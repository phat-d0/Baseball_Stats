"""DraftKings player props (strikeouts, Hits+Runs+RBIs) via The Odds API (the-odds-api.com).

Needs an API key in ODDS_API_KEY (a GitHub Actions secret in the publish workflow).
Phones never call the API; only the publish job does, under a strict budget:

* Listing today's games is free (``/events`` costs no credits) and its response
  headers report the credits left, so the budget is always based on the real balance.
* Props are priced per game: each game costs about 1 credit per market DraftKings has
  posted for it (2 markets), and nothing if none are posted yet.
* Each US-Eastern day gets an allowance of
  ``(credits left - RESERVE_CREDITS) / days until the monthly reset``,
  fixed at the day's first run. On the free plan (500/month) that's about 15 credits
  a day, enough to price ~7 games once; a bigger plan automatically buys more games
  and more refreshes.
* Within the allowance, games are priced only within ``WINDOW_HOURS`` of first pitch.
  A game is re-priced every ``EARLY_REFRESH_HOURS`` while more than ``EARLY_HOURS``
  away, then at most every ``EVENT_REFRESH_HOURS``, tightening to every
  ``FINAL_REFRESH_MINUTES`` in the last ``FINAL_HOURS`` (when lineups post and lines
  move), plus one closing pull once inside ``CLOSE_MINUTES`` of first pitch.
* When the allowance is short: closing pulls for games already priced come first (a
  graded pick with a closing price is worth more than one more ungraded game), then
  games not priced yet, then ordinary refreshes; soonest first pitch first within each.
* ``ODDS_API_MONTHLY_CAP`` (optional) limits what this app may spend per month, e.g.
  when the same key also feeds the soccer app.
* The reset day comes from ``ODDS_API_RESET_DAY`` (default 1) and is learned
  automatically the first time the balance goes up.

Everything is stored under ``data/processed/odds/`` so the publish workflow's cache
keeps the ledger and the last prices between runs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
BOOKMAKER = "draftkings"
BOOKMAKER_NAME = "DraftKings"
MARKETS = {"pitcher_strikeouts": "pitcher", "batter_hits_runs_rbis": "batter"}
RESERVE_CREDITS = 20  # never spend below this
WINDOW_HOURS = 24.0  # only price games starting within this many hours
EARLY_HOURS = 8.0  # more than this far out, a game is re-priced only every
EARLY_REFRESH_HOURS = 4.0  # this many hours;
EVENT_REFRESH_HOURS = 2.0  # closer in, at most this often...
FINAL_HOURS = 3.0  # ...except in its last hours before first pitch,
FINAL_REFRESH_MINUTES = 30  # when it's re-priced this often
CLOSE_MINUTES = 45  # one closing pull inside this many minutes of first pitch
DEFAULT_EVENT_COST = len(MARKETS)
EASTERN = ZoneInfo("America/New_York")


def odds_dir() -> Path:
    return config.PROCESSED_DIR / "odds"


@dataclass
class OddsStatus:
    bookmaker: str = BOOKMAKER_NAME
    credits_left: int | None = None
    daily_allowance: int | None = None
    spent_today: int = 0
    events_priced: int = 0  # games on today's slate with DraftKings props
    events_total: int = 0
    fetched_at: str | None = None  # most recent price download (UTC ISO)
    error: str | None = None


@dataclass
class Ledger:
    """Budget state kept between runs."""
    credits_left: int | None = None
    reset_day: int | None = None  # learned from a balance increase
    day: str | None = None  # US-Eastern date the allowance below applies to
    allowance: int = 0
    spent: int = 0
    event_cost: float = DEFAULT_EVENT_COST  # running estimate of credits per game
    period_start: str | None = None  # reset date of the current monthly period
    period_start_credits: int | None = None
    fetched: dict[str, str] = field(default_factory=dict)  # event id -> last download (UTC ISO)

    @classmethod
    def load(cls) -> Ledger:
        path = odds_dir() / "ledger.json"
        if not path.exists():
            return cls()
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in json.loads(path.read_text()).items() if k in known})

    def save(self) -> None:
        odds_dir().mkdir(parents=True, exist_ok=True)
        (odds_dir() / "ledger.json").write_text(json.dumps(asdict(self), indent=1))


def next_reset(now: datetime, reset_day: int) -> date:
    """Date (UTC) the monthly allowance next resets."""
    d = now.date()
    day = min(max(reset_day, 1), 28)
    this = d.replace(day=day)
    if this > d:
        return this
    return (this.replace(day=1) + timedelta(days=32)).replace(day=day)


def period_start(now: datetime, reset_day: int) -> date:
    nxt = next_reset(now, reset_day)
    return (nxt.replace(day=1) - timedelta(days=1)).replace(day=nxt.day)


def daily_allowance(credits_left: int, now: datetime, reset_day: int,
                    cap_left: int | None = None) -> int:
    available = credits_left - RESERVE_CREDITS
    if cap_left is not None:
        available = min(available, cap_left)
    days_left = max((next_reset(now, reset_day) - now.date()).days, 1)
    return max(int(available // days_left), 0)


# --------------------------------------------------------------------------- #
# Matching The Odds API names to MLB Stats API ids
# --------------------------------------------------------------------------- #


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’]", "", s)
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in {"jr", "sr", "ii", "iii", "iv"}]


def same_team(a: str, b: str) -> bool:
    """'Oakland Athletics' ~ 'Athletics', 'Boston Red Sox' ~ 'Boston Red Sox'."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return long_[-len(short):] == short


def match_events(events: list[dict], games: pd.DataFrame) -> dict[int, dict]:
    """game_pk -> Odds API event, by team names and the closest start time."""
    out: dict[int, dict] = {}
    for g in games.itertuples(index=False):
        start = pd.Timestamp(g.game_datetime) if g.game_datetime else None
        best, best_gap = None, None
        for ev in events:
            if not (same_team(ev.get("home_team", ""), g.home_team)
                    and same_team(ev.get("away_team", ""), g.away_team)):
                continue
            gap = abs((pd.Timestamp(ev["commence_time"]) - start).total_seconds()) if start is not None else 0
            if gap <= 6 * 3600 and (best_gap is None or gap < best_gap):
                best, best_gap = ev, gap
        if best is not None:
            out[int(g.game_pk)] = best
    return out


class PlayerIndex:
    """Resolve a sportsbook player name to an MLB id among a game's players."""

    def __init__(self, players: list[tuple[int, str]]):
        self.full: dict[str, set[int]] = {}
        self.short: dict[tuple[str, str], set[int]] = {}
        for pid, name in players:
            t = _tokens(name)
            if not t:
                continue
            self.full.setdefault(" ".join(t), set()).add(pid)
            self.short.setdefault((t[0][0], t[-1]), set()).add(pid)

    def find(self, name: str) -> int | None:
        """The one matching player, or None when unknown or ambiguous."""
        t = _tokens(name)
        if not t:
            return None
        hits = self.full.get(" ".join(t)) or self.short.get((t[0][0], t[-1]), set())
        return next(iter(hits)) if len(hits) == 1 else None


def parse_props(event: dict) -> list[dict]:
    """One row per (market, player, line) with American over/under prices."""
    book = next((b for b in event.get("bookmakers", []) if b.get("key") == BOOKMAKER), None)
    if not book:
        return []
    rows: dict[tuple, dict] = {}
    for m in book.get("markets", []):
        if m.get("key") not in MARKETS:
            continue
        for o in m.get("outcomes", []):
            side = (o.get("name") or "").lower()
            if side not in ("over", "under") or o.get("point") is None:
                continue
            key = (m["key"], o.get("description") or "", float(o["point"]))
            r = rows.setdefault(key, {"market": m["key"], "kind": MARKETS[m["key"]],
                                      "player": key[1], "line": key[2],
                                      "updated": m.get("last_update") or book.get("last_update")})
            r[side] = o.get("price")
    return list(rows.values())


# --------------------------------------------------------------------------- #
# Prices maths
# --------------------------------------------------------------------------- #


def american_to_decimal(a: float | None) -> float | None:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return None
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def no_vig_over(over: float | None, under: float | None) -> float | None:
    """Bookmaker's chance of the over with the margin removed (proportional)."""
    do, du = american_to_decimal(over), american_to_decimal(under)
    if not do or not du:
        return None
    po, pu = 1 / do, 1 / du
    return po / (po + pu)


# --------------------------------------------------------------------------- #
# Fetching under the budget
# --------------------------------------------------------------------------- #


CLOSING, UNPRICED, REFRESH = 0, 1, 2


def fetch_priority(start: pd.Timestamp, last: pd.Timestamp | None,
                   now: pd.Timestamp) -> int | None:
    """Why a game should be priced now (lower = more urgent), or None if it shouldn't."""
    minutes = (start - now) / pd.Timedelta(minutes=1)
    if minutes <= 0 or minutes > WINDOW_HOURS * 60:
        return None
    if last is None:
        return UNPRICED
    if minutes <= CLOSE_MINUTES:
        closed = (start - last) / pd.Timedelta(minutes=1) <= CLOSE_MINUTES
        return None if closed else CLOSING  # at most one pull inside the closing window
    gap = (FINAL_REFRESH_MINUTES if minutes <= FINAL_HOURS * 60
           else EVENT_REFRESH_HOURS * 60 if minutes <= EARLY_HOURS * 60
           else EARLY_REFRESH_HOURS * 60)
    return REFRESH if (now - last) / pd.Timedelta(minutes=1) >= gap else None


def _headers(resp: requests.Response) -> tuple[int | None, int | None]:
    left, last = resp.headers.get("x-requests-remaining"), resp.headers.get("x-requests-last")
    return (int(float(left)) if left else None, int(float(last)) if last else None)


def _event_path(event_id: str) -> Path:
    return odds_dir() / "events" / f"{event_id}.json"


def fetch_props(games: pd.DataFrame, *, now: datetime, api_key: str | None = None,
                session: requests.Session | None = None) -> tuple[dict[int, dict], OddsStatus]:
    """DraftKings props for today's games: game_pk -> cached event JSON. Never raises.

    ``games`` has game_pk, home_team, away_team, game_datetime (today's slate only).
    Downloads only what today's allowance can pay for (see module docstring).
    """
    api_key = api_key if api_key is not None else os.environ.get("ODDS_API_KEY", "")
    http = session or requests
    status = OddsStatus(events_total=len(games))
    ledger = Ledger.load()
    reset_day = int(ledger.reset_day or os.environ.get("ODDS_API_RESET_DAY", 1))
    cap = os.environ.get("ODDS_API_MONTHLY_CAP")
    cap = cap if cap and int(cap) > 0 else None  # empty or 0 = no cap
    today = now.astimezone(EASTERN).date().isoformat()

    if not api_key:
        status.error = "no ODDS_API_KEY configured"
        return {}, status
    if games.empty:
        return {}, status

    # 1) Free call: today's events + the real credit balance.
    try:
        resp = http.get(f"{BASE}/events", params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    except requests.RequestException as exc:  # the message could contain the URL (and key)
        status.error = f"could not reach The Odds API ({type(exc).__name__})"
        return _cached(games, ledger, status)
    if not resp.ok:
        status.error = f"The Odds API returned HTTP {resp.status_code}"
        return _cached(games, ledger, status)
    events = resp.json()
    left, _ = _headers(resp)
    if left is not None:
        if ledger.credits_left is not None and left > ledger.credits_left + 5:
            reset_day = now.day  # the balance went up: the allowance just reset
            ledger.reset_day = reset_day
            # Recompute today's allowance from the new balance now (e.g. after a plan
            # upgrade) instead of keeping the one fixed at the day's first run.
            ledger.day = None
            ledger.period_start = None
        ledger.credits_left = left

    pstart = period_start(now, reset_day).isoformat()
    if ledger.period_start != pstart:
        ledger.period_start, ledger.period_start_credits = pstart, ledger.credits_left
    cap_left = None
    if cap and ledger.period_start_credits is not None and ledger.credits_left is not None:
        cap_left = int(cap) - (ledger.period_start_credits - ledger.credits_left)
    if ledger.day != today:
        ledger.day, ledger.spent = today, 0
        ledger.allowance = (daily_allowance(ledger.credits_left, now, reset_day, cap_left)
                            if ledger.credits_left is not None else 0)

    # 2) Paid calls: price the most useful games the allowance covers.
    matched = match_events(events, games)
    starts = {pk: pd.Timestamp(ev["commence_time"]) for pk, ev in matched.items()}
    nowts = pd.Timestamp(now)

    def last_fetch(ev):
        t = ledger.fetched.get(ev["id"])
        return pd.Timestamp(t) if t else None

    due = []
    for pk, ev in matched.items():
        last = last_fetch(ev)
        prio = fetch_priority(starts[pk], last, nowts)
        if prio is not None:
            due.append((prio, last if last is not None else nowts, starts[pk], pk, ev))
    due.sort(key=lambda x: (x[0], x[1], x[2]))

    for _, _, _, pk, ev in due:
        floor = RESERVE_CREDITS + math.ceil(ledger.event_cost)
        if ledger.spent + ledger.event_cost > ledger.allowance or (
                ledger.credits_left is not None and ledger.credits_left < floor):
            break
        try:
            r = http.get(f"{BASE}/events/{ev['id']}/odds", params={
                "apiKey": api_key, "bookmakers": BOOKMAKER, "markets": ",".join(MARKETS),
                "oddsFormat": "american", "dateFormat": "iso"}, timeout=30)
        except requests.RequestException as exc:
            status.error = f"could not reach The Odds API ({type(exc).__name__})"
            break
        left, cost = _headers(r)
        if left is not None:
            ledger.credits_left = left
        if cost is not None:
            ledger.spent += cost
            if cost > 0:  # 0 means DraftKings hasn't posted props yet
                ledger.event_cost = 0.7 * ledger.event_cost + 0.3 * cost
        if not r.ok:
            status.error = f"The Odds API returned HTTP {r.status_code}"
            break
        path = _event_path(ev["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        ledger.fetched[ev["id"]] = now.isoformat(timespec="seconds")

    # Forget events from earlier days.
    keep = {ev["id"] for ev in matched.values()}
    ledger.fetched = {k: v for k, v in ledger.fetched.items() if k in keep}
    ledger.save()
    return _cached(games, ledger, status, matched)


def _cached(games, ledger: Ledger, status: OddsStatus,
            matched: dict[int, dict] | None = None) -> tuple[dict[int, dict], OddsStatus]:
    status.credits_left = ledger.credits_left
    status.daily_allowance = ledger.allowance
    status.spent_today = ledger.spent
    out = {}
    for pk, ev in (matched or {}).items():
        path = _event_path(ev["id"])
        if path.exists():
            out[pk] = json.loads(path.read_text())
    status.events_priced = sum(1 for e in out.values() if parse_props(e))
    times = [ledger.fetched[ev["id"]] for ev in (matched or {}).values() if ev["id"] in ledger.fetched]
    status.fetched_at = max(times) if times else None
    return out, status


def props_by_player(events: dict[int, dict], players_by_game: dict[int, list[tuple[int, str]]],
                    fetched: dict[str, str] | None = None
                    ) -> dict[tuple[int, int, str], list[dict]]:
    """(game_pk, player_id, kind) -> list of {line, over, under, updated, fetched_at}.

    ``fetched_at`` is when this app downloaded the game's prices (from the ledger);
    ``updated`` is DraftKings' own last-update time.
    """
    fetched = Ledger.load().fetched if fetched is None else fetched
    out: dict[tuple[int, int, str], list[dict]] = {}
    for pk, ev in events.items():
        fetched_at = fetched.get(ev.get("id"))
        idx = PlayerIndex(players_by_game.get(pk, []))
        for row in parse_props(ev):
            pid = idx.find(row["player"])
            if pid is None:
                continue
            out.setdefault((pk, pid, row["kind"]), []).append(
                {"line": row["line"], "over": row.get("over"), "under": row.get("under"),
                 "updated": row.get("updated"), "fetched_at": fetched_at})
    for v in out.values():
        v.sort(key=lambda r: r["line"])
    return out
