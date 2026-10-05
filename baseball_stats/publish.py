"""Build the static phone app: update data, train models, write `data.json` next to the web files.

    python -m baseball_stats publish --out _site

The web app (web/) is copied as-is; everything it shows comes from data.json, so the
site needs no server and can be hosted on GitHub Pages.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import collect, features, mlb_api, model, odds, slate, storage

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
EASTERN = ZoneInfo("America/New_York")  # MLB's "game day" follows US Eastern time
BATTER_LINES = [0.5, 1.5, 2.5]
PITCHER_LINES = [3.5, 4.5, 5.5, 6.5, 7.5]
LOOKAHEAD_DAYS = 7  # with no games today (off days, off-season) look this far ahead
SLATES = 2  # how many game days to show (today + next)
STATCAST_DAYS_PER_RUN = 45


def _clean(obj):
    """Make numpy/pandas values JSON-safe (NaN -> null, rounded floats)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return None if math.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def _num(row: pd.Series, col: str):
    v = row.get(col)
    return None if v is None or pd.isna(v) else float(v)


def update_data(today: date, history_start: date | None) -> None:
    """Collect every finished game not yet stored, up to yesterday."""
    start = history_start or date(today.year - 2, 3, 1)
    games = storage.read("games")
    if not games.empty:
        final = games[games["status"] == "Final"]
        if not final.empty:
            # A few days of overlap picks up suspended/late-finishing games.
            last = pd.to_datetime(final["game_date"]).max().date()
            start = max(start, last - timedelta(days=3))
    end = today - timedelta(days=1)
    if start <= end:
        collect.collect_games(start, end)


def update_statcast(today: date, history_start: date | None) -> None:
    """Fetch Statcast for days not yet stored. Failures only cost the Statcast features."""
    start = history_start or date(today.year - 2, 3, 1)
    games = storage.read("games")
    have = storage.read("statcast_pitcher")
    if not have.empty and not games.empty:
        dates = pd.to_datetime(games.loc[games["game_pk"].isin(have["game_pk"]), "game_date"])
        if not dates.empty:
            start = max(start, dates.max().date() + timedelta(days=1))
    # Savant is slow (one request per day), so backfill at most this many days per run.
    end = min(today - timedelta(days=1), start + timedelta(days=STATCAST_DAYS_PER_RUN - 1))
    if start > end:
        return
    try:
        collect.collect_statcast(start, end, game_type=mlb_api.ALL_GAME_TYPES)
    except Exception as exc:  # Savant is slow/flaky at times; keep publishing without it
        log.warning("statcast update failed (%s); continuing without new Statcast data", exc)


def find_slates(today: date, inputs: dict) -> list[tuple[date, dict]]:
    """Feature rows for the next ``SLATES`` days with games; the first carries history."""
    out = []
    d = today
    while len(out) < SLATES and d <= today + timedelta(days=LOOKAHEAD_DAYS):
        s = slate.build_slate(d, inputs, include_history=not out)
        if not s["pitcher_k"].empty or not s["batter_hrr"].empty:
            out.append((d, s))
        d += timedelta(days=1)
    return out


def book_lines(entries: list[dict] | None, pmf: np.ndarray) -> list[dict] | None:
    """DraftKings lines for one player, with the model's chance and expected value per side."""
    if not entries:
        return None
    out = []
    for e in entries:
        p = float(model.p_over(pmf[None, :], e["line"])[0])
        do, du = odds.american_to_decimal(e.get("over")), odds.american_to_decimal(e.get("under"))
        out.append({
            **e,
            "p_model": p,
            "p_book": odds.no_vig_over(e.get("over"), e.get("under")),
            "ev_over": p * do - 1 if do else None,
            "ev_under": (1 - p) * du - 1 if du else None,
        })
    return out


def _pitcher_json(row: pd.Series, mu: float, pmf: np.ndarray, names: dict) -> dict:
    return {
        "id": int(row["player_id"]),
        "name": row.get("player_name") or names.get(int(row["player_id"]), "TBD"),
        "hand": row.get("pitch_hand"),
        "mu": mu,
        "pmf": pmf,
        "stats": {
            "k_pct_szn": _num(row, "k_pct_szn"),
            "k_per_start_l5": _num(row, "k_per_start_l5"),
            "k_per_start_szn": _num(row, "k_per_start_szn"),
            "outs_per_start_l5": _num(row, "outs_per_start_l5"),
            "pitches_per_start_l5": _num(row, "pitches_per_start_l5"),
            "starts_szn": _num(row, "starts_szn"),
            "era_szn": _num(row, "era_szn"),
            "whiff_pct_szn": _num(row, "whiff_pct_szn"),
            "days_rest": _num(row, "days_rest"),
            "lineup_k_pct": _num(row, "lineup_k_pct_vs_hand"),
            "opp_team_k_pct": _num(row, "opp_team_k_pct_szn"),
            "ump_k_factor": _num(row, "ump_k_factor"),
            "park_k_factor": _num(row, "park_so_factor"),
        },
    }


def _batter_json(row: pd.Series, mu: float, pmf: np.ndarray, names: dict) -> dict:
    return {
        "id": int(row["player_id"]),
        "name": row.get("player_name") or names.get(int(row["player_id"]), ""),
        "order": int(row["batting_order"]) if pd.notna(row.get("batting_order")) else None,
        "pos": row.get("position"),
        "bats": row.get("bat_side"),
        "vs": row.get("opp_sp_hand"),
        "confirmed": bool(row.get("lineup_confirmed", True)),
        "mu": mu,
        "pmf": pmf,
        "stats": {
            "hrr_per_g_szn": _num(row, "hrr_per_g_szn"),
            "hrr_per_g_l15": _num(row, "hrr_per_g_l15"),
            "hrr_per_pa_vs_hand": _num(row, "hrr_per_pa_vs_hand_shr"),
            "pa_per_g_l15": _num(row, "pa_per_g_l15"),
            "games_szn": _num(row, "games_szn"),
            "xwoba_szn": _num(row, "xwoba_szn"),
            "opp_sp_k_pct": _num(row, "opp_sp_k_pct_shr"),
            "opp_sp_era": _num(row, "opp_sp_era_szn"),
            "park_r_factor": _num(row, "park_r_factor"),
        },
    }


def slate_json(day: date, s: dict, models: dict[str, model.CountModel], names: dict,
               label: str, props: dict | None = None) -> dict:
    bat, pit, games = s["batter_hrr"], s["pitcher_k"], s["games"]
    by_game: dict[int, dict] = {}
    for g in games.itertuples(index=False):
        by_game[g.game_pk] = {
            "game_pk": g.game_pk,
            "time": g.game_datetime,
            "status": g.detailed_state,
            "venue": g.venue_name,
            "away": {"id": g.away_team_id, "name": g.away_team, "abbr": getattr(g, "away_abbr", None)},
            "home": {"id": g.home_team_id, "name": g.home_team, "abbr": getattr(g, "home_abbr", None)},
            "temp_f": g.temp_f, "wind": " ".join(
                str(x) for x in (f"{g.wind_mph:.0f} mph" if pd.notna(g.wind_mph) else None, g.wind_dir)
                if x) or None,
            "umpire": g.hp_umpire,
            "pitchers": {"away": None, "home": None},
            "lineups": {"away": [], "home": []},
        }
    if not pit.empty:
        mu, dist = models["pitcher"].distribution(pit)
        for (_, row), m, d in zip(pit.iterrows(), mu, dist):
            side = "home" if row["is_home"] else "away"
            pj = _pitcher_json(row, m, d, names)
            pj["book"] = book_lines((props or {}).get((row["game_pk"], pj["id"], "pitcher")), d)
            by_game[row["game_pk"]]["pitchers"][side] = pj
    if not bat.empty:
        mu, dist = models["batter"].distribution(bat)
        for (_, row), m, d in zip(bat.iterrows(), mu, dist):
            side = "home" if row["is_home"] else "away"
            bj = _batter_json(row, m, d, names)
            bj["book"] = book_lines((props or {}).get((row["game_pk"], bj["id"], "batter")), d)
            by_game[row["game_pk"]]["lineups"][side].append(bj)
        for g in by_game.values():
            for side in ("away", "home"):
                g["lineups"][side].sort(key=lambda b: b["order"] or 99)
    ordered = sorted(by_game.values(), key=lambda g: (g["time"] or "", g["game_pk"]))
    return {"date": day.isoformat(), "label": label, "games": ordered}


def publish(out: str | Path, *, history_start: date | None = None,
            now: datetime | None = None, statcast: bool = False) -> dict:
    now = now or datetime.now(UTC)
    today = now.astimezone(EASTERN).date()
    update_data(today, history_start)
    if statcast:
        update_statcast(today, history_start)

    inputs = features.load_inputs()
    slates = find_slates(today, inputs)
    if slates:
        hist = {"batter": slates[0][1]["history_batter_hrr"], "pitcher": slates[0][1]["history_pitcher_k"]}
    else:
        built = features.build_features(inputs)
        hist = {"batter": built["batter_hrr"], "pitcher": built["pitcher_k"]}

    models, record = {}, {}
    for kind, lines in (("batter", BATTER_LINES), ("pitcher", PITCHER_LINES)):
        summary, alpha = model.evaluate(hist[kind], kind, lines=lines)
        record[kind] = summary
        if len(model.training_rows(hist[kind], kind)) < 50:
            raise RuntimeError(f"not enough finished games to train the {kind} model; "
                               "collect more history with --history-start")
        m = model.CountModel(kind).fit(hist[kind])
        m.alpha = alpha
        models[kind] = m
        log.info("%s model: alpha=%.3f %s", kind, alpha,
                 {k: v for k, v in summary.items() if k.startswith(("mae", "n"))})

    # build_slate may have fetched bios for new players, so read the table fresh.
    players = storage.read("players")
    names = dict(zip(players.get("player_id", []), players.get("full_name", [])))

    def label(d: date) -> str:
        if d == today:
            return "Today"
        if d == today + timedelta(days=1):
            return "Tomorrow"
        return d.strftime("%a %b %-d")

    # DraftKings props for today's games only, under the credit budget (see odds.py).
    props, odds_status = {}, odds.OddsStatus(error="no games today")
    if slates and slates[0][0] == today:
        s = slates[0][1]
        roster = pd.concat([s["batter_hrr"][["game_pk", "player_id"]],
                            s["pitcher_k"][["game_pk", "player_id"]]], ignore_index=True)
        roster["name"] = roster["player_id"].map(names)
        players_by_game = {pk: list(zip(g["player_id"], g["name"].fillna("")))
                           for pk, g in roster.groupby("game_pk")}
        events, odds_status = odds.fetch_props(s["games"], now=now)
        props = odds.props_by_player(events, players_by_game)
        log.info("odds: %s", odds_status)

    stored = storage.read("games")
    final = stored[stored["status"] == "Final"] if not stored.empty else stored
    data = {
        "generated_at": now.isoformat(),
        "data_through": (pd.to_datetime(final["game_date"]).max().date().isoformat()
                         if not final.empty else None),
        "lines": {"batter": BATTER_LINES, "pitcher": PITCHER_LINES},
        "model": {k: {"alpha": m.alpha, "trained_rows": int(len(model.training_rows(hist[k], k)))}
                  for k, m in models.items()},
        "slates": [slate_json(d, s, models, names, label(d), props if d == today else None)
                   for d, s in slates],
        "odds_source": asdict(odds_status),
        "record": record,
    }
    data = _clean(data)

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB_DIR, out)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    log.info("wrote %s (%d slates)", out / "data.json", len(data["slates"]))
    return data
