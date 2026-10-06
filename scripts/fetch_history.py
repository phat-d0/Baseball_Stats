"""Download historical DraftKings player props from The Odds API for a backtest.

For each day in the range, list the day's MLB events (1 credit), then pull each
event's strikeout and H+R+RBI props as they stood ``--minutes-before`` first pitch
(10 credits per market returned). One JSON file per day under ``--out``; days already
there are skipped, so an interrupted run resumes where it stopped. Stops before the
balance falls under ``--reserve`` or this run has spent ``--max-credits``.

The default snapshot (an hour before) goes straight into ``--out``; any other time goes
into its own subfolder, ``--out/m360`` for six hours, so the snapshots never mix and
backtests read one time at a time.

    ODDS_API_KEY=... python scripts/fetch_history.py --start 2026-08-01 --end 2026-09-28
    ODDS_API_KEY=... python scripts/fetch_history.py --start 2026-08-01 --end 2026-09-27 \
        --minutes-before 360
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb"
MARKETS = "pitcher_strikeouts,batter_hits_runs_rbis"
DEFAULT_MINUTES = 60


def iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Client:
    def __init__(self, key: str, max_credits: int, reserve: int):
        self.key, self.max_credits, self.reserve = key, max_credits, reserve
        self.spent, self.remaining = 0, None
        self.s = requests.Session()

    def get(self, path: str, **params) -> dict | None:
        if self.spent >= self.max_credits:
            raise SystemExit(f"stopping: spent {self.spent} credits (limit {self.max_credits})")
        if self.remaining is not None and self.remaining - 20 < self.reserve:
            raise SystemExit(f"stopping: {self.remaining} credits left (reserve {self.reserve})")
        for attempt in range(4):
            r = self.s.get(f"{BASE}{path}", params={"apiKey": self.key, **params}, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            break
        self.spent += int(r.headers.get("x-requests-last") or 0)
        rem = r.headers.get("x-requests-remaining")
        self.remaining = int(float(rem)) if rem is not None else self.remaining
        if r.status_code == 404 or r.status_code == 422:
            return None
        r.raise_for_status()
        return r.json()


def snapshot_dir(out: Path, minutes_before: int) -> Path:
    return out if minutes_before == DEFAULT_MINUTES else out / f"m{minutes_before}"


def day_events(c: Client, d: date) -> list[dict]:
    """Events starting from 12:00 UTC on ``d`` to 12:00 UTC the next day (US game day)."""
    lo = datetime(d.year, d.month, d.day, 12, tzinfo=UTC)
    res = c.get("/events", date=iso(lo)) or {}
    out = []
    for ev in res.get("data", []):
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if lo <= start < lo + timedelta(days=1):
            out.append(ev)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--minutes-before", type=int, default=DEFAULT_MINUTES)
    ap.add_argument("--max-credits", type=int, default=20000)
    ap.add_argument("--reserve", type=int, default=25000,
                    help="leave at least this many credits for the live apps")
    ap.add_argument("--out", default="odds_history")
    args = ap.parse_args(argv)
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY is not set", file=sys.stderr)
        return 1
    out = snapshot_dir(Path(args.out), args.minutes_before)
    out.mkdir(parents=True, exist_ok=True)
    c = Client(key, args.max_credits, args.reserve)
    d, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    try:
        while d <= end:
            path = out / f"{d.isoformat()}.json"
            if path.exists():
                d += timedelta(days=1)
                continue
            events = day_events(c, d)
            snaps = []
            for ev in events:
                start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                res = c.get(f"/events/{ev['id']}/odds", regions="us", markets=MARKETS,
                            bookmakers="draftkings", oddsFormat="american",
                            date=iso(start - timedelta(minutes=args.minutes_before)))
                if res and res.get("data"):
                    snaps.append({"snapshot": res.get("timestamp"), "event": res["data"]})
            path.write_text(json.dumps({"day": d.isoformat(), "events": len(events),
                                        "minutes_before": args.minutes_before, "snapshots": snaps}))
            print(f"{d}: {len(events)} events, {len(snaps)} priced; spent {c.spent}, "
                  f"{c.remaining} left", flush=True)
            d += timedelta(days=1)
    except SystemExit as exc:
        print(exc, flush=True)
    print(f"done: spent {c.spent} credits, {c.remaining} left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
