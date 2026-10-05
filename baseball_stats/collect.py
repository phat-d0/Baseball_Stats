"""Download and store game logs for a date range."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from . import mlb_api, parse, statcast, storage

log = logging.getLogger(__name__)

NOT_PLAYED = {"Postponed", "Cancelled", "Suspended"}


def _month_chunks(start: date, end: date):
    s = start
    while s <= end:
        nxt = (s.replace(day=1) + timedelta(days=32)).replace(day=1)
        e = min(end, nxt - timedelta(days=1))
        yield s, e
        s = e + timedelta(days=1)


def played_games(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games
    mask = (games["status"] == "Final") & ~games["detailed_state"].isin(NOT_PLAYED)
    return games[mask]


def collect_games(start: date, end: date, *, game_type: str = "R") -> dict[str, pd.DataFrame]:
    """Fetch schedule + boxscores for finished games in [start, end] and upsert them."""
    game_rows, lineup_rows = [], []
    today = date.today()
    for s, e in _month_chunks(start, end):
        log.info("schedule %s..%s", s, e)
        data = mlb_api.schedule(s, e, game_type=game_type, cache=e < today - timedelta(days=1))
        g, lu = parse.parse_schedule(data)
        game_rows += g
        lineup_rows += lu
    games = pd.DataFrame(game_rows)
    if games.empty:
        log.warning("no games found in %s..%s", start, end)
        return {}

    done = played_games(games)
    batters, pitchers, extras = [], [], []
    for i, pk in enumerate(done["game_pk"], start=1):
        if i % 100 == 0:
            log.info("boxscores %d/%d", i, len(done))
        b, p, extra = parse.parse_boxscore(int(pk), mlb_api.boxscore(int(pk)))
        batters += b
        pitchers += p
        extras.append(extra)

    # Boxscore officials are authoritative for finished games.
    if extras:
        ex = pd.DataFrame(extras).set_index("game_pk")
        g = games.set_index("game_pk")
        for col in ("hp_umpire_id", "hp_umpire"):
            g.loc[ex.index, col] = ex[col].where(ex[col].notna(), g.loc[ex.index, col])
        games = g.reset_index()

    meta = games[["game_pk", "game_date", "season", "venue_id"]]
    bat_df = pd.DataFrame(batters)
    pit_df = pd.DataFrame(pitchers)
    if not bat_df.empty:
        bat_df = bat_df.merge(meta, on="game_pk", how="left")
    if not pit_df.empty:
        pit_df = pit_df.merge(meta, on="game_pk", how="left")

    out = {
        "games": storage.upsert("games", games),
        "lineups": storage.upsert("lineups", pd.DataFrame(lineup_rows)),
        "batter_games": storage.upsert("batter_games", bat_df),
        "pitcher_games": storage.upsert("pitcher_games", pit_df),
    }
    out["players"] = update_players(out["batter_games"], out["pitcher_games"])
    log.info("stored %d games, %d batter rows, %d pitcher rows",
             len(done), len(bat_df), len(pit_df))
    return out


def update_players(*frames: pd.DataFrame, extra_ids: list[int] | None = None) -> pd.DataFrame:
    """Fetch handedness/bio for any player id not yet in the players table."""
    known = storage.read("players")
    have = set(known["player_id"]) if not known.empty else set()
    ids: set[int] = set(extra_ids or [])
    for f in frames:
        if not f.empty:
            ids |= set(f["player_id"].dropna().astype(int))
    missing = sorted(ids - have)
    if not missing:
        return known
    log.info("fetching bio for %d players", len(missing))
    rows = parse.parse_people(mlb_api.people(missing))
    return storage.upsert("players", pd.DataFrame(rows))


def collect_statcast(start: date, end: date, *, game_type: str = "R") -> dict[str, pd.DataFrame]:
    """Fetch pitch-level Statcast data and store per-game batter/pitcher aggregates."""
    out = {}
    for s, e in _month_chunks(start, end):
        log.info("statcast %s..%s", s, e)
        pitches = statcast.fetch_range(s, e, game_type=game_type)
        out["statcast_batter"] = storage.upsert("statcast_batter", statcast.aggregate(pitches, "batter"))
        out["statcast_pitcher"] = storage.upsert("statcast_pitcher", statcast.aggregate(pitches, "pitcher"))
    return out
