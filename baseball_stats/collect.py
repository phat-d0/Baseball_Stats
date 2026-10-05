"""Download and store game logs for a date range."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from . import mlb_api, pa_data, parse, statcast, storage

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


def dedupe_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game_pk.

    A suspended game is listed on its original date and again on the day it's finished,
    under the same game_pk: keep the finished ("Final") listing, else the latest one.
    """
    if games.empty:
        return games
    order = games.assign(_final=(games["status"] == "Final").astype(int))
    order = order.sort_values(["_final", "game_date"], kind="mergesort")
    return order.drop_duplicates("game_pk", keep="last").drop(columns="_final").reset_index(drop=True)


def collect_games(start: date, end: date, *, game_type: str = mlb_api.ALL_GAME_TYPES,
                  skip_existing: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch schedule + boxscores for finished games in [start, end] and upsert them.

    Works a month at a time and saves after each month, so an interrupted backfill
    keeps what it downloaded. With ``skip_existing`` (default), box scores already in
    ``batter_games`` aren't downloaded again, so a daily run only fetches new games.
    """
    out: dict[str, pd.DataFrame] = {}
    for s, e in _month_chunks(start, end):
        out = _collect_chunk(s, e, game_type=game_type, skip_existing=skip_existing) or out
    if not out:
        log.warning("no games found in %s..%s", start, end)
        return out
    out["players"] = update_players(out["batter_games"], out["pitcher_games"])
    return out


def _collect_chunk(start: date, end: date, *, game_type: str,
                   skip_existing: bool) -> dict[str, pd.DataFrame]:
    log.info("schedule %s..%s", start, end)
    data = mlb_api.schedule(start, end, game_type=game_type,
                            cache=end < date.today() - timedelta(days=1))
    game_rows, lineup_rows = parse.parse_schedule(data)
    games = dedupe_games(pd.DataFrame(game_rows))
    if games.empty:
        return {}

    done = played_games(games)
    if skip_existing:
        stored = storage.read("batter_games")
        if not stored.empty:
            done = done[~done["game_pk"].isin(stored["game_pk"])]
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
        ex = pd.DataFrame(extras).drop_duplicates("game_pk").set_index("game_pk")
        g = games.set_index("game_pk")
        for col in ("hp_umpire_id", "hp_umpire"):
            g.loc[ex.index, col] = ex[col].where(ex[col].notna(), g.loc[ex.index, col])
        games = g.reset_index()

    # Never let a stale schedule copy (e.g. "Scheduled") overwrite a stored final game.
    stored_games = storage.read("games")
    if not stored_games.empty and "status" in stored_games:
        final_before = set(stored_games.loc[stored_games["status"] == "Final", "game_pk"])
        games = games[(games["status"] == "Final") | ~games["game_pk"].isin(final_before)]

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
    log.info("%s..%s: stored %d new games", start, end, len(done))
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


def store_statcast(pitches: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-game Statcast aggregates plus the plate appearance table, from pitch rows."""
    if pitches.empty:
        return {}
    pa = pa_data.build(pitches, storage.read("batter_games"), storage.read("pitcher_games"))
    return {
        "statcast_batter": storage.upsert("statcast_batter", statcast.aggregate(pitches, "batter")),
        "statcast_pitcher": storage.upsert("statcast_pitcher", statcast.aggregate(pitches, "pitcher")),
        "plate_appearances": storage.upsert("plate_appearances", pa),
    }


def collect_statcast(start: date, end: date, *, game_type: str = mlb_api.ALL_GAME_TYPES) -> dict[str, pd.DataFrame]:
    """Fetch pitch-level Statcast data; store per-game aggregates and plate appearances."""
    out = {}
    for s, e in _month_chunks(start, end):
        log.info("statcast %s..%s", s, e)
        out = store_statcast(statcast.fetch_range(s, e, game_type=game_type)) or out
    return out


def collect_statcast_days(days: list[date], *, game_type: str = mlb_api.ALL_GAME_TYPES) -> list[date]:
    """Fetch and store the given days; returns the days Savant had no pitches for."""
    empty = []
    for d in days:
        pitches = statcast.fetch_day(d, game_type=game_type)
        if pitches.empty:
            empty.append(d)
            continue
        store_statcast(pitches)
    log.info("statcast: %d days fetched, %d empty", len(days), len(empty))
    return empty
