"""Feature rows for upcoming games (probable starters + posted lineups)."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from . import collect, features, mlb_api, parse

log = logging.getLogger(__name__)


def pending_rows(games: pd.DataFrame, lineups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Placeholder batter/pitcher rows (no stats) for not-yet-played games."""
    pit_rows, bat_rows = [], []
    for g in games.itertuples(index=False):
        for side, opp in (("home", "away"), ("away", "home")):
            sp = getattr(g, f"{side}_probable_id")
            if pd.notna(sp):
                pit_rows.append({
                    "game_pk": g.game_pk, "player_id": int(sp), "player_name": None,
                    "team_id": getattr(g, f"{side}_team_id"),
                    "opp_team_id": getattr(g, f"{opp}_team_id"),
                    "is_home": side == "home", "is_starter": True, "pitcher_order": 1,
                    "_played": 0,
                })
    probable = {}
    for g in games.itertuples(index=False):
        probable[(g.game_pk, g.home_team_id)] = g.away_probable_id
        probable[(g.game_pk, g.away_team_id)] = g.home_probable_id
    opp_of = {}
    for g in games.itertuples(index=False):
        opp_of[(g.game_pk, g.home_team_id)] = g.away_team_id
        opp_of[(g.game_pk, g.away_team_id)] = g.home_team_id
    for r in lineups.itertuples(index=False):
        if r.game_pk not in set(games["game_pk"]):
            continue
        opp_sp = probable.get((r.game_pk, r.team_id))
        bat_rows.append({
            "game_pk": r.game_pk, "player_id": r.player_id, "player_name": r.player_name,
            "team_id": r.team_id, "opp_team_id": opp_of[(r.game_pk, r.team_id)],
            "is_home": r.is_home, "position": r.position, "batting_order": r.batting_order,
            "is_starter": True,
            "opp_starter_id": int(opp_sp) if pd.notna(opp_sp) else None,
            "_played": 0,
        })
    return pd.DataFrame(bat_rows), pd.DataFrame(pit_rows)


def build_slate(day: date, inputs: dict[str, pd.DataFrame] | None = None,
                *, game_type: str = "R") -> dict[str, pd.DataFrame]:
    """Feature rows for every probable starter and lineup batter on ``day``.

    Lineups are usually posted 1-4 hours before first pitch; batters for games
    without a posted lineup are left out (re-run closer to game time).
    """
    inputs = dict(inputs or features.load_inputs())
    sched = mlb_api.schedule(day, day, game_type=game_type, cache=False)
    games, lineups = parse.parse_schedule(sched)
    games = pd.DataFrame(games)
    if games.empty:
        return {"batter_hrr": pd.DataFrame(), "pitcher_k": pd.DataFrame()}
    games = games[~games["status"].eq("Final")]
    lineups = pd.DataFrame(lineups, columns=["game_pk", "team_id", "is_home", "player_id",
                                             "player_name", "batting_order", "position"])
    bat_p, pit_p = pending_rows(games, lineups)
    log.info("%s: %d games, %d probable starters, %d lineup batters",
             day, len(games), len(pit_p), len(bat_p))

    ids = list(pit_p.get("player_id", [])) + list(bat_p.get("player_id", []))
    inputs["players"] = collect.update_players(extra_ids=ids)

    hist_games = inputs["games"]
    if not hist_games.empty:
        hist_games = hist_games[~hist_games["game_pk"].isin(games["game_pk"])]
        inputs["batter_games"] = inputs["batter_games"][
            ~inputs["batter_games"]["game_pk"].isin(games["game_pk"])]
        inputs["pitcher_games"] = inputs["pitcher_games"][
            ~inputs["pitcher_games"]["game_pk"].isin(games["game_pk"])]
    inputs["games"] = pd.concat([hist_games, games], ignore_index=True)
    inputs["batter_games"] = pd.concat([inputs["batter_games"], bat_p], ignore_index=True)
    inputs["pitcher_games"] = pd.concat([inputs["pitcher_games"], pit_p], ignore_index=True)

    out = features.build_features(inputs)
    pks = set(games["game_pk"])
    return {k: v[v["game_pk"].isin(pks)].reset_index(drop=True) for k, v in out.items()}
