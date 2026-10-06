"""Feature rows for upcoming games (probable starters + posted or projected lineups)."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from . import collect, features, mlb_api, parse

log = logging.getLogger(__name__)

LINEUP_COLS = ["game_pk", "team_id", "is_home", "player_id", "player_name",
               "batting_order", "position"]


def recent_lineups(batter_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each team's starting nine from its most recent game, as a lineup projection."""
    if batter_games.empty:
        return pd.DataFrame(columns=["team_id", "player_id", "player_name", "batting_order", "position"])
    st = batter_games[batter_games["is_starter"].astype(bool)]
    dates = games.drop_duplicates("game_pk").set_index("game_pk")["game_date"]
    st = st.assign(_d=pd.to_datetime(st["game_pk"].map(dates)))
    last_pk = st.sort_values(["_d", "game_pk"]).groupby("team_id")["game_pk"].last()
    out = st[st["game_pk"].isin(last_pk.values)]
    out = out[out["game_pk"] == out["team_id"].map(last_pk)]
    return out.sort_values(["team_id", "batting_order"])[
        ["team_id", "player_id", "player_name", "batting_order", "position"]]


def pending_rows(games: pd.DataFrame, lineups: pd.DataFrame,
                 projected: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Placeholder batter/pitcher rows (no stats) for not-yet-played games.

    Teams without a posted lineup get ``projected`` (their last starting nine),
    flagged with ``lineup_confirmed = False``. Starters carry ``opp_lineup_confirmed``:
    whether the lineup they face is posted (strikeout edges on projected lineups are
    much weaker, docs/edge_lineups.md).
    """
    pit_rows, bat_rows = [], []
    probable, opp_of = {}, {}
    for g in games.itertuples(index=False):
        for side, opp in (("home", "away"), ("away", "home")):
            team, opp_team = getattr(g, f"{side}_team_id"), getattr(g, f"{opp}_team_id")
            probable[(g.game_pk, team)] = getattr(g, f"{opp}_probable_id")
            opp_of[(g.game_pk, team)] = opp_team
            sp = getattr(g, f"{side}_probable_id")
            if pd.notna(sp):
                pit_rows.append({
                    "game_pk": g.game_pk, "player_id": int(sp), "player_name": None,
                    "team_id": team, "opp_team_id": opp_team, "is_home": side == "home",
                    "is_starter": True, "pitcher_order": 1, "_played": 0,
                })

    posted = {(r.game_pk, r.team_id) for r in lineups.itertuples(index=False)}
    for r in pit_rows:
        r["opp_lineup_confirmed"] = (r["game_pk"], r["opp_team_id"]) in posted
    rows = [(r, True) for r in lineups.itertuples(index=False) if r.game_pk in set(games["game_pk"])]
    if projected is not None and not projected.empty:
        by_team = {t: grp for t, grp in projected.groupby("team_id")}
        for g in games.itertuples(index=False):
            for side in ("home", "away"):
                team = getattr(g, f"{side}_team_id")
                if (g.game_pk, team) in posted or team not in by_team:
                    continue
                for p in by_team[team].itertuples(index=False):
                    rows.append((pd.Series({
                        "game_pk": g.game_pk, "team_id": team, "is_home": side == "home",
                        "player_id": p.player_id, "player_name": p.player_name,
                        "batting_order": p.batting_order, "position": p.position,
                    }), False))
    for r, confirmed in rows:
        opp_sp = probable.get((r.game_pk, r.team_id))
        bat_rows.append({
            "game_pk": r.game_pk, "player_id": int(r.player_id), "player_name": r.player_name,
            "team_id": r.team_id, "opp_team_id": opp_of[(r.game_pk, r.team_id)],
            "is_home": bool(r.is_home), "position": r.position,
            "batting_order": int(r.batting_order), "is_starter": True,
            "opp_starter_id": int(opp_sp) if pd.notna(opp_sp) else None,
            "lineup_confirmed": confirmed, "_played": 0,
        })
    return pd.DataFrame(bat_rows), pd.DataFrame(pit_rows)


def build_slate(day: date, inputs: dict[str, pd.DataFrame] | None = None,
                *, game_type: str = mlb_api.ALL_GAME_TYPES, project_lineups: bool = True,
                include_history: bool = False) -> dict[str, pd.DataFrame]:
    """Feature rows for every probable starter and lineup batter on ``day``.

    Lineups are usually posted 1-4 hours before first pitch. Until then, with
    ``project_lineups``, a team's most recent starting nine stands in.
    With ``include_history`` the result also has ``history_batter_hrr`` /
    ``history_pitcher_k``: the training rows from the same feature build.
    """
    inputs = dict(inputs or features.load_inputs())
    sched = mlb_api.schedule(day, day, game_type=game_type, cache=False)
    games, lineups = parse.parse_schedule(sched)
    games = pd.DataFrame(games)
    empty = {"batter_hrr": pd.DataFrame(), "pitcher_k": pd.DataFrame(), "games": games}
    if games.empty:
        return empty
    games = games[~games["status"].eq("Final") & ~games["detailed_state"].isin(collect.NOT_PLAYED)]
    if games.empty:
        return empty
    lineups = pd.DataFrame(lineups, columns=LINEUP_COLS)
    hist_games = inputs["games"]
    hist_bat, hist_pit = inputs["batter_games"], inputs["pitcher_games"]
    if not hist_games.empty:
        keep = ~hist_games["game_pk"].isin(games["game_pk"])
        hist_games = hist_games[keep & (pd.to_datetime(hist_games["game_date"]) < pd.Timestamp(day))]
        hist_bat = hist_bat[hist_bat["game_pk"].isin(hist_games["game_pk"])]
        hist_pit = hist_pit[hist_pit["game_pk"].isin(hist_games["game_pk"])]

    projected = recent_lineups(hist_bat, hist_games) if project_lineups else None
    bat_p, pit_p = pending_rows(games, lineups, projected)
    log.info("%s: %d games, %d probable starters, %d lineup batters (%d posted)",
             day, len(games), len(pit_p), len(bat_p), len(lineups))

    ids = list(pit_p.get("player_id", [])) + list(bat_p.get("player_id", []))
    inputs["players"] = collect.update_players(extra_ids=ids)
    inputs["games"] = pd.concat([hist_games, games], ignore_index=True)
    inputs["batter_games"] = pd.concat([hist_bat, bat_p], ignore_index=True)
    inputs["pitcher_games"] = pd.concat([hist_pit, pit_p], ignore_index=True)

    out = features.build_features(inputs)
    pks = set(games["game_pk"])
    result = {k: v[v["game_pk"].isin(pks)].reset_index(drop=True) for k, v in out.items()}
    if "lineup_confirmed" in bat_p:
        conf = bat_p.set_index(["game_pk", "player_id"])["lineup_confirmed"]
        idx = pd.MultiIndex.from_frame(result["batter_hrr"][["game_pk", "player_id"]])
        result["batter_hrr"]["lineup_confirmed"] = conf.reindex(idx).to_numpy()
    if "opp_lineup_confirmed" in pit_p:
        conf = pit_p.set_index(["game_pk", "player_id"])["opp_lineup_confirmed"]
        idx = pd.MultiIndex.from_frame(result["pitcher_k"][["game_pk", "player_id"]])
        result["pitcher_k"]["opp_lineup_confirmed"] = conf.reindex(idx).to_numpy()
    result["games"] = games.reset_index(drop=True)
    if include_history:
        for k, v in out.items():
            result[f"history_{k}"] = v[~v["game_pk"].isin(pks)].reset_index(drop=True)
    return result
