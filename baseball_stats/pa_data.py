"""Plate appearance table and base-out transitions, from Baseball Savant pitch rows.

``plate_appearances`` (one row per PA, key game_pk + at_bat_number) is what the
plate-appearance models train on. ``base_out_transitions`` gives, for each outcome
class and base-out state, how often each next state followed and how many runs
scored; the game simulation draws runner movement from it.

Pure functions, no network calls.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OUTCOMES = ["K", "BB", "1B", "2B", "3B", "HR", "OUT"]
EVENT_CLASS = {
    "strikeout": "K", "strikeout_double_play": "K",
    "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "BB",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
}
# Events on a pitch row that do not end the plate appearance (or end it unfinished).
NOT_PA_PREFIXES = ("caught_stealing", "pickoff", "stolen_base", "wild_pitch", "passed_ball",
                   "other_advance", "truncated_pa", "game_advisory", "ejection", "balk",
                   "runner_double_play", "other_out")
NO_RBI_EVENTS = {"grounded_into_double_play", "double_play", "strikeout_double_play",
                 "field_error"}
HITS = {"1B", "2B", "3B", "HR"}
ON_BASE = HITS | {"BB"}


def outcome_class(event: str) -> str | None:
    if not isinstance(event, str) or not event:
        return None
    if event.startswith(NOT_PA_PREFIXES):
        return None
    return EVENT_CLASS.get(event, "OUT")


def build(pitches: pd.DataFrame, batter_games: pd.DataFrame | None = None,
          pitcher_games: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per completed plate appearance, in game order.

    ``batter_games``/``pitcher_games`` (optional) supply the lineup slot and whether the
    pitcher started.
    """
    if pitches.empty:
        return pd.DataFrame()
    p = pitches.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    n_pitches = p.groupby(["game_pk", "at_bat_number"]).size().rename("pitches")
    last = p.groupby(["game_pk", "at_bat_number"]).tail(1).copy()
    last["outcome"] = last["events"].map(outcome_class)
    pa = last[last["outcome"].notna()].merge(n_pitches.reset_index(), on=["game_pk", "at_bat_number"])
    pa = pa.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)

    out = pd.DataFrame({
        "game_pk": pa["game_pk"].astype(int),
        "at_bat_number": pa["at_bat_number"].astype(int),
        "game_date": pd.to_datetime(pa["game_date"]),
        "inning": pa["inning"].astype(int),
        "is_top": pa["inning_topbot"].eq("Top"),
        "batter": pa["batter"].astype(int),
        "pitcher": pa["pitcher"].astype(int),
        "stand": pa["stand"],
        "p_throws": pa["p_throws"],
        "event": pa["events"],
        "outcome": pa["outcome"],
        "pitches": pa["pitches"].astype(int),
        "outs_before": pa["outs_when_up"].astype(int),
        "on_1b": pa["on_1b"].notna(),
        "on_2b": pa["on_2b"].notna(),
        "on_3b": pa["on_3b"].notna(),
        "runs_scored": (pa["post_bat_score"] - pa["bat_score"]).clip(lower=0).astype(int),
    })
    if "n_thruorder_pitcher" in pa:
        out["times_through_order"] = pa["n_thruorder_pitcher"].clip(upper=3).astype("Int64")
    out["rbi"] = out["runs_scored"].where(~out["event"].isin(NO_RBI_EVENTS), 0)

    gp = out.groupby(["game_pk", "pitcher"], sort=False)
    out["bf_before"] = gp.cumcount()
    out["pitches_before"] = gp["pitches"].cumsum() - out["pitches"]
    reached = out["outcome"].isin(ON_BASE).astype(int)
    out["baserunners_before"] = reached.groupby([out["game_pk"], out["pitcher"]]).cumsum() - reached
    if "times_through_order" not in out:
        out["times_through_order"] = (out["bf_before"] // 9 + 1).clip(upper=3)
    out["is_last_bf"] = ~out.duplicated(["game_pk", "pitcher"], keep="last")
    out["batter_pa_number"] = out.groupby(["game_pk", "batter"]).cumcount() + 1

    if batter_games is not None and not batter_games.empty:
        slots = batter_games.drop_duplicates(["game_pk", "player_id"]).set_index(
            ["game_pk", "player_id"])["batting_order"]
        idx = pd.MultiIndex.from_arrays([out["game_pk"], out["batter"]])
        out["batter_slot"] = slots.reindex(idx).to_numpy()
    else:
        out["batter_slot"] = np.nan
    if pitcher_games is not None and not pitcher_games.empty:
        st = pitcher_games.drop_duplicates(["game_pk", "player_id"]).set_index(
            ["game_pk", "player_id"])["is_starter"]
        idx = pd.MultiIndex.from_arrays([out["game_pk"], out["pitcher"]])
        out["pitcher_is_starter"] = st.reindex(idx).astype("boolean").to_numpy()
    else:
        out["pitcher_is_starter"] = pd.array([pd.NA] * len(out), dtype="boolean")
    return out


# --------------------------------------------------------------------------- #
# Base-out states
# --------------------------------------------------------------------------- #

END = 24  # three outs: the half-inning is over


def state_code(outs, on1, on2, on3) -> np.ndarray:
    """0..23 = outs * 8 + bases bitmask (1st=1, 2nd=2, 3rd=4)."""
    return (np.asarray(outs) * 8 + np.asarray(on1, int) + 2 * np.asarray(on2, int)
            + 4 * np.asarray(on3, int))


def transitions_from(pa: pd.DataFrame) -> pd.DataFrame:
    """Observed (outcome, state before) -> (state after, runs) for each PA."""
    p = pa.sort_values(["game_pk", "at_bat_number"])
    before = state_code(p["outs_before"], p["on_1b"], p["on_2b"], p["on_3b"])
    half = p["game_pk"].astype(str) + "-" + p["inning"].astype(str) + p["is_top"].astype(str)
    nxt = pd.Series(before, index=p.index).groupby(half.to_numpy()).shift(-1)
    after = nxt.fillna(END).astype(int).to_numpy()
    return pd.DataFrame({"outcome": p["outcome"].to_numpy(), "state": before,
                         "next_state": after, "runs": p["runs_scored"].to_numpy()})


def build_transitions(pa: pd.DataFrame, before=None) -> pd.DataFrame:
    """``base_out_transitions``: share of each (next state, runs) per (outcome, state).

    Only plate appearances from games before ``before`` are used, so a model evaluated
    on a month never sees that month's runner movement.
    """
    if before is not None:
        pa = pa[pd.to_datetime(pa["game_date"]) < pd.Timestamp(before)]
    t = transitions_from(pa)
    counts = t.groupby(["outcome", "state", "next_state", "runs"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby(["outcome", "state"])["n"].transform("sum")
    return counts


# --------------------------------------------------------------------------- #
# Reconciliation with the box scores
# --------------------------------------------------------------------------- #


def reconcile(pa: pd.DataFrame, batter_games: pd.DataFrame,
              pitcher_games: pd.DataFrame) -> dict:
    """Do strikeouts per pitcher, hits per batter and RBIs from PAs match the box scores?

    Spec: >= 99% of games match on K and hits; season RBIs within 2%.
    """
    if pa.empty:
        return {"games": 0}
    games = pa["game_pk"].unique()
    pk = pa.assign(k=pa["outcome"].eq("K").astype(int), h=pa["outcome"].isin(HITS).astype(int))
    k_pa = pk.groupby(["game_pk", "pitcher"])["k"].sum()
    h_pa = pk.groupby(["game_pk", "batter"])["h"].sum()
    pg = pitcher_games[pitcher_games["game_pk"].isin(games)].set_index(["game_pk", "player_id"])["k"]
    bg = batter_games[batter_games["game_pk"].isin(games)].set_index(["game_pk", "player_id"])
    k_ok = (k_pa.reindex(pg.index, fill_value=0) == pg).groupby(level=0).all()
    h_ok = (h_pa.reindex(bg.index, fill_value=0) == bg["h"]).groupby(level=0).all()
    ok = (k_ok & h_ok.reindex(k_ok.index, fill_value=False))
    bad = sorted(int(g) for g in ok.index[~ok])
    rbi_pa, rbi_box = int(pa["rbi"].sum()), int(bg["rbi"].sum())
    out = {
        "games": int(len(ok)),
        "match_share": float(ok.mean()) if len(ok) else None,
        "mismatched_games": bad[:50],
        "rbi_pa": rbi_pa,
        "rbi_box": rbi_box,
        "rbi_gap": abs(rbi_pa - rbi_box) / rbi_box if rbi_box else None,
    }
    out["passed"] = bool(out["match_share"] is not None and out["match_share"] >= 0.99
                         and out["rbi_gap"] is not None and out["rbi_gap"] <= 0.02)
    if bad:
        log.warning("PA/box score mismatch in %d of %d games, e.g. %s", len(bad), len(ok), bad[:5])
    return out
