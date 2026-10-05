"""Strikeout distributions for starters, by simulating their start batter by batter.

For each start: the opposing lineup bats in order; each batter's outcome is drawn from
the plate-appearance outcome model (for that batter, this pitcher and the time through
the order); pitches are drawn per outcome; after each batter the stay-or-go model
decides whether the starter is done. The strikeout distribution is whatever the
simulation produces, blended with 1% of a negative binomial curve so no count has
zero probability.

Vectorised across simulations with NumPy. A fixed seed per game and pitcher keeps
repeated runs identical when the inputs haven't changed.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model
from .pa_data import OUTCOMES
from .pa_model import STAY_INGAME, OutcomeModel, PitchSampler, StayModel

N_SIMS = 10_000
MAX_BF = 45
BLEND = 0.01
PITCH_STEP = 2  # stay-or-go grid resolution in pitches
MAX_PITCHES = 160
MAX_RUNNERS = 15
K, BB, OUT = OUTCOMES.index("K"), OUTCOMES.index("BB"), OUTCOMES.index("OUT")
ON_BASE_IDX = np.array([o in ("BB", "1B", "2B", "3B", "HR") for o in OUTCOMES])
OUT_IDX = np.array([o in ("K", "OUT") for o in OUTCOMES])


def seed_for(game_pk: int, pitcher: int) -> int:
    return zlib.crc32(f"{int(game_pk)}-{int(pitcher)}".encode())


def stay_grid(stay: StayModel, pregame: pd.Series) -> np.ndarray:
    """P(exit) for every (batters faced, pitch bucket, baserunners) state, for one starter.

    The pitcher's pre-game inputs are fixed for the whole game, so the model is asked
    once per state instead of once per simulated batter.
    """
    bf = np.arange(1, MAX_BF + 1)
    pb = np.arange(0, MAX_PITCHES + 1, PITCH_STEP)
    rn = np.arange(0, MAX_RUNNERS + 1)
    B, P, R = np.meshgrid(bf, pb, rn, indexing="ij")
    g = pd.DataFrame({"bf": B.ravel(), "pitches": P.ravel(), "baserunners": R.ravel()})
    g["outs"] = (g["bf"] - g["baserunners"]).clip(lower=0)
    g["times_through_order"] = np.minimum(3, (g["bf"] - 1) // 9 + 1)
    for c, v in pregame.items():
        g[c] = v
    x = g[STAY_INGAME + [c for c in pregame.index]].astype(float)
    return stay.predict(x).reshape(B.shape)


def simulate_start(probs: np.ndarray, exit_grid: np.ndarray | None, sampler: PitchSampler,
                   ppb_ratio: float, *, seed: int, n_sims: int = N_SIMS,
                   fixed_bf: int | None = None, exit_after: int | None = None) -> dict:
    """Simulate one start.

    ``probs``: (9, 3, 7) outcome probabilities per lineup slot and time through the
    order. ``exit_grid`` from :func:`stay_grid`. ``fixed_bf`` (tests) faces exactly that
    many batters; ``exit_after`` (tests) forces the exit after that many batters.
    Returns strikeout counts and batters faced per simulation.
    """
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(probs, axis=-1)
    k = np.zeros(n_sims, int)
    bf = np.zeros(n_sims, int)
    pitches = np.zeros(n_sims, int)
    runners = np.zeros(n_sims, int)
    outs = np.zeros(n_sims, int)
    alive = np.ones(n_sims, bool)
    for step in range(MAX_BF):
        if not alive.any():
            break
        slot, tto = step % 9, min(2, step // 9)
        u = rng.random(n_sims)
        o = (u[:, None] > cdf[slot, tto][None, :]).sum(axis=1).clip(max=len(OUTCOMES) - 1)
        live = alive
        k += live & (o == K)
        outs += live & OUT_IDX[o]
        runners += live & ON_BASE_IDX[o]
        pitches += np.where(live, sampler.draw(o, ppb_ratio, rng), 0)
        bf += live
        if fixed_bf is not None:
            alive = live & (bf < fixed_bf)
            continue
        if exit_after is not None:
            alive = live & (bf < exit_after)
            continue
        pb = np.minimum(pitches // PITCH_STEP, exit_grid.shape[1] - 1)
        p_exit = exit_grid[np.minimum(bf, MAX_BF) - 1, pb, np.minimum(runners, MAX_RUNNERS)]
        leave = rng.random(n_sims) < p_exit
        alive = live & ~leave & (outs < 27)
    return {"k": k, "bf": bf, "pitches": pitches}


def distribution(counts: np.ndarray, max_count: int, alpha: float = 0.02) -> np.ndarray:
    """Simulated counts -> P(0..max_count-1, >= max_count), with the 1% NB blend."""
    hist = np.bincount(np.minimum(counts, max_count), minlength=max_count + 1)[: max_count + 1]
    sim = hist / hist.sum()
    nb = model.pmf(np.array([max(counts.mean(), 1e-3)]), max(alpha, 1e-4), max_count)[0]
    out = (1 - BLEND) * sim + BLEND * nb
    return out / out.sum()


# --------------------------------------------------------------------------- #
# Lineups and per-batter probabilities for a set of starts
# --------------------------------------------------------------------------- #

def lineup_matchups(starts: pd.DataFrame, batters: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Matchup rows (9 slots x 3 times through) for each start's opposing lineup.

    ``starts``: pitcher_k rows (game_pk, player_id, is_home). ``batters``: batter_hrr
    rows with is_starter, batting_order, opp_starter_id.
    """
    hand = players.set_index("player_id")
    lu = batters[batters["is_starter"].astype(bool) & batters["batting_order"].between(1, 9)]
    lu = lu.drop_duplicates(["game_pk", "opp_starter_id", "batting_order"])
    m = starts[["game_pk", "player_id", "is_home"]].rename(columns={"player_id": "pitcher"}).merge(
        lu[["game_pk", "opp_starter_id", "player_id", "batting_order"]].rename(
            columns={"opp_starter_id": "pitcher", "player_id": "batter"}),
        on=["game_pk", "pitcher"], how="inner")
    m["p_throws"] = m["pitcher"].map(hand["pitch_hand"]).fillna("R")
    bats = m["batter"].map(hand["bat_side"]).fillna("R")
    m["stand"] = np.where(bats == "S", np.where(m["p_throws"] == "R", "L", "R"), bats)
    m["batter_is_home"] = ~m["is_home"].astype(bool)
    m = m.loc[m.index.repeat(3)].reset_index(drop=True)
    m["times_through_order"] = np.tile([1, 2, 3], len(m) // 3)
    return m


def lineup_probs(m: pd.DataFrame, outcome: OutcomeModel, built: dict) -> dict[tuple[int, int], np.ndarray]:
    """(game_pk, pitcher) -> (9, 3, 7) probabilities, slots ordered 1..9."""
    p = outcome.proba_for(m, built)
    out = {}
    m = m.assign(row_i=np.arange(len(m)))
    for (pk, pit), grp in m.groupby(["game_pk", "pitcher"]):
        arr = np.full((9, 3, len(OUTCOMES)), np.nan)
        for r in grp.itertuples(index=False):
            arr[int(r.batting_order) - 1, int(r.times_through_order) - 1] = p[r.row_i]
        if np.isnan(arr).any():  # a missing slot: use the lineup's average
            mean = np.nanmean(arr, axis=0)
            arr = np.where(np.isnan(arr), mean[None], arr)
        out[(int(pk), int(pit))] = arr
    return out


@dataclass
class SimResult:
    mu: np.ndarray
    dist: np.ndarray
    exp_bf: np.ndarray
