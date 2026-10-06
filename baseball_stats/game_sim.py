"""Whole-game simulation for hitters: H+R+RBI for every starting hitter at once.

Both lineups bat in order through nine innings (up to three extra innings with the
automatic runner on second; the bottom of the 9th is skipped when the home team leads,
and a walk-off ends the game). Each plate appearance draws an outcome from the
plate-appearance outcome model: against the starter until the stay-or-go model removes
him, then against a composite reliever built from that team's bullpen rates. Runner
movement, outs, runs and RBIs come from a draw on the observed base-out transitions.
Runners never pass each other: the most advanced score first. Starting hitters can be
replaced (at the rate observed for their lineup slot and trip to the plate); a replaced
hitter earns nothing more.

Every run is credited to someone: slots 0-8 are the starting hitters, slot 9 collects
bench hitters and runners the transition table adds without a known identity, so the
hitters' runs always add up to the team's score.

Vectorised with NumPy across simulations; one fixed seed per game.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model
from .pa_data import END, OUTCOMES, transitions_from
from .simulate import MAX_BF, MAX_RUNNERS, PITCH_STEP

N_SIMS = 10_000
MAX_INNINGS = 12
BENCH = 9  # credit bucket for bench hitters and unidentified runners
HIT_IDX = np.array([o in ("1B", "2B", "3B", "HR") for o in OUTCOMES])
ON_BASE_IDX = np.array([o in ("BB", "1B", "2B", "3B", "HR") for o in OUTCOMES])
OUT_IDX = np.array([o in ("K", "OUT") for o in OUTCOMES])
MAX_TRIPS = 7


# --------------------------------------------------------------------------- #
# Base-out transitions as lookup arrays
# --------------------------------------------------------------------------- #

@dataclass
class Transitions:
    """Per (outcome, state): cumulative probabilities of (next state, runs, RBIs)."""

    cum: np.ndarray   # (7, 24, M)
    nxt: np.ndarray   # (7, 24, M)
    runs: np.ndarray  # (7, 24, M)
    rbi: np.ndarray   # (7, 24, M)

    @classmethod
    def from_pa(cls, pa: pd.DataFrame, before=None) -> Transitions:
        if before is not None:
            pa = pa[pd.to_datetime(pa["game_date"]) < pd.Timestamp(before)]
        t = transitions_from(pa)
        g = t.groupby(["outcome", "state", "next_state", "runs", "rbi"]).size().rename("n").reset_index()
        return cls.from_counts(g)

    @classmethod
    def from_counts(cls, g: pd.DataFrame) -> Transitions:
        oi = {o: i for i, o in enumerate(OUTCOMES)}
        groups = {(oi[o], int(s)): grp for (o, s), grp in g.groupby(["outcome", "state"])}
        M = max(len(x) for x in groups.values())
        cum = np.ones((len(OUTCOMES), 24, M))
        nxt = np.full((len(OUTCOMES), 24, M), END, int)
        runs = np.zeros((len(OUTCOMES), 24, M), int)
        rbi = np.zeros((len(OUTCOMES), 24, M), int)
        for o in range(len(OUTCOMES)):
            for s in range(24):
                grp = groups.get((o, s))
                if grp is None:
                    grp = _default_transition(o, s)
                k = len(grp)
                p = grp["n"].to_numpy(float)
                cum[o, s, :k] = np.cumsum(p / p.sum())
                cum[o, s, k - 1:] = 1.0
                nxt[o, s, :k] = grp["next_state"].to_numpy()
                runs[o, s, :k] = grp["runs"].to_numpy()
                rbi[o, s, :k] = grp["rbi"].to_numpy()
                # Pad with the last outcome so a draw never lands on an empty cell.
                nxt[o, s, k:], runs[o, s, k:], rbi[o, s, k:] = nxt[o, s, k - 1], runs[o, s, k - 1], rbi[o, s, k - 1]
        return cls(cum, nxt, runs, rbi)

    @classmethod
    def simple_rules(cls) -> Transitions:
        """Textbook advancement (walks force, hits advance runners by the hit's bases,
        outs advance nobody). Used in tests and when no data exists for a state."""
        rows = []
        for o in range(len(OUTCOMES)):
            for s in range(24):
                rows.append(_default_transition(o, s).assign(outcome=OUTCOMES[o], state=s))
        return cls.from_counts(pd.concat(rows, ignore_index=True))

    def draw(self, o: np.ndarray, s: np.ndarray, u: np.ndarray):
        c = self.cum[o, s]
        k = (u[:, None] > c).sum(axis=1).clip(max=c.shape[1] - 1)
        return self.nxt[o, s, k], self.runs[o, s, k], self.rbi[o, s, k]


def _default_transition(o: int, s: int) -> pd.DataFrame:
    outs, b = divmod(s, 8)
    on = [bool(b & 1), bool(b & 2), bool(b & 4)]
    name = OUTCOMES[o]
    runs = 0
    if name in ("K", "OUT"):
        outs += 1
    elif name == "BB":
        if on[0]:
            if on[1]:
                if on[2]:
                    runs += 1
                on[2] = True
            on[1] = True
        on[0] = True
    else:
        adv = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}[name]
        new = [False, False, False]
        for i in (2, 1, 0):
            if on[i]:
                if i + adv >= 3:
                    runs += 1
                else:
                    new[i + adv] = True
        if adv == 4:
            runs += 1
        else:
            new[adv - 1] = True
        on = new
    nxt = END if outs >= 3 else outs * 8 + on[0] + 2 * on[1] + 4 * on[2]
    return pd.DataFrame({"next_state": [nxt], "runs": [runs], "rbi": [runs], "n": [1]})


# --------------------------------------------------------------------------- #
# Substitutions
# --------------------------------------------------------------------------- #

def stay_in_lineup(pa: pd.DataFrame, batter_games: pd.DataFrame) -> np.ndarray:
    """P(the starter in each lineup slot takes his t-th trip | the slot gets a t-th trip).

    Returns (9, MAX_TRIPS) for slots 1..9 and trips 1..MAX_TRIPS.
    """
    st = batter_games[batter_games["is_starter"].astype(bool) & batter_games["batting_order"].between(1, 9)]
    starters = st.set_index(["game_pk", "team_id", "batting_order"])["player_id"]
    p = pa.dropna(subset=["batter_slot"]).copy()
    p = p[p["batter_slot"].between(1, 9)]
    # Batting team: the batter's team in that game.
    team = batter_games.drop_duplicates(["game_pk", "player_id"]).set_index(["game_pk", "player_id"])["team_id"]
    p["team_id"] = team.reindex(pd.MultiIndex.from_arrays([p["game_pk"], p["batter"]])).to_numpy()
    p = p.dropna(subset=["team_id"]).sort_values(["game_pk", "at_bat_number"])
    p["trip"] = p.groupby(["game_pk", "team_id", "batter_slot"]).cumcount() + 1
    key = pd.MultiIndex.from_arrays([p["game_pk"], p["team_id"], p["batter_slot"].astype(int)])
    p["is_starter"] = starters.reindex(key).to_numpy() == p["batter"].to_numpy()
    tab = p[p["trip"] <= MAX_TRIPS].groupby(["batter_slot", "trip"])["is_starter"].mean()
    out = np.ones((9, MAX_TRIPS))
    for (slot, trip), v in tab.items():
        out[int(slot) - 1, int(trip) - 1] = v
    # Once replaced, a starter doesn't come back: keep it non-increasing.
    return np.minimum.accumulate(out, axis=1)


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #

@dataclass
class Team:
    """One team's inputs: how its hitters fare (batting) and its starter (fielding)."""

    probs_sp: np.ndarray   # (9, 3, 7) its hitters vs the opposing starter, per time through
    probs_rp: np.ndarray   # (9, 7) its hitters vs the opposing composite reliever
    exit_grid: np.ndarray  # its own starter's stay-or-go grid (see simulate.stay_grid)
    ppb_ratio: float = 1.0  # its own starter's pitches per batter vs the league


def seed_for(game_pk: int) -> int:
    return zlib.crc32(f"game-{int(game_pk)}".encode())


def simulate_game(away: Team, home: Team, trans: Transitions, stay_prob: np.ndarray,
                  sampler, *, seed: int, n_sims: int = N_SIMS,
                  max_innings: int = MAX_INNINGS) -> dict:
    """Simulate one game. Returns per team ("away", "home") H, R, RBI arrays of shape
    (n_sims, 10) (slot 9 = bench/unknown) and the score (n_sims, 2)."""
    rng = np.random.default_rng(seed)
    teams = [away, home]
    n = n_sims
    idx = np.arange(n)
    H = [np.zeros((n, 10), int) for _ in teams]
    R = [np.zeros((n, 10), int) for _ in teams]
    RBI = [np.zeros((n, 10), int) for _ in teams]
    score = np.zeros((n, 2), int)
    next_slot = [np.zeros(n, int), np.zeros(n, int)]
    trips = [np.zeros((n, 9), int), np.zeros((n, 9), int)]
    replaced = [np.zeros((n, 9), bool), np.zeros((n, 9), bool)]
    # Pitching state of each team's starter.
    sp_in = [np.ones(n, bool), np.ones(n, bool)]
    bf = [np.zeros(n, int), np.zeros(n, int)]
    pitches = [np.zeros(n, int), np.zeros(n, int)]
    allowed = [np.zeros(n, int), np.zeros(n, int)]
    over = np.zeros(n, bool)
    cdf_sp = [np.cumsum(t.probs_sp, axis=-1) for t in teams]
    cdf_rp = [np.cumsum(t.probs_rp, axis=-1) for t in teams]
    bench_sp = [c.mean(axis=0) for c in cdf_sp]  # (3, 7): average hitter of that lineup
    bench_rp = [c.mean(axis=0) for c in cdf_rp]  # (7,)

    for inning in range(1, max_innings + 1):
        for bat in (0, 1):
            fld = 1 - bat
            active = ~over.copy()
            if bat == 1 and inning >= 9:
                active &= ~(score[:, 1] > score[:, 0])  # home leads: no bottom half
                over |= (score[:, 1] > score[:, 0])
            outs = np.zeros(n, int)
            bases = np.full((n, 3), -1, int)  # runner credit id on 1st, 2nd, 3rd (-1 = empty)
            if inning > 9:
                bases[:, 1] = np.where(active, slot_id(next_slot[bat] - 1, replaced[bat], idx), -1)
            while True:
                live = active & (outs < 3)
                if not live.any():
                    break
                slot = next_slot[bat] % 9
                trip = trips[bat][idx, slot] + 1
                # Substitution: the starter may be gone by this trip.
                keep = np.ones(n)
                t_i = np.minimum(trip, MAX_TRIPS) - 1
                prev = np.where(t_i > 0, stay_prob[slot, np.maximum(t_i - 1, 0)], 1.0)
                keep = np.where(prev > 0, stay_prob[slot, t_i] / np.maximum(prev, 1e-9), 0.0)
                newly = live & ~replaced[bat][idx, slot] & (rng.random(n) >= keep)
                replaced[bat][idx[newly], slot[newly]] = True
                bench = replaced[bat][idx, slot]
                tto = np.minimum(2, bf[fld] // 9)
                c = np.where(
                    sp_in[fld][:, None],
                    np.where(bench[:, None], bench_sp[bat][tto], cdf_sp[bat][slot, tto]),
                    np.where(bench[:, None], bench_rp[bat][None, :], cdf_rp[bat][slot]))
                o = (rng.random(n)[:, None] > c).sum(axis=1).clip(max=len(OUTCOMES) - 1)
                state = outs * 8 + (bases[:, 0] >= 0) + 2 * (bases[:, 1] >= 0) + 4 * (bases[:, 2] >= 0)
                nxt, runs, rbi = trans.draw(o, np.minimum(state, 23), rng.random(n))
                batter_id = np.where(bench, BENCH, slot)

                # Runs: the most advanced runners score first, then the batter.
                cand = np.stack([bases[:, 2], bases[:, 1], bases[:, 0], batter_id], axis=1)
                order = np.argsort(cand < 0, axis=1, kind="stable")
                cand = np.take_along_axis(cand, order, axis=1)
                n_valid = (cand >= 0).sum(axis=1)
                runs = np.where(live, runs, 0)
                for k in range(4):
                    who = np.where(k < np.minimum(runs, n_valid), cand[:, k], -1)
                    m = live & (who >= 0)
                    np.add.at(R[bat], (idx[m], who[m]), 1)
                extra = np.where(live, runs - np.minimum(runs, n_valid), 0)
                np.add.at(R[bat], (idx, np.full(n, BENCH)), extra)
                # Remaining runners fill the new occupied bases, most advanced first.
                r_used = np.minimum(runs, n_valid)
                new_bases = np.full((n, 3), -1, int)
                ptr = r_used.copy()
                occ = [(nxt % 8) & 4, (nxt % 8) & 2, (nxt % 8) & 1]  # 3rd, 2nd, 1st
                for j, base in enumerate((2, 1, 0)):
                    occupied = (occ[j] > 0) & (nxt != END)
                    pick = np.take_along_axis(cand, np.minimum(ptr, 3)[:, None], axis=1)[:, 0]
                    pick = np.where(ptr < n_valid, pick, BENCH)
                    new_bases[:, base] = np.where(occupied, pick, -1)
                    ptr = ptr + occupied
                bases = np.where(live[:, None], new_bases, bases)
                outs = np.where(live, np.where(nxt == END, 3, nxt // 8), outs)
                score[:, bat] += runs
                hit = live & HIT_IDX[o]
                np.add.at(H[bat], (idx[hit], batter_id[hit]), 1)
                np.add.at(RBI[bat], (idx[live], batter_id[live]), np.where(live, rbi, 0)[live])
                trips[bat][idx[live], slot[live]] += 1
                next_slot[bat] = np.where(live, next_slot[bat] + 1, next_slot[bat])

                # The fielding team's starter: count the batter, then stay or go.
                sp = live & sp_in[fld]
                bf[fld] += sp
                pitches[fld] += np.where(sp, sampler.draw(o, teams[fld].ppb_ratio, rng), 0)
                allowed[fld] += sp & ON_BASE_IDX[o]
                g = teams[fld].exit_grid
                pb = np.minimum(pitches[fld] // PITCH_STEP, g.shape[1] - 1)
                p_exit = g[np.clip(bf[fld], 1, MAX_BF) - 1, pb, np.minimum(allowed[fld], MAX_RUNNERS)]
                sp_in[fld] &= ~(sp & (rng.random(n) < p_exit))

                if bat == 1 and inning >= 9:  # walk-off
                    walk = live & (score[:, 1] > score[:, 0])
                    over |= walk
                    active &= ~walk
            if inning >= 9 and bat == 1:
                over |= score[:, 0] != score[:, 1]
        if over.all():
            break
    return {"away": {"H": H[0], "R": R[0], "RBI": RBI[0]},
            "home": {"H": H[1], "R": R[1], "RBI": RBI[1]}, "score": score}


def slot_id(slot: np.ndarray, replaced: np.ndarray, idx: np.ndarray) -> np.ndarray:
    s = slot % 9
    return np.where(replaced[idx, s], BENCH, s)


def hrr_distribution(sim_team: dict, slot: int, max_count: int) -> tuple[np.ndarray, dict]:
    """A starting hitter's H+R+RBI distribution (with the 1% blend) and his averages."""
    total = sim_team["H"][:, slot] + sim_team["R"][:, slot] + sim_team["RBI"][:, slot]
    hist = np.bincount(np.minimum(total, max_count), minlength=max_count + 1)[: max_count + 1]
    sim = hist / hist.sum()
    nb = model.pmf(np.array([max(total.mean(), 1e-3)]), 0.3, max_count)[0]
    dist = 0.99 * sim + 0.01 * nb
    parts = {k: float(sim_team[k][:, slot].mean()) for k in ("H", "R", "RBI")}
    return dist / dist.sum(), parts
