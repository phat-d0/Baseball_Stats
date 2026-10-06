"""Strikeout models built on plate appearances, behind the same interface as CountModel.

Everything downstream (publish, the price log, the app) calls
``model.distribution(df)`` and gets ``(expected counts, probability matrix)``. The
models here need the game context of the rows they price (lineups, ratings), so
publish binds them to each slate with ``for_slate(slate)`` first.

* ``pa_simple`` (:class:`PASimpleK`): each opposing hitter's strikeout chance from the
  plate-appearance outcome model; batters faced from a direct model of target_bf.
  Passed the Phase 1 gate (docs/gate_phase1_pitcher.md).
* ``pa_seq`` (:class:`PASeqK`): the batter-by-batter simulation with the stay-or-go
  model. Did not pass the gate; run in shadow.

Rows without a known opposing lineup fall back to the ``current`` model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config, model, pa_model, simulate

log = logging.getLogger(__name__)

NAMES = ("current", "pa_simple", "pa_seq")
K_MAX = model.MAX_COUNT["pitcher"]


def k_from_bf(pk: np.ndarray, p_bf: np.ndarray) -> np.ndarray:
    """Strikeout distribution: Poisson-binomial over batters, mixed over batters faced.

    ``pk[j]``: strikeout chance of the (j+1)-th batter; ``p_bf[j]``: P(faces exactly j+1).
    """
    state = np.zeros(K_MAX + 1)
    state[0] = 1.0
    out = np.zeros(K_MAX + 1)
    for j in range(len(pk)):
        nxt = state * (1 - pk[j])
        nxt[1:] += state[:-1] * pk[j]
        nxt[-1] += state[-1] * pk[j]
        state = nxt
        out += p_bf[j] * state
    return out / out.sum()


def blend(dist: np.ndarray) -> np.ndarray:
    mu = max(float((dist * np.arange(len(dist))).sum()), 1e-3)
    out = (1 - simulate.BLEND) * dist + simulate.BLEND * model.pmf(np.array([mu]), 0.02, K_MAX)[0]
    return out / out.sum()


@dataclass
class _Base:
    """Shared pieces: the outcome model, a fallback ``current`` model and the slate context."""

    name: str = ""
    outcome: pa_model.OutcomeModel | None = None
    fallback: model.CountModel | None = None
    ctx: dict | None = None
    players: pd.DataFrame | None = None
    alpha: float | None = None  # no single spread: the distribution comes from the model
    last_exp_bf: np.ndarray | None = field(default=None, repr=False)

    def for_slate(self, slate: dict, players: pd.DataFrame | None = None) -> _Base:
        bound = self.__class__(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        bound.ctx, bound.players = slate, players if players is not None else self.players
        return bound

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.distribution(df)[0]

    def _probs(self, df: pd.DataFrame) -> dict:
        m = simulate.lineup_matchups(df, self.ctx["batter_hrr"], self.players)
        return simulate.lineup_probs(m, self.outcome, self.ctx) if len(m) else {}


@dataclass
class PASimpleK(_Base):
    bf_model: model.CountModel | None = None
    name: str = "pa_simple"

    def distribution(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu, dist = self.fallback.distribution(df)
        mu, dist = mu.copy(), dist.copy()
        exp_bf = np.full(len(df), np.nan)
        probs = self._probs(df)
        mu_bf = self.bf_model.predict(df)
        b = np.arange(1, simulate.MAX_BF + 1)
        for i, r in enumerate(df.itertuples(index=False)):
            key = (int(r.game_pk), int(r.player_id))
            if key not in probs:
                continue
            p_bf = model.pmf(np.array([mu_bf[i]]), 1e-4, simulate.MAX_BF + 1)[0][1:simulate.MAX_BF + 1]
            p_bf = p_bf / p_bf.sum()
            pk = np.array([probs[key][(j - 1) % 9, min(2, (j - 1) // 9), simulate.K] for j in b])
            dist[i] = blend(k_from_bf(pk, p_bf))
            mu[i] = float((dist[i] * np.arange(K_MAX + 1)).sum())
            exp_bf[i] = float((p_bf * b).sum())
        self.last_exp_bf = exp_bf
        return mu, dist


@dataclass
class PASeqK(_Base):
    stay: pa_model.StayModel | None = None
    sampler: pa_model.PitchSampler | None = None
    n_sims: int = simulate.N_SIMS
    name: str = "pa_seq"

    def distribution(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu, dist = self.fallback.distribution(df)
        mu, dist = mu.copy(), dist.copy()
        exp_bf = np.full(len(df), np.nan)
        probs = self._probs(df)
        pre = pa_model.starter_pregame(self.ctx).set_index(["game_pk", "pitcher"])
        pre_cols = [c for c in self.stay.columns if c not in pa_model.STAY_INGAME]
        ratio = pa_model.pitcher_ppb_ratio(df, self.sampler.league_ppb)
        for i, r in enumerate(df.itertuples(index=False)):
            key = (int(r.game_pk), int(r.player_id))
            if key not in probs:
                continue
            row = pre.loc[key].reindex(pre_cols) if key in pre.index else pd.Series(np.nan, index=pre_cols)
            grid = simulate.stay_grid(self.stay, row)
            sim = simulate.simulate_start(probs[key], grid, self.sampler, float(ratio.iloc[i]),
                                          seed=simulate.seed_for(*key), n_sims=self.n_sims)
            dist[i] = simulate.distribution(sim["k"], K_MAX)
            mu[i] = float(sim["k"].mean())
            exp_bf[i] = float(sim["bf"].mean())
        self.last_exp_bf = exp_bf
        return mu, dist


# --------------------------------------------------------------------------- #
# Training, saving and loading (once a day)
# --------------------------------------------------------------------------- #

def train(name: str, pa: pd.DataFrame, built: dict, starts: pd.DataFrame,
          fallback: model.CountModel, players: pd.DataFrame, *,
          batter_games: pd.DataFrame | None = None, n_sims: int | None = None) -> _Base:
    """Fit a plate-appearance model. ``starts``: training pitcher_k rows (pitcher models)."""
    outcome = pa_model.OutcomeModel().fit(pa, built)
    if name == "pa_sim":
        from . import game_sim
        bg = batter_games if batter_games is not None else pd.DataFrame()
        return GameSimH(outcome=outcome, fallback=fallback, players=players,
                        stay=pa_model.StayModel().fit(pa, built), sampler=pa_model.PitchSampler().fit(pa),
                        transitions=game_sim.Transitions.from_pa(pa),
                        stay_prob=game_sim.stay_in_lineup(pa, bg[bg["game_pk"].isin(pa["game_pk"])]),
                        n_sims=n_sims or game_sim.N_SIMS)
    if name == "pa_simple":
        bf = model.CountModel("pitcher").fit(starts.assign(target_k=starts["target_bf"]))
        return PASimpleK(outcome=outcome, fallback=fallback, players=players, bf_model=bf)
    if name == "pa_seq":
        return PASeqK(outcome=outcome, fallback=fallback, players=players,
                      stay=pa_model.StayModel().fit(pa, built), sampler=pa_model.PitchSampler().fit(pa))
    raise ValueError(f"unknown model {name!r}")


def model_dir() -> Path:
    return config.PROCESSED_DIR / "models"


KIND_OF = {"pa_simple": "pitcher", "pa_seq": "pitcher", "pa_sim": "batter"}


def _stamp_path(name: str) -> Path:
    return model_dir() / f"{KIND_OF[name]}_{name}.json"


def _model_path(name: str) -> Path:
    return model_dir() / f"{KIND_OF[name]}_{name}.joblib"


def is_stale(name: str, now: datetime, latest_pa: str | None) -> bool:
    """Retrain when missing, built by another scikit-learn, or a newer day of plate
    appearances has arrived (normally once a day, when yesterday's Statcast lands)."""
    path = _model_path(name)
    if not path.exists() or not _stamp_path(name).exists():
        return True
    stamp = json.loads(_stamp_path(name).read_text())
    if stamp.get("sklearn") != sklearn.__version__:
        return True
    if latest_pa and stamp.get("latest_game_date", "") < latest_pa:
        return True
    return False


def save(m: _Base, now: datetime, latest_pa: str | None, record: dict | None = None) -> None:
    model_dir().mkdir(parents=True, exist_ok=True)
    keep = {k: getattr(m, k) for k in m.__dataclass_fields__
            if k not in ("ctx", "players", "fallback", "last_exp_bf", "last_parts")}
    joblib.dump(keep, _model_path(m.name))
    _stamp_path(m.name).write_text(json.dumps({
        "sklearn": sklearn.__version__, "trained_on": now.date().isoformat(),
        "trained_at": now.isoformat(timespec="seconds"), "latest_game_date": latest_pa,
        "record": record or {}}, indent=1, default=float))


def load(name: str, fallback: model.CountModel, players: pd.DataFrame) -> tuple[_Base, dict]:
    keep = joblib.load(_model_path(name))
    cls = {"pa_simple": PASimpleK, "pa_seq": PASeqK, "pa_sim": GameSimH}[name]
    m = cls(**keep, fallback=fallback, players=players)
    stamp = json.loads(_stamp_path(name).read_text())
    return m, stamp.get("record") or {}


def holdout_record(name: str, pa: pd.DataFrame, built: dict, hist: pd.DataFrame,
                   fallback_params: dict | None, players: pd.DataFrame, lines: list[float],
                   test_days: int = 30, batter_games: pd.DataFrame | None = None,
                   n_sims: int | None = None) -> dict:
    """The Record tab's 30-day check for a plate-appearance model (computed daily)."""
    kind = KIND_OF[name]
    rows = model.training_rows(hist, kind)
    cutoff = rows["game_date"].max() - pd.Timedelta(days=test_days)
    train_rows, test = rows[rows["game_date"] <= cutoff], rows[rows["game_date"] > cutoff]
    pa_train = pa[pd.to_datetime(pa["game_date"]) <= cutoff]
    if len(test) < 50 or len(pa_train) < 5000:
        return {}
    target = model.TARGETS[kind]
    fb = model.CountModel(kind).fit(train_rows)
    fb.alpha = model.fit_alpha(test[target].to_numpy(), fb.predict(test))
    m = train(name, pa_train, built, train_rows if kind == "pitcher" else None, fb, players,
              batter_games=batter_games, n_sims=n_sims).for_slate(built, players)
    mu, dist = m.distribution(test)
    return model.summarize_holdout(test, kind, mu, dist, lines, alpha_for_baseline=fb.alpha)


def now_utc() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Hitters: whole-game simulation (Phase 2)
# --------------------------------------------------------------------------- #

def reliever_probs(bat: pd.DataFrame, outcome: pa_model.OutcomeModel, ctx: dict) -> np.ndarray:
    """Each hitter vs a composite reliever from the opposing bullpen's regressed rates
    (handedness left blank). Returns (len(bat), 7)."""
    from .features import log5
    m = pd.DataFrame({"game_pk": bat["game_pk"].to_numpy(), "batter": bat["player_id"].to_numpy(),
                      "pitcher": -1, "stand": bat["bat_side"].fillna("R").to_numpy(),
                      "p_throws": None, "times_through_order": 1,
                      "batter_is_home": bat["is_home"].astype(bool).to_numpy()})
    x = pa_model.matchup_features(m, ctx)
    lg_bb, lg_hr = x.get("lg_bb", 0.08), x.get("lg_hr", 0.03)
    rp_k = pd.Series(bat.get("opp_team_rp_k_pct_l15", np.nan), index=bat.index).to_numpy()
    rp_ob = pd.Series(bat.get("opp_team_rp_onbase_pct_l15", np.nan), index=bat.index).to_numpy()
    x["p_k"] = np.where(np.isfinite(rp_k), rp_k, x.get("lg_k", 0.22))
    x["p_bb"] = lg_bb
    x["p_h"] = np.where(np.isfinite(rp_ob), np.maximum(rp_ob - lg_bb, 0.1), x.get("lg_h", 0.22))
    x["p_hr"] = lg_hr
    for name, b, p, lg in (("k", "b_k_vs_hand", "p_k", "lg_k"), ("h", "b_h_vs_hand", "p_h", "lg_h"),
                           ("bb", "b_bb", "p_bb", "lg_bb"), ("hr", "b_hr", "p_hr", "lg_hr")):
        if b in x and lg in x:
            x[f"m_{name}"] = log5(x[b], x[p], x[lg])
    x["platoon_adv"] = np.nan
    return outcome.predict_proba(x)


@dataclass
class GameSimH(_Base):
    """H+R+RBI from simulating each whole game (both lineups) with the PA models."""

    stay: pa_model.StayModel | None = None
    sampler: pa_model.PitchSampler | None = None
    transitions: object = None
    stay_prob: np.ndarray | None = None
    n_sims: int = 10_000
    name: str = "pa_sim"
    last_parts: dict | None = field(default=None, repr=False)

    def distribution(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from . import game_sim
        mu, dist = self.fallback.distribution(df)
        mu, dist = mu.copy(), dist.copy()
        H_MAX = model.MAX_COUNT["batter"]
        parts = {}
        games = set(df["game_pk"].astype(int))
        allbat = self.ctx["batter_hrr"]
        bat = allbat[allbat["game_pk"].isin(games) & allbat["is_starter"].astype(bool)
                     & allbat["batting_order"].between(1, 9)].drop_duplicates(["game_pk", "team_id", "batting_order"])
        sp = self.ctx["pitcher_k"]
        sp = sp[sp["game_pk"].isin(games)].drop_duplicates(["game_pk", "is_home"])
        if bat.empty or sp.empty:
            return mu, dist
        probs_sp = simulate.lineup_probs(simulate.lineup_matchups(sp, bat, self.players), self.outcome, self.ctx)
        rp = reliever_probs(bat, self.outcome, self.ctx)
        rp_by = {}
        for (r, row) in zip(rp, bat.itertuples(index=False)):
            rp_by.setdefault((int(row.game_pk), bool(row.is_home)), np.full((9, rp.shape[1]), np.nan))[
                int(row.batting_order) - 1] = r
        pre = pa_model.starter_pregame(self.ctx).set_index(["game_pk", "pitcher"])
        pre_cols = [c for c in self.stay.columns if c not in pa_model.STAY_INGAME]
        ratio = pa_model.pitcher_ppb_ratio(sp, self.sampler.league_ppb)
        ratio.index = pd.MultiIndex.from_arrays([sp["game_pk"].astype(int), sp["player_id"].astype(int)])
        lineup = {(int(r.game_pk), bool(r.is_home), int(r.batting_order)): int(r.player_id)
                  for r in bat.itertuples(index=False)}
        row_of = {(int(pk), int(pid)): i for i, (pk, pid) in enumerate(zip(df["game_pk"], df["player_id"]))}
        for pk in sorted(games):
            s = sp[sp["game_pk"] == pk]
            if len(s) != 2:
                continue
            ids = {bool(r.is_home): int(r.player_id) for r in s.itertuples(index=False)}
            teams = {}
            for home in (False, True):
                opp_sp = ids[not home]
                ps = probs_sp.get((pk, opp_sp))
                pr = rp_by.get((pk, home))
                if ps is None or pr is None or np.isnan(pr).any():
                    break
                key = (pk, ids[home])
                row = pre.loc[key].reindex(pre_cols) if key in pre.index else pd.Series(np.nan, index=pre_cols)
                teams[home] = game_sim.Team(probs_sp=ps, probs_rp=pr, exit_grid=simulate.stay_grid(self.stay, row),
                                            ppb_ratio=float(ratio.get(key, 1.0)))
            if len(teams) != 2:
                continue
            sim = game_sim.simulate_game(teams[False], teams[True], self.transitions, self.stay_prob,
                                         self.sampler, seed=game_sim.seed_for(pk), n_sims=self.n_sims)
            for home, side in ((False, "away"), (True, "home")):
                for slot in range(9):
                    pid = lineup.get((pk, home, slot + 1))
                    i = row_of.get((pk, pid))
                    if i is None:
                        continue
                    d, p = game_sim.hrr_distribution(sim[side], slot, H_MAX)
                    dist[i] = d
                    mu[i] = float((d * np.arange(len(d))).sum())
                    parts[(pk, pid)] = p
        self.last_parts = parts
        return mu, dist
