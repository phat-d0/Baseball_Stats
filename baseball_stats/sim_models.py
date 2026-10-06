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
          fallback: model.CountModel, players: pd.DataFrame) -> _Base:
    """Fit a plate-appearance strikeout model. ``starts``: training pitcher_k rows."""
    outcome = pa_model.OutcomeModel().fit(pa, built)
    if name == "pa_simple":
        bf = model.CountModel("pitcher").fit(starts.assign(target_k=starts["target_bf"]))
        return PASimpleK(outcome=outcome, fallback=fallback, players=players, bf_model=bf)
    if name == "pa_seq":
        return PASeqK(outcome=outcome, fallback=fallback, players=players,
                      stay=pa_model.StayModel().fit(pa, built), sampler=pa_model.PitchSampler().fit(pa))
    raise ValueError(f"unknown model {name!r}")


def model_dir() -> Path:
    return config.PROCESSED_DIR / "models"


def _stamp_path(name: str) -> Path:
    return model_dir() / f"pitcher_{name}.json"


def is_stale(name: str, now: datetime, latest_pa: str | None) -> bool:
    """Retrain when missing, built by another scikit-learn, or not trained today after 10:00 UTC."""
    path = model_dir() / f"pitcher_{name}.joblib"
    if not path.exists() or not _stamp_path(name).exists():
        return True
    stamp = json.loads(_stamp_path(name).read_text())
    if stamp.get("sklearn") != sklearn.__version__:
        return True
    if latest_pa and stamp.get("latest_game_date", "") < latest_pa and now.hour >= 10 \
            and stamp.get("trained_on") != now.date().isoformat():
        return True
    return False


def save(m: _Base, now: datetime, latest_pa: str | None, record: dict | None = None) -> None:
    model_dir().mkdir(parents=True, exist_ok=True)
    keep = {k: getattr(m, k) for k in m.__dataclass_fields__ if k not in ("ctx", "players", "fallback", "last_exp_bf")}
    joblib.dump(keep, model_dir() / f"pitcher_{m.name}.joblib")
    _stamp_path(m.name).write_text(json.dumps({
        "sklearn": sklearn.__version__, "trained_on": now.date().isoformat(),
        "trained_at": now.isoformat(timespec="seconds"), "latest_game_date": latest_pa,
        "record": record or {}}, indent=1, default=float))


def load(name: str, fallback: model.CountModel, players: pd.DataFrame) -> tuple[_Base, dict]:
    keep = joblib.load(model_dir() / f"pitcher_{name}.joblib")
    cls = PASimpleK if name == "pa_simple" else PASeqK
    m = cls(**keep, fallback=fallback, players=players)
    stamp = json.loads(_stamp_path(name).read_text())
    return m, stamp.get("record") or {}


def holdout_record(name: str, pa: pd.DataFrame, built: dict, hist: pd.DataFrame,
                   fallback_params: dict | None, players: pd.DataFrame, lines: list[float],
                   test_days: int = 30) -> dict:
    """The Record tab's 30-day check for a plate-appearance model (computed daily)."""
    rows = model.training_rows(hist, "pitcher")
    cutoff = rows["game_date"].max() - pd.Timedelta(days=test_days)
    train_rows, test = rows[rows["game_date"] <= cutoff], rows[rows["game_date"] > cutoff]
    pa_train = pa[pd.to_datetime(pa["game_date"]) <= cutoff]
    if len(test) < 50 or len(pa_train) < 5000:
        return {}
    fb = model.CountModel("pitcher").fit(train_rows)
    fb.alpha = model.fit_alpha(test["target_k"].to_numpy(), fb.predict(test))
    m = train(name, pa_train, built, train_rows, fb, players).for_slate(built, players)
    mu, dist = m.distribution(test)
    return model.summarize_holdout(test, "pitcher", mu, dist, lines, alpha_for_baseline=fb.alpha)


def now_utc() -> datetime:
    return datetime.now(UTC)
