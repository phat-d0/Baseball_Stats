"""Count models for the two targets, with over/under probabilities.

A gradient-boosted Poisson regression predicts the mean (expected H+R+RBI or
strikeouts). Real outcomes are more spread out than Poisson, so the mean is
turned into a negative binomial distribution whose extra spread (``alpha``) is
fitted on held-out games. That distribution gives P(over any line).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

# Columns that identify a row or are known only after the game.
NON_FEATURES = {
    "game_pk", "game_date", "season", "player_id", "player_name", "team_id", "opp_team_id",
    "opp_starter_id", "venue_id", "position", "is_starter", "lineup_confirmed",
}
HANDS = {"L": 0.0, "R": 1.0, "S": 2.0}

TARGETS = {"batter": "target_hrr", "pitcher": "target_k"}
MAX_COUNT = {"batter": 8, "pitcher": 15}


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in df.columns if c not in NON_FEATURES and not c.startswith("target_")]].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].map(HANDS)  # bat_side / pitch_hand codes
    return X.astype(float)


def training_rows(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = df[df[TARGETS[kind]].notna()]
    if kind == "batter":
        rows = rows[rows["is_starter"]]
    return rows


# Chosen with scripts/evaluate.py (walk-forward over five held-out months). Single-game
# hitter totals are very noisy, so the hitter model is far more heavily smoothed.
PARAMS = {
    "batter": dict(learning_rate=0.02, max_iter=800, max_leaf_nodes=7,
                   min_samples_leaf=800, l2_regularization=10.0),
    "pitcher": dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                    min_samples_leaf=80, l2_regularization=1.0),
}


def _regressor(kind: str = "pitcher") -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=0, **PARAMS[kind],
    )


def fit_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """Negative binomial dispersion: Var = mu + alpha * mu^2 (method of moments)."""
    excess = np.mean((y - mu) ** 2 - mu)
    return float(max(excess / np.mean(mu ** 2), 1e-4))


def pmf(mu: np.ndarray, alpha: float, max_count: int) -> np.ndarray:
    """P(Y = 0..max_count-1) and, in the last column, P(Y >= max_count)."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)[:, None]
    n = 1.0 / alpha
    k = np.arange(max_count)[None, :]
    probs = stats.nbinom.pmf(k, n, n / (n + mu))
    tail = 1.0 - probs.sum(axis=1, keepdims=True)
    return np.hstack([probs, np.clip(tail, 0, 1)])


def p_over(dist: np.ndarray, line: float) -> np.ndarray:
    """P(Y > line) for a half-point line, from a :func:`pmf` matrix."""
    return dist[:, int(np.floor(line)) + 1:].sum(axis=1)


@dataclass
class CountModel:
    kind: str
    reg: HistGradientBoostingRegressor | None = None
    alpha: float = 0.1
    columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.reg is None:
            self.reg = _regressor(self.kind)

    def fit(self, df: pd.DataFrame) -> CountModel:
        rows = training_rows(df, self.kind)
        X = feature_frame(rows)
        X = X.loc[:, X.notna().any()]  # e.g. Statcast columns when it wasn't collected
        self.columns = list(X.columns)
        self.reg.fit(X, rows[TARGETS[self.kind]])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = feature_frame(df).reindex(columns=self.columns)
        return self.reg.predict(X)

    def distribution(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = self.predict(df)
        return mu, pmf(mu, self.alpha, MAX_COUNT[self.kind])


def baseline(df: pd.DataFrame, kind: str) -> np.ndarray:
    """What you'd guess without a model: the player's own season (else career) average."""
    if kind == "batter":
        cols = ["hrr_per_g_szn", "hrr_per_g_car"]
    else:
        cols = ["k_per_start_szn", "k_per_start_car"]
    b = df[cols[0]].fillna(df[cols[1]])
    return b.fillna(training_rows(df, kind)[TARGETS[kind]].mean()).to_numpy()


def evaluate(df: pd.DataFrame, kind: str, *, test_days: int = 30,
             lines: list[float]) -> tuple[dict, float]:
    """Train on everything before the last ``test_days`` days, score those days.

    Returns the summary shown in the app's Record tab and the fitted alpha.
    """
    rows = training_rows(df, kind)
    cutoff = rows["game_date"].max() - pd.Timedelta(days=test_days)
    train, test = rows[rows["game_date"] <= cutoff], rows[rows["game_date"] > cutoff]
    if len(train) < 500 or len(test) < 50:
        return {}, 0.1
    m = CountModel(kind).fit(train)
    y = test[TARGETS[kind]].to_numpy()
    # The spread is fitted on games the model didn't train on: residuals on its own
    # training games are too small and would make the over/under chances overconfident.
    m.alpha = fit_alpha(y, m.predict(test))
    mu, dist = m.distribution(test)
    base = baseline(test, kind)
    base_dist = pmf(base, m.alpha, MAX_COUNT[kind])

    def logloss(d):
        p = np.clip(np.concatenate([p_over(d, ln) for ln in lines]), 1e-6, 1 - 1e-6)
        h = np.concatenate([y > ln for ln in lines])
        return float(-np.mean(np.where(h, np.log(p), np.log(1 - p))))

    # Calibration: every (row, line) pair, bucketed by predicted P(over).
    probs = np.concatenate([p_over(dist, ln) for ln in lines])
    hits = np.concatenate([(y > ln).astype(float) for ln in lines])
    bins = np.clip((probs * 10).astype(int), 0, 9)
    calib = [
        {"p": float(probs[bins == b].mean()), "actual": float(hits[bins == b].mean()),
         "n": int((bins == b).sum())}
        for b in range(10) if (bins == b).sum() >= 30
    ]

    # Each day's 10 most confident "over" calls at the most common line.
    main = lines[len(lines) // 2]
    t = test.assign(_p=p_over(dist, main), _hit=(y > main).astype(float))
    top = t.sort_values("_p", ascending=False).groupby("game_date").head(10)

    by_line = {}
    for ln in lines:
        pl, hl = p_over(dist, ln), (y > ln).astype(float)
        bl = np.clip((pl * 10).astype(int), 0, 9)
        by_line[f"{ln:g}"] = [
            {"p": float(pl[bl == b].mean()), "actual": float(hl[bl == b].mean()),
             "n": int((bl == b).sum())}
            for b in range(10) if (bl == b).sum() >= 20
        ]

    summary = {
        "test_from": test["game_date"].min().date().isoformat(),
        "test_to": test["game_date"].max().date().isoformat(),
        "n": int(len(test)),
        "mean_actual": float(y.mean()),
        "mae_model": float(np.mean(np.abs(y - mu))),
        "mae_baseline": float(np.mean(np.abs(y - base))),
        "rmse_model": float(np.sqrt(np.mean((y - mu) ** 2))),
        "rmse_baseline": float(np.sqrt(np.mean((y - base) ** 2))),
        # Log loss of P(over) across the app's lines: how good the over/under chances are.
        "logloss_model": logloss(dist),
        "logloss_baseline": logloss(base_dist),
        "calibration": calib,
        "calibration_by_line": by_line,
        "top_picks": {"line": main, "n": int(len(top)),
                      "predicted": float(top["_p"].mean()), "actual": float(top["_hit"].mean())},
    }
    return summary, m.alpha
