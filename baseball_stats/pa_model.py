"""Plate-appearance models: what happens when this batter faces this pitcher, and
whether the starter stays in after it.

* :class:`OutcomeModel` - a gradient-boosted classifier over the seven outcome classes
  (K, BB, 1B, 2B, 3B, HR, OUT), from pre-game ratings of the pitcher and batter, their
  log5 matchup values and context. It describes the matchup only: no in-game results.
  Probabilities are calibrated per class on held-out games.
* :class:`StayModel` - for starters, the chance that this batter was his last one,
  from where he is in his game (batters, pitches, baserunners, outs, times through the
  order) and his usual workload.
* :class:`PitchSampler` - pitches per plate appearance, by outcome class, scaled to
  the pitcher's own pitches-per-batter rate.

All inputs are as of the start of the game day (the no-leakage feature tables).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from .features import log5
from .pa_data import OUTCOMES, ON_BASE

HANDS = {"L": 0.0, "R": 1.0, "S": 2.0}
MIN_PROB = 1e-4

BATTER_COLS = {  # batter_hrr column -> feature name
    "k_pct_shr": "b_k", "h_per_pa_shr": "b_h", "bb_pct_shr": "b_bb", "hr_per_pa_shr": "b_hr",
    "k_pct_vs_hand_shr": "b_k_vs_hand", "h_per_pa_vs_hand_shr": "b_h_vs_hand",
    "onbase_shr": "b_onbase", "iso_szn": "b_iso", "whiff_pct_szn": "b_whiff",
    "chase_pct_szn": "b_chase", "xwoba_szn": "b_xwoba", "lg_k_pa": "lg_k", "lg_h_pa": "lg_h",
    "lg_bb_pa": "lg_bb", "lg_hr_pa": "lg_hr", "park_h_factor": "park_h",
    "park_hr_factor": "park_hr", "park_r_factor": "park_r", "temp_f": "temp_f",
}
PITCHER_RATE_COLS = {"k_pct_shr": "p_k", "h_per_bf_shr": "p_h", "bb_per_bf_shr": "p_bb",
                     "hr_per_bf_shr": "p_hr"}
STARTER_COLS = {  # pitcher_k (starters only) -> feature name
    "whiff_pct_szn": "p_whiff", "csw_pct_szn": "p_csw", "chase_pct_szn": "p_chase",
    "fb_velo_szn": "p_velo", "fb_velo_trend": "p_velo_trend", "k_pct_l5": "p_k_l5",
}
GAME_COLS = {"park_so_factor": "park_k", "ump_k_factor": "ump_k"}


def game_context(built: dict) -> pd.DataFrame:
    """One row per game: park strikeout factor and umpire factor."""
    pk = built["pitcher_k"]
    cols = [c for c in GAME_COLS if c in pk]
    return pk.groupby("game_pk")[cols].first().rename(columns=GAME_COLS).reset_index()


def matchup_features(m: pd.DataFrame, built: dict) -> pd.DataFrame:
    """Model inputs for each (game_pk, batter, pitcher) matchup row.

    ``m`` needs game_pk, batter, pitcher, stand, p_throws, times_through_order and
    batter_is_home. Works the same for real plate appearances (training) and for
    hypothetical ones (simulation).
    """
    bat = built["batter_hrr"][["game_pk", "player_id"] + [c for c in BATTER_COLS if c in built["batter_hrr"]]]
    bat = bat.drop_duplicates(["game_pk", "player_id"]).rename(columns={"player_id": "batter", **BATTER_COLS})
    pr = built["pitcher_rates"][["game_pk", "player_id", *PITCHER_RATE_COLS]]
    pr = pr.drop_duplicates(["game_pk", "player_id"]).rename(columns={"player_id": "pitcher", **PITCHER_RATE_COLS})
    st = built["pitcher_k"][["game_pk", "player_id"] + [c for c in STARTER_COLS if c in built["pitcher_k"]]]
    st = st.drop_duplicates(["game_pk", "player_id"]).rename(columns={"player_id": "pitcher", **STARTER_COLS})
    x = (m[["game_pk", "batter", "pitcher", "stand", "p_throws", "times_through_order", "batter_is_home"]]
         .merge(bat, on=["game_pk", "batter"], how="left")
         .merge(pr, on=["game_pk", "pitcher"], how="left")
         .merge(st, on=["game_pk", "pitcher"], how="left")
         .merge(game_context(built), on="game_pk", how="left"))
    x.index = m.index
    for name, b, p, lg in (("k", "b_k_vs_hand", "p_k", "lg_k"), ("h", "b_h_vs_hand", "p_h", "lg_h"),
                           ("bb", "b_bb", "p_bb", "lg_bb"), ("hr", "b_hr", "p_hr", "lg_hr")):
        if b in x and p in x and lg in x:
            x[f"m_{name}"] = log5(x[b], x[p], x[lg])
    x["platoon_adv"] = ((x["stand"] != x["p_throws"]) | (x["stand"] == "S")).astype(float)
    x["stand"] = x["stand"].map(HANDS)
    x["p_throws"] = x["p_throws"].map(HANDS)
    x["batter_is_home"] = x["batter_is_home"].astype(float)
    x["times_through_order"] = pd.to_numeric(x["times_through_order"], errors="coerce").clip(upper=3)
    return x.drop(columns=["game_pk", "batter", "pitcher"]).astype(float)


def pa_matchups(pa: pd.DataFrame) -> pd.DataFrame:
    """The matchup columns of real plate appearances."""
    return pa.assign(batter_is_home=~pa["is_top"].astype(bool))


@dataclass
class OutcomeModel:
    """P(outcome class) for one plate appearance."""

    max_iter: int = 300
    clf: HistGradientBoostingClassifier | None = None
    calibrated: CalibratedClassifierCV | None = None
    columns: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=lambda: list(OUTCOMES))

    def _new(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=self.max_iter, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=0)

    def fit(self, pa: pd.DataFrame, built: dict, *, calib_days: int = 30) -> OutcomeModel:
        """Fit on all but the last ``calib_days`` of ``pa``; calibrate on those days."""
        x = matchup_features(pa_matchups(pa), built)
        x = x.loc[:, x.notna().any()]
        self.columns = list(x.columns)
        y = pa["outcome"].to_numpy()
        dates = pd.to_datetime(pa["game_date"])
        cut = dates.max() - pd.Timedelta(days=calib_days)
        fit_rows, cal_rows = (dates <= cut).to_numpy(), (dates > cut).to_numpy()
        if cal_rows.sum() < 500 or fit_rows.sum() < 1000:
            fit_rows, cal_rows = np.ones(len(pa), bool), None
        self.clf = self._new().fit(x[fit_rows], y[fit_rows])
        self.calibrated = None
        if cal_rows is not None and set(y[cal_rows]) == set(self.clf.classes_):
            self.calibrated = CalibratedClassifierCV(FrozenEstimator(self.clf), method="isotonic")
            self.calibrated.fit(x[cal_rows], y[cal_rows])
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Probabilities in ``OUTCOMES`` order; every class at least ``MIN_PROB``, rows sum to 1."""
        x = x.reindex(columns=self.columns)
        est = self.calibrated or self.clf
        raw = est.predict_proba(x)
        cls = list(est.classes_)
        p = np.zeros((len(x), len(OUTCOMES)))
        for j, c in enumerate(OUTCOMES):
            if c in cls:
                p[:, j] = raw[:, cls.index(c)]
        p = np.maximum(p, MIN_PROB)
        return p / p.sum(axis=1, keepdims=True)

    def proba_for(self, matchups: pd.DataFrame, built: dict) -> np.ndarray:
        return self.predict_proba(matchup_features(matchups, built))


# --------------------------------------------------------------------------- #
# Stay or go
# --------------------------------------------------------------------------- #

STAY_PREGAME = {"pitches_per_start_l3": "pps_l3", "pitches_per_start_l5": "pps_l5",
                "last_pitches": "last_pitches", "days_rest": "days_rest",
                "team_sp_outs_l30": "team_sp_outs", "team_rp_k_pct_l15": "team_rp_k",
                "team_rp_onbase_pct_l15": "team_rp_onbase", "outs_per_start_l5": "outs_l5"}
STAY_INGAME = ["bf", "pitches", "baserunners", "outs", "times_through_order"]


def starter_pregame(built: dict) -> pd.DataFrame:
    pk = built["pitcher_k"]
    cols = [c for c in STAY_PREGAME if c in pk]
    return (pk[["game_pk", "player_id"] + cols].drop_duplicates(["game_pk", "player_id"])
            .rename(columns={"player_id": "pitcher", **STAY_PREGAME}))


def starter_ingame(pa: pd.DataFrame) -> pd.DataFrame:
    """Where the starter stands after each of his plate appearances."""
    s = pa[pa["pitcher_is_starter"].fillna(False).astype(bool)].sort_values(["game_pk", "at_bat_number"])
    outs = s["outcome"].isin(["K", "OUT"]).astype(int)
    on = s["outcome"].isin(ON_BASE).astype(int)
    g = [s["game_pk"], s["pitcher"]]
    return pd.DataFrame({
        "game_pk": s["game_pk"], "pitcher": s["pitcher"],
        "bf": s["bf_before"] + 1,
        "pitches": s["pitches_before"] + s["pitches"],
        "baserunners": s["baserunners_before"] + on,
        "outs": outs.groupby(g).cumsum(),
        "times_through_order": pd.to_numeric(s["times_through_order"], errors="coerce"),
        "is_last_bf": s["is_last_bf"].astype(int),
        "game_date": s["game_date"],
    }, index=s.index)


@dataclass
class StayModel:
    """P(this was the starter's last batter | his game so far)."""

    clf: HistGradientBoostingClassifier | None = None
    columns: list[str] = field(default_factory=list)

    def features(self, ingame: pd.DataFrame, pregame: pd.DataFrame) -> pd.DataFrame:
        x = ingame[["game_pk", "pitcher", *STAY_INGAME]].merge(pregame, on=["game_pk", "pitcher"], how="left")
        x.index = ingame.index
        return x.drop(columns=["game_pk", "pitcher"]).astype(float)

    def fit(self, pa: pd.DataFrame, built: dict) -> StayModel:
        ing = starter_ingame(pa)
        x = self.features(ing, starter_pregame(built))
        self.columns = list(x.columns)
        self.clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=0).fit(x, ing["is_last_bf"])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(x.reindex(columns=self.columns))[:, 1]


# --------------------------------------------------------------------------- #
# Pitches per plate appearance
# --------------------------------------------------------------------------- #

@dataclass
class PitchSampler:
    """Observed pitches per PA by outcome class, scaled to a pitcher's own rate."""

    by_class: dict[str, np.ndarray] = field(default_factory=dict)  # P(pitches = 1..MAX)
    league_ppb: float = 3.9
    MAX: int = 15

    def fit(self, pa: pd.DataFrame) -> PitchSampler:
        self.league_ppb = float(pa["pitches"].mean())
        for c in OUTCOMES:
            n = pa.loc[pa["outcome"] == c, "pitches"].clip(1, self.MAX)
            counts = np.bincount(n, minlength=self.MAX + 1)[1:].astype(float) + 0.5
            self.by_class[c] = counts / counts.sum()
        return self

    def draw(self, outcome_idx: np.ndarray, ratio: np.ndarray | float, rng: np.random.Generator) -> np.ndarray:
        """Pitches for each simulated PA; ``ratio`` = pitcher's pitches-per-batter / league."""
        cdf = np.cumsum(np.stack([self.by_class[c] for c in OUTCOMES]), axis=1)
        u = rng.random(outcome_idx.shape)
        base = (u[..., None] > cdf[outcome_idx]).sum(axis=-1) + 1
        return np.maximum(1, np.rint(base * ratio)).astype(int)


def pitcher_ppb_ratio(pitcher_k: pd.DataFrame, league_ppb: float, k: float = 60.0) -> pd.Series:
    """Each starter's regressed pitches-per-batter, as a ratio to the league."""
    pitches = pitcher_k["pitches_per_start_l10"] * pitcher_k["starts_l10"]
    bf = pitcher_k["bf_per_start_l10"] * pitcher_k["starts_l10"]
    ppb = (pitches.fillna(0) + league_ppb * k) / (bf.fillna(0) + k)
    return (ppb / league_ppb).fillna(1.0)


def k_calibration(p_k: np.ndarray, is_k: np.ndarray, buckets: int = 10) -> pd.DataFrame:
    """Predicted vs observed strikeout rate in equal-count buckets."""
    order = np.argsort(p_k)
    rows = []
    for chunk in np.array_split(order, buckets):
        rows.append({"predicted": float(p_k[chunk].mean()), "observed": float(is_k[chunk].mean()),
                     "n": int(len(chunk))})
    return pd.DataFrame(rows)
