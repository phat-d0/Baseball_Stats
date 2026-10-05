"""Compare model variants on several held-out months (walk-forward).

    python scripts/evaluate.py                 # both targets, default variants
    python scripts/evaluate.py --kind pitcher

For each test month, every variant is trained only on games before that month.
Reported per variant (averaged over folds, lower is better):

* mae       average absolute miss
* deviance  Poisson deviance (rewards getting the whole mean right, not just the median)
* logloss   log loss of P(over) across the app's lines - what the over/under chances
            are judged on; the baseline uses the player's season average with the same
            negative binomial spread
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseball_stats import features, model  # noqa: E402

LINES = {"batter": [0.5, 1.5, 2.5], "pitcher": [3.5, 4.5, 5.5, 6.5, 7.5]}
FOLDS = ["2025-06", "2025-08", "2026-05", "2026-07", "2026-09"]

# Columns added by the matchup work; the "base" variant leaves them out.
MATCHUP_COLS = {
    "batter": ["m_k", "m_h", "m_bb", "m_hr", "m_onbase", "pa_vs_sp", "exp_h_vs_sp",
               "exp_onbase_vs_sp", "ahead_onbase", "behind_onbase", "slot_pa_exp",
               "opp_sp_k_pct_shr", "opp_sp_h_per_bf_shr", "opp_sp_bb_per_bf_shr",
               "opp_sp_hr_per_bf_shr", "opp_sp_exp_bf", "h_per_pa_shr", "bb_pct_shr",
               "hr_per_pa_shr", "onbase_shr", "h_per_pa_vs_hand_shr",
               "lg_k_pa", "lg_h_pa", "lg_bb_pa", "lg_hr_pa"],
    "pitcher": ["exp_k_matchup", "lineup_k_log5", "exp_onbase_matchup", "exp_hr_matchup",
                "matchup_batters", "exp_bf", "h_per_bf_shr", "bb_per_bf_shr", "hr_per_bf_shr"],
}


def nb_logloss(y: np.ndarray, dist: np.ndarray, lines: list[float]) -> float:
    ll = []
    for ln in lines:
        p = np.clip(model.p_over(dist, ln), 1e-6, 1 - 1e-6)
        hit = y > ln
        ll.append(-np.where(hit, np.log(p), np.log(1 - p)))
    return float(np.mean(np.concatenate(ll)))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


def score(y, mu, alpha, kind):
    dist = model.pmf(mu, alpha, model.MAX_COUNT[kind])
    return {"mae": float(np.mean(np.abs(y - mu))), "deviance": poisson_deviance(y, mu),
            "logloss": nb_logloss(y, dist, LINES[kind])}


def run_fold(rows: pd.DataFrame, kind: str, month: str, variants: dict) -> dict:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthEnd(1)
    train, test = rows[rows["game_date"] < start], rows[rows["game_date"].between(start, end)]
    if len(test) < 50:
        return {}
    target = model.TARGETS[kind]
    y = test[target].to_numpy()
    out = {}
    calib = train[train["game_date"] >= start - pd.Timedelta(days=30)]
    for name, fn in variants.items():
        t0 = time.time()
        mu_test, mu_calib = fn(train, test, calib)
        alpha = model.fit_alpha(calib[target].to_numpy(), mu_calib)
        out[name] = {**score(y, mu_test, alpha, kind), "alpha": alpha, "secs": time.time() - t0}
    out["_n"] = len(test)
    return out


def gbm_variant(kind: str, drop: list[str] | None = None, **params):
    def fn(train, test, calib):
        tr, te, ca = (d.drop(columns=[c for c in (drop or []) if c in d]) for d in (train, test, calib))
        m = model.CountModel(kind)
        if params:
            m.reg.set_params(**params)
        m.fit(tr)
        return m.predict(te), m.predict(ca)
    return fn


def components_variant(drop: list[str] | None = None, **params):
    """Hitters: separate models for hits, runs and RBIs; the projection is their sum."""
    def fn(train, test, calib):
        mu_t, mu_c = np.zeros(len(test)), np.zeros(len(calib))
        for part in ("target_h", "target_r", "target_rbi"):
            m = model.CountModel("batter")
            if params:
                m.reg.set_params(**params)
            cols = [c for c in (drop or [])]
            tr = train.drop(columns=[c for c in cols if c in train]).assign(target_hrr=train[part])
            m.fit(tr)
            mu_t += m.predict(test.drop(columns=[c for c in cols if c in test]))
            mu_c += m.predict(calib.drop(columns=[c for c in cols if c in calib]))
        return mu_t, mu_c
    return fn


def baseline_variant(kind: str):
    def fn(train, test, calib):
        return model.baseline(test, kind), model.baseline(calib, kind)
    return fn


def variants_for(kind: str) -> dict:
    return {
        "season_avg": baseline_variant(kind),
        "base": gbm_variant(kind, drop=MATCHUP_COLS[kind]),
        "matchup": gbm_variant(kind),
        "matchup_smooth": gbm_variant(kind, learning_rate=0.03, max_leaf_nodes=15,
                                      min_samples_leaf=300, l2_regularization=5.0),
        **({"components": components_variant()} if kind == "batter" else {}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["batter", "pitcher", "both"], default="both")
    ap.add_argument("--folds", nargs="*", default=FOLDS)
    args = ap.parse_args(argv)

    t0 = time.time()
    built = features.build_features(features.load_inputs())
    print(f"features built in {time.time() - t0:.0f}s", flush=True)
    kinds = ["batter", "pitcher"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        table = built["batter_hrr" if kind == "batter" else "pitcher_k"]
        rows = model.training_rows(table, kind)
        variants = variants_for(kind)
        results = []
        for month in args.folds:
            r = run_fold(rows, kind, month, variants)
            if r:
                results.append(r)
                print(f"{kind} {month} n={r['_n']}: " + "  ".join(
                    f"{v}: mae {r[v]['mae']:.4f} dev {r[v]['deviance']:.4f} ll {r[v]['logloss']:.4f}"
                    for v in variants), flush=True)
        if not results:
            continue
        print(f"\n== {kind}: mean over {len(results)} folds")
        for v in variants:
            m = {k: np.mean([r[v][k] for r in results]) for k in ("mae", "deviance", "logloss", "alpha", "secs")}
            print(f"  {v:12s} mae {m['mae']:.4f}  deviance {m['deviance']:.4f}  "
                  f"logloss {m['logloss']:.4f}  alpha {m['alpha']:.3f}  ({m['secs']:.0f}s)")


if __name__ == "__main__":
    main()
