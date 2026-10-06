"""Fit the model/DraftKings blend (baseball_stats/blend.py) on one set of historical
prices and test it on another.

    python scripts/fit_blend.py --fit odds_history/2025 --test odds_history/2026 \
        --weights baseball_stats/blend.json --report docs/blend_2026.json

Fits ``logit(p) = a * logit(p_model) + b * logit(p_book) + c`` per kind on the fit set
(walk-forward model chances, as in scripts/backtest.py), shows how stable the weights
are month to month, then scores the blend on the test set: log loss against the model
and DraftKings, and the return of betting its edges.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest  # noqa: E402
from baseball_stats import odds, tracking  # noqa: E402

THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05]
BANDS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, None)]


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(g: pd.DataFrame) -> dict:
    m = LogisticRegression(C=1e6).fit(np.c_[logit(g["p_model"]), logit(g["p_book"])], g["over_won"])
    a, b = m.coef_[0]
    return {"model": float(a), "book": float(b), "intercept": float(m.intercept_[0]), "lines": int(len(g))}


def apply(g: pd.DataFrame, w: dict) -> pd.DataFrame:
    z = w["model"] * logit(g["p_model"]) + w["book"] * logit(g["p_book"]) + w["intercept"]
    g = g.copy()
    g["p_blend"] = 1 / (1 + np.exp(-z))
    do = g["over"].map(odds.american_to_decimal).astype(float)
    du = g["under"].map(odds.american_to_decimal).astype(float)
    g["ev_over"] = g["p_blend"] * do - 1
    g["ev_under"] = (1 - g["p_blend"]) * du - 1
    return g


def load(folder: str, batter: str = "current") -> pd.DataFrame:
    snaps = backtest.load_prices(Path(folder))
    months = sorted({s["day"][:7] for s in snaps})
    pred, inputs = backtest.predictions(months, 10_000, batter)
    names = inputs["players"].set_index("player_id")["full_name"]
    roster = pred[["game_pk", "player_id", "kind"]].assign(name=pred["player_id"].map(names)).dropna()
    g = backtest.grades(backtest.price_rows(snaps, inputs["games"], roster), pred)
    g["month"] = pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m")
    print(f"{folder}: {len(snaps)} games, {len(g)} lines", flush=True)
    return g


def returns(p: pd.DataFrame) -> dict:
    out = {"thresholds": {}, "bands": []}
    for t in THRESHOLDS:
        out["thresholds"][f"{t:g}"] = tracking.pick_metrics(p[p["ev"] >= t])
    for lo, hi in BANDS:
        out["bands"].append({"lo": lo, "hi": hi,
                             **tracking.pick_metrics(p[(p["ev"] >= lo) & ((p["ev"] < hi) if hi else True)])})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="folder of historical prices to fit on")
    ap.add_argument("--test", required=True, help="folder of later historical prices to test on")
    ap.add_argument("--weights", default=None, help="write the fitted weights here")
    ap.add_argument("--report", default=None)
    ap.add_argument("--batter", default="current", help="hitter model (an evaluate.py variant)")
    args = ap.parse_args(argv)

    g_fit, g_test = load(args.fit, args.batter), load(args.test, args.batter)
    report: dict = {"fit": {}, "by_month": {}, "test": {}}
    weights = {}
    for kind in ("batter", "pitcher"):
        k = g_fit[g_fit["kind"] == kind]
        weights[kind] = fit(k)
        report["by_month"][kind] = {m: fit(x) for m, x in k.groupby("month") if len(x) >= 200}
        t = apply(g_test[g_test["kind"] == kind], weights[kind])
        ll = lambda p: tracking._logloss(p, t["over_won"])  # noqa: E731
        report["test"][kind] = {"lines": int(len(t)), "logloss_model": ll(t["p_model"]),
                                "logloss_book": ll(t["p_book"]), "logloss_blend": ll(t["p_blend"]),
                                **returns(tracking.picks(t, -1.0))}
        print(kind, json.dumps(weights[kind]), {m: round(w["model"], 2) for m, w in report["by_month"][kind].items()})
        print("  test log loss model %.4f book %.4f blend %.4f" % (
            report["test"][kind]["logloss_model"], report["test"][kind]["logloss_book"],
            report["test"][kind]["logloss_blend"]))
    both = pd.concat([apply(g_test[g_test["kind"] == k], weights[k]) for k in weights])
    report["test"]["all"] = returns(tracking.picks(both, -1.0))
    for t, m in report["test"]["all"]["thresholds"].items():
        if m.get("n"):
            print(f"  blended edge >= {float(t):.0%}: {m['n']} bets, won {m['win']:.1%} "
                  f"(said {m['expected']:.1%}), return {m['roi']:+.1%} [{m['roi_lo']:+.0%}, {m['roi_hi']:+.0%}]")
    report["fit"] = weights
    if args.weights:
        Path(args.weights).write_text(json.dumps({
            "fitted": date.today().isoformat(), "fit_prices": args.fit, "weights": weights}, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
