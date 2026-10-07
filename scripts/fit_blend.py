"""Fit the model/DraftKings blend (baseball_stats/blend.py) on historical prices, and
test it on later prices or month by month.

    python scripts/fit_blend.py --fit odds_history/2025 --test odds_history/2026 \
        --weights baseball_stats/blend.json --report docs/blend_2026.json

Fits ``logit(p) = a * logit(p_model) + b * logit(p_book) + c`` per kind on the fit set
(walk-forward model chances, as in scripts/backtest.py), shows how stable the weights
are month to month, then scores the blend on the test set: log loss against the model
and DraftKings, and the return of betting its edges. Without ``--test`` (a refit on
everything, see scripts/refit_blend.sh) it scores the blend walk-forward instead: each
month fitted only on the months before it.

``--live`` adds the live price log's closing lines (graded, at most an hour before
first pitch, like the historical snapshots). Their ``p_model`` is whatever model was
live when they were logged.
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
from baseball_stats import odds, storage, tracking  # noqa: E402

THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05]
BANDS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, None)]


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(g: pd.DataFrame) -> dict:
    m = LogisticRegression(C=1e6).fit(np.c_[logit(g["p_model"]), logit(g["p_book"])], g["over_won"])
    a, b = m.coef_[0]
    return {"model": float(a), "book": float(b), "intercept": float(m.intercept_[0]), "lines": int(len(g))}


def apply(g: pd.DataFrame, w: dict, sd: float | None = None) -> pd.DataFrame:
    """The blend's chance and each side's expected return; with ``sd`` (the kind's
    sigma) each side's z too, as tracking.line_values works it out live."""
    z = w["model"] * logit(g["p_model"]) + w["book"] * logit(g["p_book"]) + w["intercept"]
    g = g.copy()
    g["p_blend"] = 1 / (1 + np.exp(-z))
    do = g["over"].map(odds.american_to_decimal).astype(float)
    du = g["under"].map(odds.american_to_decimal).astype(float)
    g["ev_over"] = g["p_blend"] * do - 1
    g["ev_under"] = (1 - g["p_blend"]) * du - 1
    if sd:
        g["z_over"] = (g["p_blend"] - 1 / do) / sd
        g["z_under"] = ((1 - g["p_blend"]) - 1 / du) / sd
    return g


def sigma(g: pd.DataFrame, w: dict) -> float:
    """How far the blend usually strays from DraftKings: the sd of p_blend - p_book."""
    b = apply(g, w)
    return float((b["p_blend"] - b["p_book"]).std(ddof=0))


def tiers(both: pd.DataFrame) -> dict:
    """Bets by confidence tier (tracking.EDGE_TIERS), on rows that carry z."""
    return {t["key"]: tracking.tier_metrics(tracking.picks(both, t["min_z"], by="z")) for t in tracking.EDGE_TIERS}


def load(folder: str, cache: str | None = None, batter: str = "current") -> pd.DataFrame:
    """Graded lines with walk-forward model chances. The predictions take a while, so
    ``cache`` (a parquet path) keeps them for the next run on the same prices."""
    if cache and Path(cache).exists():
        g = pd.read_parquet(cache)
        print(f"{folder}: {len(g)} lines from {cache}", flush=True)
        return g
    snaps = backtest.load_prices(Path(folder))
    months = sorted({s["day"][:7] for s in snaps})
    pred, inputs = backtest.predictions(months, 10_000, batter)
    names = inputs["players"].set_index("player_id")["full_name"]
    roster = pred[["game_pk", "player_id", "kind"]].assign(name=pred["player_id"].map(names)).dropna()
    g = backtest.grades(backtest.price_rows(snaps, inputs["games"], roster), pred)
    g["month"] = pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m")
    print(f"{folder}: {len(snaps)} games, {len(g)} lines", flush=True)
    if cache:
        g.to_parquet(cache)
    return g


def load_live(path: str) -> pd.DataFrame:
    """Closing lines from the live price log, graded against the stored box scores."""
    g = tracking.grade(pd.read_parquet(path), storage.read("games"), storage.read("batter_games"),
                       storage.read("pitcher_games"))
    if g.empty:
        return g
    g = g[(g["status"] == "graded") & g["is_close"].astype(bool) & g["p_book"].notna()
          & g["p_model"].notna() & (g["hours_before"] * 60 <= tracking.CLOSE_MAX_MINUTES)].copy()
    g["over_won"] = g["over_won"].astype(bool)
    g["month"] = pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m")
    print(f"{path}: {len(g)} live closing lines", flush=True)
    return g


def walk_forward(g: pd.DataFrame, min_months: int = 3) -> dict:
    """Each month scored with weights fitted only on the months before it."""
    out = {}
    months = sorted(g["month"].unique())
    for m in months[min_months:]:
        before, t = g[g["month"] < m], g[g["month"] == m]
        if len(t) < 100:
            continue
        t = apply(t, fit(before))
        ll = lambda p: tracking._logloss(p, t["over_won"])  # noqa: E731
        out[m] = {"lines": int(len(t)), "logloss_model": ll(t["p_model"]),
                  "logloss_book": ll(t["p_book"]), "logloss_blend": ll(t["p_blend"])}
    return out


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
    ap.add_argument("--fit", required=True, nargs="+", help="folders of historical prices to fit on")
    ap.add_argument("--test", default=None, help="folder of later historical prices to test on")
    ap.add_argument("--live", default=None, help="price log (prop_snapshots.parquet) to add to the fit")
    ap.add_argument("--weights", default=None, help="write the fitted weights here")
    ap.add_argument("--report", default=None)
    ap.add_argument("--batter", default="current", help="hitter model (an evaluate.py variant)")
    ap.add_argument("--cache", default=None,
                    help="folder to keep the graded lines in, by folder and hitter model (delete it after a model change)")
    args = ap.parse_args(argv)

    cache = ((lambda f: str(Path(args.cache) / f"{Path(f).name}-{args.batter}.parquet")) if args.cache
             else (lambda f: None))
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
    parts = [load(f, cache(f), args.batter) for f in args.fit]
    if args.live:
        parts.append(load_live(args.live))
    g_fit = pd.concat([x for x in parts if not x.empty], ignore_index=True)
    report: dict = {"fit_prices": args.fit, "live": args.live, "batter_model": args.batter,
                    "fit": {}, "by_month": {}, "test": {}}
    weights = {}
    for kind in ("batter", "pitcher"):
        k = g_fit[g_fit["kind"] == kind]
        weights[kind] = fit(k)
        report["by_month"][kind] = {m: fit(x) for m, x in k.groupby("month") if len(x) >= 200}
        print(kind, json.dumps(weights[kind]), {m: round(w["model"], 2) for m, w in report["by_month"][kind].items()})
    report["fit"] = weights
    sigmas = {k: sigma(g_fit[g_fit["kind"] == k], weights[k]) for k in weights}
    report["sigma"] = sigmas
    print("sigma (sd of p_blend - p_book on the fit prices):", {k: round(v, 4) for k, v in sigmas.items()})
    if args.test:
        test(load(args.test, cache(args.test), args.batter), weights, report, sigmas)
    else:
        report["walk_forward"] = {}
        for kind in weights:
            wf = report["walk_forward"][kind] = walk_forward(g_fit[g_fit["kind"] == kind])
            for m, r in wf.items():
                print(f"  {kind} {m} (fitted on earlier months): log loss model {r['logloss_model']:.4f} "
                      f"book {r['logloss_book']:.4f} blend {r['logloss_blend']:.4f}")
    if args.weights:
        days = pd.to_datetime(g_fit["game_date"]).dt.date
        live = f", {len(parts[-1])} live closing lines" if args.live else ""
        Path(args.weights).write_text(json.dumps({
            "fitted": date.today().isoformat(),
            "fit_prices": f"historical prices {days.min()}..{days.max()}{live}", "weights": weights,
            "sigma": sigmas}, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=float))
    return 0


def test(g_test: pd.DataFrame, weights: dict, report: dict, sigmas: dict | None = None) -> None:
    for kind in ("batter", "pitcher"):
        t = apply(g_test[g_test["kind"] == kind], weights[kind])
        ll = lambda p: tracking._logloss(p, t["over_won"])  # noqa: E731
        report["test"][kind] = {"lines": int(len(t)), "logloss_model": ll(t["p_model"]),
                                "logloss_book": ll(t["p_book"]), "logloss_blend": ll(t["p_blend"]),
                                **returns(tracking.picks(t, -1.0))}
        print(f"  {kind} test log loss model %.4f book %.4f blend %.4f" % (
            report["test"][kind]["logloss_model"], report["test"][kind]["logloss_book"],
            report["test"][kind]["logloss_blend"]))
    both = pd.concat([apply(g_test[g_test["kind"] == k], weights[k], (sigmas or {}).get(k)) for k in weights])
    report["test"]["all"] = returns(tracking.picks(both, -1.0))
    if sigmas:
        report["test"]["tiers"] = tiers(both)
        for key, m in report["test"]["tiers"].items():
            if m.get("n"):
                print(f"  tier {key}: {m['n']} bets, won {m['win']:.1%} (break-even {m['breakeven']:.1%}, "
                      f"z {m['z_realized']:+.2f}), return {m['roi']:+.1%} [{m['roi_lo']:+.0%}, {m['roi_hi']:+.0%}]")
    for t, m in report["test"]["all"]["thresholds"].items():
        if m.get("n"):
            print(f"  blended edge >= {float(t):.0%}: {m['n']} bets, won {m['win']:.1%} "
                  f"(said {m['expected']:.1%}), return {m['roi']:+.1%} [{m['roi_lo']:+.0%}, {m['roi_hi']:+.0%}]")


if __name__ == "__main__":
    raise SystemExit(main())
