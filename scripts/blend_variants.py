"""Test refinements of the model/DraftKings blend out of sample.

Every variant is a logistic regression per kind, fitted on one set of historical prices
and scored on a later one, like scripts/fit_blend.py. With ``d = logit(p_model) -
logit(p_book)`` (how far the model leans away from DraftKings):

- ``base``: the live blend, ``a * logit(p_model) + b * logit(p_book) + c``.
- ``side``: separate weights for leaning over (d > 0) and under (d < 0).
- ``line``: base fitted separately per line (lines with too few fit rows use base).
- ``shrink``: adds ``d * |d|``; a negative weight trusts big disagreements less.
- ``side_shrink``: both.

    python scripts/blend_variants.py --fit odds_history/2025 --test odds_history/2026 \
        --cache .blend_cache --report docs/edge_blend_variants.json

A confirmed vs projected lineup term can't be tested here: historical prices are
graded with the lineups actually used, so every row is "confirmed".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fit_blend  # noqa: E402
from baseball_stats import odds, tracking  # noqa: E402

MIN_LINE_ROWS = 300
BOOT = 2000


def design(g: pd.DataFrame, variant: str) -> np.ndarray:
    lm, lb = fit_blend.logit(g["p_model"]), fit_blend.logit(g["p_book"])
    d = lm - lb
    cols = {"base": [lm, lb], "line": [lm, lb],
            "side": [lb, np.maximum(d, 0), np.minimum(d, 0)],
            "shrink": [lb, d, d * np.abs(d)],
            "side_shrink": [lb, np.maximum(d, 0), np.minimum(d, 0), d * np.abs(d)]}[variant]
    return np.column_stack(cols)


def fit_predict(fit: pd.DataFrame, test: pd.DataFrame, variant: str) -> tuple[np.ndarray, dict]:
    def one(f, t, v):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(design(f, v), f["over_won"])
        return m.predict_proba(design(t, v))[:, 1], [*map(float, m.coef_[0]), float(m.intercept_[0])]

    if variant != "line":
        return one(fit, test, variant)
    p, coefs = one(fit, test, "base")
    by_line = {}
    for line, f in fit.groupby("line"):
        mask = (test["line"] == line).to_numpy()
        if len(f) >= MIN_LINE_ROWS and mask.any():
            p[mask], by_line[f"{line:g}"] = one(f, test[mask], "base")
    return p, {"all": coefs, "by_line": by_line}


def with_blend(g: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    g = g.copy()
    g["p_blend"] = p
    g["ev_over"] = p * g["over"].map(odds.american_to_decimal).astype(float) - 1
    g["ev_under"] = (1 - p) * g["under"].map(odds.american_to_decimal).astype(float) - 1
    return g


def ll_rows(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, bool)
    return -np.where(y, np.log(p), np.log(1 - p))


def paired_gap(test: pd.DataFrame, p_new, p_base) -> dict:
    """Log loss of a variant minus base, with a 90% range from resampling whole games."""
    diff = pd.Series(ll_rows(p_new, test["over_won"]) - ll_rows(p_base, test["over_won"]))
    per_game = diff.groupby(test["game_pk"].to_numpy()).agg(["sum", "count"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per_game), size=(BOOT, len(per_game)))
    s, n = per_game["sum"].to_numpy(), per_game["count"].to_numpy()
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"gap": float(diff.mean()), "gap_lo": float(np.percentile(boot, 5)),
            "gap_hi": float(np.percentile(boot, 95))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cache", default=None, help="folder of graded lines from fit_blend.py --cache")
    ap.add_argument("--variants", nargs="+", default=["base", "side", "line", "shrink", "side_shrink"])
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    cache = (lambda f: str(Path(args.cache) / f"{Path(f).name}.parquet")) if args.cache else (lambda f: None)
    g_fit, g_test = fit_blend.load(args.fit, cache(args.fit)), fit_blend.load(args.test, cache(args.test))
    report: dict = {}
    blended: dict[str, list] = {v: [] for v in args.variants}
    for kind in ("batter", "pitcher"):
        f = g_fit[g_fit["kind"] == kind].reset_index(drop=True)
        t = g_test[g_test["kind"] == kind].reset_index(drop=True)
        p_base, _ = fit_predict(f, t, "base")
        report[kind] = {"lines_fit": int(len(f)), "lines_test": int(len(t)),
                        "logloss_model": tracking._logloss(t["p_model"], t["over_won"]),
                        "logloss_book": tracking._logloss(t["p_book"], t["over_won"]), "variants": {}}
        print(f"{kind}: fit {len(f)}, test {len(t)} lines; log loss model "
              f"{report[kind]['logloss_model']:.4f} book {report[kind]['logloss_book']:.4f}")
        for v in args.variants:
            p, coefs = fit_predict(f, t, v)
            b = with_blend(t, p)
            blended[v].append(b)
            res = {"coef": coefs, "logloss": tracking._logloss(pd.Series(p), t["over_won"]),
                   **paired_gap(t, p, p_base), **fit_blend.returns(tracking.picks(b, -1.0))}
            report[kind]["variants"][v] = res
            m1 = res["thresholds"]["0.01"]
            print(f"  {v:12s} log loss {res['logloss']:.4f} (vs base {res['gap']:+.5f} "
                  f"[{res['gap_lo']:+.5f}, {res['gap_hi']:+.5f}]); 1%+: {m1.get('n', 0)} bets, "
                  + (f"won {m1['win']:.1%} said {m1['expected']:.1%} return {m1['roi']:+.1%} "
                     f"[{m1['roi_lo']:+.0%}, {m1['roi_hi']:+.0%}]" if m1.get("n") else ""), flush=True)
    report["all"] = {}
    for v, parts in blended.items():
        r = fit_blend.returns(tracking.picks(pd.concat(parts, ignore_index=True), -1.0))
        report["all"][v] = r
        print(f"all {v}:")
        for th, m in r["thresholds"].items():
            if m.get("n"):
                print(f"  >= {float(th):.0%}: {m['n']} bets, won {m['win']:.1%} (said {m['expected']:.1%}), "
                      f"return {m['roi']:+.1%} [{m['roi_lo']:+.0%}, {m['roi_hi']:+.0%}]")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
