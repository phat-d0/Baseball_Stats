"""Compare model variants on several held-out months (walk-forward).

    python scripts/evaluate.py --kind pitcher                  # current, pa_simple, pa_seq
    python scripts/evaluate.py --kind batter
    python scripts/evaluate.py --kind pitcher --variants current pa_seq --sims 4000

For each test month every variant is trained only on games before that month. Each
variant returns the full probability distribution for every test row, scored on
(lower is better unless noted):

* logloss    log loss of P(over) across the app's lines (primary)
* rps        ranked probability score of the whole distribution
* calib@L    calibration error at line L: mean |predicted - observed over rate| over ten
             equal-count buckets
* cover50/80 share of results inside the central 50% / 80% range (closer to nominal is better)
* mae        average miss of the expected count (secondary)

Results are reported overall and by cut (pitchers: thirds of recent pitches per start).
The ship gate from the plate-appearance spec is applied against ``current``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseball_stats import features, model, pa_model, sim_models, simulate, storage  # noqa: E402

LINES = {"batter": [0.5, 1.5, 2.5], "pitcher": [3.5, 4.5, 5.5, 6.5, 7.5]}
FOLDS = ["2025-06", "2025-08", "2026-05", "2026-07", "2026-09"]


# --------------------------------------------------------------------------- #
# Measures
# --------------------------------------------------------------------------- #

def line_logloss(y: np.ndarray, dist: np.ndarray, lines: list[float]) -> float:
    ll = []
    for ln in lines:
        p = np.clip(model.p_over(dist, ln), 1e-6, 1 - 1e-6)
        ll.append(-np.where(y > ln, np.log(p), np.log(1 - p)))
    return float(np.mean(np.concatenate(ll)))


nb_logloss = line_logloss  # older name, kept for scripts that import it


def rps(y: np.ndarray, dist: np.ndarray) -> float:
    """Ranked probability score over 0..MAX (last cell is 'MAX or more')."""
    cdf = np.cumsum(dist, axis=1)[:, :-1]
    obs = (np.minimum(y, dist.shape[1] - 1)[:, None] <= np.arange(dist.shape[1] - 1)[None, :])
    return float(np.mean(np.sum((cdf - obs) ** 2, axis=1) / (dist.shape[1] - 1)))


def calib_error(y: np.ndarray, dist: np.ndarray, line: float, buckets: int = 10) -> float:
    p = model.p_over(dist, line)
    hit = (y > line).astype(float)
    order = np.argsort(p)
    gaps = [abs(p[c].mean() - hit[c].mean()) for c in np.array_split(order, buckets) if len(c)]
    return float(np.mean(gaps))


def coverage(y: np.ndarray, dist: np.ndarray, level: float) -> float:
    cdf = np.cumsum(dist, axis=1)
    lo = (cdf < (1 - level) / 2).sum(axis=1)
    hi = (cdf < 1 - (1 - level) / 2).sum(axis=1)
    return float(np.mean((y >= lo) & (y <= hi)))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-6, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


def measures(y: np.ndarray, mu: np.ndarray, dist: np.ndarray, kind: str) -> dict:
    out = {"n": int(len(y)), "logloss": line_logloss(y, dist, LINES[kind]), "rps": rps(y, dist),
           "cover50": coverage(y, dist, 0.5), "cover80": coverage(y, dist, 0.8),
           "mae": float(np.mean(np.abs(y - mu))), "deviance": poisson_deviance(y, mu)}
    for ln in LINES[kind]:
        out[f"calib@{ln:g}"] = calib_error(y, dist, ln)
    return out


# --------------------------------------------------------------------------- #
# Folds and variants
# --------------------------------------------------------------------------- #

@dataclass
class Fold:
    kind: str
    start: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame
    calib: pd.DataFrame
    built: dict
    pa: pd.DataFrame  # plate appearances before the fold (may be empty)
    players: pd.DataFrame
    sims: int


def _pmf_variant(predict):
    """Wrap a point forecast as a distribution: NB with alpha fitted on the calib window."""
    def fn(f: Fold):
        mu_test, mu_calib = predict(f)
        alpha = model.fit_alpha(f.calib[model.TARGETS[f.kind]].to_numpy(), mu_calib)
        return mu_test, model.pmf(mu_test, alpha, model.MAX_COUNT[f.kind])
    return fn


def gbm_variant(kind: str, drop: list[str] | None = None, **params):
    def predict(f: Fold):
        tr, te, ca = (d.drop(columns=[c for c in (drop or []) if c in d]) for d in (f.train, f.test, f.calib))
        m = model.CountModel(kind)
        if params:
            m.reg.set_params(**params)
        m.fit(tr)
        return m.predict(te), m.predict(ca)
    return _pmf_variant(predict)


def components_variant(**params):
    """Hitters: separate models for hits, runs and RBIs; the projection is their sum."""
    def predict(f: Fold):
        mu_t, mu_c = np.zeros(len(f.test)), np.zeros(len(f.calib))
        for part in ("target_h", "target_r", "target_rbi"):
            m = model.CountModel("batter")
            if params:
                m.reg.set_params(**params)
            m.fit(f.train.assign(target_hrr=f.train[part]))
            mu_t += m.predict(f.test)
            mu_c += m.predict(f.calib)
        return mu_t, mu_c
    return _pmf_variant(predict)


def dist_variant():
    """Hitters: the whole distribution from a multiclass model (model.DistModel)."""
    def fn(f: Fold):
        m = model.DistModel("batter").fit(f.train)
        return m.distribution(f.test)
    return fn


def baseline_variant(kind: str):
    return _pmf_variant(lambda f: (model.baseline(f.test, kind), model.baseline(f.calib, kind)))


_PA_CACHE: dict = {}


def _pa_models(f: Fold, stay: bool = True):
    """Outcome/stay models trained on the fold's plate appearances (shared by variants)."""
    if f.pa.empty:
        raise RuntimeError("no plate appearances before this fold")
    key = (f.start, f.kind)
    if key not in _PA_CACHE:
        t0 = time.time()
        outcome = pa_model.OutcomeModel().fit(f.pa, f.built)
        sampler = pa_model.PitchSampler().fit(f.pa)
        stay_m = pa_model.StayModel().fit(f.pa, f.built)
        probs = {}
        if f.kind == "pitcher":  # the lineups facing each test start
            m = simulate.lineup_matchups(f.test, f.built["batter_hrr"], f.players)
            probs = simulate.lineup_probs(m, outcome, f.built)
        _PA_CACHE.clear()
        _PA_CACHE[key] = (outcome, sampler, stay_m, probs)
        print(f"  trained PA models on {len(f.pa)} plate appearances in {time.time() - t0:.0f}s", flush=True)
    return _PA_CACHE[key]


def _sim_model(f: Fold, name: str):
    """The live app's plate-appearance strikeout model, trained on this fold's history."""
    outcome, sampler, stay, _ = _pa_models(f)
    fb = model.CountModel("pitcher").fit(f.train)
    fb.alpha = model.fit_alpha(f.calib["target_k"].to_numpy(), fb.predict(f.calib))
    if name == "pa_simple":
        bf = model.CountModel("pitcher").fit(f.train.assign(target_k=f.train["target_bf"]))
        m = sim_models.PASimpleK(outcome=outcome, fallback=fb, players=f.players, bf_model=bf)
    else:
        m = sim_models.PASeqK(outcome=outcome, fallback=fb, players=f.players, stay=stay,
                              sampler=sampler, n_sims=f.sims)
    return m.for_slate(f.built, f.players)


def pa_seq_variant():
    """Batter-by-batter simulation: outcome model + stay-or-go + pitch counts."""
    return lambda f: _sim_model(f, "pa_seq").distribution(f.test)


def pa_simple_variant():
    """Outcome model for each batter's strikeout chance; batters faced from a direct
    model of target_bf, independent of what happens in the game."""
    return lambda f: _sim_model(f, "pa_simple").distribution(f.test)


def pa_sim_variant():
    """Hitters: simulate each whole game with the plate-appearance models."""
    def fn(f: Fold):
        from baseball_stats import game_sim
        outcome, sampler, stay, _ = _pa_models(f)
        fb = model.CountModel("batter").fit(f.train)
        fb.alpha = model.fit_alpha(f.calib["target_hrr"].to_numpy(), fb.predict(f.calib))
        bg = storage.read("batter_games")
        m = sim_models.GameSimH(
            outcome=outcome, fallback=fb, players=f.players, stay=stay, sampler=sampler,
            transitions=game_sim.Transitions.from_pa(f.pa),
            stay_prob=game_sim.stay_in_lineup(f.pa, bg[bg["game_pk"].isin(f.pa["game_pk"])]),
            n_sims=f.sims)
        return m.for_slate(f.built, f.players).distribution(f.test)
    return fn


def variants_for(kind: str, names: list[str] | None = None) -> dict:
    allv = {
        "batter": {"season_avg": baseline_variant("batter"), "current": gbm_variant("batter"),
                   "components": components_variant(**model.PARAMS["batter"]),
                   "pa_sim": pa_sim_variant(), "dist": dist_variant()},
        "pitcher": {"season_avg": baseline_variant("pitcher"), "current": gbm_variant("pitcher"),
                    "pa_simple": pa_simple_variant(), "pa_seq": pa_seq_variant()},
    }[kind]
    return {k: v for k, v in allv.items() if names is None or k in names}


# --------------------------------------------------------------------------- #
# Cuts, gate, main
# --------------------------------------------------------------------------- #

def cuts(test: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
    out = {"all": np.ones(len(test), bool)}
    if kind == "pitcher" and "pitches_per_start_l5" in test:
        pps = test["pitches_per_start_l5"].to_numpy()
        lo, hi = np.nanpercentile(pps, [33.3, 66.7]) if np.isfinite(pps).any() else (np.nan, np.nan)
        out["short leash"] = pps <= lo
        out["middle"] = (pps > lo) & (pps <= hi)
        out["workhorse"] = pps > hi
    if kind == "batter" and "batting_order" in test:
        bo = test["batting_order"].to_numpy()
        out["slots 1-3"], out["slots 4-6"], out["slots 7-9"] = bo <= 3, (bo >= 4) & (bo <= 6), bo >= 7
    if "lineup_confirmed" in test and (~test["lineup_confirmed"].fillna(True).astype(bool)).any():
        conf = test["lineup_confirmed"].fillna(True).astype(bool).to_numpy()
        out["confirmed"], out["projected"] = conf, ~conf
    return out


def gate(results: list[dict], candidate: str, kind: str) -> dict:
    """The spec's ship gate for ``candidate`` against ``current`` (steps 1-3)."""
    cur = [r["current"]["all"] for r in results]
    new = [r[candidate]["all"] for r in results]
    folds_better = sum(n["logloss"] < c["logloss"] for n, c in zip(new, cur))
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    calib_ok = {f"{ln:g}": mean(new, f"calib@{ln:g}") <= mean(cur, f"calib@{ln:g}") + 0.005
                for ln in LINES[kind]}
    checks = {
        "logloss_mean_lower": mean(new, "logloss") < mean(cur, "logloss"),
        "logloss_lower_in_4_of_5": folds_better >= min(4, len(results)),
        "rps_mean_lower": mean(new, "rps") < mean(cur, "rps"),
        "calibration_within_0.5pp_every_line": all(calib_ok.values()),
    }
    return {"candidate": candidate, "folds_better": folds_better, "folds": len(results),
            "calib_ok": calib_ok, **checks, "passed": all(checks.values())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["batter", "pitcher"], default="pitcher")
    ap.add_argument("--folds", nargs="*", default=FOLDS)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--sims", type=int, default=simulate.N_SIMS)
    ap.add_argument("--out", default=None, help="write results as JSON here")
    args = ap.parse_args(argv)

    t0 = time.time()
    inputs = features.load_inputs()
    built = features.build_features(inputs)
    pa_all = storage.read("plate_appearances")
    print(f"features built in {time.time() - t0:.0f}s; {len(pa_all)} plate appearances", flush=True)
    kind = args.kind
    table = built["batter_hrr" if kind == "batter" else "pitcher_k"]
    rows = model.training_rows(table, kind)
    variants = variants_for(kind, args.variants)
    target = model.TARGETS[kind]
    results = []
    for month in args.folds:
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthEnd(1)
        train, test = rows[rows["game_date"] < start], rows[rows["game_date"].between(start, end)]
        if len(test) < 50:
            continue
        pa = pa_all[pd.to_datetime(pa_all["game_date"]) < start] if not pa_all.empty else pa_all
        f = Fold(kind, start, train, test, train[train["game_date"] >= start - pd.Timedelta(days=30)],
                 built, pa, inputs["players"], args.sims)
        y = test[target].to_numpy()
        fold_res = {"month": month, "n": len(test)}
        for name, fn in variants.items():
            t1 = time.time()
            try:
                mu, dist = fn(f)
            except RuntimeError as exc:
                print(f"  {month} {name}: skipped ({exc})", flush=True)
                continue
            fold_res[name] = {c: measures(y[m], mu[m], dist[m], kind)
                              for c, m in cuts(test, kind).items() if m.sum() >= 30}
            a = fold_res[name]["all"]
            print(f"{kind} {month} {name:11s} n={a['n']} logloss {a['logloss']:.4f} rps {a['rps']:.4f} "
                  f"mae {a['mae']:.3f} cover50 {a['cover50']:.2f} cover80 {a['cover80']:.2f} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        results.append(fold_res)

    names = [v for v in variants if all(v in r for r in results)]
    print(f"\n== {kind}: mean over {len(results)} folds")
    summary = {}
    for v in names:
        cut_names = sorted({c for r in results for c in r[v]}, key=lambda c: (c != "all", c))
        summary[v] = {}
        for c in cut_names:
            rs = [r[v][c] for r in results if c in r[v]]
            m = {k: float(np.mean([x[k] for x in rs])) for k in rs[0] if k != "n"}
            m["n"] = int(sum(x["n"] for x in rs))
            summary[v][c] = m
            calib = " ".join(f"{k[6:]}:{m[k] * 100:.1f}" for k in m if k.startswith("calib@"))
            print(f"  {v:11s} {c:12s} n={m['n']:6d} logloss {m['logloss']:.4f} rps {m['rps']:.4f} "
                  f"mae {m['mae']:.3f} cover50 {m['cover50']:.2f} cover80 {m['cover80']:.2f} calib(pp) {calib}")
    gates = {v: gate(results, v, kind) for v in names if v not in ("current", "season_avg")}
    for v, g in gates.items():
        print(f"\nGATE {v} vs current: {'PASSED' if g['passed'] else 'NOT PASSED'} -> "
              + ", ".join(f"{k}={g[k]}" for k in g if k not in ("candidate", "calib_ok", "passed")))
    if args.out:
        Path(args.out).write_text(json.dumps({"folds": results, "summary": summary, "gates": gates},
                                             indent=1, default=float))


if __name__ == "__main__":
    main()
