"""Backtest the paper strategies and bet sizing on historical DraftKings prices.

Prices come from scripts/fetch_history.py (one snapshot per game, an hour before first
pitch). The model chances are walk-forward: for each month, the live models (hitters:
the game-level model; strikeouts: pa_simple) are trained only on games before it, as
in scripts/evaluate.py. Lineups are the ones actually used, so this is the "confirmed
lineup" case.

    python scripts/backtest.py --prices odds_history --out docs/backtest_2026.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from baseball_stats import features, model, odds, storage, tracking  # noqa: E402

BANDS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, None)]
STRATEGIES = {"12%+": (0.12, None), "8–12%": (0.08, 0.12), "5–12%": (0.05, 0.12),
              "8%+": (0.08, None), "5%+": (0.05, None)}
STAKE = 10.0


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #

def load_prices(folder: Path) -> list[dict]:
    snaps = []
    for f in sorted(folder.glob("*.json")):
        day = json.loads(f.read_text())
        for s in day["snapshots"]:
            snaps.append({"day": day["day"], "snapshot": s["snapshot"], "event": s["event"]})
    return snaps


def price_rows(snaps: list[dict], games: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, player, kind, line) with DraftKings prices, matched to MLB ids.

    ``roster``: game_pk, player_id, kind, name for the players we have predictions for.
    """
    games = games.assign(game_date=pd.to_datetime(games["game_date"]).dt.date.astype(str))
    by_day = {d: g for d, g in games.groupby("game_date")}
    names = {pk: list(zip(r["player_id"], r["name"])) for pk, r in roster.groupby("game_pk")}
    rows = []
    for s in snaps:
        cand = pd.concat([by_day.get(s["day"], games.iloc[:0]),
                          by_day.get((pd.Timestamp(s["day"]) + pd.Timedelta(days=1)).date().isoformat(),
                                     games.iloc[:0])])
        matched = odds.match_events([s["event"]], cand)
        if not matched:
            continue
        pk = next(iter(matched))
        idx = odds.PlayerIndex(names.get(pk, []))
        for p in odds.parse_props(s["event"]):
            pid = idx.find(p["player"])
            if pid is None or p.get("over") is None or p.get("under") is None:
                continue
            rows.append({"game_pk": pk, "player_id": pid, "kind": p["kind"], "player_name": p["player"],
                         "line": p["line"], "over": p["over"], "under": p["under"],
                         "fetched_at": pd.Timestamp(s["snapshot"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Walk-forward predictions
# --------------------------------------------------------------------------- #

def predictions(months: list[str], sims: int) -> tuple[pd.DataFrame, dict]:
    """Test rows of every month with ``pmf`` (list) for hitters and starters."""
    inputs = features.load_inputs()
    built = features.build_features(inputs)
    pa_all = storage.read("plate_appearances")
    out = []
    for kind, fn in (("batter", evaluate.gbm_variant("batter")), ("pitcher", evaluate.pa_simple_variant())):
        rows = model.training_rows(built["batter_hrr" if kind == "batter" else "pitcher_k"], kind)
        for month in months:
            start = pd.Timestamp(f"{month}-01")
            end = start + pd.offsets.MonthEnd(1)
            train, test = rows[rows["game_date"] < start], rows[rows["game_date"].between(start, end)]
            if len(test) < 50:
                continue
            pa = pa_all[pd.to_datetime(pa_all["game_date"]) < start]
            f = evaluate.Fold(kind, start, train, test,
                              train[train["game_date"] >= start - pd.Timedelta(days=30)],
                              built, pa, inputs["players"], sims)
            t0 = time.time()
            mu, dist = fn(f)
            print(f"  {kind} {month}: {len(test)} predictions in {time.time() - t0:.0f}s", flush=True)
            out.append(pd.DataFrame({"game_pk": test["game_pk"].astype(int).to_numpy(),
                                     "player_id": test["player_id"].astype(int).to_numpy(),
                                     "kind": kind, "game_date": test["game_date"].to_numpy(),
                                     "actual": test[model.TARGETS[kind]].to_numpy(),
                                     "mu": mu, "pmf": list(dist)}))
    return pd.concat(out, ignore_index=True), inputs


def grades(prices: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Rows shaped like the live price log's grades, so tracking.picks works on them."""
    g = prices.merge(pred, on=["game_pk", "player_id", "kind"], how="inner")
    vals = [tracking.line_values({"line": r.line, "over": r.over, "under": r.under}, np.asarray(r.pmf))
            for r in g.itertuples(index=False)]
    g = pd.concat([g.drop(columns=["pmf"]).reset_index(drop=True), pd.DataFrame(vals)], axis=1)
    g["status"] = "graded"
    g["over_won"] = g["actual"] > g["line"]
    g["clv_over"] = np.nan
    g["close_line"] = np.nan
    g["lineup_confirmed"] = True
    g["is_close"] = True
    return g


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, bool)
    return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))


def band_table(p: pd.DataFrame) -> list[dict]:
    out = []
    for lo, hi in BANDS:
        sel = p[(p["ev"] >= lo) & ((p["ev"] < hi) if hi else True)]
        out.append({"lo": lo, "hi": hi, **tracking.pick_metrics(sel)})
    return out


def strategy(p: pd.DataFrame, lo: float, hi: float | None) -> pd.DataFrame:
    s = p[(p["ev"] >= lo) & ((p["ev"] < hi) if hi else True)].copy()
    dec = s["price"].map(odds.american_to_decimal).astype(float)
    s["profit"] = np.where(s["won"].astype(bool), STAKE * (dec - 1), -STAKE)
    return s.sort_values("fetched_at")


def summarize_strategy(s: pd.DataFrame) -> dict:
    if s.empty:
        return {"n": 0}
    m = tracking.pick_metrics(s)
    daily = s.groupby(pd.to_datetime(s["game_date"]).dt.date)["profit"].sum()
    cum = daily.cumsum()
    return {**m, "profit": float(s["profit"].sum()), "staked": float(STAKE * len(s)),
            "max_drawdown": float((cum.cummax().clip(lower=0) - cum).max()),
            "by_kind": {k: tracking.pick_metrics(x) for k, x in s.groupby("kind")}}


def blend_weight(g: pd.DataFrame) -> float:
    """w minimising log loss of w * model + (1 - w) * DraftKings no-vig, over lines."""
    ws = np.linspace(0, 1, 101)
    ll = [_logloss(w * g["p_model"] + (1 - w) * g["p_book"], g["over_won"]) for w in ws]
    return float(ws[int(np.argmin(ll))])


def sizing(p_all: pd.DataFrame, fit_month: str, months: list[str], bankroll: float = 1000.0) -> dict:
    """Flat $10 vs fractional Kelly on a model/market blend fitted on ``fit_month``.

    Every line where the blended chance still shows an edge is a candidate; one bet per
    player and prop (the best side), sized by quarter Kelly and capped at 2% of bankroll.
    """
    fit = p_all[p_all["month"] == fit_month]
    out = {"fit_month": fit_month, "weights": {}}
    for kind in ("batter", "pitcher"):
        k = fit[fit["kind"] == kind]
        out["weights"][kind] = blend_weight(k) if len(k) > 100 else 0.5
    test = p_all[p_all["month"].isin([m for m in months if m != fit_month])].copy()
    w = test["kind"].map(out["weights"]).astype(float)
    test["p_blend"] = w * test["p_model"] + (1 - w) * test["p_book"]
    rows = []
    for (_, _, _), grp in test.groupby(["game_pk", "player_id", "kind"]):
        best = None
        for r in grp.itertuples(index=False):
            for side in ("over", "under"):
                price = getattr(r, side)
                dec = odds.american_to_decimal(price)
                p = r.p_blend if side == "over" else 1 - r.p_blend
                ev = p * dec - 1
                if best is None or ev > best["ev"]:
                    won = r.over_won if side == "over" else not r.over_won
                    best = {"ev": ev, "p": p, "dec": dec, "won": bool(won), "game_date": r.game_date,
                            "kind": r.kind, "ev_model": (r.p_model if side == "over" else 1 - r.p_model) * dec - 1}
        if best and best["ev"] > 0:
            rows.append(best)
    bets = pd.DataFrame(rows).sort_values("game_date") if rows else pd.DataFrame()
    out["test_months"] = sorted(set(test["month"]))
    out["n_lines"] = int(len(test))
    out["logloss"] = {k: {"model": _logloss(x["p_model"], x["over_won"]),
                          "book": _logloss(x["p_book"], x["over_won"]),
                          "blend": _logloss(x["p_blend"], x["over_won"])}
                      for k, x in test.groupby("kind")}
    if bets.empty:
        out["bets"] = 0
        return out
    res = {}
    for name, frac, cap in (("quarter_kelly", 0.25, 0.02), ("half_kelly", 0.5, 0.04)):
        bank = bankroll
        for day, grp in bets.groupby(pd.to_datetime(bets["game_date"]).dt.date):
            start = bank  # stakes for a day's games are set from the morning bankroll
            for b in grp.itertuples(index=False):
                f = max((b.p * b.dec - 1) / (b.dec - 1), 0) * frac
                stake = min(f, cap) * start
                bank += stake * (b.dec - 1) if b.won else -stake
        res[name] = {"start": bankroll, "end": float(bank), "return": float(bank / bankroll - 1)}
    flat = np.where(bets["won"], STAKE * (bets["dec"] - 1), -STAKE)
    res["flat_10"] = {"bets": int(len(bets)), "profit": float(flat.sum()),
                      "roi": float(flat.sum() / (STAKE * len(bets)))}
    for lo in (0.02, 0.05):
        sel = bets[bets["ev"] >= lo]
        pr = np.where(sel["won"], STAKE * (sel["dec"] - 1), -STAKE)
        res[f"flat_10_blend_edge_{int(lo * 100)}"] = {
            "bets": int(len(sel)), "profit": float(pr.sum()),
            "roi": float(pr.sum() / (STAKE * len(sel))) if len(sel) else None,
            "win": float(sel["won"].mean()) if len(sel) else None,
            "expected": float(sel["p"].mean()) if len(sel) else None}
    out["bets"] = int(len(bets))
    out["results"] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="odds_history")
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    snaps = load_prices(Path(args.prices))
    if not snaps:
        print("no price files found", file=sys.stderr)
        return 1
    days = sorted({s["day"] for s in snaps})
    months = sorted({d[:7] for d in days})
    print(f"{len(snaps)} priced games, {days[0]} .. {days[-1]}", flush=True)

    pred, inputs = predictions(months, args.sims)
    players = inputs["players"].set_index("player_id")["full_name"]
    roster = pred[["game_pk", "player_id", "kind"]].assign(name=pred["player_id"].map(players))
    roster = roster[roster["name"].notna()]
    prices = price_rows(snaps, inputs["games"], roster)
    g = grades(prices, pred)
    g["month"] = pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m")
    print(f"{len(prices)} priced lines matched, {len(g)} with predictions", flush=True)

    result: dict = {"days": [days[0], days[-1]], "games_priced": len(snaps), "lines": int(len(g)),
                    "by_kind": {}, "strategies": {}}
    for kind, k in g.groupby("kind"):
        p = tracking.picks(k, -1.0)  # every player's best side, whatever the edge
        result["by_kind"][kind] = {
            "lines": int(len(k)),
            "logloss_model": _logloss(k["p_model"], k["over_won"]),
            "logloss_book": _logloss(k["p_book"], k["over_won"]),
            "picks": int(len(p)),
            "bands": band_table(p),
        }
    allp = tracking.picks(g, -1.0)
    result["bands_all"] = band_table(allp)
    for name, (lo, hi) in STRATEGIES.items():
        result["strategies"][name] = summarize_strategy(strategy(allp, lo, hi))
    if len(months) > 1:
        result["sizing"] = sizing(g, months[0], months)

    print(json.dumps(result, indent=1, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
