"""A permanent log of DraftKings prices with the model's numbers, graded after each game.

``data.json`` is rebuilt on every run, so without this nothing the app knew about a
price survives the next refresh. Here every prop price the app downloads is stored
once, next to the model's probability at that moment (``prop_snapshots``, append-only),
and later graded against the box score and the closing line (``prop_grades``, rebuilt
in full from the log on every run so grading rules can change without losing anything).

No network calls in this module, so it can be unit tested like ``parse.py``.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import blend, model, odds, storage

EASTERN = ZoneInfo("America/New_York")
SNAPSHOT_KEY = ["game_pk", "player_id", "kind", "line", "fetched_at"]


def line_values(entry: dict, pmf: np.ndarray, kind: str | None = None) -> dict:
    """Model chance, DraftKings' no-vig chance, the blend of the two (see ``blend``) and
    the expected return per $1 on each side at the blended chance.

    The one place these are worked out, so the log stores exactly what the app shows.
    """
    p = float(model.p_over(pmf[None, :], entry["line"])[0])
    p_book = odds.no_vig_over(entry.get("over"), entry.get("under"))
    pb = blend.blended(p, p_book, kind)
    do = odds.american_to_decimal(entry.get("over"))
    du = odds.american_to_decimal(entry.get("under"))
    return {
        "p_model": p,
        "p_book": p_book,
        "p_blend": pb,
        "ev_over": pb * do - 1 if do else None,
        "ev_under": (1 - pb) * du - 1 if du else None,
    }


def _ts(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def snapshot_rows(slate: dict, models: dict[str, model.CountModel], props: dict,
                  shadow=None) -> pd.DataFrame:
    """``prop_snapshots`` rows for every player on the slate with DraftKings lines."""
    games = slate["games"].set_index("game_pk")
    sha = os.environ.get("GITHUB_SHA", "local")
    rows = []
    for kind, table in (("pitcher", slate["pitcher_k"]), ("batter", slate["batter_hrr"])):
        if table.empty:
            continue
        keys = list(zip(table["game_pk"], table["player_id"]))
        have = [i for i, (pk, pid) in enumerate(keys) if (pk, pid, kind) in props]
        if not have:
            continue
        sub = table.iloc[have]
        active = models[kind].for_slate(slate)
        mu, dist = active.distribution(sub)
        sh_model = shadow.get(kind) if isinstance(shadow, dict) else (shadow if kind == "pitcher" else None)
        sh_dist = None
        if sh_model is not None:
            _, sh_dist = sh_model.for_slate(slate).distribution(sub)
        for j, ((_, r), m, d) in enumerate(zip(sub.iterrows(), mu, dist)):
            g = games.loc[r["game_pk"]]
            start = _ts(g.get("game_datetime"))
            for e in props[(r["game_pk"], r["player_id"], kind)]:
                if not e.get("fetched_at"):
                    continue
                rows.append({
                    "fetched_at": _ts(e["fetched_at"]),
                    "book_updated": _ts(e.get("updated")),
                    "game_pk": int(r["game_pk"]),
                    "game_date": pd.Timestamp(g.get("game_date")).normalize(),
                    "game_start": start,
                    "player_id": int(r["player_id"]),
                    "player_name": r.get("player_name"),
                    "kind": kind,
                    "line": float(e["line"]),
                    "over": e.get("over"),
                    "under": e.get("under"),
                    "mu": float(m),
                    "alpha": None if active.alpha is None else float(active.alpha),
                    "model_name": active.name,
                    "p_shadow": float(model.p_over(sh_dist[j][None, :], e["line"])[0])
                    if sh_dist is not None else None,
                    "shadow_name": sh_model.name if sh_dist is not None else None,
                    **line_values(e, d, kind),
                    "lineup_confirmed": bool(r.get("lineup_confirmed", True))
                    if kind == "batter" else True,
                    "batting_order": r.get("batting_order") if kind == "batter" else None,
                    "code_sha": sha,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("over", "under", "batting_order"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("p_book", "p_blend", "ev_over", "ev_under", "alpha", "p_shadow"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def log_snapshots(slate: dict, models: dict[str, model.CountModel], props: dict,
                  now: datetime | None = None, base=None, shadow=None) -> pd.DataFrame:
    """Append new price snapshots to ``prop_snapshots``; returns the rows added.

    Only downloads not already stored are added: re-running a publish on the same
    cached prices adds nothing, and earlier rows are never changed. ``now`` is unused
    beyond the signature the spec names; ``fetched_at`` comes from the ledger.
    """
    new = snapshot_rows(slate, models, props, shadow=shadow)
    if new.empty:
        return new
    old = storage.read("prop_snapshots", base)
    if not old.empty:
        seen = pd.MultiIndex.from_frame(old[SNAPSHOT_KEY].assign(
            fetched_at=pd.to_datetime(old["fetched_at"], utc=True)))
        idx = pd.MultiIndex.from_frame(new[SNAPSHOT_KEY])
        new = new[~idx.isin(seen)]
    if not new.empty:
        storage.upsert("prop_snapshots", new, base)
    return new.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

CLOSE_MAX_MINUTES = 90  # a "closing" price must be taken this close to first pitch
NOT_PLAYED = {"Postponed", "Cancelled"}


def void_reason(kind: str, game: dict | None, box: dict | None, line: float,
                actual: float | None, game_date) -> str | None:
    """Why a snapshot can't be graded as a bet (None = it stands).

    One small function on purpose: sportsbooks differ on edge cases (e.g. whether a
    batter must start or only get a plate appearance), so check DraftKings' current
    house rules before trusting these.
    """
    if game is None:
        return None  # not known yet -> pending, handled by the caller
    if game.get("detailed_state") in NOT_PLAYED:
        return "not played"
    if pd.Timestamp(game.get("game_date")).normalize() != pd.Timestamp(game_date).normalize():
        return "played on another day"  # postponed and replayed under the same game_pk
    if box is None:
        return "no box score"
    if not bool(box.get("is_starter")):
        return "did not start"
    if actual is not None and float(line).is_integer() and actual == line:
        return "push"
    return None


def grade(snapshots: pd.DataFrame, games: pd.DataFrame, batter_games: pd.DataFrame,
          pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """``prop_grades``: every snapshot with its result and closing-line comparison."""
    if snapshots.empty:
        return pd.DataFrame()
    s = snapshots.copy()
    s["fetched_at"] = pd.to_datetime(s["fetched_at"], utc=True)
    s["game_start"] = pd.to_datetime(s["game_start"], utc=True)
    g = games.drop_duplicates("game_pk").set_index("game_pk") if not games.empty else pd.DataFrame()
    boxes = {
        "batter": batter_games.drop_duplicates(["game_pk", "player_id"]).set_index(["game_pk", "player_id"])
        if not batter_games.empty else pd.DataFrame(),
        "pitcher": pitcher_games.drop_duplicates(["game_pk", "player_id"]).set_index(["game_pk", "player_id"])
        if not pitcher_games.empty else pd.DataFrame(),
    }
    stat = {"batter": "hrr", "pitcher": "k"}

    status, actual, reason = [], [], []
    for r in s.itertuples(index=False):
        game = g.loc[r.game_pk].to_dict() if r.game_pk in g.index else None
        finished = (game is not None and game.get("status") == "Final"
                    and game.get("detailed_state") != "Suspended")
        if game is not None and game.get("detailed_state") in NOT_PLAYED:
            finished = True
        if not finished:
            status.append("pending"); actual.append(None); reason.append(None)
            continue
        b = boxes[r.kind]
        key = (r.game_pk, r.player_id)
        box = b.loc[key].to_dict() if not b.empty and key in b.index else None
        a = float(box[stat[r.kind]]) if box is not None and pd.notna(box.get(stat[r.kind])) else None
        why = void_reason(r.kind, game, box, r.line, a, r.game_date)
        status.append("void" if why else "graded")
        actual.append(a if not why else None)
        reason.append(why)
    s["status"] = status
    s["actual"] = pd.array([None if x is None else int(x) for x in actual], dtype="Int64")
    s["void_reason"] = reason
    s["over_won"] = pd.array([None if x is None else bool(x > ln) for x, ln in zip(actual, s["line"])],
                             dtype="boolean")
    s["hours_before"] = (s["game_start"] - s["fetched_at"]) / pd.Timedelta(hours=1)
    return _closing(s)


def _closing(s: pd.DataFrame) -> pd.DataFrame:
    """Mark each player's closing snapshot and compare every earlier price with it."""
    s = s.copy()
    s["is_close"] = False
    s["close_line"] = np.nan
    s["p_book_close"] = np.nan
    s["clv_over"] = np.nan
    before = s[s["fetched_at"] <= s["game_start"]]
    for _, grp in before.groupby(["game_pk", "player_id", "kind"]):
        t_close = grp["fetched_at"].max()
        close = grp[grp["fetched_at"] == t_close]
        s.loc[close.index, "is_close"] = True
        start = grp["game_start"].iloc[0]
        if (start - t_close) > pd.Timedelta(minutes=CLOSE_MAX_MINUTES):
            continue  # too early to count as a closing price
        close_lines = close.set_index("line")["p_book"]
        for i, row in grp[grp["fetched_at"] < t_close].iterrows():
            nearest = min(close_lines.index, key=lambda ln: (abs(ln - row["line"]), ln))
            s.at[i, "close_line"] = nearest
            if nearest == row["line"] and pd.notna(close_lines[nearest]) and pd.notna(row["p_book"]):
                s.at[i, "p_book_close"] = close_lines[nearest]
                s.at[i, "clv_over"] = close_lines[nearest] - row["p_book"]
    return s


def update_grades(base=None) -> pd.DataFrame:
    """Rebuild ``prop_grades`` from the full log and the stored box scores."""
    snaps = storage.read("prop_snapshots", base)
    grades = grade(snaps, storage.read("games", base), storage.read("batter_games", base),
                   storage.read("pitcher_games", base))
    if not grades.empty:
        path = storage.table_path("prop_grades", base)
        storage.write(grades, path)  # derived: replaced in full, never merged
    return grades


# --------------------------------------------------------------------------- #
# Picks and the Record tab's "vs DraftKings" numbers
# --------------------------------------------------------------------------- #

EDGE_STEPS = [0.01, 0.02, 0.03, 0.05]  # same as the app's minimum-edge control
EDGE_BUCKETS = [(0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, None)]
BOOTSTRAP = 2000


def _p_bet(r) -> float:
    """The chance a logged price's edge was worked out from (rows before the blend
    carry only the model's)."""
    pb = getattr(r, "p_blend", None)
    return r.p_model if pb is None or pd.isna(pb) else pb


def picks(grades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The app's value pick per player, game and kind, as first seen above ``threshold``.

    Mirrors ``bestBet`` in web/app.js: at each download, the line and side with the
    highest expected return; the pick is the first download where that clears the
    threshold. Staked at a flat $1.
    """
    if grades.empty:
        return pd.DataFrame()
    rows = []
    g = grades.sort_values("fetched_at")
    for _, grp in g.groupby(["game_pk", "player_id", "kind"], sort=False):
        for _, snap in grp.groupby("fetched_at", sort=True):
            best = None
            for r in snap.itertuples(index=False):
                for side in ("over", "under"):
                    ev, price = getattr(r, f"ev_{side}"), getattr(r, side)
                    if pd.isna(ev) or pd.isna(price):
                        continue
                    if best is None or ev > best[0]:
                        best = (ev, side, price, r)
            if best is None or best[0] < threshold:
                continue
            ev, side, price, r = best
            won = None
            if r.status == "graded" and not pd.isna(r.over_won):
                won = bool(r.over_won) if side == "over" else not bool(r.over_won)
            clv = None if pd.isna(r.clv_over) else (r.clv_over if side == "over" else -r.clv_over)
            rows.append({
                "game_pk": r.game_pk, "player_id": r.player_id, "kind": r.kind,
                "player_name": getattr(r, "player_name", None),
                "game_start": getattr(r, "game_start", None), "actual": getattr(r, "actual", None),
                "game_date": pd.Timestamp(r.game_date), "fetched_at": r.fetched_at,
                "line": r.line, "side": side, "price": int(price), "ev": float(ev),
                "p": float(_p_bet(r) if side == "over" else 1 - _p_bet(r)),
                "status": r.status, "won": won, "clv": clv,
                "close_line": None if pd.isna(r.close_line) else float(r.close_line),
                "lineup_confirmed": bool(r.lineup_confirmed) if not pd.isna(r.lineup_confirmed) else True,
            })
            break  # one pick per player, game and kind
    return pd.DataFrame(rows)


def _move(p: pd.Series) -> str | None:
    """Did the closing market move toward the pick, away from it, or stay?"""
    if p["close_line"] is not None and not pd.isna(p["close_line"]) and p["close_line"] != p["line"]:
        up = p["close_line"] > p["line"]  # a higher line means the market expects more
        return "toward" if up == (p["side"] == "over") else "away"
    if p["clv"] is None or pd.isna(p["clv"]):
        return None
    return "toward" if p["clv"] > 1e-9 else "away" if p["clv"] < -1e-9 else "stayed"


def pick_metrics(p: pd.DataFrame, seed: int = 0) -> dict:
    """Win rate, expected and break-even rates, return per $1 (90% bootstrap), CLV."""
    if p.empty:
        return {"n": 0, "n_void": 0}
    graded = p[p["status"] == "graded"]
    out = {"n": int(len(graded)), "n_void": int((p["status"] == "void").sum()),
           "n_pending": int((p["status"] == "pending").sum())}
    if graded.empty:
        return out
    dec = graded["price"].map(odds.american_to_decimal).astype(float)
    won = graded["won"].astype(bool).to_numpy()
    profit = np.where(won, dec - 1, -1.0)
    boot = np.random.default_rng(seed).choice(profit, size=(BOOTSTRAP, len(profit))).mean(axis=1)
    out.update({
        "win": float(won.mean()),
        "expected": float(graded["p"].mean()),
        "breakeven": float((1 / dec).mean()),
        "roi": float(profit.sum() / len(profit)),
        "roi_lo": float(np.percentile(boot, 5)),
        "roi_hi": float(np.percentile(boot, 95)),
    })
    clv = graded["clv"].dropna().astype(float)
    if len(clv):
        out.update({"clv": float(clv.mean() * 100), "clv_pos": float((clv > 0).mean()), "n_clv": int(len(clv))})
    moves = graded.apply(_move, axis=1).dropna()
    if len(moves):
        out["moves"] = {k: float((moves == k).mean()) for k in ("toward", "away", "stayed")}
    return out


def _logloss(p: pd.Series, y: pd.Series) -> float:
    p = np.clip(p.astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    y = y.astype(bool).to_numpy()
    return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))


def summary(grades: pd.DataFrame, *, days: int | None = None) -> dict:
    """``record.market`` in data.json: picks vs DraftKings per kind and edge threshold.

    ``days`` limits everything to the most recent days of games; each threshold also
    carries a ``last30`` cut.
    """
    if grades is None or grades.empty:
        return {}
    g = grades.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    if days is not None:
        g = g[g["game_date"] > g["game_date"].max() - pd.Timedelta(days=days)]
    out: dict = {"first_snapshot": g["game_date"].min().date().isoformat()}
    recent_from = g["game_date"].max() - pd.Timedelta(days=30)
    for kind in ("pitcher", "batter"):
        k = g[g["kind"] == kind]
        if k.empty:
            out[kind] = {}
            continue
        res: dict = {}
        # Model vs DraftKings on the same rows: the closing snapshot, one row per line.
        close = k[k["is_close"].astype(bool) & (k["status"] == "graded") & k["p_book"].notna()]
        if len(close):
            res.update({"logloss_model": _logloss(close["p_model"], close["over_won"]),
                        "logloss_book": _logloss(close["p_book"], close["over_won"]),
                        "n_lines": int(len(close))})
            if "p_shadow" in close and close["p_shadow"].notna().any():
                sh = close[close["p_shadow"].notna()]
                res["shadow"] = {
                    "name": str(sh["shadow_name"].dropna().iloc[-1]),
                    "n_lines": int(len(sh)),
                    "logloss_shadow": _logloss(sh["p_shadow"], sh["over_won"]),
                    "logloss_model": _logloss(sh["p_model"], sh["over_won"]),
                    "logloss_book": _logloss(sh["p_book"], sh["over_won"]),
                }
        res["by_threshold"] = {}
        for t in EDGE_STEPS:
            p = picks(k, t)
            m = pick_metrics(p)
            m["last30"] = pick_metrics(p[p["game_date"] > recent_from]) if not p.empty else {"n": 0}
            res["by_threshold"][f"{t:g}"] = m
        base = picks(k, EDGE_STEPS[0])
        res["by_edge"] = []
        for lo, hi in EDGE_BUCKETS:
            sel = base[(base["ev"] >= lo) & ((base["ev"] < hi) if hi else True)] if not base.empty else base
            res["by_edge"].append({"lo": lo, "hi": hi, **pick_metrics(sel)})
        if kind == "batter" and not base.empty:
            res["by_lineup"] = {
                "confirmed": pick_metrics(base[base["lineup_confirmed"]]),
                "projected": pick_metrics(base[~base["lineup_confirmed"]]),
            }
        out[kind] = res
    return out


# --------------------------------------------------------------------------- #
# Paper portfolio
# --------------------------------------------------------------------------- #

PAPER_STAKE = 10.0
PAPER_EDGE = 0.12
# Strategies paper traded side by side. ``source`` says which edges a strategy trades:
# "blend" (DraftKings tilted by the model, from Oct 6, 2026) or "model" (the model's own
# chance, used before then; those strategies keep their record but take no new trades).
# A band strategy takes the first price at or above its floor and keeps it only if that
# edge is below the ceiling.
PAPER_STRATEGIES = [
    {"key": "blend1", "label": "Blended 1%+", "threshold": 0.01, "ceiling": None, "source": "blend"},
    {"key": "edge12", "label": "Model 12%+ (retired)", "threshold": 0.12, "ceiling": None, "source": "model"},
    {"key": "edge8_12", "label": "Model 8–12% (retired)", "threshold": 0.08, "ceiling": 0.12,
     "source": "model"},
]


def paper_trades(grades: pd.DataFrame, *, stake: float = PAPER_STAKE,
                 threshold: float = PAPER_EDGE, ceiling: float | None = None,
                 source: str | None = None) -> pd.DataFrame:
    """Paper bets: $``stake`` on every value pick at or above ``threshold`` edge
    (and, with ``ceiling``, below it at the moment it first cleared ``threshold``).

    A trade is placed at the first logged DraftKings price whose best side clears the
    threshold (one per player, game and prop), exactly as the Value picks list shows it:
    hitters from projected lineups are skipped. The log is append-only and stores the
    model's numbers at download time, so recorded trades never change afterwards.
    """
    if grades is None or grades.empty:
        return pd.DataFrame()
    g = grades
    if source is not None:
        blended = g["p_blend"].notna() if "p_blend" in g else pd.Series(False, index=g.index)
        g = g[blended if source == "blend" else ~blended]
    if "lineup_confirmed" in g:
        g = g[(g["kind"] == "pitcher") | g["lineup_confirmed"].fillna(True).astype(bool)]
    t = picks(g, threshold)
    if ceiling is not None and not t.empty:
        t = t[t["ev"] < ceiling].copy()
    if t.empty:
        return t
    dec = t["price"].map(odds.american_to_decimal).astype(float)
    t["stake"] = stake
    t["profit"] = np.select(
        [t["status"].eq("graded") & t["won"].eq(True), t["status"].eq("graded")],
        [stake * (dec - 1), -stake], 0.0)
    t.loc[t["status"].eq("pending"), "profit"] = np.nan
    t["result"] = np.select(
        [t["status"].eq("pending"), t["status"].eq("void"), t["won"].eq(True)],
        ["open", "void", "won"], "lost")
    return t.sort_values("fetched_at").reset_index(drop=True)


def paper_portfolio(grades: pd.DataFrame, *, stake: float = PAPER_STAKE,
                    threshold: float = PAPER_EDGE, ceiling: float | None = None,
                    names: dict | None = None, key: str | None = None,
                    label: str | None = None, source: str | None = None) -> dict:
    """One paper strategy in data.json: its trades, their totals and daily profit."""
    t = paper_trades(grades, stake=stake, threshold=threshold, ceiling=ceiling, source=source)
    base = {"stake": stake, "threshold": threshold, "ceiling": ceiling, "key": key, "label": label,
            "source": source}
    if names and not t.empty:  # probable pitchers are logged without a name
        t["player_name"] = t["player_name"].astype(object)  # all-missing names load as float
        missing = t["player_name"].isna() | (t["player_name"].astype(str).str.strip() == "")
        t.loc[missing, "player_name"] = t.loc[missing, "player_id"].map(names)
    if t.empty:
        return {**base, "trades": [], "summary": {"n": 0}}
    settled = t[t["result"].isin(["won", "lost"])]
    staked = float(stake * len(settled))
    profit = float(settled["profit"].sum())
    daily = (settled.assign(day=pd.to_datetime(settled["game_date"]).dt.date)
             .groupby("day")["profit"].sum().sort_index())
    summary = {
        "n": int(len(t)), "open": int((t["result"] == "open").sum()),
        "won": int((t["result"] == "won").sum()), "lost": int((t["result"] == "lost").sum()),
        "void": int((t["result"] == "void").sum()), "staked": staked, "profit": profit,
        "roi": profit / staked if staked else None,
        "at_risk": float(stake * (t["result"] == "open").sum()),
        "first_trade": pd.Timestamp(t["fetched_at"].min()).isoformat(),
        "avg_edge": float(t["ev"].mean()),
        "clv": float(settled["clv"].dropna().astype(float).mean() * 100) if settled["clv"].notna().any() else None,
    }
    curve = [{"date": d.isoformat(), "profit": float(v), "cum": float(c)}
             for (d, v), c in zip(daily.items(), daily.cumsum())]
    cols = ["fetched_at", "game_date", "game_start", "game_pk", "player_id", "player_name", "kind",
            "line", "side", "price", "ev", "p", "result", "actual", "profit", "clv", "stake"]
    trades = t[[c for c in cols if c in t]].iloc[::-1]  # newest first
    trades = trades.assign(fetched_at=trades["fetched_at"].astype(str),
                           game_date=pd.to_datetime(trades["game_date"]).dt.date.astype(str),
                           game_start=trades["game_start"].astype(str) if "game_start" in trades else None)
    return {**base, "summary": summary, "curve": curve, "trades": trades.to_dict("records")}


def paper_strategies(grades: pd.DataFrame, names: dict | None = None) -> list[dict]:
    """``paper_strategies`` in data.json: every strategy in ``PAPER_STRATEGIES``."""
    return [paper_portfolio(grades, threshold=st["threshold"], ceiling=st["ceiling"], names=names,
                            key=st["key"], label=st["label"], source=st["source"])
            for st in PAPER_STRATEGIES]
