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

from . import model, odds, storage

EASTERN = ZoneInfo("America/New_York")
SNAPSHOT_KEY = ["game_pk", "player_id", "kind", "line", "fetched_at"]


def line_values(entry: dict, pmf: np.ndarray) -> dict:
    """Model chance, DraftKings' no-vig chance and expected return per $1 on each side.

    The one place these are worked out, so the log stores exactly what the app shows.
    """
    p = float(model.p_over(pmf[None, :], entry["line"])[0])
    do = odds.american_to_decimal(entry.get("over"))
    du = odds.american_to_decimal(entry.get("under"))
    return {
        "p_model": p,
        "p_book": odds.no_vig_over(entry.get("over"), entry.get("under")),
        "ev_over": p * do - 1 if do else None,
        "ev_under": (1 - p) * du - 1 if du else None,
    }


def _ts(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def snapshot_rows(slate: dict, models: dict[str, model.CountModel], props: dict) -> pd.DataFrame:
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
        mu, dist = models[kind].distribution(sub)
        for (_, r), m, d in zip(sub.iterrows(), mu, dist):
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
                    "alpha": float(models[kind].alpha),
                    **line_values(e, d),
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
    for c in ("p_book", "ev_over", "ev_under"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def log_snapshots(slate: dict, models: dict[str, model.CountModel], props: dict,
                  now: datetime | None = None, base=None) -> pd.DataFrame:
    """Append new price snapshots to ``prop_snapshots``; returns the rows added.

    Only downloads not already stored are added: re-running a publish on the same
    cached prices adds nothing, and earlier rows are never changed. ``now`` is unused
    beyond the signature the spec names; ``fetched_at`` comes from the ledger.
    """
    new = snapshot_rows(slate, models, props)
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
