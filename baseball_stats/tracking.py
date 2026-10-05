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
