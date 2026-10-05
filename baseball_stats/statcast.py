"""Pitch-level Statcast data from Baseball Savant, aggregated per player-game.

Savant caps a CSV search at ~25k rows, so we request one day at a time
(a full slate is ~4-5k pitches). Each day's CSV is cached on disk.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd

from . import config
from .http import get

SWINGS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
CALLED_STRIKES = {"called_strike"}

COLUMNS = [
    "game_pk", "game_date", "batter", "pitcher", "stand", "p_throws",
    "description", "events", "zone", "pitch_type", "release_speed",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
]


def fetch_day(day: date, *, game_type: str = "R") -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "hfGT": f"{game_type}|",
        "game_date_gt": day.isoformat(),
        "game_date_lt": day.isoformat(),
    }
    body = get(config.SAVANT_CSV_URL, params, suffix=".csv",
               cache=day < date.today(), timeout=180)
    if not body.strip():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(io.BytesIO(body), low_memory=False, encoding="utf-8-sig")
    if "game_pk" not in df.columns:  # Savant returns an HTML/empty page on no games
        return pd.DataFrame(columns=COLUMNS)
    return df[[c for c in COLUMNS if c in df.columns]]


def fetch_range(start: date, end: date, **kw) -> pd.DataFrame:
    frames = []
    d = start
    while d <= end:
        frames.append(fetch_day(d, **kw))
        d += timedelta(days=1)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def _flags(pitches: pd.DataFrame) -> pd.DataFrame:
    p = pitches.copy()
    desc = p["description"].fillna("")
    p["pitch"] = 1
    p["swing"] = desc.isin(SWINGS).astype(int)
    p["whiff"] = desc.isin(WHIFFS).astype(int)
    p["called_strike"] = desc.isin(CALLED_STRIKES).astype(int)
    p["in_zone"] = p["zone"].between(1, 9).astype(int)
    p["chase"] = ((p["in_zone"] == 0) & (p["swing"] == 1)).astype(int)
    p["out_zone"] = (p["in_zone"] == 0).astype(int)
    bip = p["launch_speed"].notna() & (desc == "hit_into_play")
    p["bip"] = bip.astype(int)
    p["hard_hit"] = (bip & (p["launch_speed"] >= 95)).astype(int)
    p["barrel"] = (bip & (p["launch_speed_angle"] == 6)).astype(int)
    # xwOBA: use the batted-ball estimate on balls in play, actual wOBA otherwise
    # (walks, strikeouts, HBP). Only rows with woba_denom count as PA.
    denom = p["woba_denom"].fillna(0)
    xw = p["estimated_woba_using_speedangle"].where(bip, p["woba_value"])
    p["xwoba_num"] = xw.fillna(0) * (denom > 0)
    p["xwoba_den"] = denom
    p["fb_velo"] = p["release_speed"].where(p["pitch_type"].isin(["FF", "SI"]))
    return p


_SUM_COLS = ["pitch", "swing", "whiff", "called_strike", "in_zone", "out_zone",
             "chase", "bip", "hard_hit", "barrel", "xwoba_num", "xwoba_den"]


def aggregate(pitches: pd.DataFrame, who: str) -> pd.DataFrame:
    """Sum per (game_pk, ``who``) where ``who`` is 'batter' or 'pitcher'."""
    if pitches.empty:
        return pd.DataFrame(columns=["game_pk", who, *_SUM_COLS])
    p = _flags(pitches)
    agg = p.groupby(["game_pk", who], as_index=False)[_SUM_COLS].sum()
    if who == "pitcher":
        velo = p.groupby(["game_pk", who], as_index=False)["fb_velo"].mean()
        agg = agg.merge(velo, on=["game_pk", who], how="left")
    return agg
