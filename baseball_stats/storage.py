"""Parquet tables with upsert-by-key semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

# Primary keys for each processed table.
KEYS = {
    "games": ["game_pk"],
    "lineups": ["game_pk", "player_id"],
    "batter_games": ["game_pk", "player_id"],
    "pitcher_games": ["game_pk", "player_id"],
    "players": ["player_id"],
    "statcast_batter": ["game_pk", "batter"],
    "statcast_pitcher": ["game_pk", "pitcher"],
}


def table_path(name: str, base: Path | None = None) -> Path:
    return (base or config.PROCESSED_DIR) / f"{name}.parquet"


def read(name: str, base: Path | None = None) -> pd.DataFrame:
    path = table_path(name, base)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def upsert(name: str, df: pd.DataFrame, base: Path | None = None) -> pd.DataFrame:
    """Merge ``df`` into the stored table; new rows win on key collisions."""
    if df.empty:
        return read(name, base)
    keys = KEYS[name]
    old = read(name, base)
    merged = pd.concat([old, df], ignore_index=True) if not old.empty else df.copy()
    merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    path = table_path(name, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
