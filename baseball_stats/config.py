"""Filesystem layout for raw caches, processed tables and feature sets."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("BASEBALL_DATA_DIR", "data"))

RAW_DIR = DATA_DIR / "raw"  # cached API responses (JSON / CSV)
PROCESSED_DIR = DATA_DIR / "processed"  # one row per player-game, games, players
FEATURES_DIR = DATA_DIR / "features"  # model-ready tables

MLB_API_BASE = "https://statsapi.mlb.com/api"
SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Seconds to wait between uncached requests, to be polite to public endpoints.
REQUEST_DELAY = float(os.environ.get("BASEBALL_REQUEST_DELAY", "0.25"))
