"""Thin wrappers around the public MLB Stats API (statsapi.mlb.com)."""

from __future__ import annotations

from datetime import date

from . import config
from .http import get_json

SCHEDULE_HYDRATE = "team,probablePitcher,venue,weather,lineups,officials"
# Regular season plus every postseason round (wild card, division, LCS, World Series).
ALL_GAME_TYPES = "R,F,D,L,W"


def schedule(start: date, end: date, *, game_type: str = ALL_GAME_TYPES, cache: bool = True) -> dict:
    """Schedule for a date range, hydrated with probables, lineups and weather.

    ``game_type``: comma-separated codes: R regular season, F/D/L/W postseason rounds.
    Pass ``cache=False`` for today/future dates, where probables and lineups change.
    """
    return get_json(
        f"{config.MLB_API_BASE}/v1/schedule",
        {
            "sportId": 1,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "gameType": game_type,
            "hydrate": SCHEDULE_HYDRATE,
        },
        cache=cache,
    )


def boxscore(game_pk: int, *, cache: bool = True) -> dict:
    return get_json(f"{config.MLB_API_BASE}/v1/game/{game_pk}/boxscore", cache=cache)


def people(person_ids: list[int], *, chunk: int = 100) -> list[dict]:
    """Biographical info (bat side, pitch hand, birth date) for players."""
    out: list[dict] = []
    ids = sorted(set(int(i) for i in person_ids))
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        data = get_json(
            f"{config.MLB_API_BASE}/v1/people",
            {"personIds": ",".join(map(str, part))},
        )
        out.extend(data.get("people", []))
    return out
