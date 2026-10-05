"""Pure functions turning MLB Stats API JSON into flat rows.

Kept free of I/O so they can be unit tested against saved fixtures.
"""

from __future__ import annotations

import re

BATTING_FIELDS = {
    "plateAppearances": "pa",
    "atBats": "ab",
    "hits": "h",
    "runs": "r",
    "rbi": "rbi",
    "doubles": "doubles",
    "triples": "triples",
    "homeRuns": "hr",
    "totalBases": "tb",
    "baseOnBalls": "bb",
    "intentionalWalks": "ibb",
    "hitByPitch": "hbp",
    "strikeOuts": "so",
    "stolenBases": "sb",
    "sacFlies": "sf",
}

PITCHING_FIELDS = {
    "outs": "outs",
    "battersFaced": "bf",
    "numberOfPitches": "pitches",
    "strikes": "strikes",
    "strikeOuts": "k",
    "baseOnBalls": "bb",
    "hitByPitch": "hbp",
    "hits": "h",
    "homeRuns": "hr",
    "runs": "r",
    "earnedRuns": "er",
}

_WIND_RE = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.I)


def parse_wind(wind: str | None) -> tuple[float | None, str | None]:
    """'12 mph, Out To CF' -> (12.0, 'Out To CF')."""
    if not wind:
        return None, None
    m = _WIND_RE.match(wind.strip())
    if not m:
        return None, wind.strip() or None
    return float(m.group(1)), (m.group(2).strip() or None)


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _hp_umpire(officials: list[dict] | None) -> tuple[int | None, str | None]:
    for o in officials or []:
        if o.get("officialType") == "Home Plate":
            off = o.get("official", {})
            return off.get("id"), off.get("fullName")
    return None, None


def parse_schedule(data: dict) -> tuple[list[dict], list[dict]]:
    """Return (games, lineups) rows from a hydrated /schedule response."""
    games: list[dict] = []
    lineups: list[dict] = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            weather = g.get("weather") or {}
            wind_speed, wind_dir = parse_wind(weather.get("wind"))
            ump_id, ump_name = _hp_umpire(g.get("officials"))
            status = g.get("status", {})
            row = {
                "game_pk": g["gamePk"],
                "game_date": g.get("officialDate") or day.get("date"),
                "game_datetime": g.get("gameDate"),
                "season": int(g.get("season") or (g.get("officialDate") or day["date"])[:4]),
                "game_type": g.get("gameType"),
                "status": status.get("abstractGameState"),
                "detailed_state": status.get("detailedState"),
                "double_header": g.get("doubleHeader"),
                "game_number": g.get("gameNumber"),
                "day_night": g.get("dayNight"),
                "venue_id": g.get("venue", {}).get("id"),
                "venue_name": g.get("venue", {}).get("name"),
                "home_team_id": home.get("team", {}).get("id"),
                "home_team": home.get("team", {}).get("name"),
                "away_team_id": away.get("team", {}).get("id"),
                "away_team": away.get("team", {}).get("name"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_probable_id": (home.get("probablePitcher") or {}).get("id"),
                "away_probable_id": (away.get("probablePitcher") or {}).get("id"),
                "temp_f": _to_float(weather.get("temp")),
                "weather_condition": weather.get("condition"),
                "wind_mph": wind_speed,
                "wind_dir": wind_dir,
                "hp_umpire_id": ump_id,
                "hp_umpire": ump_name,
            }
            games.append(row)

            lu = g.get("lineups") or {}
            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                team_id = row[f"{side}_team_id"]
                for slot, p in enumerate(lu.get(key) or [], start=1):
                    lineups.append({
                        "game_pk": row["game_pk"],
                        "team_id": team_id,
                        "is_home": side == "home",
                        "player_id": p["id"],
                        "player_name": p.get("fullName"),
                        "batting_order": slot,
                        "position": (p.get("primaryPosition") or {}).get("abbreviation"),
                    })
    return games, lineups


def parse_boxscore(game_pk: int, data: dict) -> tuple[list[dict], list[dict], dict]:
    """Return (batter_rows, pitcher_rows, game_extra) for one finished game.

    ``game_extra`` holds game-level info only the boxscore has (HP umpire).
    """
    teams = data["teams"]
    starters = {side: (teams[side].get("pitchers") or [None])[0] for side in ("home", "away")}
    team_ids = {side: teams[side]["team"]["id"] for side in ("home", "away")}
    batters: list[dict] = []
    pitchers: list[dict] = []

    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        t = teams[side]
        players = t.get("players", {})

        for pid in t.get("batters", []):
            p = players.get(f"ID{pid}")
            if not p:
                continue
            bat = p.get("stats", {}).get("batting") or {}
            if not bat:
                continue
            order = p.get("battingOrder")
            order_i = int(order) if order and str(order).isdigit() else None
            row = {
                "game_pk": game_pk,
                "player_id": pid,
                "player_name": p.get("person", {}).get("fullName"),
                "team_id": team_ids[side],
                "opp_team_id": team_ids[opp],
                "is_home": side == "home",
                "position": (p.get("position") or {}).get("abbreviation"),
                "batting_order": order_i // 100 if order_i else None,
                "is_starter": bool(order_i) and order_i % 100 == 0,
                "opp_starter_id": starters[opp],
            }
            for src, dst in BATTING_FIELDS.items():
                row[dst] = bat.get(src, 0) or 0
            row["hrr"] = row["h"] + row["r"] + row["rbi"]
            batters.append(row)

        for i, pid in enumerate(t.get("pitchers", [])):
            p = players.get(f"ID{pid}")
            if not p:
                continue
            pit = p.get("stats", {}).get("pitching") or {}
            if not pit:
                continue
            row = {
                "game_pk": game_pk,
                "player_id": pid,
                "player_name": p.get("person", {}).get("fullName"),
                "team_id": team_ids[side],
                "opp_team_id": team_ids[opp],
                "is_home": side == "home",
                "is_starter": i == 0,
                "pitcher_order": i + 1,
            }
            for src, dst in PITCHING_FIELDS.items():
                row[dst] = pit.get(src, 0) or 0
            if not row["pitches"]:
                row["pitches"] = pit.get("pitchesThrown", 0) or 0
            pitchers.append(row)

    ump_id, ump_name = _hp_umpire(data.get("officials"))
    return batters, pitchers, {"game_pk": game_pk, "hp_umpire_id": ump_id, "hp_umpire": ump_name}


def parse_people(people: list[dict]) -> list[dict]:
    return [
        {
            "player_id": p["id"],
            "full_name": p.get("fullName"),
            "bat_side": (p.get("batSide") or {}).get("code"),
            "pitch_hand": (p.get("pitchHand") or {}).get("code"),
            "birth_date": p.get("birthDate"),
            "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
        }
        for p in people
    ]
