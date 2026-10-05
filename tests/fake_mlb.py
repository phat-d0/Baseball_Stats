"""Synthetic MLB Stats API responses shaped like the real endpoints."""

from __future__ import annotations

import random
from datetime import date, timedelta

TEAMS = [(101, "Aces"), (102, "Bears"), (103, "Cats"), (104, "Dogs")]
VENUES = {101: (1, "Ace Park"), 102: (2, "Bear Field"), 103: (3, "Cat Dome"), 104: (4, "Dog Yard")}


def roster(team_id: int) -> tuple[list[int], list[int]]:
    hitters = [team_id * 100 + i for i in range(1, 11)]  # 10 hitters, 1 bench
    pitchers = [team_id * 100 + 50 + i for i in range(1, 9)]  # 5 SP + 3 RP
    return hitters, pitchers


class FakeMLB:
    def __init__(self, start: date, days: int, seed: int = 0, doubleheader_on: date | None = None):
        self.rng = random.Random(seed)
        self.games: dict[int, dict] = {}
        self.box: dict[int, dict] = {}
        pk = 1000
        for d in range(days):
            day = start + timedelta(days=d)
            pairs = [(TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])] if d % 2 else \
                    [(TEAMS[0], TEAMS[2]), (TEAMS[1], TEAMS[3])]
            if doubleheader_on == day:
                pairs = pairs + [pairs[0]]
            for n, (home, away) in enumerate(pairs):
                pk += 1
                self._make_game(pk, day, home, away, d, n)

    def _starter(self, team_id: int, d: int) -> int:
        return roster(team_id)[1][d % 5]

    def _make_game(self, pk, day, home, away, d, n):
        hsp, asp = self._starter(home[0], d), self._starter(away[0], d)
        self.games[pk] = {
            "gamePk": pk, "gameType": "R", "season": str(day.year),
            "gameDate": f"{day}T{17 + n * 4:02d}:05:00Z", "officialDate": day.isoformat(),
            "status": {"abstractGameState": "Final", "detailedState": "Final"},
            "doubleHeader": "N", "gameNumber": 1, "dayNight": "night",
            "venue": {"id": VENUES[home[0]][0], "name": VENUES[home[0]][1]},
            "teams": {
                "home": {"team": {"id": home[0], "name": home[1]}, "score": 0,
                         "probablePitcher": {"id": hsp}},
                "away": {"team": {"id": away[0], "name": away[1]}, "score": 0,
                         "probablePitcher": {"id": asp}},
            },
            "weather": {"condition": "Clear", "temp": "75", "wind": "8 mph, Out To CF"},
            "officials": [{"official": {"id": 900 + pk % 3, "fullName": "Ump"},
                           "officialType": "Home Plate"}],
        }
        teams = {}
        for side, (tid, name), sp in (("home", home, hsp), ("away", away, asp)):
            hitters, staff = roster(tid)
            players, batters, order = {}, [], []
            for i, pid in enumerate(hitters[:9]):
                pa = self.rng.randint(3, 5)
                h = self.rng.randint(0, min(3, pa))
                bat = {"plateAppearances": pa, "atBats": pa - 1, "hits": h,
                       "runs": self.rng.randint(0, h + 1), "rbi": self.rng.randint(0, h + 1),
                       "doubles": 0, "triples": 0, "homeRuns": int(h == 3), "totalBases": h,
                       "baseOnBalls": 1, "strikeOuts": self.rng.randint(0, 2)}
                players[f"ID{pid}"] = {"person": {"id": pid, "fullName": f"Hitter {pid}"},
                                       "position": {"abbreviation": "RF"},
                                       "battingOrder": str((i + 1) * 100),
                                       "stats": {"batting": bat, "pitching": {}}}
                batters.append(pid)
                order.append(pid)
            relievers = staff[5:5 + self.rng.randint(1, 3)]
            for j, pid in enumerate([sp, *relievers]):
                bf = self.rng.randint(18, 28) if j == 0 else self.rng.randint(3, 8)
                pit = {"outs": bf - 6 if j == 0 else bf - 2, "battersFaced": bf,
                       "numberOfPitches": bf * 4, "strikes": bf * 2 + 5,
                       "strikeOuts": self.rng.randint(2, 10) if j == 0 else self.rng.randint(0, 3),
                       "baseOnBalls": 1, "hits": 4 if j == 0 else 1, "homeRuns": 0,
                       "runs": 2, "earnedRuns": 2}
                players[f"ID{pid}"] = {"person": {"id": pid, "fullName": f"Pitcher {pid}"},
                                       "position": {"abbreviation": "P"},
                                       "stats": {"batting": {}, "pitching": pit}}
            teams[side] = {"team": {"id": tid, "name": name}, "players": players,
                           "batters": batters, "battingOrder": order,
                           "pitchers": [sp, *relievers]}
        self.box[pk] = {"teams": teams, "officials": self.games[pk]["officials"]}

    # --- endpoint stand-ins -------------------------------------------------
    def schedule(self, start, end, *, game_type="R", cache=True):
        by_day: dict[str, list] = {}
        for g in self.games.values():
            if start.isoformat() <= g["officialDate"] <= end.isoformat():
                by_day.setdefault(g["officialDate"], []).append(g)
        return {"dates": [{"date": d, "games": gs} for d, gs in sorted(by_day.items())]}

    def boxscore(self, pk, *, cache=True):
        return self.box[pk]

    def people(self, ids, chunk=100):
        return [{"id": i, "fullName": f"P{i}",
                 "batSide": {"code": "L" if i % 3 == 0 else "R"},
                 "pitchHand": {"code": "L" if i % 2 else "R"},
                 "birthDate": "1995-05-05",
                 "primaryPosition": {"abbreviation": "P" if i % 100 > 50 else "RF"}} for i in ids]

    def upcoming(self, day: date, pk: int = 9999) -> dict:
        """Schedule response for one unplayed game with posted lineups."""
        home, away = TEAMS[0], TEAMS[3]
        lineup = lambda t: [{"id": p, "fullName": f"Hitter {p}",
                             "primaryPosition": {"abbreviation": "RF"}} for p in roster(t)[0][:9]]
        return {"dates": [{"date": day.isoformat(), "games": [{
            "gamePk": pk, "gameType": "R", "season": str(day.year),
            "gameDate": f"{day}T23:05:00Z", "officialDate": day.isoformat(),
            "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
            "venue": {"id": 1, "name": "Ace Park"},
            "teams": {"home": {"team": {"id": home[0], "name": home[1]},
                               "probablePitcher": {"id": roster(home[0])[1][0]}},
                      "away": {"team": {"id": away[0], "name": away[1]},
                               "probablePitcher": {"id": roster(away[0])[1][1]}}},
            "lineups": {"homePlayers": lineup(home[0]), "awayPlayers": lineup(away[0])},
        }]}]}
