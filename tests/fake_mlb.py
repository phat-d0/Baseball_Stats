"""Synthetic MLB Stats API responses shaped like the real endpoints."""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

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
        self.pitch_rows: dict[int, list] = {}
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
        teams, pitch_rows, score = self._play(pk, day, home, away, hsp, asp)
        self.games[pk]["teams"]["home"]["score"] = score["home"]
        self.games[pk]["teams"]["away"]["score"] = score["away"]
        self.box[pk] = {"teams": teams, "officials": self.games[pk]["officials"]}
        self.pitch_rows[pk] = pitch_rows

    # --- plate-appearance simulation: box scores and Savant pitch rows agree ---
    BASE_P = {"K": 0.22, "BB": 0.08, "1B": 0.15, "2B": 0.045, "3B": 0.005, "HR": 0.03}
    EVENT = {"K": "strikeout", "BB": "walk", "1B": "single", "2B": "double",
             "3B": "triple", "HR": "home_run", "OUT": "field_out"}

    def _outcome(self, batter, pitcher):
        # Batters and pitchers differ by id so the models have something to learn.
        bk = 0.8 + (batter % 7) * 0.07
        pk_ = 0.8 + (pitcher % 5) * 0.1
        probs = dict(self.BASE_P)
        probs["K"] *= bk * pk_
        probs["1B"] *= 1.6 - 0.6 * bk
        r, acc = self.rng.random(), 0.0
        for o, pr in probs.items():
            acc += pr
            if r < acc:
                return o
        return "OUT"

    def _play(self, pk, day, home, away, hsp, asp):
        sides = {"home": home, "away": away}
        lineups = {s: roster(t[0])[0][:9] for s, t in sides.items()}
        staffs = {"home": [hsp, *roster(home[0])[1][5:8]], "away": [asp, *roster(away[0])[1][5:8]]}
        # The fielding side's current pitcher and how long the starter goes.
        cur = {"home": 0, "away": 0}
        leash = {s: self.rng.randint(18, 27) for s in sides}
        pbf = {}  # batters faced per pitcher
        bat = {s: {pid: dict(pa=0, ab=0, h=0, r=0, rbi=0, d=0, t=0, hr=0, bb=0, so=0, sf=0)
                   for pid in lineups[s]} for s in sides}
        pit = {}
        nxt = {"home": 0, "away": 0}
        score = {"home": 0, "away": 0}
        rows, ab_no = [], 0
        inning = 0
        while True:
            inning += 1
            for half, batting, fielding in (("Top", "away", "home"), ("Bot", "home", "away")):
                if half == "Bot" and inning >= 9 and score["home"] > score["away"]:
                    break
                outs, bases = 0, [None, None, None]
                if inning > 9:
                    bases[1] = lineups[batting][(nxt[batting] - 1) % 9]  # automatic runner
                while outs < 3:
                    staff = staffs[fielding]
                    pitcher = staff[min(cur[fielding], len(staff) - 1)]
                    if cur[fielding] == 0 and pbf.get(pitcher, 0) >= leash[fielding]:
                        cur[fielding] = 1
                    elif cur[fielding] > 0 and pbf.get(pitcher, 0) >= 6 and cur[fielding] < len(staff) - 1:
                        cur[fielding] += 1
                    pitcher = staff[min(cur[fielding], len(staff) - 1)]
                    batter = lineups[batting][nxt[batting] % 9]
                    nxt[batting] += 1
                    o = self._outcome(batter, pitcher)
                    ps = pit.setdefault(pitcher, dict(bf=0, outs=0, k=0, bb=0, h=0, hr=0, r=0, pitches=0))
                    b = bat[batting][batter]
                    bat_score = score[batting]
                    before = dict(outs=outs, on=list(bases))
                    runs, scorers, event = 0, [], self.EVENT[o]
                    if o == "OUT" and bases[2] is not None and outs < 2 and self.rng.random() < 0.3:
                        event, scorers, bases[2] = "sac_fly", [bases[2]], None
                        outs += 1
                        b["sf"] += 1
                    elif o in ("K", "OUT"):
                        outs += 1
                    elif o == "BB":
                        if bases[0] is not None:
                            if bases[1] is not None:
                                if bases[2] is not None:
                                    scorers.append(bases[2])
                                bases[2] = bases[1]
                            bases[1] = bases[0]
                        bases[0] = batter
                    else:
                        adv = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}[o]
                        new = [None, None, None]
                        for i in (2, 1, 0):
                            if bases[i] is not None:
                                if i + adv >= 3:
                                    scorers.append(bases[i])
                                else:
                                    new[i + adv] = bases[i]
                        if adv == 4:
                            scorers.append(batter)
                        else:
                            new[adv - 1] = batter
                        bases = new
                    runs = len(scorers)
                    score[batting] += runs
                    for runner in scorers:
                        bat[batting][runner]["r"] += 1
                    b["pa"] += 1
                    b["rbi"] += runs
                    b["h"] += o in ("1B", "2B", "3B", "HR")
                    b["d"] += o == "2B"; b["t"] += o == "3B"; b["hr"] += o == "HR"
                    b["bb"] += o == "BB"; b["so"] += o == "K"
                    n_p = self.rng.randint(1, 6)
                    ps["bf"] += 1; ps["pitches"] += n_p; ps["k"] += o == "K"; ps["bb"] += o == "BB"
                    ps["h"] += o in ("1B", "2B", "3B", "HR"); ps["hr"] += o == "HR"; ps["r"] += runs
                    ps["outs"] += o in ("K", "OUT")
                    tto = pbf.get(pitcher, 0) // 9 + 1
                    pbf[pitcher] = pbf.get(pitcher, 0) + 1
                    ab_no += 1
                    for n in range(1, n_p + 1):
                        last = n == n_p
                        desc = ("swinging_strike" if o == "K" else "ball" if o == "BB" else "hit_into_play") \
                            if last else self.rng.choice(["ball", "called_strike", "foul"])
                        ev = float(self.rng.uniform(80, 110)) if last and o not in ("K", "BB") else np.nan
                        rows.append({
                            "game_pk": pk, "game_date": day.isoformat(), "batter": batter, "pitcher": pitcher,
                            "stand": "L" if batter % 3 == 0 else "R", "p_throws": "L" if pitcher % 2 else "R",
                            "description": desc, "events": event if last else None,
                            "zone": self.rng.randint(1, 14), "pitch_type": "FF",
                            "release_speed": 92.0 + (pitcher % 5), "launch_speed": ev, "launch_angle": 15.0,
                            "launch_speed_angle": 6 if ev == ev and ev > 105 else 3,
                            "estimated_woba_using_speedangle": 0.4 if ev == ev else np.nan,
                            "woba_value": 0.0, "woba_denom": 1 if last else np.nan,
                            "at_bat_number": ab_no, "pitch_number": n, "inning": inning,
                            "inning_topbot": half, "outs_when_up": before["outs"],
                            "on_1b": before["on"][0], "on_2b": before["on"][1], "on_3b": before["on"][2],
                            "home_team": "HOM", "away_team": "AWY", "n_thruorder_pitcher": tto,
                            "bat_score": bat_score, "post_bat_score": score[batting] if last else bat_score,
                        })
            if inning >= 9 and score["home"] != score["away"] or inning >= 12:
                break

        teams = {}
        for side in ("home", "away"):
            tid, name = sides[side]
            players, order = {}, []
            for i, pid in enumerate(lineups[side]):
                b = bat[side][pid]
                tb = b["h"] + b["d"] + 2 * b["t"] + 3 * b["hr"]
                players[f"ID{pid}"] = {
                    "person": {"id": pid, "fullName": f"Hitter {pid}"},
                    "position": {"abbreviation": "RF"}, "battingOrder": str((i + 1) * 100),
                    "stats": {"pitching": {}, "batting": {
                        "plateAppearances": b["pa"], "atBats": b["pa"] - b["bb"] - b["sf"],
                        "hits": b["h"], "runs": b["r"], "rbi": b["rbi"], "doubles": b["d"],
                        "triples": b["t"], "homeRuns": b["hr"], "totalBases": tb,
                        "baseOnBalls": b["bb"], "strikeOuts": b["so"], "sacFlies": b["sf"]}}}
                order.append(pid)
            used = [p for p in staffs[side] if p in pit]
            for pid in used:
                ps = pit[pid]
                players[f"ID{pid}"] = {
                    "person": {"id": pid, "fullName": f"Pitcher {pid}"},
                    "position": {"abbreviation": "P"},
                    "stats": {"batting": {}, "pitching": {
                        "outs": ps["outs"], "battersFaced": ps["bf"], "numberOfPitches": ps["pitches"],
                        "strikes": int(ps["pitches"] * 0.62), "strikeOuts": ps["k"],
                        "baseOnBalls": ps["bb"], "hits": ps["h"], "homeRuns": ps["hr"],
                        "runs": ps["r"], "earnedRuns": ps["r"]}}}
            teams[side] = {"team": {"id": tid, "name": name}, "players": players,
                           "batters": order, "battingOrder": order, "pitchers": used}
        return teams, rows, score

    def pitches(self, day) -> pd.DataFrame:
        """Savant-style pitch rows for every game on ``day`` (stand-in for statcast.fetch_day)."""
        rows = [r for pk, g in self.games.items() if g["officialDate"] == day.isoformat()
                for r in self.pitch_rows.get(pk, [])]
        return pd.DataFrame(rows)

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
