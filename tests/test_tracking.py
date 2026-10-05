"""Pick logging and market grading (spec: "Pick Logging and Market Grading")."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from baseball_stats import odds, publish, storage, tracking
from test_publish import START, FakeDK, fake  # noqa: F401  (fixture)

RUN1 = datetime(2025, 6, 15, 16, 0, tzinfo=UTC)  # 7 h before the 23:05 UTC first pitch


@pytest.fixture
def dk(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("ODDS_API_RESET_DAY", "1")
    monkeypatch.delenv("ODDS_API_MONTHLY_CAP", raising=False)
    api = FakeDK()
    monkeypatch.setattr(odds.requests, "get", api.get)
    return api


def _publish(tmp_path, now):
    return publish.publish(tmp_path / "site", history_start=START, now=now)


def test_snapshots_match_data_json(fake, dk, tmp_path):
    data = _publish(tmp_path, RUN1)
    snaps = storage.read("prop_snapshots")
    # One row per player and line: P10151 (K 4.5) and P10101 (H+R+RBI 1.5).
    assert len(snaps) == 2
    assert not snaps.duplicated(["game_pk", "player_id", "kind", "line"]).any()
    g = data["slates"][0]["games"][0]
    shown = {("pitcher", g["pitchers"]["home"]["id"]): g["pitchers"]["home"]["book"][0]}
    hitter = next(b for b in g["lineups"]["home"] if b["id"] == 10101)
    shown[("batter", 10101)] = hitter["book"][0]
    for r in snaps.itertuples():
        b = shown[(r.kind, r.player_id)]
        for col in ("p_model", "p_book", "ev_over", "ev_under"):
            assert getattr(r, col) == pytest.approx(b[col], abs=1e-4), col
        assert r.fetched_at == pd.Timestamp(RUN1)
        assert r.game_start == pd.Timestamp("2025-06-15T23:05:00Z")


def test_rerun_on_same_prices_adds_nothing(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    before = storage.read("prop_snapshots")
    _publish(tmp_path, RUN1 + timedelta(minutes=20))  # inside the 2 h refresh gap
    after = storage.read("prop_snapshots")
    assert len(after) == len(before)


def test_second_download_adds_rows_and_keeps_old(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    first = storage.read("prop_snapshots")
    _publish(tmp_path, RUN1 + timedelta(hours=2, minutes=30))
    after = storage.read("prop_snapshots")
    assert len(after) == 2 * len(first)
    old = after[after["fetched_at"] == pd.Timestamp(RUN1)].sort_values(["kind"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(old, first.sort_values(["kind"]).reset_index(drop=True),
                                  check_dtype=False)


def test_no_key_still_publishes(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    data = _publish(tmp_path, RUN1)
    assert data["slates"] and storage.read("prop_snapshots").empty


# ---- pricing schedule -------------------------------------------------------

START_TS = pd.Timestamp("2026-06-10T23:00:00Z")


@pytest.mark.parametrize("minutes_before,last_before,expected", [
    (600, None, None),                       # outside the 8 h window
    (300, None, odds.UNPRICED),
    (300, 360, None),                        # 1 h since last pull, 2 h gap
    (300, 430, odds.REFRESH),                # 2 h 10 m since last
    (150, 170, None),                        # final 3 h: 20 min since last
    (150, 185, odds.REFRESH),                # final 3 h: 35 min since last
    (40, 60, odds.CLOSING),                  # inside 45 min, last pull before it
    (30, 40, None),                          # closing pull already taken
])
def test_fetch_priority(minutes_before, last_before, expected):
    now = START_TS - pd.Timedelta(minutes=minutes_before)
    last = None if last_before is None else START_TS - pd.Timedelta(minutes=last_before)
    assert odds.fetch_priority(START_TS, last, now) == expected


# ---- grading ----------------------------------------------------------------

def _finish_game(k=7, hrr=2, pitcher_starts=True, batter_starts=True, state="Final"):
    """Make the fake upcoming game (9999) final with box score lines for both props."""
    storage.upsert("games", pd.DataFrame([{
        "game_pk": 9999, "game_date": "2025-06-15", "status": "Final", "detailed_state": state,
        "home_team_id": 101, "away_team_id": 104, "season": 2025, "venue_id": 1}]))
    storage.upsert("pitcher_games", pd.DataFrame([{
        "game_pk": 9999, "player_id": 10151, "team_id": 101, "is_starter": pitcher_starts, "k": k}]))
    storage.upsert("batter_games", pd.DataFrame([{
        "game_pk": 9999, "player_id": 10101, "team_id": 101, "is_starter": batter_starts, "hrr": hrr}]))


def test_graded_after_final(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    pending = tracking.update_grades()
    assert set(pending["status"]) == {"pending"}
    _finish_game(k=7, hrr=1)
    g = tracking.update_grades().set_index("kind")
    assert g.loc["pitcher", "status"] == "graded" and g.loc["pitcher", "actual"] == 7
    assert bool(g.loc["pitcher", "over_won"]) is True  # 7 > 4.5
    assert g.loc["batter", "actual"] == 1 and bool(g.loc["batter", "over_won"]) is False  # 1 < 1.5
    assert storage.read("prop_grades").shape[0] == 2


def test_scratched_starter_and_bench_batter_void(fake, dk, tmp_path):
    _publish(tmp_path, RUN1)
    _finish_game(pitcher_starts=False, batter_starts=False)
    g = tracking.update_grades()
    assert set(g["status"]) == {"void"} and g["actual"].isna().all()


def _snap(minutes_before, line, over, under, player=1, kind="pitcher", start="2026-06-10T23:00:00Z"):
    start = pd.Timestamp(start)
    p = odds.no_vig_over(over, under)
    return {"game_pk": 1, "player_id": player, "kind": kind, "line": line, "over": over,
            "under": under, "p_book": p, "p_model": 0.5, "ev_over": 0.0, "ev_under": 0.0,
            "fetched_at": start - pd.Timedelta(minutes=minutes_before), "game_start": start,
            "game_date": pd.Timestamp("2026-06-10"), "lineup_confirmed": True}


def _grade(rows, k=6, starter=True, state="Final"):
    games = pd.DataFrame([{"game_pk": 1, "game_date": "2026-06-10", "status": "Final",
                           "detailed_state": state}])
    pit = pd.DataFrame([{"game_pk": 1, "player_id": 1, "is_starter": starter, "k": k}])
    return tracking.grade(pd.DataFrame(rows), games, pd.DataFrame(), pit).sort_values("fetched_at")


def test_clv_between_two_snapshots_at_same_line():
    g = _grade([_snap(300, 5.5, -110, -110), _snap(30, 5.5, -140, 115)])
    early, late = g.iloc[0], g.iloc[1]
    assert bool(late["is_close"]) and not bool(early["is_close"])
    assert early["clv_over"] == pytest.approx(odds.no_vig_over(-140, 115) - 0.5)
    assert np.isnan(late["clv_over"])  # never compared with itself


def test_lone_early_snapshot_has_no_close():
    g = _grade([_snap(150, 5.5, -110, -110)])
    assert bool(g.iloc[0]["is_close"]) and np.isnan(g.iloc[0]["p_book_close"])


def test_close_at_a_different_line():
    g = _grade([_snap(300, 5.5, -110, -110), _snap(20, 6.5, -110, -110)])
    early = g.iloc[0]
    assert np.isnan(early["p_book_close"]) and early["close_line"] == 6.5


def test_postponed_and_push_are_void():
    assert _grade([_snap(30, 5.5, -110, -110)], state="Postponed")["status"].iloc[0] == "void"
    assert _grade([_snap(30, 6.0, -110, -110)], k=6)["void_reason"].iloc[0] == "push"
    assert _grade([_snap(30, 6.0, -110, -110)], k=7)["status"].iloc[0] == "graded"


def test_closing_pull_paid_before_unpriced_games(tmp_path, monkeypatch):
    from baseball_stats import config
    from test_odds import FakeOddsAPI, NOW
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    api = FakeOddsAPI(n_games=2, credits=400)
    # Game 0 starts in 2 h; game 1 in 6 h. First run prices both.
    api.events[0]["commence_time"] = (NOW + timedelta(hours=2)).isoformat()
    api.events[1]["commence_time"] = (NOW + timedelta(hours=6)).isoformat()
    games = pd.DataFrame([{"game_pk": 100 + i, "home_team": ev["home_team"], "away_team": ev["away_team"],
                           "game_datetime": ev["commence_time"]} for i, ev in enumerate(api.events)])
    odds.fetch_props(games.iloc[[0]], now=NOW, api_key="k", session=api)  # prices game 0 only
    assert api.paid_calls == 1
    # 1 h 20 m later: game 0 is inside 45 min -> closing pull; game 1 is unpriced.
    # Leave room in today's allowance for exactly one game.
    ledger = odds.Ledger.load()
    ledger.spent = ledger.allowance - 2
    ledger.save()
    later = NOW + timedelta(hours=1, minutes=20)
    odds.fetch_props(games, now=later, api_key="k", session=api)
    assert api.paid_calls == 2
    fetched = odds.Ledger.load().fetched
    assert fetched["ev0"] == later.isoformat(timespec="seconds") and "ev1" not in fetched
    # And the closing pull happens only once.
    ledger = odds.Ledger.load(); ledger.spent = 0; ledger.save()
    odds.fetch_props(games.iloc[[0]], now=later + timedelta(minutes=20), api_key="k", session=api)
    assert api.paid_calls == 2
