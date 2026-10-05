"""Pick logging and market grading (spec: "Pick Logging and Market Grading")."""

from datetime import UTC, datetime, timedelta

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
