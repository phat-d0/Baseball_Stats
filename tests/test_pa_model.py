"""Plate appearance data, models and their leakage checks."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from baseball_stats import collect, config, mlb_api, pa_data, publish, statcast, storage
from fake_mlb import FakeMLB

START = date(2025, 4, 1)
DAYS = 30


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    f = FakeMLB(START, days=DAYS)
    for name in ("schedule", "boxscore", "people"):
        monkeypatch.setattr(mlb_api, name, getattr(f, name))
    monkeypatch.setattr(statcast, "fetch_day", lambda d, **kw: f.pitches(d))
    return f


def _all_pitches(f):
    return pd.concat([f.pitches(date.fromordinal(START.toordinal() + i)) for i in range(DAYS)],
                     ignore_index=True)


# ---- data ---------------------------------------------------------------------

def test_outcome_classes():
    assert pa_data.outcome_class("strikeout_double_play") == "K"
    assert pa_data.outcome_class("hit_by_pitch") == "BB"
    assert pa_data.outcome_class("home_run") == "HR"
    assert pa_data.outcome_class("grounded_into_double_play") == "OUT"
    assert pa_data.outcome_class("field_error") == "OUT"
    assert pa_data.outcome_class("caught_stealing_2b") is None
    assert pa_data.outcome_class("truncated_pa") is None
    assert pa_data.outcome_class(None) is None


def test_one_row_per_pa_and_matches_box_scores(fake):
    collect.collect_games(START, date(2025, 4, 30))
    pitches = _all_pitches(fake)
    pa = pa_data.build(pitches, storage.read("batter_games"), storage.read("pitcher_games"))
    bg = storage.read("batter_games")
    assert len(pa) == int(bg["pa"].sum())
    assert not pa.duplicated(["game_pk", "at_bat_number"]).any()
    last = pitches.dropna(subset=["events"]).set_index(["game_pk", "at_bat_number"])["events"]
    got = pa.set_index(["game_pk", "at_bat_number"])
    assert (got["outcome"] == last.reindex(got.index).map(pa_data.outcome_class)).all()
    check = pa_data.reconcile(pa, bg, storage.read("pitcher_games"))
    assert check["match_share"] == 1.0 and check["rbi_gap"] == 0 and check["passed"]
    assert pa["batter_slot"].between(1, 9).all() and pa["pitcher_is_starter"].notna().all()


def test_in_game_counters_hand_built():
    rows = []
    # Pitcher 1 faces batters 11, 12, 13 (3, 1, 2 pitches); pitcher 2 then faces 11.
    for ab, (pitcher, batter, n, event) in enumerate(
            [(1, 11, 3, "strikeout"), (1, 12, 1, "single"), (1, 13, 2, "walk"), (2, 11, 4, "field_out")], 1):
        for k in range(1, n + 1):
            rows.append({"game_pk": 5, "game_date": "2025-05-01", "at_bat_number": ab, "pitch_number": k,
                         "batter": batter, "pitcher": pitcher, "stand": "R", "p_throws": "R",
                         "events": event if k == n else None, "inning": 1, "inning_topbot": "Top",
                         "outs_when_up": 0, "on_1b": None, "on_2b": None, "on_3b": None,
                         "bat_score": 0, "post_bat_score": 0, "n_thruorder_pitcher": 1})
    pa = pa_data.build(pd.DataFrame(rows))
    assert list(pa["pitches"]) == [3, 1, 2, 4]
    assert list(pa["bf_before"]) == [0, 1, 2, 0]
    assert list(pa["pitches_before"]) == [0, 3, 4, 0]
    assert list(pa["baserunners_before"]) == [0, 0, 1, 0]
    assert list(pa["is_last_bf"]) == [False, False, True, True]
    assert list(pa["batter_pa_number"]) == [1, 1, 1, 2]
    assert list(pa["times_through_order"]) == [1, 1, 1, 1]


def test_transitions_sum_to_one_and_use_earlier_games(fake):
    collect.collect_games(START, date(2025, 4, 30))
    pa = pa_data.build(_all_pitches(fake))
    t = pa_data.build_transitions(pa)
    assert np.allclose(t.groupby(["outcome", "state"])["share"].sum(), 1.0)
    assert set(t["state"]) <= set(range(24)) and t["next_state"].between(0, 24).all()
    cut = pd.Timestamp("2025-04-15")
    early = pa_data.build_transitions(pa, before=cut)
    assert early["n"].sum() == (pa["game_date"] < cut).sum()
    # A home run with the bases empty scores exactly one run.
    hr = t[(t["outcome"] == "HR") & (t["state"] % 8 == 0)]
    assert (hr["runs"] == 1).all()


def test_publish_backfills_plate_appearances(fake, tmp_path, monkeypatch):
    today = date(2025, 5, 1)
    monkeypatch.setattr(publish, "STATCAST_DAYS_PER_RUN", 10)
    monkeypatch.setattr(mlb_api, "schedule", lambda s, e, **k: fake.schedule(s, e))
    publish.update_data(today, START)
    check = publish.update_statcast(today, START)
    pa = storage.read("plate_appearances")
    days = sorted(pd.to_datetime(pa["game_date"]).dt.date.unique())
    assert days[-1] == date(2025, 4, 30) and len(days) == 10  # newest first
    assert check["passed"]
    publish.update_statcast(today, START)
    publish.update_statcast(today, START)
    assert len(publish.statcast_days_missing(today, START)) == 0
