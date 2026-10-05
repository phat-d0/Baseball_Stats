from datetime import date

import numpy as np
import pandas as pd
import pytest

from baseball_stats import collect, config, features, mlb_api, parse, slate
from fake_mlb import FakeMLB

START = date(2025, 4, 1)
DH_DAY = date(2025, 4, 10)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    f = FakeMLB(START, days=20, doubleheader_on=DH_DAY)
    for name in ("schedule", "boxscore", "people"):
        monkeypatch.setattr(mlb_api, name, getattr(f, name))
    return f


def test_parse_wind():
    assert parse.parse_wind("12 mph, Out To CF") == (12.0, "Out To CF")
    assert parse.parse_wind("0 mph, None") == (0.0, "None")
    assert parse.parse_wind(None) == (None, None)


def test_parse_boxscore(fake):
    pk = next(iter(fake.box))
    bat, pit, extra = parse.parse_boxscore(pk, fake.box[pk])
    assert len(bat) == 18
    b = bat[0]
    assert b["hrr"] == b["h"] + b["r"] + b["rbi"]
    assert b["batting_order"] == 1 and b["is_starter"]
    home_sp = fake.box[pk]["teams"]["home"]["pitchers"][0]
    assert all(r["opp_starter_id"] == home_sp for r in bat if not r["is_home"])
    assert sum(r["is_starter"] for r in pit) == 2
    assert extra["hp_umpire_id"] is not None


def _build(fake):
    collect.collect_games(START, date(2025, 4, 20))
    return features.build_features(features.load_inputs())


def test_end_to_end_shapes(fake):
    out = _build(fake)
    b, p = out["batter_hrr"], out["pitcher_k"]
    n_games = len(fake.games)
    assert len(b) == n_games * 18
    assert len(p) == n_games * 2  # starters only
    assert p["target_k"].notna().all() and b["target_hrr"].notna().all()
    for col in ("hrr_per_g_l15", "opp_sp_k_pct_shr", "park_r_factor", "team_r_per_g_l15"):
        assert col in b
    for col in ("k_pct_l5", "lineup_k_pct", "opp_team_k_pct_l15", "ump_k_factor", "days_rest"):
        assert col in p


def test_no_leakage(fake):
    b = _build(fake)["batter_hrr"]
    raw = collect.storage.read("batter_games")
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    pid = 10101
    mine = b[b["player_id"] == pid].reset_index(drop=True)
    hist = raw[raw["player_id"] == pid]
    for row in mine.itertuples():
        before = hist[hist["game_date"] < row.game_date]
        exp = before["hrr"].sum() / before["pa"].sum() if len(before) else np.nan
        got = row.hrr_per_pa_car
        assert (np.isnan(exp) and np.isnan(got)) or got == pytest.approx(exp)
    # First game has no history at all.
    assert np.isnan(mine.loc[0, "hrr_per_g_car"])


def test_doubleheader_games_share_pre_day_features(fake):
    b = _build(fake)["batter_hrr"]
    dh = b[(b["game_date"] == pd.Timestamp(DH_DAY)) & (b["player_id"] == 10101)]
    assert len(dh) == 2
    feat_cols = [c for c in b if c.endswith(("_l7", "_car", "_szn"))]
    a, c = dh.iloc[0][feat_cols], dh.iloc[1][feat_cols]
    assert ((a == c) | (a.isna() & c.isna())).all()


def test_slate(fake, monkeypatch):
    collect.collect_games(START, date(2025, 4, 20))
    day = date(2025, 4, 21)
    monkeypatch.setattr(mlb_api, "schedule", lambda *a, **k: fake.upcoming(day))
    out = slate.build_slate(day)
    b, p = out["batter_hrr"], out["pitcher_k"]
    assert len(p) == 2 and len(b) == 18
    assert p["target_k"].isna().all() and b["target_hrr"].isna().all()
    assert p["k_pct_l5"].notna().all() and p["lineup_k_pct"].notna().all()
    assert b["hrr_per_g_l15"].notna().all() and b["opp_sp_k_pct_shr"].notna().all()
    # Slate features equal the pre-day state of a hypothetical training row.
    ace_sp = 10151
    hist = features.build_features(features.load_inputs())["pitcher_k"]
    last = hist[hist["player_id"] == ace_sp].iloc[-1]
    row = p[p["player_id"] == ace_sp].iloc[0]
    assert row["starts_car"] == last["starts_car"] + 1


def _fake_pitches(fake) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for pk, box in fake.box.items():
        for side, opp in (("home", "away"), ("away", "home")):
            sp = box["teams"][opp]["pitchers"][0]
            for batter in box["teams"][side]["batters"]:
                for _ in range(15):
                    desc = rng.choice(["ball", "called_strike", "swinging_strike", "foul",
                                       "hit_into_play"])
                    ev = float(rng.uniform(70, 110)) if desc == "hit_into_play" else np.nan
                    rows.append({
                        "game_pk": pk, "batter": batter, "pitcher": sp,
                        "description": desc, "zone": int(rng.integers(1, 15)),
                        "pitch_type": "FF", "release_speed": float(rng.normal(94, 1)),
                        "launch_speed": ev, "launch_angle": 20.0,
                        "launch_speed_angle": 6 if ev == ev and ev > 100 else 3,
                        "estimated_woba_using_speedangle": 0.4 if ev == ev else np.nan,
                        "woba_value": 0.0, "woba_denom": 1 if desc == "hit_into_play" else 0,
                    })
    return pd.DataFrame(rows)


def test_statcast_features(fake):
    from baseball_stats import statcast, storage
    pitches = _fake_pitches(fake)
    storage.upsert("statcast_batter", statcast.aggregate(pitches, "batter"))
    storage.upsert("statcast_pitcher", statcast.aggregate(pitches, "pitcher"))
    agg = statcast.aggregate(pitches, "pitcher")
    assert (agg["whiff"] <= agg["swing"]).all() and agg["fb_velo"].between(85, 100).all()
    out = _build(fake)
    p, b = out["pitcher_k"], out["batter_hrr"]
    later = p[p["starts_car"] > 0]
    assert later["csw_pct_car"].between(0, 1).all()
    assert later["fb_velo_car"].between(85, 100).all()
    assert b.loc[b["games_car"] >= 3, "xwoba_car"].notna().all()


def test_suspended_game_listed_twice(fake):
    """A suspended game appears on two dates with one game_pk; keep the finished one."""
    pk = next(k for k, g in fake.games.items() if g["officialDate"] == "2025-04-02")
    first = dict(fake.games[pk], officialDate="2025-04-01",
                 status={"abstractGameState": "Final", "detailedState": "Suspended"})
    orig_schedule = fake.schedule

    def schedule(start, end, **kw):
        data = orig_schedule(start, end, **kw)
        data["dates"][0]["games"].append(first)
        return data

    collect.mlb_api.schedule = schedule
    collect.collect_games(START, date(2025, 4, 5))
    games = collect.storage.read("games")
    row = games[games["game_pk"] == pk]
    assert len(row) == 1 and row.iloc[0]["detailed_state"] == "Final"
    assert row.iloc[0]["game_date"] == fake.games[pk]["officialDate"]
