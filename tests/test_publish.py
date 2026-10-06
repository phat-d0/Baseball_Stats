import json
from datetime import UTC, date, datetime

import numpy as np
import pytest

from baseball_stats import collect, config, mlb_api, model, publish, slate, storage
from fake_mlb import FakeMLB

START = date(2025, 4, 1)
DAYS = 75
TODAY = date(2025, 6, 15)  # first day after the fake history


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    f = FakeMLB(START, days=DAYS)

    def schedule(start, end, *, game_type="R", cache=True):
        if start == end == TODAY:
            return f.upcoming(TODAY)
        return f.schedule(start, end)

    monkeypatch.setattr(mlb_api, "schedule", schedule)
    monkeypatch.setattr(mlb_api, "boxscore", f.boxscore)
    monkeypatch.setattr(mlb_api, "people", f.people)
    return f


def test_pmf_and_p_over():
    dist = model.pmf(np.array([1.0, 5.0]), 0.1, 15)
    assert dist.sum(axis=1) == pytest.approx([1, 1])
    assert model.p_over(dist, 4.5)[1] == pytest.approx(dist[1, 5:].sum())
    assert model.p_over(dist, 0.5)[0] == pytest.approx(1 - dist[0, 0])


def test_fit_alpha_recovers_overdispersion():
    rng = np.random.default_rng(0)
    mu = rng.uniform(1, 3, 20000)
    alpha = 0.3
    y = rng.negative_binomial(1 / alpha, (1 / alpha) / (1 / alpha + mu))
    assert model.fit_alpha(y, mu) == pytest.approx(alpha, abs=0.05)


def test_projected_lineups(fake):
    collect.collect_games(START, date(2025, 6, 14))
    games = storage.read("games")
    proj = slate.recent_lineups(storage.read("batter_games"), games)
    assert proj.groupby("team_id").size().eq(9).all()


def test_publish_end_to_end(fake, tmp_path):
    now = datetime(2025, 6, 15, 16, 0, tzinfo=UTC)
    data = publish.publish(tmp_path / "site", history_start=START, now=now)
    site = tmp_path / "site"
    assert (site / "index.html").exists() and (site / "icons" / "icon-192.png").exists()
    loaded = json.loads((site / "data.json").read_text())
    assert loaded["slates"][0]["label"] == "Today"
    g = loaded["slates"][0]["games"][0]
    sp = g["pitchers"]["home"]
    assert sp["mu"] > 0 and sum(sp["pmf"]) == pytest.approx(1, abs=1e-3)
    assert len(g["lineups"]["home"]) == 9 and all(b["confirmed"] for b in g["lineups"]["home"])
    assert [b["order"] for b in g["lineups"]["away"]] == list(range(1, 10))
    rec = loaded["record"]["batter"]
    assert rec["n"] > 0 and rec["calibration"]
    # A second run is incremental and gives the same answer.
    again = publish.publish(tmp_path / "site", history_start=START, now=now)
    assert again["slates"][0]["games"][0]["pitchers"]["home"]["mu"] == pytest.approx(sp["mu"], rel=1e-3)


class FakeDK:
    """The Odds API for the fake upcoming game (Dogs @ Aces)."""

    def __init__(self):
        self.credits = 300

    def get(self, url, params=None, timeout=None):
        from test_odds import Resp
        if url.endswith("/events"):
            return Resp([{"id": "e1", "home_team": "Aces", "away_team": "Dogs",
                          "commence_time": f"{TODAY}T23:05:00Z"}], self.credits, 0)
        self.credits -= 2
        line = lambda who, pt, o, u, key: {"key": key, "outcomes": [
            {"name": "Over", "description": who, "price": o, "point": pt},
            {"name": "Under", "description": who, "price": u, "point": pt}]}
        return Resp({"id": "e1", "bookmakers": [{"key": "draftkings", "markets": [
            line("P10151", 4.5, -110, -110, "pitcher_strikeouts"),
            line("P10101", 1.5, 150, -190, "batter_hits_runs_rbis"),
        ]}]}, self.credits, 2)


def test_publish_with_draftkings(fake, tmp_path, monkeypatch):
    from baseball_stats import odds
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("ODDS_API_RESET_DAY", "1")
    monkeypatch.setattr(odds.requests, "get", FakeDK().get)
    data = publish.publish(tmp_path / "site", history_start=START,
                           now=datetime(2025, 6, 15, 16, 0, tzinfo=UTC))
    g = data["slates"][0]["games"][0]
    sp = g["pitchers"]["home"]
    assert sp["id"] == 10151 and sp["book"][0]["line"] == 4.5
    b = sp["book"][0]
    p_over = sum(sp["pmf"][5:])
    assert b["p_model"] == pytest.approx(p_over, abs=1e-3)
    # Edges use the blend of DraftKings' chance and the model's (baseball_stats/blend.py).
    assert b["p_book"] < b["p_blend"] < b["p_model"] or b["p_model"] < b["p_blend"] < b["p_book"]
    assert b["ev_over"] == pytest.approx(b["p_blend"] * (1 + 100 / 110) - 1, abs=1e-3)
    assert b["p_book"] == pytest.approx(0.5)
    hitter = next(x for x in g["lineups"]["home"] if x["id"] == 10101)
    assert hitter["book"][0]["over"] == 150
    assert data["odds_source"]["events_priced"] == 1 and data["odds_source"]["spent_today"] == 2
    assert g["pitchers"]["away"]["book"] is None


def test_dist_model_gives_a_distribution():
    import pandas as pd
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 4000)
    df = pd.DataFrame({"x": x, "is_starter": True,
                       "target_hrr": rng.poisson(0.5 + 2 * x)})
    m = model.DistModel("batter").fit(df)
    mu, dist = m.distribution(df.iloc[:5])
    assert dist.shape == (5, model.MAX_COUNT["batter"] + 1)
    assert dist.sum(axis=1) == pytest.approx(np.ones(5))
    assert mu == pytest.approx(dist @ np.arange(dist.shape[1]))
    lo, hi = m.predict(pd.DataFrame({"x": [0.05, 0.95]}))
    assert hi > lo + 1


def test_publish_with_dist_hitters(fake, tmp_path, monkeypatch):
    """BASEBALL_MODEL_BATTER=dist switches hitters to model.DistModel; 'current' can shadow it."""
    from baseball_stats import odds
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("ODDS_API_RESET_DAY", "1")
    monkeypatch.setenv("BASEBALL_MODEL_BATTER", "dist")
    monkeypatch.setenv("BASEBALL_SHADOW_BATTER", "current")
    monkeypatch.setattr(odds.requests, "get", FakeDK().get)
    data = publish.publish(tmp_path / "site", history_start=START,
                           now=datetime(2025, 6, 15, 16, 0, tzinfo=UTC))
    assert data["record"]["batter"]["model_name"] == "dist"
    hitter = next(x for x in data["slates"][0]["games"][0]["lineups"]["home"] if x["id"] == 10101)
    assert sum(hitter["pmf"]) == pytest.approx(1, abs=1e-3)
    log = storage.read("prop_snapshots")
    row = log[log["kind"] == "batter"].iloc[0]
    assert row["model_name"] == "dist" and row["shadow_name"] == "current"
    assert 0 < row["p_shadow"] < 1
