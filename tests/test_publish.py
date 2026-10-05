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
