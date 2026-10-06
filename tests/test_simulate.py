"""Strikeout simulation."""

import numpy as np
import pytest
from scipy import stats

from baseball_stats import model, simulate
from baseball_stats.pa_data import OUTCOMES
from baseball_stats.pa_model import PitchSampler


def _probs(p_k):
    """Every batter: strikeout with p_k, otherwise an out."""
    p = np.zeros((9, 3, len(OUTCOMES)))
    p[..., OUTCOMES.index("K")] = p_k
    p[..., OUTCOMES.index("OUT")] = 1 - p_k
    return p


def _sampler():
    s = PitchSampler()
    s.by_class = {c: np.r_[np.zeros(3), 1.0, np.zeros(s.MAX - 4)] for c in OUTCOMES}  # always 4
    return s


def test_fixed_bf_matches_binomial():
    r = simulate.simulate_start(_probs(0.25), None, _sampler(), 1.0, seed=1, n_sims=40000, fixed_bf=24)
    assert (r["bf"] == 24).all()
    sim = np.bincount(r["k"], minlength=25)[:25] / len(r["k"])
    exact = stats.binom.pmf(np.arange(25), 24, 0.25)
    assert np.abs(sim - exact).max() < 0.01


def test_forced_exit_after_18():
    grid = np.zeros((simulate.MAX_BF, simulate.MAX_PITCHES // simulate.PITCH_STEP + 1,
                     simulate.MAX_RUNNERS + 1))
    grid[17:] = 1.0  # certain exit once he has faced 18
    r = simulate.simulate_start(_probs(0.2), grid, _sampler(), 1.0, seed=3, n_sims=5000)
    assert r["bf"].max() == 18 and (r["bf"] == 18).all()
    assert r["pitches"].max() == 18 * 4


def test_complete_game_stops_at_27_outs():
    grid = np.zeros((simulate.MAX_BF, simulate.MAX_PITCHES // simulate.PITCH_STEP + 1,
                     simulate.MAX_RUNNERS + 1))
    r = simulate.simulate_start(_probs(0.3), grid, _sampler(), 1.0, seed=4, n_sims=2000)
    assert (r["bf"] == 27).all()  # every batter is an out, never pulled


def test_seeds_and_distribution():
    grid = np.full((simulate.MAX_BF, simulate.MAX_PITCHES // simulate.PITCH_STEP + 1,
                    simulate.MAX_RUNNERS + 1), 0.06)
    a = simulate.simulate_start(_probs(0.24), grid, _sampler(), 1.0, seed=7)
    b = simulate.simulate_start(_probs(0.24), grid, _sampler(), 1.0, seed=7)
    c = simulate.simulate_start(_probs(0.24), grid, _sampler(), 1.0, seed=8)
    assert (a["k"] == b["k"]).all()
    da, dc = simulate.distribution(a["k"], 15), simulate.distribution(c["k"], 15)
    assert np.isclose(da.sum(), 1) and (da > 0).all()
    for line in (3.5, 4.5, 5.5, 6.5, 7.5):
        pa_, pc = model.p_over(da[None], line)[0], model.p_over(dc[None], line)[0]
        assert abs(pa_ - pc) < 0.03
    assert simulate.seed_for(1, 2) == simulate.seed_for(1, 2) != simulate.seed_for(1, 3)


# ---- integration: switch, shadow, saved models ---------------------------------

import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import pandas as pd  # noqa: E402

from baseball_stats import odds, publish, sim_models, statcast, storage  # noqa: E402
from test_publish import START, FakeDK, fake  # noqa: E402,F401

RUN = datetime(2025, 6, 15, 16, 0, tzinfo=UTC)


@pytest.fixture
def pa_env(fake, monkeypatch):
    monkeypatch.setattr(statcast, "fetch_day", lambda d, **kw: fake.pitches(d))
    monkeypatch.setattr(publish, "STATCAST_DAYS_PER_RUN", 400)
    monkeypatch.setattr(publish, "MIN_PA_ROWS", 1000)
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("ODDS_API_RESET_DAY", "1")
    monkeypatch.setattr(odds.requests, "get", FakeDK().get)
    return fake


def _run(tmp_path, monkeypatch, active="current", shadow=""):
    monkeypatch.setenv("BASEBALL_MODEL_PITCHER", active)
    monkeypatch.setenv("BASEBALL_SHADOW_PITCHER", shadow)
    return publish.publish(tmp_path / "site", history_start=START, now=RUN, statcast=True)


def _shape(x):
    if isinstance(x, dict):
        return {k: _shape(v) for k, v in x.items() if k not in ("exp_bf",)}
    if isinstance(x, list):
        return [_shape(x[0])] if x else []
    return type(x).__name__ if x is not None else None


def test_publish_with_pa_seq_keeps_data_shape(pa_env, tmp_path, monkeypatch):
    base = _run(tmp_path, monkeypatch, "current")
    data = _run(tmp_path, monkeypatch, "pa_seq")
    assert data["model"]["pitcher"]["name"] == "pa_seq" and data["model"]["pitcher"]["alpha"] is None
    sp = data["slates"][0]["games"][0]["pitchers"]["home"]
    assert sp["pmf"] and sum(sp["pmf"]) == pytest.approx(1, abs=1e-3) and min(sp["pmf"]) > 0
    assert sp["stats"]["exp_bf"] > 0
    g0, g1 = base["slates"][0]["games"][0], data["slates"][0]["games"][0]
    assert set(g0) == set(g1) and set(g0["pitchers"]["home"]) == set(g1["pitchers"]["home"])
    snaps = storage.read("prop_snapshots")
    assert set(snaps["model_name"]) >= {"current"}


def test_shadow_logs_without_changing_the_app(pa_env, tmp_path, monkeypatch):
    plain = _run(tmp_path, monkeypatch, "current")
    storage.table_path("prop_snapshots").unlink()
    odds.Ledger().save()  # forget the download so the prices are logged again
    shadowed = _run(tmp_path, monkeypatch, "current", shadow="pa_simple")
    a = plain["slates"][0]["games"][0]["pitchers"]["home"]
    b = shadowed["slates"][0]["games"][0]["pitchers"]["home"]
    assert a["mu"] == b["mu"] and a["pmf"] == b["pmf"] and a["book"] == b["book"]
    snaps = storage.read("prop_snapshots")
    pit = snaps[snaps["kind"] == "pitcher"]
    assert pit["p_shadow"].notna().all() and set(pit["shadow_name"]) == {"pa_simple"}
    assert snaps.loc[snaps["kind"] == "batter", "p_shadow"].isna().all()
    assert shadowed["model"]["pitcher"]["shadow"] == "pa_simple"


def test_saved_model_with_other_sklearn_is_retrained(pa_env, tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "pa_simple")
    stamp_path = sim_models._stamp_path("pa_simple")
    stamp = json.loads(stamp_path.read_text())
    assert not sim_models.is_stale("pa_simple", RUN, stamp["latest_game_date"])
    stamp["sklearn"] = "0.0.1"
    stamp_path.write_text(json.dumps(stamp))
    assert sim_models.is_stale("pa_simple", RUN, stamp["latest_game_date"])
    before = stamp_path.stat().st_mtime_ns
    _run(tmp_path, monkeypatch, "pa_simple")
    after = json.loads(stamp_path.read_text())
    assert after["sklearn"] != "0.0.1" and stamp_path.stat().st_mtime_ns != before
