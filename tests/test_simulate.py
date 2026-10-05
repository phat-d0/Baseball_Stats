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
