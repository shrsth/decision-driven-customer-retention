import numpy as np
import pandas as pd

from src.simulation import simulate_decision_quality


def _act(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "churn_probability": rng.uniform(0.3, 0.9, n),
        "CLV": rng.uniform(500, 6000, n),
        "retention_cost": rng.uniform(30, 150, n),
    })


def test_simulation_structure_and_bounds():
    sim = simulate_decision_quality(_act(), assumed_save_rate=0.3, n_sims=1000)
    assert set(["roi_mean", "roi_p5", "roi_p95", "prob_profitable", "rois"]) <= set(sim)
    assert sim["roi_p5"] <= sim["roi_mean"] <= sim["roi_p95"]
    assert 0.0 <= sim["prob_profitable"] <= 1.0
    assert len(sim["rois"]) == 1000


def test_empty_act_returns_none():
    empty = pd.DataFrame(
        {"churn_probability": [], "CLV": [], "retention_cost": []}
    )
    assert simulate_decision_quality(empty, 0.3) is None


def test_higher_save_rate_shifts_roi_up():
    act = _act()
    lo = simulate_decision_quality(act, assumed_save_rate=0.2, rate_std=0.01)
    hi = simulate_decision_quality(act, assumed_save_rate=0.6, rate_std=0.01)
    assert hi["roi_mean"] > lo["roi_mean"]
