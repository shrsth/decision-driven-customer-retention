"""Uplift tests. The Qini metric is tested offline on synthetic data; the
Hillstrom-dependent model test is skipped when the dataset isn't downloaded."""

import numpy as np
import pytest

from src.config import HILLSTROM_PATH
from src.uplift import qini_curve


def _synthetic_experiment(n=4000, seed=0):
    """Randomized experiment where a hidden 'persuadable' score drives uplift."""
    rng = np.random.default_rng(seed)
    persuadable = rng.uniform(0, 1, n)
    treat = rng.integers(0, 2, n)
    # base response 0.1, plus a treatment effect proportional to persuadability
    p = 0.1 + treat * persuadable * 0.6
    outcome = (rng.uniform(0, 1, n) < p).astype(int)
    return persuadable, treat, outcome


def test_qini_rewards_a_good_ranking():
    persuadable, treat, outcome = _synthetic_experiment()
    rng = np.random.default_rng(1)

    _, _, q_true = qini_curve(persuadable, treat, outcome)       # perfect ranking
    _, _, q_rand = qini_curve(rng.random(len(treat)), treat, outcome)  # random

    # ranking by true persuadability captures far more incremental responders
    assert q_true > q_rand


def test_qini_curve_shapes():
    persuadable, treat, outcome = _synthetic_experiment(n=500)
    fractions, qini, coef = qini_curve(persuadable, treat, outcome)
    assert len(fractions) == len(qini) == 500
    assert fractions[-1] == pytest.approx(1.0)
    assert isinstance(coef, float)


@pytest.mark.skipif(
    not HILLSTROM_PATH.exists(),
    reason="needs the Hillstrom dataset; run the uplift notebook or load_hillstrom() first",
)
def test_two_model_uplift_beats_propensity():
    import warnings

    from src.uplift import (
        fit_two_model_uplift,
        load_hillstrom,
        response_scores,
    )

    df = load_hillstrom()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        up = fit_two_model_uplift(df)
        pr = response_scores(df)
    _, _, q_up = qini_curve(up, df["treat"], df["outcome"])
    _, _, q_pr = qini_curve(pr, df["treat"], df["outcome"])
    assert q_up > q_pr
