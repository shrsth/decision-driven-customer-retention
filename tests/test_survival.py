import numpy as np

from src.survival import (
    expected_remaining_life,
    kaplan_meier,
)


def test_kaplan_meier_matches_hand_computation():
    # 5 customers: churn at t=1, censored at t=2, churn at t=3,
    # churn at t=4, censored at t=5.
    durations = [1, 2, 3, 4, 5]
    events = [1, 0, 1, 1, 0]
    times, surv = kaplan_meier(durations, events)

    # S(1) = 1 - 1/5 = 0.8
    # S(2) = 0.8 (censored, no drop)
    # S(3) = 0.8 * (1 - 1/3) = 0.5333
    # S(4) = 0.5333 * (1 - 1/2) = 0.2667
    # S(5) = 0.2667 (censored)
    expected = [0.8, 0.8, 0.5333, 0.2667, 0.2667]
    assert np.allclose(surv, expected, atol=1e-3)


def test_survival_is_non_increasing():
    rng = np.random.default_rng(0)
    durations = rng.integers(1, 60, 200)
    events = rng.integers(0, 2, 200)
    _, surv = kaplan_meier(durations, events)
    assert np.all(np.diff(surv) <= 1e-9)


def test_cox_expected_remaining_individualizes_within_contract():
    """Cox uses all covariates, so two customers on the same contract but with
    different charges/services get different expected lifetimes — something the
    per-contract KM estimator cannot do."""
    import warnings

    import pandas as pd

    from src.survival import cox_expected_remaining

    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "tenure": rng.integers(1, 60, n),
        "MonthlyCharges": rng.uniform(20, 110, n),
        "TotalCharges": rng.uniform(50, 6000, n),
        "Contract": rng.choice(["Month-to-month", "Two year"], n),
        "InternetService": rng.choice(["Fiber optic", "DSL", "No"], n),
    })
    # churn more likely for high charges + month-to-month
    p = 0.15 + 0.3 * (df["Contract"] == "Month-to-month") + 0.002 * df["MonthlyCharges"]
    df["churned"] = (rng.uniform(0, 1, n) < p.clip(0, 1)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rem = cox_expected_remaining(
            df, ["MonthlyCharges", "TotalCharges"],
            ["Contract", "InternetService"], "tenure", "churned", 60,
        )

    assert (rem > 0).all() and rem.notna().all()
    assert (rem <= 60 + 1e-6).all()
    # within a single contract, expected lifetime still varies (KM would not)
    m2m = rem[df["Contract"] == "Month-to-month"]
    assert m2m.std() > 0.5


def test_expected_remaining_life_bounded_and_higher_for_healthier_group():
    # Group A churns fast; group B rarely churns.
    times_a, surv_a = kaplan_meier([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    times_b, surv_b = kaplan_meier([10, 20, 30, 40, 50], [0, 0, 0, 0, 1])

    rem_a = expected_remaining_life(times_a, surv_a, at_tenure=0, forward_months=36)
    rem_b = expected_remaining_life(times_b, surv_b, at_tenure=0, forward_months=36)

    assert 0 <= rem_a <= 36
    assert 0 <= rem_b <= 36
    assert rem_b > rem_a  # the loyal group has more expected life ahead
