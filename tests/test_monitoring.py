import numpy as np
import pandas as pd

from src.monitoring import drift_report, population_stability_index


def test_psi_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    assert population_stability_index(base, base) < 0.01
    assert population_stability_index(base, shifted) > 0.25


def test_drift_report_flags_a_shifted_feature():
    rng = np.random.default_rng(0)
    n = 3000
    ref = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "plan": rng.choice(["A", "B", "C"], n),
    })
    current = pd.DataFrame({
        "x": rng.normal(2.0, 1, n),          # drifted numeric
        "plan": rng.choice(["A", "B", "C"], n),  # stable categorical
    })
    report = drift_report(ref, current, ["x"], ["plan"])

    x_row = report[report["feature"] == "x"].iloc[0]
    plan_row = report[report["feature"] == "plan"].iloc[0]
    assert bool(x_row["drift"]) is True
    assert bool(plan_row["drift"]) is False
