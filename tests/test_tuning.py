import warnings

import numpy as np
import pandas as pd

from src.models.tuning import compare_calibration, tune_gbm


def _synthetic(n=400, seed=0):
    """A Telco-schema frame big enough for CV-based calibration/tuning."""
    rng = np.random.default_rng(seed)
    cats = {
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ],
    }
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in cats.items()})
    df["tenure"] = rng.integers(1, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 120, n)
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.8, 1.2, n)
    df["churned"] = (rng.uniform(0, 1, n) < 0.27).astype(int)
    return df


def test_compare_calibration_structure():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compare_calibration(_synthetic())
    assert list(out.columns) == ["method", "brier", "best"]
    assert set(out["method"]) == {
        "Raw logistic regression", "Isotonic", "Platt (sigmoid)"
    }
    assert (out["brier"] >= 0).all()
    assert out["best"].sum() == 1  # exactly one method flagged best


def test_tune_gbm_returns_params_and_score():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = tune_gbm(_synthetic(), n_trials=3, cv=2)
    assert "best_params" in out and "best_auc" in out
    assert 0.0 <= out["best_auc"] <= 1.0
    assert "learning_rate" in out["best_params"]
