import numpy as np
import pandas as pd
import pytest

from src.ingest import REQUIRED_COLUMNS


def _telco_row(customer_id, contract, tenure, monthly, total, churn):
    return {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    }


@pytest.fixture
def raw_telco_df():
    """Small handcrafted frame mimicking the raw Telco CSV schema.

    Includes a blank-string TotalCharges row (tenure 0), like the real data.
    """
    rows = [
        _telco_row("A-1", "Month-to-month", 1, 70.0, "70.0", "Yes"),
        _telco_row("A-2", "Month-to-month", 12, 85.5, "1026.0", "Yes"),
        _telco_row("A-3", "Month-to-month", 30, 45.0, "1350.0", "No"),
        _telco_row("A-4", "One year", 24, 60.0, "1440.0", "No"),
        _telco_row("A-5", "One year", 48, 95.0, "4560.0", "No"),
        _telco_row("A-6", "Two year", 60, 105.0, "6300.0", "No"),
        _telco_row("A-7", "Two year", 72, 25.0, "1800.0", "No"),
        _telco_row("A-8", "Month-to-month", 0, 50.0, " ", "No"),  # blank TotalCharges
        _telco_row("A-9", "Month-to-month", 5, 99.0, "495.0", "Yes"),
        _telco_row("A-10", "One year", 36, 30.0, "1080.0", "Yes"),
        _telco_row("A-11", "Two year", 40, 80.0, "3200.0", "No"),
        _telco_row("A-12", "Month-to-month", 8, 20.0, "160.0", "No"),
    ]
    df = pd.DataFrame(rows)
    assert list(df.columns) == REQUIRED_COLUMNS
    return df


@pytest.fixture
def scored_df():
    """Customers already scored with churn probability — for decision tests."""
    rng = np.random.default_rng(0)
    n = 50
    probs = rng.uniform(0.05, 0.95, n)
    df = pd.DataFrame({
        "customer_id": [f"S-{i}" for i in range(n)],
        "churn_probability": probs,
        "CLV": rng.uniform(300, 6000, n),
        "retention_cost": rng.uniform(30, 150, n),
    })
    df["risk_band"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.30, 0.60, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return df
