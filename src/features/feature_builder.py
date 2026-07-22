"""Feature table for churn modeling, built from the SQLite customers table.

Model features are customer attributes from the Telco dataset. Economic
fields (CLV, retention_cost) are deliberately NOT model features — they
drive the decision layer, not the churn signal.
"""

from pathlib import Path

import pandas as pd

from src.config import DB_PATH
from src.sql_feature_queries import load_customers_from_sql

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CONTRACT_COLUMNS = ["customer_id", "plan", "MRR", "CLV", "retention_cost", "churned"]


def build_feature_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    """One row per customer: model features + decision-contract columns."""
    df = load_customers_from_sql(db_path)
    return df[CONTRACT_COLUMNS + FEATURES]
