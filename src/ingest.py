"""Download and clean the IBM Telco Customer Churn dataset."""

import urllib.request
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TELCO_URL

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def download_telco_data(
    dest: Path = RAW_DATA_PATH,
    url: str = TELCO_URL,
    force: bool = False,
) -> Path:
    """Download the Telco churn CSV to `dest`, skipping if already cached."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"[ingest] Using cached dataset at {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"[ingest] Downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"[ingest] Saved dataset to {dest}")
    return dest


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw Telco dataframe.

    - renames customerID -> customer_id
    - coerces TotalCharges (blank strings for tenure-0 customers) to numeric
    - maps Churn Yes/No -> churned 0/1
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("Dataset is empty")

    df = df.copy()
    df = df.rename(columns={"customerID": "customer_id"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["churned"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])
    return df
