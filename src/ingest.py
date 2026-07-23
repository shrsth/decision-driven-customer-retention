"""Download, validate, and clean the IBM Telco Customer Churn dataset."""

import urllib.request
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TELCO_URL
from src.logging_config import get_logger

log = get_logger("ingest")

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def _cleaned_schema():
    """Pandera schema for the cleaned frame — a loud data-quality gate."""
    import pandera.pandas as pa

    return pa.DataFrameSchema(
        {
            "customer_id": pa.Column(str, unique=True),
            "tenure": pa.Column(int, pa.Check.in_range(0, 100)),
            "MonthlyCharges": pa.Column(float, pa.Check.gt(0)),
            "TotalCharges": pa.Column(float, pa.Check.ge(0)),
            "Contract": pa.Column(
                str, pa.Check.isin(["Month-to-month", "One year", "Two year"])
            ),
            "churned": pa.Column(int, pa.Check.isin([0, 1])),
        },
        strict=False,  # allow the other passthrough columns
        coerce=True,
    )


def download_telco_data(
    dest: Path = RAW_DATA_PATH,
    url: str = TELCO_URL,
    force: bool = False,
) -> Path:
    """Download the Telco churn CSV to `dest`, skipping if already cached."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Using cached dataset at %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    log.info("Downloading %s", url)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    log.info("Saved dataset to %s", dest)
    return dest


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw Telco dataframe.

    - renames customerID -> customer_id
    - coerces TotalCharges (blank strings for tenure-0 customers) to numeric
    - maps Churn Yes/No -> churned 0/1
    - validates the result against a Pandera schema
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

    return _cleaned_schema().validate(df)
