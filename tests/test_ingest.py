import pandas as pd
import pytest

from src.ingest import clean_telco_data


def test_blank_total_charges_coerced_to_zero(raw_telco_df):
    cleaned = clean_telco_data(raw_telco_df)
    row = cleaned[cleaned["customer_id"] == "A-8"]
    assert row["TotalCharges"].iloc[0] == 0.0
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_churn_label_mapping(raw_telco_df):
    cleaned = clean_telco_data(raw_telco_df)
    assert "Churn" not in cleaned.columns
    assert set(cleaned["churned"].unique()) <= {0, 1}
    assert cleaned.loc[cleaned["customer_id"] == "A-1", "churned"].iloc[0] == 1
    assert cleaned.loc[cleaned["customer_id"] == "A-3", "churned"].iloc[0] == 0


def test_customer_id_renamed(raw_telco_df):
    cleaned = clean_telco_data(raw_telco_df)
    assert "customer_id" in cleaned.columns
    assert "customerID" not in cleaned.columns


def test_missing_column_raises(raw_telco_df):
    broken = raw_telco_df.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        clean_telco_data(broken)


def test_empty_dataframe_raises(raw_telco_df):
    with pytest.raises(ValueError):
        clean_telco_data(raw_telco_df.iloc[0:0])
