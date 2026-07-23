"""Derive per-customer economic fields from the cleaned Telco data.

These fields drive the decision engine and follow the column contract:
plan, MRR, CLV, retention_cost. Assumptions are documented in
docs/economic_assumptions.md.
"""

import pandas as pd

from src.config import (
    CLV_HORIZON_MONTHS,
    CLV_METHOD,
    COX_CATEGORICAL_COVARIATES,
    COX_NUMERIC_COVARIATES,
    DISCOUNT_RATE,
    OFFER_MONTHS,
    OUTREACH_COST,
)
from src.survival import cox_expected_remaining, expected_remaining_by_group

# Cox needs a reasonable sample to fit; below this we use Kaplan-Meier.
_MIN_COX_ROWS = 200


def expected_remaining_months(df: pd.DataFrame) -> pd.Series:
    """Per-customer expected remaining lifetime, via Cox or KM.

    Cox (per-customer, uses all covariates) is preferred; falls back to the
    per-contract Kaplan-Meier estimator if Cox is unavailable, the sample is
    too small, or the fit misbehaves.
    """
    if CLV_METHOD == "cox" and len(df) >= _MIN_COX_ROWS:
        try:
            rem = cox_expected_remaining(
                df, COX_NUMERIC_COVARIATES, COX_CATEGORICAL_COVARIATES,
                "tenure", "churned", CLV_HORIZON_MONTHS,
            )
            if rem.notna().all() and (rem > 0).all():
                return rem
        except Exception:
            pass  # fall through to KM
    return expected_remaining_by_group(
        df, "Contract", "tenure", "churned", CLV_HORIZON_MONTHS
    )


def add_economic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add plan, MRR, CLV and retention_cost columns (pure function)."""
    df = df.copy()

    df["plan"] = df["Contract"]
    df["MRR"] = df["MonthlyCharges"]

    # CLV: expected remaining revenue = MRR x expected remaining lifetime,
    # estimated from a survival model (Cox per-customer, or KM per-contract).
    df["CLV"] = df["MRR"] * expected_remaining_months(df)

    # Retention cost: contract-dependent outreach cost plus a win-back
    # discount that scales with MRR — so cost varies per customer.
    df["retention_cost"] = (
        df["Contract"].map(OUTREACH_COST)
        + DISCOUNT_RATE * df["MRR"] * OFFER_MONTHS
    )

    return df
