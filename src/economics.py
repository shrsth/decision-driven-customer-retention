"""Derive per-customer economic fields from the cleaned Telco data.

These fields drive the decision engine and follow the column contract:
plan, MRR, CLV, retention_cost. Assumptions are documented in
docs/economic_assumptions.md.
"""

import pandas as pd

from src.config import (
    CLV_HORIZON_MONTHS,
    DISCOUNT_RATE,
    OFFER_MONTHS,
    OUTREACH_COST,
)
from src.survival import expected_remaining_by_group


def add_economic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add plan, MRR, CLV and retention_cost columns (pure function)."""
    df = df.copy()

    df["plan"] = df["Contract"]
    df["MRR"] = df["MonthlyCharges"]

    # CLV: expected remaining revenue = MRR x empirical expected remaining
    # lifetime. Lifetime comes from a Kaplan-Meier survival curve fit per
    # contract type on the (right-censored) tenure/churn data — data-derived,
    # not a hand-tuned formula.
    expected_remaining_months = expected_remaining_by_group(
        df,
        group_col="Contract",
        duration_col="tenure",
        event_col="churned",
        forward_months=CLV_HORIZON_MONTHS,
    )
    df["CLV"] = df["MRR"] * expected_remaining_months

    # Retention cost: contract-dependent outreach cost plus a win-back
    # discount that scales with MRR — so cost varies per customer.
    df["retention_cost"] = (
        df["Contract"].map(OUTREACH_COST)
        + DISCOUNT_RATE * df["MRR"] * OFFER_MONTHS
    )

    return df
