from src.economics import add_economic_fields
from src.ingest import clean_telco_data


def _enriched(raw_telco_df):
    return add_economic_fields(clean_telco_data(raw_telco_df))


def test_retention_cost_varies_within_plan(raw_telco_df):
    """Regression: the old generator gave every customer on a plan the
    identical retention cost, breaking the efficiency ranking."""
    df = _enriched(raw_telco_df)
    for plan, group in df.groupby("plan"):
        if len(group) > 1:
            assert group["retention_cost"].nunique() > 1, (
                f"retention_cost is constant within plan {plan!r}"
            )


def test_clv_positive_and_increases_with_mrr(raw_telco_df):
    df = _enriched(raw_telco_df)
    assert (df["CLV"] > 0).all()

    # Same contract and tenure, different MRR -> higher MRR means higher CLV
    doubled = raw_telco_df.copy()
    doubled["MonthlyCharges"] = doubled["MonthlyCharges"] * 2
    df_doubled = _enriched(doubled)
    assert (df_doubled["CLV"].values > df["CLV"].values).all()


def test_two_year_clv_exceeds_month_to_month(raw_telco_df):
    df = _enriched(raw_telco_df)
    # Two-year customers churn less, so their KM-derived expected remaining
    # lifetime (CLV per unit MRR) should be higher on average.
    per_mrr = (df["CLV"] / df["MRR"]).groupby(df["plan"]).mean()
    assert per_mrr["Two year"] > per_mrr["Month-to-month"]


def test_contract_columns_present(raw_telco_df):
    df = _enriched(raw_telco_df)
    for col in ["plan", "MRR", "CLV", "retention_cost", "churned", "customer_id"]:
        assert col in df.columns
