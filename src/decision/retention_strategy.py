import pandas as pd

# --------------------------------------------------
# Tier 2: Strategy-aware policy layer
# --------------------------------------------------
def apply_decision_strategy(df, strategy: str):
    """
    Adjust retention priority based on business strategy.
    Strategy changes ranking, not eligibility.
    """
    df = df.copy()
    strategy = strategy.lower()

    if strategy == "conservative":
        df["strategy_weight"] = df["risk_band"].map({
            "HIGH": 1.0,
            "MEDIUM": 0.25,
            "LOW": 0.05
        })

    elif strategy == "balanced":
        df["strategy_weight"] = df["risk_band"].map({
            "HIGH": 1.0,
            "MEDIUM": 0.6,
            "LOW": 0.15
        })

    elif strategy == "aggressive":
        df["strategy_weight"] = df["risk_band"].map({
            "HIGH": 1.0,
            "MEDIUM": 0.85,
            "LOW": 0.4
        })

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df["adjusted_priority"] = (
        df["retention_priority_score"] * df["strategy_weight"]
    )

    return df


# --------------------------------------------------
# Scoring helpers
# --------------------------------------------------
def compute_revenue_at_risk(df):
    df = df.copy()
    df["revenue_at_risk"] = df["churn_probability"] * df["CLV"]
    return df


def compute_net_retention_value(df):
    df = df.copy()
    df["net_retention_value"] = df["revenue_at_risk"] - df["retention_cost"]
    return df


def build_retention_scores(df):
    df = compute_revenue_at_risk(df)
    df = compute_net_retention_value(df)

    df["retention_priority_score"] = (
        df["net_retention_value"].clip(lower=0)
    )

    return df


# --------------------------------------------------
# Budget + capacity constrained selection
# --------------------------------------------------
def select_customers_under_budget(
    df,
    total_budget,
    max_customers=None
):
    df = df.copy()

    df = df[df["retention_cost"] > 0]
    df = df[df["net_retention_value"] > 0]

    df["efficiency"] = (
        df["net_retention_value"] / df["retention_cost"]
    )

    sort_col = (
        "adjusted_priority"
        if "adjusted_priority" in df.columns
        else "efficiency"
    )

    df = df.sort_values(sort_col, ascending=False)

    selected = []
    spent = 0

    for _, row in df.iterrows():
        if spent + row["retention_cost"] > total_budget:
            continue

        selected.append(row)
        spent += row["retention_cost"]

        if max_customers and len(selected) >= max_customers:
            break

    selected_df = pd.DataFrame(selected)
    return selected_df, spent


# --------------------------------------------------
# Final action segmentation
# --------------------------------------------------
def assign_action_segments(full_df, selected_df):
    df = full_df.copy()
    selected_ids = set(selected_df["customer_id"])

    def segment(row):
        if row["customer_id"] in selected_ids:
            return "ACT"
        elif row["net_retention_value"] > 0:
            return "MONITOR"
        else:
            return "IGNORE"

    df["action_segment"] = df.apply(segment, axis=1)
    return df
