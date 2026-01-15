import pandas as pd

from src.features.feature_builder import build_feature_table
from src.models.train_logistic import pipeline, FEATURES
from src.decision.retention_strategy import (
    apply_decision_strategy,
    build_retention_scores,
    select_customers_under_budget,
    assign_action_segments
)

# --------------------------------------------------
# Risk band assignment (operational, not statistical)
# --------------------------------------------------
def assign_risk_band(churn_prob: float) -> str:
    if churn_prob >= 0.60:
        return "HIGH"
    elif churn_prob >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


# --------------------------------------------------
# Expected loss analysis (diagnostic, not decision)
# --------------------------------------------------
def expected_loss_analysis(row):
    churn_prob = row["churn_probability"]
    retention_cost = row["retention_cost"]
    clv = row["CLV"]

    return pd.Series({
        "loss_if_act": (1 - churn_prob) * retention_cost,
        "loss_if_ignore": churn_prob * clv
    })


# --------------------------------------------------
# Decision explainability helpers
# --------------------------------------------------
def generate_decision_reason(row, clv_median, cost_median):
    reasons = []

    if row["risk_band"] == "HIGH":
        reasons.append("High churn risk")
    elif row["risk_band"] == "MEDIUM":
        reasons.append("Moderate churn risk")

    if row["CLV"] >= clv_median:
        reasons.append("High customer value")

    if row["retention_cost"] <= cost_median:
        reasons.append("Low retention cost")

    if row["engagement_velocity"] < 0:
        reasons.append("Declining engagement")

    return " + ".join(reasons[:3])


def recommend_action(row):
    if row["risk_band"] == "HIGH" and row["CLV"] >= 3000:
        return "Personal retention call + premium discount"
    elif row["risk_band"] == "HIGH":
        return "Targeted discount offer"
    elif row["risk_band"] == "MEDIUM":
        return "Re-engagement email campaign"
    else:
        return "Standard follow-up"


# --------------------------------------------------
# Tier 1: Core decision engine (PURE PYTHON)
# --------------------------------------------------
def load_and_compute_decisions(
    budget: float,
    max_customers: int,
    strategy: str
):
    # -----------------------------
    # Load data
    # -----------------------------
    customers = pd.read_csv("data/raw/customers.csv")
    behavior = pd.read_csv("data/raw/behavior.csv")

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df = build_feature_table(customers, behavior)

    # -----------------------------
    # Churn prediction (signal only)
    # -----------------------------
    X = df[FEATURES]
    df["churn_probability"] = pipeline.predict_proba(X)[:, 1]

    # -----------------------------
    # Risk bands (operational)
    # -----------------------------
    df["risk_band"] = df["churn_probability"].apply(assign_risk_band)

    # -----------------------------
    # Diagnostic expected loss
    # -----------------------------
    loss_df = df.apply(expected_loss_analysis, axis=1)
    df = pd.concat([df, loss_df], axis=1)

    high_risk = df[df["risk_band"] == "HIGH"]
    loss_comparison = (
        high_risk["loss_if_ignore"] > high_risk["loss_if_act"]
    ).mean()

    # -----------------------------
    # Economic scoring
    # -----------------------------
    df = build_retention_scores(df)

    # -----------------------------
    # Strategy-aware prioritization
    # -----------------------------
    strategy_df = apply_decision_strategy(df, strategy)

    # -----------------------------
    # Budget + capacity constraints
    # -----------------------------
    selected_df, spent_budget = select_customers_under_budget(
        strategy_df,
        total_budget=budget,
        max_customers=max_customers
    )

    # -----------------------------
    # Final action segmentation
    # -----------------------------
    final_df = assign_action_segments(df, selected_df)

    # -----------------------------
    # Explainability & recommendations
    # -----------------------------
    clv_median = final_df["CLV"].median()
    cost_median = final_df["retention_cost"].median()

    final_df["decision_reason"] = final_df.apply(
        generate_decision_reason,
        axis=1,
        args=(clv_median, cost_median)
    )

    final_df["recommended_action"] = final_df.apply(
        recommend_action,
        axis=1
    )

    final_df["efficiency"] = (
        final_df["net_retention_value"] / final_df["retention_cost"]
    )

    # -----------------------------
    # Outputs
    # -----------------------------
    return final_df, selected_df, spent_budget, loss_comparison
