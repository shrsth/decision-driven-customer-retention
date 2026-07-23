import numpy as np
import pandas as pd

from src.config import SAVE_RATE
from src.decision.retention_strategy import (
    apply_decision_strategy,
    assign_action_segments,
    build_retention_scores,
    select_customers_under_budget,
)
from src.features.feature_builder import FEATURES, build_feature_table
from src.models.train_logistic import load_model

# Model is loaded once and reused (joblib load is cheap but not free per call).
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


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
# Scoring: churn model + risk-independent economics.
# Depends only on the data + model, so it can be computed once and cached.
# --------------------------------------------------
def score_customers() -> pd.DataFrame:
    df = build_feature_table()

    probs = _get_model().predict_proba(df[FEATURES])[:, 1]
    df["churn_probability"] = probs

    # Risk bands (vectorized)
    df["risk_band"] = np.select(
        [probs >= 0.60, probs >= 0.30], ["HIGH", "MEDIUM"], default="LOW"
    )

    # Diagnostic expected loss (vectorized)
    df["loss_if_act"] = (1 - probs) * df["retention_cost"]
    df["loss_if_ignore"] = probs * df["CLV"]

    return df


# --------------------------------------------------
# Decision explainability helpers (run on the small ACT subset only)
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

    if row["Contract"] == "Month-to-month":
        reasons.append("No contract commitment")

    if row["tenure"] <= 6:
        reasons.append("New customer")

    return " + ".join(reasons[:3])


def recommend_action(row, clv_high):
    if row["risk_band"] == "HIGH" and row["CLV"] >= clv_high:
        return "Personal retention call + premium discount"
    elif row["risk_band"] == "HIGH":
        return "Targeted discount offer"
    elif row["risk_band"] == "MEDIUM":
        return "Re-engagement email campaign"
    else:
        return "Standard follow-up"


# --------------------------------------------------
# Tier 1: Decision engine — cheap given a scored table.
# --------------------------------------------------
def decide(
    scored_df: pd.DataFrame,
    budget: float,
    max_customers: int,
    strategy: str,
    save_rate: float = SAVE_RATE,
):
    df = scored_df.copy()

    high_risk = df[df["risk_band"] == "HIGH"]
    loss_comparison = (
        high_risk["loss_if_ignore"] > high_risk["loss_if_act"]
    ).mean()

    # Economic scoring + strategy prioritization (vectorized)
    df = build_retention_scores(df, save_rate)
    strategy_df = apply_decision_strategy(df, strategy)

    # Budget + capacity constrained selection
    selected_df, spent_budget = select_customers_under_budget(
        strategy_df, total_budget=budget, max_customers=max_customers
    )

    # Final action segmentation (vectorized inside)
    final_df = assign_action_segments(df, selected_df)
    final_df["efficiency"] = (
        final_df["net_retention_value"] / final_df["retention_cost"]
    )

    # Explanations only for the ACT rows that get displayed.
    final_df["decision_reason"] = ""
    final_df["recommended_action"] = ""
    act_mask = final_df["action_segment"] == "ACT"
    if act_mask.any():
        clv_median = final_df["CLV"].median()
        cost_median = final_df["retention_cost"].median()
        clv_high = final_df["CLV"].quantile(0.75)
        act_rows = final_df[act_mask]
        # Real per-customer attributions from the model (SHAP); fall back to the
        # rule-based reason if SHAP is unavailable.
        try:
            from src.models.explain import shap_reasons
            final_df.loc[act_mask, "decision_reason"] = shap_reasons(
                _get_model(), act_rows[FEATURES]
            )
        except Exception:
            final_df.loc[act_mask, "decision_reason"] = act_rows.apply(
                generate_decision_reason, axis=1, args=(clv_median, cost_median)
            )
        final_df.loc[act_mask, "recommended_action"] = act_rows.apply(
            recommend_action, axis=1, args=(clv_high,)
        )

    return final_df, selected_df, spent_budget, loss_comparison


def load_and_compute_decisions(
    budget: float,
    max_customers: int,
    strategy: str,
    save_rate: float = SAVE_RATE,
):
    """Score all customers and produce the constrained decision table."""
    return decide(score_customers(), budget, max_customers, strategy, save_rate)
