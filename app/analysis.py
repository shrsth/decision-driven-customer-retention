import pandas as pd
import streamlit as st

from app.core import decide, score_customers
from src.config import SAVE_RATE
from src.decision.retention_strategy import compare_policies, save_rate_sensitivity

STRATEGIES = ["Conservative", "Balanced", "Aggressive"]


# --------------------------------------------------
# The expensive part (SQL read + model prediction) runs ONCE per session
# and is reused by every decision computation below.
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_scored_customers():
    return score_customers()


# --------------------------------------------------
# One cached decision run per (constraints, strategy, save_rate).
# Every Tier 1/2/3 view reuses these — the decide() step is cheap.
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_decision_result(
    budget: float,
    max_customers: int,
    strategy: str,
    save_rate: float = SAVE_RATE
):
    return decide(get_scored_customers(), budget, max_customers, strategy, save_rate)


def compute_decisions_cached(
    budget: float,
    max_customers: int,
    strategy: str,
    save_rate: float = SAVE_RATE
):
    final_df, _, _, _ = compute_decision_result(
        budget, max_customers, strategy, save_rate
    )
    return final_df


def compute_roi_sensitivity(budget, max_customers, strategy, save_rate=SAVE_RATE):
    """ROI across a sweep of save rates — the business case's sensitivity."""
    rates = [round(0.1 * i, 2) for i in range(1, 11)]
    rois = []
    for rate in rates:
        df = compute_decisions_cached(budget, max_customers, strategy, rate)
        act = df[df["action_segment"] == "ACT"]
        rois.append(
            act["net_retention_value"].sum() / max(act["retention_cost"].sum(), 1)
        )
    return rates, rois


@st.cache_data(show_spinner=False)
def compute_policy_comparison(budget, max_customers, save_rate=SAVE_RATE):
    """Economic engine vs. naive targeting baselines (same budget)."""
    return compare_policies(
        get_scored_customers(), budget, max_customers, save_rate
    )


def compute_sensitivity(budget, max_customers, strategy, save_rate=SAVE_RATE):
    """Break-even and net-value curve for the current ACT set."""
    final_df = compute_decisions_cached(budget, max_customers, strategy, save_rate)
    act = final_df[final_df["action_segment"] == "ACT"]
    return save_rate_sensitivity(act, save_rate)


def _act_sets(budget, max_customers, save_rate):
    return {
        strat: set(
            compute_decisions_cached(budget, max_customers, strat, save_rate)
            .query("action_segment == 'ACT'")["customer_id"]
        )
        for strat in STRATEGIES
    }


# --------------------------------------------------
# Tier 2: Strategy comparison
# --------------------------------------------------
def compute_strategy_comparison(
    budget: float,
    max_customers: int,
    save_rate: float = SAVE_RATE
):
    results = []

    for strat in STRATEGIES:
        final_df = compute_decisions_cached(
            budget, max_customers, strat, save_rate
        )

        act_df = final_df[final_df["action_segment"] == "ACT"]

        results.append({
            "Strategy": strat,
            "ACT Customers": len(act_df),
            "% HIGH Risk": (act_df["risk_band"] == "HIGH").mean() * 100,
            "% MEDIUM Risk": (act_df["risk_band"] == "MEDIUM").mean() * 100,
            "Revenue Saved ($)": act_df["net_retention_value"].sum(),
            "Budget Used ($)": act_df["retention_cost"].sum(),
            "ROI": (
                act_df["net_retention_value"].sum()
                / max(act_df["retention_cost"].sum(), 1)
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Tier 3: Counterfactual strategy impact
# --------------------------------------------------
def compute_counterfactual_impact(
    budget: float,
    max_customers: int,
    save_rate: float = SAVE_RATE
):
    act_sets = _act_sets(budget, max_customers, save_rate)

    conservative = act_sets["Conservative"]
    balanced = act_sets["Balanced"]
    aggressive = act_sets["Aggressive"]

    return {
        "Aggressive_only": aggressive - balanced,
        "Balanced_only": balanced - conservative,
        "Dropped_in_Conservative": aggressive - conservative
    }


# --------------------------------------------------
# Tier 3: Decision Boundary Zone
# --------------------------------------------------
def compute_decision_boundary_zone(
    budget: float,
    max_customers: int,
    save_rate: float = SAVE_RATE
):
    act_sets = _act_sets(budget, max_customers, save_rate)

    union_act = set.union(*act_sets.values())
    intersection_act = set.intersection(*act_sets.values())
    dbz_ids = union_act - intersection_act

    balanced_df = compute_decisions_cached(
        budget, max_customers, "Balanced", save_rate
    )

    dbz_df = balanced_df[
        balanced_df["customer_id"].isin(dbz_ids)
    ].copy()

    return {
        "dbz_count": len(dbz_ids),
        "conservative_only": len(act_sets["Conservative"] - act_sets["Balanced"]),
        "balanced_only": len(act_sets["Balanced"] - act_sets["Conservative"]),
        "aggressive_only": len(act_sets["Aggressive"] - act_sets["Balanced"]),
        "dbz_df": dbz_df.sort_values(
            "net_retention_value",
            ascending=False
        )
    }


# --------------------------------------------------
# Tier 3: Decision Stability
# --------------------------------------------------
def compute_decision_stability(
    budget: float,
    max_customers: int,
    strategy: str,
    save_rate: float = SAVE_RATE
):
    """
    Measures how stable ACT decisions are under
    small budget and capacity perturbations.
    """

    def act_ids(b, m):
        df = compute_decisions_cached(b, m, strategy, save_rate)
        return set(df[df["action_segment"] == "ACT"]["customer_id"])

    base_act = act_ids(budget, max_customers)

    if len(base_act) == 0:
        return {
            "budget_stability": 0.0,
            "capacity_stability": 0.0,
            "note": "No ACT customers under current constraints"
        }

    budget_act = act_ids(budget * 0.9, max_customers)
    capacity_act = act_ids(budget, int(max_customers * 0.9))

    return {
        "budget_stability": len(base_act & budget_act) / len(base_act),
        "capacity_stability": len(base_act & capacity_act) / len(base_act)
    }


# --------------------------------------------------
# Tier 3: Stability Attribution
# --------------------------------------------------
def compute_stability_attribution(
    budget: float,
    max_customers: int,
    strategy: str,
    save_rate: float = SAVE_RATE
):
    """
    Explains why some customers drop out when constraints tighten.
    """

    base_df = compute_decisions_cached(
        budget, max_customers, strategy, save_rate
    )
    reduced_df = compute_decisions_cached(
        budget * 0.9, max_customers, strategy, save_rate
    )

    base_act = base_df[base_df["action_segment"] == "ACT"]
    reduced_act = reduced_df[reduced_df["action_segment"] == "ACT"]

    dropped_ids = set(base_act["customer_id"]) - set(reduced_act["customer_id"])
    dropped = base_act[base_act["customer_id"].isin(dropped_ids)]
    retained = reduced_act

    if dropped.empty:
        return None

    return {
        "Dropped Avg Churn": dropped["churn_probability"].mean(),
        "Retained Avg Churn": retained["churn_probability"].mean(),
        "Dropped Avg CLV": dropped["CLV"].mean(),
        "Retained Avg CLV": retained["CLV"].mean(),
        "Dropped Avg Efficiency": dropped["efficiency"].mean(),
        "Retained Avg Efficiency": retained["efficiency"].mean(),
    }
