import pandas as pd
import streamlit as st

from app.core import load_and_compute_decisions

# --------------------------------------------------
# Tier 2: Strategy comparison (CACHED)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_strategy_comparison(
    budget: float,
    max_customers: int
):
    strategies = ["Conservative", "Balanced", "Aggressive"]
    results = []

    for strat in strategies:
        final_df, _, _, _ = load_and_compute_decisions(
            budget=budget,
            max_customers=max_customers,
            strategy=strat
        )

        act_df = final_df[final_df["action_segment"] == "ACT"]

        results.append({
            "Strategy": strat,
            "ACT Customers": len(act_df),
            "% HIGH Risk": (act_df["risk_band"] == "HIGH").mean() * 100,
            "% MEDIUM Risk": (act_df["risk_band"] == "MEDIUM").mean() * 100,
            "Revenue Saved (₹)": act_df["net_retention_value"].sum(),
            "Budget Used (₹)": act_df["retention_cost"].sum(),
            "ROI": (
                act_df["net_retention_value"].sum()
                / max(act_df["retention_cost"].sum(), 1)
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Tier 3: Counterfactual strategy impact (CACHED)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_counterfactual_impact(
    budget: float,
    max_customers: int
):
    strategies = ["Conservative", "Balanced", "Aggressive"]
    act_sets = {}

    for strat in strategies:
        final_df, _, _, _ = load_and_compute_decisions(
            budget=budget,
            max_customers=max_customers,
            strategy=strat
        )

        act_sets[strat] = set(
            final_df.loc[
                final_df["action_segment"] == "ACT",
                "customer_id"
            ]
        )

    conservative = act_sets["Conservative"]
    balanced = act_sets["Balanced"]
    aggressive = act_sets["Aggressive"]

    return {
        "Aggressive_only": aggressive - balanced,
        "Balanced_only": balanced - conservative,
        "Dropped_in_Conservative": aggressive - conservative
    }


# --------------------------------------------------
# Tier 3: Decision Boundary Zone (CACHED)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_decision_boundary_zone(
    budget: float,
    max_customers: int
):
    strategies = ["Conservative", "Balanced", "Aggressive"]
    act_sets = {}

    for strat in strategies:
        final_df, _, _, _ = load_and_compute_decisions(
            budget=budget,
            max_customers=max_customers,
            strategy=strat
        )

        act_sets[strat] = set(
            final_df.loc[
                final_df["action_segment"] == "ACT",
                "customer_id"
            ]
        )

    union_act = set.union(*act_sets.values())
    intersection_act = set.intersection(*act_sets.values())
    dbz_ids = union_act - intersection_act

    balanced_df, _, _, _ = load_and_compute_decisions(
        budget=budget,
        max_customers=max_customers,
        strategy="Balanced"
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
# Tier 3: Decision Stability (CACHED)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_decision_stability(
    budget: float,
    max_customers: int,
    strategy: str
):
    """
    Measures how stable ACT decisions are under
    small budget and capacity perturbations.
    """

    base_df, _, _, _ = load_and_compute_decisions(
        budget=budget,
        max_customers=max_customers,
        strategy=strategy
    )

    base_act = set(
        base_df[base_df["action_segment"] == "ACT"]["customer_id"]
    )

    if len(base_act) == 0:
        return {
            "budget_stability": 0.0,
            "capacity_stability": 0.0,
            "note": "No ACT customers under current constraints"
        }

    # Budget -10%
    budget_df, _, _, _ = load_and_compute_decisions(
        budget=budget * 0.9,
        max_customers=max_customers,
        strategy=strategy
    )

    budget_act = set(
        budget_df[budget_df["action_segment"] == "ACT"]["customer_id"]
    )

    # Capacity -10%
    capacity_df, _, _, _ = load_and_compute_decisions(
        budget=budget,
        max_customers=int(max_customers * 0.9),
        strategy=strategy
    )

    capacity_act = set(
        capacity_df[capacity_df["action_segment"] == "ACT"]["customer_id"]
    )

    return {
        "budget_stability": len(base_act & budget_act) / len(base_act),
        "capacity_stability": len(base_act & capacity_act) / len(base_act)
    }


# --------------------------------------------------
# Tier 3: Stability Attribution (CACHED)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_stability_attribution(
    budget: float,
    max_customers: int,
    strategy: str
):
    """
    Explains why some customers drop out when constraints tighten.
    """

    base_df, _, _, _ = load_and_compute_decisions(
        budget=budget,
        max_customers=max_customers,
        strategy=strategy
    )

    reduced_df, _, _, _ = load_and_compute_decisions(
        budget=budget * 0.9,
        max_customers=max_customers,
        strategy=strategy
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
