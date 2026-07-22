import numpy as np
import pandas as pd

from src.config import SAVE_RATE

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


def compute_net_retention_value(df, save_rate=SAVE_RATE):
    """Expected value of intervening: an intervention only saves the
    customer with probability `save_rate`, so at-risk revenue is
    discounted accordingly before subtracting the cost."""
    df = df.copy()
    df["net_retention_value"] = (
        save_rate * df["revenue_at_risk"] - df["retention_cost"]
    )
    return df


def build_retention_scores(df, save_rate=SAVE_RATE):
    df = compute_revenue_at_risk(df)
    df = compute_net_retention_value(df, save_rate)

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

    # Greedy pack over numpy scalars (same skip-and-continue semantics as an
    # iterrows loop, but ~10x faster — no per-row pandas Series overhead).
    costs = df["retention_cost"].to_numpy()
    cap = max_customers if max_customers else len(costs)
    selected_pos = []
    spent = 0.0

    for i in range(len(costs)):
        cost = costs[i]
        if spent + cost > total_budget:
            continue
        selected_pos.append(i)
        spent += cost
        if len(selected_pos) >= cap:
            break

    selected_df = df.iloc[selected_pos]
    return selected_df, spent


# --------------------------------------------------
# Final action segmentation
# --------------------------------------------------
def assign_action_segments(full_df, selected_df):
    df = full_df.copy()
    selected_ids = set(selected_df["customer_id"])

    df["action_segment"] = np.where(
        df["customer_id"].isin(selected_ids), "ACT",
        np.where(df["net_retention_value"] > 0, "MONITOR", "IGNORE"),
    )
    return df


# --------------------------------------------------
# Baseline policy comparison — does the economic engine beat naive targeting?
# --------------------------------------------------
def _greedy_pack(ordered_df, budget, max_customers):
    """Fill the budget in the given order (skip-and-continue greedy)."""
    costs = ordered_df["retention_cost"].to_numpy()
    cap = max_customers if max_customers else len(costs)
    selected, spent = [], 0.0
    for i in range(len(costs)):
        if spent + costs[i] > budget:
            continue
        selected.append(i)
        spent += costs[i]
        if len(selected) >= cap:
            break
    return ordered_df.iloc[selected]


def compare_policies(
    scored_df, budget, max_customers, save_rate=SAVE_RATE, random_state=42
):
    """Compare the economic decision engine against naive targeting policies
    under the same budget and capacity.

    Naive policies spend on whoever ranks top by their single criterion, even
    if the intervention loses money; the engine only spends where expected
    value is positive and ranks by that value. Value captured per policy is
    the sum of net_retention_value (save_rate * p * CLV - cost) over the
    customers it funds.
    """
    df = build_retention_scores(scored_df, save_rate)
    df = df[df["retention_cost"] > 0].copy()
    rng = np.random.default_rng(random_state)
    df["_rand"] = rng.random(len(df))

    engine = df[df["net_retention_value"] > 0]
    policies = {
        "Random": df.sort_values("_rand"),
        "Target highest churn": df.sort_values("churn_probability", ascending=False),
        "Target highest CLV": df.sort_values("CLV", ascending=False),
        "Decision engine": engine.sort_values("net_retention_value", ascending=False),
    }

    rows = []
    for name, ordered in policies.items():
        chosen = _greedy_pack(ordered, budget, max_customers)
        cost = float(chosen["retention_cost"].sum())
        value = float(chosen["net_retention_value"].sum())
        rows.append({
            "policy": name,
            "customers": len(chosen),
            "budget_used": cost,
            "expected_value": value,
            "roi": value / cost if cost else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# Assumption sensitivity — how fragile is the business case?
# --------------------------------------------------
def save_rate_sensitivity(act_df, assumed_save_rate=SAVE_RATE):
    """How the funded action set's economics respond to the true save rate.

    The engine commits to an ACT set assuming `assumed_save_rate`. If offers
    actually succeed at rate r, realized net value is
    r * sum(p * CLV) - sum(cost). The break-even rate is the r at which that
    hits zero — below it, the whole program loses money.
    """
    revenue_at_risk = float((act_df["churn_probability"] * act_df["CLV"]).sum())
    cost = float(act_df["retention_cost"].sum())
    break_even = cost / revenue_at_risk if revenue_at_risk else float("nan")

    rates = [round(0.05 * i, 2) for i in range(0, 21)]  # 0.00 .. 1.00
    net_values = [r * revenue_at_risk - cost for r in rates]

    return {
        "break_even": break_even,
        "assumed": assumed_save_rate,
        "rates": rates,
        "net_values": net_values,
        "revenue_at_risk": revenue_at_risk,
        "cost": cost,
    }
