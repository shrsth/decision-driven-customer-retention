"""Optimal budget allocation via integer linear programming.

The production engine (`select_customers_under_budget`) uses a fast greedy
heuristic. This module solves the same problem *exactly* — a 0/1 knapsack:
pick the customer subset that maximizes total net retention value subject to a
budget and a headcount cap. It exists mainly as a benchmark: `optimality_gap`
measures how close the cheap greedy comes to the provable optimum, so the
greedy's use is an evidenced engineering trade-off rather than an assumption.
"""

import pulp

from src.config import SAVE_RATE
from src.decision.retention_strategy import _greedy_pack, build_retention_scores


def select_customers_optimal(df, total_budget, max_customers=None):
    """Exact 0/1 knapsack over customers with positive net value.

    Maximizes sum(net_retention_value) s.t. sum(retention_cost) <= budget and
    count <= max_customers. Returns (selected_df, spent) — same shape as
    `select_customers_under_budget`.
    """
    df = df.copy()
    df = df[(df["retention_cost"] > 0) & (df["net_retention_value"] > 0)]
    if df.empty:
        return df, 0.0

    df = df.reset_index(drop=True)
    values = df["net_retention_value"].to_numpy()
    costs = df["retention_cost"].to_numpy()
    n = len(df)

    prob = pulp.LpProblem("retention_knapsack", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]
    prob += pulp.lpSum(values[i] * x[i] for i in range(n))
    prob += pulp.lpSum(costs[i] * x[i] for i in range(n)) <= total_budget
    if max_customers:
        prob += pulp.lpSum(x) <= max_customers
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen = [i for i in range(n) if x[i].value() and x[i].value() > 0.5]
    selected_df = df.iloc[chosen]
    return selected_df, float(selected_df["retention_cost"].sum())


def optimality_gap(scored_df, budget, max_customers, save_rate=SAVE_RATE):
    """Greedy value vs the provable optimum under the same constraints.

    Greedy ranks by efficiency (value per dollar) — the strongest simple
    heuristic for a budget knapsack. `capture_pct` is how much of the optimum
    the greedy captures; expect it near 100% for this problem shape.
    """
    df = build_retention_scores(scored_df, save_rate)
    df = df[(df["retention_cost"] > 0) & (df["net_retention_value"] > 0)].copy()
    df["efficiency"] = df["net_retention_value"] / df["retention_cost"]

    greedy = _greedy_pack(
        df.sort_values("efficiency", ascending=False), budget, max_customers
    )
    optimal, _ = select_customers_optimal(df, budget, max_customers)

    greedy_value = float(greedy["net_retention_value"].sum())
    optimal_value = float(optimal["net_retention_value"].sum())

    return {
        "greedy_value": greedy_value,
        "optimal_value": optimal_value,
        "greedy_customers": int(len(greedy)),
        "optimal_customers": int(len(optimal)),
        "capture_pct": (greedy_value / optimal_value * 100.0)
        if optimal_value > 0 else 100.0,
    }
