from src.decision.optimizer import optimality_gap, select_customers_optimal
from src.decision.retention_strategy import build_retention_scores


def test_optimal_respects_budget_and_capacity(scored_df):
    df = build_retention_scores(scored_df, save_rate=0.5)
    selected, spent = select_customers_optimal(df, total_budget=400, max_customers=5)
    assert spent <= 400 + 1e-6
    assert len(selected) <= 5
    # only positive-value customers are ever chosen
    assert (selected["net_retention_value"] > 0).all()


def test_optimal_beats_or_matches_greedy(scored_df):
    gap = optimality_gap(scored_df, budget=400, max_customers=50, save_rate=0.5)
    # the ILP optimum can never be worse than the greedy heuristic
    assert gap["optimal_value"] >= gap["greedy_value"] - 1e-6
    assert 0 <= gap["capture_pct"] <= 100.0 + 1e-6


def test_optimal_finds_the_better_combination():
    """Classic knapsack trap: greedy-by-value grabs the big item; the optimum
    takes two smaller items worth more together."""
    import pandas as pd

    df = pd.DataFrame({
        "customer_id": ["big", "s1", "s2"],
        "retention_cost": [90.0, 50.0, 50.0],
        "net_retention_value": [100.0, 70.0, 70.0],
    })
    selected, spent = select_customers_optimal(df, total_budget=100)
    assert spent <= 100
    # optimum is the two small ones (140), not the single big one (100)
    assert set(selected["customer_id"]) == {"s1", "s2"}
    assert selected["net_retention_value"].sum() == 140.0
