import pytest

from src.decision.retention_strategy import (
    apply_decision_strategy,
    build_retention_scores,
    compare_policies,
    save_rate_sensitivity,
    select_customers_under_budget,
)


def _prepared(scored_df, strategy="balanced"):
    df = build_retention_scores(scored_df)
    return apply_decision_strategy(df, strategy)


def test_budget_respected(scored_df):
    df = _prepared(scored_df)
    selected, spent = select_customers_under_budget(df, total_budget=500)
    assert spent <= 500
    assert selected["retention_cost"].sum() == pytest.approx(spent)


def test_capacity_respected(scored_df):
    df = _prepared(scored_df)
    selected, _ = select_customers_under_budget(
        df, total_budget=1_000_000, max_customers=5
    )
    assert len(selected) <= 5


def test_only_positive_net_value_selected(scored_df):
    df = _prepared(scored_df)
    selected, _ = select_customers_under_budget(df, total_budget=1_000_000)
    assert (selected["net_retention_value"] > 0).all()


def test_strategy_weight_ordering(scored_df):
    df = build_retention_scores(scored_df)
    weights = {}
    for strategy in ["conservative", "balanced", "aggressive"]:
        out = apply_decision_strategy(df, strategy)
        weights[strategy] = out.set_index("customer_id")["strategy_weight"]

    medium_low = df[df["risk_band"].isin(["MEDIUM", "LOW"])]["customer_id"]
    assert (
        weights["aggressive"][medium_low] >= weights["balanced"][medium_low]
    ).all()
    assert (
        weights["balanced"][medium_low] >= weights["conservative"][medium_low]
    ).all()


def test_unknown_strategy_raises(scored_df):
    df = build_retention_scores(scored_df)
    with pytest.raises(ValueError):
        apply_decision_strategy(df, "reckless")


def test_budget_monotonicity(scored_df):
    df = _prepared(scored_df)
    small, _ = select_customers_under_budget(df, total_budget=500)
    large, _ = select_customers_under_budget(df, total_budget=1000)
    assert len(large) >= len(small)


def test_save_rate_discounts_net_value(scored_df):
    """An intervention only works save_rate of the time, so net value
    must shrink as save_rate falls."""
    full = build_retention_scores(scored_df, save_rate=1.0)
    partial = build_retention_scores(scored_df, save_rate=0.3)
    assert (partial["net_retention_value"] < full["net_retention_value"]).all()
    expected = 0.3 * partial["revenue_at_risk"] - partial["retention_cost"]
    assert partial["net_retention_value"].equals(expected)


def test_lower_save_rate_selects_fewer_or_equal(scored_df):
    counts = {}
    for rate in [1.0, 0.3]:
        df = apply_decision_strategy(
            build_retention_scores(scored_df, save_rate=rate), "balanced"
        )
        selected, _ = select_customers_under_budget(df, total_budget=1_000_000)
        counts[rate] = len(selected)
    assert counts[0.3] <= counts[1.0]


def test_compare_policies_engine_wins_and_respects_budget(scored_df):
    budget = 400
    result = compare_policies(scored_df, budget=budget, max_customers=50, save_rate=0.5)

    assert set(result["policy"]) == {
        "Random", "Target highest churn", "Target highest CLV", "Decision engine",
    }
    # every policy stays within budget
    assert (result["budget_used"] <= budget + 1e-6).all()
    # the engine captures at least as much expected value as any naive rule
    engine_value = result.loc[
        result["policy"] == "Decision engine", "expected_value"].iloc[0]
    naive_max = result.loc[
        result["policy"] != "Decision engine", "expected_value"].max()
    assert engine_value >= naive_max


def test_save_rate_sensitivity_break_even(scored_df):
    df = build_retention_scores(scored_df, save_rate=0.5)
    act = df[df["net_retention_value"] > 0]
    sens = save_rate_sensitivity(act, assumed_save_rate=0.5)

    # net value is exactly zero at the break-even rate
    be = sens["break_even"]
    assert be * sens["revenue_at_risk"] - sens["cost"] == pytest.approx(0, abs=1e-6)
    # net value rises monotonically with the true save rate
    assert sens["net_values"] == sorted(sens["net_values"])
