import streamlit as st
import pandas as pd

from app.core import load_and_compute_decisions

from app.analysis import (
    compute_strategy_comparison,
    compute_decision_stability,
    compute_stability_attribution,
    compute_decision_boundary_zone,
    compute_counterfactual_impact
)


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Decision-Driven Customer Retention",
    layout="wide"
)

st.title("📊 Decision-Driven Customer Retention System")

st.markdown(
    """
    This system **does not just predict churn**.
    It decides **who to retain, why, and how**, under real-world constraints.
    """
)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("🔧 Decision Controls")

budget = st.sidebar.slider(
    "Retention Budget (₹)",
    min_value=5_000,
    max_value=500_000,
    step=5_000,
    value=50_000
)

max_customers = st.sidebar.slider(
    "Max Customers to Retain",
    min_value=25,
    max_value=600,
    step=25,
    value=300
)

strategy = st.sidebar.selectbox(
    "Decision Strategy",
    options=["Conservative", "Balanced", "Aggressive"],
    index=1
)

# --------------------------------------------------
# Compute decisions
# --------------------------------------------------
with st.spinner("Computing optimal retention strategy..."):
    final_df, selected_df, spent, loss_comparison = load_and_compute_decisions(
        budget=budget,
        max_customers=max_customers,
        strategy=strategy
    )

# --------------------------------------------------
# Metrics
# --------------------------------------------------
act_df = final_df[final_df["action_segment"] == "ACT"]

act_count = len(act_df)
monitor_count = (final_df["action_segment"] == "MONITOR").sum()
ignore_count = (final_df["action_segment"] == "IGNORE").sum()

revenue_saved = act_df["net_retention_value"].sum()
budget_used = act_df["retention_cost"].sum()
efficiency = revenue_saved / max(budget_used, 1)

# --------------------------------------------------
# --------------------------------------------------
# Decision summary
# --------------------------------------------------
st.divider()
st.markdown("## 📈 Decision Summary")

c1, c2, c3 = st.columns(3)
c1.metric("🟢 ACT Customers", act_count)
c2.metric("🟡 MONITOR Customers", monitor_count)
c3.metric("⚫ IGNORE Customers", ignore_count)

c4, c5, c6 = st.columns(3)
c4.metric("💰 Revenue Saved", f"₹{revenue_saved:,.0f}")
c5.metric("💸 Budget Used", f"₹{budget_used:,.0f}")
c6.metric("📈 Retention ROI", f"{efficiency:.1f}x")

# --------------------------------------------------
# Strategy Comparison (Tier 2 CORE)
# --------------------------------------------------
st.divider()
st.markdown("## 🧠 Strategy Comparison")

comparison_df = compute_strategy_comparison(
    budget=budget,
    max_customers=max_customers
)

comparison_df["Revenue Saved (₹)"] = comparison_df["Revenue Saved (₹)"].round(0)
comparison_df["Budget Used (₹)"] = comparison_df["Budget Used (₹)"].round(0)
comparison_df["ROI"] = comparison_df["ROI"].round(2)

st.dataframe(comparison_df, use_container_width=True)

# --------------------------------------------------
# Decision Stability Test (Tier 3A)
# --------------------------------------------------
st.divider()
st.markdown("## 🧱 Decision Stability Test")

st.caption(
    "Measures how robust ACT decisions are when "
    "budget or capacity changes slightly."
)

stability = compute_decision_stability(
    budget=budget,
    max_customers=max_customers,
    strategy=strategy
)

c1, c2 = st.columns(2)

c1.metric(
    "Budget Stability (-10%)",
    f"{stability['budget_stability']*100:.1f}%"
)

c2.metric(
    "Capacity Stability (-10%)",
    f"{stability['capacity_stability']*100:.1f}%"
)

st.caption(
    "High stability indicates the system makes consistent decisions "
    "and is safe to deploy in changing operational conditions."
)
# --------------------------------------------------
# Decision Stability Attribution (Tier 3B)
# --------------------------------------------------
st.divider()
st.markdown("## 🧠 Why Do Decisions Change?")

attr = compute_stability_attribution(
    budget=budget,
    max_customers=max_customers,
    strategy=strategy
)

if attr:
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dropped Avg Efficiency",
        f"{attr['Dropped Avg Efficiency']:.2f}"
    )
    c2.metric(
        "Dropped Avg CLV",
        f"₹{attr['Dropped Avg CLV']:.0f}"
    )
    c3.metric(
        "Dropped Avg Churn",
        f"{attr['Dropped Avg Churn']:.2f}"
    )

    st.caption(
        "Customers removed under tighter constraints are "
        "lower efficiency and lower value, indicating rational prioritization."
    )
else:
    st.info("No customers dropped under current stability test.")

# --------------------------------------------------
# Decision Boundary Zone (Tier 3)
# --------------------------------------------------
st.divider()
st.markdown("## 🔍 Decision Boundary Zone")

dbz = compute_decision_boundary_zone(
    budget=budget,
    max_customers=max_customers
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Boundary Customers", dbz["dbz_count"])
c2.metric("Aggressive-only", dbz["aggressive_only"])
c3.metric("Balanced-only", dbz["balanced_only"])
c4.metric("Dropped in Conservative", dbz["conservative_only"])

with st.expander("View sample boundary customers"):
    if dbz["dbz_count"] > 0:
        st.dataframe(
            dbz["dbz_df"][
                [
                    "customer_id",
                    "risk_band",
                    "churn_probability",
                    "CLV",
                    "net_retention_value"
                ]
            ].head(10),
            use_container_width=True
        )
    else:
        st.info("No boundary customers under current settings.")

# --------------------------------------------------
# Counterfactual Impact (Tier 3)
# --------------------------------------------------
st.divider()
st.markdown("## 🔁 Counterfactual Strategy Impact")

impact = compute_counterfactual_impact(
    budget=budget,
    max_customers=max_customers
)

c1, c2, c3 = st.columns(3)
c1.metric("Aggressive-only ACT", len(impact["Aggressive_only"]))
c2.metric("Balanced-only ACT", len(impact["Balanced_only"]))
c3.metric("Dropped in Conservative", len(impact["Dropped_in_Conservative"]))

with st.expander("View sample counterfactual customers"):
    if impact["Aggressive_only"]:
        sample_ids = list(impact["Aggressive_only"])[:10]
        st.dataframe(
            final_df[
                final_df["customer_id"].isin(sample_ids)
            ][
                ["customer_id", "risk_band", "CLV", "churn_probability", "net_retention_value"]
            ],
            use_container_width=True
        )
    else:
        st.info("No counterfactual customers under current settings.")

# --------------------------------------------------
# ACT Customers (Actionable Output)
# --------------------------------------------------
st.divider()
st.markdown("## 🧾 ACT Customers — Immediate Action Required")

display_cols = [
    "customer_id",
    "risk_band",
    "churn_probability",
    "CLV",
    "retention_cost",
    "net_retention_value",
    "efficiency",
    "decision_reason",
    "recommended_action"
]

table_df = act_df[display_cols].copy()
table_df["churn_probability"] = table_df["churn_probability"].round(3)
table_df["efficiency"] = table_df["efficiency"].round(2)

for col in ["CLV", "retention_cost", "net_retention_value"]:
    table_df[col] = table_df[col].round(0)

st.dataframe(
    table_df.sort_values("net_retention_value", ascending=False),
    use_container_width=True
)

st.download_button(
    "⬇️ Download ACT Customers (CSV)",
    data=table_df.to_csv(index=False),
    file_name="act_customers.csv",
    mime="text/csv"
)
