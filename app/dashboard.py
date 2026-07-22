import json
import sys
from pathlib import Path

# Make the repo root importable. Streamlit Cloud launches this as
# `streamlit run app/dashboard.py`, which puts only app/ on sys.path — not the
# project root — so the absolute `app.*` / `src.*` imports below would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.analysis import (  # noqa: E402
    compute_decision_result,
    compute_strategy_comparison,
    compute_decision_stability,
    compute_stability_attribution,
    compute_decision_boundary_zone,
    compute_roi_sensitivity,
    compute_policy_comparison,
    compute_sensitivity,
)
from app import charts
from app.hero import render_hero
from app.ferrofluid import render_background
from app.ui import inject_css, section_header, sidebar_title
from src.config import DB_PATH, METRICS_PATH, MODEL_PATH


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Decision-Driven Customer Retention",
    page_icon="◆",
    layout="wide",
)
inject_css()
render_background(colors=("#3987e5", "#4f7ff0", "#7b6cf6"),
                  opacity=0.9, glow=1.5, speed=0.26)


@st.cache_resource(show_spinner="First run: downloading data and training model…")
def ensure_pipeline_ran():
    """Bootstrap the data artifacts on a fresh deploy (e.g. Streamlit Cloud),
    where data/ is gitignored so the model, DB and metrics don't exist yet."""
    if not (MODEL_PATH.exists() and DB_PATH.exists() and METRICS_PATH.exists()):
        from src.pipeline import run_pipeline
        run_pipeline()
    return True


ensure_pipeline_ran()


@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return None


# --------------------------------------------------
# Hero title card (MagicBento-style interactive glow)
# --------------------------------------------------
render_hero()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
sidebar_title("Decision Controls")

budget = st.sidebar.slider(
    "Retention Budget ($)",
    min_value=1_000, max_value=100_000, step=1_000, value=25_000,
)
max_customers = st.sidebar.slider(
    "Max Customers to Retain",
    min_value=25, max_value=1_000, step=25, value=300,
)
strategy = st.sidebar.selectbox(
    "Decision Strategy",
    options=["Conservative", "Balanced", "Aggressive"], index=1,
)
save_rate = st.sidebar.slider(
    "Intervention Save Rate",
    min_value=0.10, max_value=1.00, step=0.05, value=0.30,
    help=(
        "Probability that a retention offer actually saves an at-risk "
        "customer. Industry acceptance rates are typically 20-40%."
    ),
)

st.sidebar.caption(
    "Churn probability is a signal, not a decision. This engine converts it "
    "into budget- and capacity-constrained ACT / MONITOR / IGNORE actions."
)

# --------------------------------------------------
# Compute decisions
# --------------------------------------------------
with st.spinner("Computing optimal retention strategy..."):
    final_df, selected_df, spent, loss_comparison = compute_decision_result(
        budget, max_customers, strategy, save_rate
    )

act_df = final_df[final_df["action_segment"] == "ACT"]
act_count = len(act_df)
monitor_count = int((final_df["action_segment"] == "MONITOR").sum())
ignore_count = int((final_df["action_segment"] == "IGNORE").sum())
revenue_saved = act_df["net_retention_value"].sum()
budget_used = act_df["retention_cost"].sum()
roi = revenue_saved / max(budget_used, 1)

tab_decisions, tab_robustness, tab_model = st.tabs(
    ["Decisions", "Strategy & Robustness", "Model Performance"]
)

# ==================================================
# TAB 1 — Decisions
# ==================================================
with tab_decisions:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Act on", f"{act_count:,}")
    c2.metric("Expected Revenue Saved", f"${revenue_saved:,.0f}")
    c3.metric("Budget Used", f"${budget_used:,.0f}")
    c4.metric("Retention ROI", f"{roi:.1f}x")

    left, right = st.columns([1, 1])
    with left:
        section_header(
            "Action segmentation",
            f"Under a ${budget:,.0f} budget and {strategy} strategy, "
            f"{act_count:,} customers are worth acting on; "
            f"{monitor_count:,} are positive-value but unfunded.",
            icon="target",
        )
        st.plotly_chart(
            charts.action_segment_chart(act_count, monitor_count, ignore_count),
            width="stretch", config={"displayModeBar": False},
        )
    with right:
        section_header(
            "ROI vs. intervention success",
            "The business case depends on interventions working — ROI falls "
            "toward break-even as the assumed save rate drops.",
            icon="trending-up",
        )
        rates, rois = compute_roi_sensitivity(
            budget, max_customers, strategy, save_rate
        )
        st.plotly_chart(
            charts.roi_sensitivity_chart(rates, rois, current_rate=save_rate),
            width="stretch", config={"displayModeBar": False},
        )

    section_header(
        "ACT customers",
        "Ranked by net retention value — the immediate-action list.",
        icon="users",
    )
    display_cols = [
        "customer_id", "risk_band", "churn_probability", "CLV",
        "retention_cost", "net_retention_value", "efficiency",
        "decision_reason", "recommended_action",
    ]
    table_df = act_df[display_cols].sort_values(
        "net_retention_value", ascending=False
    )
    st.dataframe(
        table_df,
        width="stretch", hide_index=True,
        column_config={
            "customer_id": st.column_config.TextColumn("Customer"),
            "risk_band": st.column_config.TextColumn("Risk"),
            "churn_probability": st.column_config.NumberColumn(
                "Churn Prob", format="%.2f",
                help="Model-predicted churn probability (0-1)",
            ),
            "CLV": st.column_config.NumberColumn("CLV", format="$%.0f"),
            "retention_cost": st.column_config.NumberColumn(
                "Cost", format="$%.0f"),
            "net_retention_value": st.column_config.NumberColumn(
                "Net Value", format="$%.0f"),
            "efficiency": st.column_config.NumberColumn(
                "Efficiency", format="%.2f"),
            "decision_reason": st.column_config.TextColumn("Reason", width="large"),
            "recommended_action": st.column_config.TextColumn(
                "Recommended Action", width="large"),
        },
    )
    st.download_button(
        "Download ACT customers (CSV)",
        data=table_df.to_csv(index=False),
        file_name="act_customers.csv", mime="text/csv",
    )

# ==================================================
# TAB 2 — Strategy & Robustness
# ==================================================
with tab_robustness:
    section_header(
        "Strategy comparison",
        "Same budget and capacity, three business postures.",
        icon="git-compare",
    )
    comparison_df = compute_strategy_comparison(
        budget=budget, max_customers=max_customers, save_rate=save_rate
    )
    comparison_df["Revenue Saved ($)"] = comparison_df["Revenue Saved ($)"].round(0)
    comparison_df["Budget Used ($)"] = comparison_df["Budget Used ($)"].round(0)
    comparison_df["ROI"] = comparison_df["ROI"].round(2)
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    # --- Baseline policy comparison ---
    section_header(
        "Does the decision engine beat naive targeting?",
        "Same budget, four policies. Naive rules spend on whoever ranks top by "
        "one criterion; the engine spends only where expected value is positive. "
        "Value captured = save_rate x churn prob x CLV - cost, summed over funded "
        "customers.",
        icon="target",
    )
    policy_df = compute_policy_comparison(budget, max_customers, save_rate)
    best = policy_df.loc[policy_df["expected_value"].idxmax()]
    naive_best = policy_df[policy_df["policy"] != "Decision engine"]["expected_value"].max()
    lift = (best["expected_value"] - naive_best) / max(naive_best, 1) * 100
    st.plotly_chart(
        charts.policy_comparison_chart(policy_df.to_dict(orient="records")),
        width="stretch", config={"displayModeBar": False},
    )
    st.caption(
        f"The economic engine captures ${best['expected_value']:,.0f} — "
        f"{lift:.0f}% more than the best naive rule. Targeting highest-CLV "
        "customers is nearly worthless here: high value but low churn risk means "
        "little revenue is actually at risk."
    )

    # --- Assumption sensitivity ---
    section_header(
        "How fragile is the business case?",
        "The ACT set is chosen assuming a save rate. This shows the realized net "
        "value if interventions actually succeed at a different rate — and the "
        "break-even point where the program stops paying for itself.",
        icon="gauge",
    )
    sens = compute_sensitivity(budget, max_customers, strategy, save_rate)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Break-even save rate", f"{sens['break_even']*100:.1f}%")
        st.caption(
            f"Assumed {sens['assumed']*100:.0f}%. The program profits as long as "
            f"offers work more than {sens['break_even']*100:.1f}% of the time — a "
            f"{(sens['assumed']-sens['break_even'])*100:.0f}-point margin of safety."
        )
    with c2:
        st.plotly_chart(
            charts.sensitivity_chart(
                sens["rates"], sens["net_values"],
                sens["break_even"], sens["assumed"],
            ),
            width="stretch", config={"displayModeBar": False},
        )

    section_header(
        "Decision stability",
        "How many ACT decisions survive a 10% cut to budget or capacity. High "
        "stability means the system is safe to deploy under changing conditions.",
        icon="shield",
    )
    stability = compute_decision_stability(
        budget=budget, max_customers=max_customers,
        strategy=strategy, save_rate=save_rate,
    )
    c1, c2 = st.columns(2)
    c1.metric("Budget Stability (-10%)", f"{stability['budget_stability']*100:.1f}%")
    c2.metric("Capacity Stability (-10%)", f"{stability['capacity_stability']*100:.1f}%")

    attr = compute_stability_attribution(
        budget=budget, max_customers=max_customers,
        strategy=strategy, save_rate=save_rate,
    )
    if attr:
        st.caption(
            "Customers dropped under tighter constraints are lower-value and "
            "lower-efficiency — rational prioritization, not thrashing."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Dropped Avg Efficiency", f"{attr['Dropped Avg Efficiency']:.2f}")
        c2.metric("Dropped Avg CLV", f"${attr['Dropped Avg CLV']:.0f}")
        c3.metric("Dropped Avg Churn", f"{attr['Dropped Avg Churn']:.2f}")

    section_header(
        "Decision boundary zone",
        "Customers whose ACT status flips between strategies — the marginal, "
        "contestable calls.",
        icon="shuffle",
    )
    dbz = compute_decision_boundary_zone(
        budget=budget, max_customers=max_customers, save_rate=save_rate
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Boundary Customers", dbz["dbz_count"])
    c2.metric("Aggressive-only", dbz["aggressive_only"])
    c3.metric("Balanced-only", dbz["balanced_only"])
    c4.metric("Dropped in Conservative", dbz["conservative_only"])

    with st.expander("View sample boundary customers"):
        if dbz["dbz_count"] > 0:
            st.dataframe(
                dbz["dbz_df"][[
                    "customer_id", "risk_band", "churn_probability",
                    "CLV", "net_retention_value",
                ]].head(10),
                width="stretch", hide_index=True,
            )
        else:
            st.info("No boundary customers under current settings.")

# ==================================================
# TAB 3 — Model Performance
# ==================================================
with tab_model:
    metrics = load_metrics()
    if metrics is None:
        st.warning("No metrics artifact found. Run `python -m src.pipeline` first.")
    else:
        scores = metrics["scores"]
        section_header(
            "Churn model — holdout performance",
            f"Logistic regression on {metrics['n_customers']:,} real IBM Telco "
            f"customers ({metrics['churn_rate']*100:.1f}% churn).",
            icon="activity",
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC AUC", f"{scores['roc_auc']:.3f}")
        c2.metric("Accuracy", f"{scores['accuracy']*100:.1f}%")
        c3.metric("F1 (churn)", f"{scores['f1_churn']:.3f}")
        c4.metric("Brier Score", f"{scores['brier_score']:.3f}",
                  help="Lower is better; measures probability calibration.")

        left, right = st.columns([1, 1])
        with left:
            section_header(
                "Calibration",
                "Predicted vs. actual churn rate by decile. Points on the "
                "diagonal mean the probabilities are honest — the reason "
                "logistic regression is used here.",
                icon="gauge",
            )
            st.plotly_chart(
                charts.calibration_chart(metrics["calibration"]),
                width="stretch", config={"displayModeBar": False},
            )
        with right:
            section_header(
                "Churn rate by segment",
                "Month-to-month fiber customers churn far more than customers "
                "on two-year contracts.",
                icon="bar-chart",
            )
            st.plotly_chart(
                charts.segment_churn_chart(metrics["segments"]),
                width="stretch", config={"displayModeBar": False},
            )

        importance = metrics.get("feature_importance")
        if importance:
            section_header(
                "What drives churn",
                "Standardized logistic-regression coefficients. Red raises churn "
                "odds, green protects. Short tenure, month-to-month contracts, "
                "and fiber-optic service are the dominant risk factors.",
                icon="activity",
            )
            st.plotly_chart(
                charts.feature_importance_chart(importance),
                width="stretch", config={"displayModeBar": False},
            )

        comparison = metrics.get("model_comparison")
        if comparison:
            section_header(
                "Model bake-off — why logistic regression",
                "5-fold cross-validated comparison against gradient boosting. "
                "LR is kept because it wins on calibration (lower Brier) — and "
                "here it also wins on AUC — while staying interpretable. The "
                "engine spends real budget on these probabilities, so "
                "calibration outranks raw accuracy.",
                icon="git-compare",
            )
            comp_df = pd.DataFrame(comparison).rename(columns={
                "model": "Model",
                "auc_mean": "ROC AUC",
                "auc_std": "AUC ±",
                "brier_mean": "Brier",
                "brier_std": "Brier ±",
                "accuracy_mean": "Accuracy",
                "f1_mean": "F1 (churn)",
            })
            st.dataframe(
                comp_df, width="stretch", hide_index=True,
                column_config={
                    "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
                    "AUC ±": st.column_config.NumberColumn(format="%.3f"),
                    "Brier": st.column_config.NumberColumn(
                        format="%.3f", help="Lower is better (calibration)"),
                    "Brier ±": st.column_config.NumberColumn(format="%.3f"),
                    "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "F1 (churn)": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(
                "± is the standard deviation across the 5 folds — small values "
                "mean the metrics are stable, not an artifact of one lucky split."
            )
