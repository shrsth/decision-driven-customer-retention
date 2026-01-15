import pandas as pd
from src.features.feature_builder import build_feature_table
from src.models.train_logistic import pipeline, FEATURES
from src.decision.retention_strategy import (
    build_retention_scores,
    select_customers_under_budget
)

# Load data
customers = pd.read_csv("data/raw/customers.csv")
behavior = pd.read_csv("data/raw/behavior.csv")

# Build features
df = build_feature_table(customers, behavior)

# Predict churn probabilities
X = df[FEATURES]
df["churn_probability"] = pipeline.predict_proba(X)[:, 1]

# Decision scores
df = build_retention_scores(df)

# Apply budget constraint
BUDGET = 500_000
MAX_CUSTOMERS = 300

selected_df, spent = select_customers_under_budget(
    df,
    total_budget=BUDGET,
    max_customers=MAX_CUSTOMERS
)

print(f"Selected customers: {len(selected_df)}")
print(f"Total budget spent: {spent:,.2f}")

print("\nTop 10 selected customers:")
print(selected_df[
    [
        "customer_id",
        "churn_probability",
        "CLV",
        "retention_cost",
        "net_retention_value",
        "efficiency"
    ]
].head(10))


from src.decision.retention_strategy import assign_action_segments

# Assign action segments
final_df = assign_action_segments(df, selected_df)

print("\nAction Segment Distribution:")
print(final_df["action_segment"].value_counts())

print("\nSample ACT customers:")
print(
    final_df[final_df["action_segment"] == "ACT"]
    [
        [
            "customer_id",
            "churn_probability",
            "CLV",
            "retention_cost",
            "net_retention_value",
            "action_segment"
        ]
    ]
    .head(10)
)


scenarios = [
    {"name": "Base", "budget": 500_000, "max_customers": 300},
    {"name": "Tight Budget", "budget": 100_000, "max_customers": 300},
    {"name": "High Capacity", "budget": 500_000, "max_customers": 600},
]

print("\n=== WHAT-IF ANALYSIS ===")

for scenario in scenarios:
    selected, spent = select_customers_under_budget(
        df,
        total_budget=scenario["budget"],
        max_customers=scenario["max_customers"]
    )

    print(f"\nScenario: {scenario['name']}")
    print(f"Selected customers: {len(selected)}")
    print(f"Budget spent: {spent:,.2f}")
    print(f"Total revenue saved: {selected['net_retention_value'].sum():,.2f}")
