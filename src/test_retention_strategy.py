import pandas as pd
from src.models.train_logistic import pipeline, FEATURES
from src.features.feature_builder import build_feature_table
from src.decision.retention_strategy import build_retention_scores

# Load data
customers = pd.read_csv("data/raw/customers.csv")
behavior = pd.read_csv("data/raw/behavior.csv")

# Build features
df = build_feature_table(customers, behavior)

# Prepare X
X = df[FEATURES]

# Predict churn probabilities
df["churn_probability"] = pipeline.predict_proba(X)[:, 1]

# Apply decision logic
decision_df = build_retention_scores(df)

print(decision_df[
    [
        "customer_id",
        "churn_probability",
        "CLV",
        "retention_cost",
        "revenue_at_risk",
        "net_retention_value",
        "retention_priority_score"
    ]
].sort_values("retention_priority_score", ascending=False).head(10))
