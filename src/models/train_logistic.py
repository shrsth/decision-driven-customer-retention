import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer


# ---------------------------
# Load data
# ---------------------------
customers = pd.read_csv("data/raw/customers.csv")
behavior = pd.read_csv("data/raw/behavior.csv")

from ..features.feature_builder import build_feature_table

df = build_feature_table()

# ---------------------------
# Select features
# ---------------------------
FEATURES = [
    "avg_weekly_usage",
    "engagement_velocity",
    "recent_engagement_velocity",
    "frequency",
    "recency",
    "friction_intensity"
]

X = df[FEATURES]
y = df["churned"]

# ---------------------------
# Train / test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# ---------------------------
# Model pipeline
# ---------------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


pipeline.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

import numpy as np

feature_names = FEATURES
coefficients = pipeline.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values(by="coefficient", ascending=False)

print("\nModel Coefficients:")
print(coef_df)


df["churn_probability"] = pipeline.predict_proba(X)[:, 1]

print("\nSample churn probabilities:")
print(df[["customer_id", "churn_probability"]].head())
