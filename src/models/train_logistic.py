"""Churn model: logistic regression pipeline with train/save/load helpers.

Logistic regression is chosen deliberately: the decision engine consumes
churn probabilities (p x CLV), so calibration matters more than raw AUC.
LR is well-calibrated out of the box and its coefficients are auditable.
Upgrade path if more accuracy is ever needed: HistGradientBoostingClassifier
wrapped in CalibratedClassifierCV.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_PATH, SAVE_RATE
from src.features.feature_builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES

TARGET = "churned"


def profit_threshold(y_true, probs, clv, cost, save_rate=SAVE_RATE, n_steps=50):
    """Churn-probability cutoff that maximizes *realized* value on the holdout.

    For a candidate cutoff t, we "act" on every customer with predicted churn
    prob >= t. Backtested on the real labels, acting on a customer returns
    save_rate * (actually churned) * CLV - cost. Summing over the acted set
    gives the realized value at t; the best t beats the naive 0.5 cutoff. This
    is how a probability becomes a cost-aware decision.
    """
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    clv = np.asarray(clv, dtype=float)
    cost = np.asarray(cost, dtype=float)

    per_customer = save_rate * y_true * clv - cost
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    values = np.array([float(per_customer[probs >= t].sum()) for t in thresholds])

    best_i = int(values.argmax())
    half_i = int(np.argmin(np.abs(thresholds - 0.5)))
    return {
        "thresholds": thresholds.round(3).tolist(),
        "values": values.round(1).tolist(),
        "best_threshold": float(thresholds[best_i]),
        "best_value": float(values[best_i]),
        "value_at_half": float(values[half_i]),
    }


def build_pipeline() -> Pipeline:
    """Logistic regression — the production model (calibrated, auditable)."""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def build_gbm_pipeline() -> Pipeline:
    """Gradient boosting — the accuracy-oriented challenger model.

    HistGradientBoostingClassifier needs dense input, so the one-hot encoder
    emits a dense matrix here (unlike the LR pipeline, which keeps it sparse).
    """
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", HistGradientBoostingClassifier(random_state=42)),
    ])


def compare_models(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Cross-validated bake-off: Logistic Regression vs. Gradient Boosting.

    Reports mean +/- std across folds for discrimination (ROC AUC) AND
    calibration (Brier). The decision engine spends budget proportional to
    predicted probabilities, so calibration is a first-class criterion, not
    an afterthought — which is why LR stays the production model even if GBM
    edges it on AUC.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "auc": "roc_auc",
        "brier": "neg_brier_score",
        "accuracy": "accuracy",
        "f1": "f1",
    }
    models = {
        "Logistic Regression": build_pipeline(),
        "Gradient Boosting": build_gbm_pipeline(),
    }

    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "auc_mean": res["test_auc"].mean(),
            "auc_std": res["test_auc"].std(),
            "brier_mean": -res["test_brier"].mean(),  # un-negate
            "brier_std": res["test_brier"].std(),
            "accuracy_mean": res["test_accuracy"].mean(),
            "f1_mean": res["test_f1"].mean(),
        })
    return pd.DataFrame(rows)


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Split the frame (not just X/y) so CLV and cost stay aligned to the test
    # rows for the profit-threshold backtest.
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )
    X_train, y_train = df_train[feature_cols], df_train[TARGET]
    X_test, y_test = df_test[feature_cols], df_test[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "precision_churn": precision_score(y_test, preds),
        "recall_churn": recall_score(y_test, preds),
        "f1_churn": f1_score(y_test, preds),
        "brier_score": brier_score_loss(y_test, probs),
        "mean_predicted_prob": probs.mean(),
        "max_predicted_prob": probs.max(),
        "share_prob_ge_0.60": (probs >= 0.60).mean(),
        "calibration_table": calibration_table(y_test, probs),
        "profit_threshold": profit_threshold(
            y_test, probs, df_test["CLV"], df_test["retention_cost"]
        ),
    }
    return pipeline, metrics


def calibration_table(y_true, probs, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual churn rate by probability decile.

    The decision engine spends budget proportional to these probabilities,
    so predicted and actual rates should track each other closely.
    """
    df = pd.DataFrame({"actual": list(y_true), "predicted": list(probs)})
    df["decile"] = pd.qcut(df["predicted"], q=n_bins, labels=False, duplicates="drop")
    table = df.groupby("decile").agg(
        customers=("actual", "size"),
        avg_predicted=("predicted", "mean"),
        actual_churn_rate=("actual", "mean"),
    ).round(3)
    return table.reset_index()


def feature_importances(pipeline: Pipeline, top_n: int = 12) -> pd.DataFrame:
    """Top churn drivers as standardized logistic-regression coefficients.

    Positive coefficient => raises churn odds; negative => protective. Numeric
    features are standardized and categoricals are one-hot, so magnitudes are
    broadly comparable. This is the interpretability payoff of choosing LR.
    """
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    coefs = model.coef_[0]

    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["feature"] = (
        df["feature"].str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    df = df.reindex(df["coefficient"].abs().sort_values(ascending=False).index)
    return df.head(top_n).reset_index(drop=True)


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path} — run `python -m src.pipeline` first."
        )
    return joblib.load(path)
