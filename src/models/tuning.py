"""Hyperparameter tuning (Optuna) and calibration-method comparison.

Two rigor pieces that sit around the model bake-off:
- `tune_gbm` searches the gradient-boosting hyperparameters with Optuna
  (cross-validated AUC) — so the LR-vs-GBM comparison is against a *tuned*
  challenger, not defaults.
- `compare_calibration` checks whether post-hoc calibration (isotonic / Platt)
  beats the raw logistic regression on Brier score. LR is already well
  calibrated, so this is evidence, not decoration.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from src.features.feature_builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.train_logistic import TARGET, build_gbm_pipeline, build_pipeline


def tune_gbm(df: pd.DataFrame, n_trials: int = 20, cv: int = 3, random_state: int = 42):
    """Optuna search over gradient-boosting hyperparameters (CV ROC AUC)."""
    import optuna

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "model__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "model__l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "model__max_iter": trial.suggest_int("max_iter", 100, 400),
        }
        pipe = build_gbm_pipeline().set_params(**params)
        return cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_auc": float(study.best_value)}


def compare_calibration(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> pd.DataFrame:
    """Brier score of raw LR vs. isotonic and Platt (sigmoid) recalibration."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rows = []
    raw = build_pipeline().fit(X_tr, y_tr)
    rows.append(("Raw logistic regression",
                 brier_score_loss(y_te, raw.predict_proba(X_te)[:, 1])))
    for label, method in [("Isotonic", "isotonic"), ("Platt (sigmoid)", "sigmoid")]:
        cal = CalibratedClassifierCV(build_pipeline(), method=method, cv=3)
        cal.fit(X_tr, y_tr)
        rows.append((label, brier_score_loss(y_te, cal.predict_proba(X_te)[:, 1])))

    out = pd.DataFrame(rows, columns=["method", "brier"])
    out["brier"] = out["brier"].round(4)
    out["best"] = out["brier"] == out["brier"].min()
    return out
