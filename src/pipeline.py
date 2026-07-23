"""Single entrypoint: download -> clean -> economics -> SQLite -> train -> save.

Usage:
    python -m src.pipeline [--force-download]
"""

import argparse
import json

import pandas as pd

from src.config import (
    CLV_HORIZON_MONTHS,
    CLV_METHOD,
    DB_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)
from src.economics import add_economic_fields
from src.survival import expected_remaining_by_group
from src.features.feature_builder import build_feature_table
from src.ingest import clean_telco_data, download_telco_data
from src.load_to_sqlite import load_to_sqlite
from src.models.train_logistic import (
    compare_models,
    feature_importances,
    save_model,
    train_and_evaluate,
)
from src.models.tuning import compare_calibration, tune_gbm
from src.sql_feature_queries import churn_summary_by_segment


def run_pipeline(force_download: bool = False, tune: bool = False) -> dict:
    csv_path = download_telco_data(RAW_DATA_PATH, force=force_download)

    raw = pd.read_csv(csv_path)
    customers = add_economic_fields(clean_telco_data(raw))
    print(
        f"[pipeline] Cleaned {len(customers)} customers "
        f"(churn rate {customers['churned'].mean():.1%})"
    )

    # CLV survival-method comparison: KM gives one lifetime per contract; Cox
    # individualizes it per customer using all covariates.
    cox_rem = customers["CLV"] / customers["MRR"]
    km_rem = expected_remaining_by_group(
        customers, "Contract", "tenure", "churned", CLV_HORIZON_MONTHS
    )
    within = cox_rem.groupby(customers["Contract"]).std().mean()
    print(
        f"[pipeline] CLV lifetime via '{CLV_METHOD}' "
        f"(corr with KM {cox_rem.corr(km_rem):.2f}; Cox adds "
        f"{within:.1f}-month within-contract spread that KM cannot)"
    )

    load_to_sqlite(customers, DB_PATH)

    features = build_feature_table(DB_PATH)
    pipeline, metrics = train_and_evaluate(features)
    artifact = save_model(pipeline, MODEL_PATH)
    print(f"[pipeline] Model saved to {artifact}")

    calibration = metrics.pop("calibration_table")
    profit_thr = metrics.pop("profit_threshold")
    segments = churn_summary_by_segment(DB_PATH)

    print("[pipeline] Cross-validated model bake-off (5-fold)...")
    comparison = compare_models(features)
    importances = feature_importances(pipeline)

    print("[pipeline] Calibration-method comparison...")
    calibration_methods = compare_calibration(features)

    gbm_tuning = None
    if tune:
        print("[pipeline] Optuna gradient-boosting tuning (this takes ~30s)...")
        gbm_tuning = tune_gbm(features)

    # Persist a metrics artifact (model card) the dashboard reads without retraining.
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as fh:
        json.dump(
            {
                "scores": {k: float(v) for k, v in metrics.items()},
                "calibration": calibration.to_dict(orient="records"),
                "segments": segments.to_dict(orient="records"),
                "model_comparison": comparison.round(4).to_dict(orient="records"),
                "feature_importance": importances.round(4).to_dict(orient="records"),
                "profit_threshold": profit_thr,
                "calibration_methods": calibration_methods.to_dict(orient="records"),
                "gbm_tuning": gbm_tuning,
                "churn_rate": float(customers["churned"].mean()),
                "n_customers": int(len(customers)),
            },
            fh,
            indent=2,
        )
    print(f"[pipeline] Metrics saved to {METRICS_PATH}")

    print("\n[pipeline] Holdout metrics (logistic regression):")
    for name, value in metrics.items():
        print(f"  {name:>22}: {value:.4f}")

    print("\n[pipeline] Model comparison (5-fold CV, mean +/- std):")
    print(comparison.round(4).to_string(index=False))

    print("\n[pipeline] Top churn drivers (logistic-regression coefficients):")
    print(importances.round(3).to_string(index=False))

    print("\n[pipeline] Calibration methods (Brier, lower is better):")
    print(calibration_methods.to_string(index=False))

    if gbm_tuning:
        print(
            f"\n[pipeline] Tuned gradient boosting: CV AUC {gbm_tuning['best_auc']:.4f} "
            f"(still <= logistic regression; LR kept for calibration + interpretability)"
        )

    print(
        f"\n[pipeline] Profit-maximizing threshold: {profit_thr['best_threshold']:.2f} "
        f"(value ${profit_thr['best_value']:,.0f} vs ${profit_thr['value_at_half']:,.0f} "
        "at the naive 0.5 cutoff)"
    )

    print("\n[pipeline] Calibration by probability decile "
          "(predicted vs. actual should track):")
    print(calibration.to_string(index=False))

    print("\n[pipeline] Churn summary by segment (SQL):")
    print(segments.to_string(index=False))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retention data pipeline.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if a cached copy exists.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna gradient-boosting tuning (~30s extra).",
    )
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download, tune=args.tune)


if __name__ == "__main__":
    main()
