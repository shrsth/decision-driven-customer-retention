"""Single entrypoint: download -> clean -> economics -> SQLite -> train -> save.

Usage:
    python -m src.pipeline [--force-download]
"""

import argparse
import json

import pandas as pd

from src.config import DB_PATH, METRICS_PATH, MODEL_PATH, RAW_DATA_PATH
from src.economics import add_economic_fields
from src.features.feature_builder import build_feature_table
from src.ingest import clean_telco_data, download_telco_data
from src.load_to_sqlite import load_to_sqlite
from src.models.train_logistic import (
    compare_models,
    feature_importances,
    save_model,
    train_and_evaluate,
)
from src.sql_feature_queries import churn_summary_by_segment


def run_pipeline(force_download: bool = False) -> dict:
    csv_path = download_telco_data(RAW_DATA_PATH, force=force_download)

    raw = pd.read_csv(csv_path)
    customers = add_economic_fields(clean_telco_data(raw))
    print(
        f"[pipeline] Cleaned {len(customers)} customers "
        f"(churn rate {customers['churned'].mean():.1%})"
    )

    load_to_sqlite(customers, DB_PATH)

    features = build_feature_table(DB_PATH)
    pipeline, metrics = train_and_evaluate(features)
    artifact = save_model(pipeline, MODEL_PATH)
    print(f"[pipeline] Model saved to {artifact}")

    calibration = metrics.pop("calibration_table")
    segments = churn_summary_by_segment(DB_PATH)

    print("[pipeline] Cross-validated model bake-off (5-fold)...")
    comparison = compare_models(features)
    importances = feature_importances(pipeline)

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
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download)


if __name__ == "__main__":
    main()
