import pytest

from src.economics import add_economic_fields
from src.features.feature_builder import FEATURES
from src.ingest import clean_telco_data
from src.models.train_logistic import (
    build_gbm_pipeline,
    build_pipeline,
    compare_models,
    feature_importances,
    load_model,
    save_model,
)


@pytest.fixture
def fitted_pipeline(raw_telco_df):
    df = add_economic_fields(clean_telco_data(raw_telco_df))
    pipeline = build_pipeline()
    pipeline.fit(df[FEATURES], df["churned"])
    return pipeline, df


def test_pipeline_fits_and_predicts(fitted_pipeline):
    pipeline, df = fitted_pipeline
    probs = pipeline.predict_proba(df[FEATURES])
    assert probs.shape == (len(df), 2)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_save_load_round_trip(fitted_pipeline, tmp_path):
    pipeline, df = fitted_pipeline
    path = tmp_path / "model.joblib"
    save_model(pipeline, path)
    loaded = load_model(path)

    original = pipeline.predict_proba(df[FEATURES])[:, 1]
    restored = loaded.predict_proba(df[FEATURES])[:, 1]
    assert original == pytest.approx(restored)


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.pipeline"):
        load_model(tmp_path / "nope.joblib")


def test_gbm_pipeline_fits_and_predicts(raw_telco_df):
    df = add_economic_fields(clean_telco_data(raw_telco_df))
    pipeline = build_gbm_pipeline()
    pipeline.fit(df[FEATURES], df["churned"])
    probs = pipeline.predict_proba(df[FEATURES])
    assert probs.shape == (len(df), 2)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_compare_models_structure(raw_telco_df):
    df = add_economic_fields(clean_telco_data(raw_telco_df))
    # 2-fold keeps enough of each class per fold on the small fixture
    result = compare_models(df, n_splits=2)

    assert set(result["model"]) == {"Logistic Regression", "Gradient Boosting"}
    expected_cols = {
        "model", "auc_mean", "auc_std", "brier_mean",
        "brier_std", "accuracy_mean", "f1_mean",
    }
    assert expected_cols <= set(result.columns)
    assert result["auc_mean"].between(0, 1).all()
    assert result["brier_mean"].between(0, 1).all()


def test_feature_importances(fitted_pipeline):
    pipeline, _ = fitted_pipeline
    imp = feature_importances(pipeline, top_n=8)
    assert list(imp.columns) == ["feature", "coefficient"]
    assert len(imp) == 8
    # sorted by absolute magnitude (descending)
    abs_coefs = imp["coefficient"].abs().tolist()
    assert abs_coefs == sorted(abs_coefs, reverse=True)
    # names are cleaned of the ColumnTransformer prefixes
    assert not imp["feature"].str.startswith(("num__", "cat__")).any()


def test_profit_threshold():
    import numpy as np

    from src.models.train_logistic import profit_threshold

    rng = np.random.default_rng(0)
    n = 500
    probs = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < probs).astype(int)
    clv = rng.uniform(500, 5000, n)
    cost = rng.uniform(50, 150, n)

    pt = profit_threshold(y, probs, clv, cost, save_rate=0.3)
    assert 0.0 <= pt["best_threshold"] <= 1.0
    # the optimum is at least as good as the naive 0.5 cutoff, by construction
    assert pt["best_value"] >= pt["value_at_half"] - 1e-6
    assert len(pt["thresholds"]) == len(pt["values"])


def test_calibration_table_shape_and_bounds():
    import numpy as np

    from src.models.train_logistic import calibration_table

    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < probs).astype(int)

    table = calibration_table(y, probs)
    assert set(table.columns) == {
        "decile", "customers", "avg_predicted", "actual_churn_rate"
    }
    assert table["customers"].sum() == 500
    assert table["avg_predicted"].between(0, 1).all()
    assert table["actual_churn_rate"].between(0, 1).all()
    # predicted probability must increase across deciles
    assert table["avg_predicted"].is_monotonic_increasing
