"""End-to-end integration: the full pipeline data path, offline.

Unlike the unit tests (which exercise one function on a fixture), this runs
the real chain of modules together — clean -> economics -> SQLite -> features
-> train -> save -> load -> predict — catching wiring/contract mismatches
between stages. No network: it feeds the raw-schema fixture instead of the
live download.
"""

from src.economics import add_economic_fields
from src.features.feature_builder import CONTRACT_COLUMNS, FEATURES, build_feature_table
from src.ingest import clean_telco_data
from src.load_to_sqlite import load_to_sqlite
from src.models.train_logistic import build_pipeline, load_model, save_model


def test_pipeline_data_path_end_to_end(raw_telco_df, tmp_path):
    db_path = tmp_path / "e2e.db"
    model_path = tmp_path / "e2e.joblib"

    # 1-2. clean + derive economics (incl. survival-based CLV)
    customers = add_economic_fields(clean_telco_data(raw_telco_df))

    # 3. persist to SQLite, 4. read features back out
    load_to_sqlite(customers, db_path)
    features = build_feature_table(db_path)

    assert len(features) == len(raw_telco_df)
    for col in CONTRACT_COLUMNS:
        assert col in features.columns

    # 5. train, 6. save, 7. reload
    pipeline = build_pipeline().fit(features[FEATURES], features["churned"])
    save_model(pipeline, model_path)
    loaded = load_model(model_path)

    # 8. the reloaded model scores the same feature table end-to-end
    probs = loaded.predict_proba(features[FEATURES])[:, 1]
    assert len(probs) == len(features)
    assert ((probs >= 0) & (probs <= 1)).all()
