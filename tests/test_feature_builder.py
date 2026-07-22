from src.economics import add_economic_fields
from src.features.feature_builder import CONTRACT_COLUMNS, FEATURES, build_feature_table
from src.ingest import clean_telco_data
from src.load_to_sqlite import load_to_sqlite


def test_sqlite_round_trip(raw_telco_df, tmp_path):
    db_path = tmp_path / "test.db"
    customers = add_economic_fields(clean_telco_data(raw_telco_df))
    load_to_sqlite(customers, db_path)

    features = build_feature_table(db_path)

    assert len(features) == len(raw_telco_df)
    for col in CONTRACT_COLUMNS + FEATURES:
        assert col in features.columns, f"missing column {col}"


def test_missing_db_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        build_feature_table(tmp_path / "does_not_exist.db")
