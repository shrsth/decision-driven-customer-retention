import warnings

from src.economics import add_economic_fields
from src.features.feature_builder import FEATURES
from src.ingest import clean_telco_data
from src.models.explain import shap_reasons
from src.models.train_logistic import build_pipeline


def test_shap_reasons_readable(raw_telco_df):
    df = add_economic_fields(clean_telco_data(raw_telco_df))
    pipeline = build_pipeline().fit(df[FEATURES], df["churned"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reasons = shap_reasons(pipeline, df[FEATURES])

    assert len(reasons) == len(df)
    assert all(isinstance(r, str) and r for r in reasons)
    # gender is excluded, so no reason should mention it
    joined = " ".join(reasons).lower()
    assert "female" not in joined and "male" not in joined
    # no grammatical "no a partner" artifact
    assert "no a partner" not in joined
