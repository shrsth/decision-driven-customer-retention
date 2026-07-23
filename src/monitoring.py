"""Data-drift monitoring: is today's data still like the training data?

A deployed model silently degrades when the incoming population drifts away from
what it was trained on. This computes the Population Stability Index (PSI) per
feature between a reference sample (training data) and a current batch, plus a
Kolmogorov-Smirnov test for numeric features, and flags features that have
drifted. It's the "is the model still valid?" alarm that makes the deployability
claim real.

Run a demonstration:  python -m src.monitoring
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# PSI rule-of-thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant drift.
PSI_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.01
_EPS = 1e-6


def population_stability_index(expected, actual, bins=10):
    """PSI between two numeric samples using quantile bins of `expected`."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, _EPS, None)
    a = np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _psi_categorical(expected: pd.Series, actual: pd.Series):
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    psi = 0.0
    for cat in set(e.index) | set(a.index):
        ep = max(float(e.get(cat, 0.0)), _EPS)
        ap = max(float(a.get(cat, 0.0)), _EPS)
        psi += (ap - ep) * np.log(ap / ep)
    return float(psi)


def drift_report(reference, current, numeric_cols, categorical_cols) -> pd.DataFrame:
    """Per-feature PSI (+ KS for numeric) and a drift flag, worst first."""
    rows = []
    for col in numeric_cols:
        psi = population_stability_index(reference[col], current[col])
        ks_p = float(ks_2samp(reference[col], current[col]).pvalue)
        rows.append({
            "feature": col, "type": "numeric", "psi": round(psi, 3),
            "ks_pvalue": round(ks_p, 4),
            "drift": bool(psi > PSI_THRESHOLD or ks_p < KS_PVALUE_THRESHOLD),
        })
    for col in categorical_cols:
        psi = _psi_categorical(reference[col], current[col])
        rows.append({
            "feature": col, "type": "categorical", "psi": round(psi, 3),
            "ks_pvalue": None, "drift": bool(psi > PSI_THRESHOLD),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _demo():
    """Compare the Telco training data against a deliberately drifted batch."""
    from src.features.feature_builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from src.ingest import clean_telco_data
    from src.config import RAW_DATA_PATH

    ref = clean_telco_data(pd.read_csv(RAW_DATA_PATH))
    drifted = ref.copy()
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * 1.4      # price hike
    drifted["Contract"] = "Month-to-month"                          # mix shift

    print("[monitoring] reference vs. itself (expect no drift):")
    print(drift_report(ref, ref, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
          .head(4).to_string(index=False))
    print("\n[monitoring] reference vs. drifted batch (expect drift flags):")
    print(drift_report(ref, drifted, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
          .head(4).to_string(index=False))


if __name__ == "__main__":
    _demo()
