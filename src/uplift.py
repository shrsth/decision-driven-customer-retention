"""Uplift (causal) modeling on the Hillstrom email A/B-test dataset.

The main Telco engine ranks by churn probability x CLV — it targets who is
likely to leave, not who can be *persuaded* to stay. The honest version ranks
by **uplift**: the change in outcome caused by the intervention. Uplift needs
experimental data (a treated group and a control group), which Telco lacks but
Hillstrom has — 64k customers randomly split into email vs no email.

This module fits a two-model (T-learner) uplift estimator and evaluates it with
a Qini curve, showing that targeting by uplift captures more incremental
responders than targeting by response propensity (the Telco-style approach).
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import HILLSTROM_PATH, HILLSTROM_URL

NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["history_segment", "zip_code", "channel"]


def load_hillstrom(dest: Path = HILLSTROM_PATH, url: str = HILLSTROM_URL, force=False):
    """Download (cached) and clean the Hillstrom dataset.

    treat = any email sent (vs 'No E-Mail' control); outcome = site visit.
    """
    dest = Path(dest)
    if not dest.exists() or force:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)

    df = pd.read_csv(dest)
    df["treat"] = (df["segment"] != "No E-Mail").astype(int)
    df["outcome"] = df["visit"].astype(int)
    return df


def _model():
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(random_state=42)),
    ])


def fit_two_model_uplift(df):
    """T-learner: separate outcome models for treated and control groups.

    Uplift(x) = P(outcome | treated, x) - P(outcome | control, x).
    """
    treated = df[df["treat"] == 1]
    control = df[df["treat"] == 0]
    X = df[NUMERIC + CATEGORICAL]

    m_t = _model().fit(treated[NUMERIC + CATEGORICAL], treated["outcome"])
    m_c = _model().fit(control[NUMERIC + CATEGORICAL], control["outcome"])
    uplift = m_t.predict_proba(X)[:, 1] - m_c.predict_proba(X)[:, 1]
    return uplift


def response_scores(df):
    """A plain response-propensity model (ignores treatment) — the Telco-style
    'target whoever is most likely to respond' baseline."""
    X = df[NUMERIC + CATEGORICAL]
    m = _model().fit(X, df["outcome"])
    return m.predict_proba(X)[:, 1]


def qini_curve(scores, treat, outcome):
    """Qini curve for a ranking, plus its coefficient (area over random).

    Targeting the top-k by `scores`, cumulative incremental responders are
    responders_treated - responders_control * (n_treated / n_control). The Qini
    coefficient is the area between this curve and the random-targeting line.
    """
    scores = np.asarray(scores)
    treat = np.asarray(treat)
    outcome = np.asarray(outcome)

    order = np.argsort(scores)[::-1]
    t = treat[order]
    y = outcome[order]

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    r_t = np.cumsum(y * t)
    r_c = np.cumsum(y * (1 - t))

    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t, dtype=float), where=n_c > 0)
    qini = r_t - r_c * ratio

    n = len(scores)
    fractions = np.arange(1, n + 1) / n
    random_line = fractions * qini[-1]
    coefficient = float(np.trapz(qini - random_line, fractions))
    return fractions, qini, coefficient
