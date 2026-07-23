"""Kaplan-Meier survival estimation for data-driven customer lifetime.

The Telco data is right-censored: churned customers had the event at their
tenure; still-active customers are censored at their current tenure. A
Kaplan-Meier estimator turns this into a survival curve S(t) per contract
type, from which we derive an *empirical* expected remaining lifetime — used
by the CLV model instead of a hand-tuned formula.

Implemented from scratch (no lifelines dependency) to keep the method explicit.
"""

import numpy as np
import pandas as pd


def kaplan_meier(durations, events):
    """Kaplan-Meier survival curve.

    Parameters
    ----------
    durations : array-like of observed times (tenure in months)
    events    : array-like, 1 if the event (churn) occurred, 0 if censored

    Returns
    -------
    (times, survival) : sorted unique event/censor times and S(t) at each.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)

    times = np.unique(durations)
    survival = np.empty(len(times))
    s = 1.0
    for i, t in enumerate(times):
        at_risk = np.count_nonzero(durations >= t)
        deaths = np.count_nonzero((durations == t) & (events == 1))
        if at_risk > 0:
            s *= 1.0 - deaths / at_risk
        survival[i] = s
    return times, survival


def _survival_at(times, survival, query_months):
    """Step-function lookup: S(query). Before the first event, S = 1."""
    idx = np.searchsorted(times, query_months, side="right") - 1
    if idx < 0:
        return 1.0
    return float(survival[idx])


def expected_remaining_life(times, survival, at_tenure, forward_months):
    """Restricted mean residual life over a forward window.

    Expected additional months a customer at `at_tenure` stays, looking
    `forward_months` ahead:  sum over u in 1..forward of S(t+u) / S(t).
    Survival is non-increasing, so the result is bounded by `forward_months`.
    The KM tail is flat-extrapolated beyond the last observed time (standard
    convention), so long-tenured loyal customers approach the full window.
    """
    s_t = _survival_at(times, survival, at_tenure)
    if s_t <= 1e-9:
        return 0.0
    months = np.arange(1, int(forward_months) + 1)
    conditional = np.array(
        [_survival_at(times, survival, at_tenure + m) for m in months]
    ) / s_t
    return float(conditional.sum())


def expected_remaining_by_group(
    df, group_col, duration_col, event_col, forward_months
) -> pd.Series:
    """Expected remaining months for every row, from a per-group KM curve.

    Fits one survival curve per `group_col` value and evaluates each customer's
    conditional expected remaining life at their own tenure.
    """
    remaining = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(group_col):
        times, survival = kaplan_meier(group[duration_col], group[event_col])
        remaining.loc[group.index] = [
            expected_remaining_life(times, survival, tenure, forward_months)
            for tenure in group[duration_col]
        ]
    return remaining


def cox_expected_remaining(
    df, numeric_covariates, categorical_covariates,
    duration_col, event_col, forward_months, penalizer=0.1,
) -> pd.Series:
    """Expected remaining months per customer from a Cox proportional-hazards fit.

    Where the KM approach fits one curve per contract group, a Cox model uses
    *all* covariates to give each customer their own survival curve S_i(t) —
    a fully individualized, statistically grounded expected lifetime. Remaining
    life is the restricted mean residual life over a forward window:
    sum over u in (tenure, tenure+forward] of S_i(u) / S_i(tenure).
    """
    from lifelines import CoxPHFitter

    X = pd.get_dummies(
        df[numeric_covariates + categorical_covariates],
        columns=categorical_covariates, drop_first=True,
    ).astype(float)

    fit_df = X.copy()
    fit_df[duration_col] = df[duration_col].to_numpy()
    fit_df[event_col] = df[event_col].to_numpy()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(fit_df, duration_col=duration_col, event_col=event_col)

    max_t = int(df[duration_col].max()) + int(forward_months)
    times = np.arange(0, max_t + 1)
    sf = cph.predict_survival_function(X, times=times).to_numpy()  # (len(times), n)

    tenures = df[duration_col].to_numpy().astype(int)
    remaining = np.zeros(len(df))
    for j in range(len(df)):
        t0 = tenures[j]
        s_t0 = sf[t0, j] if t0 < len(times) else sf[-1, j]
        if s_t0 <= 1e-9:
            continue
        hi = min(t0 + int(forward_months), max_t)
        if hi > t0:
            remaining[j] = float((sf[t0 + 1:hi + 1, j] / s_t0).sum())
    return pd.Series(remaining, index=df.index)
