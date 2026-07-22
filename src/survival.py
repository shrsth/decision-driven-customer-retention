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
