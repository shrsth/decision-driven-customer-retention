"""Monte Carlo simulation of decision quality under uncertainty.

The engine reports a single ROI, but that number rests on uncertain inputs —
above all the intervention save rate. This simulates the committed ACT set many
times, drawing the true save rate from a distribution and the per-customer
churn / save outcomes as coin flips, to produce a *distribution* of realized
ROI: a mean, a confidence interval, and the probability the program is
profitable. It reframes "7.3x" as "2.3x, 90% CI 1.6-3.1x, profitable 94% of the
time" — decisions under uncertainty, not a point estimate.
"""

import numpy as np


def simulate_decision_quality(
    act_df, assumed_save_rate, n_sims=3000, rate_std=0.08, seed=42
):
    """Distribution of realized net value / ROI for the funded ACT set.

    Two sources of randomness per simulation: a systematic draw of the true
    save rate (Beta with the assumed mean and `rate_std` spread), and
    idiosyncratic per-customer outcomes — the customer actually churns
    (Bernoulli on churn_probability) and the offer works (Bernoulli on the
    drawn save rate). Realized value = saved CLV - cost spent.
    """
    p = act_df["churn_probability"].to_numpy(dtype=float)
    clv = act_df["CLV"].to_numpy(dtype=float)
    cost = act_df["retention_cost"].to_numpy(dtype=float)
    total_cost = float(cost.sum())
    n = len(p)

    if n == 0 or total_cost <= 0:
        return None

    rng = np.random.default_rng(seed)

    # Save-rate draws: Beta with mean = assumed rate and the requested spread.
    m = float(assumed_save_rate)
    var = min(rate_std ** 2, m * (1 - m) * 0.99)
    conc = m * (1 - m) / var - 1
    save_rates = rng.beta(m * conc, (1 - m) * conc, n_sims)

    # Vectorized: (n_sims x n) coin flips.
    churned = rng.random((n_sims, n)) < p
    saved = rng.random((n_sims, n)) < save_rates[:, None]
    gross = ((churned & saved) * clv).sum(axis=1)
    net_values = gross - total_cost
    rois = net_values / total_cost

    return {
        "assumed_save_rate": m,
        "cost": total_cost,
        "net_mean": float(net_values.mean()),
        "net_p5": float(np.percentile(net_values, 5)),
        "net_p95": float(np.percentile(net_values, 95)),
        "roi_mean": float(rois.mean()),
        "roi_p5": float(np.percentile(rois, 5)),
        "roi_p95": float(np.percentile(rois, 95)),
        "prob_profitable": float((net_values > 0).mean()),
        "rois": rois.round(3).tolist(),
    }
