# Economic Assumptions

The decision engine needs three economic quantities per customer: **MRR**,
**CLV**, and **retention_cost**. The IBM Telco dataset provides monthly
charges and contract terms but no lifetime value or intervention cost, so
these are derived with the assumptions below (constants in `src/config.py`,
logic in `src/economics.py`).

## MRR (Monthly Recurring Revenue)

Taken directly from the dataset's `MonthlyCharges`.

## CLV (Customer Lifetime Value)

CLV is modeled as **expected remaining revenue**:

```
CLV = MRR × expected_remaining_months
```

`expected_remaining_months` is **derived from the data**, not a hand-tuned
formula. The tenure/churn columns are right-censored survival data (churned
customers had the event at their tenure; active customers are censored at
theirs), so a **Kaplan-Meier estimator** (`src/survival.py`, implemented from
scratch) fits a survival curve `S(t)` per contract type. Each customer's
expected remaining life is the restricted mean residual life over a 60-month
forward window:

```
expected_remaining_months = Σ_{u=1..60}  S(tenure + u) / S(tenure)
```

Because two-year customers churn far less, their survival curves decay slowly
and yield longer expected lifetimes — entirely from the data:

| Contract | Churn rate | KM expected remaining months |
|---|---|---|
| Month-to-month | ~43% | ~35 |
| One year | ~11% | ~47 |
| Two year | ~3% | ~59 |

This replaces the previous hand-tuned `BASE[contract] × (1 + tenure/72)`
formula: the ordering is the same, but the numbers are now estimated rather
than assumed, which is far easier to defend. Resulting CLV range: roughly
$430–$7,100 per customer.

### Cox proportional-hazards (per-customer lifetime)

The production default (`CLV_METHOD = "cox"`) upgrades the per-*contract* KM
curve to a per-*customer* one. A **Cox proportional-hazards model**
(`lifelines`, in `src/survival.py::cox_expected_remaining`) fits a single
baseline hazard modulated by each customer's covariates (monthly/total charges,
contract, internet service, payment method, paperless billing, partner,
dependents), giving every customer their own survival curve `S_i(t)` and hence
their own expected remaining lifetime.

The difference that matters: KM assigns *every* month-to-month customer the same
~35-month lifetime, whereas Cox spreads them by ~9 months (a high-charge fiber
customer gets a shorter lifetime than a low-charge DSL one on the same
contract). Means still track KM by contract (~27 / 45 / 55 months, correlation
~0.81), and concordance is ~0.90. Economics falls back to KM automatically if
Cox is unavailable or the sample is under 200 rows (`src/economics.py`).

## Retention cost

A retention intervention is modeled as a **3-month win-back discount of 30%
off MRR**, plus a contract-dependent outreach cost:

```
retention_cost = OUTREACH[contract] + 0.30 × MRR × 3
```

| Contract | Outreach cost |
|---|---|
| Month-to-month | $10 |
| One year | $25 |
| Two year | $40 |

Rationale:

- The discount scales with MRR, so **cost varies per customer** — this is
  essential, because the engine ranks customers by
  `net_retention_value / retention_cost`; a constant cost per segment would
  collapse that ranking (a bug the previous synthetic generator had).
- Outreach cost rises with contract tier: longer-contract customers warrant
  higher-touch outreach (calls vs. automated email).

Resulting range: roughly $26–$150 per customer.

## Save rate (intervention success probability)

Retention offers do not always work. The engine discounts at-risk revenue
by a **save rate** — the probability an intervention actually retains the
customer:

```
net_retention_value = save_rate × churn_probability × CLV − retention_cost
```

Default `SAVE_RATE = 0.30` (industry retention-offer acceptance is typically
20-40%), adjustable via a dashboard slider so decisions can be stress-tested
against optimistic and pessimistic assumptions. Without this discount the
system would implicitly assume every intervention succeeds, inflating ROI
roughly 3x.

## Validating the decision layer (not just the model)

Two analyses in the dashboard address the fair question "how do you know these
decisions are any good, given the assumptions?"

**Baseline policy comparison.** Under the same budget, the economic engine is
compared against naive targeting rules — random, target-highest-churn, and
target-highest-CLV. On the real data the engine captures the most expected
value (~20% more than the best naive rule, and vastly more than targeting
high-CLV customers, who often have little revenue actually at risk). This shows
the *decision layer* creates value beyond the churn model alone.

**Assumption sensitivity / break-even.** Because the ROI depends on the save
rate, the dashboard reports the **break-even save rate** — the rate at which the
chosen ACT set's realized net value (`r × Σ(p·CLV) − Σcost`) hits zero. On the
default settings this is ~9%, well below the assumed 30%, so the program has a
wide margin of safety. This reframes the key assumption as a quantified,
falsifiable risk rather than a hidden one.

## Limitations

- CLV ignores discounting, upsell/downsell, and cost-to-serve.
- Retention offers are assumed to always be a discount; real playbooks vary.
- The tenure multiplier is a linear proxy for a survival model; a proper
  survival analysis (e.g., Kaplan-Meier by contract) would be the upgrade path.
- The save rate is constant across customers; the honest upgrade is uplift
  modeling — targeting customers whose behavior changes *because* of the
  intervention, not just those likely to churn.

These assumptions are intentionally simple, documented, and centralized so
they can be challenged and replaced without touching the decision logic.

## Model choice note

The churn model is logistic regression because the engine consumes
*probabilities* (`p × CLV`), so calibration matters more than raw AUC.
This is verified, not assumed: the pipeline reports the Brier score and a
per-decile calibration table (predicted vs. actual churn rate) on the
holdout set every run. If more accuracy is ever needed, the upgrade path is
`HistGradientBoostingClassifier` wrapped in `CalibratedClassifierCV`.
