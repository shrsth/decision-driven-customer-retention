# Decision Memo
## Decision-Driven Customer Retention System

### Author
Shresth Modi

---

## 1. Executive Summary

This system is designed to make retention decisions, not just predict churn.

While a churn model estimates the probability that a customer may leave, business value is created only when predictions are converted into economically justified actions. This project implements a decision engine that selects who to act on, why, and how under real-world budget and operational constraints, and validates that these decisions remain stable under changing conditions.

---

## 2. Data Foundation

The system runs on the IBM Telco Customer Churn dataset: 7,043 real telecom
customers with a real churn label (26.5% churn rate). Churn probability is
learned from contract, tenure, services, and billing attributes.

The dataset does not include lifetime value or intervention cost, so CLV and
retention cost are derived per customer from contract, tenure, and monthly
charges under documented assumptions (see economic_assumptions.md). These
derivations are centralized and replaceable without touching decision logic.

---

## 3. Why Churn Prediction Alone Is Insufficient

A churn probability answers only one question:

“How likely is this customer to leave?”

It does not answer:
- Whether the customer is worth saving
- Whether the retention cost is justified
- Whether the business has capacity to act

For example:
- A customer with very high churn probability but low lifetime value may not justify intervention.
- A customer with moderate churn probability and high lifetime value may be economically critical.

Therefore, churn probability is treated as a signal — not a decision rule.

---

## 4. Decision Framework Used

The system follows a decision-first architecture:

1. Predict churn probability using Logistic Regression
2. Estimate economic impact:
   - Revenue at risk = churn_probability × CLV
   - Net retention value = save_rate × revenue at risk − retention cost

   Interventions are not guaranteed to succeed, so at-risk revenue is
   discounted by a save rate (default 30%) before costs are subtracted.
   Assuming every intervention works would inflate ROI roughly threefold.
3. Prioritize customers based on economic efficiency
4. Apply business strategy (Conservative / Balanced / Aggressive)
5. Enforce real-world constraints:
   - Retention budget
   - Operational capacity
6. Assign actions:
   - ACT / MONITOR / IGNORE

Only customers with positive net retention value are eligible for action.

---

## 5. Why Economic Efficiency Drives Decisions

Customers are ranked using economic efficiency rather than raw churn probability:

efficiency = net_retention_value / retention_cost

This ensures:
- Each unit of budget maximizes retained revenue
- High-cost, low-return interventions are avoided
- Decisions scale rationally as constraints change

This mirrors how real retention teams operate under finite resources.

---

## 6. Risk Bands Are Operational, Not Statistical

Risk bands (LOW / MEDIUM / HIGH) are used to:
- Simplify decision logic
- Trigger different retention playbooks
- Improve executive interpretability

They are not probability calibration artifacts or confidence intervals.

Risk bands represent operational urgency, not model certainty.

---

## 7. Why a Greedy, Constraint-Aware Selector Is Used

Customer selection is performed using a greedy, monotonic algorithm under budget and capacity constraints.

This approach is chosen because it is:
- Deterministic
- Explainable
- Auditable
- Easy to reason about in production systems

While more complex optimization techniques exist, this design prioritizes clarity and safety over theoretical optimality.

---

## 8. Decision Stability as a Deployment Requirement

A deployable decision system must behave consistently under small operational changes.

This system explicitly measures:
- Stability under ±10% budget changes
- Stability under ±10% capacity changes

High stability indicates:
- Strong prioritization
- Clear separation between high-value and marginal customers
- Low risk of decision thrashing during real operations

Instability is treated as a design flaw, not an acceptable outcome.

---

## 9. Tiered Architecture Rationale

The system is intentionally split into tiers:

- Tier 1 (core.py):
  Pure decision engine — framework-agnostic, testable, reusable

- Tier 2/3 (analysis.py):
  Strategy comparison, robustness testing, counterfactual analysis

- Dashboard (dashboard.py):
  Simulation and communication layer only

This separation ensures the decision logic can be deployed independently of the UI.

---

## 10. Known Limitation: Persuadability (Uplift)

The engine ranks customers by `save_rate × p(churn) × CLV − cost`. This has a
subtle flaw worth stating plainly, because a careful reviewer will raise it:
**it does not distinguish customers who can be *persuaded* to stay from those
who will churn no matter what.**

A customer with churn probability near 1.0 is top-ranked here — yet if that
customer has already decided to leave, the retention offer is wasted money.
Conversely, a customer who would have stayed anyway yields no incremental
value even if "saved." The quantity that actually matters is the **uplift** (or
treatment effect): the *change* in retention probability caused by the
intervention, not the retention probability itself. The truly persuadable
customers sit in the middle, not at the extremes.

**Why it isn't implemented here:** uplift modeling requires experimental data —
a treatment group that received offers and a control group that did not — so
the incremental effect can be estimated. The IBM Telco dataset has no such
treatment/outcome record, so any uplift curve would be invented, which would
undermine the honesty the rest of this system is built on (see the baseline
comparison and break-even analysis).

**The honest path forward:** run a randomized retention campaign, log who was
offered what and who stayed, then fit an uplift model (e.g., a two-model or
transformed-outcome learner) and rank by predicted uplift × CLV instead of
`p(churn) × CLV`. The decision engine's architecture already supports this —
only the scoring column would change. Until that experimental data exists, the
current `save_rate` is a deliberately simple, transparent stand-in, and the
sensitivity analysis quantifies exactly how much the conclusions depend on it.

---

## 11. Key Design Principle

Models do not create value. Decisions do.

This project demonstrates how machine learning becomes valuable only when embedded inside a constrained, economically grounded decision system.

---

## 12. Final Takeaway

This is not a churn prediction dashboard.

It is a decision-driven retention engine that:
- Converts predictions into actions
- Optimizes under constraints
- Explains decisions
- Validates robustness before deployment
