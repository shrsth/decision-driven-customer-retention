# Decision Memo
## Decision-Driven Customer Retention System

### Author
Shresth Modi

---

## 1. Executive Summary

This system is designed to make retention decisions, not just predict churn.

While a churn model estimates the probability that a customer may leave, business value is created only when predictions are converted into economically justified actions. This project implements a decision engine that selects who to act on, why, and how under real-world budget and operational constraints, and validates that these decisions remain stable under changing conditions.

---

## 2. Why Churn Prediction Alone Is Insufficient

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

## 3. Decision Framework Used

The system follows a decision-first architecture:

1. Predict churn probability using Logistic Regression
2. Estimate economic impact:
   - Revenue at risk = churn_probability × CLV
   - Net retention value = revenue at risk − retention cost
3. Prioritize customers based on economic efficiency
4. Apply business strategy (Conservative / Balanced / Aggressive)
5. Enforce real-world constraints:
   - Retention budget
   - Operational capacity
6. Assign actions:
   - ACT / MONITOR / IGNORE

Only customers with positive net retention value are eligible for action.

---

## 4. Why Economic Efficiency Drives Decisions

Customers are ranked using economic efficiency rather than raw churn probability:

efficiency = net_retention_value / retention_cost

This ensures:
- Each unit of budget maximizes retained revenue
- High-cost, low-return interventions are avoided
- Decisions scale rationally as constraints change

This mirrors how real retention teams operate under finite resources.

---

## 5. Risk Bands Are Operational, Not Statistical

Risk bands (LOW / MEDIUM / HIGH) are used to:
- Simplify decision logic
- Trigger different retention playbooks
- Improve executive interpretability

They are not probability calibration artifacts or confidence intervals.

Risk bands represent operational urgency, not model certainty.

---

## 6. Why a Greedy, Constraint-Aware Selector Is Used

Customer selection is performed using a greedy, monotonic algorithm under budget and capacity constraints.

This approach is chosen because it is:
- Deterministic
- Explainable
- Auditable
- Easy to reason about in production systems

While more complex optimization techniques exist, this design prioritizes clarity and safety over theoretical optimality.

---

## 7. Decision Stability as a Deployment Requirement

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

## 8. Tiered Architecture Rationale

The system is intentionally split into tiers:

- Tier 1 (core.py):
  Pure decision engine — framework-agnostic, testable, reusable

- Tier 2/3 (analysis.py):
  Strategy comparison, robustness testing, counterfactual analysis

- Dashboard (dashboard.py):
  Simulation and communication layer only

This separation ensures the decision logic can be deployed independently of the UI.

---

## 9. Key Design Principle

Models do not create value. Decisions do.

This project demonstrates how machine learning becomes valuable only when embedded inside a constrained, economically grounded decision system.

---

## 10. Final Takeaway

This is not a churn prediction dashboard.

It is a decision-driven retention engine that:
- Converts predictions into actions
- Optimizes under constraints
- Explains decisions
- Validates robustness before deployment
