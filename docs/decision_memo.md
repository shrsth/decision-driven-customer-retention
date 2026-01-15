# Decision Memo  
## Decision-Driven Customer Retention & Revenue Impact System

**Author:** Shresth Modi  
**Role Focus:** AI / ML Engineer · Data Science Engineer  

---

## 1. Objective

Design a **decision-support system** that identifies **which customers should be prioritized for retention** under **limited budget and operational constraints**, with the goal of **minimizing expected revenue loss** — not merely predicting churn.

---

## 2. Business Problem

Customer churn negatively impacts recurring revenue, but **retention resources are finite**.  
Attempting to retain every at-risk customer is economically inefficient and operationally infeasible.

The real business question is:

> **Given limited retention budget and capacity, which customers should we prioritize to maximize revenue saved?**

---

## 3. Design Philosophy

This project was intentionally built as a **decision system**, not a pure ML model.

Key principles:
- Behavior-driven churn modeling (not random labels)
- Strong feature engineering before modeling
- Interpretable ML used as a *supporting tool*
- Explicit cost–value trade-off analysis
- Budget- and capacity-aware decision logic

---

## 4. Data & Signals

### Behavioral Signals
- Engagement velocity (trend of usage over time)
- Recent engagement change
- Usage frequency
- Usage recency
- Friction intensity (support interactions per usage)

### Economic Signals
- Customer Lifetime Value (CLV)
- Estimated retention cost per customer

Churn is modeled as an **outcome that emerges from behavioral degradation**, not a random event.

---

## 5. Modeling Approach

### Churn Risk Estimation
- **Logistic Regression** used for probabilistic and interpretable outputs
- Focus on understanding *drivers of churn*, not maximizing accuracy

The model outputs **churn probability**, not retention decisions.

---

## 6. Revenue-at-Risk Framework

For each customer:

\[
\text{Revenue at Risk} = P(\text{churn}) \times \text{CLV}
\]

This quantifies **expected revenue loss**, not hypothetical loss.

---

## 7. Net Retention Value

Retention is justified only if expected revenue saved exceeds intervention cost:

\[
\text{Net Retention Value} = \text{Revenue at Risk} - \text{Retention Cost}
\]

This ensures decisions are **economically rational**.

---

## 8. Optimization Under Constraints

Retention was framed as a **budget-constrained optimization problem**.

**Objective:**  
Maximize total expected revenue saved.

**Constraints:**
- Fixed retention budget
- Limited operational capacity

### Selection Strategy
Customers are ranked by:

\[
\text{Efficiency} = \frac{\text{Net Retention Value}}{\text{Retention Cost}}
\]

This prioritizes **high-ROI interventions** first.

---

## 9. Action Segmentation

The system outputs **explicit action segments**:

| Segment | Definition | Action |
|------|---------|-------|
| **ACT** | Selected under constraints | Immediate retention action |
| **MONITOR** | Positive value but not selected | Observe / low-cost nudges |
| **IGNORE** | Negative net value | No intervention |

### Final Distribution
- **ACT:** 300 customers  
- **MONITOR:** 824 customers  
- **IGNORE:** 3,876 customers  

This confirms that **most customers should not be targeted**, even if churn risk exists.

---

## 10. What-If / Sensitivity Analysis

Scenario testing demonstrated:

- Reducing budget had **no impact** when capacity was the binding constraint
- Increasing capacity increased revenue saved, but with **diminishing marginal returns**
- High-efficiency customers were consistently prioritized first

This validated the **robustness and stability** of the decision logic.

---

## 11. Key Insights

1. Churn probability alone is insufficient for decision-making  
2. Retention should be framed as an **investment decision**, not a classification task  
3. A small subset of customers drives most retention ROI  
4. Budget and capacity constraints materially change optimal strategy  

---

## 12. Recommendation

Adopt a **decision-driven retention strategy** that:

- Uses churn risk as an input, not the final decision
- Explicitly models customer value and retention cost
- Respects operational and financial constraints
- Produces clear, executable action segments

This approach maximizes revenue efficiency while avoiding unnecessary retention spend.

---

## 13. Limitations & Future Work

- Retention success probability assumed uniform
- Retention cost treated as static
- No channel-level uplift modeling

Future extensions may include:
- Dynamic retention cost estimation
- Channel-specific intervention modeling
- Real-time dashboard integration

---

## 14. Conclusion

This project demonstrates how **data science, machine learning, and business reasoning** can be combined to move from *prediction* to **decision-making**.

The result is a **practical, interpretable, and constraint-aware retention system** suitable for real-world deployment.
