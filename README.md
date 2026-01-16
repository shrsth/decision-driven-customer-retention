# Decision-Driven Customer Retention System

A **decision-first customer retention system** that converts churn predictions into **economically justified actions** under **budget and operational constraints**, and validates decision robustness before deployment.

> **Key idea:** Models do not create value. Decisions do.

---

## 🚀 What This Project Does

This is **not** a churn prediction dashboard.

It is a **decision-driven retention system** that:

- Predicts customer churn probability  
- Quantifies economic impact (CLV, retention cost, net retention value)  
- Prioritizes customers based on **economic efficiency**  
- Enforces real-world constraints (budget and operational capacity)  
- Outputs **ACT / MONITOR / IGNORE** decisions  
- Explains *why* each decision was made  
- Tests decision stability under changing constraints  

---

## 🧠 Why This Matters

Most churn projects stop at:
- “Here is the churn probability”
- “Here is a dashboard”

Real retention teams need answers to:
- Who should we act on right now?
- Is this action worth the cost?
- What happens if our budget or capacity changes?
- Are these decisions stable enough to deploy?

This project is explicitly designed to answer **those questions**, not just generate predictions.

---

## 🏗️ System Architecture (Tiered by Design)

The project follows a **tiered architecture** to clearly separate decision logic, analysis, and presentation.

### App Layer
- **core.py** — Tier 1 decision engine containing all business logic, economic modeling, and constraint handling. Framework-agnostic and independent of Streamlit.
- **analysis.py** — Tier 2/3 analytical layer for strategy comparison, robustness checks, decision stability analysis, and counterfactual evaluation.
- **dashboard.py** — Streamlit-based UI used only for simulation, visualization, and communication of decisions.

### Source Layer
- **features/** — Behavioral feature engineering built from structured customer and interaction data.
- **models/** — Logistic regression churn model used strictly as a risk signal.
- **decision/** — Strategy logic and budget/capacity-constrained customer selection policies.
- **tests/** — Unit tests validating feature pipelines and decision correctness.

### Why Tier Separation Matters
- Decision logic is framework-agnostic  
- Core logic can be deployed as a backend service  
- UI can be replaced without touching decision logic  

---

## ⚙️ Decision Framework

1. **Churn Prediction**  
   Logistic Regression is used strictly as a **risk signal**, not as the final decision.

2. **Economic Modeling**
   - Revenue at risk = churn_probability × CLV  
   - Net retention value = revenue at risk − retention cost  

3. **Prioritization**
   Customers are ranked by **economic efficiency**:  
   `efficiency = net_retention_value / retention_cost`

4. **Strategy Layer**
   Business posture is applied:
   - Conservative  
   - Balanced  
   - Aggressive  

5. **Constraints**
   - Total retention budget  
   - Maximum number of customers that can be handled  

6. **Final Actions**
   - ACT  
   - MONITOR  
   - IGNORE  

Only customers with **positive net retention value** are eligible for action.

---

## 📊 Decision Robustness & Safety

Before deployment, the system evaluates:

- Stability under ±10% budget changes  
- Stability under ±10% capacity changes  
- Boundary customers sensitive to strategy shifts  
- Counterfactual differences across strategies  

High stability indicates the system is **safe to operate under real-world uncertainty**.

---

## 🧪 Key Design Choices

- **Logistic Regression** for interpretability and stability  
- **Greedy, constraint-aware selection** for determinism and auditability  
- **Risk bands** represent operational urgency, not statistical confidence  
- **No end-to-end deep learning**, prioritizing decision clarity over model complexity  

---

## 🖥️ Dashboard Purpose

The Streamlit dashboard is **not for exploratory data analysis**.

It exists to:
- Simulate real retention decisions  
- Compare strategies under identical constraints  
- Communicate decisions to non-technical stakeholders  
- Demonstrate system robustness visually  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app/dashboard.py
