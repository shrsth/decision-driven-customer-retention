# Decision-Driven Customer Retention System

A decision-first customer retention system that converts churn predictions into economically justified actions under budget and operational constraints, and validates decision robustness before deployment.

> **Key idea:** Models do not create value. Decisions do.

---

## 🚀 What This Project Does

This is **not** a churn prediction dashboard.

It is a **decision-driven retention system** that:
- Predicts customer churn probability
- Quantifies economic impact (CLV, retention cost, net retention value)
- Prioritizes customers based on **economic efficiency**
- Enforces real-world constraints (budget + operational capacity)
- Outputs **ACT / MONITOR / IGNORE** decisions
- Explains *why* each decision was made
- Tests decision stability under changing constraints

---

## 📦 Dataset

The system runs on the **IBM Telco Customer Churn** dataset — 7,043 real
telecom customers with a real churn label (26.5% churn rate), contract terms,
tenure, services, and monthly charges. The pipeline downloads it automatically
on first run (and caches it locally) from:

```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

The dataset has no lifetime value or intervention cost, so **CLV and
retention cost are derived per customer** from contract, tenure, and monthly
charges. The formulas and their rationale are documented in
[docs/economic_assumptions.md](docs/economic_assumptions.md).

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

This project explicitly answers those questions.

---

## 🏗️ System Architecture (Tiered by Design)

The project is organized using a tiered architecture to clearly separate decision logic, analysis, and presentation.

App Layer

core.py – Tier 1 decision engine containing all core business, economic, and constraint logic. This layer is framework-agnostic and does not depend on Streamlit.

analysis.py – Tier 2/3 analytical layer responsible for strategy comparison, robustness checks, decision stability analysis, and counterfactual evaluation.

dashboard.py – Streamlit-based user interface used only for simulation, visualization, and communication of decisions.

Source Layer (src/)

pipeline.py – Single entrypoint: download → clean → economics → SQLite → train → save model.

ingest.py – Dataset download (cached) and cleaning.

economics.py – Per-customer CLV and retention-cost derivation.

load_to_sqlite.py / sql_feature_queries.py – SQLite persistence and in-database churn summaries.

features/ – Feature table construction from the SQLite customers table.

models/ – Logistic regression churn model (trained by the pipeline, serialized with joblib, used strictly as a risk signal).

decision/ – Strategy logic and budget/capacity-constrained customer selection policies.

tests/ (repo root) – pytest suite validating ingestion, economics, features, decisions, and the model artifact.


### Why tier separation matters
- Decision logic is framework-agnostic
- Can be deployed as a backend service
- UI is replaceable without touching core logic

---

## ⚙️ Decision Framework

1. **Churn prediction**  
   Logistic Regression is used strictly as a risk signal.

2. **Economic modeling**
   - Revenue at risk = churn_probability × CLV
   - Net retention value = save_rate × revenue at risk − retention cost
   - (save_rate = probability the intervention actually works, default 30%)

3. **Prioritization**
   Customers are ranked by economic efficiency:
   efficiency = net_retention_value / retention_cost

4. **Strategy layer**
Business posture is applied:
- Conservative
- Balanced
- Aggressive

5. **Constraints**
- Retention budget
- Maximum number of customers that can be handled

6. **Final actions**
- ACT
- MONITOR
- IGNORE

Only customers with positive net retention value are eligible for action.

---

## 📊 Decision Robustness & Safety

Before deployment, the system validates:

- Stability under ±10% budget changes
- Stability under ±10% capacity changes
- Boundary customers sensitive to strategy shifts
- Counterfactual differences across strategies

High stability indicates the system is **safe to operate in real-world conditions**.

---

## 🧪 Key Design Choices

- **Logistic Regression** chosen for interpretability and stability  
- **Greedy, constraint-aware selection** for determinism and auditability  
- **Risk bands** represent operational urgency, not statistical confidence  
- **No end-to-end deep learning**, prioritizing decision clarity over model complexity  

---

## 🖥️ Dashboard Purpose

The Streamlit dashboard is **not for EDA**.

It exists to:
- Simulate real retention decisions
- Compare strategies under identical constraints
- Communicate decisions to non-technical stakeholders
- Demonstrate system robustness visually

---

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the data pipeline (downloads the dataset on first run,
#    loads SQLite, trains and saves the churn model)
python -m src.pipeline

# 3. Run the tests (offline, no network needed)
python -m pytest tests -q

# 4. Launch the dashboard
python -m streamlit run app/dashboard.py
```

Note: after re-running the pipeline, restart Streamlit (or clear its cache)
so the dashboard picks up the new data and model.
