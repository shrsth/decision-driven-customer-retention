# Decision-Driven Customer Retention System

### 🔗 [Live Demo](https://decision-driven-customer-retention.streamlit.app/) &nbsp;·&nbsp; [Source](https://github.com/shrsth/decision-driven-customer-retention)

A **decision-first customer retention system** that converts churn predictions into **economically justified actions** under **budget and operational constraints**, and validates those decisions before deployment.

> **Key idea:** Models do not create value. Decisions do.

> The live demo may take ~30s to wake on first load (free tier sleeps when idle) and then bootstraps its data and model.

---

## 🚀 What This Project Does

This is **not** a churn prediction dashboard.

It is a **decision-driven retention system** that:

- Predicts customer churn probability
- Quantifies economic impact (CLV, retention cost, net retention value)
- Prioritizes customers by **economic efficiency**
- Enforces real-world constraints (budget and operational capacity)
- Outputs **ACT / MONITOR / IGNORE** decisions
- Explains *why* each decision was made
- Validates the decisions: beats naive targeting, and quantifies its own break-even assumption

---

## 📦 Dataset

The system runs on the **IBM Telco Customer Churn** dataset — 7,043 real telecom customers with a real churn label (~26.5% churn rate), contract terms, tenure, services, and monthly charges. The pipeline downloads it automatically on first run (and caches it locally) from:

```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

The dataset has no lifetime value or intervention cost, so **CLV and retention cost are derived per customer**. CLV uses a **Kaplan-Meier survival model** fit per contract type (data-driven, not a hand-tuned formula). Assumptions are documented in [docs/economic_assumptions.md](docs/economic_assumptions.md).

---

## 🏗️ System Architecture

The project separates **decision logic**, **analysis**, and **presentation** so the core is framework-agnostic and deployable without the UI.

**Source layer (`src/`)**
- **pipeline.py** — single entrypoint: download → clean → economics → SQLite → train → save
- **ingest.py** — dataset download (cached) and cleaning
- **economics.py** — per-customer MRR, CLV, retention cost
- **survival.py** — Kaplan-Meier estimator for data-driven expected lifetime
- **load_to_sqlite.py / sql_feature_queries.py** — SQLite persistence and in-database churn summaries
- **features/** — feature table construction from the SQLite customers table
- **models/** — logistic-regression churn model (serialized with joblib), cross-validated bake-off vs. gradient boosting, calibration, feature importance
- **decision/** — economic scoring, strategy weights, budget/capacity selection, policy baselines, sensitivity

**App layer (`app/`)**
- **core.py** — Tier 1 decision engine (pure Python, no Streamlit dependency)
- **analysis.py** — Tier 2/3: strategy comparison, stability, boundary zone, policy comparison, sensitivity
- **dashboard.py / charts.py / ui.py / darkveil.py** — Streamlit UI, Plotly charts, styling, animated banner

---

## ⚙️ Decision Framework

1. **Churn prediction** — Logistic Regression, used strictly as a **risk signal** (chosen for calibration + interpretability, verified against gradient boosting).
2. **Economic modeling**
   - Revenue at risk = churn_probability × CLV
   - Net retention value = **save_rate** × revenue at risk − retention cost
   - (`save_rate` = probability the intervention actually works, default 30%)
3. **Prioritization** — rank by economic value; only customers with **positive net retention value** are eligible.
4. **Strategy layer** — Conservative / Balanced / Aggressive business postures.
5. **Constraints** — total retention budget and maximum customers handled.
6. **Final actions** — ACT / MONITOR / IGNORE.

---

## 📊 Model & Decision Validation

- **Cross-validated bake-off** — 5-fold CV comparing Logistic Regression vs. Gradient Boosting on AUC *and* calibration (Brier). LR is kept on evidence, not assertion.
- **Calibration curve** — predicted vs. actual churn by decile, since the engine spends budget proportional to probabilities.
- **Baseline policy comparison** — the engine captures more expected value than random / highest-churn / highest-CLV targeting at the same budget.
- **Break-even sensitivity** — reports the save rate at which the program stops paying for itself, quantifying the key assumption instead of hiding it.
- **Stability tests** — how ACT decisions hold up under ±10% budget/capacity changes.

---

## 🧪 Key Design Choices

- **Logistic Regression** for calibrated probabilities and interpretability
- **Greedy, constraint-aware selection** for determinism and auditability
- **Risk bands** represent operational urgency, not statistical confidence
- **Survival-analysis CLV** so lifetime is estimated from data, not assumed

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

The dashboard **self-bootstraps**: on first launch it runs the pipeline automatically if the model/database are missing, so a fresh deploy works out of the box. After re-running the pipeline manually, restart Streamlit so it picks up the new artifacts.

---

## ☁️ Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file to `app/dashboard.py` and deploy.

The app bootstraps its own data and model on first load, so no manual setup is needed on the server. (The dataset, database, and model artifacts live under `data/` and are gitignored — they are regenerated on deploy.)
