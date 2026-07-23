# Interview Study Guide — Decision-Driven Customer Retention

Everything you need to explain and defend this project in an interview. Read the
**Pitches** and **Numbers to memorize** first; the rest is depth for follow-ups.

Live demo: https://decision-driven-customer-retention.streamlit.app/ ·
Repo: https://github.com/shrsth/decision-driven-customer-retention

---

## 1. Pitches

**One line:** "Most churn projects stop at a prediction; mine turns churn
probabilities into budget-constrained retention *decisions*, and proves the
decision layer actually creates value."

**30 seconds:** "I built an end-to-end system on the real IBM Telco churn
dataset. A logistic-regression model predicts churn, but the model is just a
signal — the core is a decision engine that, given a fixed retention budget and
team capacity, picks exactly which customers to act on to maximize expected
retained revenue. I derive each customer's lifetime value from a survival model,
weigh it against the cost of intervening, and output ACT / MONITOR / IGNORE
decisions. Then I validate those decisions: I show they beat naive targeting,
quantify how sensitive the ROI is to my assumptions with Monte Carlo, and even
demonstrate the causal (uplift) version on an A/B-test dataset. It's deployed,
tested, and also exposed as a REST API."

**2 minutes:** expand each of the above with: the economic model (CLV = MRR ×
Cox-survival expected lifetime; retention cost = outreach + discount; save-rate
assumption), the model choice (LR for calibration, verified against tuned
gradient boosting), the optimizer benchmark (greedy ≈ 100% of the ILP optimum),
the validation suite (policy baselines, break-even, Monte Carlo, stability), the
honest limitation (uplift needs experimental data — demonstrated on Hillstrom),
and the engineering (51 tests, CI, Docker, FastAPI, drift monitoring).

---

## 2. The core idea (the thing that makes it different)

> **Models do not create value. Decisions do.**

A churn probability answers "how likely is this customer to leave?" It does *not*
answer the questions a retention team actually has:
- Is this customer *worth* saving? (a high-churn, low-value customer may not be)
- Is the intervention worth its cost?
- Do we have the budget/capacity to act?
- Are these decisions stable enough to trust?

So churn probability is treated as an **input signal**, and the deliverable is an
economically justified, **budget- and capacity-constrained** action per customer.
This "decision-first" framing is the project's spine and the answer to "what's
novel here?"

---

## 3. Data

- **IBM Telco Customer Churn** — 7,043 real telecom customers, **26.5% churn**.
- Columns: tenure, monthly/total charges, contract, internet service, add-on
  services, payment method, demographics, and a real `Churn` label.
- Downloaded automatically and cached; `TotalCharges` has ~11 blank strings
  (tenure-0 customers) coerced to 0.
- **Why it matters:** the churn label is real (no leakage), which is why the
  honest AUC is ~0.84 — close to the dataset's ceiling. Earlier the project used
  synthetic data whose label was a deterministic rule; that was replaced.

---

## 4. Architecture & tech stack

**Flow:** download → clean/validate → economics → SQLite → features → train →
serialize → decision engine → (Streamlit dashboard | FastAPI | React client).

```
src/
  config.py            paths, dataset URLs, economic constants
  ingest.py            download + clean + Pandera schema validation
  economics.py         MRR / CLV / retention_cost (per-customer)
  survival.py          Kaplan-Meier + Cox proportional-hazards lifetime
  load_to_sqlite.py    persist the customer table
  sql_feature_queries.py  read features / churn summary from SQLite
  features/feature_builder.py  the model feature set
  models/train_logistic.py  LR pipeline, GBM challenger, CV bake-off,
                            calibration table, feature importance, profit threshold
  models/tuning.py     Optuna GBM tuning + calibration-method comparison
  models/explain.py    SHAP per-customer explanations
  decision/retention_strategy.py  scoring, strategy weights, greedy selection,
                                  policy comparison, break-even sensitivity
  decision/optimizer.py  exact ILP (knapsack) + greedy-vs-optimal gap
  simulation.py        Monte Carlo decision-quality distribution
  uplift.py            Hillstrom load + two-model uplift + Qini
  monitoring.py        PSI / KS data-drift report
  pipeline.py          single entrypoint tying it together
  logging_config.py    structured logging
app/
  core.py              the pure decision engine (no Streamlit)
  analysis.py          cached Streamlit wrappers
  dashboard.py, charts.py, ui.py, ferrofluid.py, hero.py  the UI
api.py                 FastAPI service (POST /decisions)
frontend/index.html    minimal React client on the API
notebooks/             EDA + uplift, executed with outputs
tests/                 51 pytest tests
Dockerfile, pyproject.toml, .github/workflows/ci.yml
```

**Stack:** Python, pandas/numpy, scikit-learn, **lifelines** (survival),
**SHAP**, **Optuna**, **PuLP** (ILP), scipy, **Pandera**, SQLite, Streamlit,
Plotly, **FastAPI**/uvicorn, React (CDN), Docker, GitHub Actions, ruff/mypy.

**Why the tiered split (core / analysis / UI)?** `app/core.py` has zero
Streamlit imports, so the exact same engine runs behind the FastAPI service and
the React client. That separation is what makes "framework-agnostic, deployable
backend" a demonstrated fact, not a claim.

---

## 5. The economics (how a probability becomes money)

Per customer:
- **MRR** = MonthlyCharges.
- **CLV** = MRR × expected remaining lifetime (months). Lifetime is **estimated
  from a survival model**, not assumed:
  - *Kaplan-Meier* fits one survival curve per contract type from the
    right-censored tenure/churn data.
  - *Cox proportional-hazards* (the production default) fits one baseline hazard
    modulated by all covariates, giving **each customer their own** survival
    curve — so lifetime varies *within* a contract (~7-month spread) which KM
    can't capture. Concordance ~0.90; correlation with KM ~0.81.
- **retention_cost** = contract-based outreach cost + a 3-month, 30%-of-MRR
  win-back discount → varies per customer (fixing an earlier bug where it was
  constant per contract).
- **save_rate** = probability an intervention actually works (default **0.30** —
  industry offer-acceptance is ~20-40%). This is an *assumption*, handled
  transparently (see validation).

**Key equations:**
- revenue_at_risk = churn_prob × CLV
- net_retention_value = **save_rate × churn_prob × CLV − retention_cost**
- You only ACT where net_retention_value > 0.

Without the save_rate the model would implicitly assume every intervention
succeeds, inflating ROI ~3×.

---

## 6. The model

- **Logistic regression** (pipeline: median impute + scale numerics, one-hot
  categoricals, `LogisticRegression`). Holdout: **ROC AUC ≈ 0.842**, accuracy
  ~81%, F1 (churn) ~0.60, **Brier ≈ 0.138**.
- **Why LR, not gradient boosting?** The engine spends real budget *proportional
  to predicted probabilities*, so **calibration matters more than raw accuracy**.
  I proved this rather than asserting it:
  - **5-fold CV bake-off:** LR AUC 0.845 ± 0.013 vs GBM ~0.835 — LR wins on AUC
    *and* Brier.
  - **Optuna-tuned GBM** still only reaches ~0.844 AUC — no better than LR, and
    worse on calibration/interpretability.
  - **Calibration methods:** isotonic/Platt recalibration barely moves LR's Brier
    (0.1380 → 0.1379) — LR is already well-calibrated.
- **Interpretability:** standardized LR coefficients rank churn drivers (short
  tenure, month-to-month, fiber-optic raise churn; two-year contracts protect).
- **SHAP** gives genuine *per-customer* attributions for the ACT list
  ("low tenure, no partner, electronic check") — replacing hand-written rules.
- **Profit-maximizing threshold:** a probability isn't a decision until you pick
  a cutoff. Backtesting on holdout labels, the value-maximizing cutoff is **~6%**
  (act on anyone above 6% churn risk), which beats the naive 0.5 cutoff by tens
  of thousands of dollars. (0.5 is arbitrary; 6% is where acting stops paying.)

---

## 7. The decision engine

Given a scored customer table + budget + capacity + strategy + save_rate:
1. **Risk bands** (operational, not statistical): HIGH ≥ 0.60, MEDIUM ≥ 0.30.
2. **Economic scoring:** net_retention_value per customer.
3. **Strategy weights:** Conservative / Balanced / Aggressive scale how much
   MEDIUM/LOW-risk customers are prioritized — strategy changes *ranking*, not
   *eligibility*.
4. **Constrained selection:** a **greedy knapsack** fills the budget by
   priority/efficiency, respecting the customer cap.
5. **Segmentation:** funded → **ACT**; positive-value but unfunded → **MONITOR**;
   negative-value → **IGNORE**.
6. **Explainability:** SHAP reason + a recommended-action playbook per ACT row.

**Greedy vs optimal (the ILP benchmark):** greedy is fast but not provably
optimal, so I solve the same problem *exactly* as a 0/1 knapsack with **PuLP**
and measure the gap. On this data greedy captures **≈100% of the optimum** — so
using the fast heuristic is an *evidenced trade-off*, not an assumption. (This is
the same "measure, don't assume" move as the model bake-off, applied to the
decision layer.)

---

## 8. Validation — proving the decisions are good

This is the project's strongest section, because the economics rest on
assumptions the dataset can't confirm (no intervention outcomes). Instead of
hiding that, I quantify it:

- **Baseline policy comparison:** at the same budget, the engine captures more
  expected value than *random*, *target-highest-churn*, and *target-highest-CLV*
  targeting — ~20% more than the best naive rule, and vastly more than chasing
  high-CLV customers (who often have low churn risk, so little revenue is
  actually at risk). This shows the *decision layer*, not just the model,
  creates value.
- **Break-even sensitivity:** the ACT set is chosen assuming a save rate; the
  **break-even save rate** (where the program stops paying for itself) is **~9%**
  vs the assumed 30% — a wide margin of safety, and the key assumption reframed
  as a quantified, falsifiable risk.
- **Monte Carlo decision quality:** 3,000 simulations draw the true save rate
  from a distribution and each customer's churn/save outcome as coin flips,
  turning the point ROI into a distribution: **mean ≈ 3.5×, 90% CI ≈ 1.4–5.8×,
  profitable in ~100% of scenarios.** Decisions under uncertainty, not a point
  estimate.
- **Stability:** how many ACT decisions survive a 10% cut to budget/capacity
  (high = safe to deploy); and *why* decisions change (dropped customers are
  lower-value/efficiency — rational, not thrashing).

---

## 9. The honest limitation: uplift (and how it's addressed)

**The critique a sharp interviewer will raise:** ranking by `churn × CLV` targets
people *likely to leave*, not people who can be *persuaded* to stay. Someone
certain to churn regardless is wasted budget; the right target is **uplift** —
the change in outcome *caused* by the intervention.

**Why it isn't in the Telco engine:** uplift needs experimental data (a treated
group and a control group). Telco has none, so any uplift number would be
invented — which would break the project's honesty.

**How it's demonstrated anyway:** `src/uplift.py` + `notebooks/02_uplift_hillstrom.ipynb`
run it on the **Hillstrom email A/B test** (64k customers randomly split into
email vs no email — a real experiment). A **two-model (T-learner)** uplift
estimator, evaluated with a **Qini curve**, shows ranking by uplift captures
**~7× more incremental responders** than ranking by response propensity
(Qini ≈ 1083 vs 148 vs 62 random). That gap is exactly the value the Telco engine
leaves on the table for want of experimental data — and the blueprint to close
it (run a randomized campaign, then rank by `uplift × CLV`).

---

## 10. Engineering & MLOps

- **51 pytest tests**, offline (network-dependent tests skip cleanly in CI).
- **CI (GitHub Actions):** ruff lint (blocking), mypy (advisory), pytest + coverage.
- **Single pipeline entrypoint** (`python -m src.pipeline`), serialized model
  (joblib), metrics artifact (JSON model card) the dashboard reads without
  retraining.
- **Self-bootstrapping deploy:** the dashboard/API run the pipeline on first
  launch if artifacts are missing — a fresh Streamlit Cloud deploy just works.
- **Pandera** schema validation as a data-quality gate on ingest.
- **Structured logging** (timestamps + levels).
- **Drift monitoring** (`src/monitoring.py`): PSI + KS per feature between a
  training reference and a current batch — the "is the model still valid?" alarm.
- **Dockerfile** for reproducible runs.
- **FastAPI** service (`POST /decisions`) + a **React** client — proving the
  engine is consumable outside Streamlit.

---

## 11. Numbers to memorize

| Thing | Value |
|---|---|
| Dataset | IBM Telco, 7,043 customers, 26.5% churn |
| Model | Logistic regression |
| ROC AUC | ~0.845 (5-fold CV, ±0.013) |
| Brier score | ~0.138 (LR already well-calibrated) |
| GBM (tuned) | ~0.844 AUC — no better than LR |
| CLV method | Cox proportional-hazards, concordance ~0.90 |
| Profit-max threshold | ~6% churn probability (vs naive 0.5) |
| Greedy vs optimal | greedy ≈ 100% of ILP optimum |
| Best naive rule beaten by | ~20% more value (policy comparison) |
| Break-even save rate | ~9% (assumed 30%) |
| Monte Carlo ROI | mean ~3.5×, 90% CI ~1.4–5.8×, ~100% profitable |
| Uplift (Hillstrom) | Qini 1083 (uplift) vs 148 (propensity) vs 62 (random) |
| Tests | 51 |

---

## 12. Likely interview questions (with strong answers)

**"Walk me through the project."** → Use the 30-second pitch, then the flow in §4.

**"Why logistic regression and not XGBoost?"** → The engine spends budget
proportional to predicted probabilities, so calibration > raw accuracy. I
verified with a 5-fold bake-off and Optuna-tuned GBM — LR wins on Brier and ties
on AUC, while staying interpretable. (§6)

**"How do you know your decisions are actually good?"** → I don't just assert it:
(a) the engine beats naive targeting baselines at the same budget, (b) Monte
Carlo shows it's profitable in ~100% of scenarios with a 90% CI, (c) the
break-even save rate (~9%) is far below my assumption (30%). (§8)

**"Your ROI depends on the save_rate assumption — isn't that a problem?"** →
Yes, and I make it explicit rather than hiding it: the break-even analysis and
Monte Carlo quantify exactly how much the conclusion depends on it, and it stays
profitable across the plausible range. (§5, §8)

**"You're targeting likely-churners, not persuadable ones."** → Correct — that's
the uplift critique, and I address it head-on: uplift needs experimental data
Telco lacks, so I demonstrate the correct method on the Hillstrom A/B test, where
uplift targeting captures ~7× more incremental responders. The engine's
architecture already supports swapping the score. (§9)

**"How is CLV computed — is it made up?"** → No: it's MRR × expected remaining
lifetime, and lifetime is estimated from a Cox survival model on the censored
tenure/churn data (concordance ~0.90), giving each customer an individualized
value. (§5)

**"Is the greedy selection optimal?"** → Not provably, so I benchmark it against
an exact ILP (PuLP knapsack) and measure the gap — greedy captures ~100% of the
optimum here, so it's a justified engineering trade-off. (§7)

**"What would you do next / what are the weaknesses?"** → See §13.

**"How would this run in production?"** → It already can: FastAPI endpoint,
Docker image, drift monitoring to catch when retraining is due, structured
logging, and CI. The missing piece for real value is a randomized retention
campaign to enable uplift modeling.

---

## 13. Honest limitations (say these before they're asked)

1. **Unvalidated economics.** No real intervention outcomes exist in Telco, so
   CLV, save_rate, and retention_cost are *plausible but unproven*. Everything
   downstream inherits that. (Mitigated by making assumptions explicit and
   quantifying sensitivity.)
2. **No persuadability/uplift in the live engine** — inherent to the data;
   demonstrated separately on Hillstrom.
3. **Standard benchmark model** — LR on Telco isn't novel; the *decision layer*
   is the contribution, not the model.
4. **Save_rate is a single global constant** — a real system would model it per
   segment or via uplift.
5. **Snapshot, not a living system** — drift monitoring exists, but there's no
   automated retraining trigger wired up.

Stating these first signals maturity; hiding them and getting caught is worse.

---

## 14. Concept glossary (be ready to define these)

- **Calibration / Brier score:** a model is *calibrated* if, among customers it
  says have 30% churn risk, ~30% actually churn. Brier score = mean squared error
  of predicted probabilities (lower = better calibrated). Matters because the
  engine multiplies probabilities by money.
- **ROC AUC:** probability the model ranks a random churner above a random
  non-churner. Measures *ranking*, not calibration.
- **Cox proportional hazards:** a survival model where each subject's hazard =
  a shared baseline hazard × exp(β·covariates). Gives per-customer survival
  curves from right-censored time-to-event data.
- **Concordance index:** the survival analog of AUC (ranking of who fails first).
- **SHAP:** game-theoretic per-prediction feature attributions — how much each
  feature pushed *this* customer's score up/down. Exact and fast for linear
  models.
- **ILP / 0-1 knapsack:** integer linear program that picks items (customers) to
  maximize value under a budget — the provably optimal version of the greedy
  selector.
- **Uplift / treatment effect:** P(outcome | treated) − P(outcome | control) for
  a given customer — the *causal* effect of the intervention, not the outcome
  rate itself.
- **Qini curve / coefficient:** uplift's evaluation curve — cumulative
  incremental responders as you target more of the population ranked by uplift;
  the area over random is the Qini coefficient.
- **PSI (Population Stability Index):** measures distribution shift of a feature
  between training and current data (>0.25 = significant drift).
- **Monte Carlo simulation:** repeatedly sampling uncertain inputs to turn a
  point estimate into a distribution (mean, CI, probability of an outcome).
- **T-learner (two-model uplift):** fit one outcome model on the treated group
  and one on the control group; uplift = difference of their predictions.

---

## 15. How to run it (for a live demo)

```bash
pip install -r requirements.txt
python -m src.pipeline          # downloads data, trains, saves model + metrics
python -m pytest tests -q        # 51 tests
python -m streamlit run app/dashboard.py   # dashboard
uvicorn api:app                  # REST API (docs at /docs); open frontend/index.html
python -m src.monitoring         # drift demo
```

The three dashboard tabs: **Decisions** (ACT list, ROI), **Strategy &
Robustness** (policy comparison, optimizer gap, break-even, Monte Carlo,
stability), **Model Performance** (calibration, feature importance, profit
threshold, bake-off).
