"""Central configuration: paths, dataset source, and economic assumptions."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "telco_churn.csv"
DB_PATH = BASE_DIR / "data" / "db" / "retention.db"
MODEL_PATH = BASE_DIR / "data" / "models" / "churn_model.joblib"
METRICS_PATH = BASE_DIR / "data" / "models" / "metrics.json"

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# Hillstrom email A/B-test dataset (has treatment/control) for uplift modeling.
HILLSTROM_PATH = BASE_DIR / "data" / "raw" / "hillstrom.csv"
HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

# --------------------------------------------------
# Economic assumptions (see docs/economic_assumptions.md)
# --------------------------------------------------
# CLV forward horizon: how many months ahead expected remaining lifetime is
# accumulated. Expected remaining life is derived from a survival model, not a
# hand-tuned formula.
CLV_HORIZON_MONTHS = 60

# CLV survival method: "cox" (per-customer Cox proportional-hazards, uses all
# covariates) or "km" (per-contract Kaplan-Meier). Cox is the richer model;
# economics falls back to KM if Cox is unavailable or the sample is too small.
CLV_METHOD = "cox"
COX_NUMERIC_COVARIATES = ["MonthlyCharges", "TotalCharges"]
COX_CATEGORICAL_COVARIATES = [
    "Contract", "InternetService", "PaymentMethod",
    "PaperlessBilling", "Partner", "Dependents",
]

# Retention offer: a win-back discount of DISCOUNT_RATE off MRR for
# OFFER_MONTHS, plus a contract-dependent outreach cost.
DISCOUNT_RATE = 0.30
OFFER_MONTHS = 3

# Probability that an intervention actually saves an at-risk customer.
# Interventions are not guaranteed to work; industry retention-offer
# acceptance rates are typically 20-40%.
SAVE_RATE = 0.30
OUTREACH_COST = {
    "Month-to-month": 10.0,
    "One year": 25.0,
    "Two year": 40.0,
}
