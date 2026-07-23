"""FastAPI service exposing the retention decision engine as a backend.

Proves the "framework-agnostic, deployable as a service" claim: this reuses the
exact decision engine (`app.core`) with zero Streamlit involvement. Any system
(a CRM, a scheduled job) can POST constraints and get back the ranked action
list as JSON.

Run:  python -m src.pipeline   # once, to build the model/DB
      uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core import load_and_compute_decisions
from src.config import DB_PATH, MODEL_PATH, SAVE_RATE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Self-bootstrap: build the model/DB on first launch if they're missing.
    if not (MODEL_PATH.exists() and DB_PATH.exists()):
        from src.pipeline import run_pipeline
        run_pipeline()
    yield


app = FastAPI(
    title="Decision-Driven Customer Retention API",
    version="1.0.0",
    summary="Turns churn predictions into budget-constrained retention actions.",
    lifespan=lifespan,
)

# Allow the static React frontend (served from file:// or any host) to call it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecisionRequest(BaseModel):
    budget: float = Field(25_000, gt=0, description="Total retention budget ($)")
    max_customers: int = Field(300, gt=0, description="Max customers to act on")
    strategy: Literal["Conservative", "Balanced", "Aggressive"] = "Balanced"
    save_rate: float = Field(
        SAVE_RATE, gt=0, le=1, description="Assumed intervention success probability"
    )


class ActCustomer(BaseModel):
    customer_id: str
    risk_band: str
    churn_probability: float
    clv: float
    retention_cost: float
    net_retention_value: float
    recommended_action: str


class DecisionResponse(BaseModel):
    act_count: int
    budget_used: float
    expected_revenue_saved: float
    roi: float
    act_customers: list[ActCustomer]


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/decisions", response_model=DecisionResponse)
def decisions(req: DecisionRequest):
    """Rank the customers worth acting on under the given budget and strategy."""
    final_df, _, spent, _ = load_and_compute_decisions(
        req.budget, req.max_customers, req.strategy, req.save_rate
    )
    act = final_df[final_df["action_segment"] == "ACT"].sort_values(
        "net_retention_value", ascending=False
    )
    saved = float(act["net_retention_value"].sum())

    customers = [
        ActCustomer(
            customer_id=str(r.customer_id),
            risk_band=str(r.risk_band),
            churn_probability=round(float(r.churn_probability), 3),
            clv=round(float(r.CLV), 2),
            retention_cost=round(float(r.retention_cost), 2),
            net_retention_value=round(float(r.net_retention_value), 2),
            recommended_action=str(r.recommended_action),
        )
        for r in act.itertuples()
    ]
    return DecisionResponse(
        act_count=len(act),
        budget_used=round(float(spent), 2),
        expected_revenue_saved=round(saved, 2),
        roi=round(saved / max(spent, 1.0), 2),
        act_customers=customers,
    )
