"""API tests. Skipped when the trained model isn't present (e.g. fresh CI),
so the suite stays offline — run `python -m src.pipeline` first to exercise
these locally."""

import warnings

import pytest
from fastapi.testclient import TestClient

from src.config import DB_PATH, MODEL_PATH

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and DB_PATH.exists()),
    reason="needs a trained model/DB; run `python -m src.pipeline` first",
)


@pytest.fixture(scope="module")
def client():
    from api import app
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_decisions_returns_ranked_act_list(client):
    r = client.post("/decisions", json={
        "budget": 25000, "max_customers": 300,
        "strategy": "Balanced", "save_rate": 0.30,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["act_count"] > 0
    assert body["budget_used"] <= 25000 + 1e-6
    assert len(body["act_customers"]) == body["act_count"]
    # sorted by net value, descending
    nv = [c["net_retention_value"] for c in body["act_customers"]]
    assert nv == sorted(nv, reverse=True)


def test_invalid_strategy_rejected(client):
    r = client.post("/decisions", json={"strategy": "Reckless"})
    assert r.status_code == 422  # pydantic validation error
