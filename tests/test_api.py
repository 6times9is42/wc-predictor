import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_match_endpoint_rejects_same_team():
    response = client.get("/predictions/match", params={"home": "Spain", "away": "Spain"})
    assert response.status_code == 400
    assert "different teams" in response.json()["detail"]


def test_simulate_endpoint(monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.simulate.run_simulation_task",
        lambda job_id, n_simulations: None,
    )
    response = client.post("/simulate", json={"n_simulations": 10})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_simulate_status():
    response = client.get("/simulate/status/invalid-id")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"
