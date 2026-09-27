"""
Unit Tests for FastAPI Serving Endpoints.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, FEATURE_NAMES

client = TestClient(app)


def test_index_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data
    assert data["status"] == "operational"


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["model_loaded"] is True


def test_single_prediction_valid():
    payload = {
        "features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
        "feature_names": FEATURE_NAMES
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in (0, 1)
    assert "latency_ms" in data
    assert "timestamp" in data


def test_single_prediction_invalid_features_count():
    # Only 5 features instead of 11
    payload = {
        "features": [7.4, 0.70, 0.00, 1.9, 0.076]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_batch_prediction():
    payload = {
        "samples": [
            [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
            [10.2, 0.35, 0.40, 2.2, 0.065, 15.0, 45.0, 0.9960, 3.20, 0.75, 11.5]
        ]
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "api_requests_total" in text
    assert "model_prediction" in text


def test_model_info_endpoint():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_name" in data
    assert len(data["features"]) == 11
