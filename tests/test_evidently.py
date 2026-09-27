"""
Unit Tests for Evidently AI Drift Monitoring Service.
"""
from __future__ import annotations

import pytest
import numpy as np
from fastapi.testclient import TestClient

from evidently_service.main import app, store

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_reference():
    """Seed test reference dataset into in-memory store without modifying disk."""
    import pandas as pd
    np.random.seed(42)
    ref_dict = {
        "f1": np.random.normal(10.0, 1.0, 60),
        "f2": np.random.normal(5.0, 0.5, 60),
        "quality": np.random.choice([5, 6, 7], 60)
    }
    store.reference_data = pd.DataFrame(ref_dict)


def test_evidently_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["reference_loaded"] is True


def test_capture_and_analyze():
    # Capture 40 samples from similar distribution
    for _ in range(40):
        client.post("/capture", json={
            "features": {
                "f1": float(np.random.normal(10.0, 1.0)),
                "f2": float(np.random.normal(5.0, 0.5))
            },
            "prediction": 1
        })

    # Trigger analysis
    resp = client.post("/analyze", json={"window_size": 40, "threshold": 0.2})
    assert resp.status_code == 200
    res = resp.json()
    assert "dataset_drift_detected" in res
    assert "drift_share" in res
    assert "html_report" in res


def test_evidently_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "evidently_drift_score" in resp.text
