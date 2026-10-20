"""Integration tests for the HeartGuard FastAPI model serving endpoints and Prometheus metrics."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.config import settings

client = TestClient(app)


@pytest.fixture
def sample_patient_json():
    return {
        "age": 60.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 140.0,
        "chol": 260.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 120.0,
        "exang": 1.0,
        "oldpeak": 2.0,
        "slope": 2.0,
        "ca": 2.0,
        "thal": 7.0
    }


def test_root_endpoint():
    """Verify root endpoint provides API details, student attribution, and navigation endpoints."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["student_id"] == "2025AE05839"
    assert "HeartGuard" in data["service"]
    assert "endpoints" in data
    assert "drift" in data


def test_health_endpoint():
    """Verify health endpoint reports model loaded, environment, and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == settings.MODEL_VERSION
    assert "environment" in data


def test_metrics_endpoint():
    """Verify Prometheus scrape endpoint exposes custom metric names and drift indicators."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "heart_disease_api_requests_total" in content
    assert "heart_disease_predictions_total" in content
    assert "heart_disease_drift_detected" in content


def test_predict_endpoint_valid(sample_patient_json):
    """Verify /predict returns valid prediction, confidence, and latency."""
    response = client.post("/predict", json=sample_patient_json)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in (0, 1)
    assert isinstance(data["label"], str)
    assert 0.0 <= data["probability_disease"] <= 1.0
    assert 0.5 <= data["confidence"] <= 1.0
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0
    assert data["model_version"] == settings.MODEL_VERSION


def test_predict_endpoint_validation_error():
    """Verify invalid physiological values return HTTP 422 with validation errors."""
    invalid_payload = {
        "age": -10.0,  # Invalid: negative age
        "sex": 5.0,    # Invalid: not binary
        "cp": 1.0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_batch_predict_endpoint(sample_patient_json):
    """Verify batch inference scores all records and aggregates latency."""
    batch_payload = {
        "patients": [sample_patient_json, sample_patient_json]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["prediction"] in (0, 1)


def test_drift_get_status():
    """Verify GET /drift reports baseline metadata and active statistical thresholds."""
    response = client.get("/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "baseline_reference" in data
    assert "monitored_features" in data
    assert len(data["monitored_features"]) >= 13


def test_drift_post_evaluation(sample_patient_json):
    """Verify POST /drift evaluates batch patient records against training baseline."""
    batch_payload = {
        "records": [sample_patient_json] * 5
    }
    response = client.post("/drift", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "total_features_evaluated" in data
    assert "feature_metrics" in data
    assert data["sample_size"] == 5


def test_drift_post_empty_records():
    """Verify empty drift batch returns HTTP 422 validation error."""
    response = client.post("/drift", json={"records": []})
    assert response.status_code == 422
