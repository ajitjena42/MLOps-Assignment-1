"""Unit tests for model pipeline inference, predictions, and probability calibration."""

import numpy as np
import pytest

from src.models.evaluate import compute_metrics
from src.models.predict import MODEL_PATH, predict_batch, predict_single


@pytest.fixture
def valid_patient_payload():
    """Valid diagnostic record for a test patient."""
    return {
        "age": 58.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 150.0,
        "chol": 270.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 111.0,
        "exang": 1.0,
        "oldpeak": 0.8,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 7.0
    }


def test_model_artifact_exists():
    """Verify that trained model pipeline artifact is present on disk."""
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"


def test_predict_single_structure(valid_patient_payload):
    """Verify single patient prediction output schema, value types, and bounds."""
    res = predict_single(valid_patient_payload)

    assert "prediction" in res
    assert "label" in res
    assert "probability_disease" in res
    assert "confidence" in res
    assert "probabilities" in res

    assert res["prediction"] in (0, 1)
    assert isinstance(res["label"], str)
    assert 0.0 <= res["probability_disease"] <= 1.0
    assert 0.5 <= res["confidence"] <= 1.0

    # Probabilities must sum to ~1.0
    probs = res["probabilities"]
    assert pytest.approx(probs["absence"] + probs["presence"], abs=1e-3) == 1.0


def test_predict_single_with_missing_fields():
    """Verify pipeline gracefully handles missing fields via imputer without crashing."""
    incomplete_payload = {
        "age": 62.0,
        "sex": 0.0,
        "cp": 2.0
        # remaining features intentionally omitted
    }
    res = predict_single(incomplete_payload)
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["probability_disease"] <= 1.0


def test_predict_batch(valid_patient_payload):
    """Verify batch inference handles multiple records correctly."""
    batch = [valid_patient_payload, valid_patient_payload]
    results = predict_batch(batch)

    assert len(results) == 2
    for r in results:
        assert r["prediction"] in (0, 1)
        assert 0.0 <= r["probability_disease"] <= 1.0


def test_compute_metrics():
    """Verify calculation of metrics with known confusion matrix."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.4, 0.9])

    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 0.5
    assert m["true_positives"] == 1
    assert m["true_negatives"] == 1
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["roc_auc"] is not None
