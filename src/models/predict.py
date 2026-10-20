"""Model inference service module for HeartGuard — An Observable ML Pipeline.

Loads the fitted pipeline (preprocessing + estimator) and serves predictions
for individual dictionaries, batch records, or pandas DataFrames with sub-millisecond execution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.features.build_features import ALL_FEATURE_COLUMNS

logger = logging.getLogger("predict")

MODEL_PATH = settings.get_model_path()
_MODEL = None


def load_model(model_path: Optional[Union[str, Path]] = None, reload: bool = False):
    """Load and cache the trained pipeline across joblib, pickle, or MLflow formats.

    Args:
        model_path: Optional path to serialized model artifact.
        reload: If True, forces reloading artifact from disk into memory.
    """
    global _MODEL
    path = Path(model_path) if model_path else settings.get_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Fitted model artifact not found at {path.resolve()}. "
            f"Ensure training has been executed via 'python src/models/train.py'."
        )

    if _MODEL is None or reload:
        logger.info(f"Loading production pipeline from {path.resolve()}")
        if path.is_dir() and (path / "MLmodel").exists():
            import mlflow.pyfunc

            _MODEL = mlflow.pyfunc.load_model(str(path))
        elif str(path).endswith(".pkl"):
            import pickle

            with open(path, "rb") as f:
                _MODEL = pickle.load(f)
        else:
            _MODEL = joblib.load(path)
    return _MODEL


def predict_single(
    data: dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
    pipeline: Optional[Any] = None,
) -> dict[str, Any]:
    """Execute prediction on a single patient record.

    Args:
        data: Dictionary of clinical features.
        model_path: Optional path to serialized joblib model pipeline.
        pipeline: Optional pre-loaded pipeline instance for zero-IO serving.

    Returns:
        Structured prediction dictionary with class, label, probability, confidence.
    """
    active_pipeline = pipeline if pipeline is not None else load_model(model_path)
    df = pd.DataFrame([data])

    # Ensure all expected feature columns are present; fill missing with NaN for imputer
    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[ALL_FEATURE_COLUMNS]

    # Predict
    pred = int(active_pipeline.predict(df)[0])
    probas = active_pipeline.predict_proba(df)[0]
    prob_disease = float(probas[1])
    confidence = float(np.max(probas))

    label = "Heart Disease Risk Detected" if pred == 1 else "No Heart Disease Detected"

    return {
        "prediction": pred,
        "label": label,
        "probability_disease": round(prob_disease, 4),
        "confidence": round(confidence, 4),
        "probabilities": {"absence": round(float(probas[0]), 4), "presence": round(float(probas[1]), 4)},
    }


def predict_batch(
    records: list[dict[str, Any]],
    model_path: Optional[Union[str, Path]] = None,
    pipeline: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Execute high-throughput batch predictions on a list of patient records.

    Args:
        records: List of patient attribute dictionaries.
        model_path: Optional path to serialized model pipeline.
        pipeline: Optional pre-loaded pipeline instance.
    """
    active_pipeline = pipeline if pipeline is not None else load_model(model_path)
    df = pd.DataFrame(records)

    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[ALL_FEATURE_COLUMNS]

    preds = active_pipeline.predict(df)
    probas = active_pipeline.predict_proba(df)

    results = []
    for pred, prob in zip(preds, probas):
        p_class = int(pred)
        prob_disease = float(prob[1])
        results.append(
            {
                "prediction": p_class,
                "label": "Heart Disease Risk Detected" if p_class == 1 else "No Heart Disease Detected",
                "probability_disease": round(prob_disease, 4),
                "confidence": round(float(np.max(prob)), 4),
                "probabilities": {"absence": round(float(prob[0]), 4), "presence": round(float(prob[1]), 4)},
            }
        )
    return results


if __name__ == "__main__":
    sample_input = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 0.0,
        "ca": 0.0,
        "thal": 1.0,
    }
    print(json.dumps(predict_single(sample_input), indent=2))
