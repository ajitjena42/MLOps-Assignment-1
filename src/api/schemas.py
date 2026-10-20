"""Pydantic data validation schemas for HeartGuard — An Observable ML Pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PatientData(BaseModel):
    """Clinical features schema for a patient undergoing cardiovascular risk assessment."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    age: float = Field(..., ge=18.0, le=110.0, description="Age in years (18-110)")
    sex: float = Field(..., ge=0.0, le=1.0, description="Biological sex (1.0 = Male, 0.0 = Female)")
    cp: float = Field(..., ge=0.0, le=4.0, description="Chest pain type (0-4 or 1-4)")
    trestbps: float = Field(..., ge=70.0, le=260.0, description="Resting blood pressure in mm Hg (70-260)")
    chol: float = Field(..., ge=80.0, le=650.0, description="Serum cholesterol in mg/dl (80-650)")
    fbs: float = Field(..., ge=0.0, le=1.0, description="Fasting blood sugar > 120 mg/dl (1.0 = True, 0.0 = False)")
    restecg: float = Field(..., ge=0.0, le=2.0, description="Resting ECG results (0.0 = Normal, 1.0 = ST-T, 2.0 = LVH)")
    thalach: float = Field(..., ge=50.0, le=240.0, description="Maximum heart rate achieved in bpm (50-240)")
    exang: float = Field(..., ge=0.0, le=1.0, description="Exercise-induced angina (1.0 = Yes, 0.0 = No)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise (0.0 - 10.0)")
    slope: float = Field(..., ge=0.0, le=3.0, description="Peak exercise ST slope (0-3)")
    ca: float = Field(..., ge=0.0, le=3.0, description="Number of major vessels colored by fluoroscopy (0.0 - 3.0)")
    thal: float = Field(..., ge=0.0, le=7.0, description="Thalassemia status (3 = normal, 6 = fixed, 7 = reversible)")


class BatchPatientData(BaseModel):
    """Payload container for batch predictions."""

    patients: list[PatientData] = Field(..., min_length=1, max_length=1000)


class Probabilities(BaseModel):
    absence: float = Field(..., description="Estimated probability of absence of disease (Class 0)")
    presence: float = Field(..., description="Estimated probability of presence of disease (Class 1)")


class PredictionResponse(BaseModel):
    """Individual prediction response with confidence and latency metrics."""

    prediction: int = Field(..., description="Binary prediction: 0 (Absence) or 1 (Presence)")
    label: str = Field(..., description="Human-readable diagnosis label")
    probability_disease: float = Field(..., description="Calibrated risk probability of heart disease [0.0 - 1.0]")
    confidence: float = Field(..., description="Confidence score associated with top predicted class [0.5 - 1.0]")
    probabilities: Probabilities
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds")
    model_version: str = Field(default="1.0.0", description="Active deployed model pipeline version")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_records: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    environment: str
    uptime_seconds: float
    total_predictions_served: int


class DriftEvaluationRequest(BaseModel):
    """Batch payload submitted for covariate and statistical feature drift evaluation."""

    records: list[PatientData] = Field(
        ..., min_length=1, max_length=2000, description="Batch records to evaluate against training baseline"
    )


class DriftReportResponse(BaseModel):
    """Statistical drift evaluation report."""

    drift_detected: bool = Field(..., description="True if any feature exceeds significance drift threshold")
    sample_size: int = Field(..., description="Number of records evaluated in current batch")
    total_features_evaluated: int
    drifted_features_count: int
    drifted_features: list[str]
    drift_ratio: float
    baseline_reference: str
    feature_metrics: dict[str, Any]
