"""Centralized configuration module for HeartGuard — An Observable ML Pipeline.

Loads settings from environment variables with sensible defaults and provides
portable path resolutions for macOS, Linux, Docker, and Kubernetes environments.
"""

import os
from pathlib import Path

# Dynamically resolve project root: 2 levels up from src/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings:
    """Application and pipeline configuration parameters."""

    def __init__(self):
        # Project & Academic Metadata
        self.PROJECT_NAME: str = "HeartGuard — An Observable ML Pipeline"
        self.STUDENT_ID: str = "2025AE05839"
        self.COURSE: str = "AIMLCZG523 — Machine Learning Operations (MLOps)"
        self.INSTITUTION: str = "Birla Institute of Technology & Science (BITS), Pilani"

        # Environment & Serving
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.HOST: str = os.getenv("APP_HOST", os.getenv("HOST", "0.0.0.0"))
        self.PORT: int = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
        self.WORKERS: int = int(os.getenv("WORKERS", "2"))
        self.MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")

        # Storage Paths (Portable & Overridable)
        self.DATA_RAW_PATH: Path = PROJECT_ROOT / os.getenv("DATA_RAW_PATH", "data/raw/heart.csv")
        self.DATA_CLEAN_PATH: Path = PROJECT_ROOT / os.getenv("DATA_CLEAN_PATH", "data/processed/heart_cleaned.csv")
        self.MODEL_PATH: Path = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/model_pipeline.joblib")
        self.BASELINE_STATS_PATH: Path = PROJECT_ROOT / os.getenv(
            "BASELINE_STATS_PATH", "models/baseline_distribution.json"
        )
        self.REPORTS_DIR: Path = PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports")
        self.FIGURES_DIR: Path = PROJECT_ROOT / os.getenv("FIGURES_DIR", "reports/figures")

        # MLflow Governance
        self.MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}")
        self.MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "heart-disease-classification")

        # Observability & Monitoring
        self.PROMETHEUS_METRICS_ENABLED: bool = os.getenv("PROMETHEUS_METRICS_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.DRIFT_PVALUE_THRESHOLD: float = float(os.getenv("DRIFT_PVALUE_THRESHOLD", "0.05"))
        self.DRIFT_PSI_THRESHOLD: float = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))

    def get_model_path(self) -> Path:
        """Resolve model path, respecting environment overrides."""
        env_path = os.getenv("MODEL_PATH")
        if env_path:
            p = Path(env_path)
            return p if p.is_absolute() else PROJECT_ROOT / p
        return self.MODEL_PATH


settings = Settings()
