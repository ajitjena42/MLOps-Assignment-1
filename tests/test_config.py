"""Unit tests for centralized configuration and path portability."""

from src.config import PROJECT_ROOT, Settings, settings


def test_default_settings():
    """Verify default settings reflect project and academic metadata."""
    assert settings.PROJECT_NAME == "HeartGuard — An Observable ML Pipeline"
    assert settings.STUDENT_ID == "2025AE05839"
    assert settings.COURSE == "AIMLCZG523 — Machine Learning Operations (MLOps)"
    assert settings.PORT == 8000
    assert settings.MODEL_VERSION == "1.0.0"


def test_portable_paths_exist():
    """Verify paths are resolved dynamically without hardcoded machine prefixes."""
    assert settings.DATA_RAW_PATH.is_absolute()
    assert settings.DATA_CLEAN_PATH.is_absolute()
    assert str(PROJECT_ROOT) in str(settings.MODEL_PATH)
    assert settings.MODEL_PATH.exists() or (PROJECT_ROOT / "models").exists()


def test_environment_override(monkeypatch):
    """Verify environment variables correctly override configuration attributes."""
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("MODEL_VERSION", "2.0.0")

    custom_settings = Settings()
    assert custom_settings.PORT == 9000
    assert custom_settings.ENVIRONMENT == "staging"
    assert custom_settings.MODEL_VERSION == "2.0.0"
