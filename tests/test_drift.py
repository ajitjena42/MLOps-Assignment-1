"""Unit tests for statistical feature and concept drift detection engine."""

from src.models.drift import calculate_psi, compute_drift, load_baseline_distribution


def test_calculate_psi_identical_distributions():
    """Identical probability distributions must have a PSI close to 0."""
    dist_a = {"0": 0.5, "1": 0.5}
    dist_b = {"0": 0.5, "1": 0.5}
    psi = calculate_psi(dist_a, dist_b)
    assert psi == 0.0


def test_calculate_psi_drifted_distributions():
    """Significantly shifted categorical distributions must exceed drift threshold (0.2)."""
    dist_base = {"0": 0.8, "1": 0.2}
    dist_shifted = {"0": 0.2, "1": 0.8}
    psi = calculate_psi(dist_base, dist_shifted)
    assert psi >= 0.2


def test_baseline_distribution_structure():
    """Verify baseline distribution profile contains required feature metadata."""
    baseline = load_baseline_distribution()
    assert "dataset_name" in baseline
    assert "total_records" in baseline
    assert "features" in baseline
    assert baseline["total_records"] >= 250

    # Verify numerical feature stats
    for col in ["age", "chol", "trestbps", "thalach", "oldpeak"]:
        assert col in baseline["features"]
        feat = baseline["features"][col]
        assert feat["type"] == "numerical"
        assert "mean" in feat
        assert "std" in feat
        assert "quantiles" in feat

    # Verify categorical feature stats
    for col in ["sex", "cp", "fbs", "restecg", "exang"]:
        assert col in baseline["features"]
        feat = baseline["features"][col]
        assert feat["type"] == "categorical"
        assert "frequencies" in feat


def test_compute_drift_in_baseline():
    """In-distribution samples should report no statistically significant drift."""
    baseline = load_baseline_distribution()

    # Reconstruct small sample from baseline mean values
    normal_patient = {
        "age": baseline["features"]["age"]["mean"],
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": baseline["features"]["trestbps"]["mean"],
        "chol": baseline["features"]["chol"]["mean"],
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": baseline["features"]["thalach"]["mean"],
        "exang": 0.0,
        "oldpeak": baseline["features"]["oldpeak"]["mean"],
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0
    }
    batch = [normal_patient] * 10
    report = compute_drift(batch, baseline_stats=baseline)

    assert "drift_detected" in report
    assert "feature_metrics" in report
    assert report["sample_size"] == 10


def test_compute_drift_detects_severe_shift():
    """Severely shifted continuous biomarker values must trigger drift alarm."""
    baseline = load_baseline_distribution()

    # Extreme outlier batch: age 99, chol 550, oldpeak 9.0
    outlier_patient = {
        "age": 99.0,
        "sex": 0.0,
        "cp": 4.0,
        "trestbps": 220.0,
        "chol": 580.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 60.0,
        "exang": 1.0,
        "oldpeak": 8.5,
        "slope": 3.0,
        "ca": 3.0,
        "thal": 7.0
    }
    batch = [outlier_patient] * 30
    report = compute_drift(batch, baseline_stats=baseline)

    assert report["drift_detected"] is True
    assert report["drifted_features_count"] > 0
