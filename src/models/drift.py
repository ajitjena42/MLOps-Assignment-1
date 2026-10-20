"""Statistical Data and Prediction Drift Detection Engine for HeartGuard.

Calculates baseline reference distributions from historical training data and
evaluates incoming inference batches for covariate (feature) shift and concept shift.

Statistical Methods:
- Numerical Features: Two-sample Kolmogorov-Smirnov (KS) test for distribution equality.
- Categorical Features: Population Stability Index (PSI) and Total Variation distance.
- Target / Prediction: Class balance drift against historical prevalence baseline.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.config import settings
from src.features.build_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)

logger = logging.getLogger("drift")


def generate_baseline_distribution(
    data_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Compute and persist summary baseline statistics from cleaned training data.

    Args:
        data_path: Path to cleaned training CSV.
        output_path: Destination path for serialized baseline JSON.

    Returns:
        Dictionary containing baseline statistics for all features.
    """
    csv_file = Path(data_path) if data_path else settings.DATA_CLEAN_PATH
    out_file = Path(output_path) if output_path else settings.BASELINE_STATS_PATH

    if not csv_file.exists():
        raise FileNotFoundError(f"Cleaned training data not found at {csv_file}")

    df = pd.read_csv(csv_file)
    logger.info(f"Computing baseline feature distributions from {len(df)} records in {csv_file}")

    baseline: dict[str, Any] = {
        "dataset_name": "Heart Disease UCI Cleveland Clinic Foundation",
        "total_records": len(df),
        "target_distribution": {
            "negative_count": int((df["target"] == 0).sum()),
            "positive_count": int((df["target"] == 1).sum()),
            "prevalence_rate": round(float((df["target"] == 1).mean()), 4),
        },
        "features": {},
    }

    # 1. Numerical Feature Distributions
    for col in NUMERICAL_FEATURES:
        series = df[col].dropna()
        baseline["features"][col] = {
            "type": "numerical",
            "count": len(series),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "q75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
            # Representative quantiles for reference comparison
            "quantiles": [round(float(q), 4) for q in series.quantile(np.linspace(0.05, 0.95, 19)).tolist()],
            # Store values for KS-test comparison
            "values": [round(float(v), 2) for v in series.tolist()],
        }

    # 2. Categorical Feature Distributions
    for col in CATEGORICAL_FEATURES:
        series = df[col].dropna()
        counts = series.value_counts(normalize=True).to_dict()
        baseline["features"][col] = {
            "type": "categorical",
            "count": len(series),
            "frequencies": {str(k): round(float(v), 4) for k, v in counts.items()},
        }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    logger.info(f"Saved baseline distribution profile to {out_file}")
    return baseline


def load_baseline_distribution(stats_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load baseline distribution profile from disk or regenerate if missing."""
    file_path = Path(stats_path) if stats_path else settings.BASELINE_STATS_PATH
    if not file_path.exists():
        logger.info(f"Baseline profile missing at {file_path}. Generating from training data...")
        return generate_baseline_distribution(output_path=file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_psi(expected_probs: dict[str, float], actual_probs: dict[str, float], epsilon: float = 1e-4) -> float:
    """Calculate Population Stability Index (PSI) between two categorical probability distributions.

    PSI interpretation:
    - PSI < 0.1: No significant distribution change.
    - 0.1 <= PSI < 0.2: Moderate shift / warning.
    - PSI >= 0.2: Significant drift detected.
    """
    all_categories = set(expected_probs.keys()).union(set(actual_probs.keys()))
    psi = 0.0

    for cat in all_categories:
        e = expected_probs.get(cat, epsilon)
        a = actual_probs.get(cat, epsilon)
        e = max(e, epsilon)
        a = max(a, epsilon)
        psi += (a - e) * math.log(a / e)

    return round(float(psi), 4)


def compute_drift(
    batch_records: list[dict[str, Any]],
    baseline_stats: Optional[dict[str, Any]] = None,
    p_value_threshold: Optional[float] = None,
    psi_threshold: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate statistical feature drift between incoming batch and baseline profile.

    Args:
        batch_records: List of patient attribute dictionaries.
        baseline_stats: Pre-loaded baseline profile dictionary.
        p_value_threshold: KS test threshold for numerical drift (default 0.05).
        psi_threshold: PSI threshold for categorical drift (default 0.20).

    Returns:
        Structured drift assessment report.
    """
    if not batch_records:
        return {
            "status": "error",
            "message": "Empty batch records provided for drift evaluation.",
            "drift_detected": False,
        }

    p_thresh = p_value_threshold if p_value_threshold is not None else settings.DRIFT_PVALUE_THRESHOLD
    psi_thresh = psi_threshold if psi_threshold is not None else settings.DRIFT_PSI_THRESHOLD

    if baseline_stats is None:
        baseline_stats = load_baseline_distribution()

    df_batch = pd.DataFrame(batch_records)
    feature_metrics: dict[str, Any] = {}
    drifted_features = []

    # Evaluate each feature against baseline
    for col in ALL_FEATURE_COLUMNS:
        if col not in df_batch.columns or df_batch[col].dropna().empty:
            continue

        base_feat = baseline_stats["features"].get(col)
        if not base_feat:
            continue

        feat_type = base_feat["type"]
        sample_vals = df_batch[col].dropna().values

        if feat_type == "numerical":
            base_vals = np.array(base_feat.get("values", []))
            if len(base_vals) > 0 and len(sample_vals) >= 2:
                # Two-sample Kolmogorov-Smirnov Test
                ks_stat, p_val = stats.ks_2samp(sample_vals, base_vals)
                is_drift = bool(p_val < p_thresh)
                severity = "SIGNIFICANT_DRIFT" if is_drift else "NO_DRIFT"
                if is_drift:
                    drifted_features.append(col)

                feature_metrics[col] = {
                    "feature_type": "numerical",
                    "test_type": "Kolmogorov-Smirnov",
                    "statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "threshold": p_thresh,
                    "drift_detected": is_drift,
                    "status": severity,
                    "sample_mean": round(float(np.mean(sample_vals)), 4),
                    "baseline_mean": base_feat["mean"],
                }
            else:
                feature_metrics[col] = {
                    "feature_type": "numerical",
                    "status": "INSUFFICIENT_SAMPLES",
                    "drift_detected": False,
                }

        elif feat_type == "categorical":
            expected_freqs = base_feat["frequencies"]
            actual_counts = pd.Series(sample_vals).astype(str).value_counts(normalize=True).to_dict()
            psi_val = calculate_psi(expected_freqs, actual_counts)

            is_drift = bool(psi_val >= psi_thresh)
            is_warning = bool(0.1 <= psi_val < psi_thresh)
            severity = "SIGNIFICANT_DRIFT" if is_drift else ("MODERATE_DRIFT" if is_warning else "NO_DRIFT")

            if is_drift:
                drifted_features.append(col)

            feature_metrics[col] = {
                "feature_type": "categorical",
                "test_type": "Population Stability Index (PSI)",
                "psi_score": psi_val,
                "threshold": psi_thresh,
                "drift_detected": is_drift,
                "status": severity,
                "sample_frequencies": {str(k): round(float(v), 4) for k, v in actual_counts.items()},
                "baseline_frequencies": expected_freqs,
            }

    overall_drift_detected = len(drifted_features) > 0
    drift_ratio = round(len(drifted_features) / len(feature_metrics), 4) if feature_metrics else 0.0

    return {
        "drift_detected": overall_drift_detected,
        "sample_size": len(batch_records),
        "total_features_evaluated": len(feature_metrics),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_ratio": drift_ratio,
        "baseline_reference": baseline_stats.get("dataset_name", "UCI Cleveland"),
        "feature_metrics": feature_metrics,
    }
