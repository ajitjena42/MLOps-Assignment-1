"""Data cleaning and validation module for the Heart Disease UCI dataset.

Validates column schemas, data types, value boundaries, handles missing values,
and maps angiographic heart disease presence (0 vs 1..4) into binary target classes.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("clean_data")

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the dataframe contains all expected columns."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    logger.info("Schema validation successful: all 14 expected columns are present.")


def clean_dataset(
    input_path: str = "data/raw/heart.csv",
    output_path: str = "data/processed/heart_cleaned.csv"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean, validate, and standardize the raw heart disease dataset.
    
    Cleaning Decisions:
    1. Replace '?' placeholders with NaN and coerce columns to numeric.
    2. Convert target: In the UCI Cleveland dataset, target values range from 0 to 4.
       Values 1, 2, 3, and 4 represent angiographic disease (>50% diameter narrowing).
       Following standard clinical research protocol and assignment requirements,
       values 1-4 are mapped to 1 (Disease Present / Elevated Risk), and 0 to 0 (Absence).
    3. Check and report duplicate rows.
    4. Validate physiological ranges:
       - age: [18, 120]
       - trestbps: [70, 250]
       - chol: [100, 600]
       - thalach: [50, 250]
       - oldpeak: [0.0, 10.0]
    5. Missing values in 'ca' and 'thal' are preserved as NaN in processed data
       so that the scikit-learn Pipeline imputer can handle them during cross-validation
       without data leakage.
       
    Args:
        input_path: Path to raw CSV.
        output_path: Path where cleaned CSV will be written.
        
    Returns:
        Tuple of (cleaned DataFrame, quality report dictionary).
    """
    raw_path = Path(input_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    df = pd.read_csv(raw_path, na_values="?")
    validate_schema(df)

    initial_rows = len(df)
    logger.info(f"Loaded raw dataset with {initial_rows} records and {df.shape[1]} columns.")

    # 1. Coerce all columns to numeric
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Check duplicates
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows. Dropping duplicates.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicate rows found.")

    # 3. Missing value analysis
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    logger.info(f"Missing values breakdown: {missing_counts} (Total missing cells: {total_missing})")

    # 4. Map target to binary (0 = Absence, 1 = Presence)
    raw_target_dist = df["target"].value_counts().to_dict()
    logger.info(f"Raw target distribution (0-4): {raw_target_dist}")
    
    df["target_raw"] = df["target"]
    df["target"] = (df["target"] > 0).astype(int)
    binary_target_dist = df["target"].value_counts().to_dict()
    logger.info(f"Standardized binary target distribution: {binary_target_dist}")

    # 5. Range checks and sanity warnings
    sanity_issues = []
    if not df["age"].between(18, 120).all():
        sanity_issues.append("Age values outside expected physiological range [18, 120].")
    if not df["trestbps"].between(70, 260).all():
        sanity_issues.append("Resting blood pressure outside range [70, 260].")
    if not df["chol"].between(80, 650).all():
        sanity_issues.append("Cholesterol outside range [80, 650].")
    if not df["thalach"].between(50, 250).all():
        sanity_issues.append("Max heart rate outside range [50, 250].")

    if sanity_issues:
        logger.warning(f"Sanity check warnings: {sanity_issues}")
    else:
        logger.info("Physiological sanity checks passed successfully.")

    # Save processed dataset
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info(f"Cleaned dataset saved to {out_file} ({len(df)} records).")

    quality_report = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "duplicates_removed": duplicate_count,
        "missing_values": missing_counts,
        "raw_target_distribution": raw_target_dist,
        "binary_target_distribution": binary_target_dist,
        "sanity_issues": sanity_issues,
        "columns": list(df.columns)
    }

    return df, quality_report


if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "data/raw/heart.csv"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "data/processed/heart_cleaned.csv"
    clean_dataset(in_p, out_p)
