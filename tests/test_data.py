"""Unit tests for data acquisition, cleaning, and schema validation."""

import pandas as pd
import pytest

from src.data.clean_data import EXPECTED_COLUMNS, clean_dataset, validate_schema


def test_schema_validation_success():
    """Verify that dataframe with all 14 expected columns passes validation."""
    data = {col: [1.0] for col in EXPECTED_COLUMNS}
    df = pd.DataFrame(data)
    # Should not raise exception
    validate_schema(df)


def test_schema_validation_missing_column():
    """Verify that dataframe missing required column raises ValueError."""
    data = {col: [1.0] for col in EXPECTED_COLUMNS if col != "target"}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df)


def test_clean_dataset_binary_target(tmp_path):
    """Verify that multi-class target (0-4) is mapped strictly to binary (0, 1)."""
    raw_csv = tmp_path / "test_raw.csv"
    sample_data = {
        "age": [50.0, 60.0, 70.0, 45.0, 55.0],
        "sex": [1.0, 0.0, 1.0, 0.0, 1.0],
        "cp": [1.0, 2.0, 3.0, 4.0, 1.0],
        "trestbps": [120.0, 140.0, 160.0, 110.0, 130.0],
        "chol": [200.0, 250.0, 300.0, 180.0, 220.0],
        "fbs": [0.0, 1.0, 0.0, 0.0, 1.0],
        "restecg": [0.0, 1.0, 2.0, 0.0, 1.0],
        "thalach": [150.0, 130.0, 110.0, 170.0, 140.0],
        "exang": [0.0, 1.0, 1.0, 0.0, 0.0],
        "oldpeak": [1.0, 2.0, 3.0, 0.0, 0.5],
        "slope": [1.0, 2.0, 3.0, 1.0, 2.0],
        "ca": [0.0, 1.0, "?", 0.0, 2.0],
        "thal": [3.0, "?", 7.0, 3.0, 6.0],
        "target": [0, 1, 2, 3, 4],  # test all target values
    }
    pd.DataFrame(sample_data).to_csv(raw_csv, index=False)
    out_csv = tmp_path / "test_cleaned.csv"

    cleaned_df, report = clean_dataset(str(raw_csv), str(out_csv))

    assert len(cleaned_df) == 5
    # Binary targets: 0 should be 0; 1, 2, 3, 4 should all map to 1
    assert list(cleaned_df["target"]) == [0, 1, 1, 1, 1]
    assert set(cleaned_df["target"].unique()).issubset({0, 1})
    assert report["missing_values"]["ca"] == 1
    assert report["missing_values"]["thal"] == 1


def test_clean_dataset_duplicate_handling(tmp_path):
    """Verify that duplicate rows are deduplicated."""
    raw_csv = tmp_path / "test_dup.csv"
    row = {col: 1.0 for col in EXPECTED_COLUMNS}
    row["target"] = 0
    df = pd.DataFrame([row, row])  # duplicate identical rows
    df.to_csv(raw_csv, index=False)
    out_csv = tmp_path / "test_cleaned_dup.csv"

    cleaned_df, report = clean_dataset(str(raw_csv), str(out_csv))
    assert len(cleaned_df) == 1
    assert report["duplicates_removed"] == 1
