"""Unit tests for feature engineering, preprocessors, and train/test splitting."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    ALL_FEATURE_COLUMNS,
    get_feature_names,
    get_preprocessor,
    prepare_train_test_data,
)


@pytest.fixture
def sample_features_df():
    """Create reproducible sample dataframe for transformer tests."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "age": np.random.uniform(30, 75, n),
            "sex": np.random.choice([0.0, 1.0], n),
            "cp": np.random.choice([1.0, 2.0, 3.0, 4.0], n),
            "trestbps": np.random.uniform(90, 180, n),
            "chol": np.random.uniform(150, 400, n),
            "fbs": np.random.choice([0.0, 1.0], n),
            "restecg": np.random.choice([0.0, 1.0, 2.0], n),
            "thalach": np.random.uniform(80, 200, n),
            "exang": np.random.choice([0.0, 1.0], n),
            "oldpeak": np.random.uniform(0.0, 5.0, n),
            "slope": np.random.choice([1.0, 2.0, 3.0], n),
            "ca": np.random.choice([0.0, 1.0, 2.0, 3.0], n),
            "thal": np.random.choice([3.0, 6.0, 7.0], n),
            "target": np.random.choice([0, 1], n),
        }
    )


def test_preprocessor_fitting(sample_features_df):
    """Verify that ColumnTransformer fits without data leakage and produces numeric array."""
    preprocessor = get_preprocessor()
    X = sample_features_df[ALL_FEATURE_COLUMNS]
    transformed = preprocessor.fit_transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_features_df)
    # Output columns should be >= input features due to one-hot encoding
    assert transformed.shape[1] >= len(ALL_FEATURE_COLUMNS)
    assert not np.isnan(transformed).any()


def test_preprocessor_feature_names(sample_features_df):
    """Verify that get_feature_names matches output column dimensions."""
    preprocessor = get_preprocessor()
    X = sample_features_df[ALL_FEATURE_COLUMNS]
    transformed = preprocessor.fit_transform(X)
    names = get_feature_names(preprocessor)

    assert len(names) == transformed.shape[1]
    assert all(isinstance(n, str) for n in names)


def test_prepare_train_test_data_stratification(sample_features_df):
    """Verify train_test_split preserves stratification and disjoint indices."""
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        sample_features_df, target_col="target", test_size=0.2, random_state=42
    )

    # Disjoint
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    # Expected sizes
    assert len(X_train) == 40
    assert len(X_test) == 10
    # Both classes present in both splits
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
