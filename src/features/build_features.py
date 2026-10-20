"""Feature engineering and preprocessing pipeline construction.

Constructs scikit-learn ColumnTransformer pipelines combining imputation,
standardization, and one-hot encoding without data leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_FEATURES: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ALL_FEATURE_COLUMNS: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def get_preprocessor() -> ColumnTransformer:
    """Build a reusable ColumnTransformer for numerical scaling and categorical encoding.

    Returns:
        Fitted or unfitted ColumnTransformer object.
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve output feature names from a fitted ColumnTransformer."""
    feature_names = []
    # Numerical names
    feature_names.extend(NUMERICAL_FEATURES)
    # One-hot encoded names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(cat_feature_names))
    return feature_names


def prepare_train_test_data(
    df: pd.DataFrame, target_col: str = "target", test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into stratified training and testing sets.

    Args:
        df: Input cleaned dataframe.
        target_col: Name of binary target column.
        test_size: Fraction reserved for held-out evaluation (default: 0.2).
        random_state: Random seed for deterministic reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Exclude non-feature columns
    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
