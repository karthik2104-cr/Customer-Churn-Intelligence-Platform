"""Leakage-safe preprocessing pipelines.

Imputers, encoders, and scalers live inside the pipeline and are fit only when
the pipeline is fit. Callers must fit on training rows only.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("cat", Pipeline(categorical_steps), list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def make_model_pipeline(estimator, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric)),
            ("model", estimator),
        ]
    )
