"""API fixtures.

The model in these tests is fit on a synthetic table so the suite does not
download data or run the full search. test_training.py covers that search.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data.loader import clean_customers, filter_trainable
from ml.data.schema import MODEL_FEATURES, TARGET_COLUMN
from ml.data.synthetic import make_synthetic_frame, write_synthetic_csv
from ml.evaluation.metrics import binary_metrics
from ml.features.preprocessing import make_model_pipeline


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    csv_path = write_synthetic_csv(root / "synthetic_customers.csv", n=160, seed=11)
    raw = make_synthetic_frame(160, seed=11)
    cleaned, _notes = clean_customers(raw)
    trainable, _reasons = filter_trainable(cleaned)
    features = trainable.loc[:, list(MODEL_FEATURES)].copy()
    target = (trainable[TARGET_COLUMN] == "Yes").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=11
    )
    pipeline = make_model_pipeline(
        LogisticRegression(max_iter=400, solver="liblinear", class_weight="balanced", random_state=11),
        scale_numeric=True,
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(y_test, probabilities, 0.5)
    baselines = {}
    numeric = {}
    categorical = {}
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        values = x_train[column]
        baselines[column] = float(values.median())
        numeric[column] = {"min": float(values.min()), "max": float(values.max()), "median": float(values.median())}
    for column in MODEL_FEATURES:
        if column in numeric:
            continue
        baselines[column] = str(x_train[column].mode().iloc[0])
        categorical[column] = sorted(str(value) for value in trainable[column].dropna().unique())
    metadata = {
        "selected_model": "logistic_regression",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "selection_reason": "Synthetic fixture for API tests. Not an IBM telco evaluation.",
        "selection_metric": "average_precision",
        "comparison_scope": "fixture",
        "comparison": [],
        "excluded_features": ["gender", "SeniorCitizen"],
        "exclusion_reason": "Excluded from this fixture the same way the product model excludes them.",
        "feature_spec": {"numeric": numeric, "categorical": categorical},
        "baselines": baselines,
        "model_features": list(MODEL_FEATURES),
        "thresholds": {
            "medium": 0.30,
            "high": 0.50,
            "decision": 0.50,
            "rationale": "Fixture cutoffs, not validation-tuned thresholds.",
        },
        "feature_importance": {
            "method": "grouped_permutation",
            "split": "internal_validation",
            "used_for_model_selection": False,
            "features": [{"feature": "Contract", "importance": 0.0, "std": 0.0}],
            "note": "Placeholder ranking for the fixture. The training script writes the real ranking.",
        },
        "associations": None,
        "test_metrics": metrics,
        "dataset": {"label": "synthetic"},
        "calibration": {"applied": False, "reason": "Fixture model is not calibrated."},
        "resampling_experiment": {"adopted": False, "applied_to_deployed_model": False},
        "split": {"n_train": int(len(x_train)), "n_test": int(len(x_test)), "splits_are_disjoint": True},
        "cost_assumption": {"false_negative_cost": 5, "false_positive_cost": 1},
    }
    model_path = root / "model.joblib"
    metadata_path = root / "metadata.json"
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": list(MODEL_FEATURES),
            "baselines": baselines,
            "feature_spec": metadata["feature_spec"],
            "shap_background": x_train.head(40),
            "positive_label": "Yes",
        },
        model_path,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {"csv": csv_path, "model": model_path, "metadata": metadata_path, "root": root}


@pytest.fixture()
def client(fixture_paths, tmp_path, monkeypatch):
    database = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + database.resolve().as_posix())
    monkeypatch.setenv("MODEL_PATH", str(fixture_paths["model"]))
    monkeypatch.setenv("METADATA_PATH", str(fixture_paths["metadata"]))
    monkeypatch.setenv("SEGMENTS_PATH", str(tmp_path / "missing-segments.joblib"))
    monkeypatch.setenv("DATA_CSV_PATH", str(fixture_paths["csv"]))
    monkeypatch.setenv("AUTO_IMPORT", "true")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    from app.core.config import get_settings
    from app.core.database import reset_db_state
    from app.services.runtime import reset_runtime

    get_settings.cache_clear()
    reset_db_state()
    reset_runtime()
    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client
    reset_db_state()
    reset_runtime()
    get_settings.cache_clear()


