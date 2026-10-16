"""Unit tests for data handling, metrics, explanations inputs, and scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.business.impact import ImpactAssumptions, calculate_impact
from ml.business.recommendations import recommend
from ml.business.risk import predicted_label, priority_score, risk_category
from ml.data.loader import clean_customers, filter_trainable
from ml.data.quality import build_quality_report
from ml.data.schema import MODEL_FEATURES, SchemaError, TARGET_COLUMN
from ml.data.synthetic import make_synthetic_frame
from ml.data.validate import validate_record
from ml.evaluation.metrics import binary_metrics, tune_cost_threshold
from ml.features.preprocessing import make_model_pipeline
from app.services.csv_safety import sanitize_cell


def _spec_from_frame(frame: pd.DataFrame) -> dict:
    categorical = {}
    for column in MODEL_FEATURES:
        if column in ("tenure", "MonthlyCharges", "TotalCharges"):
            continue
        categorical[column] = sorted(frame[column].dropna().astype(str).unique().tolist())
    return {
        "numeric": {
            "tenure": {"min": 0, "max": 72, "median": 10},
            "MonthlyCharges": {"min": 18, "max": 120, "median": 70},
            "TotalCharges": {"min": 0, "max": 9000, "median": 1000},
        },
        "categorical": categorical,
    }


def test_blank_total_charges_with_zero_tenure_are_kept():
    raw = make_synthetic_frame(30, seed=3)
    raw.loc[0, "tenure"] = "0"
    raw.loc[0, "TotalCharges"] = " "
    cleaned, notes = clean_customers(raw)
    assert cleaned.iloc[0]["TotalCharges"] == 0
    assert any("TotalCharges" in note for note in notes)
    trainable, reasons = filter_trainable(cleaned)
    assert reasons["rows_excluded"] == 0
    assert len(trainable) == 30


def test_missing_required_column_is_rejected():
    raw = make_synthetic_frame(10, seed=4).drop(columns=["Contract"])
    with pytest.raises(SchemaError, match="Contract"):
        clean_customers(raw)


def test_duplicates_are_reported_and_not_silently_ignored():
    raw = make_synthetic_frame(12, seed=5)
    raw.loc[1, "customerID"] = raw.loc[0, "customerID"]
    cleaned, notes = clean_customers(raw)
    quality = build_quality_report(raw, cleaned, notes)
    assert quality["duplicate_customer_ids"] == 1
    assert quality["duplicate_id_examples"]
    assert quality["outliers"]["MonthlyCharges"]["note"].startswith("Flagged")


def test_preprocessor_learns_only_training_rows():
    raw = make_synthetic_frame(120, seed=6)
    cleaned, _notes = clean_customers(raw)
    trainable, _reasons = filter_trainable(cleaned)
    features = trainable.loc[:, list(MODEL_FEATURES)]
    target = (trainable[TARGET_COLUMN] == "Yes").astype(int)
    x_train, x_test, y_train, _y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=6
    )
    leaked = x_test.copy()
    leaked["tenure"] = 500
    pipeline = make_model_pipeline(LogisticRegression(max_iter=300, solver="liblinear"), scale_numeric=True)
    pipeline.fit(x_train, y_train)
    scaler = pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]
    assert scaler.mean_[0] == pytest.approx(float(x_train["tenure"].mean()))
    combined = float(pd.concat([x_train["tenure"], leaked["tenure"]]).mean())
    assert scaler.mean_[0] != pytest.approx(combined)
    probabilities = pipeline.predict_proba(x_test.iloc[:8])
    assert probabilities.shape == (8, 2)
    assert np.all((probabilities[:, 1] >= 0) & (probabilities[:, 1] <= 1))


def test_metrics_and_threshold_use_probabilities():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.45, 0.8, 0.9])
    best, _grid = tune_cost_threshold(y_true, probabilities, fn_cost=5, fp_cost=1)
    metrics = binary_metrics(y_true, probabilities, best["threshold"])
    assert metrics["confusion_matrix"]["matrix"][0][0] >= 0
    assert 0 <= metrics["recall"] <= 1
    assert metrics["roc_auc"] > 0.5


def test_risk_bands_follow_configured_thresholds():
    assert risk_category(0.2, 0.3, 0.5) == "Low"
    assert risk_category(0.3, 0.3, 0.5) == "Medium"
    assert risk_category(0.5, 0.3, 0.5) == "High"
    assert predicted_label(0.5, 0.5) == "Yes"
    assert predicted_label(0.49, 0.5) == "No"
    assert priority_score(0.5, 80, 0, monthly_p95=100) > priority_score(0.5, 80, 72, monthly_p95=100)


def test_recommendations_ignore_protected_attributes_and_state_the_rule():
    frame = make_synthetic_frame(20, seed=8)
    spec = _spec_from_frame(frame)
    base = frame.iloc[0].to_dict()
    base.update({"Contract": "Month-to-month", "tenure": 2, "MonthlyCharges": 90, "TechSupport": "No"})
    cleaned, errors, _warnings = validate_record(base, spec)
    assert not errors and cleaned is not None
    first = recommend(cleaned, 0.8, 0.3, 0.5, high_charge_cutoff=80)
    changed = dict(cleaned)
    other = recommend({**changed, "gender": "Male", "SeniorCitizen": 1}, 0.8, 0.3, 0.5, high_charge_cutoff=80)
    assert [item.as_dict() for item in first] == [item.as_dict() for item in other]
    text = " ".join(item.reason + item.action for item in first).lower()
    assert "gender" not in text
    assert "senior" not in text
    assert any(item.category == "contract" for item in first)


def test_financial_scenario_uses_the_documented_formula():
    probabilities = np.array([0.5, 0.5])
    assumptions = ImpactAssumptions(
        avg_monthly_revenue=100,
        contribution_margin=0.5,
        campaign_cost=25,
        success_rate=0.2,
        horizon_months=12,
        intervention_capacity=None,
    )
    result = calculate_impact(probabilities, assumptions)
    base = next(item for item in result["scenarios"] if item["name"] == "base")
    assert base["customers_targeted"] == 2
    assert base["expected_churners"] == pytest.approx(1.0)
    assert base["revenue_exposure"] == pytest.approx(1200)
    assert base["contribution_exposure"] == pytest.approx(600)
    assert base["expected_retained_customers"] == pytest.approx(0.2)
    assert base["estimated_contribution_benefit"] == pytest.approx(120)
    assert base["campaign_cost"] == pytest.approx(50)
    assert base["estimated_net_benefit"] == pytest.approx(70)
    assert base["break_even_success_rate"] == pytest.approx(50 / 600)
    assert result["outcomes_available"] is False


def test_capacity_limits_the_targeted_group():
    assumptions = ImpactAssumptions(100, 0.5, 10, 0.3, 12, intervention_capacity=1)
    result = calculate_impact(np.array([0.2, 0.9]), assumptions)
    base = next(item for item in result["scenarios"] if item["name"] == "base")
    assert base["customers_targeted"] == 1
    assert base["expected_churners"] == pytest.approx(0.9)


def test_csv_formula_injection_is_neutralized():
    assert sanitize_cell("=cmd()").startswith("'")
    assert sanitize_cell("@sum").startswith("'")
    assert sanitize_cell("7043") == "7043"


def _segment_row(customer_id: str, **overrides):
    row = {
        "customerID": customer_id,
        "MonthlyCharges": 50.0,
        "tenure": 24,
        "Contract": "Two year",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "risk_category": "Low",
        "probability": 0.05,
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_segment_rules_use_every_condition_and_boundaries():
    from ml.business.segments import customer_ids_for_rule, summarize_rules

    cutoff = 80.0
    frame = pd.DataFrame(
        [
            _segment_row("early-edge", tenure=6, risk_category="Medium", Contract="One year", probability=0.2),
            _segment_row("early-over", tenure=7, risk_category="Medium", Contract="One year", probability=0.2),
            _segment_row("value-edge", tenure=3, MonthlyCharges=80, risk_category="High", Contract="Month-to-month", probability=0.7, Churn="Yes"),
            _segment_row("value-under", tenure=3, MonthlyCharges=79.99, risk_category="High", Contract="Month-to-month", probability=0.7),
            _segment_row("loyal-edge", tenure=48, risk_category="Low"),
            _segment_row("loyal-under", tenure=47, risk_category="Low"),
            _segment_row("expand", tenure=12, risk_category="Low", InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes"),
            _segment_row("expand-addons", tenure=12, risk_category="Low", InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes"),
            _segment_row("expand-no-net", tenure=20, risk_category="Low", InternetService="No"),
            _segment_row("unscored-charge", tenure=10, MonthlyCharges=90, risk_category=None, probability=None),
            _segment_row("residual", tenure=20, MonthlyCharges=40, risk_category="Low", InternetService="No"),
        ]
    )
    summaries = {item["id"]: item for item in summarize_rules(frame, cutoff)}
    expected = {
        "high_risk_high_value": ["value-edge"],
        "early_tenure_elevated_risk": ["early-edge", "value-edge", "value-under"],
        "month_to_month_elevated_risk": ["value-edge", "value-under"],
        "high_monthly_charge": ["value-edge", "unscored-charge"],
        "long_tenure_loyal": ["loyal-edge"],
        "low_risk_expansion": ["loyal-edge", "loyal-under", "expand"],
        "other": ["early-over", "expand-addons", "expand-no-net", "residual"],
    }
    for rule_id, ids in expected.items():
        assert customer_ids_for_rule(frame, cutoff, rule_id) == ids
        assert summaries[rule_id]["customers"] == len(ids)
    assert customer_ids_for_rule(frame.iloc[0:0], cutoff, "high_risk_high_value") == []
    with pytest.raises(KeyError):
        customer_ids_for_rule(frame, cutoff, "not-a-segment")


def test_missing_model_message_omits_the_filesystem_path(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    from app.core.config import get_settings
    from app.services.runtime import load_model, reset_runtime

    get_settings.cache_clear()
    reset_runtime()
    try:
        state = load_model()
        assert state.model_loaded is False
        assert state.load_error == "No model file is available. Train the model before starting the API."
        assert str(missing) not in (state.load_error or "")
    finally:
        reset_runtime()
        get_settings.cache_clear()


def test_upload_filename_cannot_escape_the_label():
    from app.services.csv_safety import safe_csv_filename, safe_download_token

    assert safe_csv_filename("../../outside.csv") == "outside.csv"
    assert safe_csv_filename("..\\..\\windows\\system.ini.csv") == "system.ini.csv"
    assert safe_csv_filename("notes.txt") == ""
    assert safe_csv_filename("=cmd.csv") == "=cmd.csv"
    assert "/" not in safe_csv_filename("../../outside.csv")
    assert safe_download_token('batch"\r\n.html') == "batchhtml"
