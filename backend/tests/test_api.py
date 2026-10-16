"""API behavior against a synthetic fixture and a real fitted pipeline."""

from __future__ import annotations

import io

from ml.data.synthetic import make_synthetic_frame


def sample_customer() -> dict:
    frame = make_synthetic_frame(160, seed=11)
    row = frame.iloc[10].to_dict()
    row["customer_id"] = row["customerID"]
    if row["TotalCharges"] == "":
        row["TotalCharges"] = None
    else:
        row["TotalCharges"] = float(row["TotalCharges"])
    row["tenure"] = int(float(row["tenure"]))
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    return row


def test_health_and_summary(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["dataset_loaded"] is True
    assert body["dataset_label"] == "synthetic"
    summary = client.get("/api/dashboard/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["total_customers"] > 0
    assert payload["observed_churn_rate"] is not None
    assert "decision_support_warning" in payload
    assert payload["time_series_available"] is False


def test_customer_filter_and_pagination(client):
    page = client.get("/api/customers", params={"page": 1, "page_size": 5, "contract": "Month-to-month"})
    assert page.status_code == 200
    body = page.json()
    assert body["page_size"] == 5
    assert body["total"] >= len(body["items"])
    assert all(item["contract"] == "Month-to-month" for item in body["items"])
    if body["items"]:
        detail = client.get(f"/api/customers/{body['items'][0]['customer_id']}")
        assert detail.status_code == 200
        assert detail.json()["profile"]["customerID"] == body["items"][0]["customer_id"]
        assert detail.json()["prediction"]["probability"] is not None


def test_single_prediction_and_invalid_request(client):
    row = sample_customer()
    response = client.post("/api/predictions/single", json=row)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_category"] in {"Low", "Medium", "High"}
    assert body["predicted_class"] in {"Yes", "No"}
    assert body["factors"]
    assert body["recommendations"]
    assert "certainty" in body["reliability"]["message"] or "probability" in body["reliability"]["message"]
    invalid = client.post("/api/predictions/single", json={"tenure": -1})
    assert invalid.status_code == 422


def test_batch_upload_keeps_invalid_rows(client):
    row = sample_customer()
    header = [
        "customerID",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    good = [row["customerID"], row["Partner"], row["Dependents"], row["tenure"], row["PhoneService"], row["MultipleLines"], row["InternetService"], row["OnlineSecurity"], row["OnlineBackup"], row["DeviceProtection"], row["TechSupport"], row["StreamingTV"], row["StreamingMovies"], row["Contract"], row["PaperlessBilling"], row["PaymentMethod"], row["MonthlyCharges"], row["TotalCharges"]]
    bad = good.copy()
    bad[0] = "SYN-BAD"
    bad[13] = "Lifetime"
    lines = [",".join(header), ",".join(str(value) for value in good), ",".join(str(value) for value in bad)]
    upload = client.post(
        "/api/predictions/batch",
        files={"file": ("batch.csv", io.BytesIO("\n".join(lines).encode("utf-8")), "text/csv")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["rows"] == 2
    assert body["invalid"] == 1
    assert body["scored"] == 1
    downloaded = client.get(f"/api/predictions/batches/{body['batch_id']}/download")
    assert downloaded.status_code == 200
    assert "SYN-BAD" in downloaded.text
    assert "invalid" in downloaded.text


def test_campaign_create_and_status_update(client):
    customers = client.get("/api/customers", params={"page_size": 1}).json()
    customer_id = customers["items"][0]["customer_id"]
    created = client.post(
        "/api/campaigns",
        json={
            "customer_id": customer_id,
            "action": "Prioritize a term-contract conversation",
            "reason": "Month-to-month contract and elevated score.",
            "effort": "medium",
            "status": "follow_up_required",
        },
    )
    assert created.status_code == 200
    campaign_id = created.json()["id"]
    updated = client.patch(f"/api/campaigns/{campaign_id}", json={"status": "contacted", "notes": "Called the customer."})
    assert updated.status_code == 200
    assert updated.json()["status"] == "contacted"
    listed = client.get("/api/campaigns", params={"status": "contacted"})
    assert any(item["id"] == campaign_id for item in listed.json()["items"])
    missing = client.post(
        "/api/campaigns",
        json={"customer_id": "NO-SUCH", "action": "Call", "reason": "Missing customer should fail."},
    )
    assert missing.status_code == 404


def test_import_rejects_bad_files_and_impact_recalculates(client):
    rejected = client.post(
        "/api/data/import",
        files={"file": ("notes.txt", io.BytesIO(b"not,a,csv"), "text/plain")},
    )
    assert rejected.status_code == 400
    broken = client.post(
        "/api/data/import",
        files={"file": ("bad.csv", io.BytesIO(b"only_one_column\nvalue\n"), "text/csv")},
    )
    assert broken.status_code == 400
    assert broken.json()["code"] == "schema_error"
    before = client.get("/api/business-impact").json()
    updated = client.put(
        "/api/settings",
        json={"campaign_cost": 40, "success_rate": 0.25, "horizon_months": 6, "contribution_margin": 0.4},
    )
    assert updated.status_code == 200
    after = client.get("/api/business-impact").json()
    assert after["assumptions"]["campaign_cost"] == 40
    assert after["assumptions"]["horizon_months"] == 6
    assert after["outcomes_available"] is False
    assert before["scenarios"][1]["campaign_cost"] != after["scenarios"][1]["campaign_cost"]


def test_segment_membership_matches_counts_and_rejects_unknown_rules(client):
    segments = client.get("/api/analytics/segments")
    assert segments.status_code == 200
    rules = segments.json()["rules"]
    assert rules
    for rule in rules:
        first = client.get("/api/customers", params={"segment": rule["id"], "page": 1, "page_size": 1})
        assert first.status_code == 200
        body = first.json()
        assert body["total"] == rule["customers"]
        assert body["segment"]["members"] == rule["customers"]
        assert body["segment"]["id"] == rule["id"]
        assert len(body["items"]) == min(1, rule["customers"])
        if rule["customers"] > 1:
            second = client.get("/api/customers", params={"segment": rule["id"], "page": 2, "page_size": 1})
            assert second.json()["total"] == rule["customers"]
            assert second.json()["items"][0]["customer_id"] != body["items"][0]["customer_id"]
    narrowed = client.get(
        "/api/customers",
        params={"segment": "month_to_month_elevated_risk", "contract": "Two year", "page_size": 5},
    )
    month_rule = next(item for item in rules if item["id"] == "month_to_month_elevated_risk")
    assert narrowed.status_code == 200
    assert narrowed.json()["total"] == 0
    assert narrowed.json()["segment"]["members"] == month_rule["customers"]
    unknown = client.get("/api/customers", params={"segment": "not-a-rule"})
    assert unknown.status_code == 400
    assert unknown.json()["code"] == "unknown_segment"
    assert "Traceback" not in unknown.text


def test_traversal_filename_is_not_stored_as_a_path(client):
    rejected = client.post(
        "/api/data/import",
        files={"file": ("../../outside.csv", io.BytesIO(b"only_one_column\nvalue\n"), "text/csv")},
    )
    assert rejected.status_code == 400
    assert ".." not in rejected.text
    assert "Traceback" not in rejected.text
    not_csv = client.post(
        "/api/data/import",
        files={"file": ("../../secret.txt", io.BytesIO(b"not,a,csv"), "text/plain")},
    )
    assert not_csv.status_code == 400


def test_validation_error_omits_submitted_values_and_tracebacks(client):
    response = client.post("/api/predictions/single", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["message"] == "The request is invalid."
    assert body["errors"]
    assert "Traceback" not in response.text
    for item in body["errors"]:
        assert set(item) == {"type", "loc", "msg"}


def test_upload_larger_than_the_limit_is_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        payload = b"x" * (2 * 1024 * 1024 + 100)
        response = client.post(
            "/api/predictions/batch",
            files={"file": ("rows.csv", payload, "text/csv")},
        )
        assert response.status_code == 413
        assert response.json()["code"] == "file_too_large"
        assert "Traceback" not in response.text
    finally:
        get_settings.cache_clear()


def test_model_and_explanation_endpoints(client):
    models = client.get("/api/models")
    assert models.status_code == 200
    assert models.json()["model_loaded"] is True
    performance = client.get("/api/models/performance")
    assert performance.status_code == 200
    assert "test_metrics" in performance.json()
    assert performance.json()["test_metrics"]["average_precision"] is not None
    global_view = client.get("/api/explanations/global")
    assert global_view.status_code == 200
    assert "limitation" in global_view.json()
    quality = client.get("/api/data-quality")
    assert quality.status_code == 200
    assert quality.json()["dataset_label"] == "synthetic"
