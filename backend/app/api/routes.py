"""REST routes for the churn intelligence API."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.errors import api_error
from app.core.config import get_settings
from app.core.database import get_session
from app.models.tables import AppSetting, Campaign, Customer, Prediction, utcnow
from app.services.customers import customer_count, import_raw_frame, latest_import, list_customers, population_frame
from ml.business.segments import UnknownSegmentError
from app.services.csv_safety import rows_to_csv, safe_csv_filename, safe_download_token
from app.services.insights import (
    analytics_payload,
    business_impact,
    campaign_dict,
    create_campaign,
    dashboard_summary,
    list_campaigns,
    segment_payload,
    suggest_campaigns,
    update_campaign,
)
from app.services.mapping import customer_to_features
from app.services.runtime import (
    active_thresholds,
    get_runtime,
    load_model,
    record_model,
    require_model,
    sync_settings,
)
from app.services.scoring import (
    batch_records_from_frame,
    explain_customer,
    new_batch_id,
    parse_batch_csv,
    prediction_to_dict,
    relabel_stored_predictions,
    score_records,
    score_saved_customers,
)
from ml.data.loader import read_raw_csv
from ml.data.schema import DECISION_SUPPORT_WARNING, EXCLUSION_REASON, EXPLANATION_LIMITATION
from ml.evaluation.serialize import to_jsonable

router = APIRouter()


def _upload_limit_bytes() -> int:
    return get_settings().max_upload_mb * 1024 * 1024


async def _read_limited_csv(request: Request, file: UploadFile) -> tuple[str, bytes]:
    """Read a CSV upload into memory. The returned name is a label, not a path."""
    filename = safe_csv_filename(file.filename)
    if not filename:
        raise api_error(400, "invalid_file_type", "Upload a .csv file.")
    limit = _upload_limit_bytes()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + 1_048_576:
        raise api_error(413, "file_too_large", f"Uploads are limited to {get_settings().max_upload_mb} MB.")
    content = await file.read()
    if len(content) > limit:
        raise api_error(413, "file_too_large", f"Uploads are limited to {get_settings().max_upload_mb} MB.")
    return filename, content


class PredictionInput(BaseModel):
    customer_id: str | None = None
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=240)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class CampaignCreate(BaseModel):
    customer_id: str
    action: str = Field(min_length=3, max_length=200)
    reason: str = Field(min_length=3)
    effort: str = "medium"
    category: str = "retention"
    status: str = "follow_up_required"
    campaign_name: str = "Retention queue"
    priority_score: float = 0
    notes: str | None = None


class CampaignUpdate(BaseModel):
    status: str | None = None
    notes: str | None = None
    campaign_name: str | None = None


class SuggestRequest(BaseModel):
    limit: int = Field(default=25, ge=1, le=200)
    campaign_name: str = "Retention queue"


class SettingsUpdate(BaseModel):
    medium_threshold: float | None = Field(default=None, gt=0, lt=1)
    high_threshold: float | None = Field(default=None, gt=0, lt=1)
    avg_monthly_revenue: float | None = Field(default=None, gt=0)
    contribution_margin: float | None = Field(default=None, ge=0, le=1)
    campaign_cost: float | None = Field(default=None, ge=0)
    success_rate: float | None = Field(default=None, ge=0, le=1)
    horizon_months: int | None = Field(default=None, ge=1, le=60)
    intervention_capacity: int | None = Field(default=None, ge=1)
    clear_capacity: bool = False

    @model_validator(mode="after")
    def thresholds_are_ordered(self):
        if (
            self.medium_threshold is not None
            and self.high_threshold is not None
            and self.medium_threshold >= self.high_threshold
        ):
            raise ValueError("The medium threshold must be below the high threshold.")
        return self


def _settings_payload(row: AppSetting) -> dict[str, Any]:
    return {
        "medium_threshold": row.medium_threshold,
        "high_threshold": row.high_threshold,
        "thresholds_customized": bool(row.thresholds_customized),
        "avg_monthly_revenue": row.avg_monthly_revenue,
        "revenue_customized": bool(row.revenue_customized),
        "contribution_margin": row.contribution_margin,
        "campaign_cost": row.campaign_cost,
        "success_rate": row.success_rate,
        "horizon_months": row.horizon_months,
        "intervention_capacity": row.intervention_capacity,
        "decision_support_warning": DECISION_SUPPORT_WARNING,
        "excluded_features_reason": EXCLUSION_REASON,
        "threshold_note": (
            "The high threshold is the positive-class cutoff. Medium is an elevated band below that cutoff. "
            "Defaults come from the validation cost search and can be changed here without retraining."
        ),
    }


@router.get("/health")
def health(session: Session = Depends(get_session)) -> dict:
    state = get_runtime()
    return {
        "status": "ok",
        "model_loaded": state.model_loaded,
        "model_error": state.load_error,
        "dataset_loaded": customer_count(session) > 0,
        "dataset_label": state.dataset_label,
        "customers": customer_count(session),
        "database": "ok",
    }


@router.get("/dashboard/summary")
def summary(session: Session = Depends(get_session)) -> dict:
    return dashboard_summary(session, get_runtime().dataset_label)


@router.get("/analytics/churn")
def analytics(
    contract: str | None = None,
    internet_service: str | None = None,
    payment_method: str | None = None,
    paperless_billing: str | None = None,
    tech_support: str | None = None,
    tenure_min: float | None = None,
    tenure_max: float | None = None,
    senior_citizen: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    return analytics_payload(
        session,
        {
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "tech_support": tech_support,
            "tenure_min": tenure_min,
            "tenure_max": tenure_max,
            "senior_citizen": senior_citizen,
        },
    )


@router.get("/analytics/segments")
def segments(session: Session = Depends(get_session)) -> dict:
    return segment_payload(session, get_runtime())


@router.get("/customers")
def customers(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    search: str | None = None,
    contract: str | None = None,
    internet_service: str | None = None,
    risk: str | None = None,
    observed_churn: str | None = None,
    tenure_min: int | None = None,
    tenure_max: int | None = None,
    monthly_min: float | None = None,
    monthly_max: float | None = None,
    segment: str | None = None,
    sort: str = "priority_score",
    order: str = "desc",
    session: Session = Depends(get_session),
) -> dict:
    if order not in {"asc", "desc"}:
        raise api_error(400, "invalid_sort", "Order must be asc or desc.")
    try:
        return list_customers(
            session,
            page=page,
            page_size=page_size,
            search=search,
            contract=contract,
            internet_service=internet_service,
            risk=risk,
            observed_churn=observed_churn,
            tenure_min=tenure_min,
            tenure_max=tenure_max,
            monthly_min=monthly_min,
            monthly_max=monthly_max,
            segment=segment,
            sort=sort,
            order=order,
        )
    except UnknownSegmentError:
        raise api_error(400, "unknown_segment", "That segment id is not one of the documented rules.") from None


@router.get("/customers/{customer_id}")
def customer_detail(customer_id: str, refresh_explanation: bool = False, session: Session = Depends(get_session)) -> dict:
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise api_error(404, "customer_not_found", "No stored customer matches that id.")
    latest = session.scalar(
        select(Prediction)
        .where(Prediction.customer_id == customer_id, Prediction.status == "scored")
        .order_by(Prediction.id.desc())
        .limit(1)
    )
    needs_explanation = latest is None or not latest.top_factors or latest.top_factors == "[]"
    prediction = None
    if get_runtime().model_loaded and (refresh_explanation or needs_explanation):
        prediction = explain_customer(get_runtime(), session, customer)
    elif latest is not None:
        prediction = prediction_to_dict(latest)
    if prediction is not None:
        medium, high = active_thresholds(session)
        prediction["thresholds"] = {"medium": medium, "high": high, "decision": high}
    history = [
        prediction_to_dict(row)
        for row in session.scalars(
            select(Prediction).where(Prediction.customer_id == customer_id).order_by(Prediction.id.desc()).limit(20)
        ).all()
    ]
    campaigns = [
        campaign_dict(row)
        for row in session.scalars(
            select(Campaign).where(Campaign.customer_id == customer_id).order_by(Campaign.id.desc())
        ).all()
    ]
    return {
        "profile": customer_to_features(customer),
        "profile_note": (
            "Gender and senior-citizen status are shown only when they were present in the source file. "
            "They are not used to score this customer or to choose a retention action."
        ),
        "prediction": prediction,
        "history": history,
        "campaigns": campaigns,
        "explanation_limitation": EXPLANATION_LIMITATION,
        "decision_support_warning": DECISION_SUPPORT_WARNING,
    }


@router.post("/predictions/single")
def predict_single(payload: PredictionInput, session: Session = Depends(get_session)) -> dict:
    record = payload.model_dump()
    if payload.customer_id:
        record["customerID"] = payload.customer_id
    results = score_records(
        get_runtime(),
        session,
        [record],
        explain=True,
        prefer_shap=True,
        source="single",
        persist=True,
    )
    result = results[0]
    if result["status"] != "scored":
        raise api_error(422, "invalid_customer", " ".join(result["errors"]))
    return result


@router.post("/predictions/batch")
async def predict_batch(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict:
    require_model()
    filename, content = await _read_limited_csv(request, file)
    frame = parse_batch_csv(content)
    records, notes = batch_records_from_frame(frame)
    batch_id = new_batch_id()
    results = score_records(
        get_runtime(),
        session,
        records,
        explain=True,
        prefer_shap=False,
        source="batch",
        batch_id=batch_id,
        persist=True,
    )
    scored = sum(1 for row in results if row["status"] == "scored")
    return {
        "batch_id": batch_id,
        "filename": filename,
        "rows": len(results),
        "scored": scored,
        "invalid": len(results) - scored,
        "notes": notes,
        "preview": results[:20],
        "decision_support_warning": DECISION_SUPPORT_WARNING,
    }


@router.get("/predictions/batches/{batch_id}")
def batch_page(
    batch_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    risk: str | None = None,
    status: str | None = None,
    sort: str = "probability",
    order: str = "desc",
    session: Session = Depends(get_session),
) -> dict:
    statement = select(Prediction).where(Prediction.batch_id == batch_id)
    if risk in {"Low", "Medium", "High"}:
        statement = statement.where(Prediction.risk_category == risk)
    if status in {"scored", "invalid"}:
        statement = statement.where(Prediction.status == status)
    total = int(session.scalar(select(func.count()).select_from(statement.order_by(None).subquery())) or 0)
    stored = int(session.scalar(select(func.count()).select_from(Prediction).where(Prediction.batch_id == batch_id)) or 0)
    if total == 0 and stored == 0:
        raise api_error(404, "batch_not_found", "No batch matches that id.")
    sort_column = {
        "probability": Prediction.probability,
        "priority_score": Prediction.priority_score,
        "row_number": Prediction.row_number,
    }.get(sort, Prediction.probability)
    sort_expression = sort_column.asc() if order == "asc" else sort_column.desc()
    rows = session.scalars(
        statement.order_by(sort_expression).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return {
        "batch_id": batch_id,
        "items": [prediction_to_dict(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/predictions/batches/{batch_id}/download")
def batch_download(batch_id: str, session: Session = Depends(get_session)):
    from fastapi.responses import Response

    rows = session.scalars(select(Prediction).where(Prediction.batch_id == batch_id).order_by(Prediction.row_number)).all()
    if not rows:
        raise api_error(404, "batch_not_found", "No batch matches that id.")
    export_rows = []
    for row in rows:
        factors = []
        if row.top_factors:
            try:
                parsed = json.loads(row.top_factors)
                factors = [f"{item.get('feature')} ({item.get('direction')})" for item in parsed[:3]]
            except json.JSONDecodeError:
                factors = []
        export_rows.append(
            {
                "row_number": row.row_number,
                "customer_id": row.customer_id or row.external_id,
                "status": row.status,
                "probability": row.probability,
                "risk_category": row.risk_category,
                "predicted_class": row.predicted_class,
                "priority_score": row.priority_score,
                "top_factors": "; ".join(factors),
                "errors": row.errors or "",
            }
        )
    headers = [
        "row_number",
        "customer_id",
        "status",
        "probability",
        "risk_category",
        "predicted_class",
        "priority_score",
        "top_factors",
        "errors",
    ]
    content = rows_to_csv(headers, export_rows)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch-{safe_download_token(batch_id)}.csv"'},
    )


@router.get("/models")
def models() -> dict:
    state = get_runtime()
    metadata = state.metadata or {}
    return {
        "model_loaded": state.model_loaded,
        "model_error": state.load_error,
        "selected_model": metadata.get("selected_model"),
        "trained_at": metadata.get("trained_at"),
        "selection_reason": metadata.get("selection_reason"),
        "selection_metric": metadata.get("selection_metric"),
        "excluded_features": metadata.get("excluded_features"),
        "exclusion_reason": metadata.get("exclusion_reason"),
        "feature_spec": metadata.get("feature_spec"),
        "model_features": metadata.get("model_features"),
        "library_versions": metadata.get("library_versions"),
        "model_availability_note": metadata.get("model_availability_note"),
    }


@router.get("/models/performance")
def model_performance() -> dict:
    state = require_model()
    metadata = state.metadata or {}
    if not metadata:
        raise api_error(404, "metrics_unavailable", "Model metadata has not been written yet.")
    return {
        "selected_model": metadata.get("selected_model"),
        "selection_reason": metadata.get("selection_reason"),
        "comparison_scope": metadata.get("comparison_scope"),
        "comparison": metadata.get("comparison"),
        "resampling_experiment": metadata.get("resampling_experiment"),
        "calibration": metadata.get("calibration"),
        "cost_assumption": metadata.get("cost_assumption"),
        "thresholds": metadata.get("thresholds"),
        "validation_metrics": metadata.get("validation_metrics"),
        "test_metrics": metadata.get("test_metrics"),
        "feature_importance": metadata.get("feature_importance"),
        "split": metadata.get("split"),
        "dataset": metadata.get("dataset"),
        "excluded_features": metadata.get("excluded_features"),
        "exclusion_reason": metadata.get("exclusion_reason"),
    }


@router.post("/models/reload")
def reload_model(session: Session = Depends(get_session)) -> dict:
    state = load_model()
    sync_settings(session, state)
    if state.model_loaded and customer_count(session):
        score_saved_customers(state, session)
        record_model(session, state)
    return {"model_loaded": state.model_loaded, "model_error": state.load_error, "selected_model": (state.metadata or {}).get("selected_model")}


@router.get("/explanations/global")
def global_explanation() -> dict:
    state = require_model()
    metadata = state.metadata or {}
    return {
        "feature_importance": metadata.get("feature_importance"),
        "associations": metadata.get("associations"),
        "excluded_features": metadata.get("excluded_features"),
        "exclusion_reason": metadata.get("exclusion_reason"),
        "limitation": EXPLANATION_LIMITATION,
        "shap_global": metadata.get("shap_global"),
        "note": (
            "Global importance is grouped permutation importance on the internal validation split. "
            "It was not used to choose the model, and it is not a causal ranking."
        ),
    }


@router.get("/explanations/{customer_id}")
def customer_explanation(customer_id: str, session: Session = Depends(get_session)) -> dict:
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise api_error(404, "customer_not_found", "No stored customer matches that id.")
    return explain_customer(get_runtime(), session, customer)


@router.get("/campaigns")
def campaigns(status: str | None = None, search: str | None = None, session: Session = Depends(get_session)) -> dict:
    items = list_campaigns(session, status, search)
    return {"items": items, "total": len(items), "outcomes_available": False}


@router.post("/campaigns")
def post_campaign(payload: CampaignCreate, session: Session = Depends(get_session)) -> dict:
    return create_campaign(session, payload.model_dump())


@router.patch("/campaigns/{campaign_id}")
def patch_campaign(campaign_id: int, payload: CampaignUpdate, session: Session = Depends(get_session)) -> dict:
    return update_campaign(session, campaign_id, payload.model_dump(exclude_unset=True))


@router.post("/campaigns/suggest")
def post_suggestions(payload: SuggestRequest, session: Session = Depends(get_session)) -> dict:
    return suggest_campaigns(session, get_runtime(), payload.limit, payload.campaign_name)


@router.get("/business-impact")
def get_impact(session: Session = Depends(get_session)) -> dict:
    return business_impact(session)


@router.get("/settings")
def get_settings_route(session: Session = Depends(get_session)) -> dict:
    return _settings_payload(sync_settings(session))


@router.put("/settings")
def put_settings(payload: SettingsUpdate, session: Session = Depends(get_session)) -> dict:
    row = sync_settings(session)
    data = payload.model_dump(exclude_unset=True)
    medium = data.get("medium_threshold", row.medium_threshold)
    high = data.get("high_threshold", row.high_threshold)
    if medium >= high:
        raise api_error(422, "invalid_thresholds", "The medium threshold must be below the high threshold.")
    threshold_changed = False
    if "medium_threshold" in data or "high_threshold" in data:
        row.medium_threshold = float(medium)
        row.high_threshold = float(high)
        row.thresholds_customized = 1
        threshold_changed = True
    if "avg_monthly_revenue" in data and data["avg_monthly_revenue"] is not None:
        row.avg_monthly_revenue = float(data["avg_monthly_revenue"])
        row.revenue_customized = 1
    for field in ("contribution_margin", "campaign_cost", "success_rate", "horizon_months"):
        if field in data and data[field] is not None:
            setattr(row, field, data[field])
    if data.get("clear_capacity"):
        row.intervention_capacity = None
    elif "intervention_capacity" in data:
        row.intervention_capacity = data["intervention_capacity"]
    row.updated_at = utcnow()
    session.commit()
    if threshold_changed:
        relabel_stored_predictions(session, float(row.medium_threshold), float(row.high_threshold))
    return _settings_payload(row)


@router.get("/data-quality")
def data_quality(session: Session = Depends(get_session)) -> dict:
    record = latest_import(session)
    if record is None:
        return {
            "available": False,
            "message": "No dataset has been imported. Place the CSV in data/raw and restart, or upload it here.",
            "expected_file": "data/raw/Telco-Customer-Churn.csv",
        }
    quality = json.loads(record.quality_report) if record.quality_report else {}
    return {
        "available": True,
        "import_id": record.id,
        "filename": record.filename,
        "dataset_label": record.dataset_label,
        "imported_at": record.created_at.isoformat() if record.created_at else None,
        "message": record.message,
        "stored_customers": record.stored_count,
        "quality": quality,
        "customers_in_database": customer_count(session),
    }


@router.post("/data/import")
async def import_data(request: Request, file: UploadFile = File(...), session: Session = Depends(get_session)) -> dict:
    filename, content = await _read_limited_csv(request, file)
    import io

    import pandas as pd

    try:
        raw = pd.read_csv(io.BytesIO(content), dtype=str, encoding="utf-8-sig")
    except Exception as exc:
        raise api_error(400, "invalid_csv", "The file could not be read as CSV.") from exc
    try:
        result = import_raw_frame(session, raw, filename)
    except Exception as exc:
        from ml.data.schema import SchemaError

        if isinstance(exc, SchemaError):
            raise api_error(400, "schema_error", str(exc)) from exc
        raise
    state = get_runtime()
    state.dataset_label = result["dataset_label"]
    if state.model_loaded:
        result["scored_customers"] = score_saved_customers(state, session)
    else:
        result["scored_customers"] = 0
        result["scoring_note"] = "Customers were stored. Scores will appear after a model is trained and reloaded."
    sync_settings(session, state)
    return to_jsonable(result)


def import_configured_csv(session: Session) -> None:
    """Used at startup when the database is empty and the configured CSV exists."""
    path = get_settings().resolved_csv_path()
    if not path.exists():
        return
    raw = read_raw_csv(path)
    result = import_raw_frame(session, raw, path.name)
    state = get_runtime()
    state.dataset_label = result["dataset_label"]
    sync_settings(session, state)
    if state.model_loaded:
        score_saved_customers(state, session)
