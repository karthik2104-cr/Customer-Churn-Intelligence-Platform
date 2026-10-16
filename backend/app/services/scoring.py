"""Score customers with the deployed pipeline and persist the results."""

from __future__ import annotations

import io
import json
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.errors import api_error
from app.models.tables import Customer, Prediction, utcnow
from app.services.mapping import customer_to_features, features_to_model_row
from app.services.runtime import RuntimeState, active_thresholds, require_model
from ml.business.recommendations import recommend
from ml.business.risk import predicted_label, priority_score, reliability, risk_category
from ml.data.schema import DECISION_SUPPORT_WARNING, MODEL_FEATURES
from ml.data.validate import record_identifier, validate_record
from ml.explainability.explain import explain_frame
from ml.evaluation.serialize import to_jsonable


def _cutoffs(session: Session, state: RuntimeState, charges: pd.Series | None = None) -> tuple[float | None, float | None]:
    if charges is not None and charges.notna().sum() >= 20:
        values = pd.to_numeric(charges, errors="coerce").dropna()
        return float(values.quantile(0.75)), float(values.quantile(0.95))
    stored = session.scalars(select(Customer.monthly_charges)).all()
    if len(stored) >= 20:
        values = pd.Series(stored, dtype=float)
        return float(values.quantile(0.75)), float(values.quantile(0.95))
    background = (state.bundle or {}).get("shap_background")
    if isinstance(background, pd.DataFrame) and "MonthlyCharges" in background and len(background) >= 20:
        return float(background["MonthlyCharges"].quantile(0.75)), float(background["MonthlyCharges"].quantile(0.95))
    return None, None


def _feature_spec(state: RuntimeState) -> dict:
    return (state.bundle or {}).get("feature_spec") or (state.metadata or {}).get("feature_spec") or {}


def _baselines(state: RuntimeState) -> dict:
    return (state.bundle or {}).get("baselines") or (state.metadata or {}).get("baselines") or {}


def _associations(state: RuntimeState) -> dict | None:
    return (state.metadata or {}).get("associations")


def _top_features(state: RuntimeState, limit: int = 8) -> list[str]:
    rows = ((state.metadata or {}).get("feature_importance") or {}).get("features") or []
    names = [row["feature"] for row in rows if row.get("feature") in MODEL_FEATURES]
    return names[:limit] or list(MODEL_FEATURES)


def _frame_from_feature_dicts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([features_to_model_row(row) for row in rows], columns=list(MODEL_FEATURES))


def _prediction_payload(
    *,
    customer_id: str | None,
    probability: float,
    medium: float,
    high: float,
    factors: list[dict],
    recommendations: list[dict],
    reliability_info: dict,
    method: str,
    priority: float | None,
    warnings: list[str] | None = None,
    fallback_reason: str | None = None,
) -> dict:
    return {
        "customer_id": customer_id,
        "probability": float(probability),
        "risk_category": risk_category(probability, medium, high),
        "predicted_class": predicted_label(probability, high),
        "priority_score": priority,
        "factors": factors,
        "recommendations": recommendations,
        "reliability": reliability_info,
        "explanation_method": method,
        "explanation_fallback": fallback_reason,
        "thresholds": {"medium": medium, "high": high, "decision": high},
        "decision_support_warning": DECISION_SUPPORT_WARNING,
        "warnings": warnings or [],
    }


def score_records(
    state: RuntimeState,
    session: Session,
    records: list[dict],
    *,
    explain: bool,
    prefer_shap: bool,
    source: str,
    batch_id: str | None = None,
    persist: bool = True,
) -> list[dict]:
    state = require_model(state)
    medium, high = active_thresholds(session)
    spec = _feature_spec(state)
    prepared: list[dict | None] = []
    errors: list[list[str]] = []
    warnings: list[list[str]] = []
    identifiers: list[str | None] = []
    for record in records:
        cleaned, row_errors, row_warnings = validate_record(record, spec)
        prepared.append(cleaned)
        errors.append(row_errors)
        warnings.append(row_warnings)
        identifiers.append(record_identifier(record))

    valid_index = [index for index, row in enumerate(prepared) if row is not None]
    probabilities = np.array([], dtype=float)
    explanations: list[dict] = []
    if valid_index:
        frame = _frame_from_feature_dicts([prepared[index] for index in valid_index])
        probabilities = np.clip(np.asarray(state.pipeline.predict_proba(frame)[:, 1], dtype=float), 0.0, 1.0)
        if explain:
            background = state.bundle.get("shap_background") if prefer_shap else None
            explanations = explain_frame(
                state.pipeline,
                frame,
                _baselines(state),
                background=background if isinstance(background, pd.DataFrame) else None,
                top_features=None if prefer_shap else _top_features(state),
                prefer_shap=prefer_shap,
                top_n=6,
            )
        else:
            explanations = [{} for _ in valid_index]

    charge_series = None
    if valid_index:
        charge_series = pd.Series([prepared[index]["MonthlyCharges"] for index in valid_index], dtype=float)
    high_charge, p95 = _cutoffs(session, state, charge_series)
    associations = _associations(state)
    results: list[dict] = []
    valid_cursor = 0
    for index, record in enumerate(records):
        if prepared[index] is None:
            results.append(
                {
                    "row_number": index + 2,
                    "customer_id": identifiers[index],
                    "status": "invalid",
                    "errors": errors[index],
                    "warnings": warnings[index],
                    "probability": None,
                    "risk_category": None,
                    "predicted_class": None,
                    "priority_score": None,
                    "factors": [],
                    "recommendations": [],
                }
            )
            continue
        probability = float(probabilities[valid_cursor])
        explanation = explanations[valid_cursor] if explanations else {}
        valid_cursor += 1
        features = prepared[index]
        recommendations = [
            item.as_dict()
            for item in recommend(
                features,
                probability,
                medium,
                high,
                high_charge_cutoff=high_charge,
                associations=associations,
            )
        ]
        priority = None
        if features.get("MonthlyCharges") is not None and features.get("tenure") is not None and p95:
            priority = priority_score(probability, features["MonthlyCharges"], features["tenure"], p95)
        payload = _prediction_payload(
            customer_id=identifiers[index],
            probability=probability,
            medium=medium,
            high=high,
            factors=explanation.get("factors") or [],
            recommendations=recommendations,
            reliability_info=reliability(features, probability, spec),
            method=explanation.get("method") or "probability_only",
            priority=priority,
            warnings=warnings[index],
            fallback_reason=explanation.get("fallback_reason"),
        )
        payload["row_number"] = index + 2
        payload["status"] = "scored"
        payload["errors"] = []
        results.append(payload)

    if persist:
        _persist_results(session, results, source=source, batch_id=batch_id, records=records, prepared=prepared)
    return results


def _persist_results(
    session: Session,
    results: list[dict],
    *,
    source: str,
    batch_id: str | None,
    records: list[dict],
    prepared: list[dict | None],
) -> None:
    known_ids = set(session.scalars(select(Customer.customer_id)).all())
    for index, result in enumerate(results):
        customer_id = result.get("customer_id")
        linked_id = customer_id if customer_id in known_ids else None
        if source in {"single", "batch"} and customer_id and prepared[index] is not None and linked_id is None:
            _insert_customer_from_features(session, customer_id, prepared[index])
            linked_id = customer_id
            known_ids.add(customer_id)
        session.add(
            Prediction(
                customer_id=linked_id,
                external_id=customer_id,
                batch_id=batch_id,
                row_number=result.get("row_number"),
                probability=result.get("probability"),
                risk_category=result.get("risk_category"),
                predicted_class=result.get("predicted_class"),
                priority_score=result.get("priority_score"),
                top_factors=json.dumps(to_jsonable(result.get("factors") or [])),
                recommendations=json.dumps(to_jsonable(result.get("recommendations") or [])),
                explanation_method=result.get("explanation_method"),
                reliability=json.dumps(to_jsonable(result.get("reliability") or {})),
                status=result.get("status") or "invalid",
                errors=json.dumps(result.get("errors") or []),
                warnings=json.dumps(result.get("warnings") or []),
                source=source,
                created_at=utcnow(),
            )
        )
    session.commit()


def _insert_customer_from_features(session: Session, customer_id: str, features: dict) -> None:
    total = features.get("TotalCharges")
    session.add(
        Customer(
            customer_id=customer_id,
            partner=features.get("Partner"),
            dependents=features.get("Dependents"),
            tenure=int(features["tenure"]),
            phone_service=features.get("PhoneService"),
            multiple_lines=features.get("MultipleLines"),
            internet_service=features.get("InternetService"),
            online_security=features.get("OnlineSecurity"),
            online_backup=features.get("OnlineBackup"),
            device_protection=features.get("DeviceProtection"),
            tech_support=features.get("TechSupport"),
            streaming_tv=features.get("StreamingTV"),
            streaming_movies=features.get("StreamingMovies"),
            contract=features.get("Contract"),
            paperless_billing=features.get("PaperlessBilling"),
            payment_method=features.get("PaymentMethod"),
            monthly_charges=float(features["MonthlyCharges"]),
            total_charges=None if total is None else float(total),
            churn=None,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
    )
    session.flush()


def score_saved_customers(state: RuntimeState, session: Session) -> int:
    """Replace import scores. Manual and batch history are kept."""
    state = require_model(state)
    customers = session.scalars(select(Customer).order_by(Customer.customer_id)).all()
    if not customers:
        return 0
    session.execute(delete(Prediction).where(Prediction.source == "import"))
    session.commit()
    records = []
    for customer in customers:
        features = customer_to_features(customer)
        features["customerID"] = customer.customer_id
        records.append(features)
    # Persisting explanations for every saved customer is unnecessary at import time.
    score_records(state, session, records, explain=False, prefer_shap=False, source="import", persist=True)
    return len(records)


def explain_customer(state: RuntimeState, session: Session, customer: Customer) -> dict:
    state = require_model(state)
    features = customer_to_features(customer)
    features["customerID"] = customer.customer_id
    results = score_records(
        state,
        session,
        [features],
        explain=True,
        prefer_shap=True,
        source="explanation",
        persist=True,
    )
    return results[0]


def parse_batch_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise api_error(400, "empty_file", "The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, encoding="utf-8-sig")
    except Exception as exc:
        raise api_error(400, "invalid_csv", "The file could not be read as CSV.") from exc
    if frame.empty:
        raise api_error(400, "empty_file", "The CSV has a header but no data rows.")
    return frame


def batch_records_from_frame(frame: pd.DataFrame) -> tuple[list[dict], list[str]]:
    from ml.data.loader import normalize_columns

    renamed, notes = normalize_columns(frame)
    records = renamed.where(pd.notna(renamed), None).to_dict(orient="records")
    cleaned_records = []
    for record in records:
        cleaned_records.append({key: (None if value is None else value) for key, value in record.items()})
    return cleaned_records, notes


def new_batch_id() -> str:
    return str(uuid.uuid4())


def relabel_stored_predictions(session: Session, medium: float, high: float) -> int:
    rows = session.scalars(select(Prediction).where(Prediction.status == "scored")).all()
    for row in rows:
        if row.probability is None:
            continue
        row.risk_category = risk_category(float(row.probability), medium, high)
        row.predicted_class = predicted_label(float(row.probability), high)
    session.commit()
    return len(rows)


def prediction_to_dict(row: Prediction) -> dict:
    def load(value, fallback):
        if not value:
            return fallback
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback

    return {
        "id": row.id,
        "customer_id": row.customer_id or row.external_id,
        "batch_id": row.batch_id,
        "row_number": row.row_number,
        "probability": row.probability,
        "risk_category": row.risk_category,
        "predicted_class": row.predicted_class,
        "priority_score": row.priority_score,
        "factors": load(row.top_factors, []),
        "recommendations": load(row.recommendations, []),
        "reliability": load(row.reliability, {}),
        "explanation_method": row.explanation_method,
        "status": row.status,
        "errors": load(row.errors, []),
        "warnings": load(row.warnings, []),
        "source": row.source,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
