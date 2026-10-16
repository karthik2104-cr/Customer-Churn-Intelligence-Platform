"""Customer import, directory queries, and the in-memory analysis table."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.tables import Customer, DataImport, Prediction, utcnow
from app.services.mapping import CSV_FIELDS, customer_to_features, frame_to_customer_rows
from ml.business.segments import RULES, UnknownSegmentError, customer_ids_for_rule
from ml.data.loader import clean_customers
from ml.data.quality import build_quality_report
from ml.data.schema import CATEGORICAL_FEATURES, DATASET_NAME, ID_COLUMN
from ml.evaluation.serialize import to_jsonable

SORT_FIELDS = {
    "customer_id": Customer.customer_id,
    "tenure": Customer.tenure,
    "monthly_charges": Customer.monthly_charges,
    "contract": Customer.contract,
    "probability": Prediction.probability,
    "priority_score": Prediction.priority_score,
    "risk_category": Prediction.risk_category,
}


def infer_dataset_label(filename: str, frame: pd.DataFrame) -> str:
    lowered = filename.lower()
    if "synthetic" in lowered:
        return "synthetic"
    if ID_COLUMN in frame.columns:
        identifiers = frame[ID_COLUMN].fillna("").astype(str)
        if len(identifiers) and float(identifiers.str.startswith("SYN-").mean()) > 0.5:
            return "synthetic"
    return DATASET_NAME


def _usable_mask(frame: pd.DataFrame) -> pd.Series:
    mask = frame[ID_COLUMN].notna() & frame["tenure"].notna() & frame["MonthlyCharges"].notna()
    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            mask = mask & frame[column].notna()
    return mask


def import_raw_frame(session: Session, raw: pd.DataFrame, filename: str) -> dict[str, Any]:
    cleaned, notes = clean_customers(raw, require_target=False, require_identifier=True)
    quality = build_quality_report(raw, cleaned, notes)
    label = infer_dataset_label(filename, cleaned)
    quality["dataset_label"] = label
    mask = _usable_mask(cleaned)
    invalid = cleaned.loc[~mask]
    usable = cleaned.loc[mask].copy()
    duplicate_count = int(usable[ID_COLUMN].duplicated(keep="last").sum()) if len(usable) else 0
    usable = usable.drop_duplicates(ID_COLUMN, keep="last")
    rows = frame_to_customer_rows(usable)
    stored = _upsert_customers(session, rows)
    message = (
        f"Stored {stored} customers. {int((~mask).sum())} rows were rejected for missing identifiers or features. "
        f"{duplicate_count} duplicate ids were reported and the last copy was kept."
    )
    record = DataImport(
        filename=filename,
        dataset_label=label,
        row_count=int(len(raw)),
        stored_count=stored,
        duplicate_count=duplicate_count,
        invalid_count=int((~mask).sum()),
        quality_report=json.dumps(to_jsonable(quality)),
        status="completed",
        message=message,
        created_at=utcnow(),
    )
    session.add(record)
    session.commit()
    examples = []
    for _, row in invalid.head(15).iterrows():
        examples.append(
            {
                "customer_id": None if pd.isna(row.get(ID_COLUMN)) else str(row.get(ID_COLUMN)),
                "reason": "Missing customer id, tenure, monthly charges, or a required service field.",
            }
        )
    return {
        "import_id": record.id,
        "dataset_label": label,
        "filename": filename,
        "rows_read": int(len(raw)),
        "stored_customers": stored,
        "duplicate_count": duplicate_count,
        "invalid_count": int((~mask).sum()),
        "invalid_examples": examples,
        "message": message,
        "quality": quality,
    }


def _upsert_customers(session: Session, rows: list[dict]) -> int:
    if not rows:
        return 0
    bind = session.get_bind()
    dialect = bind.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    update_columns = [column for column in rows[0] if column != "customer_id"]
    for start in range(0, len(rows), 300):
        chunk = rows[start : start + 300]
        for row in chunk:
            row["updated_at"] = utcnow()
            row.setdefault("created_at", utcnow())
        statement = dialect_insert(Customer).values(chunk)
        assignments = {column: statement.excluded[column] for column in update_columns}
        assignments["updated_at"] = statement.excluded.updated_at
        statement = statement.on_conflict_do_update(
            index_elements=["customer_id"],
            set_=assignments,
        )
        session.execute(statement)
    session.commit()
    return len(rows)


def latest_prediction_subquery():
    return (
        select(func.max(Prediction.id).label("prediction_id"), Prediction.customer_id.label("customer_id"))
        .where(Prediction.status == "scored", Prediction.customer_id.is_not(None))
        .group_by(Prediction.customer_id)
        .subquery()
    )


def list_customers(
    session: Session,
    *,
    page: int,
    page_size: int,
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
) -> dict[str, Any]:
    latest = latest_prediction_subquery()
    statement = (
        select(Customer, Prediction)
        .outerjoin(latest, latest.c.customer_id == Customer.customer_id)
        .outerjoin(Prediction, Prediction.id == latest.c.prediction_id)
    )
    if search:
        statement = statement.where(Customer.customer_id.contains(search.strip()))
    if contract:
        statement = statement.where(Customer.contract == contract)
    if internet_service:
        statement = statement.where(Customer.internet_service == internet_service)
    if observed_churn in {"Yes", "No"}:
        statement = statement.where(Customer.churn == observed_churn)
    if risk in {"Low", "Medium", "High"}:
        statement = statement.where(Prediction.risk_category == risk)
    if tenure_min is not None:
        statement = statement.where(Customer.tenure >= tenure_min)
    if tenure_max is not None:
        statement = statement.where(Customer.tenure <= tenure_max)
    if monthly_min is not None:
        statement = statement.where(Customer.monthly_charges >= monthly_min)
    if monthly_max is not None:
        statement = statement.where(Customer.monthly_charges <= monthly_max)
    sort_column = SORT_FIELDS.get(sort, Prediction.priority_score)
    sort_expression = sort_column.asc() if order == "asc" else sort_column.desc()
    ordered = statement.order_by(sort_expression, Customer.customer_id.asc())
    segment_info = _segment_membership(session, segment) if segment else None
    if segment_info is not None:
        member_ids = set(segment_info.pop("ids"))
        matched = [
            (customer, prediction)
            for customer, prediction in session.execute(ordered).all()
            if customer.customer_id in member_ids
        ]
        total = len(matched)
        start = (page - 1) * page_size
        rows = matched[start : start + page_size]
    else:
        count_statement = select(func.count()).select_from(statement.order_by(None).subquery())
        total = int(session.scalar(count_statement) or 0)
        rows = session.execute(ordered.offset((page - 1) * page_size).limit(page_size)).all()
    return {
        "items": [_list_item(customer, prediction) for customer, prediction in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "segment": segment_info,
    }


def _segment_membership(session: Session, segment: str) -> dict[str, Any]:
    rule = next((item for item in RULES if item["id"] == segment), None)
    if rule is None:
        raise UnknownSegmentError(segment)
    frame = population_frame(session)
    charges = (
        pd.to_numeric(frame["MonthlyCharges"], errors="coerce").dropna()
        if "MonthlyCharges" in frame.columns
        else pd.Series(dtype=float)
    )
    cutoff = float(charges.quantile(0.75)) if len(charges) else None
    ids = [] if cutoff is None else customer_ids_for_rule(frame, cutoff, segment)
    return {
        "id": rule["id"],
        "name": rule["name"],
        "definition": rule["definition"],
        "members": len(ids),
        "ids": ids,
    }


def _list_item(customer: Customer, prediction: Prediction | None) -> dict:
    return {
        "customer_id": customer.customer_id,
        "contract": customer.contract,
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "total_charges": customer.total_charges,
        "internet_service": customer.internet_service,
        "payment_method": customer.payment_method,
        "paperless_billing": customer.paperless_billing,
        "tech_support": customer.tech_support,
        "observed_churn": customer.churn,
        "probability": None if prediction is None else prediction.probability,
        "risk_category": None if prediction is None else prediction.risk_category,
        "predicted_class": None if prediction is None else prediction.predicted_class,
        "priority_score": None if prediction is None else prediction.priority_score,
    }


def population_frame(session: Session) -> pd.DataFrame:
    latest = latest_prediction_subquery()
    rows = session.execute(
        select(Customer, Prediction)
        .outerjoin(latest, latest.c.customer_id == Customer.customer_id)
        .outerjoin(Prediction, Prediction.id == latest.c.prediction_id)
    ).all()
    records = []
    for customer, prediction in rows:
        features = customer_to_features(customer)
        features["probability"] = None if prediction is None else prediction.probability
        features["risk_category"] = None if prediction is None else prediction.risk_category
        features["predicted_class"] = None if prediction is None else prediction.predicted_class
        features["priority_score"] = None if prediction is None else prediction.priority_score
        records.append(features)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        frame = pd.DataFrame(columns=list(CSV_FIELDS.values()) + ["probability", "risk_category", "predicted_class", "priority_score"])
    return frame


def customer_count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(Customer)) or 0)


def latest_import(session: Session) -> DataImport | None:
    return session.scalar(select(DataImport).order_by(DataImport.id.desc()).limit(1))
