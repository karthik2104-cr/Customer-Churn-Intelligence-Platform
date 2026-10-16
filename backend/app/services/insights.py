"""Dashboard, analytics, segments, campaigns, and financial scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.errors import api_error
from app.models.tables import CAMPAIGN_STATUSES, Campaign, Customer, Prediction, utcnow
from app.services.customers import population_frame
from app.services.runtime import RuntimeState, ensure_settings
from ml.business.impact import ImpactAssumptions, calculate_impact
from ml.business.recommendations import recommend
from ml.business.segments import assign_clusters, summarize_rules
from ml.data.eda import apply_population_filters, build_eda, quartile_edges
from ml.data.schema import DECISION_SUPPORT_WARNING, TARGET_COLUMN


def _observed_rate(frame: pd.DataFrame) -> tuple[int, int, float | None]:
    if frame.empty or TARGET_COLUMN not in frame.columns:
        return 0, 0, None
    labeled = frame[frame[TARGET_COLUMN].isin(["Yes", "No"])]
    churned = int((labeled[TARGET_COLUMN] == "Yes").sum())
    return int(len(labeled)), churned, (churned / len(labeled)) if len(labeled) else None


def _risk_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "risk_category" not in frame.columns:
        return {"High": 0, "Medium": 0, "Low": 0, "Unscored": int(len(frame))}
    counts = frame["risk_category"].fillna("Unscored").value_counts().to_dict()
    return {
        "High": int(counts.get("High", 0)),
        "Medium": int(counts.get("Medium", 0)),
        "Low": int(counts.get("Low", 0)),
        "Unscored": int(counts.get("Unscored", 0)),
    }


def recent_activity(session: Session, limit: int = 8) -> list[dict]:
    predictions = session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)).all()
    campaigns = session.scalars(select(Campaign).order_by(Campaign.updated_at.desc()).limit(limit)).all()
    events = []
    for row in predictions:
        events.append(
            {
                "type": "prediction",
                "at": row.created_at.isoformat() if row.created_at else None,
                "customer_id": row.customer_id,
                "summary": (
                    f"{row.source} score"
                    + (f" {row.probability:.1%} {row.risk_category}" if row.probability is not None else " invalid row")
                ),
            }
        )
    for row in campaigns:
        events.append(
            {
                "type": "campaign",
                "at": row.updated_at.isoformat() if row.updated_at else None,
                "customer_id": row.customer_id,
                "summary": f"{row.campaign_name}: {row.status.replace('_', ' ')}",
            }
        )
    events.sort(key=lambda item: item["at"] or "", reverse=True)
    return events[:limit]


def dashboard_summary(session: Session, dataset_label: str | None) -> dict[str, Any]:
    frame = population_frame(session)
    labeled, churned, rate = _observed_rate(frame)
    risks = _risk_counts(frame)
    scored = frame["probability"].dropna() if "probability" in frame.columns else pd.Series(dtype=float)
    follow_ups = int(
        session.scalar(select(func.count()).select_from(Campaign).where(Campaign.status == "follow_up_required")) or 0
    )
    eda = build_eda(frame) if len(frame) else None
    priority = []
    if not frame.empty and "priority_score" in frame.columns:
        ranked = frame.dropna(subset=["priority_score"]).sort_values("priority_score", ascending=False).head(8)
        for record in ranked.to_dict(orient="records"):
            priority.append(
                {
                    "customer_id": record.get("customerID"),
                    "probability": record.get("probability"),
                    "risk_category": record.get("risk_category"),
                    "priority_score": record.get("priority_score"),
                    "contract": record.get("Contract"),
                    "tenure": record.get("tenure"),
                    "monthly_charges": record.get("MonthlyCharges"),
                    "observed_churn": record.get("Churn"),
                }
            )
    return {
        "dataset_label": dataset_label,
        "total_customers": int(len(frame)),
        "observed_labeled_customers": labeled,
        "observed_churn_count": churned,
        "observed_retained_count": labeled - churned,
        "observed_churn_rate": rate,
        "predictions_available": bool(len(scored)),
        "high_risk_count": risks["High"],
        "medium_risk_count": risks["Medium"],
        "low_risk_count": risks["Low"],
        "unscored_count": risks["Unscored"],
        "average_predicted_probability": float(scored.mean()) if len(scored) else None,
        "follow_up_required": follow_ups,
        "contract_churn": [] if eda is None else eda["by_contract"],
        "risk_breakdown": [
            {"name": name, "customers": risks[name]} for name in ("High", "Medium", "Low", "Unscored")
        ],
        "priority_customers": priority,
        "recent_activity": recent_activity(session),
        "time_series_available": False,
        "time_series_note": "This table is a snapshot with no signup or churn date, so no monthly trend is shown.",
        "decision_support_warning": DECISION_SUPPORT_WARNING,
        "denominator_note": (
            "Observed churn uses historical Yes/No labels. Risk counts use the current model score. "
            "Those are different measures."
        ),
    }


def analytics_payload(session: Session, filters: dict) -> dict[str, Any]:
    frame = population_frame(session)
    monthly_edges = quartile_edges(frame["MonthlyCharges"]) if len(frame) else [0, 1]
    total_edges = quartile_edges(frame["TotalCharges"]) if len(frame) else [0, 1]
    filtered = apply_population_filters(frame, **filters) if len(frame) else frame
    report = build_eda(filtered, monthly_edges=monthly_edges, total_edges=total_edges)
    risks = _risk_counts(filtered)
    scored = filtered["probability"].dropna() if "probability" in filtered.columns and len(filtered) else pd.Series(dtype=float)
    report["predicted"] = {
        "scored_customers": int(len(scored)),
        "average_probability": float(scored.mean()) if len(scored) else None,
        "risk_counts": risks,
    }
    report["filters"] = filters
    report["population_before_filters"] = int(len(frame))
    return report


def segment_payload(session: Session, state: RuntimeState) -> dict[str, Any]:
    frame = population_frame(session)
    if frame.empty:
        return {"rules": [], "clusters": [], "high_charge_cutoff": None, "dataset_label": state.dataset_label}
    charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").dropna()
    cutoff = float(charges.quantile(0.75)) if len(charges) else None
    rules = summarize_rules(frame, cutoff) if cutoff is not None else []
    clusters: list[dict] = []
    note = "Rule segments are the operational view. A customer can match more than one rule."
    if state.segments is not None and len(frame):
        labels = assign_clusters(frame, state.segments)
        working = frame.copy()
        working["cluster_label"] = labels
        for label, group in working.groupby("cluster_label"):
            labeled = group[group[TARGET_COLUMN].isin(["Yes", "No"])]
            churned = int((labeled[TARGET_COLUMN] == "Yes").sum()) if len(labeled) else 0
            clusters.append(
                {
                    "label": str(label),
                    "customers": int(len(group)),
                    "observed_churned": churned,
                    "observed_churn_rate": (churned / len(labeled)) if len(labeled) else None,
                    "average_tenure": float(pd.to_numeric(group["tenure"], errors="coerce").mean()),
                    "average_monthly_charges": float(pd.to_numeric(group["MonthlyCharges"], errors="coerce").mean()),
                    "average_predicted_probability": (
                        float(group["probability"].dropna().mean()) if group["probability"].notna().any() else None
                    ),
                }
            )
        clusters.sort(key=lambda item: item["customers"], reverse=True)
        meta_note = ((state.metadata or {}).get("segments") or {}).get("note")
        artifact_note = state.segments.get("note") if isinstance(state.segments, dict) else None
        note = artifact_note or meta_note or note
    return {
        "high_charge_cutoff": cutoff,
        "cutoff_note": "The high-charge cutoff is the 75th percentile of monthly charges in the current customer table.",
        "rules": rules,
        "clusters": clusters,
        "cluster_note": note,
        "dataset_label": state.dataset_label,
    }


def campaign_dict(row: Campaign) -> dict:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "campaign_name": row.campaign_name,
        "action": row.action,
        "reason": row.reason,
        "effort": row.effort,
        "category": row.category,
        "status": row.status,
        "priority_score": row.priority_score,
        "notes": row.notes,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "outcomes_available": False,
        "outcomes_note": "A retained status is a user record. It is not a measured save rate.",
    }


def list_campaigns(session: Session, status: str | None, search: str | None) -> list[dict]:
    statement = select(Campaign).order_by(Campaign.priority_score.desc(), Campaign.id.desc())
    if status:
        statement = statement.where(Campaign.status == status)
    if search:
        statement = statement.where(Campaign.customer_id.contains(search.strip()))
    return [campaign_dict(row) for row in session.scalars(statement).all()]


def create_campaign(session: Session, payload: dict) -> dict:
    customer = session.get(Customer, payload["customer_id"])
    if customer is None:
        raise api_error(404, "customer_not_found", "No stored customer matches that id.")
    if payload["status"] not in CAMPAIGN_STATUSES:
        raise api_error(400, "invalid_status", "Status must be a supported follow-up state.")
    row = Campaign(
        customer_id=customer.customer_id,
        campaign_name=payload.get("campaign_name") or "Retention queue",
        action=payload["action"],
        reason=payload["reason"],
        effort=payload.get("effort") or "medium",
        category=payload.get("category") or "retention",
        status=payload["status"],
        priority_score=float(payload.get("priority_score") or 0),
        notes=payload.get("notes"),
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return campaign_dict(row)


def update_campaign(session: Session, campaign_id: int, payload: dict) -> dict:
    row = session.get(Campaign, campaign_id)
    if row is None:
        raise api_error(404, "campaign_not_found", "That campaign does not exist.")
    if "status" in payload and payload["status"] is not None:
        if payload["status"] not in CAMPAIGN_STATUSES:
            raise api_error(400, "invalid_status", "Status must be a supported follow-up state.")
        row.status = payload["status"]
    if "notes" in payload:
        row.notes = payload["notes"]
    if payload.get("campaign_name"):
        row.campaign_name = payload["campaign_name"]
    row.updated_at = utcnow()
    session.commit()
    session.refresh(row)
    return campaign_dict(row)


def suggest_campaigns(session: Session, state: RuntimeState, limit: int, campaign_name: str) -> dict:
    frame = population_frame(session)
    if frame.empty or "probability" not in frame.columns:
        raise api_error(409, "scores_unavailable", "Score the customer table before building a retention queue.")
    open_ids = set(
        session.scalars(
            select(Campaign.customer_id).where(Campaign.status.in_(["follow_up_required", "contacted", "offer_proposed"]))
        ).all()
    )
    charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    cutoff = float(charges.quantile(0.75)) if charges.notna().any() else None
    ranked = frame[frame["risk_category"].isin(["High", "Medium"])].copy()
    ranked = ranked.sort_values(["priority_score", "probability"], ascending=False)
    created = []
    associations = (state.metadata or {}).get("associations")
    settings = ensure_settings(session)
    for record in ranked.to_dict(orient="records"):
        if len(created) >= limit:
            break
        customer_id = record.get("customerID")
        if not customer_id or customer_id in open_ids:
            continue
        recommendations = recommend(
            record,
            float(record["probability"]),
            float(settings.medium_threshold),
            float(settings.high_threshold),
            high_charge_cutoff=cutoff,
            associations=associations,
        )
        primary = recommendations[0]
        created.append(
            create_campaign(
                session,
                {
                    "customer_id": customer_id,
                    "campaign_name": campaign_name,
                    "action": primary.action,
                    "reason": primary.reason,
                    "effort": primary.effort,
                    "category": primary.category,
                    "status": "follow_up_required",
                    "priority_score": float(record.get("priority_score") or 0),
                },
            )
        )
        open_ids.add(customer_id)
    return {
        "created": len(created),
        "campaigns": created,
        "outcomes_available": False,
        "outcomes_note": "Suggestions are rules over model scores. No campaign result is invented.",
    }


def business_impact(session: Session) -> dict[str, Any]:
    settings = ensure_settings(session)
    frame = population_frame(session)
    labeled, churned, rate = _observed_rate(frame)
    if frame.empty or "risk_category" not in frame.columns:
        targeted = np.array([], dtype=float)
        at_risk = 0
    else:
        mask = frame["risk_category"].isin(["Medium", "High"])
        at_risk = int(mask.sum())
        targeted = pd.to_numeric(frame.loc[mask, "probability"], errors="coerce").dropna().to_numpy(dtype=float)
    average_reference = None
    if len(frame) and "MonthlyCharges" in frame.columns:
        values = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").dropna()
        average_reference = float(values.mean()) if len(values) else None
    revenue = settings.avg_monthly_revenue or average_reference or 0
    assumptions = ImpactAssumptions(
        avg_monthly_revenue=float(revenue),
        contribution_margin=float(settings.contribution_margin),
        campaign_cost=float(settings.campaign_cost),
        success_rate=float(settings.success_rate),
        horizon_months=int(settings.horizon_months),
        intervention_capacity=settings.intervention_capacity,
    )
    payload = calculate_impact(targeted, assumptions)
    payload["predictions_available"] = bool(len(targeted))
    payload["reference"] = {
        "customers_total": int(len(frame)),
        "customers_at_risk_before_capacity": at_risk,
        "data_average_monthly_charges": average_reference,
        "observed_labeled_customers": labeled,
        "observed_churned": churned,
        "observed_churn_rate": rate,
        "revenue_source": "user" if settings.revenue_customized else "data_average_unless_edited",
    }
    return payload
