"""Process-wide model state. The model file is loaded from configuration, never from an upload."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.tables import AppSetting, Customer, ModelRecord, utcnow

logger = logging.getLogger(__name__)


class RuntimeState:
    def __init__(self) -> None:
        self.pipeline = None
        self.bundle: dict = {}
        self.metadata: dict = {}
        self.segments = None
        self.model_loaded = False
        self.load_error: str | None = None
        self.dataset_label: str | None = None

    def clear(self) -> None:
        self.__init__()


STATE = RuntimeState()


def get_runtime() -> RuntimeState:
    return STATE


def reset_runtime() -> None:
    STATE.clear()


def load_model(state: RuntimeState | None = None) -> RuntimeState:
    state = state or STATE
    settings = get_settings()
    model_path = settings.resolved_model_path()
    state.pipeline = None
    state.bundle = {}
    state.metadata = {}
    state.segments = None
    state.model_loaded = False
    state.load_error = None
    if not model_path.exists():
        logger.error("Configured model file is missing: %s", model_path)
        state.load_error = "No model file is available. Train the model before starting the API."
        return state
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        state.load_error = f"The model file could not be read ({type(exc).__name__})."
        logger.exception("Model load failed")
        return state
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        state.load_error = "The model file does not contain a deployed pipeline."
        return state
    state.bundle = bundle
    state.pipeline = bundle["pipeline"]
    metadata_path = settings.resolved_metadata_path()
    if metadata_path.exists():
        state.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        state.metadata = {}
        state.load_error = "Model weights loaded, but metadata.json is missing."
    segments_path = settings.resolved_segments_path()
    if segments_path.exists():
        try:
            state.segments = joblib.load(segments_path)
        except Exception:
            logger.exception("Segment artifact could not be loaded")
            state.segments = None
    state.model_loaded = True
    return state


def ensure_settings(session: Session) -> AppSetting:
    row = session.get(AppSetting, 1)
    if row is None:
        row = AppSetting(id=1, updated_at=utcnow())
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def sync_settings(session: Session, state: RuntimeState | None = None) -> AppSetting:
    state = state or STATE
    row = ensure_settings(session)
    thresholds = (state.metadata or {}).get("thresholds") or {}
    medium = thresholds.get("medium")
    high = thresholds.get("high")
    if not row.thresholds_customized and medium is not None and high is not None and float(high) > float(medium):
        row.medium_threshold = float(medium)
        row.high_threshold = float(high)
    if not row.revenue_customized:
        average = session.scalar(select(func.avg(Customer.monthly_charges)))
        if average is not None:
            row.avg_monthly_revenue = round(float(average), 2)
    row.updated_at = utcnow()
    session.commit()
    return row


def record_model(session: Session, state: RuntimeState) -> None:
    if not state.model_loaded:
        return
    metrics = (state.metadata or {}).get("test_metrics") or {}
    session.add(
        ModelRecord(
            model_name=str((state.metadata or {}).get("selected_model") or "unknown"),
            trained_at=(state.metadata or {}).get("trained_at"),
            artifact_path=str(get_settings().resolved_model_path()),
            test_average_precision=metrics.get("average_precision"),
            test_roc_auc=metrics.get("roc_auc"),
            created_at=utcnow(),
        )
    )
    session.commit()


def active_thresholds(session: Session) -> tuple[float, float]:
    row = ensure_settings(session)
    return float(row.medium_threshold), float(row.high_threshold)


def require_model(state: RuntimeState | None = None) -> RuntimeState:
    from app.api.errors import api_error

    state = state or STATE
    if not state.model_loaded or state.pipeline is None:
        message = state.load_error or "Train a model with python -m ml.training.train before scoring customers."
        raise api_error(503, "model_not_loaded", message)
    return state


def artifact_path_is_configured(path: Path) -> bool:
    """Uploads must never be passed to joblib. Only the configured artifact path is loaded."""
    return path.resolve() == get_settings().resolved_model_path().resolve()
