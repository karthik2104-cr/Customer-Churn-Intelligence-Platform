"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from app import __version__
from app.api.body_limit import BodySizeLimitMiddleware
from app.api.routes import import_configured_csv, router
from app.core.config import get_settings
from app.core.database import get_engine, init_db, session_factory
from app.models.tables import Prediction
from app.services.customers import customer_count, latest_import
from app.services.runtime import get_runtime, load_model, record_model, sync_settings
from app.services.scoring import relabel_stored_predictions, score_saved_customers

logger = logging.getLogger("churn")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    init_db()
    state = load_model()
    session = session_factory()()
    try:
        if settings.auto_import and customer_count(session) == 0:
            import_configured_csv(session)
        else:
            record = latest_import(session)
            if record is not None:
                state.dataset_label = record.dataset_label
            settings_row = sync_settings(session, state)
            if state.model_loaded:
                relabel_stored_predictions(
                    session,
                    float(settings_row.medium_threshold),
                    float(settings_row.high_threshold),
                )
            imported_scores = int(
                session.scalar(
                    select(func.count()).select_from(Prediction).where(Prediction.source == "import", Prediction.status == "scored")
                )
                or 0
            )
            if state.model_loaded and customer_count(session) and imported_scores == 0:
                score_saved_customers(state, session)
        if state.model_loaded:
            record_model(session, state)
    finally:
        session.close()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Decision-support API for telco customer churn risk, explanations, and retention scenarios.",
        lifespan=lifespan,
    )
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "The request is invalid.",
                "errors": _public_validation_errors(exc),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict):
            content = exc.detail
        else:
            content = {"code": "error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "The request could not be completed."},
        )

    return app


app = create_app()


def _public_validation_errors(exc: RequestValidationError) -> list[dict[str, object]]:
    """Return field locations and messages only. Omit submitted values and exception context."""
    public: list[dict[str, object]] = []
    for item in exc.errors():
        public.append(
            {
                "type": str(item.get("type", "")),
                "loc": [str(part) for part in item.get("loc", ())],
                "msg": str(item.get("msg", "")),
            }
        )
    return public


def _engine_ready() -> None:
    """Imported by tests that want the engine configured before the app starts."""
    get_engine()
