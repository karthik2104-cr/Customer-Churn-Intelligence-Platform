"""Persisted customers, scores, campaigns, and settings."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

CAMPAIGN_STATUSES = ("follow_up_required", "contacted", "offer_proposed", "retained")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    senior_citizen: Mapped[int | None] = mapped_column(Integer, nullable=True)
    partner: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dependents: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tenure: Mapped[int] = mapped_column(Integer, index=True)
    phone_service: Mapped[str | None] = mapped_column(String(32), nullable=True)
    multiple_lines: Mapped[str | None] = mapped_column(String(64), nullable=True)
    internet_service: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    online_security: Mapped[str | None] = mapped_column(String(64), nullable=True)
    online_backup: Mapped[str | None] = mapped_column(String(64), nullable=True)
    device_protection: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tech_support: Mapped[str | None] = mapped_column(String(64), nullable=True)
    streaming_tv: Mapped[str | None] = mapped_column(String(64), nullable=True)
    streaming_movies: Mapped[str | None] = mapped_column(String(64), nullable=True)
    contract: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    paperless_billing: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payment_method: Mapped[str | None] = mapped_column(String(80), nullable=True)
    monthly_charges: Mapped[float] = mapped_column(Float, index=True)
    total_charges: Mapped[float | None] = mapped_column(Float, nullable=True)
    churn: Mapped[str | None] = mapped_column(String(8), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="customer")
    campaigns: Mapped[list["Campaign"]] = relationship(back_populates="customer")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id"), index=True, nullable=True
    )
    external_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    row_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, index=True, nullable=True)
    risk_category: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    predicted_class: Mapped[str | None] = mapped_column(String(8), nullable=True)
    priority_score: Mapped[float | None] = mapped_column(Float, index=True, nullable=True)
    top_factors: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendations: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reliability: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="scored", index=True)
    errors: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="single")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    customer: Mapped[Customer | None] = relationship(back_populates="predictions")


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    campaign_name: Mapped[str] = mapped_column(String(120), default="Retention queue", index=True)
    action: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str] = mapped_column(Text)
    effort: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(32), default="retention")
    status: Mapped[str] = mapped_column(String(32), default="follow_up_required", index=True)
    priority_score: Mapped[float] = mapped_column(Float, default=0, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    customer: Mapped[Customer] = relationship(back_populates="campaigns")


class AppSetting(Base):
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    medium_threshold: Mapped[float] = mapped_column(Float, default=0.30)
    high_threshold: Mapped[float] = mapped_column(Float, default=0.50)
    thresholds_customized: Mapped[int] = mapped_column(Integer, default=0)
    avg_monthly_revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_customized: Mapped[int] = mapped_column(Integer, default=0)
    contribution_margin: Mapped[float] = mapped_column(Float, default=0.70)
    campaign_cost: Mapped[float] = mapped_column(Float, default=25.0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.30)
    horizon_months: Mapped[int] = mapped_column(Integer, default=12)
    intervention_capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DataImport(Base):
    __tablename__ = "data_imports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255))
    dataset_label: Mapped[str] = mapped_column(String(120))
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    stored_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    invalid_count: Mapped[int] = mapped_column(Integer, default=0)
    quality_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ModelRecord(Base):
    __tablename__ = "model_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(80))
    trained_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(400))
    test_average_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    test_roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
