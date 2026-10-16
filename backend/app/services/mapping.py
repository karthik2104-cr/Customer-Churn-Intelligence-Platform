"""Translate between CSV names and database fields."""

from __future__ import annotations

import math

import pandas as pd

from app.models.tables import Customer
from ml.data.schema import MODEL_FEATURES

CSV_FIELDS = {
    "customer_id": "customerID",
    "gender": "gender",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "churn": "Churn",
}


def _none_if_nan(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def frame_to_customer_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for record in frame.to_dict(orient="records"):
        senior = _none_if_nan(record.get("SeniorCitizen"))
        total = _none_if_nan(record.get("TotalCharges"))
        churn = _none_if_nan(record.get("Churn"))
        rows.append(
            {
                "customer_id": str(record["customerID"]).strip(),
                "gender": _none_if_nan(record.get("gender")),
                "senior_citizen": None if senior is None else int(senior),
                "partner": _none_if_nan(record.get("Partner")),
                "dependents": _none_if_nan(record.get("Dependents")),
                "tenure": int(round(float(record["tenure"]))),
                "phone_service": _none_if_nan(record.get("PhoneService")),
                "multiple_lines": _none_if_nan(record.get("MultipleLines")),
                "internet_service": _none_if_nan(record.get("InternetService")),
                "online_security": _none_if_nan(record.get("OnlineSecurity")),
                "online_backup": _none_if_nan(record.get("OnlineBackup")),
                "device_protection": _none_if_nan(record.get("DeviceProtection")),
                "tech_support": _none_if_nan(record.get("TechSupport")),
                "streaming_tv": _none_if_nan(record.get("StreamingTV")),
                "streaming_movies": _none_if_nan(record.get("StreamingMovies")),
                "contract": _none_if_nan(record.get("Contract")),
                "paperless_billing": _none_if_nan(record.get("PaperlessBilling")),
                "payment_method": _none_if_nan(record.get("PaymentMethod")),
                "monthly_charges": float(record["MonthlyCharges"]),
                "total_charges": None if total is None else float(total),
                "churn": None if churn is None else str(churn),
            }
        )
    return rows


def customer_to_features(customer: Customer) -> dict:
    return {
        "customerID": customer.customer_id,
        "gender": customer.gender,
        "SeniorCitizen": customer.senior_citizen,
        "Partner": customer.partner,
        "Dependents": customer.dependents,
        "tenure": customer.tenure,
        "PhoneService": customer.phone_service,
        "MultipleLines": customer.multiple_lines,
        "InternetService": customer.internet_service,
        "OnlineSecurity": customer.online_security,
        "OnlineBackup": customer.online_backup,
        "DeviceProtection": customer.device_protection,
        "TechSupport": customer.tech_support,
        "StreamingTV": customer.streaming_tv,
        "StreamingMovies": customer.streaming_movies,
        "Contract": customer.contract,
        "PaperlessBilling": customer.paperless_billing,
        "PaymentMethod": customer.payment_method,
        "MonthlyCharges": customer.monthly_charges,
        "TotalCharges": customer.total_charges if customer.total_charges is not None else float("nan"),
        "Churn": customer.churn,
    }


def features_to_model_row(features: dict) -> dict:
    row = {}
    for column in MODEL_FEATURES:
        value = features.get(column)
        row[column] = float("nan") if value is None else value
    return row
