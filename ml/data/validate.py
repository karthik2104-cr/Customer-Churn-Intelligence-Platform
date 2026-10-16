"""Row-level checks shared by the API and batch scoring."""

from __future__ import annotations

import math
from typing import Any

from ml.data.schema import CATEGORICAL_FEATURES, ID_COLUMN, NUMERIC_FEATURES


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "na", "n/a"}


def _match_level(value: Any, levels: list[str]) -> tuple[str | None, bool]:
    text = str(value).strip()
    if text in levels:
        return text, False
    for level in levels:
        if text.lower() == str(level).lower():
            return str(level), True
    return None, False


def validate_record(record: dict, feature_spec: dict) -> tuple[dict | None, list[str], list[str]]:
    """Return a model-ready feature dict, errors, and warnings.

    Invalid rows are not repaired into a score. Blank TotalCharges with tenure 0
    become 0 and produce a warning, matching the training cleaner.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cleaned: dict[str, Any] = {}
    numeric_spec = feature_spec.get("numeric", {})
    categorical_spec = feature_spec.get("categorical", {})

    tenure_value = record.get("tenure")
    tenure: int | None = None
    if _is_missing(tenure_value):
        errors.append("tenure is required.")
    else:
        try:
            numeric = float(str(tenure_value).strip())
        except ValueError:
            errors.append("tenure must be a whole number of months.")
            numeric = None
        if numeric is not None:
            if not math.isfinite(numeric) or numeric < 0 or abs(numeric - round(numeric)) > 1e-6:
                errors.append("tenure must be a whole number of months, zero or greater.")
            elif numeric > 240:
                errors.append("tenure is above 240 months and was rejected as implausible for this schema.")
            else:
                tenure = int(round(numeric))
                cleaned["tenure"] = tenure

    monthly_value = record.get("MonthlyCharges")
    if _is_missing(monthly_value):
        errors.append("MonthlyCharges is required.")
    else:
        try:
            monthly = float(str(monthly_value).strip())
        except ValueError:
            errors.append("MonthlyCharges must be a number.")
        else:
            if not math.isfinite(monthly) or monthly < 0:
                errors.append("MonthlyCharges must be zero or greater.")
            else:
                cleaned["MonthlyCharges"] = monthly

    total_value = record.get("TotalCharges")
    if _is_missing(total_value):
        if tenure == 0:
            cleaned["TotalCharges"] = 0.0
            warnings.append("TotalCharges was blank and tenure is 0, so total charges were set to 0.")
        else:
            errors.append(
                "TotalCharges is missing. Provide a number, or leave it blank only when tenure is 0."
            )
    else:
        try:
            total = float(str(total_value).strip())
        except ValueError:
            errors.append("TotalCharges must be a number.")
        else:
            if not math.isfinite(total) or total < 0:
                errors.append("TotalCharges must be zero or greater.")
            else:
                cleaned["TotalCharges"] = total

    for column in CATEGORICAL_FEATURES:
        levels = list(categorical_spec.get(column, []))
        value = record.get(column)
        if _is_missing(value):
            errors.append(f"{column} is required.")
            continue
        if not levels:
            cleaned[column] = str(value).strip()
            continue
        matched, changed_case = _match_level(value, levels)
        if matched is None:
            preview = ", ".join(str(level) for level in levels[:8])
            errors.append(f"{column} has an unrecognized value '{value}'. Expected one of: {preview}.")
        else:
            cleaned[column] = matched
            if changed_case:
                warnings.append(f"{column} was matched to '{matched}' after ignoring letter case.")

    if errors:
        return None, errors, warnings

    for column in NUMERIC_FEATURES:
        bounds = numeric_spec.get(column)
        if not bounds or column not in cleaned:
            continue
        value = cleaned[column]
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(
                f"{column} is outside the range seen while fitting the deployed model "
                f"({bounds['min']} to {bounds['max']}). The score will be marked as less reliable."
            )
    return cleaned, errors, warnings


def record_identifier(record: dict) -> str | None:
    value = record.get(ID_COLUMN, record.get("customer_id"))
    if _is_missing(value):
        return None
    return str(value).strip()
