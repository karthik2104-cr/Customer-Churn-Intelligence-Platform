"""CSV loading and dtype correction for the telco churn table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.schema import (
    CANONICAL_LOOKUP,
    CATEGORICAL_FEATURES,
    DATASET_NAME,
    ID_COLUMN,
    MODEL_FEATURES,
    PROTECTED_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
    SchemaError,
    column_key,
)

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise SchemaError("Only .csv files are accepted.")
    frame = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    if frame.empty:
        raise SchemaError("The CSV has a header but no data rows.")
    return frame


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    rename: dict[str, str] = {}
    for column in frame.columns:
        canonical = CANONICAL_LOOKUP.get(column_key(column))
        if canonical is None:
            notes.append(f"Ignored unrecognized column '{column}'.")
            continue
        rename[column] = canonical
    renamed = frame.rename(columns=rename)
    if renamed.columns.duplicated().any():
        duplicates = renamed.columns[renamed.columns.duplicated()].unique().tolist()
        raise SchemaError(f"Multiple columns map to the same field: {duplicates}")
    return renamed, notes


def _blank_mask(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return series.isna() | text.isin(MISSING_TOKENS)


def _to_number(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    lowered = text.str.lower()
    text = text.mask(lowered.isin(MISSING_TOKENS))
    return pd.to_numeric(text, errors="coerce")


def _normalize_senior(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    mapped = text.map({"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1})
    mapped = mapped.mask(text.isin(MISSING_TOKENS))
    return mapped.astype("Float64")


def clean_customers(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    require_identifier: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize names and dtypes. Does not drop outliers or impute model features."""
    renamed, notes = normalize_columns(frame)
    required = list(MODEL_FEATURES)
    if require_identifier:
        required.append(ID_COLUMN)
    if require_target:
        required.append(TARGET_COLUMN)
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise SchemaError("Missing required columns: " + ", ".join(missing))
    for column in PROTECTED_COLUMNS:
        if column not in renamed.columns:
            notes.append(
                f"{column} is absent. Aggregate charts that use it will be empty. "
                "It is not required for scoring."
            )

    keep = [column for column in RAW_COLUMNS if column in renamed.columns]
    cleaned = renamed.loc[:, keep].copy()

    for column in list(CATEGORICAL_FEATURES) + [ID_COLUMN, TARGET_COLUMN, "gender"]:
        if column not in cleaned.columns:
            continue
        text = cleaned[column].fillna("").astype(str).str.strip()
        lowered = text.str.lower()
        cleaned[column] = text.mask(lowered.isin(MISSING_TOKENS) | text.eq(""))
        cleaned.loc[cleaned[column].isna(), column] = np.nan

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = _normalize_senior(cleaned["SeniorCitizen"])

    tenure = _to_number(cleaned["tenure"])
    monthly = _to_number(cleaned["MonthlyCharges"])
    total_raw = cleaned["TotalCharges"]
    total_blank = _blank_mask(total_raw)
    total = _to_number(total_raw)
    tenure_zero = tenure.fillna(-1).eq(0)
    filled = total_blank & tenure_zero
    total = total.mask(filled, 0.0)
    if int(filled.sum()):
        notes.append(
            f"Set TotalCharges to 0 for {int(filled.sum())} customers with tenure 0 and a blank total. "
            "These rows are kept."
        )
    invalid_total = total.isna() & ~total_blank
    if int(invalid_total.sum()):
        notes.append(
            f"{int(invalid_total.sum())} TotalCharges values are non-numeric and will be treated as missing."
        )
    still_missing_total = int(total.isna().sum())
    if still_missing_total:
        notes.append(
            f"{still_missing_total} TotalCharges values remain missing after the tenure-0 correction. "
            "The training pipeline imputes them from the training split only."
        )

    cleaned["tenure"] = tenure
    cleaned["MonthlyCharges"] = monthly
    cleaned["TotalCharges"] = total

    if require_target and TARGET_COLUMN in cleaned.columns:
        invalid_target = ~cleaned[TARGET_COLUMN].isin(["Yes", "No"])
        if int(invalid_target.sum()):
            notes.append(
                f"{int(invalid_target.sum())} rows have a churn label other than Yes or No "
                "and cannot be used for training."
            )
    return cleaned, notes


def dataset_label_for_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "synthetic" in name:
        return "synthetic"
    return DATASET_NAME


def load_customers(path: str | Path, *, require_target: bool = True) -> tuple[pd.DataFrame, list[str]]:
    raw = read_raw_csv(path)
    return clean_customers(raw, require_target=require_target, require_identifier=True)


def filter_trainable(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that have no usable label or no tenure/monthly charge. Outliers stay."""
    reasons: dict[str, int] = {}
    target_ok = frame[TARGET_COLUMN].isin(["Yes", "No"])
    reasons["invalid_or_missing_churn_label"] = int((~target_ok).sum())
    tenure_ok = frame["tenure"].notna()
    reasons["missing_tenure"] = int((~tenure_ok).sum())
    monthly_ok = frame["MonthlyCharges"].notna()
    reasons["missing_monthly_charges"] = int((~monthly_ok).sum())
    identifier_ok = frame[ID_COLUMN].notna()
    reasons["missing_customer_id"] = int((~identifier_ok).sum())
    keep = target_ok & tenure_ok & monthly_ok & identifier_ok
    trainable = frame.loc[keep].copy()
    trainable["tenure"] = trainable["tenure"].round().astype(int)
    reasons["rows_used_for_training"] = int(len(trainable))
    reasons["rows_excluded"] = int((~keep).sum())
    if trainable[TARGET_COLUMN].nunique() < 2:
        raise SchemaError("Training data must contain both churn and retained customers.")
    return trainable, reasons
