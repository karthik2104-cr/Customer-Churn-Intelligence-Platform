"""Data-quality checks. Outliers are reported and retained."""

from __future__ import annotations

import pandas as pd

from ml.data.loader import _blank_mask, normalize_columns
from ml.data.schema import ID_COLUMN, NUMERIC_FEATURES, RAW_COLUMNS, TARGET_COLUMN


def _iqr_report(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"flagged": 0, "note": "No numeric values to inspect."}
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flagged = int(((values < lower) | (values > upper)).sum())
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "flagged": flagged,
        "note": (
            "Flagged for inspection only. These rows are retained because extreme "
            "bills or tenure can be real customers."
        ),
    }


def build_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame, notes: list[str]) -> dict:
    renamed, ignored = normalize_columns(raw)
    present = [column for column in RAW_COLUMNS if column in renamed.columns]
    missing_columns = [column for column in RAW_COLUMNS if column not in renamed.columns]
    missing_counts = {column: int(_blank_mask(renamed[column]).sum()) for column in present}

    duplicate_count = 0
    duplicate_ids: list[str] = []
    if ID_COLUMN in renamed.columns:
        identifiers = renamed[ID_COLUMN].fillna("").astype(str).str.strip()
        duplicated = identifiers.duplicated(keep=False) & identifiers.ne("")
        duplicate_count = int(identifiers.duplicated(keep="first").sum())
        duplicate_ids = identifiers.loc[duplicated].value_counts().head(20).index.astype(str).tolist()

    class_counts: dict[str, int] = {}
    churn_rate = None
    if TARGET_COLUMN in cleaned.columns:
        observed = cleaned[TARGET_COLUMN].fillna("Missing").astype(str)
        class_counts = {str(key): int(value) for key, value in observed.value_counts(dropna=False).items()}
        labeled = cleaned[TARGET_COLUMN].isin(["Yes", "No"])
        if int(labeled.sum()):
            churn_rate = float((cleaned.loc[labeled, TARGET_COLUMN] == "Yes").mean())

    outliers = {
        column: _iqr_report(cleaned[column]) for column in NUMERIC_FEATURES if column in cleaned.columns
    }

    invalid_numbers = {}
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        if column not in renamed.columns:
            continue
        source = renamed[column]
        blank = _blank_mask(source)
        numeric = pd.to_numeric(source.fillna("").astype(str).str.strip().mask(blank), errors="coerce")
        invalid_numbers[column] = int((numeric.isna() & ~blank).sum())

    return {
        "rows_raw": int(len(raw)),
        "rows_cleaned": int(len(cleaned)),
        "columns_present": present,
        "missing_columns": missing_columns,
        "ignored_columns": ignored,
        "missing_values": missing_counts,
        "duplicate_customer_ids": duplicate_count,
        "duplicate_id_examples": duplicate_ids,
        "invalid_numeric_values": invalid_numbers,
        "class_counts": class_counts,
        "observed_churn_rate": churn_rate,
        "outliers": outliers,
        "cleaning_notes": notes,
        "outlier_policy": "Outliers are inspected and retained. They are not deleted automatically.",
    }
