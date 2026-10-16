"""Training-split associations used to annotate recommendations."""

from __future__ import annotations

import pandas as pd

from ml.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def training_associations(frame: pd.DataFrame) -> dict:
    labeled = frame.loc[frame[TARGET_COLUMN].isin(["Yes", "No"])].copy()
    churn = (labeled[TARGET_COLUMN] == "Yes").astype(float)
    overall = float(churn.mean()) if len(labeled) else None
    numeric = []
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(labeled[column], errors="coerce")
        correlation = None if values.notna().sum() < 3 else float(values.corr(churn))
        numeric.append(
            {
                "feature": column,
                "correlation_with_churn": correlation,
                "note": "Pearson correlation on the training split. Association only.",
            }
        )
    categorical = []
    for column in CATEGORICAL_FEATURES:
        levels = []
        for level, group in labeled.groupby(column):
            rate = float((group[TARGET_COLUMN] == "Yes").mean())
            levels.append(
                {
                    "level": str(level),
                    "customers": int(len(group)),
                    "churn_rate": rate,
                    "difference_from_training_rate": None if overall is None else rate - overall,
                }
            )
        levels.sort(key=lambda item: item["churn_rate"], reverse=True)
        categorical.append({"feature": column, "levels": levels})
    return {
        "split": "training",
        "training_churn_rate": overall,
        "numeric": numeric,
        "categorical": categorical,
        "caveat": "Differences are observed associations in the training split, not causal effects.",
    }
