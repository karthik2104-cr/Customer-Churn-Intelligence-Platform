"""Grouped permutation importance on original columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def grouped_permutation_importance(model, frame: pd.DataFrame, y_true, n_repeats: int = 4, seed: int = 42) -> dict:
    y_true = np.asarray(y_true).astype(int)
    baseline = float(average_precision_score(y_true, model.predict_proba(frame)[:, 1]))
    rng = np.random.default_rng(seed)
    rows = []
    for column in frame.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = float(average_precision_score(y_true, model.predict_proba(shuffled)[:, 1]))
            drops.append(baseline - score)
        rows.append(
            {
                "feature": column,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
            }
        )
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "method": "grouped_permutation",
        "scoring": "average_precision",
        "split": "internal_validation",
        "used_for_model_selection": False,
        "baseline_average_precision": baseline,
        "n_repeats": n_repeats,
        "features": rows,
        "note": (
            "Each original column is shuffled on the internal validation split. "
            "The drop in average precision is an association between that column and the model's ranking, "
            "not a causal effect. The test set was not used."
        ),
    }
