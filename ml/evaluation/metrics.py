"""Threshold-aware classification metrics.

Cross-validated accuracy, precision, recall, and F1 from scikit-learn search
objects use the library default cutoff of 0.5. Held-out metrics in this module
use the threshold chosen on the internal validation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, probabilities, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any((probabilities < -1e-6) | (probabilities > 1 + 1e-6)):
        raise ValueError("Probabilities must lie between 0 and 1.")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    report = classification_report(
        y_true,
        predicted,
        labels=[0, 1],
        target_names=["retained", "churn"],
        output_dict=True,
        zero_division=0,
    )
    # Ranking metrics need both classes. Callers guarantee that on real splits.
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {
            "labels": ["retained", "churn"],
            "matrix": matrix.tolist(),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "classification_report": report,
        "support": {"retained": int((y_true == 0).sum()), "churn": int((y_true == 1).sum())},
    }


def tune_cost_threshold(
    y_true,
    probabilities,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> tuple[dict, list[dict]]:
    """Pick a decision threshold by expected misclassification cost on one split."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    best: dict | None = None
    grid: list[dict] = []
    for threshold in np.round(np.arange(0.15, 0.81, 0.01), 2):
        predicted = probabilities >= threshold
        fp = int(np.sum((y_true == 0) & predicted))
        fn = int(np.sum((y_true == 1) & ~predicted))
        tp = int(np.sum((y_true == 1) & predicted))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        cost = float(fn_cost * fn + fp_cost * fp)
        row = {
            "threshold": float(threshold),
            "cost": cost,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        grid.append(row)
        if best is None or cost < best["cost"] - 1e-9 or (
            abs(cost - best["cost"]) <= 1e-9 and recall > best["recall"]
        ):
            best = row
    if best is None:
        raise ValueError("Threshold search produced no candidates.")
    return best, grid


def choose_medium_threshold(y_true, probabilities, high_threshold: float, base_rate: float) -> float:
    """Lowest cutoff below the decision threshold that still beats the base rate on precision.

    Customers between the medium and high cutoffs are elevated risk, not automatic
    positive classifications. The high cutoff remains the decision threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates: list[float] = []
    upper = round(float(high_threshold) - 0.01, 2)
    for threshold in np.round(np.arange(0.10, max(0.10, upper) + 0.001, 0.01), 2):
        if threshold >= high_threshold:
            continue
        predicted = probabilities >= threshold
        tp = float(np.sum((y_true == 1) & predicted))
        fp = float(np.sum((y_true == 0) & predicted))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        if precision + 1e-9 >= base_rate:
            candidates.append(float(threshold))
    if candidates:
        medium = min(candidates)
    else:
        medium = round(max(0.10, float(high_threshold) * 0.6), 2)
    if medium >= high_threshold:
        medium = round(max(0.05, float(high_threshold) - 0.05), 2)
    return float(medium)
