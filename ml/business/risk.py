"""Risk bands and the ranking heuristic."""

from __future__ import annotations


def risk_category(probability: float, medium_threshold: float, high_threshold: float) -> str:
    if high_threshold <= medium_threshold:
        raise ValueError("The high-risk threshold must be greater than the medium-risk threshold.")
    if probability >= high_threshold:
        return "High"
    if probability >= medium_threshold:
        return "Medium"
    return "Low"


def predicted_label(probability: float, decision_threshold: float) -> str:
    """Positive class uses the cost-tuned decision threshold, which is also the high-risk cutoff."""
    return "Yes" if probability >= decision_threshold else "No"


def priority_score(
    probability: float,
    monthly_charges: float,
    tenure: float,
    monthly_p95: float,
    max_tenure: float = 72.0,
) -> float:
    value_ratio = 0.0 if monthly_p95 <= 0 else min(float(monthly_charges) / float(monthly_p95), 1.0)
    newness = 1.0 - min(float(tenure) / float(max_tenure), 1.0)
    score = 100.0 * (0.60 * float(probability) + 0.25 * value_ratio + 0.15 * newness)
    return round(score, 1)


def reliability(features: dict, probability: float, feature_spec: dict) -> dict:
    outside: list[str] = []
    for column, bounds in feature_spec.get("numeric", {}).items():
        value = features.get(column)
        if value is None:
            continue
        if float(value) < float(bounds["min"]) or float(value) > float(bounds["max"]):
            outside.append(column)
    for column, levels in feature_spec.get("categorical", {}).items():
        value = features.get(column)
        if value is not None and value not in levels:
            outside.append(column)
    if outside:
        return {
            "level": "limited",
            "outside_features": outside,
            "message": (
                "Some values sit outside the ranges or categories used to fit the deployed model "
                f"({', '.join(outside)}). The number shown is still the model's predicted probability, "
                "not a certainty score, and it is less reliable for this profile."
            ),
        }
    if abs(float(probability) - 0.5) <= 0.05:
        return {
            "level": "borderline",
            "outside_features": [],
            "message": (
                "The predicted probability is near 0.50, so a small change in inputs can move the risk band. "
                "This number is an estimated chance of churn, not a measure of how certain the model is."
            ),
        }
    return {
        "level": "typical",
        "outside_features": [],
        "message": (
            "Inputs fall within the values used to fit the deployed model. "
            "The predicted probability is an estimated chance of churn, not a certainty score or a guarantee."
        ),
    }
