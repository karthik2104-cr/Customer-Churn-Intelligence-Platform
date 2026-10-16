"""Local explanations for the deployed pipeline.

SHAP is used when it can be computed for the fitted estimator and the values
are consistent with the predicted probability. Otherwise the service uses
baseline occlusion on the same pipeline that produced the probability.
Neither method is a causal effect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.data.schema import EXPLANATION_LIMITATION, MODEL_FEATURES


def _python_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value) if not isinstance(value, str) else value


def _factor(feature: str, actual: Any, baseline: Any, contribution: float, method: str) -> dict:
    if contribution > 1e-8:
        direction = "higher_predicted_churn"
    elif contribution < -1e-8:
        direction = "lower_predicted_churn"
    else:
        direction = "no_change"
    detail = (
        "Difference in predicted churn probability when this feature is replaced with its training baseline."
        if method == "baseline_occlusion"
        else "SHAP value aggregated back to the original feature. Positive values push the explained output toward churn."
    )
    return {
        "feature": feature,
        "actual": _python_value(actual),
        "baseline": _python_value(baseline),
        "contribution": float(contribution),
        "direction": direction,
        "detail": detail,
    }


def occlusion_explain(
    pipeline,
    frame: pd.DataFrame,
    baselines: dict,
    features: list[str] | None = None,
    top_n: int = 6,
) -> tuple[np.ndarray, list[list[dict]], str]:
    """Change one original feature at a time to its baseline and record the probability delta."""
    columns = [feature for feature in (features or list(MODEL_FEATURES)) if feature in frame.columns]
    base = np.asarray(pipeline.predict_proba(frame)[:, 1], dtype=float)
    deltas: dict[str, np.ndarray] = {}
    for feature in columns:
        perturbed = frame.copy()
        perturbed[feature] = baselines[feature]
        probs = np.asarray(pipeline.predict_proba(perturbed)[:, 1], dtype=float)
        deltas[feature] = base - probs
    explanations: list[list[dict]] = []
    for index in range(len(frame)):
        factors = [
            _factor(
                feature,
                frame.iloc[index][feature],
                baselines.get(feature),
                float(deltas[feature][index]),
                "baseline_occlusion",
            )
            for feature in columns
        ]
        factors.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        explanations.append(factors[:top_n])
    return base, explanations, "baseline_occlusion"


def _original_feature(transformed_name: str) -> str:
    if transformed_name.startswith("num__"):
        return transformed_name.split("__", 1)[1]
    if transformed_name.startswith("cat__"):
        rest = transformed_name.split("__", 1)[1]
        for feature in sorted(MODEL_FEATURES, key=len, reverse=True):
            if rest == feature or rest.startswith(feature + "_"):
                return feature
    return transformed_name


def _positive_class_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            array = array[:, :, 1]
        elif array.shape[1] == 2:
            array = array[:, 1, :]
    if array.ndim == 2:
        array = array[0] if array.shape[0] == 1 else array.ravel()
    return np.asarray(array, dtype=float).ravel()


def _estimator_supported(estimator) -> bool:
    name = type(estimator).__name__
    return name in {
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "HistGradientBoostingClassifier",
        "XGBClassifier",
        "GradientBoostingClassifier",
    }


def shap_explain_row(pipeline, row: pd.DataFrame, background: pd.DataFrame | None, probability: float) -> dict | None:
    """Return a SHAP explanation or None when SHAP is unavailable or inconsistent."""
    try:
        import shap
    except Exception:
        return None
    if not hasattr(pipeline, "named_steps"):
        return None
    estimator = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocess")
    if estimator is None or preprocessor is None or not _estimator_supported(estimator):
        return {
            "available": False,
            "reason": (
                "SHAP is not reported for this deployed estimator. "
                "Baseline occlusion is used so the explanation matches the probability shown to the user."
            ),
        }
    try:
        transformed = preprocessor.transform(row)
        names = list(preprocessor.get_feature_names_out())
        if type(estimator).__name__ == "LogisticRegression":
            if background is None or background.empty:
                return None
            background_transformed = preprocessor.transform(background)
            explainer = shap.LinearExplainer(estimator, background_transformed)
            values = explainer.shap_values(transformed)
            base_value = explainer.expected_value
        else:
            explainer = shap.TreeExplainer(estimator, model_output="probability")
            values = explainer.shap_values(transformed)
            base_value = explainer.expected_value
        contribution = _positive_class_values(values)
        if isinstance(base_value, (list, np.ndarray)):
            base_array = np.asarray(base_value, dtype=float).ravel()
            base_number = float(base_array[-1] if len(base_array) > 1 else base_array[0])
        else:
            base_number = float(base_value)
        if len(contribution) != len(names):
            return None
        if abs((base_number + float(contribution.sum())) - float(probability)) > 0.08:
            return {
                "available": False,
                "reason": (
                    "SHAP ran but its values did not reconcile with the deployed probability, "
                    "so they were discarded. Baseline occlusion is shown instead."
                ),
            }
        grouped: dict[str, float] = {}
        for name, value in zip(names, contribution):
            feature = _original_feature(str(name))
            grouped[feature] = grouped.get(feature, 0.0) + float(value)
        factors = [
            _factor(feature, row.iloc[0][feature], None, value, "shap")
            for feature, value in grouped.items()
            if feature in row.columns
        ]
        factors.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {
            "available": True,
            "method": "shap",
            "base_value": base_number,
            "factors": factors[:8],
            "limitation": EXPLANATION_LIMITATION,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"SHAP could not explain this model ({type(exc).__name__}). Baseline occlusion is used instead.",
        }


def explain_frame(
    pipeline,
    frame: pd.DataFrame,
    baselines: dict,
    feature_spec: dict | None = None,
    background: pd.DataFrame | None = None,
    top_features: list[str] | None = None,
    prefer_shap: bool = False,
    top_n: int = 6,
) -> list[dict]:
    """Explain each row. SHAP is attempted only for a single row when requested."""
    del feature_spec
    if prefer_shap and len(frame) == 1:
        probability = float(pipeline.predict_proba(frame)[0, 1])
        shap_result = shap_explain_row(pipeline, frame, background, probability)
        if shap_result and shap_result.get("available") and shap_result.get("factors"):
            return [
                {
                    "method": "shap",
                    "factors": shap_result["factors"][:top_n],
                    "limitation": EXPLANATION_LIMITATION,
                    "shap_note": "SHAP values are aggregated from encoded features back to the original columns.",
                }
            ]
        fallback_reason = None if shap_result is None else shap_result.get("reason")
    else:
        fallback_reason = None
        if prefer_shap and len(frame) != 1:
            fallback_reason = "Batch explanations use baseline occlusion so every row is scored with the deployed pipeline."

    _base, explanations, method = occlusion_explain(
        pipeline,
        frame,
        baselines,
        features=top_features or list(MODEL_FEATURES),
        top_n=top_n,
    )
    results = []
    for factors in explanations:
        results.append(
            {
                "method": method,
                "factors": factors,
                "limitation": EXPLANATION_LIMITATION,
                "fallback_reason": fallback_reason,
            }
        )
    return results
