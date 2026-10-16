"""Explainable customer segments.

Rule segments are the operational view. Optional k-means profiles are fit
without the churn label and without protected characteristics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ml.business.recommendations import addon_count
from ml.data.schema import ADDON_COLUMNS, TARGET_COLUMN

RULES: list[dict[str, str]] = [
    {
        "id": "high_risk_high_value",
        "name": "High-risk, high-value",
        "definition": "High risk and monthly charges at or above the population 75th percentile.",
        "suggested_action": "Prioritize a personal retention review before a generic campaign.",
    },
    {
        "id": "early_tenure_elevated_risk",
        "name": "New customers with early churn signals",
        "definition": "Tenure of 0-6 months and Medium or High risk. The tenure cut matches the first analytics bin.",
        "suggested_action": "Prioritize onboarding and an early check-in.",
    },
    {
        "id": "month_to_month_elevated_risk",
        "name": "Month-to-month customers with elevated risk",
        "definition": "Month-to-month contract and Medium or High risk.",
        "suggested_action": "Prioritize a term-contract conversation.",
    },
    {
        "id": "high_monthly_charge",
        "name": "High monthly-charge customers",
        "definition": "Monthly charges at or above the population 75th percentile, regardless of risk.",
        "suggested_action": "Review plan fit. Treat this as a pricing conversation only when risk is also elevated.",
    },
    {
        "id": "long_tenure_loyal",
        "name": "Long-tenure loyal customers",
        "definition": "Tenure of at least 48 months and Low risk.",
        "suggested_action": "Keep the regular service cycle. Avoid unnecessary retention discounts.",
    },
    {
        "id": "low_risk_expansion",
        "name": "Low-risk customers with expansion room",
        "definition": "Low risk, tenure of at least 12 months, internet service, and two or fewer add-on services.",
        "suggested_action": "Consider a relevant add-on conversation. This is not a churn intervention.",
    },
    {
        "id": "other",
        "name": "Other customers",
        "definition": "Customers who do not match an earlier primary rule.",
        "suggested_action": "Use the individual recommendation rather than a segment campaign.",
    },
]

PRIMARY_PRIORITY = [rule["id"] for rule in RULES]
RULE_IDS = {rule["id"] for rule in RULES}


class UnknownSegmentError(KeyError):
    """Raised when a caller asks for a segment id that is not a documented rule."""


def matching_rules(features: dict, risk: str, high_charge_cutoff: float) -> list[str]:
    matched: list[str] = []
    monthly = float(features.get("MonthlyCharges") or 0)
    tenure = float(features.get("tenure") or 0)
    if risk == "High" and monthly >= high_charge_cutoff:
        matched.append("high_risk_high_value")
    if tenure <= 6 and risk in {"High", "Medium"}:
        matched.append("early_tenure_elevated_risk")
    if features.get("Contract") == "Month-to-month" and risk in {"High", "Medium"}:
        matched.append("month_to_month_elevated_risk")
    if monthly >= high_charge_cutoff:
        matched.append("high_monthly_charge")
    if tenure >= 48 and risk == "Low":
        matched.append("long_tenure_loyal")
    internet = features.get("InternetService")
    if risk == "Low" and tenure >= 12 and internet not in {None, "No"} and addon_count(features) <= 2:
        matched.append("low_risk_expansion")
    return matched


def primary_segment(matched: list[str]) -> str:
    for rule_id in PRIMARY_PRIORITY:
        if rule_id in matched:
            return rule_id
    return "other"


def _classify(frame: pd.DataFrame, high_charge_cutoff: float) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Return overlapping rule indexes and exclusive primary indexes for the same rules."""
    records = frame.to_dict(orient="records")
    membership = {rule["id"]: [] for rule in RULES if rule["id"] != "other"}
    primary_counts: dict[str, list[int]] = {rule["id"]: [] for rule in RULES}
    for index, record in enumerate(records):
        risk = record.get("risk_category")
        if risk in {"High", "Medium", "Low"}:
            matched = matching_rules(record, str(risk), high_charge_cutoff)
        else:
            monthly = float(record.get("MonthlyCharges") or 0)
            matched = ["high_monthly_charge"] if monthly >= high_charge_cutoff else []
        primary = primary_segment(matched)
        primary_counts[primary].append(index)
        for rule_id in matched:
            membership[rule_id].append(index)
    return membership, primary_counts


def customer_ids_for_rule(frame: pd.DataFrame, high_charge_cutoff: float, rule_id: str) -> list[str]:
    """Customer ids that satisfy every condition of one rule.

    Overlapping rules use the same match as the segment counts. The residual
    rule ``other`` is the primary assignment for customers who matched no earlier rule.
    """
    if rule_id not in RULE_IDS:
        raise UnknownSegmentError(rule_id)
    if frame.empty or "customerID" not in frame.columns:
        return []
    membership, primary_counts = _classify(frame, high_charge_cutoff)
    indexes = primary_counts["other"] if rule_id == "other" else membership[rule_id]
    return [str(frame.iloc[index]["customerID"]) for index in indexes]


def summarize_rules(frame: pd.DataFrame, high_charge_cutoff: float) -> list[dict[str, Any]]:
    """Count customers against every rule. A customer may appear in more than one rule."""
    membership, primary_counts = _classify(frame, high_charge_cutoff)

    summaries = []
    for rule in RULES:
        if rule["id"] == "other":
            indexes = primary_counts["other"]
            population = frame.iloc[indexes] if indexes else frame.iloc[0:0]
        else:
            indexes = membership[rule["id"]]
            population = frame.iloc[indexes] if indexes else frame.iloc[0:0]
        labeled = population[population[TARGET_COLUMN].isin(["Yes", "No"])] if TARGET_COLUMN in population else population
        churned = int((labeled[TARGET_COLUMN] == "Yes").sum()) if len(labeled) and TARGET_COLUMN in labeled else 0
        summaries.append(
            {
                **rule,
                "customers": int(len(population)),
                "share_of_population": float(len(population) / len(frame)) if len(frame) else 0.0,
                "observed_churned": churned,
                "observed_churn_rate": float(churned / len(labeled)) if len(labeled) else None,
                "average_predicted_probability": (
                    float(population["probability"].mean()) if len(population) and "probability" in population else None
                ),
                "average_monthly_charges": (
                    float(population["MonthlyCharges"].mean()) if len(population) else None
                ),
                "average_tenure": float(population["tenure"].mean()) if len(population) else None,
                "primary_customers": int(len(primary_counts[rule["id"]])),
            }
        )
    return summaries


def _cluster_matrix(frame: pd.DataFrame, total_median: float) -> pd.DataFrame:
    total = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(total_median)
    addon = np.zeros(len(frame))
    for column in ADDON_COLUMNS:
        addon += (frame[column] == "Yes").astype(int).to_numpy()
    return pd.DataFrame(
        {
            "tenure": pd.to_numeric(frame["tenure"], errors="coerce").to_numpy(),
            "MonthlyCharges": pd.to_numeric(frame["MonthlyCharges"], errors="coerce").to_numpy(),
            "TotalCharges": total.to_numpy(),
            "addon_count": addon,
            "month_to_month": (frame["Contract"] == "Month-to-month").astype(float).to_numpy(),
            "fiber": (frame["InternetService"] == "Fiber optic").astype(float).to_numpy(),
            "electronic_check": (frame["PaymentMethod"] == "Electronic check").astype(float).to_numpy(),
        }
    )


def _describe(profile: dict, overall: dict) -> str:
    parts: list[str] = []
    if profile["tenure_mean"] >= overall["tenure_mean"] + 12:
        parts.append("longer tenure")
    elif profile["tenure_mean"] <= max(6, overall["tenure_mean"] - 12):
        parts.append("shorter tenure")
    if profile["monthly_mean"] >= overall["monthly_mean"] * 1.15:
        parts.append("higher monthly charges")
    elif profile["monthly_mean"] <= overall["monthly_mean"] * 0.85:
        parts.append("lower monthly charges")
    if profile["month_to_month_rate"] >= 0.60:
        parts.append("mostly month-to-month contracts")
    elif profile["month_to_month_rate"] <= 0.25:
        parts.append("mostly fixed-term contracts")
    if profile["fiber_rate"] >= 0.60:
        parts.append("mostly fiber internet")
    if not parts:
        parts.append("a profile close to the training average")
    return "Customers with " + ", ".join(parts)


def fit_segments(train_frame: pd.DataFrame, seed: int = 42, fast: bool = False) -> dict[str, Any]:
    """Fit k-means on training features only. The churn label is not a feature."""
    total_median = float(pd.to_numeric(train_frame["TotalCharges"], errors="coerce").median())
    matrix = _cluster_matrix(train_frame, total_median)
    if "Churn" in matrix.columns or TARGET_COLUMN in matrix.columns:
        raise RuntimeError("Cluster features must not include the churn label.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    max_k = 2 if fast or len(matrix) < 80 else 6
    scores: dict[str, float] = {}
    best_k = 2
    best_score = -1.0
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(scaled)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(scaled, labels))
        scores[str(k)] = score
        if score > best_score:
            best_score = score
            best_k = k
    model = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = model.fit_predict(scaled)
    overall = {
        "tenure_mean": float(matrix["tenure"].mean()),
        "monthly_mean": float(matrix["MonthlyCharges"].mean()),
    }
    profiles = []
    names: dict[int, str] = {}
    for cluster_id in range(best_k):
        positions = np.flatnonzero(labels == cluster_id)
        group = train_frame.iloc[positions]
        subset = matrix.iloc[positions]
        profile = {
            "cluster": int(cluster_id),
            "customers": int(len(positions)),
            "tenure_mean": float(subset["tenure"].mean()),
            "monthly_mean": float(subset["MonthlyCharges"].mean()),
            "total_mean": float(subset["TotalCharges"].mean()),
            "addon_mean": float(subset["addon_count"].mean()),
            "month_to_month_rate": float(subset["month_to_month"].mean()),
            "fiber_rate": float(subset["fiber"].mean()),
            "electronic_check_rate": float(subset["electronic_check"].mean()),
            "observed_churn_rate": float((group[TARGET_COLUMN] == "Yes").mean()) if len(group) else None,
        }
        description = _describe(profile, overall)
        profile["description"] = description
        names[cluster_id] = description
        profiles.append(profile)
    # Disambiguate identical descriptions.
    seen: dict[str, int] = {}
    for profile in profiles:
        count = seen.get(profile["description"], 0) + 1
        seen[profile["description"]] = count
        if list(names.values()).count(profile["description"]) > 1:
            profile["description"] = f"{profile['description']} (cluster {profile['cluster'] + 1})"
            names[profile["cluster"]] = profile["description"]
    return {
        "scaler": scaler,
        "model": model,
        "total_median": total_median,
        "feature_names": list(matrix.columns),
        "k": best_k,
        "silhouette_by_k": scores,
        "selected_silhouette": best_score,
        "profiles": profiles,
        "names": names,
        "excluded": ["Churn", "gender", "SeniorCitizen"],
        "fit_rows": int(len(train_frame)),
        "note": (
            "Clusters were fit on the training split only. Churn, gender, and senior-citizen "
            "status were not used as features. Names describe the fitted centers; they are not causal segments."
        ),
    }


def assign_clusters(frame: pd.DataFrame, artifact: dict) -> list[str]:
    matrix = _cluster_matrix(frame, float(artifact["total_median"]))
    scaled = artifact["scaler"].transform(matrix)
    labels = artifact["model"].predict(scaled)
    names = artifact["names"]
    return [names[int(label)] for label in labels]
