"""Descriptive churn analysis.

Rates use labeled customers in the filtered population as the denominator.
Counts and rates are returned separately. This module does not fit the classifier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.schema import ADDON_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

TENURE_BINS = [-0.01, 6, 12, 24, 48, 10_000]
TENURE_LABELS = ["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"]


def _labeled(frame: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame.loc[frame[TARGET_COLUMN].isin(["Yes", "No"])].copy()


def churn_by(frame: pd.DataFrame, column: str) -> list[dict]:
    labeled = _labeled(frame)
    rows = []
    if column not in labeled.columns or labeled.empty:
        return rows
    for name, group in labeled.groupby(column, dropna=False):
        customers = int(len(group))
        churned = int((group[TARGET_COLUMN] == "Yes").sum())
        label = "Missing" if pd.isna(name) else str(name)
        rows.append(
            {
                "name": label,
                "customers": customers,
                "churned": churned,
                "retained": customers - churned,
                "churn_rate": (churned / customers) if customers else None,
            }
        )
    rows.sort(key=lambda item: item["customers"], reverse=True)
    return rows


def _binned(frame: pd.DataFrame, column: str, bins, labels: list[str] | None = None) -> list[dict]:
    labeled = _labeled(frame)
    if column not in labeled.columns or labeled.empty:
        return []
    groups = pd.cut(labeled[column], bins=bins, labels=labels, include_lowest=True, duplicates="drop")
    working = labeled.copy()
    working["_bin"] = groups.astype(str)
    return churn_by(working, "_bin")


def quartile_edges(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return [0, 1]
    edges = [float(values.quantile(q)) for q in (0, 0.25, 0.5, 0.75, 1)]
    fixed = [edges[0]]
    for edge in edges[1:]:
        if edge <= fixed[-1]:
            fixed.append(fixed[-1] + 1e-6)
        else:
            fixed.append(edge)
    if fixed[-1] == fixed[0]:
        fixed[-1] = fixed[0] + 1
    return fixed


def addon_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    count = np.zeros(len(working), dtype=int)
    for column in ADDON_COLUMNS:
        if column in working.columns:
            count += (working[column] == "Yes").astype(int).to_numpy()
    labels = np.full(len(working), "0 services", dtype=object)
    labels[(count >= 1) & (count <= 2)] = "1-2 services"
    labels[(count >= 3) & (count <= 4)] = "3-4 services"
    labels[count >= 5] = "5-6 services"
    working["addon_group"] = labels
    return working


def numeric_by_churn(frame: pd.DataFrame) -> dict:
    labeled = _labeled(frame)
    summary: dict[str, dict] = {}
    for column in NUMERIC_FEATURES:
        if column not in labeled.columns:
            continue
        summary[column] = {}
        for label, group in labeled.groupby(TARGET_COLUMN):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            summary[column][str(label)] = {
                "customers": int(values.shape[0]),
                "mean": float(values.mean()) if len(values) else None,
                "median": float(values.median()) if len(values) else None,
                "p25": float(values.quantile(0.25)) if len(values) else None,
                "p75": float(values.quantile(0.75)) if len(values) else None,
            }
    return summary


def correlation_pairs(frame: pd.DataFrame) -> list[dict]:
    labeled = _labeled(frame)
    if labeled.empty:
        return []
    working = pd.DataFrame(
        {
            "tenure": pd.to_numeric(labeled.get("tenure"), errors="coerce"),
            "MonthlyCharges": pd.to_numeric(labeled.get("MonthlyCharges"), errors="coerce"),
            "TotalCharges": pd.to_numeric(labeled.get("TotalCharges"), errors="coerce"),
            "SeniorCitizen": pd.to_numeric(labeled["SeniorCitizen"], errors="coerce")
            if "SeniorCitizen" in labeled.columns
            else np.nan,
            "churn_flag": (labeled[TARGET_COLUMN] == "Yes").astype(float),
        }
    )
    corr = working.corr(numeric_only=True)
    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": None if pd.isna(value) else float(value),
                }
            )
    return pairs


def apply_population_filters(
    frame: pd.DataFrame,
    *,
    contract: str | None = None,
    internet_service: str | None = None,
    payment_method: str | None = None,
    paperless_billing: str | None = None,
    tech_support: str | None = None,
    tenure_min: float | None = None,
    tenure_max: float | None = None,
    senior_citizen: str | None = None,
) -> pd.DataFrame:
    filtered = frame
    mapping = {
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "TechSupport": tech_support,
    }
    for column, value in mapping.items():
        if value and column in filtered.columns:
            filtered = filtered.loc[filtered[column] == value]
    if tenure_min is not None and "tenure" in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered["tenure"], errors="coerce") >= tenure_min]
    if tenure_max is not None and "tenure" in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered["tenure"], errors="coerce") <= tenure_max]
    if senior_citizen in {"0", "1"} and "SeniorCitizen" in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered["SeniorCitizen"], errors="coerce") == int(senior_citizen)]
    return filtered


def build_eda(
    frame: pd.DataFrame,
    *,
    monthly_edges: list[float] | None = None,
    total_edges: list[float] | None = None,
) -> dict:
    labeled = _labeled(frame)
    customers = int(len(labeled))
    churned = int((labeled[TARGET_COLUMN] == "Yes").sum()) if customers else 0
    monthly_edges = monthly_edges or quartile_edges(labeled["MonthlyCharges"]) if customers else [0, 1]
    total_edges = total_edges or quartile_edges(labeled["TotalCharges"]) if customers else [0, 1]
    with_addons = addon_frame(labeled)
    return {
        "population": int(len(frame)),
        "labeled_customers": customers,
        "unlabeled_customers": int(len(frame) - customers),
        "churned": churned,
        "retained": customers - churned,
        "churn_rate": (churned / customers) if customers else None,
        "denominator_note": (
            "Churn rate = churned customers / customers with a Yes or No label in the current filter. "
            "Counts and rates are both shown so a large group is not confused with a high rate."
        ),
        "by_contract": churn_by(labeled, "Contract"),
        "by_tenure": _binned(labeled, "tenure", TENURE_BINS, TENURE_LABELS),
        "by_monthly_charges": _binned(labeled, "MonthlyCharges", monthly_edges),
        "by_total_charges": _binned(labeled, "TotalCharges", total_edges),
        "by_internet_service": churn_by(labeled, "InternetService"),
        "by_payment_method": churn_by(labeled, "PaymentMethod"),
        "by_paperless_billing": churn_by(labeled, "PaperlessBilling"),
        "by_senior_citizen": churn_by(labeled, "SeniorCitizen"),
        "by_tech_support": churn_by(labeled, "TechSupport"),
        "by_online_security": churn_by(labeled, "OnlineSecurity"),
        "by_gender": churn_by(labeled, "gender"),
        "by_partner": churn_by(labeled, "Partner"),
        "by_dependents": churn_by(labeled, "Dependents"),
        "by_addon_group": churn_by(with_addons, "addon_group"),
        "numeric_by_churn": numeric_by_churn(labeled),
        "correlation": correlation_pairs(labeled),
        "monthly_edges": monthly_edges,
        "total_edges": total_edges,
        "protected_attribute_note": (
            "Gender and senior-citizen charts are descriptive only. Those fields are not used "
            "to score customers or to recommend retention actions."
        ),
        "time_series_available": False,
        "time_series_note": "This table is a snapshot. It has no signup or churn date, so no monthly trend is calculated.",
    }


def render_eda_markdown(report: dict) -> str:
    lines = [
        "# Exploratory data report",
        "",
        "Descriptive statistics for the cleaned table supplied to this report. "
        "These figures are not held-out model metrics.",
        "",
        f"- Labeled customers: {report['labeled_customers']}",
        f"- Churned: {report['churned']}",
        f"- Retained: {report['retained']}",
        f"- Churn rate: {report['churn_rate']}",
        "",
        report["denominator_note"],
        "",
        "## Contract",
        "",
    ]
    for row in report["by_contract"]:
        rate = "n/a" if row["churn_rate"] is None else f"{row['churn_rate']:.1%}"
        lines.append(
            f"- {row['name']}: {row['churned']} churned / {row['customers']} customers ({rate})"
        )
    lines.extend(["", "## Notes", "", report["protected_attribute_note"], "", report["time_series_note"], ""])
    return "\n".join(lines)


def write_charts(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    """Save a few static charts. Failures are returned as warnings, not raised."""
    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled = _labeled(frame)
    if labeled.empty:
        return ["No labeled rows were available for charts."]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = labeled[TARGET_COLUMN].value_counts().reindex(["No", "Yes"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Retained", "Churned"], [counts.get("No", 0), counts.get("Yes", 0)], color=["#1f7a6b", "#a33b3b"])
        ax.set_ylabel("Customers")
        ax.set_title("Churn counts")
        fig.tight_layout()
        fig.savefig(output_dir / "churn_counts.png", dpi=120)
        plt.close(fig)

        contract = (
            labeled.groupby("Contract")[TARGET_COLUMN]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(contract.index.astype(str), contract.values, color="#c46b3a")
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, 1)
        ax.set_title("Churn rate by contract")
        fig.tight_layout()
        fig.savefig(output_dir / "churn_rate_by_contract.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - environment specific
        warnings.append(f"Matplotlib charts were not written: {exc}")

    try:
        import plotly.express as px

        summary = (
            labeled.groupby("InternetService")
            .agg(customers=(TARGET_COLUMN, "size"), churned=(TARGET_COLUMN, lambda s: int((s == "Yes").sum())))
            .reset_index()
        )
        summary["churn_rate"] = summary["churned"] / summary["customers"]
        fig = px.bar(
            summary,
            x="InternetService",
            y="churn_rate",
            hover_data=["customers", "churned"],
            title="Churn rate by internet service",
            labels={"churn_rate": "Churn rate", "InternetService": "Internet service"},
        )
        fig.write_html(output_dir / "churn_rate_by_internet.html", include_plotlyjs="cdn")
    except Exception as exc:  # pragma: no cover - environment specific
        warnings.append(f"Plotly chart was not written: {exc}")
    return warnings
