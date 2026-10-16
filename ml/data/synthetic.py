"""Synthetic rows for automated tests.

These records are generated. They are not the IBM Telco Customer Churn dataset
and must not be presented as that dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.schema import ADDON_COLUMNS, RAW_COLUMNS


def make_synthetic_frame(n: int = 160, seed: int = 42) -> pd.DataFrame:
    """Build a labeled synthetic table with the telco columns and both classes."""
    rng = np.random.default_rng(seed)
    contracts = np.array(["Month-to-month", "One year", "Two year"])
    contract = rng.choice(contracts, size=n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(np.array(["DSL", "Fiber optic", "No"]), size=n, p=[0.35, 0.45, 0.20])
    phone = rng.choice(np.array(["Yes", "No"]), size=n, p=[0.9, 0.1])
    tenure = rng.integers(0, 73, size=n)
    tenure = np.where(contract == "Two year", np.maximum(tenure, 12), tenure)
    monthly = rng.normal(70, 25, size=n)
    monthly = np.where(internet == "Fiber optic", monthly + 20, monthly)
    monthly = np.where(internet == "No", rng.uniform(18, 30, size=n), monthly)
    monthly = np.clip(monthly, 18, 120)
    total = monthly * tenure
    total = np.where(tenure == 0, np.nan, total)

    def yes_no(probability: float) -> np.ndarray:
        return np.where(rng.random(n) < probability, "Yes", "No")

    partner = yes_no(0.48)
    dependents = yes_no(0.30)
    paperless = yes_no(0.6)
    senior = rng.choice(np.array(["0", "1"]), size=n, p=[0.84, 0.16])
    gender = rng.choice(np.array(["Female", "Male"]), size=n)
    payment = rng.choice(
        np.array(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
        size=n,
    )
    multiple = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], size=n))
    addon = {}
    for column in ADDON_COLUMNS:
        values = rng.choice(["Yes", "No"], size=n)
        values = np.where(internet == "No", "No internet service", values)
        addon[column] = values

    month_to_month = (contract == "Month-to-month").astype(float)
    fiber = (internet == "Fiber optic").astype(float)
    support = (addon["TechSupport"] == "No").astype(float)
    logit = (
        -1.4
        + 1.3 * month_to_month
        + 0.8 * fiber
        + 0.4 * support
        - 0.03 * tenure
        + 0.015 * (monthly - 65)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "Yes", "No")
    if len(set(churn.tolist())) < 2:
        churn[0] = "Yes"
        churn[1] = "No"

    frame = pd.DataFrame(
        {
            "customerID": [f"SYN-{index:04d}" for index in range(1, n + 1)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure.astype(int).astype(str),
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": np.round(monthly, 2).astype(str),
            "TotalCharges": ["" if np.isnan(value) else f"{value:.2f}" for value in total],
            "Churn": churn,
            **addon,
        }
    )
    return frame.loc[:, list(RAW_COLUMNS)]


def write_synthetic_csv(path: str | Path, n: int = 160, seed: int = 42) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_frame(n=n, seed=seed).to_csv(destination, index=False)
    return destination
