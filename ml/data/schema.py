"""Canonical Telco churn schema and column normalization."""

from __future__ import annotations

import re

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

# Kept for aggregate description. Excluded from scoring, explanations,
# clustering, and retention rules.
PROTECTED_COLUMNS = ("gender", "SeniorCitizen")
EXCLUSION_REASON = (
    "gender and SeniorCitizen remain available for aggregate descriptive analysis. "
    "They are excluded from the scoring model, explanations, cluster features, and "
    "retention recommendations so outreach is not driven by those characteristics."
)

NUMERIC_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")

CATEGORICAL_FEATURES = (
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)

MODEL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ADDON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

RAW_COLUMNS = (
    ID_COLUMN,
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
)

DATASET_NAME = "IBM Telco Customer Churn"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
DATASET_MIRROR_NOTE = (
    "The setup script downloads the public IBM sample CSV from the "
    "telco-customer-churn-on-icp4d project. That file matches the table published "
    "on Kaggle as Telco Customer Churn. Confirm the current Kaggle or IBM terms "
    "before redistributing the file."
)

DECISION_SUPPORT_WARNING = (
    "This application is a decision-support tool. It does not authorize denying service, "
    "automatically changing prices, or treating customers differently because of protected "
    "characteristics. A person should review every recommendation before anyone acts on it."
)

EXPLANATION_LIMITATION = (
    "Explanations describe how this model's score relates to the customer's recorded features. "
    "They are not evidence that a feature causes churn, and they are not a forecast of what will "
    "happen if the business changes that feature."
)

PRIORITY_FORMULA = (
    "Priority score = 100 × (0.60 × predicted churn probability + "
    "0.25 × monthly charges / population 95th percentile, capped at 1 + "
    "0.15 × (1 − tenure / 72), capped at 1). "
    "This is a transparent ranking heuristic, not an optimized profit model."
)


def column_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


CANONICAL_LOOKUP = {column_key(col): col for col in RAW_COLUMNS}


class SchemaError(ValueError):
    """Raised when a table cannot be mapped onto the expected churn schema."""
