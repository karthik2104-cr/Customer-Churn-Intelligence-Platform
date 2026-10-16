"""Conditional retention recommendations.

Rules use service, contract, payment, and tenure fields. They do not read
gender or senior-citizen status. Observed rates are included only when the
caller supplies associations measured on the training split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from ml.data.schema import ADDON_COLUMNS


@dataclass(frozen=True)
class Recommendation:
    action: str
    reason: str
    effort: str
    category: str

    def as_dict(self) -> dict:
        return asdict(self)


def addon_count(features: Mapping) -> int:
    return sum(1 for column in ADDON_COLUMNS if features.get(column) == "Yes")


def _rate_sentence(associations: dict | None, feature: str, level: str) -> str:
    if not associations:
        return ""
    overall = associations.get("training_churn_rate")
    for block in associations.get("categorical", []):
        if block.get("feature") != feature:
            continue
        for item in block.get("levels", []):
            if item.get("level") == level and overall is not None:
                return (
                    f" On the training split, the observed churn rate for {feature} = {level} was "
                    f"{item['churn_rate']:.1%} among {item['customers']} customers, compared with "
                    f"{overall:.1%} overall. That is an association, not a cause."
                )
    return ""


def recommend(
    features: Mapping,
    probability: float,
    medium_threshold: float,
    high_threshold: float,
    high_charge_cutoff: float | None = None,
    associations: dict | None = None,
) -> list[Recommendation]:
    elevated = probability >= medium_threshold
    high = probability >= high_threshold
    probability_text = f"{probability:.1%}"
    medium_text = f"{medium_threshold:.1%}"
    recommendations: list[Recommendation] = []

    if elevated and features.get("Contract") == "Month-to-month":
        recommendations.append(
            Recommendation(
                action="Prioritize a term-contract conversation",
                reason=(
                    f"The contract is month-to-month and the predicted churn probability is {probability_text}, "
                    f"at or above the elevated-risk threshold of {medium_text}."
                    + _rate_sentence(associations, "Contract", "Month-to-month")
                    + " A contract offer is a retention conversation, not an automatic price or eligibility change."
                ),
                effort="medium",
                category="contract",
            )
        )

    monthly = features.get("MonthlyCharges")
    if (
        elevated
        and high_charge_cutoff is not None
        and monthly is not None
        and float(monthly) >= float(high_charge_cutoff)
    ):
        recommendations.append(
            Recommendation(
                action="Review plan fit and monthly charges",
                reason=(
                    f"Monthly charges are {float(monthly):.2f}, at or above the current population 75th percentile "
                    f"of {float(high_charge_cutoff):.2f}, and the predicted probability is {probability_text}. "
                    "Review whether the current plan matches the services in use. "
                    "This is not an instruction to change the price automatically."
                ),
                effort="medium",
                category="pricing",
            )
        )

    if elevated and features.get("TechSupport") == "No":
        recommendations.append(
            Recommendation(
                action="Offer a technical-support check-in",
                reason=(
                    "Technical support is not on the account and the predicted "
                    f"probability is {probability_text}."
                    + _rate_sentence(associations, "TechSupport", "No")
                ),
                effort="low",
                category="support",
            )
        )

    if elevated and features.get("OnlineSecurity") == "No":
        recommendations.append(
            Recommendation(
                action="Mention online security during the service review",
                reason=(
                    "Online security is not on the account and the predicted probability is "
                    f"{probability_text}."
                    + _rate_sentence(associations, "OnlineSecurity", "No")
                ),
                effort="low",
                category="support",
            )
        )

    tenure = features.get("tenure")
    if elevated and tenure is not None and float(tenure) <= 6:
        recommendations.append(
            Recommendation(
                action="Prioritize onboarding and an early check-in",
                reason=(
                    f"Tenure is {int(float(tenure))} {'month' if int(float(tenure)) == 1 else 'months'} and the predicted probability is {probability_text}. "
                    "The early-tenure rule uses the same 0-6 month bin as the analytics view."
                ),
                effort="medium",
                category="onboarding",
            )
        )

    if elevated and features.get("PaymentMethod") == "Electronic check":
        recommendations.append(
            Recommendation(
                action="Offer help with the payment method",
                reason=(
                    "The payment method is electronic check and the predicted probability is "
                    f"{probability_text}."
                    + _rate_sentence(associations, "PaymentMethod", "Electronic check")
                ),
                effort="low",
                category="payment",
            )
        )

    if elevated and features.get("InternetService") == "Fiber optic":
        recommendations.append(
            Recommendation(
                action="Review fiber service experience and plan fit",
                reason=(
                    "Internet service is fiber optic and the predicted probability is "
                    f"{probability_text}."
                    + _rate_sentence(associations, "InternetService", "Fiber optic")
                    + " Do not treat fiber service itself as a proven cause of churn."
                ),
                effort="medium",
                category="service_review",
            )
        )

    if (
        not elevated
        and tenure is not None
        and float(tenure) >= 12
        and features.get("InternetService") not in {None, "No"}
        and addon_count(features) <= 2
    ):
        recommendations.append(
            Recommendation(
                action="Consider a relevant add-on conversation",
                reason=(
                    f"The predicted probability is {probability_text}, below the elevated-risk threshold of {medium_text}. "
                    f"Tenure is {int(float(tenure))} {'month' if int(float(tenure)) == 1 else 'months'} and the account has {addon_count(features)} add-on services. "
                    "This is an expansion idea for a lower-risk customer, not a retention save."
                ),
                effort="low",
                category="expansion",
            )
        )

    if high and not recommendations:
        recommendations.append(
            Recommendation(
                action="Place the customer in a personal retention review",
                reason=(
                    f"The predicted probability is {probability_text}, at or above the high-risk threshold "
                    f"of {high_threshold:.1%}, and none of the more specific service rules matched. "
                    "Start with a service review rather than a pricing or contract change."
                ),
                effort="medium",
                category="retention",
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                action="No immediate outreach",
                reason=(
                    f"The predicted probability is {probability_text}, below the elevated-risk threshold of {medium_text}. "
                    "Keep the customer on the regular service cycle."
                ),
                effort="low",
                category="monitor",
            )
        )

    rank = {
        "retention": 0,
        "contract": 1,
        "onboarding": 2,
        "pricing": 3,
        "service_review": 4,
        "support": 5,
        "payment": 6,
        "expansion": 7,
        "monitor": 8,
    }
    recommendations.sort(key=lambda item: rank.get(item.category, 9))
    return recommendations
