"""Configurable financial scenarios.

Every dollar figure is an estimate from user-supplied assumptions. The dataset
does not contain cost, margin, or campaign-outcome fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORMULAS = [
    "Customers targeted = Medium and High risk customers. When an intervention capacity is set, keep the capacity-sized group with the highest predicted probabilities.",
    "Expected churners = sum of predicted churn probabilities among targeted customers. This is not a count of certain losses.",
    "Revenue exposure = sum over targeted customers of (average monthly revenue x horizon months x that customer's predicted probability).",
    "Contribution exposure = revenue exposure x contribution margin.",
    "Expected retained customers = expected churners x scenario success rate.",
    "Estimated contribution benefit = expected retained customers x average monthly revenue x horizon months x contribution margin.",
    "Campaign cost = number of targeted customers x cost per customer.",
    "Estimated net benefit = estimated contribution benefit - campaign cost.",
    "Break-even success rate = campaign cost / (expected churners x average monthly revenue x horizon months x contribution margin).",
]

ASSUMPTIONS_NOTE = (
    "The dataset has no cost, margin, campaign, or save-rate fields. Every dollar figure is a "
    "scenario from the assumptions shown with it, not an observed financial result. Revenue is "
    "kept separate from contribution. A retained customer is counted across the horizon once; "
    "those months are not added again as a second lifetime-value term."
)

OUTCOMES_NOTE = (
    "Retention success and campaign effectiveness are unavailable. Marking a customer as retained "
    "records a user's follow-up status. It is not a measured save, and this dataset has no outcome field."
)


@dataclass(frozen=True)
class ImpactAssumptions:
    avg_monthly_revenue: float
    contribution_margin: float
    campaign_cost: float
    success_rate: float
    horizon_months: int
    intervention_capacity: int | None = None


def select_targeted(probabilities: np.ndarray, capacity: int | None) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        return values
    if capacity is None or capacity >= len(values):
        return values
    order = np.argsort(-values)
    return values[order[:capacity]]


def _scenario(name: str, multiplier: float, probabilities: np.ndarray, assumptions: ImpactAssumptions) -> dict:
    targeted = select_targeted(probabilities, assumptions.intervention_capacity)
    count = int(targeted.size)
    expected_churners = float(targeted.sum()) if count else 0.0
    success = min(1.0, float(assumptions.success_rate) * multiplier)
    capped = success + 1e-12 < float(assumptions.success_rate) * multiplier
    expected_retained = expected_churners * success
    revenue_exposure = float(targeted.sum() * assumptions.avg_monthly_revenue * assumptions.horizon_months) if count else 0.0
    contribution_exposure = revenue_exposure * assumptions.contribution_margin
    benefit = (
        expected_retained
        * assumptions.avg_monthly_revenue
        * assumptions.horizon_months
        * assumptions.contribution_margin
    )
    cost = count * assumptions.campaign_cost
    denominator = (
        expected_churners
        * assumptions.avg_monthly_revenue
        * assumptions.horizon_months
        * assumptions.contribution_margin
    )
    break_even = (cost / denominator) if denominator > 0 else None
    return {
        "name": name,
        "success_rate": success,
        "success_rate_capped_at_one": capped,
        "customers_targeted": count,
        "expected_churners": expected_churners,
        "revenue_exposure": revenue_exposure,
        "contribution_exposure": contribution_exposure,
        "expected_retained_customers": expected_retained,
        "estimated_contribution_benefit": benefit,
        "campaign_cost": cost,
        "estimated_net_benefit": benefit - cost,
        "break_even_success_rate": break_even,
    }


def calculate_impact(probabilities: np.ndarray, assumptions: ImpactAssumptions) -> dict:
    values = np.asarray(probabilities, dtype=float)
    scenarios = [
        _scenario("conservative", 0.5, values, assumptions),
        _scenario("base", 1.0, values, assumptions),
        _scenario("optimistic", 1.5, values, assumptions),
    ]
    return {
        "disclaimer": "These figures are estimates from the assumptions below, not observed campaign results.",
        "assumptions_note": ASSUMPTIONS_NOTE,
        "outcomes_available": False,
        "outcomes_note": OUTCOMES_NOTE,
        "formulas": FORMULAS,
        "assumptions": {
            "avg_monthly_revenue": assumptions.avg_monthly_revenue,
            "contribution_margin": assumptions.contribution_margin,
            "campaign_cost": assumptions.campaign_cost,
            "success_rate": assumptions.success_rate,
            "horizon_months": assumptions.horizon_months,
            "intervention_capacity": assumptions.intervention_capacity,
            "scenario_success_multipliers": {"conservative": 0.5, "base": 1.0, "optimistic": 1.5},
        },
        "scenarios": scenarios,
    }
