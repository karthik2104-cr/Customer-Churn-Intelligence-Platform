# Business metrics

Every dollar figure is an estimate. The IBM file has no cost, margin, campaign, or save-rate column. Marking a customer "retained" stores a follow-up status. It is not a measured save.

## Risk

- High risk: predicted probability at or above the decision threshold. That threshold is also the positive class.
- Medium risk: probability at or above the medium cutoff and below the decision threshold.
- Low risk: below the medium cutoff.
- The shipped defaults are 0.10 and 0.16, chosen on the validation split under a 5:1 false-negative to false-positive cost assumption. Change them in Settings. Stored scores are relabeled; the model is not retrained.

Predicted probability is not certainty. The reliability note says whether the row is near 0.5, outside the training range, or typical.

## Priority

`100 × (0.60 × probability + 0.25 × monthly charges / population 95th percentile, capped at 1 + 0.15 × (1 − tenure / 72), capped at 1)`.

This ranks outreach. It is not an optimized profit model.

## Recommendations

Rules look at contract, tenure, charges, payment method, internet service, and support add-ons. They do not read gender or senior-citizen status. A training-split association sentence is appended only when that table exists. Association is not causation.

## Financial scenarios

Targeted customers are Medium and High risk. If an intervention capacity is set, the highest probabilities are kept until the capacity is filled.

- Expected churners = sum of predicted probabilities among targeted customers. This is not a count of certain losses.
- Revenue exposure = sum of (average monthly revenue × horizon × that customer's probability).
- Contribution exposure = revenue exposure × contribution margin.
- Expected retained customers = expected churners × scenario success rate.
- Contribution benefit = expected retained customers × average monthly revenue × horizon × contribution margin.
- Campaign cost = targeted customers × cost per customer.
- Net benefit = contribution benefit − campaign cost.
- Break-even success rate = campaign cost / (expected churners × average monthly revenue × horizon × contribution margin).

Scenarios multiply the base success rate by 0.5, 1.0, and 1.5, and cap the rate at 1. Revenue and contribution are shown separately. The horizon is counted once. There is no second lifetime-value term.

Observed historical churn and predicted risk are different quantities. The overview shows both and does not add them together.
