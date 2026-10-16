# Business Framing Question Bank

Each business framing field, why it matters, what a good answer looks like, and the red flags to check. None of these need the data. The ML definition (target column, unit of prediction, prediction time, exact metric) comes later, in ml-problem-framing.

## Business goal (`business_goal`, `business_kpi`)

**Ask:** What business outcome should improve, and how is it measured?

**Why it matters:** It decides which errors are expensive, which in turn decides the metric and the threshold.

**Good answer:** "Reduce fraud losses on card not present transactions, measured as basis points of fraud on approved volume."

**Red flags:**

1. The goal is the model itself, such as "build a churn model". Ask what changes in the business if the model works.
2. There is no KPI, so nobody can say later whether the model helped.

## Decision supported (`decision_supported`, `users`, `capacity`)

**Ask:** What action does the output trigger, who takes it, and how many actions can they take?

**Why it matters:** Capacity limits turn the problem into "rank the top N well", which changes the metric. The person acting on the output decides how much explanation is needed.

**Good answer:** "The retention team calls the 500 customers most likely to lapse each week."

**Red flags:**

1. No one acts on the output. A score without an action rarely gets used.
2. The capacity is fixed, but the metric is a global one such as AUC. Suggest precision or recall at the capacity instead.

## Current approach (`current_approach`)

**Ask:** How is this decision made today without a model?

**Why it matters:** It is the most meaningful baseline for stakeholders, and the V1 bar is usually stated against it.

**Good answer:** "A rule flags customers with no purchase in 30 days."

**Red flags:** "Nothing" is a valid answer, but then the baseline skill should build a simple heuristic so V1 is still compared with something.

## Outcome in business terms (`outcome`)

**Ask:** In business terms, what should the model predict?

**Why it matters:** It is the starting point for the exact target in ml-problem-framing, and it keeps the target tied to the decision.

**Good answer:** "Customers who are about to stop ordering." The exact rule and time window wait until the data has been explored.

**Red flags:**

1. The outcome is something the business cannot act on, or learns about too late to act.
2. The outcome is a proxy that only loosely tracks the goal, such as predicting clicks when the goal is purchases. Say so, and record it as an assumption if the user keeps it.

## How the output is used (`output_use`, `v1_bar_intent`)

**Ask:** What form does the output take when someone acts on it, and what would make V1 worth using?

**Why it matters:** It decides the metric family before anyone looks at the data, so the metric follows the decision and not the other way round.

| Output use | What it means | Metric family (settled in ml-problem-framing) |
|---|---|---|
| `ranked_list` | Act on the top N or top k% each period, often because of a capacity limit | Precision or recall at k |
| `threshold_decision` | Approve or decline, flag or not, at a cut off | Precision and recall at the threshold, or an error cost |
| `probability` | The score is used as a number, for example in pricing or expected value | Log loss or Brier score, plus calibration |
| `quantity` | A number such as demand or spend | MAE, WAPE, or RMSE |
| `other` | Ranking of items for a user, segments, and so on | Decided in ml-problem-framing |

**Good V1 bar intent:** stated against the current approach, for example "finds more of the customers who lapse than the 30 day rule, at the same 500 calls a week". ml-problem-framing turns it into a number once the baseline and the class balance are known.

**Red flags:**

1. "As accurate as possible". Ask what result would change the decision to use it.
2. A bar stated only as a model metric, such as "90% accuracy", with no link to the current approach.

## Is ML needed? (`ml_needed`, `ml_needed_reason`)

**Ask yourself, then the user:** could a rule, a report, or a dashboard serve the decision as well?

**Why it matters:** A model is worth its cost only when it beats the simple option by enough to change outcomes. Concluding "not needed" early is a good result, not a failure.

**Signs ML is not needed:** the decision is rare or low value; a simple rule is already close to perfect; there is no way to act differently on a better prediction; nobody will own the model once it is built.

**Record:** `yes`, `no`, or `unsure`, with one sentence of reasoning. `unsure` is fine at this stage: ml-data-discovery and the baseline will tell.

## Constraints (`constraints.*`, `out_of_scope`)

**Ask:** Are there limits on latency, interpretability, fairness, or the data that may be used?

**Why it matters:** These rule out some model types and some features before time is spent on them.

**Examples:** "Batch scoring only, so latency does not matter." "Reason codes are required for every declined application." "No personal data or protected attributes as features."

**Red flags:**

1. Regulatory or explanation needs are discovered after a complex model is built.
2. Features are planned from data the production system cannot access.

## Feedback loops

**Ask:** Will acting on the model's output change the outcome it predicts?

**Why it matters:** If retention offers stop churn, future labels for treated customers no longer show who would have churned. Plan a holdout group or record treatment so later training is not biased.

**Record:** as an assumption or an open question for the evaluation stage.
