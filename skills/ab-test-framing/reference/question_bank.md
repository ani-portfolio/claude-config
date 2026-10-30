# Question Bank for the Test Framing

Each field of the `framing` section, why it matters, what a good answer looks like, and the red flags to push back on.

## Hypothesis and change

| Field | Why it matters | A good answer | Red flags |
|---|---|---|---|
| `hypothesis` | It says what should happen and why, so the result can confirm or refute something. | "Because the ad shows the product, users who see it will convert more often than users who see a public service ad." | "See if the ads work." No direction, no metric, or no mechanism. |
| `change` | The analysis can only credit the effect to what actually differed. | "The treatment group saw product ads; the control saw public service ads in the same slots." | The control also changed, or several things changed at once without saying so. |
| `test_state` | A planned test gets a sample size. A finished test gets a power check. | planned, running, or finished. | A running test that people want to "check early": that needs a sequential method, or the false positive rate rises. |

## Decision and owner

| Field | Why it matters | A good answer | Red flags |
|---|---|---|---|
| `decision` | A test that changes no decision is not worth its cost. | "Keep funding the campaign at the current budget, or stop it." | "Learn about users." Nobody will act differently on either result. |
| `owner` | Someone must make the call when the result is mixed. | A named role. | A committee with no tie breaker. |

## Unit and population

| Field | Why it matters | A good answer | Red flags |
|---|---|---|---|
| `unit` | Intervals assume independent units. Analysing page views when users were randomised makes them far too narrow. | "user", the thing that was assigned. | Randomised by user, analysed by session or impression. Units that can see both variants. |
| `population` | Results only hold for the people who were eligible. | "Logged in users on the shop during the test." | Generalising to a population that was not in the test. |

## Variants and planned split

| Field | Why it matters | A good answer | Red flags |
|---|---|---|---|
| `variants` | The sample ratio check needs the planned share. A data share far from the plan means broken assignment or logging. | One control, one or more treatments, planned shares that add up to 1. | "The split is whatever is in the data." The planned split must come from how the test was set up, not from the data. An unequal split such as 96 to 4 is fine if it was planned, but the control must still be large enough. |

## Metrics

| Field | Why it matters | A good answer | Red flags |
|---|---|---|---|
| `primary_metric` | One metric decides the test, so the false positive rate stays at alpha. | One metric, with a type (binary, continuous, count), the good direction, and an exact definition. | Several primary metrics. A metric that can move without the business improving (clicks without orders). |
| `secondary_metrics` | They explain the result. | Two or three, reported with a Holm correction. | Using a significant secondary to rescue a flat primary. |
| `guardrail_metrics` | They catch harm the primary does not see. | Unsubscribes, refunds, latency, with `max_harm`. | No guardrails for a change that could annoy users. |

## Success

| Field | Why it matters | A good answer | Red flags |
|---|---|---|---|
| `mde` | It sets the sample size and decides whether a significant result is big enough to matter. | "0.5 percentage points", with a reason in cost or value. | "Any improvement." Every real change is non zero, so this cannot be planned or judged. |
| `ship_rule` | Agreed before results, so the result cannot move the goalposts. | "Ship if the primary improves at the 5% level and no guardrail is harmed beyond its limit." | A rule written after seeing the numbers. |

## Common problems to raise

1. **Post-treatment columns as segments or controls.** Counts of ads seen, time on site after assignment, or anything the treatment can change. Note them for ab-data-validation, which keeps them out of adjustments and segments.
2. **Peeking.** Looking at a running test repeatedly and stopping at the first significant result. Recommend a fixed horizon or a sequential method.
3. **Novelty effects.** A short test of a visible change may measure curiosity. Recommend whole weeks and a check over time.
4. **Network effects.** Units that affect each other (marketplaces, social features) break independence. Note it as a risk.
5. **No real control.** A before and after comparison is not an A/B test. Say so, and keep the result labelled as observational.
