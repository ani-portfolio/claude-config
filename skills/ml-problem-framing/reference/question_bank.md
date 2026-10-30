# ML Framing Question Bank

Each ML framing field, why it matters, what a good answer looks like, and the red flags to check, followed by the checks of the framing against the data. ml-problem-framing builds on the business framing (`business` in the spec) and the discovery findings (`discovery` in the spec and `docs/data_dictionary.yaml`).

## Target (`target.*`, `task_type`)

**Ask:** What exact event or value is predicted, what counts as positive, and over what window?

**Why it matters:** Most V1 failures come from a vague or leaky target.

**Good answer:** "Positive if the customer has zero orders in the 60 days after the scoring date. Customers who closed their account in that window are excluded."

Fields:

* `name`: the column, if it already exists in the data.
* `definition`: the rule in plain language.
* `positive_class`: what the positive class is called and coded as.
* `outcome_window`: the time after the prediction over which the outcome is observed.
* `exists_in_data`: true, false, or null when unknown. When false, `construction` explains how to build it.
* `label_delay`: how long after the prediction the label becomes known.

**Red flags:**

1. The definition is a word, not a rule ("churn", "fraud", "high value").
2. The window is missing, so the label depends on when the data was pulled.
3. The target is built from fields that are only filled in after the outcome, such as a cancellation reason.
4. Edge cases are undefined: customers who leave the population, refunds, chargebacks that arrive late.
5. The positive class is very rare (below 1%) and there is no plan for evaluation at that rate.

## Unit of prediction (`unit_of_prediction`, `entity_id_cols`, `population`)

**Ask:** What does one row represent at prediction time, and which population is scored?

**Why it matters:** It sets the grain of the training table, the split strategy, and the evaluation unit.

**Good answer:** "One active customer on each weekly scoring date. Active means at least one order in the past 180 days."

**Red flags:**

1. The unit does not match the decision, for example predicting per transaction when the action is per customer.
2. The same entity appears many times, but nobody has said whether the split should be grouped by entity.
3. The training population includes people who would never be scored in production, or the reverse.

## Prediction time (`prediction_time.*`)

**Ask:** At what moment is the prediction made, what data is available at that moment, and how often is it made?

**Why it matters:** This is where leakage is prevented. Every feature must be computable from data available at this moment.

Fields:

* `description`: the moment in plain language.
* `column`: the date or timestamp column that marks that moment in the data, if one exists.
* `feature_cutoff`: the latest data the features may use, for example "end of the previous day".
* `cadence`: real time, hourly, daily batch, weekly batch.
* `horizon`: for forecasting, how far ahead is predicted.

**Red flags:**

1. Nobody can say when the prediction is made. Push for an answer, because the split and features depend on it.
2. Data arrives with a lag (for example, settlement data two days late) but the cutoff ignores it.
3. The outcome window starts before the feature cutoff.

## Success (`success.*`, `task_type`)

**Ask:** Which offline metric will decide whether V1 is good enough, and what value counts as good enough?

**Why it matters:** The evaluation skill reports against these, and a bar agreed now prevents moving the goal posts later.

**Metric guide by task type:**

| Situation | Suggested primary metric | Avoid |
|---|---|---|
| Binary, balanced classes | ROC AUC or log loss | |
| Binary, rare positive class | PR AUC | Accuracy |
| Fixed review capacity (top N or top k%) | Precision or recall at k | Global AUC alone |
| Scores used as probabilities (pricing, expected value) | Log loss or Brier score, plus a calibration check | Ranking metrics alone |
| Multiclass | Macro F1, or log loss | Accuracy when classes are imbalanced |
| Regression | MAE, or RMSE when large errors matter most | R squared alone |
| Regression with a skewed target (spend, revenue) | MAE on the original scale, or RMSLE | RMSE dominated by a few large values |
| Forecasting | WAPE or MASE, by series and overall | MAPE when actuals can be zero |
| Ranking | NDCG at k or MAP at k | |

**Good V1 bar:** stated against the current approach or a simple baseline, for example "PR AUC at least 0.10 higher than the current rule" or "WAPE below the seasonal naive forecast".

**V1 criteria:** the same bar as checks in `success.v1_criteria`, so no script ever parses the text. "WAPE below the seasonal naive forecast on CV and on the test set" becomes two `beats_model` criteria (`model: seasonal_naive`, `on: cv` and `on: test`). A business guardrail such as "every media effect is non negative" becomes a `check` criterion naming a function in `src/v1_checks.py`. Anything that only a person can judge is `manual`, and keeps the bar from being met by a script alone. Red flag: a criterion that names a model or baseline that will not be built; the consistency checks report it once training has run.

**Guardrails:** metrics that must not get worse, such as the false positive rate for good customers, or performance in a key segment.

**Red flags:**

1. The metric does not match the decision (see the table).
2. The bar is an absolute number with no reference, such as "90% accuracy".
3. There is no guardrail for a group that could be harmed by errors.

## Checking the framing against the data

These use `discovery.framing_hints`, `discovery.sources`, `discovery.joins`, and the data dictionary.

**History against the outcome window.** A row only has a complete label once its whole outcome window has passed, or later if the label arrives with a delay. The label delay is counted from the prediction, so it already includes the window. The labelled history is roughly:

labelled history = `history_days` minus the larger of the outcome window and the label delay

Part of that history then goes to the test period, which should cover at least one full business cycle (often one to three months, and a full year when the outcome is seasonal). Example: 357 days of history with a 60 day window leaves about 297 days of labelled rows. A 90 day test period then leaves about 207 days for training. If what remains for training is short, offer a shorter window, more history, or a smaller test period, and record the choice in `decisions`.

**Candidate targets.** A candidate column only fits when it measures the business outcome from the draft over a defined window. A column such as `churned` with no window behind it is a red flag: ask when it was set and over what period. When no candidate fits, the target is built from events, and `target.construction` must say how.

**Grain and entity IDs.** The unit of prediction should match the detected grain, or the framing must say how rows are aggregated to it. When an entity ID has more than one row per value, the split must keep each entity on one side (a grouped split) or split by time.

**Prediction time.** The prediction time column should be a date that is known when the prediction is made, such as a scoring date or an event timestamp, not a date filled in later, such as a resolution or settlement date.

**Class balance.** Use the smaller class share of the chosen binary target to confirm the metric: under about 5%, prefer PR AUC or recall at k over ROC AUC or accuracy.

**Columns recorded after the outcome.** Every `post_outcome_suspects` column needs a decision now: exclude it, or confirm when it is filled in. The targeted EDA checks the rest with the target.

**Timing in the data dictionary.** Every column whose `timing` is `possibly_after_outcome` or `unknown` needs a decision before it can be a feature: exclude it, or confirm when it is filled in and update the dictionary. Columns with `timing: before_prediction` are only safe if their window ends at or before the prediction time, so check that against `prediction_time.feature_cutoff`.

**Joins.** When the target or the features come from another source, check the join in `discovery.joins`. A many to many join must be aggregated first, and a low match rate means part of the population has no features or no label.
