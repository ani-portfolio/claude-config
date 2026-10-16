# Split Strategies

The split decides what "unseen data" means for this model. The test score is only honest if the test set looks like what the model will face in production, and if nothing about the test set leaked into training.

## The rule

Decide the split before any feature work, and fit anything that learns from the data (imputers, encoders, scalers, target encoders, feature selection) on training rows only. Steps that learn nothing, such as a log, a ratio, or the day of the week, are safe on any rows.

The split is a plan, not only a cut. For a time split, lag and rolling features are built on the full timeline, with every lag at least as long as the forecast horizon, and the rows are cut by date afterwards. The test rows need the training period's history for their lags, just as the model will have that history in production.

## Choosing a strategy

| Situation | Strategy | Test set | Cross validation |
|---|---|---|---|
| Forecasting | `time` | The latest period, one or more horizons long | Rolling backtest folds of one horizon, no gap (lags are shifted by the horizon instead) |
| The prediction is made at a point in time (a scoring date, an event time) | `time` | The latest period | Rolling backtest folds, with a gap of at least the outcome window or label delay |
| An entity (customer, store, device) has several rows, and there is no time order to respect | `group` | A share of the entities | Group k fold, stratified when classification |
| Independent rows, no time order | `random` | A share of the rows, stratified when classification | Stratified k fold (k fold for regression) |

When in doubt between `time` and the others, prefer `time`: a model that predicts the future should be tested on the future. A random split of time ordered data usually gives a score that is too good.

## Gaps

A row's label is only known once its outcome window has passed. If a churn label looks 60 days ahead, the training rows in the last 60 days before the test period have labels that are decided inside the test period. A gap of `max(outcome window, label delay)` days between training and test, and inside every backtest fold, removes that overlap. The rows in the gap are saved as `gap.parquet`: they are not used to train or test, but feature engineering may use them as history.

Forecasting usually needs no gap, because the features themselves are shifted: a forecast made 28 days ahead may only use lags of 28 days or more.

## Rolling backtest folds

Each fold validates on one window and trains only on dates before it (minus the gap). The windows move forward in time, ending where the final test period begins.

* `expanding`: every fold trains from the first date. Uses all history, the usual choice.
* `sliding`: every fold trains on the last `train_days` only. Use it when old data no longer reflects the present.

Fewer, longer folds give steadier scores, and more folds show how stable the model is over time. Two to four folds is typical. The first fold must still have enough training data.

## Duplicates

Exact duplicate rows split across train and test inflate the score, because the model is tested on rows it has seen. `keep_together` (the default) puts every copy on the same side as the first one. `drop` keeps one copy. `ignore` does nothing and is only right when copies are genuine repeated events. A time split keeps copies together anyway when they share the date.

## Sizes

* Random and group: a test share of 0.2 is typical. With fewer than about 5,000 rows, consider 0.25 to 0.3, so the test score is not too noisy.
* Time: the test period should cover at least one full business cycle that matters (for example, several weeks of a weekly decision) and leave most of the labelled history for training.
* Classification: every fold should have enough rows of the smallest class to measure the metric, roughly at least 50 positives per validation fold.

## Red flags

1. A random split of data where the prediction is made at a point in time.
2. A time split without a gap when the label looks ahead.
3. The same entity in training and test when the model will score new entities in production.
4. A test set that is reused to tune the model. It is used once, at the end. Tuning uses the CV folds.
5. Class balance that differs a lot between sets, which shows a problem with stratification or a change over time.
