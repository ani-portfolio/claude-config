# Baselines and Metrics

## Why baselines come first

A model score means nothing on its own. A recall of 0.33 is excellent if today's rule gets 0.18 and poor if sorting by one column gets 0.40. Baselines are the non ML references: they show what is possible without a model, so model training can show what ML adds.

All ML models, simple and complex, are built in ml-model-training, starting with logistic or ridge regression. That first simple model answers whether the features carry signal at all: if it barely beats the reference, tuning a stronger model will not fix it.

## Non ML baselines

| Option | Type | Classification | Regression | Forecasting |
|---|---|---|---|---|
| Current business rule | `rule` | The rule as a pandas expression, such as `days_since_last_order > 30` | The rule's estimate, such as `orders_90d * 30` | Rarely used; the naive forecasts play this part |
| Base rate or mean | `prior` | The training base rate for every row. AUC 0.5; the floor | The training mean | Use `series_mean` instead |
| Best single feature | `column` | Rank by the strongest single feature from the feature check | Not useful unless a column is already an estimate | Not used |
| Segment average | `segment` | The target rate per segment (for example plan and region), learned on the fold's training rows | The mean per segment | `series_mean`: the mean of each series |
| Naive | `column` | | | The value one horizon ago (`sales_lag_28d`) |
| Seasonal naive | `column` | | | The value on the same day of the season (a lag that is a multiple of 7, or 364 days) |
| Moving average | `column` | | | The rolling mean ending one horizon before (`sales_roll_28d_mean_shift_28d`) |

A rule that returns only true or false ranks every flagged row equally. For "at k" metrics, ties are broken at random, which is how the rule would work in practice. A rule can rank within its flags if it is written as a score, for example `(days_since_last_order > 30) * days_since_last_order`.

Baselines may only use columns that a model may use: columns excluded in the framing, and features the leakage check left out, are refused. The segment average is a lookup table learned on each fold's training rows, not a model, so it counts as non ML.

## The reference

The reference is the baseline models are measured against, normally today's approach (the `rule` baseline), because the V1 bar in the framing is stated against it. When there is no current approach, the best non ML baseline is the reference. When the best non ML baseline beats the current rule (as when sorting by recency beats a 30 day cut off), a model must beat that too; otherwise a simpler change to the rule would do.

## Metrics

| Key | Meaning | Better |
|---|---|---|
| `roc_auc` | How well the score ranks positives above negatives, 0.5 random, 1.0 perfect | higher |
| `pr_auc` | Average precision; more informative than AUC when positives are rare | higher |
| `log_loss`, `brier` | Quality of the probabilities; only for baselines that output probabilities | lower |
| `recall_at_k` | Share of all positives found in the top k | higher |
| `precision_at_k` | Share of the top k that are positive | higher |
| `lift_at_k` | Precision at k divided by the base rate | higher |
| `rmse`, `mae` | Typical error, in the target's units; RMSE punishes large errors more | lower |
| `mape` | Mean absolute percentage error, skipping zero targets | lower |
| `wape` | Total absolute error divided by the total actual; the usual forecasting metric | lower |
| `r2` | Share of variance explained | higher |
| `bias` | Mean of prediction minus actual; shows systematic over or under forecasting | closer to 0 |

The "at k" metrics take `k` (rows) or `k_share` (a share), and `k_per` (a column such as the scoring date). With `k_per`, the top k are taken within each date, which matches a weekly capacity such as 500 offers a week. k must be smaller than the rows in a typical group, or every row is picked and every baseline scores the same; `--suggest` warns about this, and `k_share` is the fix when the sample is smaller than production.

The primary metric comes from `framing.success.primary_metric`. Secondary metrics give context, especially a probability metric (log loss) when the scores will be used as probabilities.
