# Feature Rules

## Three kinds of steps

| Kind | Examples | When it runs | Rule |
|---|---|---|---|
| Row steps | calendar parts, days since a date, log, ratios | Once, on every row | Safe: a row's value depends only on that row. |
| History steps | lags, rolling windows | Once, on the full timeline (training, gap, and test rows together) | Every window ends at or before the prediction time. Lags and shifts are at least `min_shift_days` (the forecast horizon), and lags of the target at least `target_min_shift_days` (the label delay). |
| Learned steps | imputing, one hot and ordinal encoding, scaling, TargetEncoder | Inside each CV fold, fitted on that fold's training rows | Never fitted on validation or test rows. |

`src/features.py` keeps the first two in `build_features(df, config)` and the third in `make_preprocessor(kind, config)`, an unfitted pipeline. The feature tables saved by the notebook hold the inputs before the learned steps, so later stages can fit the preprocessor inside every fold.

## Which columns become inputs

The data dictionary decides first:

| Dictionary timing | Treatment |
|---|---|
| `identifier` | Left out. An entity ID stays in the feature tables as a key for grouped folds. |
| `static`, `before_prediction`, `at_event` | Used. |
| `event_time` | Turned into days from that date to the prediction date (or calendar parts when there is no prediction date). |
| `possibly_after_outcome`, `unknown` | Left out until the user confirms the timing, then recorded in `decisions`. |

Then the column type: numbers and booleans are numeric; text with up to about 50 values is categorical; text with more values is high cardinality; long free text is left out in V1. Columns in `data.excluded_cols` are always left out. A column that is constant in the training rows is left out.

## Calendar parts

Month and week only help when every month has been in training at least once, which needs more than a year of history. With less, the test period has months the model never saw, and the drift check shows it. Day of week is safe for daily data.

## Two preprocessors

| Input | `linear` | `tree` |
|---|---|---|
| Numeric | median imputation with missing flags, then scaling | as is (tree models handle missing values) |
| Categorical | one hot, rare categories grouped, unseen categories ignored | ordinal codes, rare grouped, unseen as -1, missing as -2 |
| High cardinality | TargetEncoder, then scaling | TargetEncoder |

TargetEncoder cross fits inside its own fit: the encoding of each training row is learned from the other parts of the training data, so the model does not see its own target through the encoding. On new rows it uses the encoding learned on all training rows.

## Leakage check

Each input alone, scored on the CV folds: AUC for classification (the larger of AUC and 1 minus AUC), absolute Spearman correlation for regression and forecasting. AUC is used rather than correlation because it measures how well the feature ranks the outcomes, whatever the shape of the relationship and whatever the class balance; correlation misses strong curved relationships and shrinks when the positive class is rare. Categorical inputs are scored through the target rate of their category in the fold's training rows, so an ID cannot score well by memorising rows.

A feature that scores 0.9 or more alone is left out unless the user confirms it is known at the prediction time (`confirmed_safe`, with a decision and a reason). Typical real features score 0.55 to 0.75.

## Drift check

The population stability index between training and test rows: under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 large. A time split always shifts calendar parts and, when the series level changes, lags of the target; those are reported but not counted as problems. A large shift in another feature can mean its meaning changed over time, which deserves a question to the data owner.

## Forecasting

Direct forecasting in V1: lags of the target of at least the horizon (for example 28 and 35 days for a 28 day horizon, plus 364 days when there is more than a year of history), rolling means ending at least the horizon before the row, calendar parts, and the series ID as a category so the model can learn each series' level.

## Hand written features

`src/custom_features.py` belongs to the user. The skill creates it once and never changes it. It runs after the built in steps when `custom: True`. Its new columns are only used when they are added to the numeric or categorical lists. The rules are the same: only information known at the prediction time, and nothing learned from all rows.
