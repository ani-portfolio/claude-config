# Checks and Thresholds

This file explains how each check works, so the report's findings can be explained and tuned. The thresholds are constants at the top of `scripts/eda_core.py`.

## Column roles

Every column gets a role before analysis. The role decides which checks run and whether values may be shown.

| Role | How it is detected | Values shown? |
|---|---|---|
| Identifier | Given in `id_cols`, or a name like `id`, `*_id`, `uuid`, or `*_key` with more than 90% unique values. Text columns with more than 95% unique values and over 1,000 distinct values also count | No |
| Possible personal data | A name containing words such as email, phone, card number, account number, name, address, postal, birth, or password, or values that look like emails or 12 to 19 digit numbers | No |
| Free text | Text with an average length above 30 characters, or above 15 characters with more than half the values unique | No, length statistics only |
| Dates stored as text | Over 90% of sampled values look like dates and parse as dates | Date range only |
| Numbers stored as text | Over 95% of sampled values parse as numbers after removing commas, currency signs, percent signs, and spaces | Analysed as numbers |
| Categorical | Other text or category columns | Yes, if the value appears at least 5 times |
| Numeric, boolean, datetime | From the pandas dtype | Summary statistics |

## Thresholds

| Check | Rule |
|---|---|
| Near constant column | One value, including missing, covers at least 99% of rows |
| High missing | 50% or more is High; 5% or more is Medium |
| Disguised missing (text) | Values such as "", "NA", "null", "unknown", "?", or "-" |
| Disguised missing (numeric) | -1, -9, -99, -999, -9999, 999, 9999, 99999, or 999999, when the value is the column's minimum or maximum and covers at least 1% of rows |
| Missing together | Pearson correlation (phi coefficient) of the 0 or 1 missing indicators of two columns, over all rows, of 0.8 or more. Works for numeric and categorical columns alike. Listed in a table of up to 30 pairs, from up to 100 columns |
| Missingness group | Columns whose missing indicators correlate at 0.99 or more are merged into one group (a column with no partner is a group of one). Each group gets a verdict: likely structural when a rule explains it, probably structural when it has two or more columns, partly explained, or no pattern found |
| Missingness rule | A value of another column (numeric columns split into 10 equal sized ranges) that covers at least 1% of rows, is 95% or more all missing or all present, and differs from the overall missing rate by at least 20 points. 98% or more that also accounts for at least half of the group's missing rows (or present rows, for a rule where the column is almost always present) counts as explaining the group; a smaller share is called partly explained. Searched on up to 200,000 rows |
| Outliers | Outside Q1 minus 1.5 × IQR or Q3 plus 1.5 × IQR; flagged when above 5% of rows |
| Highly skewed | Absolute skew above 2 |
| High cardinality | More than 50 unique values |
| Rare category | Under 1% of rows |
| Inconsistent spelling | Values that match after lowercasing and removing spaces and punctuation |
| Placeholder dates | 1900-01-01, 1970-01-01, and 9999-12-31 |
| Date ordering | Flagged when the unusual order covers more than 0% and less than 5% of rows |
| Empty periods | Days, weeks, or months with no rows inside the date range; low periods have under half the rolling median |
| High correlation | Absolute Spearman correlation of 0.9 or more (`corr_threshold`) |
| Strong categorical association | Cramér's V of 0.8 or more (`cramers_threshold`) |
| Imbalanced target | Largest class at least 10 times the smallest |
| Possible leakage | Association with the target of 0.95 or more (High). A name that contains the target's name or suggests post outcome data (such as outcome, actual, after, final, or settled), or missingness correlated with the target at 0.5 or more (Medium) |

## Measures

* **Spearman correlation** ranks the values first, so it captures any monotonic relationship and is robust to skew and outliers.
* **Cramér's V** measures association between two categorical columns, from 0 to 1, with a bias correction.
* **Eta (correlation ratio)** measures how much of a numeric column's variance is explained by a categorical column, from 0 to 1.
* **Mutual information** captures non linear relationships. It needs scikit-learn and is computed on up to 50,000 rows.

## Flags and grouped issues

Every issue is recorded once for each column it affects. The Flags column in the profile tables shows a short label for each column's issues. The summary section groups issues with the same type and severity into one row and lists the affected columns, with up to 12 named before "and N more".

## Plots

Plots use seaborn with the whitegrid style. Numeric and categorical plots are drawn as grids with 3 columns and at most `max_plots` panels (default 24). When there are more columns than that, numeric columns are chosen by skew and categorical columns in column order. The target grid shows up to 9 of the features most related to the target.

## Inferring the target column

`run_eda.py --list-columns` ranks columns with `suggest_targets()` in `eda_core.py`. The skill always confirms the result with the user.

| Signal | Score |
|---|---|
| Name is `target`, `label`, `y`, `class`, `outcome`, or `response` | +6 |
| Name suggests an outcome, such as `is_*`, `has_*`, `churn`, `fraud`, `click`, `convert`, or `*_flag` | +3 |
| Name suggests a measure that is often predicted, such as `revenue`, `ltv`, `sales`, or `amount` | +1 |
| Binary, or 3 to 10 classes | +2, or +1 |
| Continuous numbers | +0.5 |
| First or last column | +1 |
| Constant | minus 8 |
| Almost every value unique (except float columns), or named like an ID or personal data | minus 6 each |
| Dates, or free text | minus 6 |
| Text with more than 50 unique values | minus 3 |
| More than 50% missing | minus 2 |
| Name suggests data recorded after the outcome | minus 0.5 |

## Task type for the target

The skill asks the user whether the target is continuous or categorical, and passes the answer as `target_type`. A continuous target is analysed as regression and a categorical target as classification. When no type is given, a numeric target with more than 20 unique values is treated as continuous, and everything else as categorical. The report says which rule was used.

## Checks against the ML framing

These run only when `ml_spec.yaml` exists and its framing status is `draft` or `confirmed`. Each check gets Pass, Check (a person should confirm), Fail (the data contradicts the framing), or Skipped. Every Fail is also listed as a High issue, and Fail and Check results that need an answer are added to the spec's open questions.

| Check | Fail when | Check when |
|---|---|---|
| Target column exists | `framing.target.name` is not a column and `exists_in_data` is not false | The target must still be built, or no target is named |
| Target matches the framing | | The report analysed a different target than the framing names |
| Target type matches the task type | A classification task has a continuous target, a regression or forecasting task has a categorical target, or a binary task has more than two classes | |
| Metric suits the class balance | The smaller class is under 5% and the primary metric is accuracy | The smaller class is under 5% and no metric is set |
| Entity ID columns exist | An entity ID column is not in the data | |
| Rows per entity | | An entity ID repeats, so the split must be grouped or time based |
| Grain matches the unit of prediction | | Rows repeat on the entity ID plus the prediction time column, or no entity ID is set |
| Prediction time column exists | The column is not in the data, or the task is forecasting and no date column is set | The column cannot be read as dates |
| Excluded columns | | An excluded column is not in the data |
| Leakage suspects handled | | A leakage suspect is not in `data.excluded_cols` |

