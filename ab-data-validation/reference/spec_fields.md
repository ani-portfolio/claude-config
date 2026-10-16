# Fields in ab_spec.yaml

`ab_spec.yaml` sits at the project root. Every A/B testing skill reads all of it first and writes only its own section, through `scripts/ab_common.py` or `nb_common.write_stage` from its notebook. The spec never holds raw rows. Column names, counts, rates, and test results are fine.

## project

Written by ab-project-setup. Only empty fields are filled when setup runs again.

| Field | Meaning |
|---|---|
| `name` | Project name. Also the Jupyter kernel name (`ab_common.kernel_name`). |
| `description` | One sentence on the test. ab-test-framing uses it as the first draft of the hypothesis. |
| `owner` | Who owns the project. |
| `python`, `kernel`, `kernel_display` | The `.venv` Python version and the registered kernel. |
| `created`, `updated` | Timestamps. |

## paths

The project folders, relative to the root: `data_raw`, `data_processed`, `notebooks`, `src`, `models`, `docs`. They are shared with `ml_spec.yaml` when both workflows use the same folder.

## framing

Written by ab-test-framing, from `notebooks/ab01_test_framing.ipynb`. Business terms only, with no column names.

| Field | Meaning |
|---|---|
| `status` | `not_started`, `draft`, or `confirmed`. The analysis is blocked until it is `confirmed`. |
| `test_name` | Short name for the test. |
| `test_state` | `planned`, `running`, or `finished`. A finished test skips the design questions and reports the power it had. |
| `hypothesis` | Because of the change, the metric will move in a stated direction, for a stated population. |
| `change` | What the treatment does differently from the control. |
| `decision` | What will be done with the result, for example roll out the campaign to everyone. |
| `owner` | Who makes that decision. |
| `unit` | The unit of randomisation, for example user, session, or store. The analysis unit must match it. |
| `population` | Who was eligible for the test. |
| `variants` | A list of `{name, role, planned_share}`. Exactly one `role: control`. Planned shares add up to 1, and the sample ratio check tests against them. |
| `primary_metric` | `{name, type, direction, definition}`. `type` is `binary`, `continuous`, or `count`. `direction` is `increase` or `decrease`, the direction that is good. One primary metric decides the test. |
| `secondary_metrics` | The same shape. Reported with a Holm correction, never used to rescue a failed primary. |
| `guardrail_metrics` | The same shape plus `max_harm`, the largest harm that is acceptable. A guardrail breach blocks shipping. |
| `mde` | The smallest effect worth acting on, in business terms or as a number. ab-test-design turns it into `design.mde_abs`. |
| `ship_rule` | The rule agreed before seeing results, for example ship if the primary is significant at 5% and no guardrail is harmed. |
| `out_of_scope` | What this test will not answer. |
| `risks` | A list of `{problem, resolution}` found when the framing was checked. |

## design

Written by ab-test-design, from `notebooks/ab02_design.ipynb`.

| Field | Meaning |
|---|---|
| `status` | `not_started`, `draft`, `confirmed`, or `skipped`. |
| `alpha`, `power`, `sides` | The significance level, the power, and `two` or `one` sided testing. The analysis uses the same values. |
| `baseline` | The control rate or mean the sample size is based on. `baseline_sd` for continuous metrics. |
| `mde_abs`, `mde_rel` | The minimum detectable effect, absolute and relative to the baseline. |
| `sample_size` | `{n_control, n_treatment, total}`. |
| `units_per_day`, `duration_days` | Traffic and the planned run time in whole weeks. |
| `achieved_power`, `achieved_mde` | For a finished test, the power to detect `mde_abs` and the smallest effect the actual sample could detect. |
| `looks` | `fixed` (one analysis at the end) or `sequential` (looks while running, which needs a sequential method). |
| `variance_reduction` | `none`, `cuped` with the pre-period column, or `stratification`. |

## data

Written by ab-data-validation. How the spec's names map onto the data.

| Field | Meaning |
|---|---|
| `source` | The data file, relative to the project root. |
| `unit_col` | The column that identifies a unit. |
| `group_col` | The column that holds the variant. |
| `variant_values` | `{variant name: value in group_col}`. |
| `metric_cols` | `{metric name: column}`, for every metric in the framing that the data has. |
| `date_col` | The assignment or exposure date, when there is one. |
| `pre_treatment_cols` | Columns recorded before assignment. Only these are used for balance checks, CUPED, and segments. |
| `post_treatment_cols` | Columns recorded after assignment, which the treatment can change. Never used to adjust or segment the effect. |

## validation

Written by ab-data-validation, from `notebooks/ab03_validation.ipynb`.

| Field | Meaning |
|---|---|
| `result` | `passed`, `warning`, or `failed`. The analysis is blocked on `failed`. |
| `failures`, `warnings` | Plain sentences, one per problem. |
| `override` | `{reason, by}` when the user chose to analyse despite a failure. |
| `srm` | The sample ratio test: observed and expected counts and shares, chi-square, p value, and `passed` (p of 0.001 or more). |
| `units` | Rows, distinct units, duplicate units, and units seen in more than one variant. |
| `balance` | One row per pre-treatment column: standardised mean difference or chi-square p value, and a flag. |
| `coverage` | The date range and units per day by variant, when there is a date column. |
| `run_at` | When the checks ran. |

## analysis

Written by ab-analysis, from `notebooks/ab04_analysis.ipynb`.

| Field | Meaning |
|---|---|
| `alpha`, `sides`, `mde_abs` | The settings used. They should match the design. |
| `primary` | `{metric, treatment, method, control_mean, treatment_mean, diff, diff_ci, rel_lift, rel_ci, p_value, significant, result_label}`. |
| `secondary`, `guardrails` | Lists of the same shape. Secondary metrics carry `p_holm`. Guardrails carry `breached`. |
| `checks` | The bootstrap interval and any notes on the method. |
| `cuped_col` | The pre-treatment column used for CUPED, or null. |
| `files` | The results parquet. |
| `run_at` | When the analysis ran. |

## segments

Written by ab-segment-analysis, from `notebooks/ab05_segments.ipynb`. `status: skipped` when no segments were planned.

| Field | Meaning |
|---|---|
| `results` | One entry per segment column: the effect per segment, Cochran's Q, its p value, and `p_bh` after the Benjamini Hochberg correction across columns. |
| `post_treatment_ok` | True only when the user chose to segment on a column recorded after assignment, with the reason in `decisions`. |

## readout

Written by ab-readout, from `notebooks/ab06_readout.ipynb`, together with `docs/ab_readout.md`.

| Field | Meaning |
|---|---|
| `status` | `draft` or `final`. |
| `result_label` | The primary result, copied from the analysis. |
| `recommendation` | `ship`, `do_not_ship`, `iterate`, or `rerun`, with `recommendation_reason`. The decision stays with the owner. |
| `business_impact` | The effect in business units, with its interval, for example extra conversions per 100,000 users. |
| `risks`, `next_steps` | Lists of plain sentences. |

## tracking

Written by ab-project-setup (`tracking.py setup`), or by the first step that logs a run: `uri` (the MLflow address, normally the shared server `http://127.0.0.1:5000`), `experiment` and `experiment_id` (`ab/<project>`), and `ui` (the link to the experiment).

## mlflow, in each step section

`framing`, `design`, `validation`, `analysis`, `segments`, and `readout` each get an `mlflow` entry when their notebook logs a run: `run_id`, `experiment_id`, `url`, and `logged_at`. A step without it ran while the server was down.

## Shared lists

`assumptions`, `open_questions` (each with an `id`, the `stage` that needs the answer, and `status`), `decisions` (each with `date`, `stage`, `decision`, and `reason`), `history` (every write), and `notebooks` (each stage notebook's path, decision hash, and last run). Skills append to these lists and never replace them.
