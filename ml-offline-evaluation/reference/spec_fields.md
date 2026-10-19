# Spec Fields

`ml_spec.yaml` sits at the project root, and its presence is what marks the project root for every skill. Each skill reads the whole file first and writes only the sections it owns. People can edit it by hand at any time.

## Ownership

| Section | Written by | Read by |
|---|---|---|
| `project` | ml-project-setup (name, description, Python, kernel). `ml_common.py` updates `updated` on every write | all |
| `paths` | ml-project-setup | all |
| `business` | ml-business-framing | all |
| `discovery` | ml-data-discovery | ml-problem-framing, ml-eda-report, and later skills |
| `framing` | ml-problem-framing | all |
| `data` | ml-problem-framing (date, ID, and excluded columns), ml-data-discovery and ml-eda-report (`sources`) | all |
| `eda` | ml-eda-report | split and later skills |
| `split`, `features`, `baseline`, `training`, `evaluation` | the skill for that stage | later skills |
| `review` | ml-review | ml-status |
| `tracking` | ml-project-setup | every stage that logs to MLflow, ml-status, ml-review |
| `version` | ml-project-setup (V1), then ml-iterate | every stage that logs or commits, ml-status |
| `git` | ml-project-setup | `nb_common.cli`, ml-iterate |
| `autopilot` | ml-autopilot (`scripts/autopilot.py`) | ml-autopilot, ml-status |
| `notebooks` | `nb_common.py`, when a stage notebook runs | ml-status |
| `assumptions`, `open_questions`, `decisions` | any skill, append only | all |
| `history` | `ml_common.py` | people |

A skill may suggest changes to a section it does not own, but it asks the user first and records the change in `decisions`. The data dictionary lives in its own file, `docs/data_dictionary.yaml`, because it can be long. The `discovery` section points to it and summarises it.

Each workflow step writes one numbered notebook, and the notebook is the record of that step. Its markdown cells hold the reasoning, risks, assumptions, and open questions. Its code cells marked *Decision* hold the answers. Running the notebook writes those answers to the spec, so the spec always matches the notebook that last ran. To change a decision, edit the decision cell and run the notebook again.

## Workflow

| Step | Skill | Main output | Done when |
|---|---|---|---|
| 1. setup | ml-project-setup | Project folder, `.venv`, `00_scratch.ipynb` | `project.kernel` is set |
| 2. business (anchors only) | ml-business-framing | `01_business_framing.ipynb` with the mode and the two anchors, as a draft | `business.mode` is set and its two anchors are filled in (`business_goal` and `decision_supported`, or `learning_question` and `answer_actions`), or `business.status` is `confirmed` |
| 3. discovery | ml-data-discovery | `02_data_discovery.ipynb`, and `docs/data_dictionary.yaml` | `discovery` is filled in (or the ML framing was confirmed directly) |
| 4. business (full framing) | ml-business-framing | `01_business_framing.ipynb`, and the summary `docs/business_framing.md` | `business.status` is `confirmed` |
| 5. framing | ml-problem-framing | `03_problem_framing.ipynb` | `framing.status` is `confirmed` |
| 6. eda | ml-eda-report | `04_eda_<dataset>.ipynb` | `eda.mode` is `targeted` and the ML framing is confirmed |
| 7. split | ml-train-test-split | `05_split.ipynb`, and `train.parquet`, `test.parquet`, `cv_folds.parquet` (and `gap.parquet`) in `data/processed` | `split` is filled in |
| 8. features | ml-feature-engineering | `06_features.ipynb`, `src/features.py`, `src/custom_features.py`, and `features_train.parquet` and `features_test.parquet` in `data/processed` | `features` is filled in |
| 9. baseline | ml-baseline-model | `07_baseline.ipynb`, and `baseline_cv_results.parquet` and `baseline_oof_predictions.parquet` in `data/processed` | `baseline` is filled in |
| 10. training | ml-model-training | `08_training.ipynb`, `training_runs.parquet` and `training_oof_predictions.parquet` in `data/processed`, and `models/champion_<name>.joblib` | `training` is filled in |
| 11. evaluation | ml-offline-evaluation | `09_evaluation.ipynb` | `evaluation` is filled in |

Notebooks live in the `notebooks` folder. When a person has edited a notebook and a skill would replace it, the skill stops and offers to replace it or to write the next version, such as `03_problem_framing_v2.ipynb`.

`ml_common.py status` and the ml-status skill print this progress and names the next step. `ml_common.py gate` exits with code 1 while the ML framing (step 5) or the targeted EDA (step 6) is not done, or when the business framing concluded that ML is not needed, and the modelling skills (split onwards) run it first.

## Top level

| Field | Meaning |
|---|---|
| `spec_version` | Layout version. These skills expect 1. |
| `project.name` | Short project name, used in titles and the kernel name. |
| `project.description` | One sentence on what the project is for. ml-business-framing uses it as the first draft of the business goal. |
| `project.owner` | The person responsible for the model. |
| `project.python` | Python version of the project `.venv`. |
| `project.kernel`, `project.kernel_display` | The registered Jupyter kernel name and how editors show it. |
| `project.created`, `project.updated` | Timestamps, set automatically. |

## paths

Folders relative to the project root. Skills read these instead of assuming a layout, so an older project can point them elsewhere (for example `data_raw: data`).

| Field | Default | Holds |
|---|---|---|
| `data_raw` | `data/raw` | Input data. Read only. |
| `data_processed` | `data/processed` | Split sets and feature tables written by the skills. |
| `notebooks` | `notebooks` | Numbered stage notebooks. |
| `src` | `src` | Reusable code, importable as `src`. |
| `models` | `models` | Fitted pipelines and models. |
| `docs` | `docs` | The business framing summary, the data dictionary, and later the model card. |

## business

Written by ml-business-framing, in two steps: the mode and the two anchors before anyone looks at the data, then the rest of the framing, drafted from what data discovery found.

| Field | Meaning |
|---|---|
| `status` | `not_started`, `draft`, or `confirmed`. Only ml-business-framing sets `confirmed`, after the user agrees. |
| `mode` | `standard` (a clear business goal and KPI) or `exploratory` (an experiment). Empty counts as `standard`, so specs from before modes existed read as standard. |
| `learning_question` | Exploratory only. What the project should find out, for example "can weekly media spend explain bike sales well enough to compare channels?". One of the two exploratory anchors. |
| `answer_actions` | Exploratory only. What would be done with each possible answer, as a list of `answer` and `action`, for example "yes: build the budget model" and "no: improve how spend is tracked". The second exploratory anchor; it gives the project a stopping rule. |
| `business_goal` | The business outcome to improve. |
| `business_kpi` | The business measure the model should move. |
| `decision_supported` | The action the output drives and who takes it. |
| `users` | Who uses the output. |
| `capacity` | Any limit on how many actions can be taken, for example 500 calls a week. |
| `current_approach` | How the decision is made today. Used as the heuristic baseline. |
| `outcome` | What should be predicted, in business terms, for example "customers who stop ordering". ml-problem-framing turns it into an exact target. |
| `output_use` | `ranked_list`, `threshold_decision`, `probability`, `quantity`, or `other`. Decides the metric family. |
| `v1_bar_intent` | What would make V1 worth using, stated against the current approach. ml-problem-framing makes it a number. |
| `ml_needed` | `yes`, `no`, or `unsure`, with `ml_needed_reason`. When `no`, the modelling gate stays closed. |
| `constraints.*` | Latency, interpretability, fairness, data access, and other limits. |
| `out_of_scope` | Things this model will not do. |
| `risks` | Problems found when the framing was checked, each with `problem` and `resolution`. |

## discovery

Written by ml-data-discovery each time it runs. The whole section is replaced on each run.

| Field | Meaning |
|---|---|
| `run_at`, `notebook`, `dictionary` | When it ran, the notebook path (`notebooks/02_data_discovery.ipynb`), and the dictionary path (`docs/data_dictionary.yaml`). |
| `main_source` | The source the model will most likely be trained on. The framing hints describe this one. |
| `sources` | One entry per source: `name`, `path`, `rows`, `columns`, `duplicate_rows`, `grain`, `entity_id_candidates`, `date_columns`, `main_date`, `history_days`, `days_since_last_date`, `months_with_no_rows`, `thin_months`, and the main `issues` with `issue_counts`. |
| `joins` | For each pair of sources and shared key: `left`, `right`, `keys`, `relationship` (one to one, one to many, many to many), match % on each side, rows per matched left row, and `notes`. |
| `dictionary_summary` | Counts of columns and of columns that need review, and the lists `unclear_meaning`, `timing_unknown`, `possibly_after_outcome`, `candidate_targets`, and `personal_data` as `source.column`. |
| `framing_hints` | For the main source: `candidate_targets` (column, suggested type, unique values, missing %, the smaller class share for binary columns, and reasons), `grain`, `entity_id_candidates`, `date_columns`, `main_date`, `history_days`, and `post_outcome_suspects`. |

## Data dictionary file

`docs/data_dictionary.yaml` holds, for each source, its `path`, a `description`, and one entry per column:

| Field | Meaning |
|---|---|
| `column`, `type`, `role`, `unique`, `missing_pct` | From the data. |
| `meaning` | One sentence, drafted from the name and the data, or null when the name gives no clue. |
| `meaning_source` | `name` (drafted from the name), `unclear` (could not be worked out), `reviewed` (wording improved by Claude from context), or `user` (confirmed by a person). Entries marked `reviewed` or `user` are kept when ml-data-discovery runs again. |
| `timing` | When the column is filled in: `identifier`, `static`, `event_time`, `before_prediction`, `at_event`, `possibly_after_outcome`, or `unknown`. |
| `timing_basis` | Why that timing was chosen. |
| `flags` | `unclear_meaning`, `timing_unknown`, `possibly_after_outcome`, `candidate_target`, `personal_data`, `free_text`, `mostly_missing`, `constant`. |
| `needs_review` | true when the meaning or the timing could not be worked out, or the column may be recorded after the outcome. |
| `notes` | Free text, kept across runs. |

## framing

Written by ml-problem-framing. The ML definition of the problem, built on the business framing and the discovery findings.

| Field | Meaning |
|---|---|
| `status` | `not_started`, `draft`, or `confirmed`. Only ml-problem-framing sets `confirmed`, after the user agrees. |
| `task_type` | `binary_classification`, `multiclass_classification`, `regression`, `ranking`, `forecasting`, or `other`. |
| `target.name` | The target column, if it exists in the data. |
| `target.definition` | The exact rule for the target, in plain language. |
| `target.positive_class` | The positive class name and code, for classification. |
| `target.outcome_window` | The period after the prediction over which the outcome is observed. |
| `target.exists_in_data` | true, false, or null when unknown. |
| `target.construction` | How to build the target when it is not a column yet. |
| `target.label_delay` | How long after the prediction the label becomes known. |
| `unit_of_prediction` | What one row represents at prediction time. |
| `entity_id_cols` | Columns that identify the entity being scored, such as `customer_id`. Used for grouped splits. |
| `population` | Which entities are scored. |
| `prediction_time.description` | When the prediction is made. |
| `prediction_time.column` | The date or timestamp column that marks that moment. |
| `prediction_time.feature_cutoff` | The latest data features may use. |
| `prediction_time.cadence` | Real time, daily batch, weekly batch, and so on. |
| `prediction_time.horizon` | For forecasting, how far ahead is predicted. |
| `success.primary_metric` | The offline metric that decides whether V1 is good enough. |
| `success.secondary_metrics` | Other metrics to report. |
| `success.v1_bar` | The result that makes V1 worth using, as a number against a reference. |
| `success.guardrails` | Metrics that must not get worse. |
| `success.v1_criteria` | The V1 bar as checks a script applies, so no script reads `v1_bar` text. Each has an `id`, a `kind`, and `on` (`cv`, judged in model training, or `test`, judged in offline evaluation): `beats_model` (`model`, optional `margin`), `pct_over_reference` (`pct`), `threshold` (`metric`, `op`, `value`), `check` (`function` in the project's `src/v1_checks.py`, optional `params`), or `manual` (`text`, judged by a person). `ml_common.v1_evaluate` applies them. |
| `risks` | Problems found when the framing was checked against the data, each with `problem` and `resolution`. |

Specs written before the business section existed kept `business_goal`, `decision_supported`, `users`, `current_approach`, `constraints`, `out_of_scope`, and `success.business_kpi` in `framing`. `ml_common.py` moves them to `business` when it reads the spec.

## data

| Field | Meaning |
|---|---|
| `sources` | One entry per dataset: `name`, `path`, `description`, `rows`, `columns`. ml-data-discovery and ml-eda-report add or update the entry for each file they profile. |
| `date_col` | The main date column. |
| `id_cols` | Identifier columns that are not entity IDs, such as transaction IDs. |
| `excluded_cols` | Columns that must never be used as features, each with `column` and `reason`. |

## eda

Written by ml-eda-report each time it runs. The whole section is replaced on each run.

| Field | Meaning |
|---|---|
| `mode` | `targeted` (the confirmed target) in the workflow. `discovery` appears only when ml-eda-report was run without a target on its own. |
| `run_at`, `source`, `report` | When it ran, the data file, and the notebook path (or a markdown report when one was asked for). |
| `rows`, `columns`, `sampled` | Size of the data, and whether plots and correlations used a sample. |
| `grain` | The detected unique key: `type` (`single`, `pair`, or `unclear`) and `columns`. |
| `duplicate_rows` | Number of exact duplicate rows. |
| `primary_date` | The main date column with its first and last date. |
| `columns_by_role` | Column names grouped by the role EDA inferred: numeric, categorical, boolean, datetime, identifier, free text, personal data. |
| `target` | Column, type, missing %, unique values, class balance (up to 10 classes), imbalance ratio, and skew. |
| `top_target_associations` | The ten features most associated with the target, with the measure used. |
| `leakage_suspects` | Columns flagged as possible leakage, with reasons. |
| `suggested_exclusions` | Columns EDA suggests keeping out of the features (identifiers, free text, personal data, strong leakage suspects), with reasons. Not applied automatically. |
| `issues` | Data quality issues grouped by type: severity, area, finding, columns, and suggested action. |
| `framing_checks` | Targeted mode only. Checks of the data against the framing, each with `check`, `status` (`pass`, `warn`, `fail`, or `skipped`), and `detail`. |
| `framing_hints` | The same hints ml-data-discovery writes, for the profiled file: `candidate_targets` (column, suggested type, unique values, missing %, the smaller class share for binary columns, and reasons), `grain`, `entity_id_candidates` (column, distinct values, rows per value), `date_columns` (column, first and last date, span in days, missing %), `history_days` of the main date column, and `post_outcome_suspects` (columns whose names suggest they are recorded after the outcome). |

## notebooks

Written by `nb_common.py` each time a stage notebook runs to the end. One entry per stage, such as `business` or `discovery`.

| Field | Meaning |
|---|---|
| `path` | The notebook that ran, relative to the project root. |
| `decisions_hash` | A hash of its decision cells when it ran. ml-status compares it with the notebook on disk to tell when a notebook was edited and not run again. |
| `run_at` | When it ran. |

## split

Written by `05_split.ipynb` each time it runs. The whole section is replaced on each run.

| Field | Meaning |
|---|---|
| `strategy` | `random`, `group`, or `time`. |
| `data_source`, `target`, `task_type` | The file that was split, the target column, and the task. |
| `row_id_col` | The column added to every set with the row's position in the original file, normally `row_id`. |
| `date_col`, `group_col` | The column that orders a time split, and the entity column (kept together in a group split, reported in a time split). |
| `stratify`, `seed` | Whether the sets keep the same class balance, and the random seed. |
| `test` | `rows` and `share` of the test set, the `size` asked for (random and group), or `days`, `start`, and `end` (time). |
| `gap_days` | Time split: the days left out between training and the test set, and inside each fold. |
| `cv` | `method` (`stratified_kfold`, `kfold`, `stratified_group_kfold`, `group_kfold`, or `time_backtest`), `n_folds`, for time also `window`, `valid_days`, and `train_days`, and `folds`: one entry per fold with `n_train`, `n_valid`, the class shares of the validation rows, and for time the `train_start`, `train_end`, `valid_start`, and `valid_end` dates. |
| `duplicates` | `handling` (`keep_together`, `drop`, or `ignore`), the number of duplicate `rows`, and how many were `dropped`. |
| `dropped` | Rows dropped for a missing target or a date that could not be read. |
| `rows` | Rows in `train`, `test`, and `gap`. |
| `files` | Paths of `train`, `test`, `folds`, and `gap` (time only), all parquet. |
| `checks` | `problems` the leakage checks found, and `test_rows_with_copy_in_train`. |
| `run_at` | When it ran. |

Later stages load the files with `split_core.load_sets(spec, root)` and get the folds with `split_core.cv_splits(train, spec["split"])`.

## features

Written by `06_features.ipynb` each time it runs. The whole section is replaced on each run, except `code`, which the skill sets when it installs the project code.

| Field | Meaning |
|---|---|
| `config` | The settings `src/features.py` reads (`load_config`): `numeric`, `categorical`, `high_cardinality`, `date_parts`, `days_since`, `log`, `ratios`, `lags`, `rolling`, `custom`, `date_col`, `target`, `min_shift_days`, `target_min_shift_days`, `rare_min_frequency`, and `exclude` (including features left out by the leakage check). |
| `inputs` | One entry per model input: `name`, `kind` (numeric, categorical, or high cardinality), `step` (column, date_part, days_since, log, ratio, lag, or rolling), `source` column, and `reason`. |
| `excluded` | Every column that is not an input, with `column` and `reason`. |
| `n_inputs`, `outputs` | The number of inputs, and the number of columns each preprocessor (`linear`, `tree`) produces. |
| `files` | `train` and `test` feature tables, parquet: keys, target, and inputs before the learned steps. |
| `code` | `features` and `custom` paths, and `features_hash`, the hash of `src/features.py` as the skill wrote it. |
| `checks` | `leakage_threshold`, the `strongest` single features with their scores, `flagged` (left out), `confirmed_safe`, `drift_large`, and `problems`. |
| `run_at` | When it ran. |

Later stages use `from src.features import load_config, feature_columns, make_preprocessor`, read the feature tables, and fit `make_preprocessor(kind, config)` inside each fold from `split_core.cv_splits`.

## baseline

Written by `07_baseline.ipynb` each time it runs. The whole section is replaced on each run.

| Field | Meaning |
|---|---|
| `metrics` | `primary` (a key such as `recall_at_k`, with `k` or `k_share` and `k_per` for the at k metrics) and `secondary` keys. Model training and evaluation use the same metrics. |
| `primary_label` | The primary metric as it appears in the results, for example `recall at 50 per scoring_date`. |
| `models` | Every baseline, all non ML: `name`, `type` (`prior`, `rule`, `column`, or `segment`), `params`, and `reason`. Offline evaluation refits them on the whole training set to score the test set. |
| `results` | Per baseline: the mean of every metric over the CV folds, and the spread (standard deviation) of the primary metric. |
| `reference` | The baseline models are measured against (normally today's rule), with its primary metric. |
| `best_non_ml` | The best baseline, with its primary metric. |
| `best_vs_reference_pct` | How much better the best baseline is than the reference, in %, when they differ. Positive is better, also for error metrics. |
| `files` | `cv_results` and `oof_predictions`, parquet. |
| `checks` | `problems` found. |
| `run_at` | When it ran. |

## training

Written by `08_training.ipynb` each time it runs. The whole section is replaced on each run; the run log keeps every trial.

| Field | Meaning |
|---|---|
| `models` | The models in the notebook: `name`, `type` (`logistic`, `ridge`, `tree`, `random_forest`, `hist_gbm`, `lightgbm`, `xgboost`, `catboost`), `params` (fixed), `class_weight`, `tuning` (`n_trials`, `timeout_minutes`), and `reason`. |
| `settings` | `seed` and `selection` (`best_mean`). |
| `data_version` | A hash of the feature settings and the split. Runs with another version are stale and left out of the leaderboard. |
| `primary_label` | The primary metric, the same as in the baseline section. |
| `leaderboard` | The best trial of each model and every baseline: `model`, `kind` (`ml` or `baseline`), `type`, `primary_mean`, `primary_std`, `trials`, `minutes`, and `vs reference %` (positive is better). |
| `champion` | `name`, `type`, `params` (the tuned settings), `class_weight`, `run_id` in the run log, the primary metric and `std`, `vs_reference_pct`, `beats_best_baseline`, `v1` (the CV criteria: `status` met, not_met, incomplete, needs_human, or not_set, and each criterion's `status` and `detail`), `v1_bar_met` (true when `v1.status` is met, false when not_met, otherwise null), `path` of the saved pipeline, and `serialization` (`cloudpickle_by_value`: it loads with `joblib.load` without the skills). |
| `top_features` | The ten most important inputs of the champion by permutation importance. |
| `files` | `runs`, `oof_predictions` (parquet), and `champion_model` (joblib). |
| `checks` | `problems`, and the champion's `train_score` and `cv_score`. |
| `run_at` | When it ran. |

## evaluation

Written by `09_evaluation.ipynb` each time it runs. The whole section is replaced on each run, except `test_opened`, which only grows.

| Field | Meaning |
|---|---|
| `champion` | The champion from the training section. Evaluation never changes it. |
| `primary_label` | The primary metric, the same as in the baseline and training sections. |
| `metrics` | One entry per scored model: `model`, `kind` (`champion`, `ml`, or `baseline`), `test` (the primary metric on the test set), `low` and `high` (the 95% bootstrap interval), `cv` (the mean CV score), and every secondary metric. |
| `comparisons` | Paired bootstrap differences of the champion against the reference and the best baseline: `comparison`, `difference`, `low`, `high`, and `relative %`. Positive is better, also for error metrics. |
| `operating_point` | For classification, the rows the champion acts on: the top k of the primary metric, or the threshold with the best F1 on the out of fold predictions. |
| `slices` | One entry per slice value: `slice`, `value`, `rows`, the positive share or mean target, the champion and reference scores, the difference with its interval, and the flags `too small`, `worse than reference`, and `far below overall`. |
| `slice_settings` | The slice columns used: `categorical`, `numeric_bins`, `time`, `new_entity`, `missing`, `protected`, `min_rows`, and `min_positives`. |
| `bootstrap` and `cluster` | The number of draws, the seed, and the column whose values are drawn together (single rows when empty). |
| `score_psi` | Drift of the champion's scores from the out of fold scores to the test scores. Under 0.1 is stable. |
| `target_psi` | Regression and forecasting only: drift of the actual values over the same rows, to tell a data shift from a model problem. |
| `calibration` | For classification: `mean_predicted`, `observed_rate`, and `largest_decile_gap`. |
| `evidence` | The questions a go or no go decision needs, each with a one sentence `answer`. |
| `v1` | The V1 bar: `cv` (from training), `test` (the test criteria judged here), and the overall `status`. |
| `checks` | `self_check`: one row per model (`model`, `check`, `recorded`, `recomputed`, `rel_diff`, `ok`), confirming before the test set was opened that each model's CV score reproduces through the scoring path used here; `off` when it was turned off. |
| `files` | `test_predictions`, `metrics`, and `slices` (parquet), and `model_card` (markdown). |
| `test_opened` | Every time the test set was scored: `at`, `champion`, and `data_version`. A later run with another champion or data version warns that the test score may be optimistic. |
| `decision` | Always null when written by the notebook. The decision (go, iterate, or no go) is the user's and is recorded in `decisions`. |
| `run_at` | When it ran. |

## review

Written by `10_review.ipynb` (the ml-review helper skill) each time it runs. The whole section is replaced on each run; `review_findings.parquet` keeps every finding of every review.

| Field | Meaning |
|---|---|
| `stages` | The stages reviewed in this run. |
| `open` | Open findings by severity: `blocker`, `warning`, and `note`. Findings carried over from stages not reviewed this time are included. |
| `accepted` | The number of findings accepted with a reason. |
| `fixed_since_last` | Findings of the previous review that are no longer found (scripted) or not raised again (reviewer). |
| `open_blockers` | Each open blocker: `id`, `title`, and `fix`. |
| `reviewer` | The independent reviewer's number of `findings` and the `files_read`. |
| `files` | `findings`, parquet. Columns: `id`, `stage`, `category`, `severity`, `title`, `evidence`, `fix`, `source` (`script` or `reviewer`), `status` (`open`, `accepted`, `fixed`, or `not repeated`), `accepted_reason`, `first_seen`, and `last_seen`. |
| `run_at` | When it ran. |

## iterations

Written only by the ml-iterate helper skill (`scripts/iterate.py`). Older specs get the empty section when they are next loaded.

| Field | Meaning |
|---|---|
| `current` | The number of the open iteration, or null when none is open. The original work is iteration 0. |
| `entries` | One entry per iteration, oldest first. |

Each entry:

| Field | Meaning |
|---|---|
| `id` | The iteration number, from 1. |
| `opened_at`, `closed_at` | When it was opened and closed. |
| `hypothesis` | One sentence: what is expected to improve, and why. |
| `diagnosis` | Short findings from the evaluation that led to it. |
| `changes` | Each planned change: `stage` and `change`. |
| `rerun` | The stages to run again, in order: the earliest changed stage, then every later modelling stage. |
| `start` | The state it started from: `champion`, `cv`, `test`, `test_look` (how many times the test set had been opened), and `data_version`. |
| `snapshot` | The folder with a copy of that state: `iterations/iter_<id - 1>/`. It holds the spec, notebooks, `src`, models, docs, and the small result files, plus `manifest.json` listing what was copied and the sizes of the large data files that were not. |
| `carried_forward` | The previous champion as it was added to the new training run: `name`, `type`, `params`, `inputs`, and whether its inputs still exist. |
| `status` | `open` or `closed`. |
| `outcome`, `reason` | `kept` (the new state stays), `rolled_back` (the snapshot was restored), or `stopped` (no further change is worth making), and why. |
| `version`, `previous_version` | The version this iteration builds (iteration N builds V(N+1)) and the version it started from. |
| `mlflow_parent_run` | The MLflow run of the version (`run_type = experiment`). Every stage run of the iteration is logged as its child. |
| `end` | The state it ended with: `champion`, `cv`, `test`, `test_look`, `data_version`, and `carried_forward_cv` (the carried champion's CV score on the new folds, when it was trained). |
| `experiments` | A round driven by ml-autopilot only: one record per experiment, see below. |
| `adopted` | A round only: the experiment adopted into the project (`V2-a`), or absent when none passed. |
| `bar_result` | The iteration bar applied when it closed (`ml_common.iteration_bar_evaluate`): `status` (`pass`, `fail`, or `incomplete`), `wins`, `folds`, `mean_diff`, `spread`, and `detail`. |

Opening and closing an iteration also add an entry to `decisions` with stage `iteration`.

`bar` (at the top of the section) is the iteration bar: when a new champion really beats the champion carried forward, judged on the same CV folds. It is written from `ml_common.ITERATION_BAR_DEFAULTS` when the first iteration opens, and can be edited.

| Field | Default | Meaning |
|---|---|---|
| `min_fold_win_share` | 0.5 | The new champion must win on more than this share of the paired folds. |
| `spread_multiple` | 1.0 | Its mean improvement must be larger than this many standard deviations of the per fold differences. |
| `min_folds` | 3 | With fewer paired folds, the bar is `incomplete`, never `fail`. |
| `max_iterations` | 3 | Iterating without a person stops after this many rounds. Enforced by ml-autopilot, not by ml-iterate. |

Each record in a round's `experiments`:

| Field | Meaning |
|---|---|
| `id` | A short id within the round: `a`, `b`, ... |
| `experiment` | Its name in MLflow and git: `<version>-<id>` (`V2-a`). Its branch is `exp/v2-a` and its worktree `.worktrees/v2-a`. |
| `hypothesis`, `changes`, `rerun` | Its changes (one in the screen phase), why, and the stages it reruns. The rerun never includes evaluation: an experiment is judged on CV. |
| `backlog` | The id of the `autopilot.backlog` item it came from, or a list of ids when it combines several. |
| `phase` | The phase of the round it belongs to: `screen`, `combine`, `ablate`, `verify`, or `greedy`. |
| `compare_to` | The experiment it is judged against (an ablation against the combination, a greedy step against the set before it). Absent: the champion carried forward. |
| `frozen_model` | An ablation only: the combination's model with its tuned settings, trained without tuning. |
| `refs` | The branches of the screen experiments whose changes it makes. |
| `candidate` | The model it is judged by: the new champion, or the frozen model of an ablation. |
| `status` | `planned`, `running`, then `passed`, `failed` (the bar failed or was incomplete), or `error` (the agent stopped), and at close `adopted` or `discarded`. |
| `bar`, `champion`, `cv`, `v1_cv`, `problems`, `summary` | What the experiment reported in its `experiment_result.json`. `bar.per_fold` has the per fold CV scores of every model it trained, and `bar.champion_model` its champion with the tuned settings. |
| `workspace`, `branch`, `mlflow_parent_run` | Where it ran, and its MLflow parent run (a child of the version's run). |

## autopilot

Written only by ml-autopilot. Absent in projects no agent has driven.

| Field | Meaning |
|---|---|
| `settings` | `max_parallel` (3, at most 5: experiments that run at once) and `unattended`, agreed with the person at the start of the run (`ml_common.AUTOPILOT_DEFAULTS`). Older specs with `experiments_per_round` and `max_rounds` are brought up to date when read. |
| `started_at` | When the run started. |
| `checkpoints` | Each stop for the person: `kind` (`business`, `framing`, `test_v1`, `test_final`), `status` (`open` or `cleared`), `opened_at`, `cleared_at`, `file` (`docs/checkpoint.md`), and a `note`. |
| `backlog` | Candidate experiments, one change each: `id` (`B1`), `stage`, `change`, `hypothesis`, `cost` (1 to 3), `status` (`open`, `in_round`, `tried`, `adopted`), `round`, and `result` (the screen experiment, its bar result, and a `note` on why it was adopted or not). |
| `rounds` | One record per round: `iteration`, `version`, `phase`, `status`, `passers` (screen experiments that passed, best first), `combo`, `ablations` and `ablation_verdicts`, `dropped`, `verify`, `greedy` (`order`, `set`, `current`), `discarded`, `decided`, `final` (the experiment adopted, or none), and `log` (one sentence per step). |
| `stopped` | `at` and `reason` when iterating was stopped early. |

## The workspace file

An experiment's worktree has `.ml_workspace.json` at its root (ignored by git): `experiment`, `id`, `iteration`, `version`, `parent_run`, `version_parent_run`, `main_root`, `workspace`, `branch`, `hypothesis`, `changes`, and `rerun`. `tracking.candidate` reads it, so every stage run in the worktree is logged as a candidate of the experiment without environment variables. `ml_common.autopilot_mode` treats a project with the file as driven by an agent.

## version

Written by ml-project-setup (V1) and by ml-iterate when an iteration opens or closes.

| Field | Meaning |
|---|---|
| `current` | The version being worked on: `V1` for the first pass, then `V2`, `V3`, and so on. Stage runs and commits are tagged with it. |
| `branch` | Its git branch, `ml/v1`, `ml/v2`, and so on. |
| `previous` | The version the current one was built from. |
| `mlflow_parent_run` | The MLflow run of the current version, from V2 on. |
| `history` | One entry per earlier or closed version: `version`, `branch`, `status` (`kept`, `kept (stopped iterating)`, or `rolled_back`), `superseded_by` (the version built on it, when there is one), `champion`, `cv`, `test`, `reason`, `mlflow_parent_run`, and `git_tag`. |

## git

Written by ml-project-setup.

| Field | Meaning |
|---|---|
| `auto_commit` | True by default: every successful stage build commits the stage's files on the version branch, and ml-iterate commits when it opens or closes a version. Set it to false to commit by hand. |

## tracking

Written by ml-project-setup (`tracking.py setup`), or by the first stage that logs a run.

| Field | Meaning |
|---|---|
| `uri` | The MLflow tracking address, normally the shared server `http://127.0.0.1:5000`. `MLFLOW_TRACKING_URI` overrides it for a shell. |
| `experiment`, `experiment_id` | The project's experiment, `ml/<project>`. |
| `ui` | The link to the experiment in the MLflow UI. |
| `registered_model` | The registry name of the champion, `<project>-champion`. The current champion has the alias `champion`, and each version's champion keeps the alias `champion-v1`, `champion-v2`, and so on (MLflow reserves `v1` for itself). |

## mlflow, in each stage section

`split`, `features`, `baseline`, `training`, `evaluation`, and `review` each get an `mlflow` entry when their notebook logs a run: `run_id`, `experiment_id`, `url`, and `logged_at`. A stage without it ran while the server was down.

## The answers folder

`answers/<NN>_<stage>.yaml` at the project root (not in the spec) keeps the settings each stage notebook was last built from: `skill`, `stage`, `saved_at`, the `command` line, and the `answers` file contents. Written by `nb_common.cli` after a successful build, so a stage can be rebuilt without asking again. The notebook's decision cells stay the record of what was decided.

## Shared lists

| Field | Entry fields |
|---|---|
| `assumptions` | A sentence each. |
| `open_questions` | `id` (Q1, Q2, ...), `question`, `stage` that needs the answer, `raised_by`, `status` (`open` or `resolved`), and `answer` once resolved. |
| `decisions` | `date`, `stage`, `decision`, `reason`. A decision that no longer holds keeps its entry and gets `superseded_by` and `superseded_on`; the decision that replaces it has `replaces`. Written by ml-reconcile-notes. |
| `retired_assumptions` | Assumptions that were confirmed or no longer hold: `assumption`, `reason`, `retired_on`. Written by ml-reconcile-notes. |
| `history` | `at`, `skill`, `action`. Written automatically. |
