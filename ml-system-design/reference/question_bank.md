# Question Bank

One entry per question of the interview, in order. Each says what to draft from (the fields in `design.py --context`), the question, what a good answer looks like, the red flags that call for one follow up, and whether the part depends on the champion. Draft every answer before asking. Ask one question per message.

## Platform (`platform`)

**Draft from:** the user's earlier words, `business.constraints`, and the stacks in `defaults.yaml`.
**Ask (AskUserQuestion):** "Should the design name specific tools, and if so from which platform?" Options: the stack the context points to (Recommended), the default stack, tool neutral, and Other for a platform not in the file. Each option's description says which components it covers.
**Good answer:** one stack, with any per component overrides named.
**Red flags:** a platform chosen for one feature while most components fall outside it; tool neutral when the build skills will need tools soon.

## Registry decision (`registry.settings.decides_champion`)

Only when the chosen stack has `has_registry: true`. Ask every time, never assume.
**Ask (AskUserQuestion):** "The platform has its own model registry. Which registry decides the champion?" Options: "MLflow for experiments, the platform registry for deployment, linked by the MLflow run id" (Recommended), "MLflow decides, the platform only serves", "The platform registry for both".
**Why it matters:** a gate split across two registries needs a sync that drifts.

## Components

For each component, send one message: the draft (tool from the stack, settings from the context), one sentence on what supports it, and the question "Keep this, change it, or mark it not needed?" A component marked not needed needs a reason.

### 1. Ingestion (`ingestion`)
**Draft from:** `data_sources`, `framing.prediction_time`, `label_delay`. Settings: `frequency` (hourly, daily, weekly, monthly), `source`, `limits` (rate limits, quotas, cost).
**Red flags:** no limits recorded for a rate limited API; a frequency faster than the source updates.

### 2. Data validation (`data_validation`)
**Draft from:** the data dictionary's ranges and the EDA's quality findings. Settings: `checks`, `on_failure` (reject, quarantine, alert).
**Red flags:** checks only after features are written; no freshness check.

### 3. Features and feature store (`features`) (champion dependent)
**Draft from:** `features.inputs`, `features.code`, `top_features`. Settings: `served_inputs` (every name in `features.inputs`; design.py blocks on a missing one), `offline_store`, `online_store`, `backfill`.
**Follow up:** for any lag or rolling input, ask how it is computed at request time (from the online store, or recomputed from raw).
**Red flags:** training features computed by a notebook and serving features by different code.

### 4. Training pipeline (`training`)
**Draft from:** the champion's type and settings, the split's window. Settings: `trigger` (a schedule, drift, or by hand), `window`.
**Red flags:** retraining on a schedule with no gate after it.

### 5. Registry and gate (`registry`) (champion dependent)
**Draft from:** `framing.primary_metric`, `champion.std`, `evaluation.comparisons`. Settings: `gate` with `metric` (must be the framing's primary metric), exactly one of `margin` (in the metric's units, at least the champion's CV std) or `test` (a paired significance test on the daily metric difference: `name` `diebold_mariano` or `block_bootstrap`, `alpha`, `sided`), `window_days` (the live window both models are compared on, optional with a test), `promotion` (manual or auto); `lineage`.
**Follow ups (one at a time):** a fixed margin or a significance test; then the margin, with the CV std shown, or the test and its alpha; manual or automatic promotion.
**Why a test can suit better:** the CV std includes how much harder some seasons are than others, which cancels out when both models are scored on the same days, so a margin of at least the CV std can be far larger than the noise of a paired comparison.
**Red flags:** a margin below the CV std (blocked); both a margin and a test (blocked); automatic promotion with no monitoring (warned).

### 6. Inference (`inference`)
**Draft from:** `framing.prediction_time.cadence`, `business.output_use`, the constraints. Settings: `mode` (batch or online), `schedule`, and `logging` (what each prediction log row holds: model version, features, time).
**Follow up:** batch or online, with the cadence shown. Online for a daily cadence is warned; a learning goal is a valid reason.
**Red flags:** predictions not logged, so label feedback has nothing to join.

### 7. Serving and deployment (`serving`)
**Draft from:** the inference mode and the stack. Settings: `environments`, `secrets`, `rollback`, the request and response `contract`.
**Red flags:** secrets in code; no way back to the previous model.

### 8. Label feedback (`label_feedback`)
**Draft from:** `label_delay`, `framing.target`. Settings: `label_arrives`, `join` (the keys shared by predictions and actuals).
**Red flags:** labels that never arrive, so the gate and monitoring cannot measure live performance (warned).

### 9. Monitoring (`monitoring`) (champion dependent)
**Draft from:** `evaluation.score_psi`, `target_psi`, the champion's test score, the slices. Settings: `data_drift`, `prediction_drift`, `performance`, `system_health` (latency, errors, failed jobs), `alerts` (who and how).
**Red flags:** thresholds with no starting point in the evaluation; alerts nobody reads.

### 10. Failure handling (`failure_handling`) (champion dependent)
**Draft from:** `evaluation.best_non_ml` (the natural fallback), the constraints. Settings: `fallback`, `target` (a reliability target, such as a forecast for every entity by a time).
**Follow up:** what is served when features are stale, the model fails, or ingestion is rate limited.
**Red flags:** no fallback (warned).

### 11. Orchestration (`orchestration`)
**Draft from:** the schedules above. Settings: `order`, `retries`.
**Red flags:** steps that depend on each other scheduled independently.

### 12. Testing (`testing`)
**Draft from:** the components above. Settings: `unit`, `component`, `system`, each naming what it guards.
**Good answer:** the feature skew between training and serving is covered by a test.
**Red flags:** no tests (warned).

## On a rerun

Read `out_of_date` in the context. Ask only about the champion dependent parts it lists and anything the person wants to change; keep every other answer from `answers/11_system.yaml`.
