# {System Name}: ML System Summary

**Repository:** {repo name or path}
**Commit:** {short commit hash, or "not a git repository"}
**Generated on:** {YYYY-MM-DD}

This summary treats the code as the source of truth. Where the documentation disagrees with the code, this summary describes what the code does, and section 14 lists the difference.

Status labels used in this document:

* **Confirmed** means shown directly in code that runs, or in config the code loads.
* **Inferred** means suggested by code, such as names or a partial implementation, but not fully shown.
* **Documented only** means described in documentation, with no supporting code found.
* **Not found** means there is no evidence in code or documentation.

**Discrepancies between documentation and code:** {N} found ({n} High, {n} Medium, {n} Low). See section 14.

## 1. System Overview and Problem Formulation

**Summary:** {One paragraph in plain language describing what the system does and why it exists.}

| Field | Value (from code) | Status | Code evidence |
|---|---|---|---|
| Unit of prediction | {entity and grain, such as one customer at signup} | | |
| Prediction horizon | {how far ahead the model predicts} | | |
| Inputs | {main inputs at prediction time} | | |
| Label | {target and how it is built, including the window} | | |
| Task type | {regression, classification, ranking, forecasting} | | |
| Offline metrics | {metrics used before deployment} | | |
| Serving mode | {online, micro batch with frequency, or batch} | | |
| Downstream decision | {the action the prediction drives} | | |
| Business KPI | {the business outcome to improve} | | |
| Guardrail metrics | {metrics that must not get worse} | | |
| Online experiment | {how it is tested in production} | | |
| Entry points | {main files for training, inference, and pipelines} | | |

![System diagram](system_diagram.png)

The editable diagram is in [system_diagram.excalidraw](system_diagram.excalidraw). The PNG also contains the scene, so it can be dragged into Excalidraw and edited.

### Business Goal

{The goal in one or two sentences.}

### Assumptions and Notes

{Assumptions stated in code, config, or docs, each with evidence.}

## 2. Raw Data Sources

**Status:** {Confirmed, Inferred, Documented only, or Not found}

{For each source: name (table, file, topic, or API), grain, key columns, refresh frequency, and how it is loaded. Cite the loading code.}

## 3. Data Quality Checks

**Status:**

{Schema, null, range, freshness, and drift checks that run before training or inference. Name the library and what happens when a check fails.}

## 4. Feature Pipeline

**Status:**

{How raw data becomes features: point in time correctness, joins, aggregation windows, encoders, scalers, and where the code lives.}

### Engineered Features

| Feature | Description | Computed (precomputed or on demand) | Evidence |
|---|---|---|---|

### Feature Store

**Status:**

{Offline store, online store, how the online store is populated, versioning, and the entity keys. Write "Not found in this codebase." if there is no feature store.}

## 5. Training Pipeline and Model Details

**Status:**

{Model type and library, hyperparameters and how they are tuned, label construction, split strategy, cross validation, class or sample weighting, and the training data snapshot. Flag any leakage risk.}

## 6. Evaluation and Validation

**Status:**

{Offline metrics and how they are computed, backtesting, comparison to a baseline, and the checks a model must pass before promotion.}

## 7. Model Registry

**Status:**

{Registry tool, what is logged (version, parameters, metrics, data snapshot, tags), and how a model is promoted.}

## 8. Orchestration and Scheduling

**Status:**

{Orchestrator, the tasks and their order, schedules, and the environments they run in.}

## 9. Deployment Strategy

**Status:**

{Packaging, serving infrastructure, and rollout method such as shadow, canary, or blue green, with traffic percentages and promotion gates.}

## 10. Experimentation

**Status:**

{A/B tests or holdouts: the randomization unit, treatment and control definitions, split sizes, primary and guardrail metrics, and exploration policies.}

## 11. Inference Pipeline

**Status:**

{Request flow from input to prediction: feature lookup, on demand feature computation, model loading, batching, latency targets, and prediction logging.}

### Training and Serving Skew

{Whether training and inference share one feature implementation. List any differences found.}

## 12. Outputs, Consumers, and Fallback Logic

**Status:**

{What is produced, its format, where it is written, which downstream systems consume it, guardrails such as caps or clipping, and the fallback used when inference fails.}

## 13. Monitoring and Retraining

**Status:**

{Data drift, concept drift, prediction drift, data quality, and system health monitoring, with thresholds and alert destinations. Retraining triggers: schedule, drift based, or manual. Feedback loops where predictions change future training data.}

## 14. Documentation and Code Discrepancies

The code is treated as correct in every row. The "Code does" column is what this summary reports.

| # | Topic | Type | Documentation says | Code does | Documentation source | Code evidence | Impact |
|---|---|---|---|---|---|---|---|
| 1 | {e.g. Label window} | {Conflicts, Not implemented, or Undocumented} | {one sentence, or "Not mentioned"} | {one sentence} | {file and heading or line} | {file (function) or config key} | {High, Medium, or Low} |

{If no discrepancies were found, write "No discrepancies were found between the documentation and the code." and name the documentation files that were checked. If the repository has no documentation, say so.}

**Notes on High impact rows:** {For each High impact row, two or three sentences on why it matters, such as how it changes the model's behaviour, outputs, or risks.}

## 15. Gaps and Open Questions

{Numbered list of everything that could not be confirmed from code, items marked Documented only, and risks worth raising with the owning team.}

## Appendix: Files Reviewed

**Code:** {the key code and config files read, grouped by section}

**Documentation checked against code:** {the README, docs, and design files that were compared with the code}
