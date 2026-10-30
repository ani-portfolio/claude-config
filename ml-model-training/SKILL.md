---
name: ml-model-training
description: Build and tune the ML models of a project, step 10 of the ML project skills. Asks how many models to build, the type of each (logistic or ridge regression, decision tree, random forest, scikit-learn gradient boosting, LightGBM, XGBoost, CatBoost), and the questions that matter for each (tuning budget, class weights, fixed settings). Tunes each model with Optuna on the same CV folds and metrics as the baselines, logs every trial to a parquet run log, compares every model with the baselines, checks for leakage, overfitting, stability, and the V1 bar, and saves the champion (best mean CV score) refitted on the whole training set. Writes notebooks/08_training.ipynb; running the skill again adds models to the same notebook and only trains the new ones. Use after the baselines, or when asked to train, tune, compare, or add a model.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Model Training

Build every ML model of the project, from simple to complex, on the same CV folds and metrics as the baselines, and pick the champion. The test set is not used here.

The outputs are:

* `notebooks/08_training.ipynb`: the record. Decision cells hold the models and the training settings. Other cells train and tune the models, show a leaderboard with the baselines, the tuning history, the checks, the champion's feature importance and calibration.
* `training_runs.parquet` (one row per trial, only appended to) and `training_oof_predictions.parquet` (out of fold predictions of each model's best trial) in the processed data folder.
* `models/champion_<name>.joblib`: the champion pipeline, refitted on every training row.
* The `training` section of `ml_spec.yaml`.

This is step 10 of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, ml-train-test-split, ml-feature-engineering, ml-baseline-model, **ml-model-training**, then ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. `reference/models.md` explains every model type, the tuning, the run log, and the checks.

## Ground rules

1. **Same folds, same metrics.** Models use the folds from the split and the metrics and reference from the baseline section, so every number is comparable.
2. **No test set.** The test set is used once, in ml-offline-evaluation.
3. **Never retrain what is logged.** A tuning run already in the run log is read back. Only new or changed models are trained.
4. **Ask before installing.** LightGBM, XGBoost, CatBoost, and Optuna are installed with `uv add` only after the user agrees.
5. **Push back** on a thorough search when a quick one has already flattened, on class weights when the output is used as a probability, and on a champion that does not beat the best baseline.
6. **No raw data in the reply.** Use metrics and counts.
7. **Use the scripts.** `scripts/make_notebook.py` builds the notebook or adds models to it, and runs it. Do not write the `training` section by hand.
8. **Autopilot.** When the arguments include `autopilot` (ml-autopilot or an ml-experimenter agent is driving), or the project root has `.ml_workspace.json`, no person is present. Do not call AskUserQuestion. Take the recommended answer to every question, or the change the driver gave for this stage. Record each answer you took in `assumptions` with `source: autopilot draft`, and never set a framing to `status: confirmed`. Run the stage script with `--json`, and reply with its result and one line per assumption.
9. **Standard practice first.** When you recommend or offer options for ML experimentation design or ML system design, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
10. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status` and `gate`. If the gate exits with code 1, say which step to run first and stop, unless the user explicitly chooses to continue (record that in `decisions`). If the `baseline` section is empty, run **ml-baseline-model** first: the models need its metrics and reference.

Then check that the packages this stage needs are installed: `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py need scikit-learn`. New projects do not install scikit-learn up front. If the check exits with code 1, show the user the `uv add` command it printed, and run it only after they agree.

## Step 2: Get the facts

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --suggest
```

It prints the task, the training size, the positive share, the metric, the reference and the best baseline, the V1 bar, any interpretability constraint, which packages are installed, the time of one trial measured on a real fit, the models already in the notebook, models trained on older features or split (stale), and a recommended answers YAML.

## Step 3: Ask about the models

Ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next.

1. **How many models?** Question: "How many ML models do you want to build in this run?" Options: "1", "2 (Recommended)", "3", "4". In a later run, mention the models already in the notebook, which are kept.

**Model type, one question per model, one call each.** Question: "Which type is model N?" Options from the menu in `reference/models.md`, at most four per question, chosen for the task:

* Classification: logistic regression, LightGBM (or scikit-learn gradient boosting when LightGBM is not installed), XGBoost or CatBoost, random forest.
* Regression and forecasting: ridge regression, LightGBM, XGBoost or CatBoost, random forest.

Recommend the linear model for model 1 when the notebook has no linear model yet, and LightGBM for model 2. Say in the description when a package is not installed. The user can type another type (decision tree, hist_gbm). If a chosen package is missing, ask before running `uv add <package>`.

**The questions that matter for each model.** Ask only what applies, one question per call, going through one model at a time:

* **Tuning** (every model): "How much tuning for model N?" Options: "Quick: 20 trials, about X minutes (Recommended)", "Thorough: 60 trials, about Y minutes", "None: the default settings only". Use the time per trial from `--suggest`.
* **Class weights** (classification, positive share under about 10%): "Weight the rare class for model N?" Options: "No (Recommended)" when the metric ranks or the output is a probability; "Yes, balanced" when the output drives a yes or no decision at a threshold.
* **Fixed settings** (only when the user mentioned a constraint, such as a model that must stay simple): which setting to hold fixed, as `params`.
* **Inputs subset** (only when a model must use some of the features, such as a reference model without the media features): the columns, as `inputs`.

If the business framing asks for reasons per prediction, say that logistic regression explains each score and that feature importance in the notebook explains the model as a whole.

If AskUserQuestion is unavailable or the user is not present, build the recommended models with quick tuning, record them as assumptions, and say so at the top of the reply.

## Step 4: Build or add, and run

1. Write the answers to a YAML file in the scratch folder, starting from the `--suggest` output and the keys in `templates/training_answers.example.yaml`: `models` (each with a unique `name`, `type`, `params`, `class_weight`, `tuning`, and `reason`), `training`, `reasoning` (`models`, `tuning`), `risks`, `assumptions`, `open_questions`, `decisions`, and `resolved_questions`.
2. Run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   * **No notebook yet:** it writes `08_training.ipynb` and runs it.
   * **The notebook exists:** it **adds** the models to the notebook's `MODELS` cell, keeping everything else, and runs it. Models already logged are read back, so only the new ones train. A name that is already used is refused: choose another name, or pass `--replace-model` when the user wants to replace that model.
   * `--rebuild` writes the notebook from scratch with only the models in the answers file (the run log keeps every earlier trial). It asks for `--overwrite` or `--new-version` when the notebook was edited.

   It prints the leaderboard (each model's best trial and the baselines, with the improvement over the reference), the champion and where it was saved, whether the V1 bar is met, the top features, and problems.
3. If a cell fails, read the error in the notebook. The usual causes are a missing package, or a time limit too short for even one trial. Fix the answers and run again.
4. Read the problems and the tuning history. A curve still rising at the end of a quick search can justify a thorough one; a flat curve means more trials will not help.

## Step 5: Report back

Reply with these parts, in this order:

1. **Files written:** `notebooks/08_training.ipynb`, the run log, the out of fold predictions, the champion model file, and `ml_spec.yaml`.
2. **Summary:** the leaderboard as a small table (models and baselines, primary metric, spread, improvement over the reference, trials), then two or three sentences: the champion, how much it beats the reference and the best baseline, whether the V1 bar is met, and how close the simple model is.
3. **Problems:** from the checks, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** say that the user can add models later by running `/ml-model-training` again or by adding them to `MODELS` in the notebook, and that only new models are trained. When the champion is good enough, run **ml-offline-evaluation**, which scores the champion and the baselines on the test set once.

## The spec file

`reference/spec_fields.md` describes the `training` section. `scripts/ml_common.py`, `scripts/nb_common.py`, `scripts/split_core.py`, and `scripts/model_core.py` (model menu, search spaces, tuning, and the run log) are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
