---
name: ml-baseline-model
description: Build the non ML baselines every model must beat, step 8 of the ML project skills. Offers the options for the task (the current business rule, the base rate or mean, the best single feature, a segment average, or naive, seasonal naive, and moving average forecasts) and scores them on the same CV folds the models will use, against the primary metric from the framing (including recall or precision at k per scoring date). Builds no ML models; those, simple and complex, belong to ml-model-training. Writes notebooks/07_baseline.ipynb; running it saves results and out of fold predictions as parquet and the baseline section of ml_spec.yaml with the reference to beat. Use after feature engineering, or when asked for a baseline, a benchmark, or how well today's rule performs.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Baseline Models

Score the non ML references before any model is built, so every model result has something to be compared with. Baselines show what is possible without a model, such as today's business rule. This skill builds no ML models: every ML model, from logistic regression to tuned gradient boosting, is built in **ml-model-training**, which compares them with the baselines from here. Every baseline is scored on the same CV folds that model training will use. The test set is not used.

The outputs are:

* `notebooks/07_baseline.ipynb`: the record. Decision cells hold the metrics, the baselines, and the reference. Other cells score every baseline per fold, chart the results, and check them.
* `baseline_cv_results.parquet` (metrics per baseline and fold) and `baseline_oof_predictions.parquet` (out of fold predictions) in the processed data folder.
* The `baseline` section of `ml_spec.yaml`: the metrics, the baseline definitions, the results, the reference, and the best baseline.

This is step 8 of the ML project skills: ml-project-setup, ml-business-framing, ml-data-discovery, ml-problem-framing, ml-eda-report, ml-train-test-split, ml-feature-engineering, **ml-baseline-model**, then ml-model-training and ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. `reference/baselines.md` explains every baseline, the reference, and every metric.

## Ground rules

1. **Never invent answers.** Today's rule comes from `business.current_approach`, written as an expression and confirmed by the user.
2. **Non ML only.** If the user asks for an ML baseline here, explain that ML models, simple and complex, are built and compared in ml-model-training, starting with logistic or ridge regression.
3. **Same folds, same inputs.** Baselines use the folds from the split and only columns a model may use. Columns excluded in the framing, and features the leakage check left out, are refused.
4. **No test set.** The test set is used once, in ml-offline-evaluation.
5. **Push back** on an accuracy metric for a rare positive class, on a k that is not smaller than the rows per group, and on skipping the current approach when one exists.
6. **No raw data in the reply.** Use metrics and counts.
7. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, using `scripts/model_core.py`. Do not write the `baseline` section by hand.
8. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status` and `gate`. If the gate exits with code 1, say which step to run first and stop, unless the user explicitly chooses to continue (record that in `decisions`). If the `features` section is empty, run **ml-feature-engineering** first.

Read `business` (current approach), `framing` (task, primary metric, V1 bar), `split` (folds), and `features` (inputs, strongest single features, leakage flags). If `notebooks/07_baseline.ipynb` exists, read its decision cells: the user may have edited them.

Then check that the packages this stage needs are installed: `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py need scikit-learn`. New projects do not install scikit-learn up front. If the check exits with code 1, show the user the `uv add` command it printed, and run it only after they agree.

## Step 2: Get the metric and the options

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --suggest
```

It prints the task, the current approach, the primary metric read from the framing (with a warning when k is not smaller than the rows per scoring date), the non ML options for this task, each with a description and whether it is recommended, and a recommended answers YAML.

## Step 3: Ask which baselines to build

**Round 1 (AskUserQuestion, up to two questions).**

1. **Baselines** (`multiSelect: true`). Question: "Which non ML baselines should I build?" Use the options `--suggest` printed for this task, with their descriptions, recommended ones first and marked (Recommended). AskUserQuestion allows at most four options, so leave out the least useful ones.
   * Classification and regression: the current business rule, the base rate or mean, the best single feature, a segment average.
   * Forecasting: naive (the value one horizon ago), seasonal naive, moving average, the series mean.
2. **Primary metric,** only when the framing has none or `--suggest` warned about it: the proposed metric first (Recommended), with the alternatives for the task. For "at k" metrics, offer `k_share` when k is not smaller than the rows per group.

**Round 2 (plain text, only when the current business rule was chosen).** Write `business.current_approach` as a pandas expression on the column names, for example `days_since_last_order > 30`, and ask the user to confirm it. Use only columns from the feature table or the split's training rows. If the rule uses something that is not in the data, say so and leave the rule out, with an open question.

If AskUserQuestion is unavailable or the user is not present, build the recommended baselines, write the rule from the current approach as an assumption, and say so at the top of the reply.

## Step 4: Build and run the notebook

1. Write the answers to a YAML file in the scratch folder, starting from the `--suggest` output and the keys in `templates/baseline_answers.example.yaml`: `metrics`, the chosen `baselines` (each with `name`, `type`, `params`, `reason`), `reference` (the rule baseline when there is one, otherwise null for the best baseline), `reasoning` (`metrics`, `baselines`), `risks`, `assumptions`, `open_questions`, `decisions`, and `resolved_questions`.
2. Build and run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   It prints the primary metric of every baseline with its spread across folds, the reference, the best baseline and how much it beats the reference, the files, and problems. It refuses ML model types.
3. **Exit code 2:** the user edited the notebook. Ask: "Replace it" (`--overwrite`), "Write a new version" (`--new-version`), or "Keep my edits" (change its decision cells, then `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/07_baseline.ipynb`).
4. If a cell fails, read the error in the notebook. The usual causes are a rule with a column name that does not exist, or a column that is excluded. Fix the answers and run again.
5. Read the problems. A baseline that scores too well suggests leakage: go back to the features. When another baseline beats today's rule, say so plainly: a simpler change to the rule may already help, and the model must beat that baseline too.

## Step 5: Report back

Reply with these parts, in this order:

1. **Files written:** `notebooks/07_baseline.ipynb`, the two parquet files, and `ml_spec.yaml`.
2. **Summary:** a small table of the baselines with the primary metric (mean and spread), then two or three sentences: the reference, the best baseline, and what the V1 bar in the framing means as a number against the reference.
3. **Problems:** from the checks, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** say that the user can change the metrics or baselines in the notebook's decision cells and choose Run All. Then run **ml-model-training**, which builds the ML models from simple to complex on the same folds and compares every one with the reference and the best baseline.

## The spec file

`reference/spec_fields.md` describes the `baseline` section. `scripts/ml_common.py`, `scripts/nb_common.py`, `scripts/split_core.py`, and `scripts/model_core.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
