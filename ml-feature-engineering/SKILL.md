---
name: ml-feature-engineering
description: Build the features for an ML project, step 8 of the ML project skills. Proposes the model inputs from the data dictionary and column types, builds row and history features (calendar parts, days since, logs, ratios, lags and rolling windows with horizon rules) on the full timeline, and defines the learned steps as an unfitted linear or tree preprocessor (one hot or ordinal codes, TargetEncoder for high cardinality) that later stages fit inside each CV fold. Installs config driven src/features.py and a user owned src/custom_features.py, and writes notebooks/06_features.ipynb with leakage (single feature AUC), drift, history window, and fold fit checks; running it saves feature tables as parquet and the features section of ml_spec.yaml. Use after the split, or when asked to create, select, or check features.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Feature Engineering

Decide which columns become model inputs, how new features are built, and how the inputs are prepared for each kind of model. Everything that learns from the data stays unfitted here, so later stages fit it on training rows only, inside each CV fold.

The outputs are:

* `src/features.py`: config driven feature code. `build_features(df, config)` adds the row and history features; `make_preprocessor("linear" or "tree", config)` returns the learned steps as an unfitted sklearn pipeline; `load_config(spec)` and `feature_columns(config)` give the settings and the inputs. Every later stage, and batch scoring later, imports it, so the features are always the same.
* `src/custom_features.py`: hand written features. Created once and never changed by the skill.
* `notebooks/06_features.ipynb`: the record. Decision cells hold the settings, the columns left out with reasons, and a reason for every feature. The other cells build the features and check them.
* `features_train.parquet` and `features_test.parquet` in the processed data folder: the keys, the target, and every input before the learned steps.
* The `features` section of `ml_spec.yaml`.

This is step 8 of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, ml-train-test-split, **ml-feature-engineering**, then ml-baseline-model, ml-model-training, and ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. `reference/feature_rules.md` explains every rule used here.

## Ground rules

1. **Never invent answers.** Propose features from the dictionary, the framing, and the data, and let the user confirm them. Anything unconfirmed goes in `assumptions` or `open_questions`.
2. **Timing first.** A column whose timing is `possibly_after_outcome` or `unknown` stays out until the user confirms it. A lag or rolling window never looks back less than the horizon, and lags of the target never less than the label delay.
3. **Nothing learned outside the folds.** Imputing, encoding, scaling, and TargetEncoder are only defined here, never fitted on all rows.
4. **Push back** on a feature that cannot be known at the prediction time, on month or week features with under a year of history, and on a confirmed leakage flag without a convincing reason.
5. **No raw data in the reply.** Use column names, counts, and scores.
6. **Use the scripts.** `scripts/make_notebook.py` installs the code, builds the notebook, and runs it. Do not edit `src/features.py` or the `features` section of the spec by hand.
7. **Autopilot.** When the arguments include `autopilot` (ml-autopilot or an ml-experimenter agent is driving), or the project root has `.ml_workspace.json`, no person is present. Do not call AskUserQuestion. Take the recommended answer to every question, or the change the driver gave for this stage. Record each answer you took in `assumptions` with `source: autopilot draft`, and never set a framing to `status: confirmed`. Run the stage script with `--json`, and reply with its result and one line per assumption.
8. **Standard practice first.** When you recommend or offer options for ML experimentation design or ML system design, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
9. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Then run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status` and `gate`. If the gate exits with code 1, say which step to run first and stop, unless the user explicitly chooses to continue (record that in `decisions`). If the `split` section is empty, run **ml-train-test-split** first.

Read `framing` (task, target, entity, prediction time, horizon, label delay), `data.excluded_cols`, `split` (strategy, date and group columns, folds), `eda` (leakage suspects, top associations), and `docs/data_dictionary.yaml`. If `notebooks/06_features.ipynb` exists, read its decision cells: the user may have edited them.

Then check that the packages this stage needs are installed: `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py need scikit-learn`. New projects do not install scikit-learn up front. If the check exits with code 1, show the user the `uv add` command it printed, and run it only after they agree.

## Step 2: Get the facts and a recommendation

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --suggest
```

It prints the training size, the proposed inputs by type, every column left out with its reason, and a recommended answers YAML with a reason for every feature. For forecasting it proposes lags and rolling means of the target shifted by the horizon, calendar parts, and the series ID as a category.

## Step 3: Ask one question at a time

Ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next. Put the recommendation first and mark it (Recommended).

1. **Columns left out:** list them with reasons, and ask whether any should be used (`multiSelect: true`, recommended: none). A column with unknown timing can only be used after the user says when it is filled in.
2. **New features:** the proposed row and history steps (logs, days since, ratios, lags, rolling windows), with a recommendation. Ask in plain text whether the user has domain features in mind, such as a ratio or a count; simple ones go in `ratios`, others in `src/custom_features.py`.
3. **Calendar parts,** when there is a date: recommend only what the history supports (see `reference/feature_rules.md`).

**Only if needed, still one at a time:**

4. **High cardinality columns:** confirm TargetEncoder (Recommended), or leave a column out.
5. **Forecasting:** the lag days and rolling windows, with the horizon as the minimum.

If AskUserQuestion is unavailable or the user is not present, use the recommended answers, record them as assumptions, and say so at the top of the reply.

## Step 4: Build and run the notebook

1. Write the answers to a YAML file in the scratch folder, starting from the `--suggest` output and the keys in `templates/feature_answers.example.yaml`. Write `feature_reasons` in the user's terms, and fill in `reasoning` (`inputs`, `steps`, `history`, `encoding`), `risks`, `assumptions`, `open_questions` (each with its `stage`), `decisions` (each with a `reason`), and `resolved_questions`.
2. If the user wants hand written features, write them in `src/custom_features.py` (it is theirs; show them the code), and add the new column names to the numeric or categorical lists.
3. Build and run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   It prints the number of inputs, the outputs of each preprocessor, the strongest single features, features left out by the leakage check, large drift, the files, and problems.
4. **Exit code 2:** the user edited the notebook. Ask: "Replace it" (`--overwrite`), "Write a new version" (`--new-version`), or "Keep my edits" (change its decision cells, then `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/06_features.ipynb`).
5. **Exit code 3:** `src/features.py` was edited. Explain that hand written features belong in `src/custom_features.py`, and ask before rerunning with `--replace-code`.
6. **Leakage flags:** for every feature left out by the check, tell the user its score and ask whether it is known at the prediction time. If they confirm with a convincing reason, add it to `confirmed_safe` with a decision, and rebuild. Otherwise it stays out.
7. If a cell fails, read the error in the notebook, fix the answers, and run again.

## Step 5: Report back

Reply with these parts, in this order:

1. **Files written:** `notebooks/06_features.ipynb`, `src/features.py`, `src/custom_features.py` (when created), the feature tables, and `ml_spec.yaml`.
2. **Summary:** the number of inputs by kind, the new features and why, and the outputs of each preprocessor, in two to five sentences.
3. **Problems:** leakage flags and how they were resolved, large drift, and missing values in the linear preprocessor, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** say that the user can change the settings in the notebook's decision cells and choose Run All. Then run **ml-baseline-model**, which scores the current business rule and simple models on the same folds with the linear preprocessor fitted inside each fold.

## The spec file

`reference/spec_fields.md` describes the `features` section. `scripts/ml_common.py`, `scripts/nb_common.py`, and `scripts/split_core.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`. The project code comes from `templates/features.py` and `templates/custom_features.py`.
