---
name: ml-train-test-split
description: Split the data into a final test set and cross validation folds on the training set, step 7 of the ML project skills. Chooses the strategy from the confirmed ML framing (random stratified, grouped by entity, or by time with a gap and rolling backtest folds for forecasting and point in time predictions), checks the split for leakage, and saves train, test, and fold files as parquet. Writes notebooks/05_split.ipynb with the reasoning, editable decision cells, and pandas checks; running it writes the split section of the shared ml_spec.yaml. Use after the targeted EDA, or when asked to split data, set up cross validation or backtesting, or hold out a test set.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Train, Validation, and Test Split

Decide what "unseen data" means for this model, before any feature work. The output is the notebook `notebooks/05_split.ipynb`. Its markdown cells say why the strategy, test period, gap, and folds were chosen. Its *Decision* cells hold the data path, the target, and the split settings, which the user can edit. Its other cells make the split and check it in short pandas code. Running it writes, all as parquet in the processed data folder:

* `train.parquet`: the training rows, with a `fold` column (the CV fold the row validates, or -1).
* `test.parquet`: the final test set, used once at the end.
* `cv_folds.parquet`: one row per fold, with sizes and, for a time split, date ranges.
* `gap.parquet`: time split only, the rows between training and test, kept as feature history.

It also writes the `split` section of `ml_spec.yaml`. Cross validation on the training set is how later stages compare and tune models; the test set is not touched until the final evaluation.

This is step 7 of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, **ml-train-test-split**, then ml-feature-engineering, ml-baseline-model, ml-model-training, and ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. `reference/strategies.md` explains every strategy, the gap, backtest folds, and the red flags.

## Ground rules

1. **Never invent answers.** Recommend settings from the framing and the data, and let the user confirm them. Anything still unconfirmed goes in `assumptions` or `open_questions`.
2. **Push back.** A random split of data where the prediction is made at a point in time, a time split without a gap when the label looks ahead, or entities on both sides when production scores new entities: say why it would give a score that is too good, and suggest the fix.
3. **No raw data in the reply.** Use column names, counts, shares, and dates. Do not print rows.
4. **Never change the input data.** Read from the raw data, and write only parquet files to the processed data folder.
5. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, which uses `scripts/split_core.py` for the split. Do not write the `split` section of the spec by hand.
6. **Autopilot.** When the arguments include `autopilot` (ml-autopilot or an ml-experimenter agent is driving), or the project root has `.ml_workspace.json`, no person is present. Do not call AskUserQuestion. Take the recommended answer to every question, or the change the driver gave for this stage. Record each answer you took in `assumptions` with `source: autopilot draft`, and never set a framing to `status: confirmed`. Run the stage script with `--json`, and reply with its result and one line per assumption.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use the project's Python, `.venv/bin/python`, as `PY`. If there is no `.venv`, use `python3` and suggest **ml-project-setup**. Then:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py gate
```

If the gate exits with code 1, say which step to run first (ml-problem-framing, or ml-eda-report on the confirmed target) and stop. Continue only if the user explicitly chooses to, and record that choice in `decisions` with the reason.

Read `framing` (task type, target, entity ID columns, prediction time column and horizon, outcome window, label delay), `data` (date and ID columns), and `eda` (grain, duplicate rows, target balance). If a `split` section exists, or `notebooks/05_split.ipynb` exists, read it too: the user may have edited the notebook's decision cells.

Then check that the packages this stage needs are installed: `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py need scikit-learn`. New projects do not install scikit-learn up front. If the check exits with code 1, show the user the `uv add` command it printed, and run it only after they agree.

## Step 2: Get the facts and a recommendation

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --suggest
```

It prints the gate, the data size, the class balance, the date range, rows per entity, exact duplicates, a recommended strategy with the reason for each choice, and a recommended answers YAML. It never prints rows.

If the target is not a column yet (`framing.target.exists_in_data` is false), write the body of `build_target(df)` from `framing.target.construction` with the user, as `target_code` in the answers. It must use only information known when the outcome is decided, and it runs before the split.

## Step 3: Ask one question at a time

Ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next. Put the recommendation first and mark it (Recommended), and say in each description what it means for the score.

1. **Strategy:** time, group, or random, with the reason from `--suggest`. If the framing has a prediction time column or the task is forecasting, recommend time, and explain that a random split would let the model see the future.
2. **Test set:** for time, the test period in days and the gap (offer the recommended values and one shorter and one longer test period, with how much training history each leaves). For random and group, the test share (0.2 recommended, 0.25 or 0.3 for small data).
3. **Cross validation folds:** the number of folds. For time, also the validation window and expanding or sliding window. Cross validation on the training set is the default way later stages compare models.

**Only if needed, still one at a time:**

4. **Duplicates,** when there are exact duplicate rows: keep together (Recommended), drop, or ignore.
5. **Group column,** when the group strategy is chosen and there is more than one candidate entity column.

If AskUserQuestion is unavailable or the user is not present, use the recommended answers, record them as assumptions, and say so at the top of the reply.

## Step 4: Build and run the notebook

1. Write the answers to a YAML file in the scratch folder, starting from the `--suggest` output and the keys in `templates/split_answers.example.yaml`. Fill in `reasoning` (one short paragraph each for `strategy`, `test`, `cv`, and `duplicates`, in the user's terms), `risks` with their resolution, `assumptions`, `open_questions` (each with its `stage`), `decisions` (each with a `reason`), and `resolved_questions` for open questions this split answers.
2. Build and run it:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   It prints the strategy, the rows in each set, the test period and gap, every fold, the files written, and any problems the leakage checks found.
3. **If it exits with code 2,** the user edited the notebook after the skill last wrote it. Do not overwrite it silently. Ask with AskUserQuestion: "Replace it with the new answers" (rerun with `--overwrite`), "Write a new version" (rerun with `--new-version`, which writes `05_split_v2.ipynb`), or "Keep my edits" (change the existing notebook's decision cells instead, then run `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/05_split.ipynb`).
4. If a cell fails, the notebook is saved with the outputs so far. Read the error. The usual causes are a fold with no rows (use fewer folds, a shorter validation window, or a shorter gap) or a target that is not in the data (fill in `target_code`). Fix the answers and run again.
5. If the leakage checks report a problem, tell the user and fix the settings, unless the user accepts it, which goes in `risks` with the reason.

## Step 5: Report back

Reply with these parts, in this order:

1. **Files written:** `notebooks/05_split.ipynb`, the parquet files, and `ml_spec.yaml`.
2. **Summary:** the strategy and why, the rows and class balance of each set, the test period and gap for a time split, and the folds, in two to five sentences.
3. **Problems:** anything the checks found, and how it was resolved, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** say that the user can change any setting in the notebook's decision cells and choose Run All, which rewrites the files and the spec. Then run **ml-feature-engineering**, which fits every learned step inside the CV folds on training rows only, and for a time split builds lag features on the full timeline before cutting by date.

## The spec file

`reference/spec_fields.md` describes the `split` section and every other field in `ml_spec.yaml`. `scripts/ml_common.py`, `scripts/nb_common.py`, and `scripts/split_core.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`. Later stages read the folds with `split_core.cv_splits(train, spec["split"])`.
