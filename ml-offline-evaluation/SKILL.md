---
name: ml-offline-evaluation
description: Score the champion, every other ML model, and every baseline on the test set once, step 10 of the ML project skills. Shows each metric with a 95% bootstrap interval, paired comparisons against the reference and the best baseline, charts (gain curve with capacity, metric as k changes, ROC and precision recall, residuals, metric over time, calibration, slice dot plot), metrics sliced by proposed columns, test against CV, importance on test, and score drift. Ends with an evidence table and a model card; the go or no go decision stays with the user. Writes notebooks/09_evaluation.ipynb. Use after model training, or when asked to evaluate, test, or write a model card.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Offline Evaluation

Score every model on the test set once, show the evidence in tables and charts, and leave the decision to the user. Nothing is chosen or tuned here.

The outputs are:

* `notebooks/09_evaluation.ipynb`: the record. Decision cells hold the models to score, the slices, and the bootstrap settings. Other cells show the metrics with intervals, the comparisons, the charts, the slices, the checks, and the evidence table.
* `test_predictions.parquet`, `evaluation_metrics.parquet`, and `evaluation_slices.parquet` in the processed data folder.
* `docs/model_card.md`: written from the spec. When the user edited it, the card is kept and `docs/model_card_new.md` is written next to it.
* The `evaluation` section of `ml_spec.yaml`, including a record of every time the test set was opened.

This is step 10 of the ML project skills: ml-project-setup, ml-business-framing, ml-data-discovery, ml-problem-framing, ml-eda-report, ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, **ml-offline-evaluation**. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. `reference/evaluation.md` explains every table, chart, and check.

## Ground rules

1. **The champion is fixed.** It was chosen by its mean CV score in model training. If another model does better on the test set, say so, but do not switch: choosing with the test set makes the test score optimistic.
2. **Evidence, not a verdict.** The notebook and the reply present the evidence. Do not decide go, iterate, or no go for the user. Record a decision only when the user states one.
3. **Clearly mark the champion.** Every table marks it "(champion)" and every chart draws it as a thick red line.
4. **Warn about test reuse.** When the test set was opened before for another champion or other data, say so at the top of the reply.
5. **No raw data in the reply.** Use metrics and counts.
6. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook. Do not write the `evaluation` section or the model card by hand.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status` and `gate`. If the gate exits with code 1, say which step to run first and stop, unless the user explicitly chooses to continue (record that in `decisions`). If the `training` section has no champion, run **ml-model-training** first.

Then check that the packages this stage needs are installed: `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py need scikit-learn`. New projects do not install scikit-learn up front. If the check exits with code 1, show the user the `uv add` command it printed, and run it only after they agree.

## Step 2: Get the facts

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --suggest
```

It prints whether the test set was opened before, the models to score, the test and training sizes, the bootstrap grouping, the proposed slices, and a recommended answers YAML.

If the test set was opened before, ask with AskUserQuestion before going on: "The test set has been opened N times. Another look gives an optimistic score. Score it again?" Options: "Yes, and record it as look N+1" and "No, stop at the CV results". When an iteration is open (`iterations.current` in the spec, from the ml-iterate skill), say which iteration this look belongs to. The model card states how many times the test set was opened.

## Step 3: Ask

Ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next:

1. **Models.** "Which models should be scored on the test set?" Options: "All: the champion, the other ML models, and the baselines (Recommended)", and "The champion and the baselines only". Name the champion in the question.
2. **Slices.** "Which slices should the metrics be split by?" Use `multiSelect: true`, with one option per proposed slice dimension (at most four; group the numeric bins as one option). Say in the description that slices smaller than 100 rows or 30 positives are shown but marked too small, and that the user can type other columns.
3. **Bootstrap grouping** (only when the split is random and an entity or date column repeats in the test set): "Should related rows be drawn together for the intervals?" Options: "Single rows (Recommended)", and "Rows of the same <column> together". For time and group splits, the date or the group is used without asking.

If AskUserQuestion is unavailable or the user is not present, use the recommended answers, record them as assumptions, and say so at the top of the reply.

## Step 4: Build and run

1. Write the answers to a YAML file in the scratch folder, starting from the `--suggest` output and the keys in `templates/evaluation_answers.example.yaml`: `models`, `slices`, `bootstrap`, `reasoning` (`slices`), `risks`, `assumptions`, `open_questions`, `decisions`, and `resolved_questions`.
2. Run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   It writes `09_evaluation.ipynb`, runs it, and prints the test metrics with intervals, the comparisons, and the evidence. If the notebook was edited since the skill wrote it, it exits with code 2: ask the user, then rerun with `--overwrite` or `--new-version`. To rerun the notebook as it is, use `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/09_evaluation.ipynb`.
3. If a cell fails, read the error in the notebook, fix the answers, and run again.
4. Look at the chart outputs in the notebook before you report, and check that the champion line and the capacity line make sense.

## Step 5: Report back

Reply with these parts, in this order:

1. **Warning** (only when the test set was opened before for another champion or other data).
2. **Files written:** the notebook, the three parquet files, the model card, and `ml_spec.yaml`.
3. **Test results:** a small table with every model (champion marked), the primary metric on test with its interval, and the CV score. Then the paired comparisons against the reference and the best baseline, and for classification, how many positives the champion and the reference find at the operating point.
4. **Evidence:** one sentence per question in the evidence table: the comparisons, the V1 bar, test against CV, other models, weak slices, calibration, and drift.
5. **Open questions:** one line each.
6. **Next step:** ask the user for their decision (go, iterate, or no go) and offer to record it with the reason. Mention that iterating means going back to `/ml-feature-engineering` or `/ml-model-training`, and that the test set has now been seen. `/ml-status` shows the project, and `/ml-reconcile-notes` aligns the notes.

When the user states a decision, add it to the `decisions` list in the last decision cell of the notebook with `nb_common.edit_cell`, then rerun the notebook. The decision goes to the spec with stage `evaluation`, and the model card shows it.

## The spec file

`reference/spec_fields.md` describes the `evaluation` section. `scripts/eval_core.py` holds the evaluation logic (bootstrap, slices, operating point, and the model card). `scripts/ml_common.py`, `scripts/nb_common.py`, `scripts/split_core.py`, and `scripts/model_core.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
