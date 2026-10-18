---
name: ml-problem-framing
description: Turn the business framing and the discovered data into an exact ML problem definition, step 4 of the ML project skills. Proposes the target (an existing column or a rule to build it), the unit of prediction, the prediction time column, the outcome window, the task type, the primary metric and V1 bar, and the columns to exclude, using the business section, the ml-data-discovery hints, and the data dictionary in the shared ml_spec.yaml. Checks the framing against the data (history length, grain, class balance, timing of columns), pushes back on leaky or vague choices, and confirms the framing, which unlocks the modelling skills. Writes notebooks/03_problem_framing.ipynb with the reasoning, editable decision cells, and pandas checks; running it writes the framing to the spec. Use after data discovery, or when asked to define the target, label, unit of prediction, or prediction time for a model.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# ML Problem Framing

Define exactly what the model predicts, for which unit, at what moment, and how success is measured, using what the business framing decided and what data discovery found. The output is the notebook `notebooks/03_problem_framing.ipynb`. Its markdown cells hold the reasoning, the risks, the assumptions, and the open questions. Its *Decision* cells hold the target, unit of prediction, prediction time, success measures, and excluded columns, which the user can edit. Its check cells test those decisions against the data in short pandas code. Running it writes the `framing` section of the shared spec `ml_spec.yaml` and the `data` settings (date, ID, and excluded columns).

This is step 4 of the ML project skills: ml-project-setup, ml-business-framing, ml-data-discovery, **ml-problem-framing**, ml-eda-report, then ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, and ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. A confirmed ML framing, followed by the targeted EDA, is what lets the modelling skills start (`ml_common.py gate`).

## Ground rules

1. **Never invent answers.** Propose answers from the business section, the discovery hints, and the data dictionary, and let the user confirm them. Anything still unconfirmed goes in `assumptions` or `open_questions`, never silently into the framing.
2. **Push back.** When a choice is leaky, vague, or does not fit the data or the decision, say so plainly, explain why in one or two sentences, and suggest a fix. Do not accept "churned" as a target definition without a rule and a window.
3. **One question at a time.** Ask exactly one question per message, then wait for the answer before asking the next. Never send a numbered list of questions, and never put more than one question in an AskUserQuestion call. Lead with proposals, so the user can confirm instead of typing. Keep the interview short by skipping anything the spec, the dictionary, or an earlier answer already settles.
4. **No raw data in the reply.** Use column names and the summary numbers in the spec, the dictionary, and the notebook checks. Do not print rows or values.
5. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, and the notebook writes the spec. Do not edit the `framing` section or `data.excluded_cols` by hand.
6. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and read what came before

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use the project's Python, `.venv/bin/python`, as `PY`. If there is no `.venv`, use `python3` and suggest the **ml-project-setup** skill. Then read the spec with `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status` and `show`:

* **`business`:** the goal, decision, capacity, current approach, outcome in business terms, output use, V1 bar intent, and constraints. If `business.status` is not `confirmed`, say that ml-business-framing should come first, and ask whether to continue with what is there. If `business.ml_needed` is `no`, stop and say so.
* **`discovery`:** `main_source`, `sources` (grain, entity IDs, dates, history, gaps), `joins`, `dictionary_summary`, and `framing_hints`. Read `docs/data_dictionary.yaml` too, above all the `timing` of each column.
* **No discovery section:** if the user knows the data well (they say so, or point to a data dictionary), continue with `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py columns <file>` for the column names. Otherwise recommend running **ml-data-discovery** first, because the proposals below depend on it.
* **`framing.status` is `confirmed`:** show a short summary and ask whether to revise it or keep it. Record any revision in `decisions` with the reason, and say that ml-eda-report should run again afterwards.

If `notebooks/03_problem_framing.ipynb` exists, read its decision cells too. They are the source of the framing, and the user may have edited them since the last run.

`reference/question_bank.md` lists every field, the red flags, and how to check the framing against the data.

## Step 2: Propose what is predicted, and when

Ask with AskUserQuestion, one question per call, in the order below, and wait for each answer before asking the next. Put the best proposal first and mark it (Recommended), and say in each option's description which finding supports it. If an answer is leaky or vague, push back with one follow up question before moving on, and adjust the later proposals to fit the answers so far.

1. **Target:** the candidate targets from `discovery.framing_hints` and the dictionary (`candidate_targets`) that fit `business.outcome`, each with its type and, for binary ones, the smaller class share. Always include "Build it from other columns", for targets such as "no order in the next 60 days". If chosen, ask in plain text how it is built, and record it in `target.construction` with `exists_in_data: false`. Point out when the build needs a join, and what `discovery.joins` says about that join.
2. **Unit of prediction:** built from the main source's grain and entity IDs, for example "one `customer_id` per `scoring_date`". If an entity ID has more than one row per value, say so, because the split will have to be grouped or time based.
3. **Prediction time column:** the date columns whose dictionary timing is `event_time`, with their range. Include "No column marks it", and record that as an open question for the split stage.
4. **Outcome window,** with the history check from `reference/question_bank.md`: for each option, say how much labelled history would remain, for example "60 days: about 9 of the 12 months have complete labels".

## Step 3: Propose how success is measured, and what to exclude

Keep asking one question per call, in the same way as Step 2.

5. **Task type and primary metric:** follow `business.output_use` and the metric guide in `reference/question_bank.md`, checked against the class balance. For a smaller class share under 5%, recommend PR AUC or recall at k, and push back on accuracy. Use `business.capacity` for k when the output is a ranked list.
6. **V1 bar:** turn `business.v1_bar_intent` into a number against the current approach or a simple baseline. Then write the same bar as `success.v1_criteria`, the checks a script applies, so model training and evaluation (and later an agent) judge it without reading text. Propose each criterion from the text and confirm them in the same question: a comparison with a named model or baseline (`beats_model`), a share over the reference (`pct_over_reference`), a metric threshold (`threshold`), a project guardrail as a function in `src/v1_checks.py` (`check`), or something only a person can judge (`manual`). Each says whether it is judged on CV or on the test set (`on`). Prefer machine checkable criteria; a bar with a `manual` criterion always needs a person. When a `check` criterion is used, the script creates `src/v1_checks.py` from the template; write the functions there and show the user the code.
7. **Columns to exclude** (`multiSelect: true`): every dictionary column with `timing` of `possibly_after_outcome` (other than the target), identifiers, personal data, and free text, each with its reason. Columns with `timing: unknown` are listed too, with the recommendation to exclude them until their timing is known.

**Follow ups (only if needed, still one at a time):** population, label delay, feature cutoff, and cadence, when the business framing and the data do not already answer them.

## Step 4: Build a draft notebook and check it against the data

Write the answers so far to a YAML file in the scratch folder, with the keys in `templates/framing_answers.example.yaml` exactly, and `status: draft`:

* `framing` and `data` (`date_col`, usually the prediction time column; `id_cols`; and `excluded_cols`, each with `column` and `reason`).
* `reasoning`: one short paragraph each for the target, the unit, the prediction time, success, and the exclusions, saying which finding supports the choice.
* `checks`: `data_path` (the main source file) and `test_days` (the test period the history check allows for).

Build and run it:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
```

The script prints the results of the check cells: the target's class balance, whether the unit of prediction is unique, the prediction time column's range, how many days of labelled history the outcome window and label delay leave after the test period, excluded and risky columns, and whether the metric suits the class balance.

Then check the full framing against those results and the red flags in `reference/question_bank.md`:

1. The target uses information from after the prediction time, or the outcome window overlaps the feature window.
2. The history is too short for the outcome window, any label delay, and a test period.
3. The unit of prediction does not match the grain or how the decision is made.
4. The primary metric does not suit the decision or the class balance.
5. A feature comes from a join that multiplies rows or matches poorly.
6. A column with `before_prediction` timing uses a window that may end after the prediction time.
7. The population in the data differs from the population scored in production.

Tell the user about each problem found, with a suggested fix, and ask how to resolve them one problem at a time, one AskUserQuestion call each. Record each one under `risks` with its `resolution`. A problem that cannot be resolved now goes in `open_questions` with the stage that needs it.

Then show a compact summary: one short paragraph, and a small table with the target definition, unit of prediction, prediction time, outcome window, task type, primary metric, V1 bar, and the number of excluded columns. Ask: "Does this ML framing look right?" with the options "Confirm it" (Recommended), "Change something", and "Keep it as a draft".

## Step 5: Finish the notebook

1. Update the answers file: the agreed changes, `risks`, `assumptions`, `open_questions` (each with its `stage`), `decisions` (each with a `reason`), and `resolved_questions` (the IDs of the open questions this framing answers, each with an `answer`). Set `status` to `confirmed` only when the user confirmed in Step 4 and no required field is empty.
2. Build and run it again with the same command. The draft notebook from Step 4 is replaced, because nobody edited it.
3. **If it exits with code 2,** the user edited the notebook after the skill last wrote it. Do not overwrite it silently. Ask with AskUserQuestion:
   * "Replace it with the new answers": rerun with `--overwrite`.
   * "Write a new version": rerun with `--new-version`, which writes `03_problem_framing_v2.ipynb`.
   * "Keep my edits": make the changes in the existing notebook's decision cells instead, then run it with `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/03_problem_framing.ipynb`.
4. If a cell fails, the notebook is saved with the outputs so far. Read the error, fix the answer that caused it, and run again.
5. When the user explained a column's meaning or timing along the way, it belongs in the data dictionary, which `02_data_discovery.ipynb` writes. Add it to that notebook's `dictionary_answers` cell (`"source.column": {"meaning": ..., "timing": ...}`) and run it with `nb_common.py run`, or tell the user to, if ml-data-discovery is not installed.
6. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py validate`, and fix any errors in the answers, not in the spec.

## Step 6: Report back

Reply with these parts, in this order:

1. **Files written:** `notebooks/03_problem_framing.ipynb` and `ml_spec.yaml`, and the discovery notebook and dictionary if they changed.
2. **Summary:** two or three sentences on the target, the unit and prediction time, the metric and V1 bar, and the status.
3. **Problems:** each problem from the pressure test and how it was resolved, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** say that the user can change any decision in the notebook and choose Run All to update the spec. Then run the **ml-eda-report** skill. It takes the target, date, and ID columns from the spec, analyses the target, checks for leakage, and checks the data against this framing. After that, the split stage can start.

## The spec file

`reference/spec_fields.md` describes every field in `ml_spec.yaml` and the dictionary file. `scripts/ml_common.py` is the only code that should create or restructure the spec, and `scripts/nb_common.py` builds and runs the notebooks. All three are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
