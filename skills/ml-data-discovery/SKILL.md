---
name: ml-data-discovery
description: Find out what the data is before deciding what to predict, step 3 of the ML project skills. For each data file it records the size, the grain (what makes a row unique), possible entity IDs, coverage over time with gaps and freshness, and the main quality issues; across files it checks how they join. It drafts a data dictionary from the column names and the data, with each column's meaning and when it is filled in, and flags the columns it cannot work out. Everything is in notebooks/02_data_discovery.ipynb as short pandas code the user can check and rerun; running it writes docs/data_dictionary.yaml and hints for the ML framing (candidate targets, history length, columns that may be recorded after the outcome) to the shared ml_spec.yaml. Use when starting work on new data, when asked to explore, inventory, or document data sources, or to build a data dictionary.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Data Discovery

Find out what the data is and how it comes about, so the ML framing can be built on facts instead of assumptions. The outputs are:

* `notebooks/02_data_discovery.ipynb`: the record of this step. Each finding is computed in a short pandas cell the user can read and rerun: the first rows, columns and types, the grain, entity IDs, coverage over time, missing values, joins, candidate targets, and columns that may be recorded after the outcome. Then the drafted dictionary, the columns to review, and a *Decision* cell, `dictionary_answers`, where the answers go.
* `docs/data_dictionary.yaml`: the data dictionary, written by the notebook.
* The `discovery` section of `ml_spec.yaml`, the entries in `data.sources`, and grouped open questions, also written by the notebook.

This is step 3 of the ML project skills: ml-project-setup, ml-business-framing (anchors only), **ml-data-discovery**, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, then ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, and ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the skills folder.

Data discovery describes the data as it is. It does not pick a target and it does not study which features predict anything: that waits for ml-eda-report, after ml-problem-framing has confirmed the target.

## Ground rules

1. **Summary statistics only in the reply.** Never print raw rows to the conversation, and do not run `head`, `cat`, or `df.head()` on the data yourself. The notebook's First Rows cell is the only place raw rows appear, for the user to look at. Column names, counts, shares, and date ranges are fine.
2. **Never change the data.** Read only. Suggest fixes, do not apply them.
3. **Draft, flag, never guess.** The dictionary is drafted from the column names and the data. When the meaning or the timing of a column cannot be worked out, it stays empty and flagged for review. Claude may improve the wording of a clear draft, and may fill in a flagged entry only from something the user or a document said. `reference/dictionary_rules.md` explains every rule.
4. **Use the scripts.** `scripts/discover.py` builds and runs the notebook. The notebook uses `scripts/eda_core.py` (the EDA engine), `scripts/nb_common.py` (notebooks), and `scripts/ml_common.py` (the spec). Do not edit `docs/data_dictionary.yaml` or the `discovery` section by hand, because the next run would undo it.
5. **Autopilot.** When the arguments include `autopilot` (ml-autopilot or an ml-experimenter agent is driving), or the project root has `.ml_workspace.json`, no person is present. Do not call AskUserQuestion. Take the recommended answer to every question, or the change the driver gave for this stage. Record each answer you took in `assumptions` with `source: autopilot draft`, and never set a framing to `status: confirmed`. Run the stage script with `--json`, and reply with its result and one line per assumption.
6. **Standard practice first.** When you recommend or offer options for ML experimentation design or ML system design, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use the project's Python, `.venv/bin/python`, as `PY`. If there is no `.venv`, use `python3` and suggest the **ml-project-setup** skill. Check packages with `PY ${CLAUDE_SKILL_DIR}/scripts/discover.py --check`. If something is missing, show the user `uv add <package>` (or the command `--check` implies) and ask before installing. Then read the spec:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py paths
```

Read `business` too. Only the mode and the two anchors come before discovery: the goal and the decision (`business_goal`, `decision_supported`) for a standard project, or the learning question and the answer actions (`learning_question`, `answer_actions`) for an exploratory one. They tell you which source the model will most likely use and which columns may relate to the outcome. If the anchors are not done, say that the anchors pass of ml-business-framing normally comes first, and continue if the user wants. The rest of the business framing is drafted from what this skill finds, in the full pass of ml-business-framing that comes next.

## Step 2: Choose the sources

List the files in the `data_raw` folder (names and sizes only, for example with `ls -la`). Then ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next, leaving out what the request already answers:

1. **Which files to include** (`multiSelect: true`), when there are several. Recommend all data files, and leave out obvious extras such as lookup tables only if the user agrees.
2. **The main source:** the one the model will most likely be trained on, for example the table with one row per customer and scoring date. Recommend the one that best fits the anchors (`decision_supported` or `business_goal`, or `learning_question`), then `business.outcome` if an older spec has it, otherwise the largest. The framing hints describe this source.

Give each source a short name if the file names are long or unclear (passed with `--names`).

If AskUserQuestion is unavailable or the user is not present, include every data file, use the largest as the main source, and say so at the top of the reply.

## Step 3: Build and run the notebook

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/discover.py <file> [<file> ...] [--names a,b] [--main <name>]
```

This builds `notebooks/02_data_discovery.ipynb` and runs it, which takes about as long as one EDA per file. Running it writes the dictionary and the spec. The script then prints a summary of each source, the joins, and the dictionary counts. Answers already confirmed in an earlier run are put back into the `dictionary_answers` cell, so a rerun after new data arrives keeps them.

**If it exits with code 2,** the user edited the notebook after the skill last wrote it. Do not overwrite it silently. Ask with AskUserQuestion:

* "Replace it": rerun with `--overwrite`. Confirmed dictionary answers are kept, because they are read back from the dictionary file.
* "Write a new version": rerun with `--new-version`, which writes `02_data_discovery_v2.ipynb`.
* "Keep my edits": run the edited notebook as it is with `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/02_data_discovery.ipynb`.

If a cell fails, the notebook is saved with the outputs so far. Read the error in the notebook, fix the cause (often a date column that does not parse, which `--answers` can set under `date_columns`), and run again.

## Step 4: Review the drafted dictionary

Read `docs/data_dictionary.yaml` (column names and statistics only, no data values) and decide on answers within the rules:

1. **Clear names:** better wording for awkward drafts, for example "Days since last order" becomes "Days since the customer's last order, as of the row's date". Mark these `source: reviewed`.
2. **Context the user gave:** if the request, the business framing, or a document the user pointed to explains a column, give its meaning or timing, mark it `source: reviewed`, and say where it came from in `notes`.
3. **Everything else that is flagged stays flagged.** Do not guess.

Then ask the user about the flagged columns, grouped so it stays short:

* If there are only a few (up to about eight), ask in plain text, listing each column with what is unclear: its meaning, or when it is filled in relative to the outcome.
* If there are many, show the count and the most important ones (columns that may be recorded after the outcome come first, because they can leak), and say that the notebook's Columns to Review table lists the rest. Offer to go through them together, or to leave them for the user to fill in the notebook.

Put the answers in an answers YAML file in the scratch folder, in the same form as the notebook cell (`source` is `user` for the user's answers, which is the default):

```yaml
dictionary_answers:
  customers.seg_cd:
    meaning: Marketing segment code assigned at sign up.
    timing: static
  orders.refund_flag:
    timing: possibly_after_outcome
    notes: Set when a refund is processed, which can be weeks after the order.
source_descriptions:
  customers: One row per customer, exported from the CRM.
```

and rebuild with the same files and options plus `--answers <file>`. If the user has edited the notebook, add the answers to its `dictionary_answers` cell instead and run it with `nb_common.py run`.

## Step 5: Check the spec

The notebook already wrote the `discovery` section, the `data.sources` entries, and grouped open questions (unclear columns, columns that may be recorded after the outcome, joins that need checking, and the business anchors when they are still missing), and closed dictionary questions that are now answered. Check them with `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status`. Do not change `framing` or `data.excluded_cols`: excluding columns is decided in ml-problem-framing.

## Step 6: Report back

Reply with these parts, in this order. Do not paste the notebook, and do not include data values.

1. **Files written:** `notebooks/02_data_discovery.ipynb`, `docs/data_dictionary.yaml`, and `ml_spec.yaml`.
2. **Summary:** for each source, one or two sentences on its size, grain, and date coverage. Then the joins, one sentence each.
3. **Problems:** gaps or thin months, stale data, joins that multiply rows or match poorly, and the High severity quality issues, one sentence each.
4. **Open questions:** the dictionary columns still flagged (how many, and the most important by name), and the other open questions in the spec, one line each with the stage that needs the answer.
5. **Next step:** say that the user can check every finding in the notebook, and can answer flagged columns in its `dictionary_answers` cell and choose Run All. Then answer the columns that may be recorded after the outcome, and run **ml-business-framing** again for the full pass, which drafts the rest of the business framing from these findings. After that, run **ml-problem-framing**, which proposes the target, unit of prediction, prediction time, and window.

## The spec file

`reference/spec_fields.md` describes the `discovery` section and the dictionary file. `scripts/ml_common.py`, `scripts/nb_common.py`, and `scripts/eda_core.py` are copies of the canonical files in `skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
