---
name: ab-data-validation
description: Check that an A/B test's data can be trusted before any effect is estimated, step 3 of the A/B testing skills. Maps the data onto the test framing (unit, group column and variant labels, metric columns, date), records which columns were recorded before and after assignment, and runs the trust checks, namely the sample ratio test against the planned split, units in more than one variant, duplicate units, unmapped labels, missing metrics by variant, balance on columns recorded before assignment, and coverage over time. A failure blocks the analysis. Writes notebooks/ab03_validation.ipynb; running it writes the data and validation sections of ab_spec.yaml and saves data/processed/ab_units.parquet. Use after the test framing, or when asked to check an experiment's data, test for a sample ratio mismatch, or check randomisation.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Data Validation

Check whether the test's data can be trusted before any effect is estimated. Most wrong A/B conclusions come from broken assignment or logging rather than from the statistics, and the sample ratio test catches many of them. The output is `notebooks/ab03_validation.ipynb`. Running it writes the `data` section (the column mapping and the timing of each column) and the `validation` section of `ab_spec.yaml`, and saves one row per unit to `data/processed/ab_units.parquet` for the analysis.

This is step 3 of the A/B testing skills: ab-project-setup, ab-test-framing, ab-test-design, **ab-data-validation**, ab-analysis, ab-segment-analysis, and ab-readout. The shared conventions are in `AB_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Never compare the metrics between groups here.** Validation looks at counts and the observed split, labels, column types, missing values, value ranges, duplicates, each metric pooled over all variants, and columns recorded before assignment. A pooled value cannot show which variant did better. The effect is estimated in ab-analysis, after validation has passed.
2. **The planned split comes from the framing.** Never fill it from the observed counts: that makes the sample ratio test pass by construction. If it is unknown, the check is skipped with a warning and an open question.
3. **Timing needs a person.** The script guesses which columns were recorded after assignment from their names. Confirm every guess with the user. A column with unknown timing is treated as recorded after assignment.
4. **Never change input data.** Read from the raw data folder, write only to `data_processed`, `notebooks`, and the spec.
5. **No raw rows.** Report counts, shares, and test statistics. The labels of the group column may be shown, because they must be matched to the variants.
6. **Use the scripts.** `scripts/inspect_data.py` proposes the answers and `scripts/make_notebook.py` builds and runs the notebook. Do not edit the `data` or `validation` sections by hand.
7. **Standard practice first.** When you recommend or offer options for A/B test design or analysis, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
8. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status`. The framing must have its variants (with exactly one control) and its metrics, even as a draft; if not, run **ab-test-framing** first. If the framing is a draft, say that validation can run, but the analysis stays blocked until the framing is confirmed.

## Step 2: Inspect the data

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/inspect_data.py [<data file>] --yaml <scratch>/validation_answers.yaml
```

It prints the size, each column's type and distinct count, the candidate unit, group, date, and metric columns, the labels in the group column, index columns left over from an export, and a first guess at each column's timing. It writes the recommended answers. With no file, it uses the only data file in the raw data folder (or the project root); with several, ask which one holds the test.

## Step 3: Confirm with the user

Ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next. Put the script's guess first and mark it (Recommended). Skip anything the spec or the user already settled.

1. **Unit and group columns,** and how the labels map to the variants in the framing.
2. **Metric columns** for each metric in the framing. A metric with no column is a failure for the primary and a warning for the others.
3. **Timing** (multiSelect): "Which of these columns were recorded after users were assigned to a group?" List every column that is not the unit, group, metric, or date column, with the script's post-treatment guesses first. Explain in one sentence why it matters: the treatment can change those columns, so adjusting for them or splitting by them biases the effect.
4. **Planned split,** only when the framing does not have it: "What share of units was each group meant to get?" with "Not known" as an option.

If AskUserQuestion is unavailable or the user is not present, use the script's guesses, put unconfirmed columns in `unknown_timing_cols`, and say so at the top of the reply.

Add to the answers: `reasoning` (one paragraph each for `mapping`, `timing`, and `split`), `assumptions`, `open_questions` (with stages), `decisions`, and `resolved_questions` for earlier open questions this settles (for example the planned split, as `{id: Q1, answer: ...}`).

## Step 4: Build and run the notebook

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <scratch>/validation_answers.yaml
```

It prints the result (passed, warning, or failed), each failure and warning, and the sample ratio p value. Exit code 2 means the notebook was edited since the skill wrote it: ask whether to replace it (`--overwrite`), write a new version (`--new-version`), or keep the edits and run it (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/ab03_validation.ipynb`). If a cell fails, read the error in the notebook, fix the answers, and run again.

## Step 5: Handle a failure

A failure blocks the analysis (`ab_common.py gate` exits with code 1). Explain each failure in plain language and what usually causes it:

* **Sample ratio mismatch:** assignment that depends on something users do, bots filtered from one group only, a logging change partway through, or a wrong planned split. Ask the user to check the planned split first.
* **Units in more than one variant:** assignment by session or device when the unit is the user, or a bug in bucketing.
* **Unmapped labels or a missing primary metric column:** usually a mapping to fix in the answers.

Never set an override on your own. If the user decides to analyse anyway, set `override: {reason: ..., by: ...}` in the answers and rerun with `--overwrite`. The reason is shown in the readout.

## Step 6: Report back

1. **Files written:** `notebooks/ab03_validation.ipynb`, `data/processed/ab_units.parquet`, and `ab_spec.yaml`.
2. **Summary:** the result, the sample ratio p value against the planned split, the units analysed, and the columns kept out of adjustments because they were recorded after assignment.
3. **Problems:** each failure and warning, one sentence each, with the likely cause.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** if the result is passed or warning (or failed with an override) and the framing is confirmed, run **ab-analysis**. Otherwise say what must be fixed first.
