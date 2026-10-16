---
name: ab-segment-analysis
description: Check whether an A/B test's effect differs between segments, step 5 of the A/B testing skills. For segment columns recorded before assignment, it estimates the effect within each segment, tests whether the effects differ with Cochran's Q, and controls the false discovery rate across columns with Benjamini-Hochberg. Columns recorded after assignment are refused unless the user chooses otherwise with a reason, and then the result is labelled descriptive. Writes notebooks/ab05_segments.ipynb; running it writes the segments section of ab_spec.yaml. Optional. Use after the analysis, or when asked whether a test worked differently for some users, by device, country, or another segment.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Segment Analysis

Find out whether the effect differs between groups of units, without fooling ourselves. With enough segments, some will look different by chance, so the question for each column is whether the effects differ more than chance allows, corrected across columns. The output is `notebooks/ab05_segments.ipynb`, and running it writes the `segments` section of `ab_spec.yaml`.

This is step 5 of the A/B testing skills: ab-project-setup, ab-test-framing, ab-test-design, ab-data-validation, ab-analysis, **ab-segment-analysis**, and ab-readout. It is optional. The shared conventions are in `AB_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Only columns recorded before assignment.** Splitting by something the treatment can change (ads seen, sessions during the test) compares groups the treatment has already sorted, so the segments are no longer like for like. The script refuses such columns unless `allow_post_treatment: true` is set with a reason in `decisions`, and the result is then labelled descriptive.
2. **Few, planned segments.** Prefer the segments named before the results were seen. Every extra column is another chance of a false finding. Push back on "try every column".
3. **A segment is a hypothesis, not a launch plan.** A segment where the effect differs is a reason for the next test, not for shipping to that segment alone.
4. **No raw rows.** Report segment sizes, effects, intervals, and test statistics.
5. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook. Do not edit the `segments` section by hand.
6. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

`cd` to the root (`python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root`) and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status` and `gate`. If the gate exits with code 1, say which step to run first and stop, unless the user explicitly chooses to continue (record it in `decisions`). Read `show --section data` for `pre_treatment_cols`.

**Skipping.** When no segments were planned, or the data has no columns recorded before assignment, mark the step skipped:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --skip --reason "<why>"
```

## Step 2: Choose the segments

Ask with one AskUserQuestion round (multiSelect): "Which segments should I check?" Offer only the columns in `data.pre_treatment_cols`, the planned ones first. If the user asks for a column recorded after assignment, explain the bias in one or two sentences and ask whether to drop it or keep it as descriptive only. Also ask for the metric when it is not the primary.

If AskUserQuestion is unavailable, use only the segments named in the framing; if there are none, skip the step with that reason, and say so at the top of the reply.

## Step 3: Build and run the notebook

Write the answers to a scratch YAML (`segment_cols`, `metric`, `bins`, `min_units`, `allow_post_treatment`, `reasoning` with a `segments` paragraph, `assumptions`, `open_questions`, `decisions`), then:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
```

It prints, per column, whether the effects differ after correction. Exit code 2 means the notebook was edited since the skill wrote it: ask whether to replace it (`--overwrite`), write a new version (`--new-version`), or keep the edits and run it (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/ab05_segments.ipynb`).

## Step 4: Report back

1. **Files written:** `notebooks/ab05_segments.ipynb` and `ab_spec.yaml`.
2. **Summary:** for each column, whether the effects differ after correction, and the segments with the largest and smallest effects, in plain sentences.
3. **Problems:** descriptive columns, segments too small to test, or many columns tested, one sentence each.
4. **Open questions:** one line each.
5. **Next step:** run **ab-readout** (or rerun it if it already ran, since the segments changed).
