---
name: ml-review
description: Independent review of an ML project built with the ml- skills, for leakage, settings that disagree between stages, results that look too good, honest use of the test set, statistical strength, business fit, and reproducibility. Asks whether to review the whole project or chosen stages, runs scripted checks, then has a fresh agent that did not build the work read the notebooks and the code. Writes notebooks/10_review.ipynb (with a note on any open blockers) and review_findings.parquet, which tracks each finding as open, accepted, or fixed across reviews. Changes nothing else. Use at any time, especially before ml-offline-evaluation, or when asked to review, audit, or check the project.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Project Review

Review the project the way a careful senior data scientist would before trusting its results. The review has two layers:

1. **Scripted checks** (`scripts/review_core.py`): fixed rules on the spec, the parquet files, the notebooks, and `src/`. They give the same result every time, so each finding keeps its id and can be tracked as open, accepted, or fixed.
2. **An independent reviewer**: a fresh agent with no memory of the conversation that built the project. It reads the notebooks, the decision cells, the reasoning, and the feature code with the checklist in `reference/reviewer_brief.md`, and adds what the scripts cannot see.

The outputs are:

* `notebooks/10_review.ipynb`: the record. Decision cells hold the stages reviewed, the accepted findings (each with a reason), and the reviewer's findings. Other cells run the checks, merge the findings with earlier reviews, and show a note about any open blockers.
* `review_findings.parquet` in the processed data folder: every finding of every review, with its status.
* The `review` section of `ml_spec.yaml`, which ml-status reads.

This is a helper skill, not a workflow step. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Read only.** The review never changes another stage's notebook, spec section, data, or code. Each finding names the skill that should fix it.
2. **Keep the reviewer independent.** Give the reviewer only the brief, the file list, and the scripted findings. Do not pass on this conversation, your own opinion of the work, or what you expect it to find.
3. **Do not soften the reviewer.** Put its findings in the notebook as it wrote them. If you disagree with one, say so in the reply with your evidence, and let the user decide whether to accept it.
4. **Severity means something.** A blocker makes a reported number wrong or untrustworthy. A warning could change a decision. A note is worth knowing.
5. **No raw data in the reply.** Use counts and metrics.
6. **Use the scripts.** Do not write the `review` section or the findings file by hand.
7. **Standard practice first.** When you recommend or offer options for ML experimentation design or ML system design, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
8. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. The review needs no gate: it reviews whatever stages are done. If no stage after the ML framing is done, say there is little to review yet and stop, unless the user wants the framing reviewed.

## Step 2: Get the facts

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --suggest
```

It prints the stages done, the last review (its date, open counts, and the stages that ran again since), the findings accepted earlier, and the scripted findings for every stage done.

## Step 3: Ask about the scope

**First question (AskUserQuestion).** "Should the review cover the whole project, or only some stages?" Options:

* "The whole project (Recommended)": every stage that is done.
* "Only the stages that ran again since the last review" (offer this only when there was a review and some stages ran since; name them).
* "Choose the stages".

**Second question, only for "Choose the stages" (AskUserQuestion, `multiSelect: true`), after the first is answered.** "Which stages should be reviewed?" Offer the groups that have stages done, at most four: "Framing and EDA", "Split and features", "Baseline and training", "Evaluation". Map them to `framing, eda`, `split, features`, `baseline, training`, and `evaluation`.

If AskUserQuestion is unavailable or the user is not present, review the whole project and say so at the top of the reply.

## Step 4: Brief the independent reviewer

1. Print the brief for the chosen stages:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/review_core.py brief <stage> <stage> ...
   ```

   It lists the project, the files to read, the saved parquet files, and the scripted findings.
2. Launch one agent with the Agent tool (`subagent_type: general-purpose`; it is a fresh agent, so it has no context from this conversation). Its prompt is the full text of `reference/reviewer_brief.md`, followed by the brief from step 1. Add nothing else about the project.
3. Its reply ends with a YAML block with `reviewer` (`summary`, `files_read`) and `reviewer_findings`. If the YAML is missing or malformed, ask the same agent (SendMessage) to send it again in the required form.
4. Check that the reply contains no raw rows. Do not edit the findings.

## Step 5: Build and run

1. Write the answers to a YAML file in the scratch folder, with the keys in `templates/review_answers.example.yaml`: `stages`, `accept` (only reasons the user gave you; earlier accepted findings are kept automatically), and the reviewer's `reviewer` and `reviewer_findings`. Add `reviewer.at` with the current time.
2. Run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   It writes `10_review.ipynb`, runs it, and prints the open findings by severity, the counts, and how many findings were fixed since the last review. If the notebook was edited since the skill wrote it, it exits with code 2: ask the user, then rerun with `--overwrite` or `--new-version`.
3. Findings of stages that were not reviewed this time are carried over as they were, and say so.

## Step 6: Report back

Reply with these parts, in this order:

1. **Blockers:** the note from the notebook. When there are open blockers, list each one with its fix, and say that the results of those stages and every later stage should not be relied on, shared, or used to open the test set until each is fixed or accepted with a reason. When there are none, say so in one sentence.
2. **Findings:** a small table of the open warnings and notes (stage, source, title, fix). Mention how many were accepted and how many were fixed since the last review.
3. **The reviewer's view:** its summary in one or two sentences, and any finding you disagree with, with your evidence.
4. **Files written:** `notebooks/10_review.ipynb`, `review_findings.parquet`, and `ml_spec.yaml`.
5. **Next step:** fix the blockers with the skills named, then run `/ml-review` again. To accept a finding instead, the user gives a reason.

## Accepting a finding later

When the user accepts a finding with a reason, add it to `ACCEPTED` in the first decision cell of the notebook with `nb_common.edit_cell`, then rerun the notebook as it is:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/10_review.ipynb
```

This does not launch a new reviewer. The accepted finding stays accepted in later reviews until it is removed from `ACCEPTED`.

## The spec file

`reference/spec_fields.md` describes the `review` section. `scripts/review_core.py` holds the scripted checks and the findings history; add a check there when a stage gains a new way to go wrong. `scripts/ml_common.py`, `scripts/nb_common.py`, `scripts/split_core.py`, and `scripts/model_core.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
