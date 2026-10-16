---
name: ml-reconcile-notes
description: Find and fix notes that disagree across the stages of an ML project, a helper for the ML project skills. Gathers every note (reasoning and decision cells in the stage notebooks, and the assumptions, open questions, decisions, risks, and settings in the shared ml_spec.yaml), runs fixed checks for settings that disagree, reads the free text for stale or contradicting notes (for example a V1 bar that names a baseline that was never built), and asks the user which way to align each conflict. Fixes each note in the notebook that owns it, keeps the decision history (superseded decisions, resolved questions, retired assumptions), backs up first, and runs the edited notebooks and every later one again. Use after changing a decision, when /ml-status reports notes that may disagree, or when asked to update, align, reconcile, or clean up the project notes.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Reconcile Notes

Decisions change as a project goes on, but the notes written earlier do not update themselves. This skill finds notes that no longer agree with each other and aligns them, with the user's approval for every change.

It is a helper for the ML project skills, not a workflow step: run it at any time, especially after changing a decision. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Fix a note where it is owned.** Each spec section is written by its stage notebook, so a note is changed in that notebook's decision cell or reasoning text, and the notebook is run again. Never edit a section of `ml_spec.yaml` by hand: the next run of its notebook would put the old text back. The only spec edits are to the shared lists, through the script.
2. **Ask which way to align.** The later note is usually right, but not always: the later step may have drifted by mistake. Every change needs the user's approval.
3. **Keep the history.** Decisions are never deleted: a replaced decision is marked `superseded_by`. Answered questions are marked resolved. Assumptions that no longer hold move to `retired_assumptions` with the reason.
4. **Change notes, not the work.** When aligning means doing different work (for example adding a baseline that the V1 bar expects), say so and point to the skill that does it, instead of editing results.
5. **Use the scripts.** `scripts/reconcile.py` gathers the notes, checks them, backs up, applies the plan, and runs the notebooks again.
6. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. If a notebook was edited and not run again (`/ml-status` says so), ask the user to run it first, so the notes and the spec match before comparing them.

## Step 2: Gather the notes and run the fixed checks

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/reconcile.py check
PY ${CLAUDE_SKILL_DIR}/scripts/reconcile.py inventory --out <scratch folder>/notes.json
```

`check` lists settings that disagree by fixed rules: the target, date column, or entity differing between stages; columns excluded in the framing but used as features; the primary metric or its k differing between the framing and the baseline without a note saying why; a V1 bar that names a model that is not a baseline; a gap shorter than the label delay; lags shorter than the horizon; ML being built after the business framing said it is not needed; and open questions for stages that are finished.

`inventory` prints every note by stage, with where it lives (a notebook cell, a spec field, or a shared list) and when it was written.

## Step 3: Read the notes for conflicts

Read the inventory from the first stage to the last, and look for notes that the fixed checks cannot catch:

1. **Stale references:** a note that names something that changed later, such as a model, a column, a metric, a window, or a strategy.
2. **Contradictions:** two notes that cannot both be true, such as an assumption and a later decision.
3. **Answered questions:** open questions that a later decision, assumption, or result answers.
4. **Outdated reasoning:** a reasoning paragraph that explains a choice that was later changed.
5. **Assumptions** that were confirmed (they become facts) or proved wrong.

Do not report differences that are intended and explained, such as k of 50 on the sample with an assumption saying why, or wording that differs but means the same.

## Step 4: Agree the fixes with the user

Show a short table of the findings: number, what disagrees, where each note lives, and the proposed fix. Then ask with AskUserQuestion, up to four findings per call, one question per finding. For each, the options are:

* "Update the earlier note (Recommended)" when the later decision is the current one, with the new wording in the description.
* "Keep the earlier note and change the later one" when the later step drifted. If that needs different work rather than different words, say which skill to run instead.
* "Leave it as it is," for differences that are intended; offer to record why as an assumption or decision so the checks stop reporting it.

Put the recommended option first. If AskUserQuestion is unavailable or the user is not present, make no changes: list the findings and the proposed fixes, and stop.

## Step 5: Write the plan and apply it

1. Locate each note with `PY ${CLAUDE_SKILL_DIR}/scripts/reconcile.py find "exact text"`, which prints the notebook cells and spec fields that contain it.
2. Write the plan to a YAML file in the scratch folder, with the keys in `templates/plan.example.yaml`: `edits` (notebook, cell, old text that appears once in the cell, new text, why), `supersede_decisions`, `resolve_questions`, `retire_assumptions`, and `replace_assumptions`. A retired or reworded assumption must also be changed in the decision cell that lists it.
3. Check it, then apply it:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/reconcile.py apply <plan.yaml> --dry-run
   PY ${CLAUDE_SKILL_DIR}/scripts/reconcile.py apply <plan.yaml>
   ```

   `apply` refuses the whole plan when any item does not match, so nothing is half applied. It backs up the spec and every notebook it changes to `.reconcile_backup/<date and time>/`, applies the cell edits, then runs the notebooks again: every notebook with a changed decision cell and every later notebook, in workflow order, so the spec follows. A notebook whose only change is in its reasoning text is not run. If a run fails, it stops there and says which notebook failed. Then it updates the shared lists and runs the fixed checks again.

The edited notebooks now hold the user's approved changes, so a stage skill that rebuilds one of them later asks first, as with any edit.

## Step 6: Report back

Reply with these parts, in this order:

1. **Changes:** one line per fixed note: what changed and where.
2. **Notebooks run again,** and any that failed, with the error.
3. **Left as it is:** findings the user chose to keep, with the reason recorded.
4. **Still open:** conflicts the fixed checks still find, and work another skill must do.
5. **Backup:** the backup folder, and that it can be deleted once the user is happy.

Then suggest `/ml-status` to see the project after the changes.

## The spec file

`reference/spec_fields.md` describes the shared lists, including `superseded_by` on decisions and `retired_assumptions`. `scripts/ml_common.py` (with the fixed checks, `consistency_checks`) and `scripts/nb_common.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
