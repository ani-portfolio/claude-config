---
name: ml-iterate
description: Plan, track, and close an iteration of an ML project after the offline evaluation, a helper for the ML project skills. Diagnoses the evaluation and training results (test against CV, fold windows, bias, drift, weak slices, test set looks), agrees a hypothesis and the changes by stage with the user, snapshots the starting state to iterations/iter_N, records the iteration in the shared ml_spec.yaml, starts the next version (V2, V3) on its own git branch with its own MLflow parent run, lists the stages to rerun in order and hands off to each stage skill with the user's agreement, carries the previous champion forward so it is scored on the same folds, guards a second look at the test set, and compares iterations at the end. Use after /ml-offline-evaluation when the user wants to improve the model, iterate, try again, or compare iterations.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Iterate

An evaluation often shows something to improve: folds that missed a seasonal pattern, a trend that cannot follow the test period, a model that does no better than a simpler one. Iterating means changing earlier stages and running every later one again. This skill keeps that honest and on record. It says why each iteration happened, what it changed, which stages ran again, how the new champion compares with the old one on the same folds, and how many times the test set has been seen.

It is a helper for the ML project skills, not a workflow step, and it does not redo any stage itself: each stage skill still runs with its own questions. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Hand off, do not redo.** Every change is made by the stage skill that owns it, in its notebook. This skill only records, snapshots, sequences, and compares.
2. **One iteration at a time.** Open a new iteration only after the last one is closed. If one is open, resume it.
3. **Compare on CV first.** Models are compared on the CV folds. The test set is opened again only at the end, only with the user's explicit agreement, and every later look is recorded as optimistic.
4. **Keep the history.** Snapshots are never deleted by the skill. A rollback backs up the current state first.
5. **No raw data in the reply.** Use metrics, counts, and dates.
6. **Use the script.** `scripts/iterate.py` does every write to the `iterations` section. Do not edit it by hand.
7. **Never commit to main.** The script commits on the version branches only. Merging into main is the user's choice.
8. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Then run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py remaining
```

* **An iteration is open:** resume it at Step 5, with the stages `remaining` lists.
* **No evaluation yet** (the `evaluation` section has no `test_opened`): say that iterating starts from an evaluated model, and suggest the next step from `/ml-status` instead.

## Step 2: Diagnose

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py diagnose
```

It prints the champion with its CV and test scores, a table of every model's test score, interval, CV score, per fold CV scores, and bias, the validation windows, the evaluation's evidence, and flags. The flags include:

* validation windows that cover less than a seasonal year when the project mentions seasonality
* another model level with or better than the champion on test
* test scores far from CV
* drift and weak slices
* how many times the test set was opened

Read it together with the spec's risks, decisions, and open questions. Then propose changes, cheapest first. Tie each change to the stage and skill that owns it, and give its reason in one sentence. Typical changes:

| Finding | Change | Stage |
|---|---|---|
| CV folds did not predict the test score | Longer or more validation windows | split |
| The model cannot follow a trend or a shift | New trend or event features | features |
| Fixed settings that should be learned | Several versions of a feature for the model to choose from, or a new tuned setting | features, training |
| A simpler model is as good | Stronger reference, or stop here | training, or close as stopped |

Say plainly when the honest finding may be that the data cannot support more. Stopping is a valid outcome.

## Step 3: Agree the plan

Ask with AskUserQuestion, one question per call: first which changes to make (`multiSelect: true`, the recommended ones first), then, after that answer, confirm the hypothesis, one sentence on what should improve and why. Write the plan to a YAML file in the scratch folder with the keys in `templates/iteration_plan.example.yaml`.

If AskUserQuestion is unavailable or the user is not present, make no changes: show the diagnosis and the proposed plan, and stop.

## Step 4: Open the iteration

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py open <plan.yaml> --dry-run
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py open <plan.yaml>
```

It saves the starting state to `iterations/iter_<N-1>/` and records the iteration in the `iterations` section. The snapshot holds the spec, notebooks, `src`, models, docs, and the small result files; large data files are not copied, and their sizes are listed in `manifest.json`. It adds a decision and prints the rerun order: the earliest changed stage, then every later modelling stage.

**Versions.** Iteration N builds version V(N+1): the first pass is V1, iteration 1 builds V2. Opening the iteration also:

* commits the current state and tags it with the version being left (`v1`);
* creates the branch `ml/v2` from it and switches to it, so every stage build of the iteration is committed there;
* creates the version's MLflow parent run (`experiment V2`). Every stage run of the iteration is logged as its child, tagged `version = V2` and `candidate = true`. The alias `champion` stays on the previous version's model until the iteration is kept.

The script prints a line for each of these. When git or MLflow is not available, it says so and the iteration still opens.

## Step 5: Hand off stage by stage

For each stage in the rerun order:

1. **Ask first.** Ask whether to run its skill now, and remind the user of the planned change for that stage.
2. **Run the stage skill.** Invoke it with the Skill tool and follow it as usual, using the planned change as the user's answer to the matching question. The stage notebook already exists, so the skill rebuilds it or adds to it as it normally would.
3. **Carry the old champion forward (training only).** Before ml-model-training builds its answers file, run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py carry-forward
   ```

   It prints the previous champion as a model entry, named `<champion>_iter<N-1>`, with its tuned settings fixed and no tuning. Add that entry to the models, so it is scored on the same new folds and features as the new models. It also says when feature definitions changed under the same names, and when the old champion's inputs no longer exist. In that case, the comparison falls back to the recorded scores, which are not like for like.
4. **Guard the test set (evaluation only).** Before ml-offline-evaluation, ask with AskUserQuestion whether to open the test set again. Say how many times it has been opened, and that the new score will be optimistic. Offer to close the iteration on the CV result alone instead.
5. **Check progress.** After each stage, run `iterate.py remaining` to see what is left.

## Step 6: Close the iteration

Show the comparison:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py compare
```

Ask the user for the outcome with AskUserQuestion:

* **Keep the new state (Recommended when the new champion beats the carried one on CV):** `kept`.
* **Roll back to the snapshot:** `rolled_back`. It restores the snapshot after backing up the current state to `.iterate_backup/<time>/`. The split and feature tables are not in the snapshot, so the split and features notebooks must run again to rebuild them.
* **Stop iterating:** `stopped`, when the data cannot support more.

Then:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py close --outcome <outcome> --reason "<why>" --dry-run
PY ${CLAUDE_SKILL_DIR}/scripts/iterate.py close --outcome <outcome> --reason "<why>" [--restore]
```

`--restore` is required for `rolled_back`, and only after the user agreed.

Closing settles the version:

* **Kept or stopped:** the version is promoted in MLflow (its stage runs become the current ones, and the alias `champion` moves to its champion, which also keeps the alias `champion-v2`), and git tags the version `v2`. Work stays on `ml/v2`.
* **Rolled back:** the version is marked rejected in MLflow, and the runs of the restored state are made current again. Git commits the restored files on `ml/v2`, tags it `v2-rolled-back`, switches back to `ml/v1`, and commits the spec there with the record of the rollback. The branch `ml/v2` is kept for the record. Then suggest `/ml-reconcile-notes`, because notes written in the earlier iteration may now disagree with the new decisions.

## Step 7: Report back

Reply with these parts, in this order:

1. **Files written:** the snapshot folder, `ml_spec.yaml`, and any backup folder, then the version, its git branch and tag, and its MLflow parent run.
2. **Summary:** the iteration's hypothesis and changes, then the `compare` table. Add two or three sentences: whether the new champion beats the carried old one on the same folds, and the test score with how many looks it took.
3. **Problems:** stages that were not rerun, a carry forward that was not possible, and flags from the diagnosis that are still open, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** while open, the next stage skill to run. When closed, `/ml-reconcile-notes`, then `/ml-status`, or a new iteration with `/ml-iterate`.

## The spec file

`reference/spec_fields.md` describes the `iterations` and `version` sections. `scripts/ml_common.py`, `scripts/nb_common.py`, `scripts/tracking.py`, and `scripts/git_ops.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
