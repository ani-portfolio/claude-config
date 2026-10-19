---
name: ml-autopilot
description: Run an ML project end to end as an agent, a helper for the ML project skills. Clarifies the goal with the user first (the mode, the two anchors, and the run's budget), then runs every stage skill without a person, stopping at three kinds of checkpoint (the business framing, the ML framing, and every opening of the test set), where it writes docs/checkpoint.md, sends a push notification, and stops. After the first evaluation it improves the model in rounds of experiments, each one change, run in parallel by ml-experimenter agents in their own git worktrees, judged on CV by the iteration bar, and logged to MLflow as candidates; the best one that passes is adopted and the rest are rejected. Resumes from the spec each time it is started. Use when asked to run the ML workflow end to end, to run it unattended or as an agent, to run experiments automatically, or to resume an autopilot run.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/../ml-iterate/scripts/*)
---

# Autopilot

This skill drives the ML project skills from start to finish so that a person only has to answer what only a person can answer. It does no stage work itself. Every stage is run by its own skill, called with the argument `autopilot`, which makes the skill take its proposed answers and record them as drafts. The person is asked for the goal at the start and reviews the work at checkpoints. Everything in between runs without them.

It is a helper for the ML project skills, not a workflow step. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder, under "Running without a person".

## Ground rules

1. **The spec decides what comes next.** Run `autopilot.py next` before every action and do what it says. Never skip a checkpoint it returns.
2. **Never confirm a framing without the person.** The business and ML framing are drafted unattended and confirmed only after the person reviews them at their checkpoint, in this session.
3. **Never open the test set without a checkpoint.** Evaluation runs only after the person agreed at a `test_v1` or `test_final` checkpoint.
4. **Experiments never touch the project.** Each runs in its own worktree. Only `autopilot.py round close` merges anything back, and only an experiment that passed the iteration bar.
5. **Stop when the limits say so.** The rounds stop at `max_rounds`, after `max_consecutive_fails` rounds in a row with nothing adopted, or when the diagnosis finds nothing worth trying.
6. **Use the scripts.** `scripts/autopilot.py` writes the `autopilot` section. `iterate.py` (in ml-iterate) writes the `iterations` section, and `scripts/workspace.py` makes and removes worktrees. Do not edit either section by hand.
7. **No raw data in the reply or in the checkpoint page.** Use metrics, counts, and dates.
8. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the next action

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. If there is no project yet, run **ml-project-setup** first, with the user. `IT` below is `${CLAUDE_SKILL_DIR}/../ml-iterate/scripts/iterate.py`. Then:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py next
```

`next` prints one JSON object with an `action` and a `why`. Do what the action says (Step 2 onwards), then run `next` again. Repeat until the action is `checkpoint`, `done`, or `manual_iteration`, or until something fails.

| Action | What to do |
|---|---|
| `ask_goal` | Step 2 |
| `run_stage` | Step 3 |
| `checkpoint` | Step 4: write it, notify, and stop |
| `review` | Step 5: the person is back |
| `open_round` | Step 6 |
| `await_experiments` | Step 6, part 4: wait for the agents, or record the ones that stopped |
| `close_round` | Step 6, part 5 |
| `manual_iteration` | An iteration opened with /ml-iterate is open. Say so and stop; the person finishes it first. |
| `done` | Step 7 |

## Step 2: Clarify the goal (the person is present)

This is the only part of the run before the first checkpoint that needs the person. Ask one question at a time with AskUserQuestion, and wait for each answer.

1. **The mode and the two anchors.** Invoke **ml-business-framing** with the Skill tool, without `autopilot`. It asks the mode question (a clear business goal, or exploratory), then the two anchors for that mode, then saves them as a draft and stops. Do not let it go further.
2. **The run's budget.** Ask how many experiments each round may try in parallel (2 recommended, at most 5), and the most rounds (3 recommended, never more than `iterations.bar.max_iterations`). Then record them:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py start --experiments-per-round 2 --max-rounds 3
   ```

3. Say plainly what happens next: the agent profiles the data, drafts the rest of the business framing, and stops at the first checkpoint with a push notification.

## Step 3: Run a stage without the person

Invoke the stage's skill (`skill` in the JSON) with the Skill tool and the argument `autopilot`. In that mode the skill:
* asks nothing,
* takes its recommended answer to every question,
* records each one as an assumption with `source: autopilot draft`,
* never marks a framing confirmed,
* ends with the `--json` result of its script.

When `mode` is `draft` (the business or ML framing), the skill drafts the whole section from the data and leaves it as `status: draft`.

If the stage's script exits with code 1, read the error in the notebook and fix the cause once if it is clearly a mistake in the drafted answers. Otherwise stop and tell the person, without a checkpoint. Exit codes 2 and 3 mean a person edited the notebook or `src/features.py`. Never overwrite their edits: stop and say so.

When `attended` is true (setup, or an evaluation the person agreed to), run the skill normally, without `autopilot`.

## Step 4: Write a checkpoint, notify, and stop

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py checkpoint write <kind>
```

It writes `docs/checkpoint.md`:
* For the business and ML framing, it has the drafts table, the assumptions the agent made, the red flags, and what resuming does.
* For the test set, it has the CV leaderboard, the V1 bar on CV, the rounds, and how many times the test set has been opened.

Then send the `notification` text from the JSON with the PushNotification tool. If the tool is unavailable, say so in the reply. Reply with a short summary and the path of the checkpoint page, then **stop**. Do not continue in the same turn.

## Step 5: Review a checkpoint (the person is back)

`next` returns `review` while a checkpoint is open. Read `docs/checkpoint.md` and show its tables in the reply.

**Business or ML framing:**
1. Ask with AskUserQuestion (`multiSelect: true`) which rows they dispute, with "None, accept all" first. Add every red flag as an option too.
2. Go through each disputed row and red flag one question at a time, with a proposed answer.
3. Invoke the stage's skill normally, without `autopilot`, and give it the reviewed answers: the accepted drafts as they are, and the corrected ones. The person has reviewed them, so the skill confirms the section (`status: confirmed`) as it would in an interview.
4. Clear the checkpoint:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py checkpoint clear <kind> --reason "reviewed: <n> rows changed"
   ```

**The test set (`test_v1` or `test_final`):** ask whether to open the test set now. Say how many times it has been opened, and that every look after the first is optimistic.
* **Yes:** clear the checkpoint, and run **ml-offline-evaluation** normally.
* **No:** run `autopilot.py stop --reason "<why>"`, and report the CV result as the outcome.

Then return to Step 1 and carry on unattended.

## Step 6: A round of experiments

**1. Diagnose.**

```bash
PY IT diagnose --json
```

Read the diagnosis with the spec's risks, decisions, and open questions, and the backlog (`autopilot.py backlog list`), so a change already tried is not proposed again. Propose changes, cheapest first. Each one is **one change to one stage**, with a one sentence hypothesis, so a result says what caused it. Use the table in ml-iterate's Step 2 for typical changes. When nothing is worth trying, run `autopilot.py stop --reason "<the honest finding>"` and go to Step 1.

**2. Add them to the backlog.** Write them to a YAML list, with `cost` 1 (a setting), 2 (a feature or a fold change), or 3 (a new model type). Then:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py backlog add <items.yaml>
```

**3. Open the round.**

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py round open
```

It takes the next K items, at most one per stage, and opens an iteration with one experiment each. That snapshots the project, branches the version (`ml/v2`), and creates the version's MLflow parent run. Then it makes one worktree per experiment, with its own MLflow parent run under the version's. The JSON lists each experiment's `workspace`, `experiment` (V2-a), `changes`, and `rerun`.

**4. Launch one ml-experimenter per experiment, in one message**, so they run in parallel. Use the Agent tool with `subagent_type: ml-experimenter`, and run each in the background. Give each one a prompt with:
* the workspace path,
* the experiment name,
* the change and the hypothesis,
* the rerun order from the JSON,
* the full path of `iterate.py` (`IT`, resolved to an absolute path),
* the line: "Work only inside the workspace. Never run ml-offline-evaluation."

A stage build in the project itself is refused while the round is open, so an agent that forgets to `cd` to its workspace fails safely.

Wait for every agent to finish. Do not poll. Then, for each experiment:

```bash
PY IT record-experiment <id> --result <workspace>/experiment_result.json
```

If an agent stopped without writing the file, record `--status error`.

**5. Close the round.**

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/autopilot.py round close
```

It picks the experiment that passed the iteration bar with the largest mean improvement. It merges that experiment's branch into the version branch, copies its processed data and models back, and closes the iteration as `kept`. MLflow promotes its runs, and the alias `champion` moves. Every other experiment is tagged `rejected` in MLflow and its worktree is removed. Its branch and its runs stay. When none passed, the round closes as `rolled_back` and the project is unchanged.

Report the round in two or three sentences, then go to Step 1.

## Step 7: Report back

Reply with these parts, in this order:

1. **Files written:** the checkpoint page if any, the notebooks rebuilt, and the version branch and tag.
2. **Summary:** the goal in one sentence, the champion with its CV score and its test score (and how many looks), and the rounds as a table (experiment, change, bar result, status). End with the reason iterating stopped.
3. **MLflow:** the project experiment link (`tracking.ui` in the spec), and that each experiment is a parent run under its version's run.
4. **Problems:** stages that failed, agents that stopped, and drafts that were never reviewed, one sentence each.
5. **Next step:** `/ml-reconcile-notes` after kept rounds, then `/ml-status`. Merging the version branch into main is the person's choice.

## Files

* `scripts/autopilot.py`: the next action, checkpoints, the backlog, rounds, and limits.
* `scripts/workspace.py`: the worktree of each experiment (create, adopt, discard, list).
* `reference/checkpoints.md`: what each checkpoint shows and what resuming does.
* `templates/backlog.example.yaml`: the format of backlog items.
* The subagent is `agents/ml-experimenter.md` in the claude-skills folder, linked into `~/.claude/agents/`.
* `scripts/ml_common.py`, `scripts/nb_common.py`, `scripts/tracking.py`, and `scripts/git_ops.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
