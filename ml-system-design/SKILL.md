---
name: ml-system-design
description: Design the end to end ML system around a model that passed offline evaluation, step 12 of the ML project skills. Reads the shared ml_spec.yaml (framing, features, champion, evaluation, and the go decision), asks whether to use specific tools or a platform such as Hopsworks, then drafts and walks through twelve components one question at a time (ingestion, data validation, feature pipeline and store, training, model registry and champion/challenger gate, inference, serving, label feedback, monitoring, failure handling, orchestration, testing). Blocks designs that contradict the spec and records the reason for every judgement call. Design only, no code. Writes docs/system_design.md with a Mermaid diagram and the system section of ml_spec.yaml. Use after ml-offline-evaluation, or when asked to design, plan, or architect the production system, pipelines, feature store, serving, monitoring, or champion/challenger setup for a model.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# ML System Design

Agree on how the model becomes a running system: which pipelines exist, which tools they use, how a challenger replaces the champion, what is served, what is watched, and what happens when something breaks. The output is a design, not code. `docs/system_design.md` holds the reasoning, a Mermaid diagram, and a settings table, and the `system` section of `ml_spec.yaml` holds the settings that later build skills read. Separate build skills implement each component later.

This is step 12 of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, ml-offline-evaluation, and ml-system-design. Every stage reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder. Unlike the other steps, this one writes no notebook: the design document is its record.

## Ground rules

1. **Never invent answers.** Every component is drafted from the spec and proposed for the user to confirm. Anything unconfirmed goes in `assumptions` or `open_questions` with stage `system`.
2. **Push back.** When an answer is vague, contradicts the spec, or skips a risk, say so in one or two sentences and suggest a better answer. `design.py --check` finds the contradictions; the question bank lists the red flags.
3. **One question at a time.** Ask exactly one question per message, then wait. Never send a numbered list of questions, and never put more than one question in an AskUserQuestion call. Lead with the drafted answer.
4. **Design only.** Write no pipeline, service, or test code, and install nothing. The skill writes only `docs/system_design.md`, the `system` section, and the shared lists, all through `scripts/design.py`.
5. **Never with autopilot.** Infrastructure choices commit accounts, keys, and money, so a person always answers. When the arguments include `autopilot`, or the project root has `.ml_workspace.json`, stop and say that ml-system-design needs a person and is out of the autopilot's scope.
6. **No raw rows.** Work with the spec, column names, and summary numbers only.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use the project's Python, `.venv/bin/python`, as `PY` for every later command. If there is no `.venv`, use `python3`. If the user names another spec, pass `--spec PATH` to every command. Then run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/design.py --context
```

The context is JSON: the entry rule, the business anchors, the framing (cadence, prediction time, label delay, primary metric), the champion (name, CV std), the model inputs, the evaluation results, the previous design and its out of date parts, and the tool stacks from `reference/defaults.yaml`. Read it once; every draft below comes from it. If `existing_system_summary` is set, read that file too: components that exist already are drafted as "keep the current approach", and the question becomes whether to change them.

## Step 2: Apply the entry rule

`entry.rule` in the context decides whether the design may start:

* **go:** continue, with `entry.outcome: go`.
* **refuse:** tell the user why (no evaluation, no recorded decision, or a no go or iterate decision in a standard project) and what to run instead. Continue only if the user overrides it with a reason, then use `entry.outcome: override` and `entry.reason`.
* **exploratory_no_go:** show the business `answer_actions` and ask which answer matches the result. If its action still says to build the system (for example "serve the persistence baseline as the first champion"), continue with `entry.outcome: no_go_build` and `entry.answer` set to that answer word for word; the baseline it names becomes the first champion in the drafts. Otherwise refuse as above.
* **unclear:** the decision text could not be read. Ask the user what was decided, and suggest recording it in the evaluation notebook's decisions cell.

On a rerun with an existing design, the entry rule still applies, and `out_of_date` lists the champion dependent parts to redraft (Step 4).

## Step 3: Platform and registry

Read `reference/question_bank.md` before asking. Ask with AskUserQuestion: "Should the design name specific tools, and if so from which platform?" Offer the stacks in `defaults.yaml` (each option's description says what it covers), the recommended one first, and let Other name a platform not in the file. A named platform means `platform.kind: platform` and `platform.name`; the default stack is `kind: default`; tool neutral is `kind: neutral`. For a platform that is not in `defaults.yaml`, draft its tools from what you know and say so.

If the chosen stack has `has_registry: true`, ask next, every time, which registry decides the champion (question bank, "Registry decision"). Never take the answer from an earlier project.

## Step 4: Walk the components

Go through the twelve components in the order of `defaults.yaml`, one message each. Each message has the draft (tool and settings, from the stack and the context), one sentence on which spec fact supports it, and the question "Keep this, change it, or mark it not needed?" A component marked not needed needs a reason. Ask the follow ups the question bank lists, one at a time, only for the registry gate (margin with the CV std shown, then manual or automatic promotion), inference (batch or online, with the cadence shown), and failure handling (the fallback and the reliability target).

On a rerun, ask only about the components in `out_of_date` and anything the user wants to change. Keep every other answer from `answers/11_system.yaml`.

If AskUserQuestion is unavailable or the user is not present, stop after the drafts: write them with `status: draft` and every unconfirmed component listed in `open_questions` with stage `system`.

## Step 5: Check the answers

Write the answers to a YAML file in the scratch folder with exactly the keys in `templates/system_answers.example.yaml`, including a `reasoning` paragraph for the platform, the entry, and each component in the user's terms (what was pushed back on, and why). Then run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/design.py <answers.yaml> --check
```

* **BLOCK** lines are contradictions with the spec (a gate metric that is not the primary metric, a model input not computed at serving time, a gate margin smaller than the CV spread, batch inference more often than data arrives, a missing registry decision, an entry rule not met, a component not needed without a reason). Explain each one to the user and fix the answer. A block can only be lifted by changing the earlier stage, not by accepting it.
* **WARNING** lines are judgement calls. Ask the user about each one; if they keep the choice, add it to `warnings_accepted` with the check id and their reason.

## Step 6: Confirm

Show a compact summary: one short paragraph and a table with the platform, each component's tool (or "not needed"), the gate settings, the inference mode, and the fallback. Ask with AskUserQuestion: "Does this system design look right?" with the options "Confirm it" (Recommended), "Change something", and "Keep it as a draft". Set `status` to `confirmed` only when the user confirms and no component is missing.

## Step 7: Build

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/design.py <answers.yaml>
```

It refuses (exit 1) while any block or unaccepted warning remains. On success it writes the `system` section (with the current champion, so later changes are noticed), adds the accepted warnings and any override to `decisions`, writes `docs/system_design.md`, keeps the answers in `answers/11_system.yaml`, and commits on the version branch.

If it exits with code 2, `docs/system_design.md` was edited by hand since the skill wrote it. Ask with AskUserQuestion whether to move the edits into the answers' `reasoning` and rebuild (Recommended), or replace the file as it is with `--overwrite`.

Then run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py validate` and fix any errors in the answers, not in the spec.

## Step 8: Report back

Reply with these parts, in this order:

1. **Files written:** `docs/system_design.md`, `ml_spec.yaml` (section `system`), and `answers/11_system.yaml`.
2. **Summary:** two to four sentences on the platform, the shape of the system (batch or online, the gate, the fallback), the components marked not needed, and the status.
3. **Problems:** each block that was fixed and each warning accepted, one sentence each.
4. **Open questions:** one line each.
5. **Next step:** the components are built by separate build skills, which read the `system` section. When the champion changes (after `/ml-iterate`), `/ml-status` marks this step out of date, and rerunning this skill asks only about the champion dependent parts. To change a decision, edit `answers/11_system.yaml` and run `design.py answers/11_system.yaml` again.

## The spec file

`reference/spec_fields.md` describes every field in `ml_spec.yaml`, including the `system` section. `scripts/ml_common.py`, `nb_common.py`, `git_ops.py`, and `tracking.py` are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
