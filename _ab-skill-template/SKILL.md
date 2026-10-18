---
name: ab-<skill-name>
description: <What the skill does, in one or two sentences>. Step <number> of the A/B testing skills. Reads the shared ab_spec.yaml, <what it reads>, and writes the <stage> section. Use when <the requests that should trigger it>.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

<!--
Template for a new A/B testing skill. Copy this folder, rename it to the skill name (it starts with ab-, and the name
in the frontmatter must match the folder), replace every <placeholder>, and delete this comment. Then add the folder to
_ab_shared/skills.txt (and to STATS_SKILLS in tools/sync_ab_shared.py if it computes statistics) and run
`python3 tools/sync_ab_shared.py --fix`, which adds scripts/ab_common.py, scripts/nb_common.py, and
reference/spec_fields.md (and scripts/ab_core.py). See AB_CONVENTIONS.md, "Adding a new skill".
-->

# <Skill Title>

<One paragraph: what this step produces and why it matters for the decision.>

This is step <number> of the A/B testing skills: ab-project-setup, ab-test-framing, ab-test-design, ab-data-validation, ab-analysis, ab-segment-analysis, and ab-readout. Every step reads `ab_spec.yaml` first and writes only its own section. The shared conventions are in `AB_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Never invent answers.** Propose inferred settings as drafts for the user to confirm. Anything unconfirmed goes in `assumptions` or `open_questions`.
2. **Results come last.** <Say whether this step may look at metric differences between variants. Only ab-analysis and later steps do.>
3. **Only columns recorded before assignment adjust or split the effect.**
4. **No raw rows.** Report counts, rates, and test statistics.
5. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, and statistics come from `scripts/ab_core.py`.
6. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. If there is no `ab_spec.yaml`, suggest **ab-project-setup**. Then run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status` and read the sections this step depends on: <for example framing, data, and validation>.

For a step after the validation, also run the gate:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py gate
```

If it exits with code 1, say which step to run first and stop. Continue only if the user explicitly chooses to, and record that choice in `decisions` with the reason.

## Step 2: Ask the questions

Ask one question at a time, one per message or AskUserQuestion call, and wait for each answer before asking the next. Take every answer the spec already gives. Put the best option first and mark it (Recommended).

1. <Question>: <how to build the options from the spec>.

If AskUserQuestion is unavailable or the user is not present, use the recommended answers, record them as assumptions, and say so at the top of the reply.

## Step 3: Build and run the notebook

Write the answers to a scratch YAML file, then run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py answers.yaml
```

The notebook `<abNN_stage>.ipynb` (`PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py notebook-name <stage>`) has markdown cells for the reasoning, *Decision* cells for the answers, check cells that compute from the data, and a last cell that writes the `<stage>` section with `nb_common.write_stage` and calls `nb_common.record_run`. Exit code 2 means the user edited the notebook: ask whether to replace it (`--overwrite`), write the next version (`--new-version`), or keep the edits and run it (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run <notebook>`).

## Step 4: Report back

1. **Files written:** the notebook first, then data files and documents, with paths.
2. **Summary:** two to five sentences on the result.
3. **Problems:** anything that needs the user, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** the next skill in the workflow, and anything to resolve before running it.
