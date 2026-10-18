---
name: ml-<skill-name>
description: <What the skill does, in one or two sentences>. Step <number> of the ML project skills. Reads the shared ml_spec.yaml, <what it reads>, and writes the <stage> section. Use when <the requests that should trigger it>.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

<!--
Template for a new ML project skill. Copy this folder, rename it to the skill name from the workflow table in
ML_CONVENTIONS.md (it starts with ml-, and the name in the frontmatter must match the folder), replace every <placeholder>, and delete this comment. Then add the folder to
_ml_shared/skills.txt and run `python3 tools/sync_ml_shared.py --fix`, which adds scripts/ml_common.py and
reference/spec_fields.md, and scripts/nb_common.py.
-->

# <Skill Title>

<One paragraph: what this stage produces and why it matters for a V1 model.>

This is step <number> of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, and ml-offline-evaluation. Every stage reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Never invent answers.** Propose inferred settings as drafts for the user to confirm. Anything unconfirmed goes in `assumptions` or `open_questions`.
2. **Never change input data.** Read from the `data_raw` folder, and write only to `data_processed`, `models`, `notebooks`, and `docs`.
3. **No raw rows.** Work with column names and summary statistics. Do not print rows or paste data values into the reply.
4. **Use the scripts.** <Main script> does the work, and spec changes go through `scripts/ml_common.py`.
5. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root and use the project's Python, `.venv/bin/python`, as `PY` for every later command. If there is no `.venv`, use `python3` and suggest the **ml-project-setup** skill. Then run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py paths
```

Read the sections this stage depends on: <for example framing, data, and eda>.

For a modelling stage (split onwards), also run the gate:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py gate
```

If it exits with code 1, tell the user which step to run first (ml-problem-framing, or ml-eda-report on the confirmed target) and stop. Continue only if the user explicitly chooses to, and record that choice in `decisions` with the reason.

## Step 2: Ask the setup questions

Ask one question at a time, one per message or AskUserQuestion call, and wait for each answer before asking the next. Take every answer the spec already gives, and do not ask about it again. Put the best option first and mark it (Recommended).

1. <Question>: <how to build the options from the spec>.

If AskUserQuestion is unavailable or the user is not present, use the recommended answers, record them as assumptions, and say so at the top of the reply.

## Step 3: Build and run the notebook

Write the answers to a scratch YAML file, then build the notebook `<NN>_<stage>.ipynb` in the `notebooks` folder with `scripts/make_notebook.py` (`PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py notebook-name <stage>` gives the name). Use `nb_common.py` for the cells (`setup_cells`, `md`, `code(..., decision=True)`), and follow the notebook rules in `ML_CONVENTIONS.md`:

1. Markdown cells hold the reasoning, risks, assumptions, and open questions.
2. Decision cells hold the answers as readable Python values.
3. Check cells compute from the data, so the user can verify each claim.
4. The last cells write the `<stage>` section with `nb_common.write_stage`, mark settled questions resolved, and call `nb_common.record_run`.

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py answers.yaml
```

The script runs the notebook once, which updates the spec. If it exits with code 2, the user edited the notebook after the skill wrote it. Ask whether to replace it (`--overwrite`), write the next version (`--new-version`), or keep the edits and run it as it is (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run <notebook>`). If a cell fails, read the error in the notebook, fix the cause, and run it again.

## Step 4: Check the spec

Run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py validate` and fix any errors in the notebook's decision cells, not in the spec.

## Step 5: Report back

Reply with these parts, in this order:

1. **Files written:** the notebook first, then data files, models, and documents, with paths.
2. **Summary:** two to five sentences on the result.
3. **Problems:** anything that needs the user, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** the next skill in the workflow, and anything to resolve before running it.
