---
name: ml-status
description: Show where an ML project stands in the ML project workflow. Reads the shared ml_spec.yaml and reports each step's state (done, draft, skipped, not started), outputs that are missing, results that are out of date because something they depend on changed, open questions that block the next step, data dictionary columns still to review, whether modelling can start, and the next skill to run. Read only. Always shows the status table itself in the reply. Use when asked for the status, progress, or next step of an ML project, what is left to do, or whether a project is ready for modelling.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# ML Status

Report where an ML project stands and what to do next. This skill is a helper for the ML project skills (ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, then the modelling skills). It is not a workflow step itself: it only reads `ml_spec.yaml` and the project files, and never changes anything. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Read only.** Never edit the spec, the dictionary, or any project file, and never run another skill on the user's behalf. Suggest what to run.
2. **No raw data.** The status uses the spec and file timestamps only.
3. **Standard practice first.** When you recommend or offer options for ML experimentation design or ML system design, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
4. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

`cd` to the project root. Use `.venv/bin/python` as `PY` when it exists, otherwise `python3`. If no `ml_spec.yaml` is found, say the folder is not an ML project yet and suggest **/ml-project-setup**. If the user names another spec, pass `--spec PATH`.

## Step 2: Get the status

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/status.py
```

It prints a markdown table with every workflow step in order, then the items that need attention and the next step:

| Column | Meaning |
|---|---|
| # | The order of the step in the workflow. |
| Skill | The skill for the step, as a slash command. |
| Status | One of: **Done**, **Draft**, **Skipped** (data discovery, when the ML framing was confirmed directly), **Out of date** (finished, but something it depends on changed since, or its notebook was edited and not run again), **Next**, **Not started**, or **Not built yet** (the skill does not exist yet). The step to run next is in bold. |
| Notes | Missing outputs (the step's notebook, and the business summary or data dictionary), why a step is out of date, including a notebook edited since it last ran, how many open questions belong to the step, and "needed before modelling" on the steps that keep the modelling gate closed. |

Below the table, **Needs attention** lists what does not belong to one row: columns that may leak into the features, data dictionary columns to review, the open questions for the next step or earlier steps, and notes that may disagree between stages (found by fixed checks, with a pointer to `/ml-reconcile-notes`). **Next** names the skill to run. When a finished step is out of date, the next step is to rerun it. **Optional skills** lists the helpers that can run at any time: `/ml-reconcile-notes` and `/ml-review`, marked "not built yet" when they do not exist.

Options:

* `--explain` adds a "What it does" column. Use it when the user asks what the steps are, or seems new to the workflow.
* `--detail` prints everything as plain text, including every open question and the dictionary details. Use it when the user asks about the open questions or a specific problem.
* `--brief` prints one line. `--json` prints the same facts for other tools.

## Step 3: Report back

**The user does not see the command output.** The app collapses tool output, so the table only reaches the user if your reply contains it. Your reply must follow this exact layout:

1. If the user asked a narrower question, such as "can I start modelling?" or "what are the open questions?", answer it first in one sentence. Otherwise write nothing before the table.
2. Paste the whole output of `status.py` exactly as it was printed: the project line, the markdown table with every row, **Needs attention** (when present), **Next**, and **Optional skills**. Copy it character for character. Do not rewrite it in prose, summarise it, reorder it, merge rows, or drop rows, even when every step looks fine.
3. After it, add at most two sentences of your own, and only when they help: for example what the next skill will ask for, or that earlier answers in this conversation can be reused there.

A reply that describes the status in sentences without the table is wrong, even if every fact in it is correct.

Example of a correct reply (shortened):

```
**retention**: Predict customers who will stop ordering

| # | Skill | Status | Notes |
|---|---|---|---|
| 1 | `/ml-project-setup` | Done |  |
| 2 | `/ml-business-framing` | **Next** |  |
...

**Next:** `/ml-business-framing`: the business goal, decision, current approach, and whether ML is needed.

The business framing needs no data, so it can start before the data is in place.
```

Use `--detail` output in the reply only when the user asked about the details.
