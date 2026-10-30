---
name: ab-status
description: Show where an A/B test stands in the A/B testing workflow. Reads the shared ab_spec.yaml and reports each step's state (done, draft, skipped, not started), outputs that are missing, results that are out of date because an earlier step ran again or a notebook was edited, open questions that block the next step, settings that disagree between steps, whether the analysis gate is open, and the next skill to run. Read only. Always shows the status table itself in the reply. Use when asked for the status, progress, or next step of an A/B test or experiment analysis.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# A/B Test Status

Report where an A/B test stands and what to do next. This is a helper for the A/B testing skills (ab-project-setup, ab-test-framing, ab-test-design, ab-data-validation, ab-analysis, ab-segment-analysis, and ab-readout), not a step itself: it only reads `ab_spec.yaml` and the project files. The shared conventions are in `AB_CONVENTIONS.md` in the skills folder.

## Ground rules

1. **Read only.** Never edit the spec or any project file, and never run another skill on the user's behalf. Suggest what to run.
2. **No raw data.** The status uses the spec and file timestamps only.
3. **Standard practice first.** When you recommend or offer options for A/B test design or analysis, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
4. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root
```

`cd` to the project root. Use `.venv/bin/python` as `PY` when it exists, otherwise `python3`. If there is no `ab_spec.yaml`, say the folder has no A/B test yet and suggest **/ab-project-setup**. If the folder has `ml_spec.yaml` instead, mention `/ml-status` too.

## Step 2: Get the status

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/status.py
```

It prints a markdown table of every step in order, then **Needs attention**, **Next**, and **Optional skills**. Status words: **Done**, **Draft**, **Skipped** (the optional design or segments step), **Out of date**, **Next**, **Not started**, and **Not built yet**. The notes column shows missing outputs, why a step is out of date, open questions, the validation result, the primary result label, and "needed before the analysis" on the steps that keep the gate closed.

Options: `--explain` adds a "What it does" column (use it when the user is new to the workflow), `--detail` prints everything as text, including every open question and setting that disagrees (use it when asked about them), `--brief` prints one line, and `--json` prints the same facts as JSON.

## Step 3: Report back

**The user does not see the command output.** The reply must follow this layout:

1. If the user asked a narrower question, such as "can I analyse yet?", answer it first in one sentence. Otherwise write nothing before the table.
2. Paste the whole output of `status.py` exactly as printed: the title line, the table with every row, **Needs attention** (when present), **Next**, and **Optional skills**. Do not rewrite, summarise, reorder, or drop rows.
3. After it, add at most two sentences of your own, and only when they help.

A reply that describes the status in sentences without the table is wrong, even if every fact in it is correct.
