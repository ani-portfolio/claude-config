---
name: ab-test-framing
description: Frame an A/B test before anyone looks at its results, step 1 of the A/B testing skills. Covers the hypothesis and the change, the decision the result informs and who owns it, the unit of randomisation and the population, the variants with their planned split, one primary metric with secondary and guardrail metrics, the smallest effect worth acting on, and the ship rule. Asks short rounds of questions, pushes back on vague hypotheses, several primary metrics, and unit mismatches, and writes notebooks/ab01_test_framing.ipynb; running it writes the framing section of the shared ab_spec.yaml and a test plan for stakeholders. Use when starting an experiment or campaign test, or when asked to define the hypothesis, metrics, or success criteria of a test.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Test Framing

Agree on what the test is for before looking at any result: the hypothesis, the decision it informs, how success is measured, and the rule for shipping. The output is the notebook `notebooks/ab01_test_framing.ipynb`. Its markdown cells hold the reasoning, risks, assumptions, and open questions, and its *Decision* cells hold the answers. Running it writes the `framing` section of `ab_spec.yaml` and the test plan `docs/test_plan.md`.

This is step 1 of the A/B testing skills: ab-project-setup, **ab-test-framing**, ab-test-design, ab-data-validation, ab-analysis, ab-segment-analysis, and ab-readout. Every step reads `ab_spec.yaml` first and writes only its own section. The shared conventions are in `AB_CONVENTIONS.md` in the claude-skills folder.

Nothing here needs the data, and no results are looked at. Column names are matched to the metrics in **ab-data-validation**.

## Ground rules

1. **Never invent answers.** Propose inferred answers as drafts for the user to confirm. Anything unconfirmed goes in `assumptions` or `open_questions`.
2. **Never look at results.** Do not compute group rates or differences while framing, even when the data is in the folder. Framing after seeing results is how goalposts move. Reading column names to propose drafts is fine.
3. **The planned split comes from how the test was set up, not from the data.** If the user does not know it, record it as an open question for validation, and do not fill it from the observed counts.
4. **Push back.** One primary metric. A unit that matches the randomisation. A decision that depends on the result. A minimum effect with a reason. Explain each objection in one or two sentences and suggest a better answer.
5. **Keep the interview short.** At most three rounds. Lead with proposed answers. Skip what the request, the spec, or the conversation already answers.
6. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, and the notebook writes the spec. Do not edit the `framing` section by hand.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. If there is no `ab_spec.yaml`, suggest **ab-project-setup** first (in a folder set up by ml-project-setup it only adds the spec). Then read `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status` and `show --section framing`.

* **`framing.status` is `not_started` or `draft`:** continue, and only ask about what is empty.
* **`framing.status` is `confirmed`:** show a short summary and ask whether to revise it. A revision after the analysis has run must be recorded in `decisions` with the reason, because it changes what the result means.

If `notebooks/ab01_test_framing.ipynb` exists, read its decision cells; the user may have edited them since the last run. If the project also has `ml_spec.yaml` with a business section, reuse its goal, decision, and users as drafts.

## Step 2: Gather context

Silently: the request, `project.description`, the conversation, and the data file names and column names in the raw data folder (`PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py columns <file>`, which prints names and types only). `reference/question_bank.md` lists every field, what a good answer looks like, and the red flags.

## Step 3: Ask in up to three rounds

If AskUserQuestion is unavailable or the user is not present, skip to Step 5 with the drafts as `status: draft`, and list every unconfirmed field under `open_questions`.

**Round 1 (plain text).** In one numbered message, with your draft after each, so the user can reply "yes" or correct it:

1. What changed between the groups, and what do you expect it to do (the hypothesis)?
2. What decision depends on the result, and who makes it?
3. What was randomised (user, session, store), and who was eligible?
4. What were the groups called, and what share of units was each **planned** to get?

**Round 2 (AskUserQuestion, up to four questions).** Best guess first, marked (Recommended).

1. **Primary metric:** offer the two or three candidates that fit the hypothesis, each with its type and good direction.
2. **Guardrails** (multiSelect): harms worth watching, and "None needed".
3. **Smallest effect worth acting on:** two or three options with a reason (for example the cost of the change), and "Not decided".
4. **Test state:** planned, running, or finished, if not clear.

**Round 3 (only if needed).** The ship rule, secondary metrics, and anything still unclear.

## Step 4: Pressure test and confirm

Check the answers against the red flags in `reference/question_bank.md`. Most common:

1. The hypothesis has no direction or no mechanism.
2. More than one primary metric, or a primary that can move without the business improving.
3. The analysis unit differs from the randomisation unit.
4. The planned split is unknown or was read off the data.
5. Columns that the treatment can change (exposure counts, activity after assignment) are being suggested as segments or controls. Note them for ab-data-validation.
6. A running test that will be checked repeatedly. Suggest ab-test-design with `looks: sequential`.

Tell the user about each problem in plain language with a suggested fix. Then show a compact summary table (hypothesis, decision and owner, unit, variants and planned split, primary metric, guardrails, minimum effect, ship rule) and ask with AskUserQuestion: "Does this test framing look right?" with "Confirm it" (Recommended), "Change something", and "Keep it as a draft".

## Step 5: Build and run the notebook

1. Write the answers to a YAML file in the scratch folder, with the keys in `templates/framing_answers.example.yaml`. `status` is `confirmed` only when the user confirmed and no required field is empty. Include `reasoning` (one short paragraph per section, with what was pushed back on), `risks`, `assumptions`, `open_questions` (each with its `stage`), and `decisions`.
2. Run it:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

3. **Exit code 2** means the user edited the notebook since the skill wrote it. Ask whether to replace it (`--overwrite`), write a new version (`--new-version`), or keep the edits and run it as it is (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/ab01_test_framing.ipynb`).
4. If a cell fails, read the error in the notebook, fix the answer, and run again.
5. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py validate` and fix any errors in the answers.

## Step 6: Report back

1. **Files written:** `notebooks/ab01_test_framing.ipynb`, `docs/test_plan.md`, and `ab_spec.yaml`.
2. **Summary:** two or three sentences on the hypothesis, the primary metric, and the status.
3. **Problems:** each problem from the pressure test and how it was resolved, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** the user can change any answer in the decision cells and choose Run All. For a planned or running test, run **ab-test-design**. For a finished test, put the data in the raw data folder and run **ab-data-validation**; ab-test-design is optional and reports the power the test had.
