---
name: ml-business-framing
description: Frame the business problem behind an ML project before anyone looks at the data, step 2 of the ML project skills. Covers the business goal and KPI, the decision the output supports and who acts on it, today's approach, the outcome to predict in business terms, how the output will be used, what would make V1 worth using, whether ML is needed at all, and constraints. Asks one question at a time, each with a proposed answer, pushes back on vague answers, and writes notebooks/01_business_framing.ipynb with the reasoning and decisions; running it writes the business section of the shared ml_spec.yaml and a short summary for stakeholders. Use when starting a new model or analytics project, or when asked to scope, frame, or write up the business problem.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Business Framing

Agree on the business problem before any data work: what decision the model would support, how that decision is made today, and what a useful result looks like. The output is the notebook `notebooks/01_business_framing.ipynb`. Its markdown cells hold the reasoning for each answer, the risks found, the assumptions, and the open questions. Its *Decision* cells hold the answers, which the user can edit. Running the notebook writes the `business` section of the shared spec `ml_spec.yaml` and a short summary, `docs/business_framing.md`, for stakeholders who do not read notebooks.

This is step 2 of the ML project skills: ml-project-setup, **ml-business-framing**, ml-data-discovery, ml-problem-framing, ml-eda-report, then ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, and ml-offline-evaluation. Every skill reads `ml_spec.yaml` first and writes only its own section. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

Nothing in this skill needs the data. The exact target, unit of prediction, prediction time, and metric are settled in **ml-problem-framing**, after **ml-data-discovery** has shown what the data can support.

## Ground rules

1. **Never invent answers.** When something is inferred from files or context rather than said by the user, propose it as a draft for the user to confirm. Anything still unconfirmed goes in `assumptions` or `open_questions`, never silently into the framing.
2. **Push back.** When an answer is vague or does not connect to a decision, say so plainly, explain why in one or two sentences, and suggest a better answer. "Build a churn model" is not a business goal.
3. **Concluding that ML is not needed is a good result.** If a rule, a report, or a dashboard would serve the decision as well, say so.
4. **One question at a time.** Ask exactly one question per message, then wait for the answer before asking the next. Never send a numbered list of questions, and never put more than one question in an AskUserQuestion call. Lead with a proposed answer wherever one can be inferred. Keep the interview short by skipping anything the request, the spec, the files, or an earlier answer already settles.
5. **Stay on the business side.** Do not ask about columns, tables, or time windows. If the user brings them up, note them as assumptions for ml-problem-framing.
6. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, and the notebook writes the spec through `scripts/nb_common.py`. Do not edit the `business` section of the spec by hand.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

The root is the nearest folder with `ml_spec.yaml`, then `.venv`, then `.git`. `cd` there, and run every later command with the project's Python, `.venv/bin/python`. Call it `PY`. If the project has no `.venv`, use `python3` and suggest the **ml-project-setup** skill first. If no spec exists, create one with `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py init --name <short project name>`. If the user names another spec, pass `--spec PATH` to every command.

Read the spec with `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status` and `show --section business`:

* **`business.status` is `not_started` or `draft`:** continue with Step 2, and only ask about what is empty.
* **`business.status` is `confirmed`:** show a short summary and ask with AskUserQuestion whether to revise it or keep it. Record any revision in `decisions` with the reason. If the ML framing is already confirmed, say which parts of it the revision may affect.

If `notebooks/01_business_framing.ipynb` exists, read its decision cells too. They are the source of the answers, and the user may have edited them since the last run.

## Step 2: Gather context before asking

Do this silently and quickly, so the first questions already carry good guesses: the user's request, `project.description` in the spec (the first draft of the business goal), and anything the user points to, such as a brief, a ticket, a slide, or a previous summary like `docs/system_summary.md` from the `ml-system-summary` skill. `reference/question_bank.md` lists every field, why it matters, what a good answer looks like, and the red flags.

## Step 3: Ask one question at a time

If AskUserQuestion is unavailable or the user is not present, skip to Step 5 with the drafts written as `status: draft`, and list every unconfirmed field under `open_questions`.

Go through the questions below in order, one per message. Wait for the answer before moving on. After each answer, check it against the red flags in `reference/question_bank.md`. If it is vague or does not connect to a decision, push back with one follow up question before moving on. Skip a question when its answer is already settled, and adjust the drafts of later questions to fit what the user has said so far.

**Free text questions (plain message).** Ask each one on its own, followed by your draft answer, so the user can reply "yes" or correct it:

1. What business goal is this for, and which KPI shows whether it improved?
2. What decision or action will the output drive, and who acts on it?
3. Is there a capacity limit on that action (for example 500 calls a week)?
4. How is that decision made today?
5. In business terms, what should be predicted? For example "customers who are about to stop ordering".

**Choice questions (AskUserQuestion, one question per call).** Put the best guess first and mark it (Recommended).

6. **How the output is used:** a ranked list for a fixed capacity, a yes or no decision at a threshold, a probability used as a number, a quantity, or something else. Explain in each description which metric family it leads to (see the table in `reference/question_bank.md`).
7. **What would make V1 worth using,** stated against today's approach. Offer two or three options, for example "catches more at the same volume" or "same catch rate with less manual review".
8. **Constraints:** batch or real time, explainability, and data or attributes that may not be used. `multiSelect: true`.
9. **Who uses the output,** only when it is not clear from question 2.

**Follow ups (only if needed, still one at a time).** Ask about anything still unclear, such as out of scope items or a feedback loop (will acting on the output change the outcome it predicts?).

## Step 4: Pressure test and confirm

Check the answers against the red flags in `reference/question_bank.md`, and decide whether ML is needed (`yes`, `no`, or `unsure`, with one sentence of reasoning). The most common problems:

1. The goal is the model itself, or there is no KPI.
2. Nobody acts on the output, or the action cannot change the outcome.
3. The outcome is a weak proxy for the goal.
4. There is no way to tell whether V1 beats today's approach.
5. A rule or report would do as well, so ML is not needed.
6. Acting on the output will change future outcomes (a feedback loop), so a holdout group will be needed later.

Tell the user about each problem found, in plain language, with a suggested fix. A problem that cannot be resolved now goes in `open_questions` with the stage that needs the answer.

Then show a compact summary: one short paragraph, and a small table with the goal and KPI, decision and capacity, current approach, outcome to predict, output use, V1 bar intent, and whether ML is needed. Ask with AskUserQuestion: "Does this business framing look right?" with the options "Confirm it" (Recommended), "Change something", and "Keep it as a draft".

## Step 5: Build and run the notebook

1. Write the answers to a YAML file in the scratch folder, using the keys in `templates/business_answers.example.yaml` exactly:
   * `status`: `confirmed` only when the user confirmed in Step 4 and no required field is empty, `draft` otherwise.
   * `business`: the answers.
   * `reasoning`: one short paragraph per section, saying why each answer is what it is, in the user's terms. This is what makes the notebook readable later, so include what was pushed back on and why.
   * `risks`: each problem from the pressure test, with how it was resolved.
   * `assumptions`, `open_questions` (each with the `stage` that needs it), and `decisions` (each with a `reason`).
2. Build and run the notebook:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
   ```

   Running it checks the answers, writes the `business` section and the shared lists to the spec (entries already there are not added twice), records the run, and writes `docs/business_framing.md` from the spec.
3. **If it exits with code 2,** the user edited the notebook after the skill last wrote it. Do not overwrite it silently. Ask with AskUserQuestion:
   * "Replace it with the new answers": rerun with `--overwrite`.
   * "Write a new version": rerun with `--new-version`, which writes `01_business_framing_v2.ipynb` and leaves the edited one as it is.
   * "Keep my edits": make the changes in the existing notebook's decision cells instead, then run it with `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/01_business_framing.ipynb`.
4. If a cell fails, the notebook is saved with the outputs so far. Read the error, fix the answer that caused it, and run again.
5. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py validate`, and fix any errors in the answers, not in the spec.

## Step 6: Report back

Reply with these parts, in this order:

1. **Files written:** `notebooks/01_business_framing.ipynb`, `docs/business_framing.md`, and `ml_spec.yaml`.
2. **Summary:** two or three sentences on the business problem, whether ML looks needed, and the status (draft or confirmed).
3. **Problems:** each problem found in the pressure test and how it was resolved, one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** say that the user can change any answer in the notebook's decision cells and choose Run All to update the spec. Then, if ML is not needed, say what to do instead (for example the rule or report to build) and that the modelling skills will stay blocked. When the question is whether a change worked (an experiment, a campaign with a control group, a before and after comparison), suggest the A/B testing skills, starting with **ab-project-setup** and then **ab-test-framing**, which reuse this project's folder and `.venv`. Otherwise put the data in the `data_raw` folder and run the **ml-data-discovery** skill.

## The spec file

`reference/spec_fields.md` describes every field in `ml_spec.yaml` and which skill writes it. `scripts/ml_common.py` is the only code that should create or restructure the file, and `scripts/nb_common.py` builds and runs the notebooks. All three are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.
