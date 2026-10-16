---
name: ab-test-design
description: Size an A/B test, step 2 of the A/B testing skills. Sets the significance level, power, and sidedness, turns the smallest effect worth acting on into a minimum detectable effect, and computes the sample size per variant for the planned split and the run time in whole weeks from daily traffic, with a power curve. For a test that has already run, it reports the power the test had and the smallest effect it could detect, from the unit counts only. Also records whether the test will be looked at while running and any variance reduction plan. Writes notebooks/ab02_design.ipynb; running it writes the design section of ab_spec.yaml, which the analysis reuses. Optional for finished tests. Use before launching a test, or when asked about sample size, power, minimum detectable effect, test duration, or whether a test was big enough.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Test Design

Decide how big the test must be before it runs, so a real effect of the size that matters is likely to be detected and a result that is not significant means something. For a finished test, check whether it was big enough. The output is `notebooks/ab02_design.ipynb`, and running it writes the `design` section of `ab_spec.yaml`. The analysis reuses its significance level, sidedness, and minimum effect.

This is step 2 of the A/B testing skills: ab-project-setup, ab-test-framing, **ab-test-design**, ab-data-validation, ab-analysis, ab-segment-analysis, and ab-readout. It is optional for a test that has already run. The shared conventions are in `AB_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Never invent the baseline or the minimum effect.** They come from the user, from data before the test, or from the framing's `mde`. For a finished test with no baseline, the control mean from the validated data is used, and the notebook says so.
2. **The minimum effect has a reason.** "Any improvement" cannot be sized. Push back and ask what change would pay for the work or the risk.
3. **Power for a finished test uses counts, not results.** It answers "could this test have detected the minimum effect?", never "was the observed effect real?".
4. **Peeking needs a plan.** If the user will look at a running test more than once, record `looks: sequential` and say that a fixed analysis overstates significance in that case.
5. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook. Do not edit the `design` section by hand.
6. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the spec

`cd` to the root (`python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root`) and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status` and `show --section framing`. The framing needs a control and a primary metric; if not, run **ab-test-framing** first. Note `framing.test_state`.

**Skipping.** For a finished test where the user wants no power check, mark the step skipped and move on:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py --skip --reason "<why>"
```

## Step 2: Ask the design questions

One or two AskUserQuestion rounds, best guess first and marked (Recommended), skipping what the spec answers:

1. **Baseline:** the control rate or mean before the test, and its source. For continuous metrics, also the standard deviation. Offer "Use the control mean from the test data" for a finished test.
2. **Minimum effect:** turn the framing's `mde` into a relative or absolute number, and offer two or three alternatives with what each costs in sample size.
3. **Error rates:** alpha 0.05, power 0.8, two sided (Recommended), with one sided only when no effect and harm lead to the same decision.
4. **Traffic and looks** (planned or running tests): eligible units per day, and whether the test will be looked at before the end.

If AskUserQuestion is unavailable, use alpha 0.05, power 0.8, two sided, the framing's `mde`, and the control mean for a finished test, write the design as `draft`, and say so at the top of the reply.

## Step 3: Build and run the notebook

Write the answers to a scratch YAML (`status`, `alpha`, `power`, `sides`, `baseline`, `baseline_sd`, `mde_abs` or `mde_rel`, `units_per_day`, `looks`, `variance_reduction`, `adjust_for_treatments`, `reasoning` with `error_rates`, `effect`, and `traffic` paragraphs, `assumptions`, `open_questions`, `decisions`), then:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
```

It prints the sample size, the duration, and for a finished test the power it had and the smallest detectable effect. Exit code 2 means the notebook was edited since the skill wrote it: ask whether to replace it (`--overwrite`), write a new version (`--new-version`), or keep the edits and run it (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/ab02_design.ipynb`).

## Step 4: Report back

1. **Files written:** `notebooks/ab02_design.ipynb` and `ab_spec.yaml`.
2. **Summary:** the minimum effect, the sample size and run time, or for a finished test its power and smallest detectable effect, in two or three sentences.
3. **Problems:** an unequal split that makes the control the bottleneck, a run time under a week, low power, or planned peeking, one sentence each.
4. **Open questions:** one line each.
5. **Next step:** for a planned test, launch it and come back with the data for **ab-data-validation**. For a finished test, run **ab-data-validation** (or **ab-analysis** if validation has run).
