---
name: ab-analysis
description: Estimate the effect of an A/B test, step 4 of the A/B testing skills. For every treatment and every metric in the framing, it computes the absolute and relative lift with confidence intervals (delta method for the relative lift), the p value from a two proportion z test or Welch's t test, Holm corrections for secondary metrics and for several treatments, guardrail breaches, an optional CUPED adjustment with a column recorded before assignment, a bootstrap check for continuous metrics, the effect the test could detect, and a plain label for each result (significant and meaningful, inconclusive, no meaningful effect, or harm). Writes notebooks/ab04_analysis.ipynb; running it saves data/processed/ab_results.parquet and the analysis section of ab_spec.yaml. Use after the data validation, or when asked whether a test produced a lift, or to compute the effect, confidence interval, or significance of an experiment.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Analysis

Estimate what the treatment did, and how sure we can be. The metrics, variants, and ship rule come from the framing, the significance level and minimum effect from the design, and the column mapping from the validation, so nothing is chosen after seeing the result. The output is `notebooks/ab04_analysis.ipynb`. Running it saves every comparison to `data/processed/ab_results.parquet` and writes the `analysis` section of `ab_spec.yaml`.

This is step 4 of the A/B testing skills: ab-project-setup, ab-test-framing, ab-test-design, ab-data-validation, **ab-analysis**, ab-segment-analysis, and ab-readout. The shared conventions are in `AB_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **The framing decides.** The primary metric, the variants, and the good direction come from the framing. Do not switch the primary, the sidedness, or the significance level after seeing the result. If the user insists, record the change and the reason in `decisions`; the readout shows it.
2. **One row per unit.** The unit is what was randomised. Repeated rows are combined, and units in more than one variant are dropped, as set in the notebook.
3. **Only columns recorded before assignment adjust the effect.** CUPED with a column listed in `data.post_treatment_cols` is refused.
4. **Report intervals, not only p values.** A result that is not significant is inconclusive unless its interval rules out the minimum effect. Say how large an effect the test could have detected.
5. **No raw rows.** Report rates, means, differences, intervals, and p values.
6. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook. Do not write the `analysis` section by hand.
7. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and check the gate

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root
```

`cd` to the project root and use `.venv/bin/python` as `PY`. Run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status
PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py gate
```

If the gate exits with code 1, say which step to run first (a confirmed **ab-test-framing**, or **ab-data-validation** without a failure) and stop. Continue only if the user explicitly chooses to, and record that choice in `decisions` with the reason. Read `show --section design` and `show --section data`.

## Step 2: Settle the settings

Most settings come from the spec. Ask only about what is missing or worth a choice, one question at a time, one AskUserQuestion call per question:

1. **Minimum effect** (`mde_abs`), when the design has none: offer the framing's `mde` turned into the units of the primary metric (0.005 for half a percentage point), and "Skip the practical significance check".
2. **CUPED,** only when `data.pre_treatment_cols` has a numeric column that likely predicts the metric (for example spend before the test). "None" otherwise, and say why.
3. **Sidedness,** only when the design did not set it: two sided (Recommended) unless the decision would be the same for no effect and harm.

If AskUserQuestion is unavailable, take the spec's values (alpha 0.05, two sided, no CUPED), and say so at the top of the reply.

Write the answers to a scratch YAML (`alpha`, `sides`, `mde_abs`, `cuped_col`, `n_boot`, `reasoning` with a `settings` paragraph, `assumptions`, `open_questions`, `decisions`, `resolved_questions`). When everything comes from the spec, use `--defaults` instead.

## Step 3: Build and run the notebook

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>     # or --defaults
```

It prints the primary result and any guardrail breach. Exit code 2 means the notebook was edited since the skill wrote it: ask whether to replace it (`--overwrite`), write a new version (`--new-version`), or keep the edits and run it (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/ab04_analysis.ipynb`). If a cell fails, read the error in the notebook and fix the cause.

Then run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py show --section analysis` to read the full results.

## Step 4: Interpret

Explain the primary result in one or two plain sentences: the rate or mean in each variant, the absolute difference (in percentage points for rates) and the relative lift, each with its interval, and the label. Then:

* **Inconclusive:** say the interval still includes effects worth acting on, and give the detectable effect from the checks. Do not call it "no effect".
* **Significant but not meaningful:** say the interval does not clear the minimum effect.
* **Unequal splits:** a small control (for example 4%) gives a wide interval even with a large total. Say so when it applies.
* **Guardrails:** name any breach.
* **Secondary metrics:** they explain the primary; they do not replace it.

## Step 5: Report back

1. **Files written:** `notebooks/ab04_analysis.ipynb`, `data/processed/ab_results.parquet`, and `ab_spec.yaml`.
2. **Summary:** the primary result with its interval and label, then guardrails and secondary metrics in one sentence each.
3. **Problems:** blockers overridden, settings changed after seeing results, or checks that disagree (for example the bootstrap interval), one sentence each.
4. **Open questions:** one line each, with the stage that needs the answer.
5. **Next step:** **ab-segment-analysis** if segments were planned in the framing, otherwise **ab-readout**.
