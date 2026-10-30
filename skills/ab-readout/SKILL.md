---
name: ab-readout
description: Write the readout of an A/B test for the decision owner, step 6 of the A/B testing skills. Restates the primary result from the analysis with its interval and label, checks the ship rule agreed in the framing, turns the effect into business units, summarises whether the data can be trusted (validation result, sample ratio, columns recorded after assignment, overrides), segments, risks, and next steps, and records a recommendation (ship, do not ship, iterate, or rerun) while leaving the decision to the owner. Writes notebooks/ab06_readout.ipynb; running it writes the readout section of ab_spec.yaml and docs/ab_readout.md. Use after the analysis, or when asked to summarise, write up, or present the result of an experiment.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Readout

Turn the analysis into something the decision owner can act on: what moved, how sure we are, whether the data can be trusted, and a recommendation tied to the ship rule agreed before the test. The output is `notebooks/ab06_readout.ipynb`, and running it writes the `readout` section of `ab_spec.yaml` and the stakeholder document `docs/ab_readout.md`.

This is step 6 of the A/B testing skills: ab-project-setup, ab-test-framing, ab-test-design, ab-data-validation, ab-analysis, ab-segment-analysis, and **ab-readout**. The shared conventions are in `AB_CONVENTIONS.md` in the skills folder.

## Ground rules

1. **Nothing is recomputed.** The numbers come from the analysis section. If they look wrong, fix the analysis, not the readout.
2. **The recommendation follows the agreed ship rule.** If the user wants a recommendation that the rule does not support, record the reason in `decisions`. The decision itself stays with the owner.
3. **State the caveats plainly.** An unknown planned split, a validation override, a short test, a small control group, or a result that is only inconclusive goes in `risks`.
4. **Use the scripts.** `scripts/make_notebook.py` builds and runs the notebook, and `scripts/readout_doc.py` writes the document from the spec. Do not edit `docs/ab_readout.md` by hand.
5. **Standard practice first.** When you recommend or offer options for A/B test design or analysis, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
6. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Find the project and the results

`cd` to the root (`python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root`) and use `.venv/bin/python` as `PY`. Run `PY ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status` and `show --section analysis`. If the analysis has not run, run **ab-analysis** first. Read the framing (ship rule, owner, decision), the validation (result, warnings, override), and the segments.

## Step 2: Draft the recommendation

Propose a recommendation from the ship rule and the result:

* **ship:** the primary improved significantly (and meaningfully, if a minimum effect was set), and no guardrail was breached.
* **do_not_ship:** harm on the primary or a guardrail, or an interval that rules out the minimum effect.
* **iterate:** the direction is promising but the result is inconclusive or not meaningful, and a changed treatment is worth testing.
* **rerun:** the data cannot be trusted (a validation failure that was overridden) or the test was too small to detect the minimum effect.

Ask one question at a time, one AskUserQuestion call per question, waiting for each answer: first the recommendation (your draft first, marked Recommended), then the scale for the business impact (for example per 100,000 users, or the campaign's real volume), then whether the readout is `draft` or `final`. Propose `risks` and `next_steps` from the caveats above, and let the user correct them.

If AskUserQuestion is unavailable, write the readout as `draft` with your proposed recommendation, and say so at the top of the reply.

## Step 3: Build and run the notebook

Write the answers to a scratch YAML (`status`, `recommendation`, `recommendation_reason`, `scale_units`, `scale_label`, `risks`, `next_steps`, `assumptions`, `open_questions`, `decisions`) and run:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/make_notebook.py <answers.yaml>
```

Exit code 2 means the notebook was edited since the skill wrote it: ask whether to replace it (`--overwrite`), write a new version (`--new-version`), or keep the edits and run it (`PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run notebooks/ab06_readout.ipynb`).

## Step 4: Report back

1. **Files written:** `notebooks/ab06_readout.ipynb`, `docs/ab_readout.md`, and `ab_spec.yaml`.
2. **Summary:** the primary result with its interval, the business impact, and the recommendation, in two or three sentences.
3. **Problems:** caveats a reader could miss, one sentence each.
4. **Open questions:** one line each.
5. **Next step:** share `docs/ab_readout.md` with the decision owner, then set the status to `final` in the notebook and choose Run All. Offer to publish the readout as a page if the user wants to share it.
