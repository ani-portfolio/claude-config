#!/usr/bin/env python3
"""Write notebooks/ab06_readout.ipynb from the readout answers, and run it once.

The notebook restates the primary result from the analysis, checks the standard ship rule (the primary improved
significantly and no guardrail was breached), turns the effect into business units, and holds the recommendation,
risks, and next steps as decision cells. Running it writes the readout section of ab_spec.yaml and the stakeholder
document docs/ab_readout.md. The decision itself stays with the decision owner.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml [--overwrite | --new-version] [--no-run]

answers.yaml keys: status (draft or final), recommendation (ship, do_not_ship, iterate, rerun), recommendation_reason,
scale_units (for example 100000, to state the effect per 100,000 units, or null), scale_label, risks, next_steps,
assumptions, open_questions, decisions.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ab-readout"
STAGE = "readout"


def build(answers, spec, scripts_dir, nb_name):
    cells = nb_common.setup_cells(
        STAGE, "Readout",
        "The result of the test for the decision owner: what moved, how sure we are, whether the data can be trusted, "
        "and a recommendation. The numbers come from `ab04_analysis.ipynb`; nothing is recomputed here except the "
        "business impact.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the recommendation, risks, and next steps. Edit "
        "them, then choose Run All, which writes `docs/ab_readout.md`.",
        scripts_dir, nb_name)
    cells.append(code('''
import pandas as pd
from IPython.display import display
an, fr, val = spec["analysis"], spec["framing"], spec["validation"]
p = an["primary"]
blockers = ab_common.analysis_blockers(spec)
if blockers:
    print("Warning: " + "; ".join(blockers) + ".")
'''))
    cells.append(md("## The Result\n\nCopied from the analysis section of the spec."))
    cells.append(code('''
rows = [dict(e, role="primary") for e in (p.get("all_treatments") or [p])]
rows += [dict(e, role="secondary") for e in an.get("secondary") or []] + [dict(e, role="guardrail") for e in an.get("guardrails") or []]
display(pd.DataFrame(rows)[["role", "metric", "treatment", "control_mean", "treatment_mean", "diff", "diff_ci",
                            "rel_lift", "rel_ci", "p_value", "result_label"]])
print(f"Data validation: {val.get('result')}.", *(val.get("failures") or []), *(val.get("warnings") or []), sep="\\n  ")
'''))
    cells.append(md("## Ship Rule\n\nThe rule agreed in the framing, and the standard check of it: the primary metric "
                    "improved significantly for every treatment and no guardrail was breached. If the agreed rule says "
                    "something else, the recommendation below follows the agreed rule."))
    cells.append(code('''
good = [("improvement" in e["result_label"]) for e in (p.get("all_treatments") or [p])]
breached = [g["metric"] for g in an.get("guardrails") or [] if g.get("breached")]
SHIP_RULE_MET = all(good) and not breached
print("Agreed rule:", fr.get("ship_rule") or "none")
print("Standard check:", "met" if SHIP_RULE_MET else "not met", f"(guardrails breached: {', '.join(breached) or 'none'})")
'''))
    cells.append(md("## Business Impact\n\nThe absolute effect scaled to a number of units the business recognises, with "
                    "its interval. `SCALE_UNITS` of None skips this."))
    cells.append(code(f"# Decision\nSCALE_UNITS = {answers.get('scale_units')!r}\nSCALE_LABEL = {answers.get('scale_label') or 'units'!r}",
                      decision=True))
    cells.append(code('''
business_impact = None
if SCALE_UNITS:
    lo, hi = p["diff_ci"]
    business_impact = (f"the change in {p['metric']} equals about {p['diff'] * SCALE_UNITS:+,.0f} per {SCALE_UNITS:,} "
                       f"{SCALE_LABEL} (interval {lo * SCALE_UNITS:+,.0f} to {hi * SCALE_UNITS:+,.0f}).")
    print(business_impact)
'''))
    cells.append(md("## Recommendation\n\n`RECOMMENDATION` is ship, do_not_ship, iterate, or rerun. Give the reason in a "
                    "sentence or two, tied to the ship rule. Risks and caveats include anything a reader could misread, such "
                    "as an unknown planned split, a short test, or columns recorded after assignment."))
    cells.append(code(
        "# Decision\n"
        f"RECOMMENDATION = {answers.get('recommendation')!r}\n"
        f"RECOMMENDATION_REASON = {answers.get('recommendation_reason')!r}\n"
        f"risks = {lit(list(answers.get('risks') or []))}\n"
        f"next_steps = {lit(list(answers.get('next_steps') or []))}", decision=True))
    cells.append(md("## Assumptions, Open Questions, and Decisions"))
    cells.append(code(
        "# Decision\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []))}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []))}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []))}", decision=True))
    cells.append(md("## Status\n\n`final` once the decision owner has seen it, `draft` until then."))
    cells.append(code(f"# Decision\nSTATUS = {answers.get('status') or 'draft'!r}", decision=True))
    cells.append(md("## Write to the Spec and the Readout"))
    cells.append(code(f'''
assert STATUS in ("draft", "final"), "STATUS must be draft or final"
assert RECOMMENDATION in (None, "ship", "do_not_ship", "iterate", "rerun"), "RECOMMENDATION must be ship, do_not_ship, iterate, or rerun"
readout = {{"status": STATUS, "result_label": p["result_label"], "ship_rule_met": SHIP_RULE_MET,
           "recommendation": RECOMMENDATION, "recommendation_reason": RECOMMENDATION_REASON,
           "business_impact": business_impact, "risks": risks, "next_steps": next_steps}}
errors, _, _ = nb_common.write_stage(SPEC_PATH, "readout", readout, "{SKILL}", assumptions, open_questions, decisions,
                                     replace=True)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
import readout_doc
print("Wrote", readout_doc.write(SPEC_PATH, PATHS["docs"]))
print("Spec problems:", errors or "none")
'''))
    cells += nb_common.tracking_cells(SKILL)
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers")
    ap.add_argument("--spec")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--new-version", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    spec_path = ab_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ab_common.load_or_create(spec_path)
    if not (spec.get("analysis") or {}).get("primary"):
        sys.exit("The analysis has not run yet. Run ab-analysis first.")
    answers = ab_common.load(a.answers)
    nb_name = ab_common.notebook_name(STAGE)
    path = os.path.join(ab_common.project_paths(spec, root)["notebooks"], nb_name)
    nb = build(answers, spec, HERE, nb_name)
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    if res["status"] == "edited":
        print(f"Not written: {ab_common.relpath(path, root)} was edited after the skill wrote it. "
              f"Rerun with --overwrite to replace it, or --new-version to write {os.path.basename(res['next_version'])}.")
        sys.exit(2)
    print(f"Wrote notebook: {ab_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if not res["executed"]:
        sys.exit(1)
    ro = ab_common.load(spec_path).get("readout") or {}
    print(f"Readout: status {ro.get('status')}, recommendation {ro.get('recommendation')}, ship rule check "
          f"{'met' if ro.get('ship_rule_met') else 'not met'}. Document: docs/ab_readout.md")


if __name__ == "__main__":
    main()
