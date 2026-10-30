#!/usr/bin/env python3
"""Write notebooks/ab01_test_framing.ipynb from the test framing answers, and run it once.

Markdown cells hold the reasoning, the risks found, the assumptions, and the open questions. Code cells tagged
"decision" hold the answers, so they can be edited. Running the notebook checks the answers, writes them to the
framing section of ab_spec.yaml, and writes the test plan docs/test_plan.md for stakeholders.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml                 # build and run; refuses to replace an edited notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write ab01_test_framing_v2.ipynb instead
    make_notebook.py --from-spec                  # rebuild from the framing section already in the spec

answers.yaml (see templates/framing_answers.example.yaml): status, framing (the fields), reasoning (one paragraph per
section), risks, assumptions, open_questions, decisions.
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

SKILL = "ab-test-framing"
STAGE = "framing"

SECTIONS = [
    # (key in reasoning, heading, fields, what to check)
    ("hypothesis", "Hypothesis and Change", ["test_name", "test_state", "hypothesis", "change"],
     "A hypothesis names the change, the metric, the expected direction, and the population. test_state is planned, "
     "running, or finished."),
    ("decision", "Decision and Owner", ["decision", "owner"],
     "Someone must act on the result. If no decision depends on it, the test is not worth running."),
    ("unit", "Unit and Population", ["unit", "population"],
     "The unit is what was randomised (user, session, store). The analysis must count the same unit, or the "
     "confidence intervals will be too narrow."),
    ("variants", "Variants and Planned Split", ["variants"],
     "Exactly one variant has role control. planned_share is the share of units each variant was meant to get; the "
     "sample ratio check in ab03_validation tests the data against it."),
    ("metrics", "Metrics", ["primary_metric", "secondary_metrics", "guardrail_metrics"],
     "One primary metric decides the test. type is binary, continuous, or count; direction is the direction that is "
     "good. Guardrails carry max_harm, the largest harm that is acceptable."),
    ("success", "Smallest Effect Worth Acting On and Ship Rule", ["mde", "ship_rule"],
     "Agreed before anyone sees results, so the result cannot move the goalposts."),
    ("scope", "Out of Scope", ["out_of_scope"], "What this test will not answer."),
]


def build(answers, spec, scripts_dir, nb_name):
    fr = dict(answers.get("framing") or {})
    fr.pop("risks", None)
    fr.pop("status", None)
    reasoning = answers.get("reasoning") or {}
    status = answers.get("status") or "draft"
    cells = nb_common.setup_cells(
        STAGE, "Test Framing",
        "What the test changes, what decision it informs, how success is measured, and the rule for shipping, all "
        "agreed before looking at results. Nothing here needs the data.\n\n"
        "**How to use this notebook.** The code cells marked *Decision* hold the answers. Edit them, then choose "
        "Run All. The last cells check the answers, write them to `ab_spec.yaml`, and update the test plan "
        "`docs/test_plan.md`. Column names are matched to these metrics later, in `ab03_validation.ipynb`.",
        scripts_dir, nb_name)
    cells.append(code("framing = {}  # filled in by the decision cells below"))
    for key, heading, fields, hint in SECTIONS:
        text = f"## {heading}\n\n"
        if reasoning.get(key):
            text += reasoning[key].strip() + "\n\n"
        text += f"*{hint}*"
        cells.append(md(text))
        body = ",\n".join(f"    {f!r}: {lit(fr.get(f), indent=4)}" for f in fields)
        cells.append(code(f"# Decision\nframing.update({{\n{body},\n}})", decision=True))

    cells.append(md("## Risks Found During Framing\n\nEach problem found when the answers were checked, and how it was "
                    "resolved. A problem that is still open belongs in the open questions below as well."))
    cells.append(code(f"# Decision\nrisks = {lit(list(answers.get('risks') or []))}", decision=True))
    cells.append(md("## Assumptions, Open Questions, and Decisions\n\nAssumptions are taken as true without "
                    "confirmation. Each open question names the stage that needs the answer (framing, design, "
                    "validation, analysis, segments, or readout). Decisions record choices and why they were made."))
    cells.append(code(
        "# Decision\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []))}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []))}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []))}", decision=True))
    cells.append(md("## Status\n\n`confirmed` once the decision owner agrees with the framing, `draft` until then. The "
                    "analysis stays blocked until it is confirmed."))
    cells.append(code(f"# Decision\nSTATUS = {status!r}", decision=True))

    cells.append(md("## Check the Answers\n\nThe same rules `ab_common.validate` applies to the spec."))
    cells.append(code('''
problems = []
variants = framing.get("variants") or []
roles = [v.get("role") for v in variants]
if variants and roles.count("control") != 1:
    problems.append(f"There must be exactly one control variant, there are {roles.count('control')}.")
shares = [v.get("planned_share") for v in variants]
if shares and all(isinstance(s, (int, float)) for s in shares) and abs(sum(shares) - 1) > 0.001:
    problems.append(f"The planned shares add up to {sum(shares):g}, not 1.")
if shares and any(s is None for s in shares):
    print("Some planned shares are not known. The sample ratio check needs them; it is an open question for validation.")
metrics = [framing.get("primary_metric") or {}] + list(framing.get("secondary_metrics") or []) + list(framing.get("guardrail_metrics") or [])
for m in metrics:
    if m.get("type") not in (None, "binary", "continuous", "count"):
        problems.append(f"{m.get('name')}: type must be binary, continuous, or count.")
    if m.get("direction") not in (None, "increase", "decrease"):
        problems.append(f"{m.get('name')}: direction must be increase or decrease.")
assert STATUS in ("draft", "confirmed"), "STATUS must be draft or confirmed"
assert framing.get("test_state") in (None, "planned", "running", "finished"), "test_state must be planned, running, or finished"
assert not problems, "\\n".join(problems)
print("The answers pass the checks.")
'''))
    cells.append(md("## Write to the Spec\n\nWrites the `framing` section of `ab_spec.yaml` and records this run."))
    cells.append(code(f'''
result = nb_common.write_stage(SPEC_PATH, "framing", {{**framing, "risks": risks, "status": STATUS}}, "{SKILL}",
                               assumptions, open_questions, decisions)
nb_common.show_result(*result, STATUS)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
'''))
    cells.append(md("## Test Plan\n\nWrites `docs/test_plan.md`, a short plan without code, from the spec. Edit this "
                    "notebook, not that file, so the two stay the same."))
    cells.append(code('''
import test_plan
print("Wrote", test_plan.write(SPEC_PATH, PATHS["docs"]))
'''))
    nxt = ("For a test that has not run yet, size it with `/ab-test-design`. For a finished test, put the data in the "
           "raw data folder and run `/ab-data-validation` (`/ab-test-design` can still report the power the test had).")
    cells.append(md(f"## Next Step\n\n{nxt} Check progress with `/ab-status`."))
    cells += nb_common.tracking_cells(SKILL)
    return nb_common.build(cells, spec, SKILL, STAGE)


def answers_from_spec(spec):
    fr = dict(spec.get("framing") or {})
    return {"status": fr.pop("status", "draft"), "risks": fr.pop("risks", []), "framing": fr, "reasoning": {},
            "assumptions": [], "open_questions": [], "decisions": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--from-spec", action="store_true", help="Build from the framing section of the spec")
    ap.add_argument("--spec", help="Spec path (default: ab_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()

    spec_path = ab_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ab_common.load_or_create(spec_path)
    if a.from_spec:
        answers = answers_from_spec(spec)
    elif a.answers:
        answers = ab_common.load(a.answers)
    else:
        ap.error("give an answers file or --from-spec")
    nb_name = ab_common.notebook_name(STAGE)
    path = os.path.join(ab_common.project_paths(spec, root)["notebooks"], nb_name)
    if not os.path.exists(spec_path):
        ab_common.save(spec_path, spec)
    nb = build(answers, spec, HERE, nb_name)
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    if res["status"] == "edited":
        print(f"Not written: {ab_common.relpath(path, root)} was edited after the skill wrote it. "
              f"Rerun with --overwrite to replace it, or --new-version to write {os.path.basename(res['next_version'])}.")
        sys.exit(2)
    print(f"Wrote notebook: {ab_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if res["executed"]:
        st = (ab_common.load(spec_path).get("framing") or {}).get("status")
        print(f"Spec: framing section written, status {st}. Test plan: docs/test_plan.md")
    else:
        print("Spec: not updated. Open the notebook and choose Run All to write the spec.")
        sys.exit(1)


if __name__ == "__main__":
    main()
