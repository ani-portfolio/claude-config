#!/usr/bin/env python3
"""Write notebooks/01_business_framing.ipynb from the business framing answers, and run it once.

The notebook is the working document for the business framing. Markdown cells hold the reasoning, the risks
found, the assumptions, and the open questions. Code cells tagged "decision" hold the answers, so they can be
edited. Running the notebook validates the answers, writes them to the business section of ml_spec.yaml, and
writes the stakeholder summary docs/business_framing.md.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml                 # build and run; refuses to replace an edited notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write 01_business_framing_v2.ipynb instead
    make_notebook.py --from-spec                  # rebuild from the business section already in the spec

answers.yaml (see templates/business_answers.example.yaml, or business_answers_exploratory.example.yaml for an
exploratory project): status, business (the fields, with mode), reasoning (one paragraph per section), risks,
assumptions, open_questions, decisions. Before data discovery, only the mode and the two anchors are filled in.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ml-business-framing"
STAGE = "business"

MODE_SECTION = ("mode", "Project Mode", ["mode"],
                'mode is "standard" (a clear business goal and KPI) or "exploratory" (an experiment, framed by what it '
                'should teach and what each answer would lead to).')
SECTIONS = [
    # (key in reasoning, heading, fields, what to check)
    ("goal", "Business Goal", ["business_goal", "business_kpi"],
     "The goal should be a business outcome, not the model itself, and the KPI should show whether it improved."),
    ("decision", "Decision and Users", ["decision_supported", "users", "capacity"],
     "Someone must act on the output. A capacity limit (for example 500 calls a week) turns this into a ranking problem."),
    ("current_approach", "Current Approach", ["current_approach"],
     "This is the baseline V1 has to beat."),
    ("outcome", "What to Predict", ["outcome"],
     "In business terms only. The exact target, window, and columns are settled in 03_problem_framing, after data discovery."),
    ("output_use", "How the Output Is Used", ["output_use", "v1_bar_intent"],
     "output_use is one of ranked_list, threshold_decision, probability, quantity, or other. It decides the metric family."),
    ("ml_needed", "Is ML Needed?", ["ml_needed", "ml_needed_reason"],
     'ml_needed is "yes", "no", or "unsure". "no" keeps the modelling skills blocked.'),
    ("constraints", "Constraints and Scope", ["constraints", "out_of_scope"],
     "Limits on latency, explanation, fairness, and data access, and what this model will not do."),
]
# Exploratory projects have no current approach to beat. They are framed by what they should teach, and what each
# possible answer would lead to, which also says when to stop.
EXPLORATORY_SECTIONS = [
    ("learning", "What We Want to Learn", ["learning_question", "answer_actions"],
     "answer_actions is a list of {answer, action}. Every answer should lead to a different action; if they all lead "
     "to the same one, the experiment cannot change anything."),
    ("decision", "Who Uses the Result", ["users"],
     "Who reads the answer and takes the action."),
] + [s for s in SECTIONS if s[0] in ("outcome", "output_use", "ml_needed", "constraints")]


def sections_for(mode):
    return [MODE_SECTION] + (EXPLORATORY_SECTIONS if mode == "exploratory" else SECTIONS)


def build(answers, spec, scripts_dir, nb_name):
    bus = dict(answers.get("business") or {})
    bus.pop("risks", None)
    reasoning = answers.get("reasoning") or {}
    status = answers.get("status") or bus.get("status") or "draft"
    bus.pop("status", None)
    cells = nb_common.setup_cells(
        STAGE, "Business Framing",
        "What decision the model would support, how it is made today, and what a useful result looks like, or, "
        "for an exploratory project, what it should teach and what each answer would lead to. The mode and the two "
        "anchors are settled before anyone looks at the data; the rest is drafted from what data discovery found.\n\n"
        "**How to use this notebook.** The code cells marked *Decision* hold the answers. Edit them, then choose "
        "Run All. The last cells check the answers, write them to `ml_spec.yaml`, and update the stakeholder "
        "summary `docs/business_framing.md`. The ML definition of the problem (target, window, metric) comes later, "
        "in `03_problem_framing.ipynb`.",
        scripts_dir, nb_name)
    cells.append(code("business = {}  # filled in by the decision cells below"))
    for key, heading, fields, hint in sections_for(bus.get("mode")):
        text = f"## {heading}\n\n"
        if reasoning.get(key):
            text += reasoning[key].strip() + "\n\n"
        text += f"*{hint}*"
        cells.append(md(text))
        body = ",\n".join(f"    {f!r}: {lit(bus.get(f), indent=4)}" for f in fields)
        cells.append(code(f"# Decision\nbusiness.update({{\n{body},\n}})", decision=True))

    cells.append(md("## Risks Found During Framing\n\nEach problem found when the answers were checked, and how it was "
                    "resolved. A problem that is still open belongs in the open questions below as well."))
    cells.append(code(f"# Decision\nrisks = {lit(list(answers.get('risks') or []), indent=0)}", decision=True))

    cells.append(md("## Assumptions, Open Questions, and Decisions\n\nAssumptions are things taken as true without "
                    "confirmation. Each open question names the stage that needs the answer. Decisions record choices "
                    "and why they were made. They are added to the shared lists in the spec; entries already there are "
                    "not duplicated."))
    cells.append(code(
        "# Decision\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []), indent=0)}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []), indent=0)}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []), indent=0)}", decision=True))
    cells.append(md("## Status\n\n`confirmed` once the business owner has agreed with the framing, `draft` until then."))
    cells.append(code(f"# Decision\nSTATUS = {status!r}", decision=True))

    cells.append(md("## Write to the Spec\n\nChecks the answers, writes them to the `business` section of "
                    "`ml_spec.yaml`, and records this run."))
    cells.append(code(f'''
assert STATUS in ("draft", "confirmed"), "STATUS must be draft or confirmed"
assert business.get("ml_needed") in (None, "yes", "no", "unsure"), 'ml_needed must be "yes", "no", or "unsure"'
assert business.get("output_use") in (None, "ranked_list", "threshold_decision", "probability", "quantity", "other")
assert business.get("mode") in (None, "standard", "exploratory"), 'mode must be "standard" or "exploratory"'
for a in business.get("answer_actions") or []:
    assert isinstance(a, dict) and a.get("answer") and a.get("action"), "each answer_actions item needs an answer and an action"

result = nb_common.write_stage(SPEC_PATH, "business", {{**business, "risks": risks, "status": STATUS}}, "{SKILL}",
                               assumptions, open_questions, decisions)
nb_common.show_result(*result, STATUS)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
'''))
    cells.append(md("## Stakeholder Summary\n\nWrites `docs/business_framing.md`, a short summary without code, from "
                    "the spec. Edit this notebook, not that file, so the two stay the same."))
    cells.append(code('''
import business_summary
path = business_summary.write(SPEC_PATH, PATHS["docs"])
print("Wrote", path)
'''))
    cells.append(md("## Next Step\n\nIf only the mode and the two anchors are filled in, put the data in the raw data "
                    "folder, run `/ml-data-discovery`, and then come back to `/ml-business-framing` for the rest, drafted "
                    "from the data. If the framing is complete and ML is needed, continue with `/ml-problem-framing`. "
                    "If the question is whether a change worked (an experiment or a campaign with a control group), "
                    "run `/ab-project-setup` and then `/ab-test-framing` instead. Check progress with `/ml-status`."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def answers_from_spec(spec):
    bus = dict(spec.get("business") or {})
    return {"status": bus.pop("status", "draft"), "risks": bus.pop("risks", []), "business": bus, "reasoning": {},
            "assumptions": [], "open_questions": [], "decisions": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--from-spec", action="store_true", help="Build from the business section of the spec")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()

    spec_path = ml_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ml_common.load_or_create(spec_path)
    if a.from_spec:
        answers = answers_from_spec(spec)
    elif a.answers:
        answers = ml_common.load(a.answers)
    else:
        ap.error("give an answers file or --from-spec")
    nb_name = ml_common.notebook_name(STAGE)
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    if not os.path.exists(spec_path):  # the notebook writes to the spec, so it must exist
        ml_common.save(spec_path, spec)
    nb = build(answers, spec, HERE, nb_name)
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    if res["status"] == "edited":
        print(f"Not written: {ml_common.relpath(path, root)} was edited after the skill wrote it. "
              f"Rerun with --overwrite to replace it, or --new-version to write {os.path.basename(res['next_version'])}.")
        sys.exit(2)
    print(f"Wrote notebook: {ml_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if res["executed"]:
        st = (ml_common.load(spec_path).get("business") or {}).get("status")
        print(f"Spec: business section written, status {st}. Summary: docs/business_framing.md")
    else:
        print("Spec: not updated. Open the notebook and choose Run All to write the spec.")


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "business", "ml-business-framing")
