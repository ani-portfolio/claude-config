"""Write docs/business_framing.md, the stakeholder summary of the business framing, from ml_spec.yaml.

Called from the last cells of notebooks/01_business_framing.ipynb, so the summary always matches the spec.
"""

from __future__ import annotations

import datetime as _dt
import os

import ml_common

OUTPUT_USE = {
    "ranked_list": "A ranked list, acted on from the top down",
    "threshold_decision": "A yes or no decision at a cut off",
    "probability": "A probability used as a number",
    "quantity": "A predicted quantity",
    "other": "Other",
}


def _v(x, empty="Not decided yet."):
    if x is None or x == "" or x == []:
        return empty
    return str(x)


def _render_exploratory(spec, pr, b, status, ml):
    actions = [a for a in b.get("answer_actions") or [] if isinstance(a, dict)]
    L = [f"# Business Framing: {pr.get('name')}", "",
         f"**Status:** {status}  ", "**Mode:** Exploratory  ",
         f"**Last updated:** {_dt.date.today():%Y-%m-%d}  ",
         "**Source:** `notebooks/01_business_framing.ipynb`, written to `ml_spec.yaml`. Edit the notebook, not this file.", "",
         "## Summary", "",
         " ".join(x for x in [
             f"This is an exploratory project. It should find out: {b['learning_question'].rstrip('.?')}?" if b.get("learning_question") else "",
             f"It ends with one of {len(actions)} answers, each leading to a different action." if actions else "",
             f"Is ML needed: {ml.lower()}." if b.get("ml_needed") else "",
         ] if x), "",
         "## What We Want to Learn", "", _v(b.get("learning_question")), "",
         "## What Each Answer Leads To", ""]
    if actions:
        L += ["| Answer | Action |", "|---|---|"] + [f"| {a.get('answer')} | {a.get('action')} |" for a in actions]
    else:
        L += ["Not decided yet."]
    L += ["", "## Who Uses the Result", "", _v(b.get("users")), "",
          "## What to Predict", "", _v(b.get("outcome")) + " The exact target, time window, and metric are defined in the ML problem framing.", "",
          "## How Success Is Judged", "",
          _v(b.get("v1_bar_intent"), "Not decided yet.") + " There is no current approach to beat, so V1 is compared "
          "with the best simple baseline.", ""]
    return L


def render(spec):
    pr = spec.get("project") or {}
    b = spec.get("business") or {}
    c = b.get("constraints") or {}
    status = (b.get("status") or "draft").capitalize()
    ml = {"yes": "Yes", "no": "No", "unsure": "Not sure yet"}.get(b.get("ml_needed"), "Not decided yet")
    if ml_common.business_mode(b) == "exploratory":
        L = _render_exploratory(spec, pr, b, status, ml)
        return _render_common(spec, b, c, ml, L)
    L = [f"# Business Framing: {pr.get('name')}", "",
         f"**Status:** {status}  ",
         f"**Last updated:** {_dt.date.today():%Y-%m-%d}  ",
         "**Source:** `notebooks/01_business_framing.ipynb`, written to `ml_spec.yaml`. Edit the notebook, not this file.", "",
         "## Summary", "",
         " ".join(x for x in [
             f"The goal is to {b['business_goal'][0].lower() + b['business_goal'][1:].rstrip('.')}." if b.get("business_goal") else "",
             f"The output supports this decision: {b['decision_supported'].rstrip('.')}." if b.get("decision_supported") else "",
             f"Is ML needed: {ml.lower()}" + (f", because {b['ml_needed_reason'][0].lower() + b['ml_needed_reason'][1:].rstrip('.')}." if b.get("ml_needed_reason") else "."),
         ] if x), "",
         "## Business Goal", "", f"{_v(b.get('business_goal'))} The KPI is: {_v(b.get('business_kpi'), 'not decided yet.')}", "",
         "## Decision and Users", "", _v(b.get("decision_supported")) + (f" Used by: {b['users']}." if b.get("users") else "")
         + (f" Capacity: {b['capacity']}." if b.get("capacity") else ""), "",
         "## Current Approach", "", _v(b.get("current_approach")) + " This is the baseline V1 must beat.", "",
         "## What to Predict", "", _v(b.get("outcome")) + " The exact target, time window, and metric are defined in the ML problem framing.", "",
         "## How the Output Is Used", "",
         "| Item | Value |", "|---|---|",
         f"| Output use | {OUTPUT_USE.get(b.get('output_use'), _v(b.get('output_use')))} |",
         f"| What would make V1 worth using | {_v(b.get('v1_bar_intent'))} |", "",
        ]
    return _render_common(spec, b, c, ml, L)


def _render_common(spec, b, c, ml, L):
    L += ["## Is ML Needed?", "", f"{ml}." + (f" {b['ml_needed_reason']}" if b.get("ml_needed_reason") else ""), "",
          "## Constraints", ""]
    cons = [f"* **{k.replace('_', ' ').capitalize()}:** {v}" for k, v in c.items() if k != "other" and v]
    cons += [f"* {x}" for x in c.get("other") or []]
    L += (cons or ["None recorded."]) + [""]
    L += ["## Out of Scope", ""] + ([f"* {x}" for x in b.get("out_of_scope") or []] or ["Nothing listed yet."]) + [""]
    risks = b.get("risks") or []
    L += ["## Risks Found During Framing", ""]
    L += ([f"* {r.get('problem')} {r.get('resolution') or ''}".rstrip() for r in risks if isinstance(r, dict)]
          or ["None were found."]) + [""]
    L += ["## Assumptions", ""] + ([f"* {a}" for a in spec.get("assumptions") or []] or ["None recorded."]) + [""]
    oq = [q for q in spec.get("open_questions") or [] if isinstance(q, dict) and q.get("status") == "open"]
    L += ["## Open Questions", ""]
    if oq:
        L += ["| ID | Question | Needed by stage |", "|---|---|---|"]
        L += [f"| {q.get('id')} | {q.get('question')} | {q.get('stage')} |" for q in oq]
    else:
        L += ["None."]
    return "\n".join(L).rstrip() + "\n"


def write(spec_path, docs_dir):
    spec = ml_common.migrate(ml_common.load(spec_path))
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "business_framing.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(spec))
    return path
