"""Write docs/test_plan.md, the stakeholder version of the test framing, from ab_spec.yaml.

Called from the last cells of notebooks/ab01_test_framing.ipynb, so the plan always matches the spec. When the design
has run, its sample size and duration are included too.
"""

from __future__ import annotations

import datetime as _dt
import os

import ab_common


def _v(x, empty="Not decided yet."):
    if x is None or x == "" or x == []:
        return empty
    return str(x)


def _metric_row(role, m):
    extra = f" Largest acceptable harm: {m['max_harm']}." if m.get("max_harm") else ""
    return (f"| {role} | {_v(m.get('name'), '')} | {_v(m.get('type'), '')} | {_v(m.get('direction'), '')} | "
            f"{_v(m.get('definition'), '')}{extra} |")


def render(spec):
    pr, fr, de = spec.get("project") or {}, spec.get("framing") or {}, spec.get("design") or {}
    L = [f"# Test Plan: {fr.get('test_name') or pr.get('name')}", "",
         f"**Status:** {(fr.get('status') or 'draft').capitalize()}  ",
         f"**Test state:** {_v(fr.get('test_state'), 'not stated')}  ",
         f"**Last updated:** {_dt.date.today():%Y-%m-%d}  ",
         "**Source:** `notebooks/ab01_test_framing.ipynb`, written to `ab_spec.yaml`. Edit the notebook, not this file.",
         "", "## Hypothesis", "", _v(fr.get("hypothesis")), "",
         "## What Changes", "", _v(fr.get("change")), "",
         "## Decision", "", _v(fr.get("decision")) + (f" Decided by: {fr['owner']}." if fr.get("owner") else ""), "",
         "## Unit and Population", "",
         f"Randomised by {_v(fr.get('unit'), 'an unstated unit')}. Population: {_v(fr.get('population'), 'not stated.')}", "",
         "## Variants", "", "| Variant | Role | Planned share |", "|---|---|---|"]
    for v in fr.get("variants") or []:
        s = v.get("planned_share")
        L.append(f"| {v.get('name')} | {v.get('role')} | {f'{s:.1%}' if isinstance(s, (int, float)) else 'not known'} |")
    if not fr.get("variants"):
        L.append("| Not decided yet. | | |")
    L += ["", "## Metrics", "", "| Role | Metric | Type | Good direction | Definition |", "|---|---|---|---|---|"]
    pm = fr.get("primary_metric") or {}
    if pm.get("name"):
        L.append(_metric_row("Primary", pm))
    L += [_metric_row("Secondary", m) for m in fr.get("secondary_metrics") or [] if isinstance(m, dict)]
    L += [_metric_row("Guardrail", m) for m in fr.get("guardrail_metrics") or [] if isinstance(m, dict)]
    L += ["", "## Success", "", "| Item | Value |", "|---|---|",
          f"| Smallest effect worth acting on | {_v(fr.get('mde'))} |",
          f"| Ship rule | {_v(fr.get('ship_rule'))} |", ""]
    if de.get("status") in ("draft", "confirmed"):
        ss = de.get("sample_size") or {}
        L += ["## Design", "", "| Item | Value |", "|---|---|",
              f"| Significance level and power | {de.get('alpha')} and {de.get('power')}, {de.get('sides')} sided |",
              f"| Minimum detectable effect | {de.get('mde_abs')} absolute ({de.get('mde_rel')} relative) |",
              f"| Sample size | {ss.get('total'):,} units in total |" if ss.get("total") else "| Sample size | Not computed |",
              f"| Duration | {de.get('duration_days')} days |" if de.get("duration_days") else "| Duration | Not computed |", ""]
    L += ["## Out of Scope", ""] + ([f"* {x}" for x in fr.get("out_of_scope") or []] or ["Nothing listed."]) + [""]
    risks = [r for r in fr.get("risks") or [] if isinstance(r, dict)]
    L += ["## Risks Found During Framing", ""] + ([f"* {r.get('problem')} {r.get('resolution') or ''}".rstrip()
                                                   for r in risks] or ["None were found."]) + [""]
    L += ["## Assumptions", ""] + ([f"* {a}" for a in spec.get("assumptions") or []] or ["None recorded."]) + [""]
    oq = [q for q in spec.get("open_questions") or [] if isinstance(q, dict) and q.get("status") == "open"]
    L += ["## Open Questions", ""]
    L += (["| ID | Question | Needed by stage |", "|---|---|---|"]
          + [f"| {q.get('id')} | {q.get('question')} | {q.get('stage')} |" for q in oq]) if oq else ["None."]
    return "\n".join(L).rstrip() + "\n"


def write(spec_path, docs_dir):
    spec = ab_common.load(spec_path)
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "test_plan.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(spec))
    return path
