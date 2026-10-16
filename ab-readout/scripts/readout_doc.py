"""Write docs/ab_readout.md, the stakeholder readout of an A/B test, from ab_spec.yaml.

Called from the last cells of notebooks/ab06_readout.ipynb, so the readout always matches the spec.
"""

from __future__ import annotations

import datetime as _dt
import os

import ab_common

REC = {"ship": "Ship the change", "do_not_ship": "Do not ship the change", "iterate": "Iterate on the change and test again",
       "rerun": "Rerun the test before deciding"}


def pct(x, signed=False):
    return "not computed" if x is None else (f"{x:+.2%}" if signed else f"{x:.2%}")


def fmt_p(p):
    return "not computed" if p is None else ("below 0.0001" if p < 1e-4 else f"{p:.3g}")


def num(x, signed=False):
    if x is None:
        return "not computed"
    return f"{x:+,.4g}" if signed else f"{x:,.4g}"


def points(x, signed=False):
    """A difference between two rates, in percentage points."""
    return "not computed" if x is None else f"{x * 100:{'+' if signed else ''}.2f} points"


def _effect_row(e, binary):
    m = pct if binary else num
    dm = points if binary else num
    ci = e.get("diff_ci") or [None, None]
    rel = e.get("rel_ci") or [None, None]
    return (f"| {e.get('metric')} | {e.get('treatment')} | {m(e.get('control_mean'))} | {m(e.get('treatment_mean'))} | "
            f"{dm(e.get('diff'), True)} ({dm(ci[0], True)} to {dm(ci[1], True)}) | "
            f"{pct(e.get('rel_lift'), True)} ({pct(rel[0], True)} to {pct(rel[1], True)}) | {fmt_p(e.get('p_value'))} |")


def render(spec):
    pr, fr, val = spec.get("project") or {}, spec.get("framing") or {}, spec.get("validation") or {}
    an, seg, ro = spec.get("analysis") or {}, spec.get("segments") or {}, spec.get("readout") or {}
    types = {m.get("name"): m.get("type") for m in [fr.get("primary_metric") or {}] + list(fr.get("secondary_metrics") or [])
             + list(fr.get("guardrail_metrics") or []) if isinstance(m, dict)}
    p = an.get("primary") or {}
    conf = f"{1 - (an.get('alpha') or 0.05):.0%}"
    L = [f"# A/B Test Readout: {fr.get('test_name') or pr.get('name')}", "",
         f"**Status:** {(ro.get('status') or 'draft').capitalize()}  ",
         f"**Last updated:** {_dt.date.today():%Y-%m-%d}  ",
         f"**Decision owner:** {fr.get('owner') or 'not named'}  ",
         "**Source:** `notebooks/ab06_readout.ipynb`, from `ab_spec.yaml`. Edit the notebooks, not this file.", "",
         "## Summary", ""]
    if p:
        L.append(f"{fr.get('test_name') or 'The test'} compared {p.get('treatment')} with the control on "
                 f"{p.get('metric')}. It moved from {pct(p.get('control_mean')) if types.get(p.get('metric')) == 'binary' else num(p.get('control_mean'))} "
                 f"to {pct(p.get('treatment_mean')) if types.get(p.get('metric')) == 'binary' else num(p.get('treatment_mean'))}, a relative change of "
                 f"{pct(p.get('rel_lift'), True)} ({conf} interval {pct((p.get('rel_ci') or [None])[0], True)} to "
                 f"{pct((p.get('rel_ci') or [None, None])[1], True)}). The result: {p.get('result_label')}.")
    if ro.get("business_impact"):
        L += ["", f"In business terms: {ro['business_impact']}"]
    if ro.get("recommendation"):
        L += ["", f"**Recommendation:** {REC.get(ro['recommendation'], ro['recommendation'])}. {ro.get('recommendation_reason') or ''}".rstrip()]
    L += ["", "The decision stays with the decision owner.", "",
          "## What Was Tested", "", f"**Hypothesis:** {fr.get('hypothesis') or 'not stated'}", "",
          f"**Change:** {fr.get('change') or 'not stated'}", "",
          f"**Decision it informs:** {fr.get('decision') or 'not stated'}", "",
          f"**Ship rule, agreed before the results:** {fr.get('ship_rule') or 'none was agreed'}", ""]
    if ro.get("ship_rule_met") is not None:
        L += [f"The standard check of that rule (the primary improved significantly and no guardrail was breached) is "
              f"{'met' if ro['ship_rule_met'] else 'not met'}.", ""]
    L += ["## Results", "", f"Intervals are {conf} and two sided.", "",
          "| Metric | Treatment | Control value | Treatment value | Difference (interval) | Relative lift (interval) | p value |",
          "|---|---|---|---|---|---|---|"]
    prim = p.get("all_treatments") or ([p] if p else [])
    L += [_effect_row(e, types.get(e.get("metric")) == "binary") for e in prim]
    L += [_effect_row(e, types.get(e.get("metric")) == "binary") for e in an.get("secondary") or []]
    L += [_effect_row(e, types.get(e.get("metric")) == "binary") for e in an.get("guardrails") or []]
    if an.get("secondary"):
        L += ["", "Secondary metrics use Holm adjusted p values, and they explain the result rather than decide it."]
    gs = an.get("guardrails") or []
    L += ["", "## Guardrails", ""] + ([f"* {g.get('metric')}: {'breached' if g.get('breached') else 'not breached'}." for g in gs]
                                      or ["None were defined."])
    checks = [c for c in an.get("checks") or [] if c.get("metric") == p.get("metric")]
    if checks and checks[0].get("detectable_effect_rel") is not None:
        L += ["", f"With this sample, the test could detect a relative change of about {pct(checks[0]['detectable_effect_rel'])} "
              "on the primary metric with 80% power."]
    L += ["", "## Can the Data Be Trusted?", "", f"The data validation result is **{val.get('result') or 'not run'}**."]
    srm = val.get("srm") or {}
    if srm:
        L.append(f"The split between variants {'matches' if srm.get('passed') else 'does not match'} the planned split "
                 f"(sample ratio p value {fmt_p(srm.get('p_value'))}).")
    L += [f"* Failure: {x}" for x in val.get("failures") or []] + [f"* Warning: {x}" for x in val.get("warnings") or []]
    if val.get("override"):
        L.append(f"* The analysis went ahead despite a failure: {val['override'].get('reason')}")
    post = (spec.get("data") or {}).get("post_treatment_cols") or []
    if post:
        L += ["", "Columns recorded after assignment (" + ", ".join(post) + ") were not used to adjust or split the "
              "result, because the treatment can change them."]
    if seg.get("results"):
        L += ["", "## Segments", ""]
        for r in seg["results"]:
            L.append(f"* {r.get('column')}: effects {'differ' if (r.get('p_bh') or 1) < (an.get('alpha') or 0.05) else 'do not differ clearly'} "
                     f"between segments (heterogeneity p value {fmt_p(r.get('p_value'))}, after correction {fmt_p(r.get('p_bh'))})"
                     + (". Descriptive only: the column was recorded after assignment, so its segments are not comparable groups." if r.get("post_treatment") else "."))
    elif seg.get("status") == "skipped":
        L += ["", "## Segments", "", "No segments were analysed."]
    L += ["", "## Risks and Caveats", ""] + ([f"* {x}" for x in ro.get("risks") or []] or ["None recorded."])
    L += ["", "## Next Steps", ""] + ([f"* {x}" for x in ro.get("next_steps") or []] or ["None recorded."])
    oq = [q for q in spec.get("open_questions") or [] if isinstance(q, dict) and q.get("status") == "open"]
    L += ["", "## Open Questions", ""]
    L += (["| ID | Question | Stage |", "|---|---|---|"] + [f"| {q['id']} | {q['question']} | {q['stage']} |" for q in oq]) if oq else ["None."]
    L += ["", "## Assumptions", ""] + ([f"* {a}" for a in spec.get("assumptions") or []] or ["None recorded."])
    return "\n".join(L).rstrip() + "\n"


def write(spec_path, docs_dir):
    spec = ab_common.load(spec_path)
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "ab_readout.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(spec))
    return path
