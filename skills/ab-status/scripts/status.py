#!/usr/bin/env python3
"""Where an A/B test stands: the workflow steps and their state, the outputs each step should have left, results
that are out of date, open questions that block the next step, settings that disagree between steps, the analysis
gate, and the next skill to run.

Read only: it never changes the spec or any file.

Commands (run from the project root, or anywhere inside the project):
    status.py               # a markdown table of the steps in order, then what needs attention
    status.py --explain     # the same, with a column saying what each step does
    status.py --detail      # everything as plain text, including every open question
    status.py --brief       # one line
    status.py --json        # the same facts as JSON
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common  # noqa: E402
import nb_common  # noqa: E402

STATUS_WORDS = {"done": "Done", "draft": "Draft", "skipped": "Skipped", "not started": "Not started"}
# Each stage, and the stage whose rerun makes it out of date.
CHAIN = [("framing", "design"), ("framing", "validation"), ("validation", "analysis"), ("framing", "analysis"),
         ("analysis", "segments"), ("analysis", "readout"), ("segments", "readout")]
OPTIONAL_SKILLS = [
    ("ab-reconcile-notes", "Finds notes that disagree between the steps (for example a primary metric changed after "
                           "the design), asks which way to align each one, and runs the notebooks again."),
    ("ab-sequential-monitor", "Safe looks at a running test with alpha spending or mSPRT, so stopping early does not "
                              "inflate false positives."),
    ("ab-bayesian-analysis", "Posterior lift, the probability that the treatment beats the control, and the expected "
                             "loss, as an alternative readout."),
    ("ab-review", "An independent read only audit for peeking, sample ratio problems, metrics chosen after the fact, "
                  "and adjustments on columns recorded after assignment."),
]


def _ts(value):
    try:
        return _dt.datetime.fromisoformat(str(value)).replace(tzinfo=None) if value else None
    except ValueError:
        return None


def file_time(path):
    try:
        return _dt.datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def installed_skills():
    found = set()
    for base in (os.path.dirname(os.path.dirname(os.path.realpath(HERE))), os.path.expanduser("~/.claude/skills")):
        try:
            found |= {n for n in os.listdir(base) if os.path.isfile(os.path.join(base, n, "SKILL.md"))}
        except OSError:
            pass
    return found


def expected_outputs(spec, root):
    paths = ab_common.project_paths(spec, root)
    runs = spec.get("notebooks") or {}
    j = os.path.join

    def nb(stage):
        rec = runs.get(stage) or {}
        p = j(root, rec["path"]) if rec.get("path") else j(paths["notebooks"], ab_common.notebook_name(stage))
        return (os.path.basename(p), p)

    out = {"setup": [("spec", j(root, ab_common.SPEC_NAME)), (".venv", j(root, ".venv"))],
           "framing": [nb("framing"), ("test plan", j(paths["docs"], "test_plan.md"))],
           "design": [nb("design")], "segments": [nb("segments")],
           "readout": [nb("readout"), ("readout", j(paths["docs"], "ab_readout.md"))]}
    for stage in ("validation", "analysis"):
        out[stage] = [nb(stage)] + [(os.path.basename(p), j(root, p)) for p in ((spec.get(stage) or {}).get("files") or {}).values()]
    return out


def stale_results(spec, root):
    notes = []
    runs = spec.get("notebooks") or {}
    for stage in runs:
        if nb_common.edited_since_run(spec, root, stage):
            notes.append((stage, f"`{runs[stage].get('path')}` was edited after it last ran. Run it (Run All) so the spec matches it."))
    src = (spec.get("data") or {}).get("source")
    vt = _ts((spec.get("validation") or {}).get("run_at"))
    if src and vt and (file_time(os.path.join(root, src)) or vt) > vt:
        notes.append(("validation", f"`{src}` changed after the validation ran. Rerun ab-data-validation."))
    for before, after in CHAIN:
        b, a = _ts((runs.get(before) or {}).get("run_at")), _ts((runs.get(after) or {}).get("run_at"))
        if b and a and b > a and ab_common.step_state(spec, after) in ("done", "draft"):
            notes.append((after, f"The {before} step ran again after this step. Rerun {ab_common.STAGE_SKILL[after]}."))
    return notes


def collect(spec, path):
    root = os.path.dirname(path)
    installed = installed_skills()
    nxt, nxt_skill, _ = ab_common.next_step(spec)
    order = [s[0] for s in ab_common.STEPS]
    outputs = expected_outputs(spec, root)
    steps = []
    for i, (step, skill, _, what) in enumerate(ab_common.STEPS, 1):
        state = ab_common.step_state(spec, step)
        missing = [label for label, p in outputs.get(step, []) if not os.path.exists(p)] if state == "done" else []
        steps.append({"number": i, "step": step, "skill": skill, "what": what, "state": state,
                      "missing_outputs": missing, "installed": skill in installed,
                      "optional": step in ab_common.OPTIONAL_STEPS})
    nxt_i = order.index(nxt) if nxt in order else len(order)
    blocking, later = [], []
    for q in spec.get("open_questions") or []:
        if isinstance(q, dict) and q.get("status") == "open":
            idx = order.index(q["stage"]) if q.get("stage") in order else nxt_i
            (blocking if idx <= nxt_i else later).append(q)
    val, an = spec.get("validation") or {}, spec.get("analysis") or {}
    return {"spec": path, "project": (spec.get("project") or {}).get("name"),
            "description": (spec.get("project") or {}).get("description"),
            "test_name": (spec.get("framing") or {}).get("test_name"),
            "python": ab_common.project_python(root), "steps": steps, "next": {"step": nxt, "skill": nxt_skill},
            "stale": [{"step": s, "note": n} for s, n in stale_results(spec, root)],
            "blocking_questions": blocking, "later_questions": later, "gate": ab_common.analysis_blockers(spec),
            "conflicts": ab_common.consistency_checks(spec), "validation": val.get("result"),
            "unknown_split": any(v.get("planned_share") is None for v in (spec.get("framing") or {}).get("variants") or []),
            "primary": (an.get("primary") or {}).get("result_label")}


def row_notes(r, s):
    notes = []
    if s["missing_outputs"]:
        notes.append("missing: " + ", ".join(s["missing_outputs"]))
    notes += [x["note"].split(". ")[0] for x in r["stale"] if x["step"] == s["step"]]
    n_q = sum(1 for q in r["blocking_questions"] + r["later_questions"] if q.get("stage") == s["step"])
    if n_q:
        notes.append(f"{n_q} open question" + ("s" if n_q > 1 else ""))
    if s["step"] == "validation" and r["validation"]:
        notes.append(f"result: {r['validation']}")
    if s["step"] == "analysis" and r["primary"]:
        notes.append(r["primary"])
    if s["step"] in ("framing", "validation") and s["state"] not in ("done",) and r["gate"]:
        notes.append("needed before the analysis")
    if s["optional"] and s["state"] == "not started":
        notes.append("optional")
    return "; ".join(notes)


def rerun_step(r):
    stale = {x["step"] for x in r["stale"]}
    return next((s for s in r["steps"] if s["step"] in stale and s["state"] in ("done", "skipped", "draft")), None)


def status_word(r, s):
    rerun = rerun_step(r)
    if rerun and s["step"] == rerun["step"]:
        return "**Out of date, rerun next**"
    if any(x["step"] == s["step"] for x in r["stale"]) and s["state"] in ("done", "skipped", "draft"):
        return "Out of date"
    if s["step"] == r["next"]["step"] and not rerun:
        return "**Next**" if s["installed"] else "**Next** (not built yet)"
    if not s["installed"] and s["state"] == "not started":
        return "Not built yet"
    return STATUS_WORDS.get(s["state"], s["state"])


def print_table(r, explain=False):
    title = r["test_name"] or r["project"]
    print(f"**{title}**" + (f": {r['description']}" if r["description"] else ""))
    print()
    cols = ["#", "Skill", "Status"] + (["What it does"] if explain else []) + ["Notes"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in r["steps"]:
        cells = [str(s["number"]), f"`/{s['skill']}`", status_word(r, s)]
        if explain:
            cells.append(s["what"][0].upper() + s["what"][1:])
        cells.append(row_notes(r, s).replace("|", "\\|"))
        print("| " + " | ".join(cells) + " |")
    print()
    attention = []
    if r["unknown_split"]:
        attention.append("The planned split is not known for every variant, so the sample ratio cannot be tested.")
    if r["blocking_questions"]:
        attention.append(f"{len(r['blocking_questions'])} open question" + ("s" if len(r["blocking_questions"]) > 1 else "")
                         + " for the next step or earlier steps: " + ", ".join(str(q.get("id")) for q in r["blocking_questions"])
                         + ". See them with --detail.")
    shown = [c for c in r["conflicts"] if c["check"] != "open question"]
    if shown:
        attention.append(f"{len(shown)} setting{'s' if len(shown) > 1 else ''} may disagree between steps ("
                         + ", ".join(dict.fromkeys(c["check"] for c in shown)) + "). See them with --detail.")
    if r["gate"] and r["next"]["step"] in ("analysis", "segments", "readout"):
        attention.append("The analysis is blocked: " + "; ".join(r["gate"]) + ".")
    link = None
    try:
        import tracking
        notes, link = tracking.status_notes(ab_common.load(r["spec"]),
                                            "Set it up with `setup_ab.py <folder> --add-tracking`.")
        attention += notes
    except ImportError:
        pass
    if attention:
        print("**Needs attention**")
        print()
        for a in attention:
            print(f"* {a}")
        print()
    if link:
        print(link)
        print()
    print_next(r)
    installed = installed_skills()
    print()
    print("**Optional skills** (run at any time)")
    print()
    for name, what in OPTIONAL_SKILLS:
        print(f"* `/{name}`" + ("" if name in installed else " (not built yet)") + f": {what}")


def print_next(r):
    n = r["next"]
    nxt = next((s for s in r["steps"] if s["step"] == n["step"]), None)
    rerun = rerun_step(r)
    if rerun:
        why = next(x["note"] for x in r["stale"] if x["step"] == rerun["step"]).split(". ")[0]
        if "was edited after it last ran" in why:
            print(f"**Next:** open `{why.split('`')[1]}` and choose Run All, so the spec matches your edits.")
        else:
            print(f"**Next:** rerun `/{rerun['skill']}`, because {why[0].lower() + why[1:]}.")
    elif not n["step"]:
        print("**Next:** every step is done.")
    elif nxt and not nxt["installed"]:
        print(f"**Next:** `/{n['skill']}`, which is not built yet.")
    else:
        extra = " It is optional: skip it if the test has already run and no power check is wanted." if n["step"] == "design" else \
            " It is optional: skip it if no segments were planned in the framing." if n["step"] == "segments" else ""
        print(f"**Next:** `/{n['skill']}`: {nxt['what'] if nxt else ''}.{extra}")


def print_text(r):
    print(f"Test: {r['test_name'] or r['project']}" + (f": {r['description']}" if r["description"] else ""))
    print(f"Spec: {r['spec']}")
    print(f"Project Python: {r['python'] or 'none (.venv missing, run ab-project-setup)'}")
    print("\nSteps:")
    for s in r["steps"]:
        print(f"  {s['number']}. {s['skill']:22} {s['state']:12} {s['what']}")
        if s["missing_outputs"]:
            print(f"      missing: {', '.join(s['missing_outputs'])}")
    print("\nOut of date:" + ("" if r["stale"] else " nothing found."))
    for s in r["stale"]:
        print(f"  [{s['step']}] {s['note']}")
    print(f"\nOpen questions for the next step or earlier steps: {len(r['blocking_questions'])}")
    for q in r["blocking_questions"]:
        print(f"  {q.get('id')} [{q.get('stage')}] {q.get('question')}")
    if r["later_questions"]:
        print(f"Open questions for later steps: {len(r['later_questions'])}")
        for q in r["later_questions"]:
            print(f"  {q.get('id')} [{q.get('stage')}] {q.get('question')}")
    print(f"\nSettings that may disagree between steps: {len(r['conflicts'])}")
    for c in r["conflicts"]:
        print(f"  [{c['check']}] {c['detail']}")
    print("\nAnalysis gate: " + ("open." if not r["gate"] else "closed: " + "; ".join(r["gate"]) + "."))
    n = r["next"]
    print("Next: " + (f"run /{n['skill']} ({n['step']})." if n["step"] else "every step is done."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    path = ab_common.spec_path(a.spec)
    if not os.path.exists(path):
        root, why = ab_common.find_project_root()
        print(f"No A/B test spec found ({why if not root else 'no ab_spec.yaml in ' + root}). Start with /ab-project-setup.")
        sys.exit(1)
    r = collect(ab_common.load(path), path)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
    elif a.brief:
        done = sum(1 for s in r["steps"] if s["state"] in ("done", "skipped"))
        n = r["next"]
        print(f"{r['test_name'] or r['project']}: {done} of {len(r['steps'])} steps done. "
              + (f"Next: /{n['skill']}." if n["step"] else "All steps done.")
              + (f" ({len(r['stale'])} out of date)" if r["stale"] else ""))
    elif a.detail:
        print_text(r)
    else:
        print_table(r, explain=a.explain)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
