#!/usr/bin/env python3
"""Where an ML project stands: the workflow steps and their state, the outputs each step should have left,
results that are out of date, open questions that block the next step, data dictionary items to review,
the modelling gate, and the next skill to run.

Read only: it never changes the spec or any file.

Commands (run from the project root, or anywhere inside the project):
    status.py               # a markdown table of the steps in order, with status and notes, then what needs attention
    status.py --explain     # the same, with a column saying what each step does
    status.py --detail      # everything, including every open question and the dictionary details
    status.py --brief       # one line: progress and the next skill
    status.py --json        # the same facts as JSON, for other tools
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402

# Skill names used in history entries by earlier versions of the skills.
SKILL_ALIASES = {
    "ml-project-setup": ["project-setup"],
    "ml-business-framing": ["business-framing", "problem-framing"],
    "ml-data-discovery": ["data-discovery"],
    "ml-problem-framing": ["ml-framing", "problem-framing"],
    "ml-eda-report": ["eda-report"],
}


def _ts(value):
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(str(value)).replace(tzinfo=None)
    except ValueError:
        return None


def last_write(spec, skill, contains=None):
    """The time of the last history entry by a skill (or its older names), optionally only for actions that
    contain some text."""
    names = {skill, *SKILL_ALIASES.get(skill, [])}
    best = None
    for h in spec.get("history") or []:
        if not isinstance(h, dict) or h.get("skill") not in names:
            continue
        if contains and contains not in str(h.get("action", "")):
            continue
        t = _ts(h.get("at"))
        if t and (best is None or t > best):
            best = t
    return best


def file_time(path):
    try:
        return _dt.datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def expected_outputs(spec, root):
    """Files each step should have produced, as {step: [(label, path)]}."""
    paths = ml_common.project_paths(spec, root)
    disc = spec.get("discovery") or {}
    eda = spec.get("eda") or {}
    runs = spec.get("notebooks") or {}
    j = lambda *p: os.path.join(*p)

    def nb(stage):
        # The notebook that last ran for the stage (it may be a _v2), else the standard name.
        rec = runs.get(stage) or {}
        p = j(root, rec["path"]) if rec.get("path") else j(paths["notebooks"], ml_common.notebook_name(stage))
        return (os.path.basename(p), p)

    out = {
        "setup": [("spec", j(root, ml_common.SPEC_NAME)), (".venv", j(root, ".venv")), ("CLAUDE.md", j(root, "CLAUDE.md"))],
        "business": [nb("business"), ("business summary", j(paths["docs"], "business_framing.md"))],
        "discovery": [nb("discovery"),
                      ("data dictionary", j(root, disc["dictionary"]) if disc.get("dictionary") else j(paths["docs"], "data_dictionary.yaml"))],
        "framing": [nb("framing")],
        "eda": [(os.path.basename(eda["report"]), j(root, eda["report"]))] if eda.get("report") else [],
    }
    for stage in ("split", "features", "baseline", "training", "evaluation"):
        out[stage] = [nb(stage)]
    for stage in ("split", "features", "baseline", "training", "evaluation"):
        for name, p in ((spec.get(stage) or {}).get("files") or {}).items():
            out[stage].append((os.path.basename(p), j(root, p)))
    # The system design writes no notebook; its record is the design document.
    sysd = spec.get("system") or {}
    out["system"] = [("system design", j(root, sysd["doc"]) if sysd.get("doc") else j(paths["docs"], "system_design.md"))]
    code = ((spec.get("features") or {}).get("code") or {})
    for key in ("features", "custom"):
        if code.get(key):
            out["features"].append((os.path.basename(code[key]), j(root, code[key])))
    return out


def edited_notebooks(spec, root):
    """Stages whose notebook decision cells were edited after the notebook last ran."""
    return [stage for stage in (spec.get("notebooks") or {}) if nb_common.edited_since_run(spec, root, stage)]


def stale_results(spec, root):
    """Results that were produced before something they depend on changed."""
    notes = []
    disc, eda = spec.get("discovery") or {}, spec.get("eda") or {}
    bus_t = last_write(spec, "ml-business-framing")
    fr_t = last_write(spec, "ml-problem-framing")
    disc_t = _ts(disc.get("run_at"))
    eda_t = _ts(eda.get("run_at"))
    fr_status = (spec.get("framing") or {}).get("status")

    for stage in edited_notebooks(spec, root):
        p = ((spec.get("notebooks") or {}).get(stage) or {}).get("path")
        notes.append((stage, f"`{p}` was edited after it last ran. Run it (Run All) so the spec matches it."))

    for s in disc.get("sources") or []:
        ft = file_time(os.path.join(root, s.get("path", "")))
        if disc_t and ft and ft > disc_t:
            notes.append(("discovery", f"`{s.get('path')}` changed after data discovery ran. Rerun ml-data-discovery."))
    if eda.get("source"):
        ft = file_time(os.path.join(root, eda["source"]))
        if eda_t and ft and ft > eda_t:
            notes.append(("eda", f"`{eda['source']}` changed after the EDA ran. Rerun ml-eda-report."))
    if eda_t and fr_t and fr_t > eda_t and eda.get("mode") == "targeted":
        notes.append(("eda", "The ML framing changed after the EDA ran, so the framing checks may be out of date. "
                             "Rerun ml-eda-report."))
    if fr_status == "confirmed" and disc_t and fr_t and disc_t > fr_t:
        notes.append(("framing", "Data discovery ran again after the ML framing was last updated. Check that the "
                                 "framing still fits the data, with ml-problem-framing."))
    # Only for projects framed in the current order (they record a mode); before it, discovery always came later.
    bus = spec.get("business") or {}
    if bus.get("mode") and bus.get("status") == "confirmed" and disc_t and bus_t and disc_t > bus_t:
        notes.append(("business", "Data discovery ran again after the business framing was confirmed. Check that "
                                  "the drafts it was based on still hold, with ml-business-framing."))
    if fr_status == "confirmed" and bus_t and fr_t and bus_t > fr_t:
        notes.append(("framing", "The business framing changed after the ML framing was last updated. Check that "
                                 "the ML framing still matches it, with ml-problem-framing."))
    # Modelling stages: a stage is out of date when the stage before it ran again after it.
    runs = spec.get("notebooks") or {}
    chain = [("framing", "the ML framing", "ml-problem-framing"), ("split", "the split", "ml-train-test-split"),
             ("features", "the features", "ml-feature-engineering"), ("baseline", "the baseline", "ml-baseline-model"),
             ("training", "the training", "ml-model-training"), ("evaluation", "the evaluation", "ml-offline-evaluation")]
    for (prev, prev_label, _), (stage, _, skill) in zip(chain, chain[1:]):
        before, after = _ts((runs.get(prev) or {}).get("run_at")), _ts((runs.get(stage) or {}).get("run_at"))
        if before and after and before > after and spec.get(stage):
            notes.append((stage, f"{prev_label[0].upper() + prev_label[1:]} ran again after this step. Rerun {skill}."))
    # The system design has no notebook: compare with the evaluation, the champion, and its own document.
    sysd = spec.get("system") or {}
    if sysd:
        eval_t, sys_t = _ts((runs.get("evaluation") or {}).get("run_at")), _ts(sysd.get("run_at"))
        champ = ((spec.get("training") or {}).get("champion") or {}).get("name")
        if sysd.get("champion") and champ and sysd["champion"] != champ:
            notes.append(("system", f"The champion changed from {sysd['champion']} to {champ} after the system was "
                                    "designed. Rerun ml-system-design for the parts that depend on it."))
        elif eval_t and sys_t and eval_t > sys_t:
            notes.append(("system", "The evaluation ran again after the system was designed. Rerun ml-system-design "
                                    "for the parts that depend on the champion."))
        doc = os.path.join(root, sysd.get("doc") or "")
        if sysd.get("doc_hash") and os.path.isfile(doc):
            with open(doc, encoding="utf-8") as f:
                if hashlib.sha256(f.read().encode("utf-8")).hexdigest()[:16] != sysd["doc_hash"]:
                    notes.append(("system", f"`{sysd['doc']}` was edited by hand, and the next build would replace "
                                            "it. Move the edits into `answers/11_system.yaml` and rerun ml-system-design."))
    return notes


def dictionary_risks(spec):
    """Columns the dictionary flags as possibly recorded after the outcome, or with unknown timing, that the
    framing has not excluded."""
    ds = (spec.get("discovery") or {}).get("dictionary_summary") or {}
    excluded = {e.get("column") for e in ((spec.get("data") or {}).get("excluded_cols") or []) if isinstance(e, dict)}
    target = ((spec.get("framing") or {}).get("target") or {}).get("name")
    main = (spec.get("discovery") or {}).get("main_source")
    risky = []
    for ref in ds.get("possibly_after_outcome", []) + ds.get("timing_unknown", []) + ds.get("candidate_targets", []):
        src, _, col = ref.partition(".")
        if col in excluded or col == target:
            continue
        if main and src != main:
            continue
        risky.append(ref)
    return list(dict.fromkeys(risky)), ds


def installed_skills():
    """Names of skills that exist, in the skills folder this skill lives in or in ~/.claude/skills."""
    found = set()
    here_skills = os.path.dirname(os.path.dirname(os.path.realpath(HERE)))
    for base in (here_skills, os.path.expanduser("~/.claude/skills")):
        try:
            for name in os.listdir(base):
                if os.path.isfile(os.path.join(base, name, "SKILL.md")):
                    found.add(name)
        except OSError:
            pass
    return found


STATUS_WORDS = {"done": "Done", "draft": "Draft", "skipped": "Skipped", "not started": "Not started"}


def collect(spec, path):
    root = os.path.dirname(path)
    steps = []
    installed = installed_skills()
    next_step, next_skill, _ = ml_common.next_step(spec)
    order = [s[0] for s in ml_common.STEPS]
    outputs = expected_outputs(spec, root)
    for i, (step, skill, how, what) in enumerate(ml_common.STEPS, 1):
        done = ml_common.step_done(spec, step)
        state = "done" if done else "not started"
        if step == "discovery" and done and not spec.get("discovery"):
            state = "skipped"
        elif step == "business_anchors" and not done and (spec.get("business") or {}).get("mode"):
            state = "draft"
        elif step in ("business", "framing") and not done and ((spec.get(step) or {}).get("status")) == "draft":
            state = "draft"
        missing = []
        if state == "done":
            missing = [label for label, p in outputs.get(step, []) if not os.path.exists(p)]
        steps.append({"number": i, "step": step, "skill": skill, "how": how, "what": what, "state": state,
                      "missing_outputs": missing, "installed": skill in installed})

    nxt_idx = order.index(next_step) if next_step in order else len(order)
    blocking, later = [], []
    for q in spec.get("open_questions") or []:
        if not isinstance(q, dict) or q.get("status") != "open":
            continue
        st = q.get("stage")
        idx = order.index(st) if st in order else nxt_idx
        (blocking if idx <= nxt_idx else later).append(q)

    risky, ds = dictionary_risks(spec)
    business = spec.get("business") or {}
    return {
        "spec": path,
        "project": (spec.get("project") or {}).get("name"),
        "description": (spec.get("project") or {}).get("description"),
        "python": ml_common.project_python(root),
        "steps": steps,
        "next": {"step": next_step, "skill": next_skill},
        "ml_needed": business.get("ml_needed"),
        "mode": ml_common.business_mode(spec) if business.get("mode") else None,
        "stale": [{"step": s, "note": n} for s, n in stale_results(spec, root)],
        "blocking_questions": blocking,
        "later_questions": later,
        "dictionary": {"needs_review": ds.get("needs_review", 0), "columns": ds.get("columns", 0),
                       "unexcluded_risky_columns": risky},
        "gate": ml_common.modelling_blockers(spec),
        "conflicts": ml_common.consistency_checks(spec),
        "review": review_state(spec),
        "iteration": iteration_state(spec),
        "eval_decision": ml_common.evaluation_decision(spec),
        "tracking": tracking_state(spec),
        "version": version_state(spec, root),
        "deployment_layout": deployment_layout_state(spec),
    }


def deployment_layout_state(spec):
    """'missing' when the system design is confirmed but ml-project-setup has not added the pipeline packages yet
    (paths.pipelines is empty), 'done' once it has, None before the design is confirmed."""
    if ((spec.get("system") or {}).get("status")) != "confirmed":
        return None
    return "done" if ((spec.get("paths") or {}).get("pipelines")) else "missing"


def version_state(spec, root):
    """The version being worked on (V1, V2, ...), its git branch, the branch checked out, uncommitted files, and the
    earlier versions with how each ended."""
    v = spec.get("version") or {}
    out = {"current": v.get("current") or "V1", "branch": v.get("branch") or "ml/v1",
           "history": [{"version": h.get("version"), "status": h.get("status"), "superseded_by": h.get("superseded_by")}
                       for h in v.get("history") or [] if h.get("version") != (v.get("current") or "V1")],
           "git": None}
    try:
        import git_ops
        if git_ops.is_repo(root):
            st = git_ops.status(root)
            out["git"] = {"branch": st["branch"], "last_commit": st["last_commit"], "on_main": st["on_protected"],
                          "uncommitted": len(st["uncommitted"])}
    except ImportError:
        pass
    return out


def iteration_state(spec):
    """The open iteration from the ml-iterate helper: its number, hypothesis, and the stages not yet run again."""
    it = spec.get("iterations") or {}
    e = next((x for x in it.get("entries") or [] if x.get("id") == it.get("current") and x.get("status") == "open"), None)
    if not e:
        return None
    opened, runs = _ts(e.get("opened_at")), spec.get("notebooks") or {}
    left = [s for s in e.get("rerun") or []
            if not _ts((runs.get(s) or {}).get("run_at")) or (opened and _ts(runs[s]["run_at"]) < opened)]
    return {"id": e.get("id"), "hypothesis": e.get("hypothesis"), "remaining": left}


def tracking_state(spec):
    """The project's MLflow experiment, whether the server answers, and the finished stages with no logged run."""
    tr = spec.get("tracking") or {}
    if not tr.get("experiment_id"):
        return None
    try:
        import tracking
        uri = tracking.tracking_uri(spec)
        up = tracking.reachable(uri)
        stages = tracking.LOGGED_STAGES
    except ImportError:
        uri, up, stages = tr.get("uri"), None, ()
    missing = [s for s in stages if spec.get(s) and not (spec.get(s) or {}).get("mlflow")]
    return {"experiment": tr.get("experiment"), "ui": tr.get("ui"), "uri": uri, "reachable": up, "missing": missing}


def review_state(spec):
    """The last ml-review: open counts, open blockers, and the stages that ran again since. None before a review."""
    rv = spec.get("review") or {}
    if not rv.get("run_at"):
        return None
    last = _ts(rv["run_at"])
    later = [stage for stage, rec in (spec.get("notebooks") or {}).items()
             if stage != "review" and _ts(rec.get("run_at")) and last and _ts(rec.get("run_at")) > last]
    return {"run_at": rv["run_at"], "open": rv.get("open") or {}, "open_blockers": rv.get("open_blockers") or [],
            "ran_since": later}


def row_notes(r, s):
    """The notes for one step: missing outputs, out of date results, open questions, and the modelling gate."""
    notes = []
    if s["missing_outputs"]:
        notes.append("missing: " + ", ".join(s["missing_outputs"]))
    notes += [x["note"].split(". ")[0] for x in r["stale"] if x["step"] == s["step"]]
    n_q = sum(1 for q in r["blocking_questions"] + r["later_questions"] if q.get("stage") == s["step"])
    if n_q:
        notes.append(f"{n_q} open question" + ("s" if n_q > 1 else ""))
    if s["step"] == "business" and r["ml_needed"] == "no":
        notes.append("concluded ML is not needed, so modelling stays blocked")
    if s["step"] in ("framing", "eda") and s["state"] not in ("done", "skipped") and r["gate"]:
        notes.append("needed before modelling")
    return "; ".join(notes)


def rerun_step(r):
    """The earliest finished step whose results are out of date, or None."""
    stale_steps = {x["step"] for x in r["stale"]}
    return next((s for s in r["steps"] if s["step"] in stale_steps and s["state"] in ("done", "skipped")), None)


def status_word(r, s):
    rerun = rerun_step(r)
    if rerun and s["step"] == rerun["step"]:
        return "**Out of date, rerun next**"
    if any(x["step"] == s["step"] for x in r["stale"]) and s["state"] in ("done", "skipped"):
        return "Out of date"
    if s["step"] == r["next"]["step"] and not rerun:
        return "**Next**" if s["installed"] else "**Next** (not built yet)"
    if not s["installed"] and s["state"] == "not started":
        return "Not built yet"
    return STATUS_WORDS.get(s["state"], s["state"])


def print_table(r, explain=False):
    print(f"**{r['project']}**" + (f": {r['description']}" if r["description"] else ""))
    print()
    cols = ["#", "Skill", "Status"] + (["What it does"] if explain else []) + ["Notes"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in r["steps"]:
        cells = [str(s["number"]), f"`/{s['skill']}`" + (f" ({s['how']})" if s.get("how") else ""), status_word(r, s)]
        if explain:
            cells.append(s["what"][0].upper() + s["what"][1:])
        cells.append(row_notes(r, s).replace("|", "\\|"))
        print("| " + " | ".join(cells) + " |")
    print()
    attention = []
    itr = r.get("iteration")
    if itr:
        attention.append(f"Iteration {itr['id']} is open: {itr['hypothesis']} Still to rerun: "
                         + (", ".join(f"`/{ml_common.STAGE_SKILL[s]}`" for s in itr["remaining"]) if itr["remaining"]
                            else "none, so close it with `/ml-iterate`") + ".")
    d = r["dictionary"]
    if d["unexcluded_risky_columns"]:
        attention.append("Possible leakage: " + ", ".join(f"`{c}`" for c in d["unexcluded_risky_columns"])
                         + " may be recorded after the outcome (or have unknown timing) and are not excluded yet. "
                           "Settle them in /ml-problem-framing.")
    if d["needs_review"]:
        attention.append(f"The data dictionary has {d['needs_review']} of {d['columns']} columns to review.")
    if r["blocking_questions"]:
        attention.append(f"{len(r['blocking_questions'])} open question"
                         + ("s" if len(r["blocking_questions"]) > 1 else "")
                         + " for the next step or earlier steps: "
                         + ", ".join(str(q.get("id")) for q in r["blocking_questions"]) + ". See them with --detail.")
    shown = [c for c in r["conflicts"] if c["check"] != "open question"]   # open questions are listed above
    if shown:
        n_c = len(shown)
        attention.append(f"{n_c} note{'s' if n_c > 1 else ''} may disagree between stages ("
                         + ", ".join(dict.fromkeys(c["check"] for c in shown))
                         + "). Align them with /ml-reconcile-notes.")
    rv = r.get("review")
    if rv:
        o = rv["open"]
        if o.get("blocker"):
            attention.append(f"The last review ({rv['run_at'][:10]}) has {o['blocker']} open blocker"
                             + ("s" if o["blocker"] > 1 else "") + ": "
                             + "; ".join(b["title"] for b in rv["open_blockers"][:3])
                             + ". See `notebooks/10_review.ipynb`.")
        if rv["ran_since"]:
            attention.append("Stages ran again after the last review (" + ", ".join(rv["ran_since"])
                             + "). Run /ml-review again before relying on the results.")
    tk = r.get("tracking")
    if not tk and any(s["step"] in ("split", "features", "baseline", "training", "evaluation") and s["state"] == "done"
                      for s in r["steps"]):
        attention.append("MLflow tracking is not set up for this project, so its runs are not logged. Set it up with "
                         "`/ml-project-setup` (`setup_project.py <folder> --add-tracking`).")
    if tk and tk["reachable"] is False:
        attention.append(f"The MLflow server at {tk['uri']} is not reachable, so new runs are not logged. Start it with "
                         "`python3 ~/Projects/claude-config/skills/tools/mlflow_server.py start`.")
    if tk and tk["missing"]:
        attention.append("No MLflow run is logged for " + ", ".join(tk["missing"])
                         + ". A run is logged the next time each of those notebooks runs with the server up.")
    vs = r.get("version") or {}
    g = vs.get("git")
    if g is None and vs:
        attention.append("The project is not a git repository, so the stage builds are not committed. Set it up with "
                         "`/ml-project-setup` (`setup_project.py <folder> --add-git`).")
    elif g and g["on_main"]:
        attention.append("The project is on the main branch. The next stage build moves to "
                         f"`{vs['branch']}` before it commits; the skills never commit to main.")
    elif g and g["branch"] != vs["branch"]:
        attention.append(f"The checked out branch is `{g['branch']}`, but the spec works on {vs['current']} "
                         f"(`{vs['branch']}`). Stage builds commit to the branch that is checked out.")
    if attention:
        print("**Needs attention**")
        print()
        for a in attention:
            print(f"* {a}")
        print()
    if vs:
        earlier = ", ".join(f"{h['version']} {str(h['status']).replace('_', ' ')}"
                            + (f" (built on by {h['superseded_by']})" if h.get("superseded_by") else "")
                            for h in vs["history"])
        line = f"**Version:** {vs['current']} on branch `{vs['branch']}`"
        if g and g.get("last_commit"):
            line += f", last commit `{g['last_commit']}`"
        if g and g.get("uncommitted"):
            line += f", {g['uncommitted']} uncommitted file{'s' if g['uncommitted'] > 1 else ''}"
        print(line + (f". Earlier versions: {earlier}." if earlier else "."))
        print()
    if tk:
        print(f"**MLflow:** experiment `{tk['experiment']}`" + (f", {tk['ui']}" if tk.get("ui") else "") + ".")
        print()
    print_next(r)
    print_optional(r)


def print_next(r):
    n = r["next"]
    nxt = next((s for s in r["steps"] if s["step"] == n["step"]), None)
    rerun = rerun_step(r)
    if rerun:
        why = next(x["note"].split(". ")[0] for x in r["stale"] if x["step"] == rerun["step"])
        if "was edited after it last ran" in why:
            nb_path = why.split("`")[1]
            print(f"**Next:** open `{nb_path}` and choose Run All, so the spec matches your edits. Then check the "
                  "status again, since later steps may need a rerun too.")
            return
        print(f"**Next:** rerun `/{rerun['skill']}`, because {why[0].lower() + why[1:]}. Then check the status again, "
              "since later steps may need a rerun too.")
    elif not n["step"] and r.get("iteration") and r["iteration"]["remaining"]:
        s = r["iteration"]["remaining"][0]
        print(f"**Next:** `/{ml_common.STAGE_SKILL[s]}`, the next stage of iteration {r['iteration']['id']} "
              "(see `/ml-iterate`).")
    elif n["step"] == "system" and not r.get("eval_decision") and not r.get("iteration"):
        print("**Next:** record a go, iterate, or no go decision on the evaluation (its notebook's decisions cell). "
              "After a go, design the system with `/ml-system-design`; to improve the model, use `/ml-iterate`.")
    elif not n["step"] and not r.get("eval_decision") and not r.get("iteration"):
        print("**Next:** every step is done. Record a go, iterate, or no go decision on the evaluation; to improve the "
              "model, use `/ml-iterate`.")
    elif not n["step"] and r.get("deployment_layout") == "missing":
        print("**Next:** every step is done. Add the deployment layout (one package per pipeline under src/) with "
              "`/ml-project-setup` (`setup_project.py <folder> --add-deployment`), then build the pipelines.")
    elif not n["step"]:
        print("**Next:** every step is done.")
    elif r["ml_needed"] == "no" and n["step"] in ("split", "features", "baseline", "training", "evaluation", "system"):
        print("**Next:** none in the ML workflow. The business framing concluded that ML is not needed. If the "
              "question is whether a change worked, use the A/B testing skills: `/ab-project-setup`, then "
              "`/ab-test-framing`.")
    elif nxt and not nxt["installed"]:
        print(f"**Next:** `/{n['skill']}`, which is not built yet. Start it from `_ml-skill-template/` in the "
              "skills folder.")
    else:
        print(f"**Next:** `/{n['skill']}`: {nxt['what'] if nxt else ''}.")


OPTIONAL_SKILLS = [
    ("ml-reconcile-notes", "Finds notes that disagree across the stages (for example a V1 bar that names a model "
                           "that was never built), asks which way to align each one, fixes it in the notebook that "
                           "owns it, and runs the notebooks again. Run it after changing a decision."),
    ("ml-iterate", "Plans and tracks an iteration after the evaluation: diagnoses what to improve, snapshots the "
                   "current state, lists the stages to rerun in order, carries the old champion forward so it is "
                   "compared on the same folds, and guards another look at the test set."),
    ("ml-review", "An independent audit of the whole project or chosen stages for leakage, inconsistent settings, and results that "
                  "look too good, by an agent that did not build the work. Read only."),
]


def print_optional(r):
    installed = installed_skills()
    print()
    print("**Optional skills** (run at any time)")
    print()
    for name, what in OPTIONAL_SKILLS:
        print(f"* `/{name}`" + ("" if name in installed else " (not built yet)") + f": {what}")


def print_text(r):
    print(f"Project: {r['project']}" + (f": {r['description']}" if r["description"] else ""))
    print(f"Spec: {r['spec']}")
    if r.get("mode"):
        print(f"Mode: {r['mode']}")
    print(f"Project Python: {r['python'] or 'none (.venv missing, run ml-project-setup)'}")
    print()
    print("Steps:")
    for s in r["steps"]:
        name = s["skill"] + (f" ({s['how']})" if s.get("how") else "")
        line = f"  {s['number']:2}. {name:36} {s['state']:12} {s['what']}"
        print(line)
        if s["missing_outputs"]:
            print(f"      missing: {', '.join(s['missing_outputs'])}")
    print()
    if r["ml_needed"] == "no":
        print("Note: the business framing concluded that ML is not needed, so the modelling skills stay blocked.")
    if r["stale"]:
        print("Out of date:")
        for s in r["stale"]:
            print(f"  [{s['step']}] {s['note']}")
    else:
        print("Out of date: nothing found.")
    print()
    print(f"Open questions for the next step or earlier steps: {len(r['blocking_questions'])}")
    for q in r["blocking_questions"]:
        print(f"  {q.get('id')} [{q.get('stage')}] {q.get('question')}")
    if r["later_questions"]:
        print(f"Open questions for later steps: {len(r['later_questions'])}")
        for q in r["later_questions"]:
            print(f"  {q.get('id')} [{q.get('stage')}] {q.get('question')}")
    d = r["dictionary"]
    if d["columns"]:
        print()
        print(f"Data dictionary: {d['needs_review']} of {d['columns']} columns need review.")
        if d["unexcluded_risky_columns"]:
            print("  Not yet excluded, but possibly recorded after the outcome or with unknown timing: "
                  + ", ".join(d["unexcluded_risky_columns"]))
    print()
    print(f"Notes that may disagree between stages: {len(r['conflicts'])}" + (" (align them with /ml-reconcile-notes)" if r["conflicts"] else ""))
    for c in r["conflicts"]:
        print(f"  [{c['check']}] {c['detail']}")
    print()
    print("Modelling gate: " + ("open, modelling can start." if not r["gate"] else "closed: " + "; ".join(r["gate"]) + "."))
    n = r["next"]
    print("Next: " + (f"run /{n['skill']} ({n['step']})." if n["step"] else "every step is done."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--json", action="store_true", help="Print the status as JSON")
    ap.add_argument("--brief", action="store_true", help="Print one line")
    ap.add_argument("--explain", action="store_true", help="Add a column saying what each step does")
    ap.add_argument("--detail", action="store_true", help="Print everything, including every open question")
    a = ap.parse_args()

    path = ml_common.spec_path(a.spec)
    if not os.path.exists(path):
        root, why = ml_common.find_project_root()
        print(f"No ML project spec found ({why if not root else 'no ml_spec.yaml in ' + root}). "
              "Start a project with /ml-project-setup.")
        sys.exit(1)
    spec = ml_common.migrate(ml_common.load(path))
    r = collect(spec, path)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
    elif a.brief:
        done = sum(1 for s in r["steps"] if s["state"] in ("done", "skipped"))
        n = r["next"]
        extra = []
        if r["stale"]:
            extra.append(f"{len(r['stale'])} out of date")
        if r["blocking_questions"]:
            extra.append(f"{len(r['blocking_questions'])} open questions")
        print(f"{r['project']}: {done} of {len(r['steps'])} steps done. "
              + (f"Next: /{n['skill']}." if n["step"] else "All steps done.")
              + (f" ({', '.join(extra)})" if extra else ""))
    elif a.detail:
        print_text(r)
    else:
        print_table(r, explain=a.explain)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
