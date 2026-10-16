#!/usr/bin/env python3
"""Find and fix notes that disagree across the stages of an ML project.

Notes are everything the skills wrote down along the way: reasoning paragraphs and decision cells in the stage
notebooks, and the assumptions, open questions, decisions, risks, and settings in ml_spec.yaml. A note is fixed
where it is owned, in its notebook, and the notebook is run again so the spec follows; the spec's shared lists
(decisions, open questions, assumptions) keep their history.

Commands (run from the project root with the project's Python):
    reconcile.py inventory [--out notes.json]   # every note, by stage, with where it lives and when it was written
    reconcile.py check                          # settings that disagree between stages, by fixed rules
    reconcile.py find "some text"               # the notebook cells and spec fields that contain the text
    reconcile.py apply plan.yaml [--dry-run] [--no-run]
                                                # back up, apply the approved changes, then run the edited
                                                # notebooks again, and every later notebook, in workflow order

plan.yaml:
    edits:                       # text changes in notebook cells (old must appear once in the cell)
      - {notebook: notebooks/03_problem_framing.ipynb, cell: cell-012, old: "...", new: "...", why: "..."}
    supersede_decisions:         # an old decision stays in the log, marked as replaced
      - {decision: "old decision text", superseded_by: "new decision text", reason: "...", stage: framing}
    resolve_questions:
      - {id: Q3, answer: "..."}
    retire_assumptions:          # moved from assumptions to retired_assumptions, with the reason
      - {assumption: "old text", reason: "..."}
    replace_assumptions:         # reworded in place
      - {old: "old text", new: "new text"}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402

SKILL = "ml-reconcile-notes"
ORDER = [s[0] for s in ml_common.STEPS]
# Which stage, and so which notebook, owns each spec section.
OWNER = {"project": "setup", "paths": "setup", "business": "business", "discovery": "discovery",
         "framing": "framing", "data": "framing", "eda": "eda", "split": "split", "features": "features",
         "baseline": "baseline", "training": "training", "evaluation": "evaluation"}
TEXT_SECTIONS = ("business", "framing", "data", "split", "features", "baseline", "training", "evaluation")


def notebooks(spec, root):
    """{stage: notebook path} for every stage notebook that exists, in workflow order."""
    out = {}
    runs = spec.get("notebooks") or {}
    for stage in ORDER:
        p = (runs.get(stage) or {}).get("path")
        if stage == "eda" and not p:
            rep = (spec.get("eda") or {}).get("report")
            p = rep if rep and str(rep).endswith(".ipynb") else None
        if p and os.path.exists(os.path.join(root, p)):
            out[stage] = p
    return out


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(value, list) and value and all(isinstance(x, (str, int, float)) for x in value):
        out[prefix] = ", ".join(map(str, value))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(f"{prefix}[{i}]", v, out)
    elif isinstance(value, str) and len(value) > 2:
        out[prefix] = value


BOILERPLATE = ("## Setup", "## Next Step", "## Write to the Spec", "## Save the", "## Load the", "## Build the")


def _note_text(src):
    """The part of a markdown cell that records reasoning: without headings and the italic help lines the skills add.
    Empty when nothing is left."""
    if src.lstrip().startswith(BOILERPLATE) or "**How to use this notebook.**" in src:
        return ""
    keep = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")
            and not (l.strip().startswith("*") and l.strip().endswith("*") and not l.strip().startswith("* "))]
    text = " ".join(keep).strip()
    return text if len(text) >= 60 else ""


def inventory(spec, root):
    """Every note as a dict: id, stage, kind, where, owner (the notebook to change), written, text."""
    runs = spec.get("notebooks") or {}
    nbs = notebooks(spec, root)
    when = lambda stage: (runs.get(stage) or {}).get("run_at") or ((spec.get(stage) or {}).get("run_at"))
    notes = []

    def add(stage, kind, where, text, owner=None, written=None):
        notes.append({"id": f"N{len(notes) + 1}", "stage": stage, "kind": kind, "where": where,
                      "owner": owner or nbs.get(stage), "written": written or when(stage), "text": str(text).strip()})

    # Notebook cells: reasoning (markdown) and decisions (tagged code cells)
    for stage, path in nbs.items():
        nb = nb_common.read(os.path.join(root, path))
        for c in nb.get("cells", []):
            src = c.get("source", "")
            src = src if isinstance(src, str) else "".join(src)
            tags = c.get("metadata", {}).get("tags") or []
            if c.get("cell_type") == "markdown" and stage != "eda" and _note_text(src):
                add(stage, "reasoning", f"{path}#{c.get('id')}", _note_text(src), owner=path)
            elif c.get("cell_type") == "code" and "decision" in tags:
                add(stage, "decision cell", f"{path}#{c.get('id')}", src, owner=path)

    # Spec settings written in words
    for section in TEXT_SECTIONS:
        flat = {}
        _flatten(section, spec.get(section) or {}, flat)
        for key, text in flat.items():
            if any(k in key for k in (".files", ".code", "run_at", ".results", ".checks", "hash", ".folds", "inputs[")):
                continue
            add(OWNER[section], "spec setting", key, text)

    # Shared lists
    for i, a in enumerate(spec.get("assumptions") or []):
        add(None, "assumption", f"assumptions[{i}]", a, owner=None, written=None)
    for q in spec.get("open_questions") or []:
        if isinstance(q, dict):
            add(q.get("stage"), "open question" if q.get("status") == "open" else "resolved question",
                f"open_questions.{q.get('id')}", q.get("question") + (f" Answer: {q['answer']}" if q.get("answer") else ""),
                owner=None, written=None)
    for i, d in enumerate(spec.get("decisions") or []):
        if isinstance(d, dict):
            text = d.get("decision", "") + (f" Reason: {d['reason']}" if d.get("reason") else "")
            if d.get("superseded_by"):
                text += f" (superseded by: {d['superseded_by']})"
            add(d.get("stage"), "decision", f"decisions[{i}]", text, owner=None, written=d.get("date"))
    # Documents generated from the spec
    doc = os.path.join(ml_common.project_paths(spec, root)["docs"], "business_framing.md")
    if os.path.exists(doc):
        add("business", "document", ml_common.relpath(doc, root),
            "Generated from the business section by 01_business_framing.ipynb; fixed by fixing that notebook.")
    return notes


def find(spec, root, text):
    hits = []
    for stage, path in notebooks(spec, root).items():
        for c in nb_common.read(os.path.join(root, path)).get("cells", []):
            src = c.get("source", "")
            src = src if isinstance(src, str) else "".join(src)
            if text in src:
                hits.append(f"{path}#{c.get('id')} ({c.get('cell_type')}, appears {src.count(text)} times)")
    flat = {}
    for section in spec:
        _flatten(section, spec.get(section), flat)
    hits += [f"ml_spec.yaml: {k}" for k, v in flat.items() if text in str(v)]
    return hits


# ---------------------------------------------------------------- apply

def apply(plan, spec_path, dry_run=False, run=True, log=print):
    root = os.path.dirname(spec_path)
    spec = ml_common.migrate(ml_common.load(spec_path))
    edits = plan.get("edits") or []
    nbs = notebooks(spec, root)
    stage_of = {os.path.normpath(p): s for s, p in nbs.items()}

    # Check everything first, so nothing is half applied
    problems = []
    for e in edits:
        p = os.path.join(root, e["notebook"])
        if not os.path.exists(p):
            problems.append(f"{e['notebook']}: not found")
            continue
        cell = next((c for c in nb_common.read(p)["cells"] if c.get("id") == e["cell"]), None)
        src = "" if cell is None else (cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"]))
        if cell is None:
            problems.append(f"{e['notebook']}: no cell {e['cell']}")
        elif src.count(e["old"]) != 1:
            problems.append(f"{e['notebook']} {e['cell']}: the old text appears {src.count(e['old'])} times, it must appear once")
    decisions = spec.get("decisions") or []
    for d in plan.get("supersede_decisions") or []:
        if not any(isinstance(x, dict) and x.get("decision") == d["decision"] for x in decisions):
            problems.append(f"decision not found: {d['decision'][:80]!r}")
    ids = {q.get("id") for q in spec.get("open_questions") or [] if isinstance(q, dict)}
    for q in plan.get("resolve_questions") or []:
        if q["id"] not in ids:
            problems.append(f"open question {q['id']} not found")
    assumptions = list(spec.get("assumptions") or [])
    for a in (plan.get("retire_assumptions") or []):
        if a["assumption"] not in assumptions:
            problems.append(f"assumption not found: {a['assumption'][:80]!r}")
    for a in (plan.get("replace_assumptions") or []):
        if a["old"] not in assumptions:
            problems.append(f"assumption not found: {a['old'][:80]!r}")
    # An assumption that is retired or reworded must also leave the decision cell it came from, or the next run of
    # that notebook adds it back.
    gone = [a["assumption"] for a in plan.get("retire_assumptions") or []] + [a["old"] for a in plan.get("replace_assumptions") or []]
    for stage, path in nbs.items():
        for c in nb_common.read(os.path.join(root, path)).get("cells", []):
            if "decision" not in (c.get("metadata", {}).get("tags") or []):
                continue
            src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            for e in edits:
                if os.path.normpath(e["notebook"]) == os.path.normpath(path) and e["cell"] == c.get("id"):
                    src = src.replace(e["old"], e["new"])
            for text in gone:
                if text in src:
                    problems.append(f"The assumption {text[:60]!r} is still in {path} {c.get('id')}; add an edit that "
                                    "removes or rewords it there, or the next run of that notebook adds it back.")
    if problems:
        return {"ok": False, "problems": problems}
    if dry_run:
        return {"ok": True, "dry_run": True, "edits": len(edits)}

    # Back up the spec and every notebook that changes
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(root, ".reconcile_backup", stamp)
    os.makedirs(backup, exist_ok=True)
    shutil.copy2(spec_path, backup)
    for nb in {e["notebook"] for e in edits}:
        os.makedirs(os.path.join(backup, os.path.dirname(nb)), exist_ok=True)
        shutil.copy2(os.path.join(root, nb), os.path.join(backup, nb))
    log(f"Backup: {ml_common.relpath(backup, root)}")

    # Notebook cells
    changed_stages, decision_stages = set(), set()
    for e in edits:
        kind = nb_common.edit_cell(os.path.join(root, e["notebook"]), e["cell"], e["old"], e["new"])
        stage = stage_of.get(os.path.normpath(e["notebook"]))
        changed_stages.add(stage)
        if kind == "decision":
            decision_stages.add(stage)
        log(f"Edited {e['notebook']} {e['cell']} ({kind})")

    # Rerun: every notebook with a changed decision cell, and every later notebook, in workflow order.
    runs = []
    if run and decision_stages:
        first = min(ORDER.index(s) for s in decision_stages if s in ORDER)
        todo = [s for s in nbs if ORDER.index(s) >= first]
        for stage in todo:
            res = nb_common.run_existing(os.path.join(root, nbs[stage]))
            runs.append({"stage": stage, "notebook": nbs[stage], "ran": res["executed"], "error": res["error"]})
            log(f"Ran {nbs[stage]}: " + ("yes" if res["executed"] else f"no ({res['error']})"))
            if not res["executed"]:
                log("Stopped: later notebooks were not run, because they depend on this one.")
                break

    # Shared lists, after the reruns (a rerun only adds to them)
    spec = ml_common.migrate(ml_common.load(spec_path))
    today = _dt.date.today().isoformat()
    for d in plan.get("supersede_decisions") or []:
        for x in spec.get("decisions") or []:
            if isinstance(x, dict) and x.get("decision") == d["decision"] and not x.get("superseded_by"):
                x["superseded_by"] = d["superseded_by"]
                x["superseded_on"] = today
        if not any(isinstance(x, dict) and x.get("decision") == d["superseded_by"] for x in spec.get("decisions") or []):
            spec.setdefault("decisions", []).append({"date": today, "stage": d.get("stage"), "decision": d["superseded_by"],
                                                     "reason": d.get("reason"), "replaces": d["decision"]})
    for q in plan.get("resolve_questions") or []:
        for x in spec.get("open_questions") or []:
            if isinstance(x, dict) and x.get("id") == q["id"]:
                x["status"], x["answer"] = "resolved", q["answer"]
    for a in plan.get("retire_assumptions") or []:
        if a["assumption"] in (spec.get("assumptions") or []):
            spec["assumptions"].remove(a["assumption"])
            spec.setdefault("retired_assumptions", []).append({"assumption": a["assumption"], "reason": a.get("reason"),
                                                               "retired_on": today})
    for a in plan.get("replace_assumptions") or []:
        spec["assumptions"] = [a["new"] if x == a["old"] else x for x in spec.get("assumptions") or []]
    n = sum(len(plan.get(k) or []) for k in ("edits", "supersede_decisions", "resolve_questions", "retire_assumptions",
                                             "replace_assumptions"))
    ml_common.log(spec, SKILL, f"reconciled {n} notes")
    ml_common.save(spec_path, spec)
    return {"ok": True, "backup": ml_common.relpath(backup, root), "edited": sorted(s for s in changed_stages if s),
            "runs": runs, "remaining": ml_common.consistency_checks(spec)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inventory", "check", "find", "apply"])
    ap.add_argument("arg", nargs="?", help="text for find, plan file for apply")
    ap.add_argument("--out", help="inventory: write the notes to this JSON file as well")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--dry-run", action="store_true", help="apply: check the plan without changing anything")
    ap.add_argument("--no-run", action="store_true", help="apply: do not run the notebooks again")
    a = ap.parse_args()
    spec_path = ml_common.spec_path(a.spec)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found.")
    root = os.path.dirname(spec_path)
    os.chdir(root)
    spec = ml_common.migrate(ml_common.load(spec_path))

    if a.command == "inventory":
        notes = inventory(spec, root)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=1, ensure_ascii=False)
        counts = {}
        for n in notes:
            counts[n["kind"]] = counts.get(n["kind"], 0) + 1
        print(f"{len(notes)} notes: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        print("Notebooks: " + ", ".join(f"{s} {p}" for s, p in notebooks(spec, root).items()))
        for stage in ORDER + [None]:
            group = [n for n in notes if n["stage"] == stage]
            if not group:
                continue
            print(f"\n## {stage or 'shared lists'}")
            for n in group:
                text = n["text"].replace("\n", " ")
                print(f"{n['id']} [{n['kind']}] {n['where']}" + (f" (written {n['written']})" if n["written"] else "")
                      + f"\n    {text[:600]}" + ("..." if len(text) > 600 else ""))
    elif a.command == "check":
        found = ml_common.consistency_checks(spec)
        if not found:
            print("No conflicts found by the fixed checks.")
        for i, c in enumerate(found, 1):
            print(f"C{i} [{c['check']}] {c['detail']}\n    where: {', '.join(c['where'])}")
    elif a.command == "find":
        if not a.arg:
            ap.error("give the text to find")
        hits = find(spec, root, a.arg)
        print("\n".join(hits) if hits else "Not found.")
    else:
        if not a.arg:
            ap.error("give the plan file")
        res = apply(ml_common.load(a.arg), spec_path, dry_run=a.dry_run, run=not a.no_run)
        if not res["ok"]:
            print("Nothing was changed. Problems in the plan:")
            for p in res["problems"]:
                print("  " + p)
            sys.exit(1)
        if res.get("dry_run"):
            print(f"The plan is valid: {res['edits']} cell edits. Nothing was changed.")
            return
        print("Notebooks edited: " + (", ".join(res["edited"]) or "none"))
        failed = [r for r in res["runs"] if not r["ran"]]
        print("Notebooks run again: " + (", ".join(r["stage"] for r in res["runs"]) or "none")
              + (f". Failed: {failed[0]['notebook']} ({failed[0]['error']})" if failed else ""))
        rem = res["remaining"]
        print("Conflicts left by the fixed checks: " + ("none" if not rem else "; ".join(c["detail"] for c in rem)))
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:   # output piped into head
        pass
