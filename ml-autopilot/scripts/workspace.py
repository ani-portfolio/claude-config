#!/usr/bin/env python3
"""Workspaces for the experiments of an ml-autopilot round: one git worktree per experiment.

An experiment never changes the project. It runs in a git worktree under .worktrees/ on a branch of its own
(exp/v2-a), with the project's data and models copied in, and its MLflow runs are logged as candidates under the
experiment's parent run (V2-a), itself a child of the version's parent run (V2). The experiment the round settles on
(autopilot.py round advance) is adopted: its branch is merged into the version branch and its data and models are
copied back. The others are discarded: their parent runs are tagged rejected and their worktrees removed. Their
branches and MLflow runs stay for the record.

Commands (run from the project root with the project's Python):
    workspace.py create <id> [--json]       # the worktree for experiment <id> of the open round
    workspace.py list [--json]              # the round's experiments and their worktrees
    workspace.py adopt <id> [--json]        # merge the experiment into the project (the round is closed by autopilot.py)
    workspace.py discard <id> --reason "..." [--json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import git_ops  # noqa: E402
import ml_common  # noqa: E402

SKILL = "ml-autopilot"
WORKTREES = ".worktrees"
# Ignored by git, so copied into a workspace (and back when it is adopted). data/raw is linked: it is read only.
COPIED = ["data/processed", "models"]
LINKED = [".venv", "data/raw"]


def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def iterate_script():
    """ml-iterate's iterate.py, which does every write to the iterations section."""
    skills = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    p = os.path.join(os.path.dirname(skills), "ml-iterate", "scripts", "iterate.py")
    return p if os.path.exists(p) else os.path.join(skills, "..", "ml-iterate", "scripts", "iterate.py")


def run_iterate(root, *args):
    r = subprocess.run([sys.executable, iterate_script(), *args, "--json"], cwd=root, capture_output=True, text=True)
    try:
        out = json.loads(r.stdout) if r.stdout.strip() else {}
    except ValueError:
        out = {"text": r.stdout.strip()}
    if r.returncode != 0:
        out = {"ok": False, "problems": [(r.stdout or r.stderr).strip()[-800:]]}
    return out


def _tracking():
    try:
        import tracking
        return tracking
    except ImportError:
        return None


def open_round(spec):
    it = spec.get("iterations") or {}
    return next((e for e in it.get("entries") or [] if e.get("id") == it.get("current") and e.get("status") == "open"),
                None)


def experiment_entry(spec, exp_id):
    e = open_round(spec)
    x = next((x for x in (e or {}).get("experiments") or [] if str(x.get("id")) == str(exp_id)), None)
    return e, x


def names(entry, x):
    exp = x.get("experiment") or f"{entry.get('version')}-{x['id']}"
    slug = exp.lower()
    return exp, f"exp/{slug}", slug


def _ensure_ignored(root, spec, spec_path):
    """Keep .worktrees/ and the workspace file out of git."""
    gi = os.path.join(root, ".gitignore")
    text = open(gi, encoding="utf-8").read() if os.path.exists(gi) else ""
    add = [p for p in (f"{WORKTREES}/", ml_common.WORKSPACE_FILE) if p not in text.split()]
    if add:
        with open(gi, "a", encoding="utf-8") as f:
            f.write(("\n" if text and not text.endswith("\n") else "") + "\n# Experiment workspaces (ml-autopilot)\n"
                    + "\n".join(add) + "\n")
        git_ops.commit_paths(root, [gi], "Keep experiment workspaces out of git", spec)


def create(root, spec_path, exp_id):
    spec = ml_common.load(spec_path)
    entry, x = experiment_entry(spec, exp_id)
    if x is None:
        return {"ok": False, "problems": [f"The open round has no experiment {exp_id!r}. Open the round first with "
                                          "`autopilot.py round open`."]}
    exp, branch, slug = names(entry, x)
    path = os.path.join(root, WORKTREES, slug)
    if os.path.exists(path):
        return {"ok": True, "experiment": exp, "workspace": path, "branch": branch, "existing": True}
    _ensure_ignored(root, spec, spec_path)
    # The worktree starts from the last commit, so the spec with the open round must be committed
    git_ops.commit_paths(root, [spec_path], f"{entry.get('version')}: round {entry['id']} before its workspaces",
                         spec)
    done, msg = git_ops.worktree_add(root, path, branch)
    if not done:
        return {"ok": False, "problems": [msg]}
    notes = [msg]
    for rel in LINKED:
        src, dst = os.path.join(root, rel), os.path.join(path, rel)
        if os.path.exists(src):
            if os.path.lexists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) and not os.path.islink(dst) else os.remove(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    for rel in COPIED + [entry.get("snapshot") or ""]:
        src, dst = os.path.join(root, rel), os.path.join(path, rel)
        if rel and os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=True)
    shutil.copy2(spec_path, os.path.join(path, ml_common.SPEC_NAME))   # the spec exactly as the round opened it
    parent = None
    t = _tracking()
    if t is not None:
        try:
            tags = {"iteration": entry["id"], "version": entry.get("version"), "workspace": path, "git_branch": branch,
                    "changes": "; ".join(f"{c['stage']}: {c['change']}" for c in x.get("changes") or [])[:5000]}
            if entry.get("mlflow_parent_run"):
                tags["mlflow.parentRunId"] = entry["mlflow_parent_run"]
            parent = t.start_experiment(spec_path, exp, x.get("hypothesis"), tags=tags)
        except Exception as ex:  # noqa: BLE001  tracking never blocks the work
            notes.append(f"MLflow: could not create the parent run of {exp}: {ex}")
    info = {"experiment": exp, "id": str(x["id"]), "iteration": entry["id"], "version": entry.get("version"),
            "parent_run": parent, "version_parent_run": entry.get("mlflow_parent_run"), "main_root": root,
            "workspace": path, "branch": branch, "hypothesis": x.get("hypothesis"), "changes": x.get("changes"),
            "rerun": x.get("rerun"), "phase": x.get("phase") or "screen", "compare_to": x.get("compare_to"),
            "frozen_model": x.get("frozen_model"), "refs": x.get("refs"), "created_at": _now(),
            "agent_session": os.environ.get("ML_AGENT_SESSION")}
    with open(os.path.join(path, ml_common.WORKSPACE_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, default=str)
    if parent:
        notes.append(f"MLflow: {exp} parent run {parent[:8]}, under the {entry.get('version')} run")
    rec = {"status": "running", "workspace": ml_common.relpath(path, root), "branch": branch,
           "mlflow_parent_run": parent}
    with open(os.path.join(path, "experiment_result.json"), "w", encoding="utf-8") as f:
        json.dump({**rec, "status": "running"}, f)
    r = run_iterate(root, "record-experiment", str(x["id"]), "--result", os.path.join(path, "experiment_result.json"))
    if r.get("ok") is False:
        notes.append("Could not record the workspace in the spec: " + " ".join(r.get("problems") or []))
    os.remove(os.path.join(path, "experiment_result.json"))
    return {"ok": True, **info, "notes": notes}


def _overlay(saved, merged):
    """Put back what the project owns during a round: the autopilot section, the round's experiment records, and
    history lines the workspace does not have."""
    merged["autopilot"] = saved.get("autopilot") or merged.get("autopilot") or {}
    s_it, m_it = saved.get("iterations") or {}, merged.setdefault("iterations", {})
    s_open = {e.get("id"): e for e in s_it.get("entries") or []}
    for e in m_it.get("entries") or []:
        if e.get("id") in s_open and s_open[e["id"]].get("experiments"):
            e["experiments"] = s_open[e["id"]]["experiments"]
    have = {json.dumps(h, sort_keys=True, default=str) for h in merged.get("history") or []}
    extra = [h for h in saved.get("history") or [] if json.dumps(h, sort_keys=True, default=str) not in have]
    merged["history"] = (merged.get("history") or []) + extra
    return merged


def adopt(root, spec_path, exp_id, force=False):
    spec = ml_common.load(spec_path)
    entry, x = experiment_entry(spec, exp_id)
    if x is None:
        return {"ok": False, "problems": [f"The open round has no experiment {exp_id!r}."]}
    if x.get("status") != "passed" and not force:
        return {"ok": False, "problems": [f"Experiment {x.get('experiment')} is {x.get('status')}, not passed. Only an "
                                          "experiment that passed the iteration bar is adopted."]}
    exp, branch, slug = names(entry, x)
    path = os.path.join(root, WORKTREES, slug)
    if not git_ops.branch_exists(root, branch):
        return {"ok": False, "problems": [f"The branch {branch} does not exist."]}
    if os.path.isdir(path):   # commit whatever the experiment left uncommitted, so the merge carries it
        ws_spec = ml_common.load(os.path.join(path, ml_common.SPEC_NAME))
        pp = ml_common.project_paths(ws_spec, path)
        git_ops.commit_paths(path, [os.path.join(path, ml_common.SPEC_NAME), pp["notebooks"], pp["src"], pp["docs"],
                                    os.path.join(path, "answers")], f"{exp}: final state", ws_spec)
    saved_text = open(spec_path, encoding="utf-8").read()
    saved = ml_common.load(spec_path)
    git_ops.git(root, "checkout", "HEAD", "--", ml_common.SPEC_NAME)
    # The project's spec moves during a round (each workspace and each result is recorded in it), so its hunks can
    # conflict with the experiment's. The experiment's side wins; _overlay then puts back what the project owns.
    done, msg = git_ops.merge_branch(root, branch, f"Adopt {exp}: {x.get('hypothesis')}", prefer="theirs")
    if not done:
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(saved_text)
        return {"ok": False, "problems": [msg]}
    notes = [msg]
    merged = _overlay(saved, ml_common.load(spec_path))
    ml_common.log(merged, SKILL, f"adopted {exp} into the project")
    ml_common.save(spec_path, merged)
    if os.path.isdir(path):
        for rel in COPIED:
            src, dst = os.path.join(path, rel), os.path.join(root, rel)
            if os.path.isdir(src) and not os.path.islink(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=True)
                notes.append(f"copied {rel}/ from the workspace")
        git_ops.worktree_remove(root, path)
        notes.append(f"removed the worktree {ml_common.relpath(path, root)}; the branch {branch} is kept")
    git_ops.commit_paths(root, [spec_path], f"Adopted {exp}", merged)
    return {"ok": True, "experiment": exp, "branch": branch, "notes": notes}


def discard(root, spec_path, exp_id, reason):
    spec = ml_common.load(spec_path)
    entry, x = experiment_entry(spec, exp_id)
    if x is None:
        return {"ok": False, "problems": [f"The open round has no experiment {exp_id!r}."]}
    exp, branch, slug = names(entry, x)
    path = os.path.join(root, WORKTREES, slug)
    notes = []
    t = _tracking()
    if t is not None:
        try:
            if t.reject(spec_path, exp, reason):
                notes.append(f"MLflow: {exp} tagged rejected; its runs stay for the record")
        except Exception as ex:  # noqa: BLE001
            notes.append(f"MLflow: could not tag {exp} as rejected: {ex}")
    if os.path.isdir(path):
        ws_spec = ml_common.load(os.path.join(path, ml_common.SPEC_NAME))
        pp = ml_common.project_paths(ws_spec, path)
        git_ops.commit_paths(path, [os.path.join(path, ml_common.SPEC_NAME), pp["notebooks"], pp["src"], pp["docs"],
                                    os.path.join(path, "answers")], f"{exp}: final state (discarded)", ws_spec)
        git_ops.worktree_remove(root, path)
        notes.append(f"removed the worktree; the branch {branch} is kept")
    return {"ok": True, "experiment": exp, "branch": branch, "notes": notes}


def listing(root, spec):
    e = open_round(spec)
    trees = {os.path.realpath(w["path"]): w for w in git_ops.worktrees(root)}
    out = []
    for x in (e or {}).get("experiments") or []:
        exp, branch, slug = names(e, x)
        path = os.path.join(root, WORKTREES, slug)
        out.append({"id": x["id"], "experiment": exp, "status": x.get("status"), "branch": branch,
                    "workspace": path if os.path.realpath(path) in trees else None,
                    "bar": (x.get("bar") or {}).get("status")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["create", "list", "adopt", "discard"])
    ap.add_argument("id", nargs="?", help="the experiment id (a, b, ...)")
    ap.add_argument("--reason", help="discard: why")
    ap.add_argument("--force", action="store_true", help="adopt: even when the experiment did not pass the bar")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    spec_path = ml_common.spec_path(a.spec)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found. Run this from the project root.")
    root = os.path.dirname(os.path.abspath(spec_path))
    if ml_common.workspace_info(root):
        sys.exit("This is an experiment workspace. Run workspace.py from the main project.")
    os.chdir(root)
    if a.command == "list":
        res = {"ok": True, "experiments": listing(root, ml_common.load(spec_path))}
    elif not a.id:
        ap.error(f"{a.command} needs the experiment id")
    elif a.command == "create":
        res = create(root, spec_path, a.id)
    elif a.command == "adopt":
        res = adopt(root, spec_path, a.id, force=a.force)
    else:
        if not a.reason:
            ap.error("discard needs --reason")
        res = discard(root, spec_path, a.id, a.reason)
    if a.json:
        print(json.dumps(res, indent=1, default=str))
    else:
        if not res.get("ok"):
            print("Nothing was changed. " + " ".join(res.get("problems") or []))
        elif a.command == "list":
            for x in res["experiments"]:
                print(f"{x['experiment']}: {x['status']}, bar {x['bar']}, branch {x['branch']}, "
                      f"workspace {x['workspace'] or 'none'}")
        else:
            print(f"{a.command}: {res.get('experiment')} ({res.get('branch')})")
            for n in res.get("notes") or []:
                print("  " + n)
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
