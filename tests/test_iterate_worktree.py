#!/usr/bin/env python3
"""Checks for ml-iterate's worktree option (iterate.py open --worktree, and close with a workspace).

Each check builds a small git project in a temporary folder: a spec with an evaluated champion, a model file, a
result file, and raw data. No notebooks run and MLflow is not configured, so the checks are fast.

Plain asserts, no test framework:
    python3 tests/test_iterate_worktree.py
Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)
SCRIPTS = os.path.join(SKILLS, "ml-iterate", "scripts")
ITERATE = os.path.join(SCRIPTS, "iterate.py")
sys.path.insert(0, SCRIPTS)

import ml_common as m  # noqa: E402
import nb_common  # noqa: E402
import tracking  # noqa: E402


def _git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)}: {r.stderr}"
    return r.stdout.strip()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _project():
    """A git project on ml/v1 with an evaluated champion, ready for an iteration."""
    root = os.path.realpath(tempfile.mkdtemp(prefix="iter_wt_"))
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "commit", "-q", "--allow-empty", "-m", "empty")
    _git(root, "switch", "-q", "-c", "ml/v1")
    s = m.template("t")
    s["training"] = {"champion": {"name": "lgbm", "type": "lightgbm", "mae": 1.0}, "primary_label": "mae"}
    s["evaluation"] = {"test_opened": ["2026-01-01"], "champion": "lgbm"}
    m.save(os.path.join(root, m.SPEC_NAME), s)
    _write(os.path.join(root, ".gitignore"), "data/\nmodels/\n.venv/\n")
    _write(os.path.join(root, "models", "champion.joblib"), "v1 model")
    _write(os.path.join(root, "data", "processed", "cv_folds.parquet"), "v1 folds")
    _write(os.path.join(root, "data", "raw", "weather.csv"), "raw")
    _write(os.path.join(root, "notebooks", "08_training.ipynb"), "{}")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "V1")
    return root


def _run(root, *args):
    r = subprocess.run([sys.executable, ITERATE, *args], cwd=root, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _open(root):
    plan = os.path.join(root, "..", os.path.basename(root) + "_plan.yaml")
    _write(plan, "hypothesis: Longer folds catch the season.\nchanges:\n"
                 "  - {stage: split, change: Two folds of 26 weeks.}\n")
    code, out = _run(root, "open", plan, "--worktree")
    assert code == 0, out
    return os.path.join(root, ".worktrees", "v2")


def _work_in(ws):
    """What a stage rerun leaves in the worktree: a new model, a new notebook, and a new champion in the spec."""
    _write(os.path.join(ws, "models", "champion.joblib"), "v2 model")
    _write(os.path.join(ws, "notebooks", "05_split.ipynb"), "{\"v\": 2}")
    p = os.path.join(ws, m.SPEC_NAME)
    s = m.load(p)
    s["training"]["champion"] = {"name": "ridge", "type": "ridge", "mae": 0.9}
    m.save(p, s)


def test_open_makes_the_worktree_and_leaves_the_project():
    root = _project()
    ws = _open(root)
    assert os.path.isdir(ws)
    assert _git(ws, "branch", "--show-current") == "iter/v2"
    assert _git(root, "branch", "--show-current") == "ml/v2"
    info = m.workspace_info(ws)
    assert info and info["kind"] == "iteration" and info["version"] == "V2"
    assert _read(os.path.join(ws, "models", "champion.joblib")) == "v1 model"
    assert _read(os.path.join(ws, "data", "processed", "cv_folds.parquet")) == "v1 folds"
    assert os.path.islink(os.path.join(ws, "data", "raw"))
    e = m.load(os.path.join(root, m.SPEC_NAME))["iterations"]["entries"][-1]
    assert e["workspace"] == ".worktrees/v2" and e["workspace_branch"] == "iter/v2"
    # the worktree's spec holds the same open iteration, so the stages there log under V2
    assert m.load(os.path.join(ws, m.SPEC_NAME))["iterations"]["current"] == e["id"]


def test_stage_builds_only_in_the_worktree():
    root = _project()
    ws = _open(root)
    try:
        nb_common._refuse_during_round(["--spec", os.path.join(root, m.SPEC_NAME)])
        raise AssertionError("a stage build in the main project was allowed")
    except SystemExit as e:
        assert ".worktrees/v2" in str(e.code)
    nb_common._refuse_during_round(["--spec", os.path.join(ws, m.SPEC_NAME)])   # allowed


def test_close_refused_inside_the_worktree():
    root = _project()
    ws = _open(root)
    code, out = _run(ws, "close", "--outcome", "kept", "--reason", "r")
    assert code != 0 and "main project" in out, out


def test_close_kept_merges_and_copies_back():
    root = _project()
    ws = _open(root)
    _work_in(ws)
    code, out = _run(root, "close", "--outcome", "kept", "--reason", "better")
    assert code == 0, out
    assert not os.path.exists(ws)
    assert _git(root, "branch", "--show-current") == "ml/v2"
    assert "iter/v2" in _git(root, "branch", "--list", "iter/v2")
    assert _git(root, "merge-base", "--is-ancestor", "iter/v2", "ml/v2") == ""
    assert _read(os.path.join(root, "models", "champion.joblib")) == "v2 model"
    assert _read(os.path.join(root, "notebooks", "05_split.ipynb")) == "{\"v\": 2}"
    s = m.load(os.path.join(root, m.SPEC_NAME))
    e = s["iterations"]["entries"][-1]
    assert e["status"] == "closed" and e["outcome"] == "kept"
    assert e["end"]["champion"] == "ridge"
    assert s["training"]["champion"]["name"] == "ridge"
    assert s["iterations"]["current"] is None
    assert "v2" in _git(root, "tag", "--list", "v2")


def test_close_rolled_back_drops_the_worktree():
    root = _project()
    ws = _open(root)
    _work_in(ws)
    code, out = _run(root, "close", "--outcome", "rolled_back", "--reason", "worse")
    assert code == 0, out
    assert not os.path.exists(ws)
    assert not os.path.exists(os.path.join(root, ".iterate_backup"))
    assert _git(root, "branch", "--show-current") == "ml/v1"
    assert _read(os.path.join(root, "models", "champion.joblib")) == "v1 model"
    assert not os.path.exists(os.path.join(root, "notebooks", "05_split.ipynb"))
    assert _git(root, "rev-parse", "v2-rolled-back^{commit}") == _git(root, "rev-parse", "iter/v2")
    s = m.load(os.path.join(root, m.SPEC_NAME))
    e = s["iterations"]["entries"][-1]
    assert e["status"] == "closed" and e["outcome"] == "rolled_back"
    assert e["end"]["champion"] == "ridge"   # the record of what the iteration reached
    assert s["training"]["champion"]["name"] == "lgbm"
    assert s["version"]["current"] == "V1"


def test_iteration_workspace_is_interactive_and_in_place():
    ws = tempfile.mkdtemp(prefix="iter_ws_")
    spec = {"iterations": {"current": 1, "entries": [{"id": 1, "status": "open", "version": "V2",
                                                      "mlflow_parent_run": "p"}]}}
    _write(os.path.join(ws, m.WORKSPACE_FILE), json.dumps({"kind": "iteration", "version": "V2"}))
    assert not m.autopilot_mode("", ws)
    c = tracking.candidate(spec, ws)
    assert c["experiment"] == "V2" and c["in_place"] and c["parent"] == "p"
    # a workspace file from before kinds existed is an experiment's
    _write(os.path.join(ws, m.WORKSPACE_FILE), json.dumps({"experiment": "V2-a", "parent_run": "q"}))
    assert m.autopilot_mode("", ws)
    c = tracking.candidate(spec, ws)
    assert c["experiment"] == "V2-a" and not c["in_place"]


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} of {len(tests)} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
