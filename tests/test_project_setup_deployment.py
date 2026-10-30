#!/usr/bin/env python3
"""Checks for ml-project-setup's deployment layout (setup_project.py <folder> --add-deployment).

Each check builds a small project in a temporary folder: a spec with a system design, a pyproject.toml that already
lists pytest, and a CLAUDE.md. No .venv and no git, so nothing is installed and the checks are fast.

Plain asserts, no test framework:
    python3 tests/test_project_setup_deployment.py
Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)
SCRIPTS = os.path.join(SKILLS, "ml-project-setup", "scripts")
SETUP = os.path.join(SCRIPTS, "setup_project.py")
sys.path.insert(0, SCRIPTS)

import ml_common as m  # noqa: E402

COMPONENTS = ["ingestion", "data_validation", "features", "training", "registry", "inference", "serving",
              "label_feedback", "monitoring", "failure_handling", "orchestration", "testing"]
PIPELINES = ["feature_pipeline", "training_pipeline", "inference_pipeline", "monitoring_pipeline"]

PYPROJECT = """[project]
name = "t"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[dependency-groups]
dev = [
    "pytest>=9.1.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

CLAUDE = """# t

## Layout
* `src/`: reusable code.

## Conventions
* Keep notebooks short.
"""


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _project(status="confirmed", not_needed=(), pyproject=PYPROJECT):
    root = os.path.realpath(tempfile.mkdtemp(prefix="deploy_"))
    s = m.template("t")
    if status:
        s["system"] = {"status": status, "components": {
            c: ({"needed": False, "reason": "not in V1"} if c in not_needed else
                {"needed": True, "tool": f"{c} tool", "settings": {}, "reason": "r"}) for c in COMPONENTS}}
    m.save(os.path.join(root, m.SPEC_NAME), s)
    _write(os.path.join(root, "pyproject.toml"), pyproject)
    _write(os.path.join(root, "CLAUDE.md"), CLAUDE)
    os.makedirs(os.path.join(root, "src"))
    _write(os.path.join(root, "src", "__init__.py"), "")
    return root


def _run(root, *args):
    return subprocess.run([sys.executable, SETUP, root, "--add-deployment", "--no-git", *args],
                          capture_output=True, text=True)


def _files(root):
    out = set()
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".pytest_cache")]
        out.update(os.path.relpath(os.path.join(d, f), root) for f in files)
    return out


# ---------------------------------------------------------------- checks

def test_refuses_without_a_confirmed_design():
    for status in (None, "draft"):
        root = _project(status=status)
        before = _files(root)
        r = _run(root)
        assert r.returncode == 1, r.stdout + r.stderr
        assert "ml-system-design" in r.stdout + r.stderr
        assert _files(root) == before


def test_dry_run_lists_the_files_and_changes_nothing():
    root = _project()
    before = _files(root)
    r = _run(root, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _files(root) == before
    for p in PIPELINES:
        assert f"WOULD create src/{p}/__init__.py" in r.stdout, r.stdout
    assert "WOULD create conftest.py" in r.stdout
    assert "WOULD create tests/unit/test_pipeline_boundaries.py" in r.stdout


def test_creates_the_layout_once():
    root = _project()
    r = _run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    files = _files(root)
    for p in PIPELINES:
        for f in ("__init__.py", "jobs/.gitkeep", "tests/unit/.gitkeep", "tests/component/.gitkeep"):
            assert f"src/{p}/{f}" in files, (p, f)
    assert "src/feature_pipeline/expectations/.gitkeep" in files
    assert {"conftest.py", "tests/system/.gitkeep", "tests/unit/test_pipeline_boundaries.py"} <= files
    assert "ingestion tool" in _read(os.path.join(root, "src", "feature_pipeline", "__init__.py"))
    pipes = (m.load(os.path.join(root, m.SPEC_NAME)).get("paths") or {}).get("pipelines")
    assert pipes == {"feature": "src/feature_pipeline", "training": "src/training_pipeline",
                     "inference": "src/inference_pipeline", "monitoring": "src/monitoring_pipeline"}, pipes
    snapshot = {f: _read(os.path.join(root, f)) for f in files}
    r = _run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WOULD" not in r.stdout and " create " not in r.stdout.replace("KEPT", ""), r.stdout
    assert {f: _read(os.path.join(root, f)) for f in _files(root)} == snapshot


def test_pipelines_whose_components_are_not_needed_are_left_out():
    root = _project(not_needed=("label_feedback", "monitoring", "data_validation"))
    r = _run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    files = _files(root)
    assert not any(f.startswith("src/monitoring_pipeline/") for f in files)
    assert "src/feature_pipeline/__init__.py" in files
    assert "src/feature_pipeline/expectations/.gitkeep" not in files
    pipes = m.load(os.path.join(root, m.SPEC_NAME))["paths"]["pipelines"]
    assert "monitoring" not in pipes
    assert "monitoring_pipeline" not in _read(os.path.join(root, "tests", "unit", "test_pipeline_boundaries.py"))


def test_pyproject_gains_only_missing_items():
    root = _project()
    r = _run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    text = _read(os.path.join(root, "pyproject.toml"))
    assert text.startswith(PYPROJECT.split("[tool.hatch")[0]), "the [project] and build tables must be unchanged"
    assert 'packages = ["src"]' in text and 'exclude = ["src/*/tests"]' in text, text
    assert 'testpaths = ["tests", "src"]' in text, text
    assert "--import-mode=importlib" in text and '"unit:' in text and '"component:' in text and '"system:' in text
    assert '"pytest>=9.1.1"' in text


def test_pyproject_conflict_is_reported_not_changed():
    conflicting = PYPROJECT.replace('testpaths = ["tests"]', 'testpaths = "tests"\naddopts = "--import-mode=prepend"')
    root = _project(pyproject=conflicting)
    r = _run(root)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "import-mode" in r.stdout
    text = _read(os.path.join(root, "pyproject.toml"))
    assert 'addopts = "--import-mode=prepend"' in text and 'testpaths = "tests"' in text


def test_boundary_test_passes_on_the_skeleton_and_catches_a_violation():
    root = _project()
    assert _run(root).returncode == 0
    test = os.path.join(root, "tests", "unit", "test_pipeline_boundaries.py")
    run = lambda: subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
                                 cwd=root, capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    _write(os.path.join(root, "src", "inference_pipeline", "bad.py"), "from ..training_pipeline import x\n")
    r = run()
    assert r.returncode == 1 and "inference_pipeline/bad.py imports training_pipeline" in r.stdout, r.stdout


def test_claude_md_section_added_once_and_replaced_on_update():
    root = _project()
    _run(root)
    path = os.path.join(root, "CLAUDE.md")
    text = _read(path)
    assert text.count("## Deployment layout") == 1 and "src/inference_pipeline/" in text, text
    assert text.startswith(CLAUDE.rstrip()), "the existing sections must be unchanged"
    _run(root)
    assert _read(path).count("## Deployment layout") == 1
    _write(path, _read(path).replace("src/inference_pipeline/", "src/old_name/"))
    r = _run(root)
    assert r.returncode == 1 and "--update-claude-md" in r.stdout, r.stdout
    r = _run(root, "--update-claude-md")
    assert r.returncode == 0, r.stdout + r.stderr
    text = _read(path)
    assert "src/old_name/" not in text and text.count("## Deployment layout") == 1


def test_ml_status_points_to_the_layout_after_the_design():
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("status", os.path.join(SKILLS, "ml-status", "scripts", "status.py"))
    status = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(status)
    assert status.deployment_layout_state({"system": {"status": "draft"}}) is None
    assert status.deployment_layout_state({"system": {"status": "confirmed"}}) == "missing"
    assert status.deployment_layout_state({"system": {"status": "confirmed"},
                                           "paths": {"pipelines": {"feature": "src/feature_pipeline"}}}) == "done"
    import contextlib
    import io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status.print_next({"next": {"step": None, "skill": None}, "steps": [], "stale": [], "eval_decision": "go",
                           "iteration": None, "ml_needed": "yes", "deployment_layout": "missing"})
    assert "--add-deployment" in out.getvalue(), out.getvalue()


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} of {len(tests)} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
