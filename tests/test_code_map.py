#!/usr/bin/env python3
"""Checks for code-map/scripts/code_map.py: the JSON model it builds from three fixture repos, the HTML page, and
the setup that writes the config, the VS Code task, and the local git excludes.

Plain asserts, no test framework and no project packages needed:
    python3 tests/test_code_map.py
Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "code_map"
sys.path.insert(0, str(HERE.parent / "code-map" / "scripts"))

import code_map as cm  # noqa: E402


def _ids(items):
    return sorted(i["id"] for i in items)


def _edge(model, src, dst):
    found = [e for e in model["imports"] if e["from"] == src and e["to"] == dst]
    assert found, f"no import edge {src} -> {dst}: {model['imports']}"
    return found[0]


def _git_repo(fixture):
    tmp = Path(tempfile.mkdtemp()) / fixture
    shutil.copytree(FIXTURES / fixture, tmp)
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    return tmp


# Files and imports

def test_skips_tests_folders():
    model = cm.build_model(FIXTURES / "flat", ml=False)
    assert not [f for f in model["files"] if "/tests/" in f["id"]], model["files"]


def test_flat_imports_with_names():
    model = cm.build_model(FIXTURES / "flat", ml=False)
    assert _edge(model, "src/app/cli.py", "src/app/core.py")["names"] == ["run"]
    assert _edge(model, "src/app/cli.py", "src/utils/text.py")["names"] == ["slug"]
    assert _edge(model, "src/app/core.py", "src/utils/text.py")["names"] == ["slug"]
    assert _edge(model, "src/app/core.py", "src/config.py")["names"] == ["NAME"]


def test_relative_imports_resolve():
    model = cm.build_model(FIXTURES / "srclayout", ml=False)
    assert _edge(model, "src/mylib/core/api.py", "src/mylib/io/reader.py")["names"] == ["read_csv"]
    assert _edge(model, "src/mylib/core/api.py", "src/mylib/_util.py")["names"] == ["clean"]


def test_file_records():
    model = cm.build_model(FIXTURES / "flat", ml=False)
    core = next(f for f in model["files"] if f["id"] == "src/app/core.py")
    assert core["doc"] == "The core logic of the app."
    assert core["group"] == "app"
    assert core["path"] == str((FIXTURES / "flat" / "src/app/core.py").resolve())


# Grouping

def test_flat_groups():
    model = cm.build_model(FIXTURES / "flat", ml=False)
    assert [(g["id"], g["kind"]) for g in model["groups"]] == [("app", "group"), ("utils", "group"),
                                                               ("shared", "shared")]
    config = next(f for f in model["files"] if f["id"] == "src/config.py")
    assert config["group"] == "shared"


def test_srclayout_descends_into_single_package():
    model = cm.build_model(FIXTURES / "srclayout", ml=False)
    assert [g["id"] for g in model["groups"]] == ["core", "io", "shared"]
    groups = {f["id"]: f["group"] for f in model["files"]}
    assert groups["src/mylib/_util.py"] == "shared"
    assert groups["src/mylib/__init__.py"] == "shared"


def test_ml_groups():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    assert [g["id"] for g in model["groups"]] == ["feature_pipeline", "feedback_pipeline", "inference_pipeline",
                                                  "monitoring_pipeline", "training_pipeline", "shared"]


# Callers of shared files

def test_callers_by_function():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    assert model["callers"]["src/features.py"] == {"build_rows": ["src/feature_pipeline/pipeline.py",
                                                                  "src/inference_pipeline/predictor.py",
                                                                  "src/training_pipeline/jobs/train.py"]}
    assert "src/feature_pipeline/store.py" not in model["callers"]


# ML mode

def test_detect():
    ml = cm.detect(FIXTURES / "ml")
    assert ml["pipelines"] == ["feature_pipeline", "feedback_pipeline", "inference_pipeline", "monitoring_pipeline",
                               "training_pipeline"]
    assert ml["assets"] == ["DEPLOYMENT", "FEATURES", "FEATURE_VIEW", "MODEL", "PREDICTIONS", "SCORES"]
    assert ml["suggest_ml"] is True
    flat = cm.detect(FIXTURES / "flat")
    assert flat == {"pipelines": [], "assets": [], "suggest_ml": False}


def test_asset_edges():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    assert _ids(model["assets"]) == ["DEPLOYMENT", "FEATURES", "FEATURE_VIEW", "MODEL", "PREDICTIONS", "SCORES"]
    assert {a["via"] for a in model["assets"]} == {"platform"}
    got = {(e["group"], e["asset"], e["dir"]) for e in model["asset_edges"]}
    assert got == {
        ("feature_pipeline", "FEATURES", "write"),            # through name, version = platform.FEATURES
        ("feature_pipeline", "FEATURE_VIEW", "write"),
        ("training_pipeline", "FEATURE_VIEW", "read"),
        ("training_pipeline", "FEATURE_VIEW", "touch"),       # list(platform.FEATURE_VIEW)
        ("training_pipeline", "MODEL", "write"),
        ("training_pipeline", "SCORES", "read"),
        ("shared", "MODEL", "read"),                          # src/registry.py, which inference imports
        ("inference_pipeline", "FEATURE_VIEW", "read"),
        ("inference_pipeline", "DEPLOYMENT", "read"),
        ("inference_pipeline", "PREDICTIONS", "write"),
        ("feedback_pipeline", "PREDICTIONS", "read"),
        ("feedback_pipeline", "SCORES", "write"),
    }, got
    write = next(e for e in model["asset_edges"] if e["asset"] == "MODEL" and e["dir"] == "write")
    assert write["file"] == "src/training_pipeline/registry.py" and write["line"] == 11, write


def test_no_assets_without_ml():
    model = cm.build_model(FIXTURES / "ml", ml=False)
    assert model["assets"] == [] and model["asset_edges"] == []


# Tours

def test_tour_from_declared_script():
    model = cm.build_model(FIXTURES / "flat", ml=False)
    assert model["tours"] == [{"entry": "src/app/cli.py",
                               "steps": ["src/app/cli.py", "src/app/core.py", "src/utils/text.py",
                                         "src/config.py"]}]


def test_tour_from_public_api():
    model = cm.build_model(FIXTURES / "srclayout", ml=False)
    assert model["tours"] == [{"entry": "src/mylib/__init__.py",
                               "steps": ["src/mylib/__init__.py", "src/mylib/core/api.py", "src/mylib/io/reader.py",
                                         "src/mylib/_util.py"]}]


def test_tours_from_jobs_in_ml_mode():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    assert [t["entry"] for t in model["tours"]] == ["src/feature_pipeline/jobs/hourly.py",
                                                    "src/inference_pipeline/jobs/log.py",
                                                    "src/training_pipeline/jobs/train.py"]
    hourly = model["tours"][0]["steps"]
    assert hourly == ["src/feature_pipeline/jobs/hourly.py", "src/feature_pipeline/pipeline.py",
                      "src/feature_pipeline/store.py", "src/platform.py", "src/features.py"], hourly


def test_tours_from_main_without_ml():
    model = cm.build_model(FIXTURES / "ml", ml=False)
    assert [t["entry"] for t in model["tours"]] == ["src/feature_pipeline/jobs/hourly.py",
                                                    "src/training_pipeline/jobs/train.py"]


# Warnings

def test_warnings():
    flat = cm.build_model(FIXTURES / "flat", ml=False)
    assert any("no module docstring" in w and "src/utils/__init__.py" in w for w in flat["warnings"]), flat
    ml = cm.build_model(FIXTURES / "ml", ml=True)
    assert any("platform.MISSING" in w for w in ml["warnings"]), ml["warnings"]
    plain = cm.build_model(FIXTURES / "flat", ml=True)
    assert any("no store or registry names" in w for w in plain["warnings"]), plain["warnings"]


def test_parse_error_is_a_warning():
    repo = _git_repo("flat")
    (repo / "src" / "broken.py").write_text("def (:\n")
    model = cm.build_model(repo, ml=False)
    assert "src/broken.py" not in [f["id"] for f in model["files"]]
    assert any("src/broken.py" in w and "could not be parsed" in w for w in model["warnings"]), model["warnings"]


def test_no_entry_points_warning():
    repo = _git_repo("srclayout")
    (repo / "src" / "mylib" / "__init__.py").unlink()
    model = cm.build_model(repo, ml=False)
    assert model["tours"] == []
    assert any("no entry points" in w for w in model["warnings"]), model["warnings"]


# Command line and page

def test_cli_writes_page_and_model():
    repo = _git_repo("flat")
    out = subprocess.run([sys.executable, str(HERE.parent / "code-map" / "scripts" / "code_map.py"), str(repo),
                          "--no-ml", "--json"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    page = repo / ".code_map" / "index.html"
    assert out.stdout.splitlines()[0] == str(page.resolve())
    html = page.read_text()
    assert "cytoscape" in html and "src/app/core.py" in html and "__MODEL__" not in html
    assert json.loads((repo / ".code_map" / "model.json").read_text())["root"] == str(repo.resolve())


def test_cli_without_src_exits_1():
    empty = Path(tempfile.mkdtemp())
    out = subprocess.run([sys.executable, str(HERE.parent / "code-map" / "scripts" / "code_map.py"), str(empty)],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "src" in out.stderr


def test_cli_reads_ml_from_config():
    repo = _git_repo("ml")
    cm.setup(repo, ml=False)
    cm.main([str(repo), "--json"])
    assert json.loads((repo / ".code_map" / "model.json").read_text())["ml"] is False


# Setup

def test_setup_writes_config_task_and_excludes():
    repo = _git_repo("flat")
    cm.setup(repo, ml=False)
    cm.setup(repo, ml=True)
    assert json.loads((repo / ".code_map" / "config.json").read_text()) == {"ml": True}
    tasks = json.loads((repo / ".vscode" / "tasks.json").read_text())
    assert [t["label"] for t in tasks["tasks"]] == ["Code map"]
    exclude = (repo / ".git" / "info" / "exclude").read_text().splitlines()
    assert exclude.count(".code_map/") == 1 and exclude.count(".vscode/tasks.json") == 1, exclude
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", ".code_map", ".vscode"],
                            capture_output=True, text=True).stdout
    assert status == "", status


def test_setup_merges_untracked_tasks():
    repo = _git_repo("flat")
    (repo / ".vscode").mkdir()
    (repo / ".vscode" / "tasks.json").write_text(json.dumps({"version": "2.0.0", "tasks": [{"label": "Mine"}]}))
    cm.setup(repo, ml=False)
    tasks = json.loads((repo / ".vscode" / "tasks.json").read_text())
    assert [t["label"] for t in tasks["tasks"]] == ["Mine", "Code map"]


def test_setup_leaves_tracked_tasks_alone():
    repo = _git_repo("flat")
    (repo / ".vscode").mkdir()
    original = json.dumps({"version": "2.0.0", "tasks": []})
    (repo / ".vscode" / "tasks.json").write_text(original)
    subprocess.run(["git", "-C", str(repo), "add", ".vscode/tasks.json"], check=True)
    result = cm.setup(repo, ml=False)
    assert (repo / ".vscode" / "tasks.json").read_text() == original
    assert result["task"] == "skipped: .vscode/tasks.json is tracked by git"
    exclude = (repo / ".git" / "info" / "exclude").read_text().splitlines()
    assert ".vscode/tasks.json" not in exclude


# Columns: a left to right layout

def test_rank_chain_and_unlinked():
    rank, removed = cm._rank(["a", "b", "c", "z"], {("a", "b"): 1, ("b", "c"): 1, ("a", "c"): 1})
    assert rank == {"a": 0, "b": 1, "c": 2, "z": 0} and removed == [], (rank, removed)


def test_rank_drops_self_loops():
    rank, removed = cm._rank(["a", "b"], {("a", "a"): 3, ("a", "b"): 1})
    assert rank == {"a": 0, "b": 1} and removed == []


def test_rank_breaks_tie_at_furthest_backward_edge():
    edges = {("s", "a"): 1, ("a", "b"): 1, ("b", "c"): 1, ("c", "a"): 1}
    rank, removed = cm._rank(["a", "b", "c", "s"], edges)
    assert removed == [("c", "a")], removed
    assert rank == {"s": 0, "a": 1, "b": 2, "c": 3}, rank


def test_rank_breaks_weighted_cycle_at_weakest_edge():
    rank, removed = cm._rank(["a", "b"], {("a", "b"): 1, ("b", "a"): 2})
    assert removed == [("a", "b")] and rank == {"b": 0, "a": 1}, (rank, removed)


def _cols(items):
    return {i["id"]: i["col"] for i in items}


def test_group_cols_follow_data_flow():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    assert _cols(model["groups"]) == {"feature_pipeline": 0, "training_pipeline": 2, "inference_pipeline": 4,
                                      "feedback_pipeline": 6, "monitoring_pipeline": 8, "shared": None}
    empty = {g["id"] for g in model["groups"] if g.get("empty")}
    assert empty == {"monitoring_pipeline"}, empty


def test_asset_cols_sit_after_their_writer():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    assert _cols(model["assets"]) == {"FEATURES": 1, "FEATURE_VIEW": 1, "MODEL": 3, "PREDICTIONS": 5, "SCORES": 7,
                                      "DEPLOYMENT": 3}  # no writer in src/: just left of its reader


def test_group_cols_follow_imports_without_ml():
    assert _cols(cm.build_model(FIXTURES / "flat", ml=False)["groups"]) == {"app": 0, "utils": 1, "shared": None}
    assert _cols(cm.build_model(FIXTURES / "srclayout", ml=False)["groups"]) == {"core": 0, "io": 1,
                                                                                 "shared": None}


def test_file_cols_inside_their_group():
    ml = _cols(cm.build_model(FIXTURES / "ml", ml=True)["files"])
    assert [ml[f"src/feature_pipeline/{f}"] for f in ("__init__.py", "jobs/hourly.py", "pipeline.py", "store.py")] \
        == [0, 0, 1, 2]
    flat = _cols(cm.build_model(FIXTURES / "flat", ml=False)["files"])
    assert [flat[f"src/app/{f}"] for f in ("__init__.py", "cli.py", "core.py")] == [0, 0, 1]


# Logic boxes: platforms around store names, and what each group does

def test_asset_platforms_from_call_names():
    model = cm.build_model(FIXTURES / "ml", ml=True)
    got = {a["id"]: a["platform"] for a in model["assets"]}
    assert got == {"FEATURES": "Feature store", "FEATURE_VIEW": "Feature store", "PREDICTIONS": "Feature store",
                   "SCORES": "Feature store", "MODEL": "Model registry", "DEPLOYMENT": "Serving"}, got


def test_group_doc_and_jobs():
    ml = {g["id"]: g for g in cm.build_model(FIXTURES / "ml", ml=True)["groups"]}
    assert ml["feature_pipeline"]["doc"] == "The feature pipeline."
    assert ml["feature_pipeline"]["jobs"] == ["src/feature_pipeline/jobs/hourly.py"]
    assert ml["monitoring_pipeline"]["jobs"] == []
    assert ml["shared"]["doc"] == ""
    flat = {g["id"]: g for g in cm.build_model(FIXTURES / "flat", ml=False)["groups"]}
    assert flat["app"]["doc"] == "The app package." and flat["app"]["jobs"] == ["src/app/cli.py"]
    assert flat["utils"]["doc"] == "" and flat["utils"]["jobs"] == []


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
