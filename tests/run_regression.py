#!/usr/bin/env python3
"""Regression suite for the ML project skills.

For each fixture in tests/fixtures (a project cut back to the state after EDA, with kept answers for the later
stages), it:

1. copies the fixture to a temporary folder, makes it a git repository the way ml-project-setup does (an empty
   commit on main, work on ml/v1), and sets up tracking on a throwaway MLflow store in that folder;
2. runs split, features, baseline, training, and evaluation with `--json`, as an agent would;
3. checks the results: every stage exit code 0, one commit per stage on ml/v1 and nothing on main, runs tagged V1
   and current, the champion registered with the aliases champion and champion-v1, and the answers artifact and git
   commit on each stage run;
4. with --iterate (the first fixture only, unless --iterate-all), opens V2, carries the V1 champion forward, reruns
   training and evaluation, closes it as kept, and checks that the iteration bar was applied; then opens V3 and rolls
   it back, and checks the branches, tags, aliases, and current runs.
5. with --autopilot (the first fixture only), runs one ml-autopilot round of two experiments the way the agents
   would, each in its own git worktree, adopts the one that passed the iteration bar and discards the other, and
   checks the worktrees, branches, MLflow candidates, verdicts and aliases, and the spec;
6. for an exploratory fixture (business.mode exploratory), checks that the model card frames the result by the
   learning question.

The real MLflow server and ~/.mlflow are never touched: each fixture gets a throwaway server (tools/mlflow_server.py
with its own MLFLOW_HOME in the temporary folder and a free port), and the projects log to it over HTTP with the
mlflow-skinny client, as real projects do.

Usage:
    python3 tests/run_regression.py [fixture ...] [--iterate] [--iterate-all] [--autopilot] [--keep] [--python PATH]

The Python that runs the notebooks needs the project packages (pandas, pyarrow, scikit-learn, lightgbm, optuna,
mlflow, nbformat, nbclient, ipykernel, matplotlib, pyyaml). It is, in order: --python, the ML_TEST_PYTHON
environment variable, or a cached environment that uv builds once in ~/.cache/ml-skills-regression.
Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures")
CACHE = os.path.expanduser("~/.cache/ml-skills-regression")
# mlflow-skinny, the client the projects install; the full mlflow is only the shared server.
PACKAGES = ["pandas", "pyarrow", "numpy", "scikit-learn", "lightgbm", "optuna", "mlflow-skinny>=3.17,<4", "nbformat", "nbclient",
            "ipykernel", "matplotlib", "pyyaml", "cloudpickle", "jupyter_client"]

STAGES = [  # (stage, skill folder, answers file, extra flags)
    ("split", "ml-train-test-split", "split.yaml", []),
    ("features", "ml-feature-engineering", "features.yaml", []),
    ("baseline", "ml-baseline-model", "baseline.yaml", []),
    ("training", "ml-model-training", "training.yaml", []),
    ("evaluation", "ml-offline-evaluation", "evaluation.yaml", []),
]


# ---------------------------------------------------------------- environment

def test_python(arg):
    py = arg or os.environ.get("ML_TEST_PYTHON")
    if py:
        return py
    py = os.path.join(CACHE, ".venv", "bin", "python")
    if os.path.exists(py):
        return py
    uv = shutil.which("uv")
    if not uv:
        sys.exit("No test Python: pass --python, set ML_TEST_PYTHON, or install uv so the suite can build one.")
    os.makedirs(CACHE, exist_ok=True)
    print(f"Building the test environment in {CACHE} (once)...")
    subprocess.run([uv, "venv", "-q", "--python", "3.11", os.path.join(CACHE, ".venv")], check=True)
    subprocess.run([uv, "pip", "install", "-q", "--python", py, *PACKAGES], check=True)
    return py


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


class Checks:
    def __init__(self, name):
        self.name, self.passed, self.failed = name, [], []

    def check(self, ok, what, detail=""):
        (self.passed if ok else self.failed).append(what + (f" ({detail})" if detail and not ok else ""))
        print(("  ok    " if ok else "  FAIL  ") + what + (f": {detail}" if detail and not ok else ""))
        return ok


# ---------------------------------------------------------------- project

def make_project(fixture, base, env):
    root = os.path.join(base, os.path.basename(fixture))
    shutil.copytree(fixture, root)
    os.makedirs(os.path.join(root, "kept_answers"), exist_ok=True)
    for f in os.listdir(os.path.join(root, "answers")):   # the fixture answers are inputs; answers/ is for the skills
        shutil.move(os.path.join(root, "answers", f), os.path.join(root, "kept_answers", f))
    sys.path.insert(0, os.path.join(SKILLS, "_ml_shared"))
    import ml_common  # noqa: E402
    spec_path = os.path.join(root, ml_common.SPEC_NAME)
    spec = ml_common.load(spec_path)
    spec["version"] = {"current": "V1", "branch": "ml/v1", "history": []}
    spec["git"] = {"auto_commit": True}
    ml_common.save(spec_path, spec)
    with open(os.path.join(root, ".gitignore"), "w") as f:
        f.write("__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.venv/\ndata/\nmodels/\nmlruns/\nmlflow.db\nmlflow_artifacts/\n"
                "iterations/\n.iterate_backup/\n.reconcile_backup/\nmlflow_export/\nkept_answers/\n")
    for args in (["init", "-q", "-b", "main"], ["config", "user.name", "Regression Test"],
                 ["config", "user.email", "regression@example.com"],
                 ["commit", "-q", "--allow-empty", "-m", "Initialize repository"], ["switch", "-q", "-c", "ml/v1"],
                 ["add", "-A"], ["commit", "-q", "-m", "V1 setup: regression fixture"]):
        r = git(root, *args)
        if r.returncode:
            sys.exit(f"git {' '.join(args)} failed: {r.stderr}")
    r = subprocess.run([env["PY"], os.path.join(SKILLS, "_ml_shared", "tracking.py"), "setup"], cwd=root,
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit(f"tracking setup failed: {r.stdout}{r.stderr}")
    git(root, "commit", "-q", "-am", "V1 setup: tracking")
    return root, spec_path


def run_stage(root, env, skill, answers, extra=()):
    script = os.path.join(SKILLS, skill, "scripts", "make_notebook.py")
    t0 = time.time()
    r = subprocess.run([env["PY"], script, answers, "--json", *extra], cwd=root, capture_output=True, text=True,
                       env=env, timeout=1800)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        out = {"exit_code": r.returncode, "exit_reason": "no JSON", "stderr": r.stderr[-1500:]}
    out["seconds"] = round(time.time() - t0, 1)
    if out.get("exit_code"):
        out["stderr_tail"] = r.stderr[-1500:]
    return out


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mlflow_server(store, command):
    """Start or stop a throwaway MLflow server with its data in `store`. Returns (tracking URI, environment)."""
    port_file = os.path.join(store, "port")
    if command == "start":
        os.makedirs(store, exist_ok=True)
        with open(port_file, "w") as f:
            f.write(str(free_port()))
    with open(port_file) as f:
        port = f.read().strip()
    senv = {**os.environ, "MLFLOW_HOME": store, "MLFLOW_PORT": port}
    r = subprocess.run([sys.executable, os.path.join(SKILLS, "tools", "mlflow_server.py"), command], env=senv,
                       capture_output=True, text=True)
    if command == "start" and r.returncode:
        sys.exit(f"Could not start a test MLflow server: {r.stdout}{r.stderr}")
    return f"http://127.0.0.1:{port}"


def mlflow_client(env):
    os.environ["MLFLOW_TRACKING_URI"] = env["MLFLOW_TRACKING_URI"]
    from mlflow.tracking import MlflowClient  # noqa: E402
    return MlflowClient(env["MLFLOW_TRACKING_URI"])


# ---------------------------------------------------------------- checks

def check_v1(c, root, spec_path, env, results):
    import ml_common
    spec = ml_common.load(spec_path)
    for stage, _, _, _ in STAGES:
        res = results.get(stage) or {}
        c.check(res.get("exit_code") == 0, f"{stage}: exit code 0",
                f"{res.get('exit_reason')} {res.get('stderr_tail', '')[-400:]}")
    c.check(git(root, "rev-list", "--count", "main").stdout.strip() == "1", "main holds only the first commit")
    c.check(git(root, "branch", "--show-current").stdout.strip() == "ml/v1", "work is on ml/v1")
    log = git(root, "log", "--format=%s", "main..ml/v1").stdout
    for stage, *_ in STAGES:
        c.check(f"V1 {stage}: built by" in log, f"{stage}: committed on ml/v1")
    c.check(not git(root, "status", "--porcelain", "--", "ml_spec.yaml", "notebooks", "src", "answers").stdout.strip(),
            "the spec, notebooks, code, and answers are committed", git(root, "status", "--porcelain").stdout[:300])
    c.check(not git(root, "ls-files", "data", "models").stdout.strip(), "data and models are not in git")

    client = mlflow_client(env)
    exp_id = spec["tracking"]["experiment_id"]
    for stage, *_ in STAGES:
        rid = ((spec.get(stage) or {}).get("mlflow") or {}).get("run_id")
        if not c.check(bool(rid), f"{stage}: the spec names its MLflow run"):
            continue
        run = client.get_run(rid)
        t = run.data.tags
        c.check(t.get("current") == "true", f"{stage}: run is current")
        c.check(t.get("version") == "V1", f"{stage}: run tagged version V1", t.get("version"))
        c.check(t.get("git_branch") == "ml/v1", f"{stage}: run tagged branch ml/v1", t.get("git_branch"))
        c.check(bool(t.get("git_commit")), f"{stage}: run tagged with its git commit")
        arts = [a.path for a in client.list_artifacts(rid, "answers")]
        c.check(bool(arts), f"{stage}: answers logged as an artifact")
    currents = client.search_runs([exp_id], "tags.current = 'true' and tags.run_type = 'stage'")
    stages = [r.data.tags.get("stage") for r in currents]
    c.check(len(stages) == len(set(stages)), "one current run per stage", str(sorted(stages)))
    spec = ml_common.load(spec_path)
    if (spec.get("business") or {}).get("mode") == "exploratory":
        card = os.path.join(root, "docs", "model_card.md")
        text = open(card).read() if os.path.exists(card) else ""
        c.check("An exploratory project" in text and "| Answer | Action |" in text,
                "exploratory: the model card frames the result by the learning question")
    name = spec["tracking"].get("registered_model") or f"{spec['project']['name']}-champion"
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        v1 = client.get_model_version_by_alias(name, "champion-v1")
        c.check(champ.version == v1.version, "aliases champion and champion-v1 name the same model version")
    except Exception as e:  # noqa: BLE001
        c.check(False, "champion registered with aliases champion and champion-v1", str(e)[:200])


def run_iteration(c, root, spec_path, env, results):
    import ml_common
    it = [env["PY"], os.path.join(SKILLS, "ml-iterate", "scripts", "iterate.py")]
    kept = os.path.join(root, "kept_answers")
    client = mlflow_client(env)
    spec = ml_common.load(spec_path)
    name = spec["tracking"].get("registered_model") or f"{spec['project']['name']}-champion"
    v1_version = client.get_model_version_by_alias(name, "champion").version

    # V2: change training and keep it
    plan = os.path.join(root, "kept_answers", "plan_v2.yaml")
    with open(plan, "w") as f:
        f.write("hypothesis: More tuning trials find a better logistic model.\n"
                "changes:\n  - stage: training\n    change: Tune with more trials.\n")
    r = subprocess.run(it + ["open", plan, "--json"], cwd=root, capture_output=True, text=True, env=env)
    c.check(r.returncode == 0, "V2: iteration opened", (r.stdout + r.stderr)[-400:])
    c.check(git(root, "branch", "--show-current").stdout.strip() == "ml/v2", "V2: work is on ml/v2")
    c.check(git(root, "tag", "-l", "v1").stdout.strip() == "v1", "V2: v1 is tagged")
    spec = ml_common.load(spec_path)
    entry = spec["iterations"]["entries"][-1]
    c.check(spec["version"]["current"] == "V2" and entry.get("version") == "V2", "V2: the spec says version V2")
    c.check(bool(entry.get("mlflow_parent_run")), "V2: MLflow parent run created")
    c.check(bool((spec["iterations"].get("bar") or {}).get("min_folds")), "V2: the iteration bar is written to the spec")
    # Carry the V1 champion forward, so the iteration bar compares it with the new champion on the same folds.
    r = subprocess.run(it + ["carry-forward", "--json"], cwd=root, capture_output=True, text=True, env=env)
    c.check(r.returncode == 0, "V2: champion carried forward", (r.stdout + r.stderr)[-400:])
    training_v2 = os.path.join(kept, "training_v2.yaml")
    tr_answers = ml_common.load(os.path.join(kept, "training.yaml"))
    if r.returncode == 0:
        tr_answers["models"] = list(tr_answers.get("models") or []) + [json.loads(r.stdout)["model"]]
    ml_common.save(training_v2, tr_answers)
    for stage, skill, ans, extra in STAGES[3:]:
        flags = ["--rebuild", "--overwrite"] if stage == "training" else ["--overwrite"]
        res = run_stage(root, env, skill, training_v2 if stage == "training" else os.path.join(kept, ans), flags)
        results[f"v2 {stage}"] = res
        c.check(res.get("exit_code") == 0, f"V2 {stage}: exit code 0",
                f"{res.get('exit_reason')} {res.get('stderr_tail', '')[-400:]}")
    spec = ml_common.load(spec_path)
    rid = spec["training"]["mlflow"]["run_id"]
    t = client.get_run(rid).data.tags
    c.check(t.get("version") == "V2" and t.get("candidate") == "true", "V2 training: run tagged V2 and candidate")
    c.check(t.get("mlflow.parentRunId") == entry.get("mlflow_parent_run"), "V2 training: run under the V2 parent run")
    c.check(client.get_model_version_by_alias(name, "champion").version == v1_version,
            "V2 open: alias champion still names the V1 model")
    r = subprocess.run(it + ["close", "--outcome", "kept", "--reason", "Regression test keeps V2.", "--json"],
                       cwd=root, capture_output=True, text=True, env=env)
    c.check(r.returncode == 0, "V2: closed as kept", (r.stdout + r.stderr)[-400:])
    closed = ml_common.load(spec_path)["iterations"]["entries"][0]
    br = closed.get("bar_result") or {}
    few = br.get("status") == "incomplete" and "paired folds" in str(br.get("detail"))   # fewer folds than min_folds
    c.check(br.get("status") in ("pass", "fail") or few, "V2: the iteration bar was applied on the same folds", str(br))
    c.check(git(root, "tag", "-l", "v2").stdout.strip() == "v2", "V2: v2 is tagged")
    v2 = client.get_model_version_by_alias(name, "champion-v2").version
    c.check(client.get_model_version_by_alias(name, "champion").version == v2, "V2 kept: alias champion moved to V2")
    c.check(git(root, "rev-list", "--count", "main").stdout.strip() == "1", "V2: main still holds only the first commit")

    # V3: change training again and roll it back
    plan = os.path.join(root, "kept_answers", "plan_v3.yaml")
    with open(plan, "w") as f:
        f.write("hypothesis: Fewer trials are enough.\nchanges:\n  - stage: training\n    change: Tune with fewer trials.\n")
    r = subprocess.run(it + ["open", plan, "--json"], cwd=root, capture_output=True, text=True, env=env)
    c.check(r.returncode == 0, "V3: iteration opened", (r.stdout + r.stderr)[-400:])
    res = run_stage(root, env, "ml-model-training", os.path.join(kept, "training.yaml"), ["--rebuild", "--overwrite"])
    c.check(res.get("exit_code") == 0, "V3 training: exit code 0", res.get("stderr_tail", "")[-400:])
    r = subprocess.run(it + ["close", "--outcome", "rolled_back", "--reason", "Regression test rolls V3 back.",
                             "--restore", "--json"], cwd=root, capture_output=True, text=True, env=env)
    c.check(r.returncode == 0, "V3: closed as rolled back", (r.stdout + r.stderr)[-400:])
    c.check(git(root, "tag", "-l", "v3-rolled-back").stdout.strip() == "v3-rolled-back", "V3: v3-rolled-back is tagged")
    c.check(git(root, "branch", "--show-current").stdout.strip() == "ml/v2", "V3 rolled back: back on ml/v2")
    spec = ml_common.load(spec_path)
    c.check(spec["version"]["current"] == "V2", "V3 rolled back: the spec says version V2", spec["version"].get("current"))
    hist = {h["version"]: h["status"] for h in spec["version"].get("history") or []}
    c.check(hist.get("V3") == "rolled_back", "V3 rolled back: recorded in the version history", str(hist))
    c.check(hist.get("V1") == "kept" and hist.get("V2") == "kept", "V1 and V2 recorded as kept", str(hist))
    c.check(not any(h.get("superseded_by") for h in spec["version"]["history"] if h["version"] == "V2"),
            "V3 rolled back: V2 is the working version again, not superseded")
    c.check(client.get_model_version_by_alias(name, "champion").version == v2, "V3 rolled back: alias champion stays on V2")
    rid = spec["training"]["mlflow"]["run_id"]
    c.check(client.get_run(rid).data.tags.get("current") == "true", "V3 rolled back: the V2 training run is current again")
    c.check(git(root, "rev-list", "--count", "main").stdout.strip() == "1", "V3: main still holds only the first commit")


def run_autopilot(c, root, spec_path, env):
    """One round of two experiments, as ml-autopilot and two ml-experimenter agents would run it, without a model in
    the loop: the agents' work is the stage scripts with kept answers."""
    import ml_common
    PY = env["PY"]
    ap_ = [PY, os.path.join(SKILLS, "ml-autopilot", "scripts", "autopilot.py")]
    it = [PY, os.path.join(SKILLS, "ml-iterate", "scripts", "iterate.py")]
    kept = os.path.join(root, "kept_answers")
    client = mlflow_client(env)
    spec = ml_common.load(spec_path)
    name = spec["tracking"].get("registered_model") or f"{spec['project']['name']}-champion"
    v1_version = client.get_model_version_by_alias(name, "champion").version
    v1_training_run = spec["training"]["mlflow"]["run_id"]
    exp_id = spec["tracking"]["experiment_id"]
    if not (spec.get("project") or {}).get("kernel"):   # the fixtures have no kernel; a set up project always does
        spec["project"]["kernel"] = "regression"
        ml_common.save(spec_path, spec)
        git(root, "commit", "-q", "-am", "Regression: kernel name")

    def run(cmd, cwd=root):
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        try:
            out = json.loads(r.stdout)
        except ValueError:
            out = {"raw": (r.stdout + r.stderr)[-600:]}
        return r.returncode, out

    code, _ = run(ap_ + ["start", "--experiments-per-round", "2", "--max-rounds", "1"])
    c.check(code == 0, "autopilot: settings recorded")
    code, nxt = run(ap_ + ["next"])
    c.check(nxt.get("action") == "open_round", "autopilot: after the evaluation, the next action is a round", str(nxt))
    items = os.path.join(kept, "backlog.yaml")
    with open(items, "w") as f:
        f.write("- {stage: training, change: Tune LightGBM with 20 trials, hypothesis: More trials find a better model., cost: 1}\n"
                "- {stage: training, change: Ridge only with 2 trials, hypothesis: A simpler model is as good., cost: 1}\n")
    code, out = run(ap_ + ["backlog", "add", items])
    c.check(code == 0 and len(out.get("added") or []) == 2, "autopilot: two backlog items added", str(out)[:300])
    code, rnd = run(ap_ + ["round", "open", "--scratch", kept])
    c.check(code == 0 and len(rnd.get("experiments") or []) == 2, "round: opened with two workspaces", str(rnd)[:600])
    if code:
        return
    c.check(git(root, "branch", "--show-current").stdout.strip() == "ml/v2", "round: the project is on ml/v2")
    exps = {x["id"]: x for x in rnd["experiments"]}
    for x in exps.values():
        ws = x["workspace"]
        c.check(os.path.exists(os.path.join(ws, ".ml_workspace.json")), f"{x['experiment']}: workspace file written")
        c.check(git(ws, "branch", "--show-current").stdout.strip() == x["branch"], f"{x['experiment']}: worktree on {x['branch']}")
        c.check(bool(x.get("parent_run")), f"{x['experiment']}: MLflow parent run created")
        if x.get("parent_run"):
            c.check(client.get_run(x["parent_run"]).data.tags.get("mlflow.parentRunId") == x.get("version_parent_run"),
                    f"{x['experiment']}: parent run sits under the V2 run")

    # The agents: carry forward, train with the experiment's change, apply the bar, write the result
    base_answers = ml_common.load(os.path.join(kept, "training.yaml"))
    variants = {"a": [dict(m, tuning={**m["tuning"], "n_trials": 20}) for m in base_answers["models"] if m["type"] == "lightgbm"],
                "b": [dict(m, tuning={**m["tuning"], "n_trials": 2}) for m in base_answers["models"] if m["type"] == "ridge"]}
    for i, x in exps.items():
        ws = x["workspace"]
        code, cf = run(it + ["carry-forward", "--json"], cwd=ws)
        c.check(code == 0, f"{x['experiment']}: champion carried forward in the workspace", str(cf)[:300])
        ans = {**base_answers, "models": variants[i] + ([cf["model"]] if code == 0 else [])}
        ap_path = os.path.join(kept, f"training_{i}.yaml")
        ml_common.save(ap_path, ans)
        res = run_stage(ws, env, "ml-model-training", ap_path, ["--rebuild", "--overwrite"])
        c.check(res.get("exit_code") == 0, f"{x['experiment']}: training in the workspace, exit code 0",
                res.get("stderr_tail", "")[-400:])
        _, bar = run(it + ["bar", "--json"], cwd=ws)
        ws_spec = ml_common.load(os.path.join(ws, ml_common.SPEC_NAME))
        rid = ((ws_spec.get("training") or {}).get("mlflow") or {}).get("run_id")
        if rid:
            t = client.get_run(rid).data.tags
            c.check(t.get("candidate") == "true" and t.get("experiment") == x["experiment"] and t.get("current") == "false",
                    f"{x['experiment']}: training run is a candidate of its experiment, not current", str({k: t.get(k) for k in ("candidate", "experiment", "current")}))
            c.check(t.get("mlflow.parentRunId") == x.get("parent_run"), f"{x['experiment']}: training run under its parent run")
        with open(os.path.join(ws, "experiment_result.json"), "w") as f:
            json.dump({"bar": bar, "champion": (ws_spec.get("training") or {}).get("champion", {}).get("name"),
                       "summary": "regression"}, f)
        exps[i]["training_run"] = rid
        exps[i]["bar_status"] = bar.get("status")
    c.check(client.get_model_version_by_alias(name, "champion").version == v1_version,
            "round: alias champion has not moved while the experiments ran")
    res = run_stage(root, env, "ml-model-training", os.path.join(kept, "training.yaml"), ["--rebuild", "--overwrite"])
    c.check(res.get("exit_code") == 1, "round: a stage build in the project itself is refused while the round is open",
            str(res.get("exit_code")))
    c.check(ml_common.load(spec_path)["training"]["mlflow"]["run_id"] == v1_training_run,
            "round: the project's training section is untouched")

    for i, x in exps.items():
        code, out = run(it + ["record-experiment", i, "--result", os.path.join(x["workspace"], "experiment_result.json"), "--json"])
        c.check(code == 0, f"{x['experiment']}: result recorded in the project", str(out)[:300])
    passed = [i for i, x in exps.items() if x.get("bar_status") == "pass"]
    print("  bar results: " + ", ".join(f"{x['experiment']} {x.get('bar_status')}" for x in exps.values()))
    if not passed:   # the plumbing is under test here, not the models: let a count as passed
        print("  neither experiment passed the bar; recording V2-a as passed to test the adoption")
        run(it + ["record-experiment", "a", "--status", "passed", "--json"])
        passed = ["a"]
    win = sorted(passed)[0] if len(passed) == 1 else None
    code, nxt = run(ap_ + ["next"])
    c.check(nxt.get("action") == "close_round", "round: every experiment reported, so the next action is to close", str(nxt))
    code, closed = run(ap_ + ["round", "close"])
    c.check(code == 0, "round: closed", str(closed)[:600])
    if code:
        return
    win = win or closed.get("adopted", "").split("-")[-1].lower()
    lose = "b" if win == "a" else "a"
    W, L = exps[win], exps[lose]
    spec = ml_common.load(spec_path)
    entry = spec["iterations"]["entries"][-1]
    c.check(entry.get("status") == "closed" and entry.get("outcome") == "kept" and entry.get("adopted") == W["experiment"],
            "round: the iteration is closed as kept with the adopted experiment", str({k: entry.get(k) for k in ("status", "outcome", "adopted")}))
    st = {x["id"]: x.get("status") for x in entry.get("experiments") or []}
    c.check(st.get(win) == "adopted" and st.get(lose) == "discarded", "round: experiment statuses adopted and discarded", str(st))
    c.check(spec["training"]["mlflow"]["run_id"] == W["training_run"], "adopt: the project's training section is the adopted experiment's")
    c.check(client.get_run(W["training_run"]).data.tags.get("current") == "true", "adopt: the adopted training run is current")
    c.check(client.get_run(v1_training_run).data.tags.get("current") == "false", "adopt: the V1 training run is no longer current")
    c.check(client.get_run(W["parent_run"]).data.tags.get("verdict") == "promoted", "adopt: the adopted experiment is promoted")
    c.check(client.get_run(L["parent_run"]).data.tags.get("verdict") == "rejected", "discard: the other experiment is rejected")
    c.check(bool(client.search_runs([exp_id], f"tags.experiment = '{L['experiment']}' and tags.run_type = 'stage'")),
            "discard: the rejected experiment's runs stay in MLflow")
    champ = client.get_model_version_by_alias(name, "champion")
    c.check(champ.version != v1_version and champ.tags.get("experiment") in (W["experiment"], None),
            "adopt: alias champion moved to the adopted experiment's model", f"{champ.version} {champ.tags}")
    try:
        c.check(client.get_model_version_by_alias(name, "champion-v2").version == champ.version,
                "adopt: alias champion-v2 names the adopted model")
    except Exception as e:  # noqa: BLE001
        c.check(False, "adopt: alias champion-v2 names the adopted model", str(e)[:200])
    champ_path = os.path.join(root, (spec["training"].get("champion") or {}).get("path") or "missing")
    c.check(os.path.exists(champ_path), "adopt: the adopted champion's model file is in the project")
    c.check(not os.path.exists(W["workspace"]) and not os.path.exists(L["workspace"]), "the worktrees are removed")
    c.check(bool(git(root, "branch", "--list", W["branch"]).stdout.strip()) and bool(git(root, "branch", "--list", L["branch"]).stdout.strip()),
            "both experiment branches are kept")
    c.check(f"Adopt {W['experiment']}" in git(root, "log", "--format=%s", "-20").stdout, "adopt: merge commit on ml/v2")
    bl = {b["id"]: b["status"] for b in spec["autopilot"]["backlog"]}
    c.check(sorted(bl.values()) == ["adopted", "tried"], "the backlog records adopted and tried", str(bl))
    c.check(git(root, "rev-list", "--count", "main").stdout.strip() == "1", "autopilot: main still holds only the first commit")
    c.check(not git(root, "status", "--porcelain", "--", "ml_spec.yaml").stdout.strip(), "autopilot: the spec is committed")
    code, nxt = run(ap_ + ["next"])
    c.check(nxt.get("action") == "checkpoint" and nxt.get("checkpoint") == "test_final",
            "autopilot: after the last round, it stops before the final test look", str(nxt))
    code, cp = run(ap_ + ["checkpoint", "write", "test_final"])
    c.check(code == 0 and os.path.exists(os.path.join(root, "docs", "checkpoint.md")), "checkpoint: docs/checkpoint.md written")
    code, nxt = run(ap_ + ["next"])
    c.check(nxt.get("action") == "review", "checkpoint: the next start reviews it", str(nxt))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fixtures", nargs="*", help="Fixture names (default: all)")
    ap.add_argument("--iterate", action="store_true", help="Also test V2 kept and V3 rolled back on the first fixture")
    ap.add_argument("--iterate-all", action="store_true", help="Test the iterations on every fixture")
    ap.add_argument("--autopilot", action="store_true", help="Also test one autopilot round on the first fixture")
    ap.add_argument("--keep", action="store_true", help="Keep the temporary projects")
    ap.add_argument("--python", help="Python that runs the notebooks")
    args = ap.parse_args()

    py = test_python(args.python)
    try:
        import mlflow  # noqa: F401  the checks read the MLflow store
    except ImportError:
        # Compare the paths as given, not resolved: a venv's python links to the same interpreter as python3.
        if os.path.abspath(sys.executable) != os.path.abspath(py):
            os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])
        sys.exit("The test Python has no mlflow.")
    names = args.fixtures or sorted(d for d in os.listdir(FIXTURES) if os.path.isdir(os.path.join(FIXTURES, d)))
    base = tempfile.mkdtemp(prefix="ml-regression-")
    summary, all_ok = [], True
    for i, name in enumerate(names):
        print(f"\n== {name}")
        store = os.path.join(base, f"{name}_mlflow")
        uri = mlflow_server(store, "start")
        env = {**os.environ, "PY": py, "MLFLOW_TRACKING_URI": uri, "MLFLOW_DISABLE_TELEMETRY": "true",
               "MLFLOW_DISABLE_AGENT_HINT": "1", "ML_AGENT_SESSION": f"regression-{name}"}
        for k in ("ML_ITERATE_EXPERIMENT", "ML_ITERATE_ITERATION", "ML_ITERATE_PARENT_RUN"):
            env.pop(k, None)
        root, spec_path = make_project(os.path.join(FIXTURES, name), base, env)
        c = Checks(name)
        results = {}
        for stage, skill, ans, extra in STAGES:
            res = run_stage(root, env, skill, os.path.join(root, "kept_answers", ans), extra)
            results[stage] = res
            print(f"  {stage}: exit {res.get('exit_code')} in {res['seconds']} s")
            if res.get("exit_code"):
                break
        check_v1(c, root, spec_path, env, results)
        if (args.iterate and i == 0) or args.iterate_all:
            if not c.failed:
                run_iteration(c, root, spec_path, env, results)
            else:
                c.check(False, "iteration test skipped because V1 failed")
        elif args.autopilot and i == 0:
            if not c.failed:
                run_autopilot(c, root, spec_path, env)
            else:
                c.check(False, "autopilot test skipped because V1 failed")
        mlflow_server(store, "stop")
        summary.append((name, len(c.passed), len(c.failed), c.failed))
        all_ok &= not c.failed
        with open(os.path.join(base, f"{name}_results.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)

    print("\n== Summary")
    for name, p, f, failed in summary:
        print(f"{name}: {p} passed, {f} failed")
        for x in failed:
            print(f"    {x}")
    if args.keep or not all_ok:
        print(f"Projects kept in {base}")
    else:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
