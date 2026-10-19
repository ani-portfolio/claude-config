#!/usr/bin/env python3
"""Track and sequence iterations of an ML project after an offline evaluation.

An iteration is one round of changes made because the evaluation showed something to improve. This script does not
rerun any stage itself: the stage skills do that. It records why each iteration happened and what it changed, saves
a snapshot of the state it starts from, works out which stages to run again and in which order, carries the previous
champion forward so it is scored on the same folds as the new models, and compares iterations at the end.

The original work is iteration 0. Opening iteration N saves the state it starts from to iterations/iter_<N-1>/.

Commands (run from the project root with the project's Python):
    iterate.py diagnose [--json]           # read only: the evidence from evaluation and training, to propose changes
    iterate.py open plan.yaml [--dry-run]  # snapshot, record the iteration, and print the rerun order
    iterate.py remaining [--json]          # the stages of the open iteration that have not run again yet
    iterate.py carry-forward [--json]      # the previous champion as a model entry for ml-model-training
    iterate.py compare [--json]            # every iteration side by side
    iterate.py bar [--json]                # the iteration bar: does the new champion really beat the carried one?
    iterate.py close --outcome kept|rolled_back|stopped --reason "..." [--restore] [--dry-run]
    iterate.py record-experiment <id> --result result.json   # a round's experiment: its bar result and verdict
    iterate.py close --outcome kept --adopted <id> --reason "..."     # a round: the experiment adopted into the project
    iterate.py close --outcome rolled_back --none-passed --reason "..."   # a round where no experiment passed

plan.yaml:
    hypothesis: One sentence on what should improve, and why.
    diagnosis: [short findings that led to it]
    changes:
      - {stage: split, change: "Two expanding folds of 26 weeks instead of three of 13."}
      - {stage: features, change: "Add a post COVID slope and a third Fourier pair."}

A round (ml-autopilot): the plan lists experiments instead of one set of changes. Each experiment is one change,
run by an agent in its own git worktree (ml-autopilot's workspace.py) and logged to MLflow as the experiment
<version>-<id> (V2-a, V2-b). The project itself does not change until the best experiment that passes the iteration
bar is adopted; when none passes, the round closes as rolled back with nothing to restore.
    hypothesis: What the round tries, in one sentence.
    experiments:
      - {id: a, hypothesis: "...", changes: [{stage: features, change: "..."}]}
      - {id: b, hypothesis: "...", changes: [{stage: split, change: "..."}]}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402

SKILL = "ml-iterate"
# The stages an iteration can change, in the order they must run again. Business, discovery and setup changes go
# through their own skills first; the rerun then starts at the ML framing.
CHAIN = ["framing", "eda", "split", "features", "baseline", "training", "evaluation"]
OUTCOMES = ("kept", "rolled_back", "stopped")
EXPERIMENT_STATUS = ("planned", "running", "passed", "failed", "error", "adopted", "discarded")
# Small result files copied into a snapshot. Large data (raw files, split sets, feature tables) is not copied.
SMALL_RESULTS = ["cv_folds.parquet", "baseline_cv_results.parquet", "baseline_oof_predictions.parquet",
                 "training_runs.parquet", "training_oof_predictions.parquet", "test_predictions.parquet",
                 "evaluation_metrics.parquet", "evaluation_slices.parquet"]


# ---------------------------------------------------------------- versions: git and MLflow
#
# V1 is the first pass. Opening iteration N starts version V(N+1): the previous version gets its git tag (v1), a
# branch ml/v(N+1) is made from it, and an MLflow parent run is created for the version, so every stage run of the
# iteration is logged under it (tracking.candidate reads the open iteration from the spec). Closing the iteration
# settles the version everywhere: kept or stopped promotes it in MLflow (the alias 'champion' moves to its champion)
# and tags v<N+1>; rolled back rejects it, makes the restored runs current again, tags v<N+1>-rolled-back, and goes
# back to the previous version's branch. Nothing is ever committed to main.

def _git():
    try:
        import git_ops
        return git_ops
    except ImportError:
        return None


def _tracking():
    try:
        import tracking
        return tracking
    except ImportError:
        return None


def _version(spec):
    v = spec.get("version") or {}
    v.setdefault("current", "V1")
    v.setdefault("branch", "ml/v1")
    v.setdefault("history", [])
    spec["version"] = v
    return v


def _vnum(name):
    try:
        return int(str(name).lstrip("Vv"))
    except ValueError:
        return 1


def _commit(root, spec, spec_path, message, extra=()):
    g = _git()
    if g is None or not g.is_repo(root) or not (spec.get("git") or {}).get("auto_commit", True):
        return None
    return g.commit_paths(root, [spec_path, *extra], message, spec,
                          trailers={"Version": _version(spec)["current"], "Skill": SKILL,
                                    "Agent-Session": os.environ.get("ML_AGENT_SESSION")})


def _start_version(spec, root, spec_path, entry):
    """Tag the version being left, branch the new one, and create its MLflow parent run. Returns notes."""
    notes = []
    v = _version(spec)
    prev, new = v["current"], f"V{entry['id'] + 1}"
    g = _git()
    if g is not None and g.is_repo(root):
        pp = ml_common.project_paths(spec, root)
        _commit(root, spec, spec_path, f"{prev}: state before {new} was opened",
                [pp["notebooks"], pp["src"], pp["docs"], os.path.join(root, "answers")])
        if g.tag(root, prev.lower(), f"{prev}: the state iteration {entry['id']} started from"):
            notes.append(f"git: tagged {prev.lower()}")
        target = f"ml/{new.lower()}"
        if g.branch_exists(root, target):
            g.git(root, "switch", "-q", target)
        else:
            g.git(root, "switch", "-q", "-c", target)
        notes.append(f"git: working on branch {target}")
    old = next((h for h in v["history"] if h.get("version") == prev), None)
    if old is None:   # V1, or a version from before versions were recorded: it was the working version, so kept
        old = {"version": prev, "branch": v.get("branch"), "status": "kept",
               "champion": entry["start"].get("champion"), "cv": entry["start"].get("cv"),
               "test": entry["start"].get("test"), "git_tag": prev.lower()}
        v["history"].append(old)
    old.update({"superseded_by": new, "superseded_at": _now()})
    v.update({"current": new, "branch": f"ml/{new.lower()}", "previous": prev})
    entry["version"] = new
    entry["previous_version"] = prev
    return notes


def _version_parent_run(spec_path, entry):
    t = _tracking()
    if t is None:
        return None
    try:
        return t.start_experiment(spec_path, entry["version"], entry["hypothesis"],
                                  tags={"iteration": entry["id"], "previous_version": entry.get("previous_version"),
                                        "changes": "; ".join(f"{c['stage']}: {c['change']}" for c in entry["changes"])[:5000]})
    except Exception as ex:  # noqa: BLE001  tracking never blocks the work
        print(f"MLflow: could not create the parent run of {entry['version']}: {ex}", file=sys.stderr)
        return None


def _settle_version(spec, root, spec_path, entry, outcome, reason, adopted=None):
    """Close the version in MLflow and git. Returns notes. In a round, adopted names the experiment (V2-a) whose
    runs become current; the version's own parent run is tagged with the verdict too."""
    notes = []
    v = _version(spec)
    ver = entry.get("version") or v["current"]
    t = _tracking()
    if t is not None:
        try:
            if outcome == "rolled_back":
                t.reject(spec_path, ver, reason)
                done = t.restore_current(spec_path)
                notes.append(f"MLflow: {ver} rejected; the restored runs are current again ({', '.join(done or []) or 'none'})")
            else:
                res = t.promote(spec_path, adopted or ver)
                if adopted:
                    t.promote(spec_path, ver)   # the version's parent run: its verdict; it holds no stage runs
                    res = res or {}
                if res:
                    notes.append(f"MLflow: {adopted or ver} promoted" + (
                        f"; alias champion is version {res['champion_version']}" if res.get("champion_version") else ""))
        except Exception as ex:  # noqa: BLE001
            notes.append(f"MLflow: could not settle {ver}: {ex}")
    status = {"kept": "kept", "stopped": "kept (stopped iterating)", "rolled_back": "rolled_back"}[outcome]
    v["history"] = [h for h in v["history"] if h.get("version") != ver] + [
        {"version": ver, "branch": f"ml/{ver.lower()}", "status": status, "closed_at": _now(), "reason": reason,
         "champion": (entry.get("end") or {}).get("champion"), "cv": (entry.get("end") or {}).get("cv"),
         "test": (entry.get("end") or {}).get("test"), "mlflow_parent_run": entry.get("mlflow_parent_run"),
         "git_tag": ver.lower() + ("-rolled-back" if outcome == "rolled_back" else "")}]
    if outcome == "rolled_back":
        prev = entry.get("previous_version") or "V1"
        v.update({"current": prev, "branch": f"ml/{prev.lower()}"})
        for h in v["history"]:   # the previous version is the working version again
            if h.get("version") == prev:
                h.pop("superseded_by", None)
                h.pop("superseded_at", None)
    return notes, ver


def _settle_git(spec, root, spec_path, entry, outcome, ver, restored_files):
    g = _git()
    notes = []
    if g is None or not g.is_repo(root):
        return notes
    paths = ml_common.project_paths(spec, root)
    extra = [os.path.join(root, f) for f in restored_files or []] + [paths["notebooks"], paths["src"], paths["docs"],
                                                                     os.path.join(root, "answers")]
    sha = _commit(root, spec, spec_path, f"{ver} closed: {outcome.replace('_', ' ')}", extra)
    tag = ver.lower() + ("-rolled-back" if outcome == "rolled_back" else "")
    if g.tag(root, tag, f"{ver} closed as {outcome}"):
        notes.append(f"git: tagged {tag}" + (f" ({sha[:8]})" if sha else ""))
    if outcome == "rolled_back":
        prev_branch = f"ml/{(entry.get('previous_version') or 'V1').lower()}"
        if g.branch_exists(root, prev_branch):
            text = ml_common.dump_text(spec)   # the record of the rolled back version, kept on the previous branch
            r = g.git(root, "switch", "-q", prev_branch)
            if r.returncode == 0:
                with open(spec_path, "w", encoding="utf-8") as f:
                    f.write(text)
                _commit(root, spec, spec_path, f"Record that {ver} was rolled back")
                notes.append(f"git: back on {prev_branch}")
            else:
                notes.append(f"git: could not switch back to {prev_branch}: {(r.stderr or '').strip()[:200]}")
    return notes


# ---------------------------------------------------------------- helpers

def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _ts(value):
    try:
        return _dt.datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _iters(spec):
    it = spec.get("iterations") or {}
    it.setdefault("current", None)
    it.setdefault("entries", [])
    spec["iterations"] = it
    return it


def _open_entry(spec):
    it = _iters(spec)
    return next((e for e in it["entries"] if e.get("id") == it["current"] and e.get("status") == "open"), None)


def _champion_state(spec):
    """The champion, its CV and test scores, how many times the test set was opened, and the data version."""
    tr, ev = spec.get("training") or {}, spec.get("evaluation") or {}
    champ = tr.get("champion") or {}
    label = tr.get("primary_label")
    test = next((m.get("test") for m in ev.get("metrics") or [] if m.get("kind") == "champion"), None)
    return {"champion": champ.get("name"), "type": champ.get("type"),
            "cv": champ.get(label) if label else None, "cv_std": champ.get("std"),
            "test": test if ev.get("champion") == champ.get("name") else None,
            "test_look": len(ev.get("test_opened") or []), "data_version": tr.get("data_version"),
            "metric": label}


def _decision(spec, decision, reason):
    spec.setdefault("decisions", []).append({"date": _now()[:10], "stage": "iteration", "decision": decision,
                                             "reason": reason})


def _hash_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _hash_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _rerun_order(changes):
    stages = [c.get("stage") for c in changes or []]
    idx = [CHAIN.index(s) for s in stages if s in CHAIN]
    if any(s in ("setup", "business", "discovery") for s in stages):
        idx.append(0)
    return CHAIN[min(idx):] if idx else []


def _input_names(features):
    """Input names from the features section; its inputs are dicts with a name, or plain names."""
    out = []
    for x in (features or {}).get("inputs") or []:
        out.append(x if isinstance(x, str) else (x.get("name") or x.get("feature") or x.get("column")))
    return [x for x in out if x]


def _say(data, as_json, text_lines):
    if as_json:
        print(json.dumps(data, indent=1, default=str, ensure_ascii=False))
    else:
        print("\n".join(text_lines))


# ---------------------------------------------------------------- diagnose

def diagnose(spec, root):
    """The evidence an iteration is planned from. Reads only."""
    paths = ml_common.project_paths(spec, root)
    ev, tr, sp = (spec.get(k) or {} for k in ("evaluation", "training", "split"))
    out = {"evaluated": bool(ev.get("metrics")), "state": _champion_state(spec), "evidence": ev.get("evidence") or [],
           "models": [], "folds": [], "flags": [], "conflicts": ml_common.consistency_checks(spec),
           "open_iteration": (_open_entry(spec) or {}).get("id"), "iterations": len(_iters(spec)["entries"])}
    cv = {r.get("model"): r for r in tr.get("leaderboard") or []}
    for m in ev.get("metrics") or []:
        out["models"].append({"model": m.get("model"), "kind": m.get("kind"), "test": m.get("test"),
                              "low": m.get("low"), "high": m.get("high"), "cv": m.get("cv"), "bias": m.get("bias"),
                              "cv_std": (cv.get(m.get("model")) or {}).get("primary_std")})

    per_fold, err = best_trial_folds(spec, root)
    if err:  # pragma: no cover
        out["flags"].append(f"Could not read the run log: {err}")
    out["per_fold"] = per_fold

    folds = (sp.get("cv") or {}).get("folds") or []
    out["folds"] = folds
    seasonal = _mentions_season(spec)
    short = []
    for f in folds:
        a, b = _ts(f.get("valid_start")), _ts(f.get("valid_end"))
        if a and b and (b - a).days + 7 < 364:
            short.append((b - a).days + 7)
    if folds and seasonal and short and sum(short) < 364:
        out["flags"].append(f"The {len(folds)} validation windows cover about {sum(short)} days in total, less than "
                            "one seasonal year, and the framing or EDA mentions seasonality. CV may miss what a full "
                            "test year shows. Consider longer or more validation windows (ml-train-test-split).")

    st = out["state"]
    others = [m for m in out["models"] if m["kind"] != "champion" and m.get("test") is not None]
    champ = next((m for m in out["models"] if m["kind"] == "champion"), None)
    if champ and champ.get("test") is not None:
        lower = _lower_better(tr)
        paired = {c.get("comparison"): c for c in ev.get("comparisons") or []}
        for o in others:
            better = (o["test"] < champ["test"]) if lower else (o["test"] > champ["test"])
            # The paired comparison (positive means the champion is better) settles it when the evaluation made one.
            c = next((v for k, v in paired.items() if str(k).endswith(f"against {o['model']}")), None)
            if c and c.get("low") is not None:
                if c["low"] > 0:
                    continue
                verdict = "better than" if c["high"] < 0 else "level with"
                out["flags"].append(f"On the test set {o['model']} ({o['test']:.4f}) is {verdict} the champion "
                                    f"({champ['test']:.4f}): paired difference {c['difference']:+.4f}, 95% interval "
                                    f"{c['low']:+.4f} to {c['high']:+.4f}.")
            elif better:
                out["flags"].append(f"On the test set {o['model']} ({o['test']:.4f}) scores better than the champion "
                                    f"({champ['test']:.4f}). There is no paired comparison, so check whether the "
                                    "difference is real.")
        if champ.get("cv") is not None:
            gap = champ["test"] - champ["cv"]
            if abs(gap) > 0.25 * abs(champ["cv"] or 1):
                out["flags"].append(f"The champion's test score ({champ['test']:.4f}) is far from its CV score "
                                    f"({champ['cv']:.4f}).")
        for o in others:
            if o.get("cv") is not None and o.get("test") is not None and abs(o["test"] - o["cv"]) > 0.25 * abs(o["cv"] or 1):
                out["flags"].append(f"{o['model']} scored {o['test']:.4f} on test against {o['cv']:.4f} in CV, so the "
                                    "CV folds did not predict its test score well.")
        if champ.get("bias") is not None and champ.get("test") is not None:
            out["flags"].append(f"The champion's average error (bias) on test is {champ['bias']:,.0f}.")
    if ev.get("score_psi") is not None and float(ev["score_psi"]) > 0.25:
        out["flags"].append(f"Large drift between training and test (prediction PSI {float(ev['score_psi']):.2f}, "
                            f"actual PSI {float(ev.get('target_psi') or 0):.2f}).")
    weak = [s for s in ev.get("slices") or [] if isinstance(s, dict) and s.get("worse than reference")]
    for s in weak:
        out["flags"].append(f"Weak slice: {s.get('slice')} = {s.get('value')}.")
    if st["test_look"] > 1:
        out["flags"].append(f"The test set has been opened {st['test_look']} times; scores after the first are optimistic.")
    return out


def best_trial_folds(spec, root):
    """Per fold CV scores of each model's best trial in the current data version, from the run log, as
    ({model: [scores in fold order]}, error or None)."""
    tr = spec.get("training") or {}
    runs_path = os.path.join(ml_common.project_paths(spec, root)["data_processed"], "training_runs.parquet")
    per_fold = {}
    if not os.path.exists(runs_path):
        return per_fold, None
    try:
        import pandas as pd
        runs = pd.read_parquet(runs_path)
        runs = runs[runs["data_version"] == tr.get("data_version")]
        for name, g in runs.groupby("model"):
            best = g.loc[g["primary_mean"].idxmin()] if _lower_better(tr) else g.loc[g["primary_mean"].idxmax()]
            per_fold[name] = json.loads(best["folds"])
    except Exception as e:  # pragma: no cover
        return per_fold, str(e)
    return per_fold, None


def bar_result(spec, root, entry=None):
    """The iteration bar for an iteration (the open one by default): the new champion against the champion carried
    forward, on the same folds. See ml_common.iteration_bar_evaluate."""
    e = entry or _open_entry(spec)
    bar = ml_common.iteration_bar(spec)
    out = {"status": "incomplete", "new": None, "carried": None, "detail": ""}
    if not e:
        out["detail"] = "No iteration is open."
        return out
    new = ((spec.get("training") or {}).get("champion") or {}).get("name")
    carried = (e.get("carried_forward") or {}).get("name")
    out.update(new=new, carried=carried)
    if not carried:
        out["detail"] = ("The previous champion was not carried forward into this training run, so the two cannot be "
                         "compared on the same folds. Run `iterate.py carry-forward` before training.")
        return out
    if not new:
        out["detail"] = "Training has not run in this iteration yet."
        return out
    if new == carried:
        out.update(status="fail", wins=0, detail=f"The carried champion {carried} is still the best model, so no new "
                   "model beat it.")
        return out
    per_fold, err = best_trial_folds(spec, root)
    res = ml_common.iteration_bar_evaluate(per_fold.get(new), per_fold.get(carried),
                                           _lower_better(spec.get("training") or {}), bar)
    if err:
        res["detail"] = f"Could not read the run log: {err}"
    out.update(res)
    out["detail"] = f"{new} against {carried}: " + res["detail"]
    return out


def _lower_better(tr):
    import importlib.util
    key = str(tr.get("primary_label") or "").split("@")[0]
    if importlib.util.find_spec("model_core"):
        import model_core
        return model_core.HIGHER_IS_BETTER.get(key) is False
    return key in ("rmse", "mae", "mape", "wape", "log_loss", "brier")


def _mentions_season(spec):
    text = json.dumps([spec.get("framing"), spec.get("eda"), spec.get("features"), spec.get("decisions"),
                       spec.get("open_questions")], default=str).lower()
    return "season" in text


def print_diagnosis(d):
    st = d["state"]
    L = []
    if not d["evaluated"]:
        L.append("The project has no offline evaluation yet. Run /ml-offline-evaluation before iterating.")
    L.append(f"Champion: {st['champion']} ({st['type']}). CV {st['metric']}: {_f(st['cv'])}"
             + (f" (std {_f(st['cv_std'])})" if st.get("cv_std") is not None else "")
             + f". Test: {_f(st['test'])}. Test set opened {st['test_look']} time(s).")
    if d["open_iteration"]:
        L.append(f"Iteration {d['open_iteration']} is open. Use `iterate.py remaining` to resume it.")
    L.append("\nTest against CV:")
    L.append("| Model | Kind | Test | 95% interval | CV | CV std | Per fold CV | Bias |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in d["models"]:
        pf = d["per_fold"].get(m["model"])
        L.append(f"| {m['model']} | {m['kind']} | {_f(m['test'])} | {_f(m['low'])} to {_f(m['high'])} | {_f(m['cv'])} | "
                 f"{_f(m['cv_std'])} | " + (" / ".join(f"{x:.3f}" for x in pf) if pf else "") + " | "
                 + (f"{m['bias']:,.0f}" if m.get("bias") is not None else "") + " |")
    if d["folds"]:
        L.append("\nValidation windows: " + "; ".join(
            f"fold {f.get('fold')} {f.get('valid_start')} to {f.get('valid_end')} ({f.get('n_valid')} rows, "
            f"trained on {f.get('n_train')})" for f in d["folds"]) + ".")
    L.append("\nEvidence from the evaluation:")
    L += [f"* {e.get('question')} {e.get('answer')}" for e in d["evidence"]] or ["* none recorded"]
    L.append("\nFlags:")
    L += [f"* {x}" for x in d["flags"]] or ["* none"]
    if d["conflicts"]:
        L.append("\nNotes that disagree (fix with /ml-reconcile-notes):")
        L += [f"* {c['detail']}" for c in d["conflicts"]]
    return L


def _f(x):
    return "" if x is None else (f"{x:.4f}" if isinstance(x, (int, float)) else str(x))


# ---------------------------------------------------------------- snapshot and restore

def _snapshot_files(spec, root, spec_path):
    paths = ml_common.project_paths(spec, root)
    files = [spec_path]
    files += glob.glob(os.path.join(paths["notebooks"], "*.ipynb"))
    files += glob.glob(os.path.join(paths["src"], "*.py"))
    files += glob.glob(os.path.join(paths["models"], "*.joblib"))
    files += glob.glob(os.path.join(paths["docs"], "*.md")) + glob.glob(os.path.join(paths["docs"], "*.yaml"))
    files += [os.path.join(paths["data_processed"], f) for f in SMALL_RESULTS
              if os.path.exists(os.path.join(paths["data_processed"], f))]
    return sorted(set(files))


def snapshot(spec, root, spec_path, dest):
    """Copy the current state to dest, keeping paths relative to the project root."""
    paths = ml_common.project_paths(spec, root)
    copied = []
    for src in _snapshot_files(spec, root, spec_path):
        rel = os.path.relpath(src, root)
        target = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)
    skipped = {}
    for p in glob.glob(os.path.join(paths["data_processed"], "*.parquet")):
        if os.path.basename(p) not in SMALL_RESULTS:
            skipped[os.path.relpath(p, root)] = os.path.getsize(p)
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"saved_at": _now(), "copied": copied, "not_copied_bytes": skipped}, f, indent=1)
    with open(os.path.join(dest, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("# Snapshot copies of models and result files, kept out of git like the originals.\n*.joblib\n*.parquet\n")
    return copied, skipped


def restore(root, src_dir, keep_iterations):
    """Copy a snapshot back over the project. The spec keeps the iterations section given."""
    with open(os.path.join(src_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for rel in manifest["copied"]:
        shutil.copy2(os.path.join(src_dir, rel), os.path.join(root, rel))
    spec_path = os.path.join(root, ml_common.SPEC_NAME)
    spec, _ = ml_common.load_or_create(spec_path)
    spec["iterations"] = keep_iterations
    return spec, manifest


# ---------------------------------------------------------------- open, remaining, carry forward, compare, close

def open_iteration(spec, root, spec_path, plan, dry_run=False):
    it = _iters(spec)
    problems = []
    if _open_entry(spec):
        problems.append(f"Iteration {it['current']} is still open. Close it first with `iterate.py close`.")
    if not (spec.get("evaluation") or {}).get("test_opened"):
        problems.append("There is no offline evaluation to iterate from. Run /ml-offline-evaluation first.")
    if not str(plan.get("hypothesis") or "").strip():
        problems.append("The plan needs a hypothesis: one sentence on what should improve, and why.")
    experiments = plan.get("experiments") or []
    changes = plan.get("changes") or []
    if experiments and not changes:   # a round: its changes are those of its experiments
        changes = [c for x in experiments for c in (x.get("changes") or [])]
    ids = [str(x.get("id") or "") for x in experiments]
    if any(not i or not i.replace("_", "").isalnum() for i in ids) or len(set(ids)) != len(ids):
        problems.append("Every experiment needs a short id of its own, in letters and digits (a, b, c).")
    for x in experiments:
        if not str(x.get("hypothesis") or "").strip() or not x.get("changes"):
            problems.append(f"Experiment {x.get('id')} needs a hypothesis and at least one change.")
    if not changes:
        problems.append("The plan lists no changes.")
    for c in changes:
        if c.get("stage") not in CHAIN + ["setup", "business", "discovery"]:
            problems.append(f"Unknown stage {c.get('stage')!r} in changes. Use one of: {', '.join(ml_common.STAGES)}.")
        if not str(c.get("change") or "").strip():
            problems.append(f"The change for {c.get('stage')} is empty.")
    n = max([e.get("id", 0) for e in it["entries"]], default=0) + 1
    dest = os.path.join(root, "iterations", f"iter_{n - 1}")
    if os.path.exists(dest):
        problems.append(f"{ml_common.relpath(dest, root)} already exists. Move it aside, or check the iterations section.")
    if problems:
        return {"ok": False, "problems": problems}
    rerun = _rerun_order(changes)
    entry = {"id": n, "opened_at": _now(), "hypothesis": plan["hypothesis"].strip(),
             "diagnosis": list(plan.get("diagnosis") or []), "changes": changes, "rerun": rerun,
             "start": _champion_state(spec), "snapshot": ml_common.relpath(dest, root), "status": "open",
             "start_hashes": {"features_config": _hash_obj((spec.get("features") or {}).get("config")),
                              "custom_features": _hash_file(os.path.join(ml_common.project_paths(spec, root)["src"],
                                                                         "custom_features.py")),
                              "split_cv": _hash_obj(((spec.get("split") or {}).get("cv") or {}).get("folds"))}}
    if experiments:
        entry["experiments"] = [{"id": str(x["id"]), "hypothesis": str(x["hypothesis"]).strip(),
                                 "changes": x["changes"],
                                 # an experiment never opens the test set: it is judged on CV
                                 "rerun": [st for st in _rerun_order(x["changes"]) if st != "evaluation"],
                                 "status": "planned", "backlog": x.get("backlog")} for x in experiments]
    if dry_run:
        return {"ok": True, "dry_run": True, "entry": entry, "files": _snapshot_files(spec, root, spec_path)}
    copied, skipped = snapshot(spec, root, spec_path, dest)
    notes = _start_version(spec, root, spec_path, entry)
    for x in entry.get("experiments") or []:
        x["experiment"] = f"{entry['version']}-{x['id']}"
    if not it.get("bar"):
        it["bar"] = dict(ml_common.ITERATION_BAR_DEFAULTS)
    it["entries"].append(entry)
    it["current"] = n
    _decision(spec, f"Opened iteration {n} ({entry['version']}): {entry['hypothesis']}",
              "Planned changes: " + "; ".join(f"{c['stage']}: {c['change']}" for c in changes))
    ml_common.log(spec, SKILL, f"opened iteration {n} as {entry['version']}")
    ml_common.save(spec_path, spec)
    parent = _version_parent_run(spec_path, entry)
    if parent:
        spec = ml_common.load(spec_path)
        for x in _iters(spec)["entries"]:
            if x.get("id") == n:
                x["mlflow_parent_run"] = parent
                entry["mlflow_parent_run"] = parent
        _version(spec)["mlflow_parent_run"] = parent
        ml_common.save(spec_path, spec)
        notes.append(f"MLflow: {entry['version']} parent run {parent[:8]}; every stage run of the iteration is logged under it")
    if _commit(root, spec, spec_path, f"{entry['version']} opened: {entry['hypothesis']}"):
        notes.append(f"git: committed the opening of {entry['version']}")
    return {"ok": True, "entry": entry, "copied": len(copied), "skipped": skipped, "notes": notes}


def remaining(spec):
    e = _open_entry(spec)
    if not e:
        return None, []
    opened = _ts(e["opened_at"])
    runs = spec.get("notebooks") or {}
    left = []
    for stage in e["rerun"]:
        t = _ts((runs.get(stage) or {}).get("run_at"))
        if not t or (opened and t < opened):
            left.append(stage)
    return e, left


def carry_forward(spec, root):
    """The previous champion as a model entry for the training answers, with its tuned settings fixed."""
    e = _open_entry(spec)
    if not e:
        return {"ok": False, "problems": ["No iteration is open."]}
    snap_spec_path = os.path.join(root, e["snapshot"], ml_common.SPEC_NAME)
    old = ml_common.load(snap_spec_path) if os.path.exists(snap_spec_path) else {}
    champ = (old.get("training") or {}).get("champion") or {}
    if not champ.get("name"):
        return {"ok": False, "problems": ["The snapshot has no champion to carry forward."]}
    model = {"name": f"{champ['name']}_iter{e['id'] - 1}", "type": champ.get("type"),
             "params": dict(champ.get("params") or {}), "class_weight": champ.get("class_weight"),
             "tuning": "none",
             "reason": f"The champion of iteration {e['id'] - 1}, carried forward with its tuned settings fixed, so it "
                       "is scored on the same folds and features as the new models."}
    if champ.get("inputs"):
        model["inputs"] = list(champ["inputs"])
    # Which columns does it need, and do they still exist?
    needed = set(model.get("inputs") or [])
    for v in model["params"].values():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            needed |= set(v)
    old_inputs = _input_names(old.get("features"))
    if not model.get("inputs"):
        needed |= set(old_inputs)
    current = set(_input_names(spec.get("features")))
    missing = sorted(needed - current) if current else []
    notes = []
    paths = ml_common.project_paths(spec, root)
    changed = []
    if _hash_obj((spec.get("features") or {}).get("config")) != e["start_hashes"]["features_config"]:
        changed.append("the feature settings")
    if _hash_file(os.path.join(paths["src"], "custom_features.py")) != e["start_hashes"]["custom_features"]:
        changed.append("src/custom_features.py")
    if changed:
        notes.append("Since the iteration opened, " + " and ".join(changed) + " changed. The carried model keeps its "
                     "settings but is fitted on the current features, so columns with the same name may now be built "
                     "differently.")
    if not model.get("inputs") and current and set(old_inputs) != current:
        new_cols = sorted(current - set(old_inputs))
        if new_cols:
            model["inputs"] = sorted(set(old_inputs) & current)
            notes.append("The carried model uses only the inputs it had before (" + str(len(model["inputs"]))
                         + "), not the new ones: " + ", ".join(new_cols) + ".")
    usable = not missing
    if missing:
        notes.append("These inputs no longer exist, so the old champion cannot be carried forward: "
                     + ", ".join(missing) + ". The comparison falls back to the recorded scores, which are not "
                     "like for like.")
    return {"ok": True, "usable": usable, "model": model, "missing": missing, "notes": notes}


def compare(spec, root):
    """Every iteration side by side, from the iterations section and the current state."""
    it = _iters(spec)
    rows = []
    entries = it["entries"]
    if entries:
        s = entries[0]["start"]
        rows.append({"iteration": 0, "champion": s.get("champion"), "cv": s.get("cv"), "test": s.get("test"),
                     "test_look": s.get("test_look"), "folds": entries[0]["start_hashes"]["split_cv"],
                     "outcome": "original", "carried_forward_cv": None, "bar": None})
    for e in entries:
        end = e.get("end") or (_champion_state(spec) if e.get("status") == "open" else {})
        lb = {r.get("model"): r for r in (spec.get("training") or {}).get("leaderboard") or []}
        carried = (e.get("carried_forward") or {}).get("name")
        ccv = end.get("carried_forward_cv")
        if ccv is None and carried and e.get("status") == "open":
            ccv = (lb.get(carried) or {}).get("primary_mean")
        folds = _hash_obj(((spec.get("split") or {}).get("cv") or {}).get("folds")) if e.get("status") == "open" \
            else end.get("folds")
        rows.append({"iteration": e["id"], "champion": end.get("champion"), "cv": end.get("cv"),
                     "test": end.get("test"), "test_look": end.get("test_look"), "folds": folds,
                     "outcome": e.get("outcome") or "open", "carried_forward_cv": ccv,
                     "hypothesis": e.get("hypothesis"),
                     "bar": ((e.get("bar_result") if e.get("status") != "open" else bar_result(spec, root, e))
                             or {}).get("status")})
    for r in rows:
        r["same_folds_as_previous"] = None
    for a, b in zip(rows, rows[1:]):
        b["same_folds_as_previous"] = a["folds"] == b["folds"]
    return rows


def _experiment(entry, exp_id):
    return next((x for x in (entry or {}).get("experiments") or [] if str(x.get("id")) == str(exp_id)), None)


def record_experiment(spec, spec_path, exp_id, result):
    """Record what one experiment of the open round found: its status, the iteration bar result, the CV scores,
    and where it ran. result is the experiment_result.json the experimenter wrote, or a dict with a status."""
    e = _open_entry(spec)
    if not e:
        return {"ok": False, "problems": ["No iteration is open."]}
    x = _experiment(e, exp_id)
    if x is None:
        return {"ok": False, "problems": [f"The open iteration has no experiment {exp_id!r}."]}
    status = result.get("status")
    if status is None:
        bar = (result.get("bar") or {}).get("status")
        status = "passed" if bar == "pass" else ("error" if result.get("problems") and not bar else "failed")
    if status not in EXPERIMENT_STATUS:
        return {"ok": False, "problems": [f"status must be one of {', '.join(EXPERIMENT_STATUS)}"]}
    x.update({"status": status, "recorded_at": _now()})
    for k in ("bar", "champion", "cv", "v1_cv", "workspace", "branch", "mlflow_parent_run", "problems", "summary"):
        if result.get(k) is not None:
            x[k] = result[k]
    ml_common.log(spec, SKILL, f"recorded experiment {x.get('experiment') or exp_id}: {status}")
    ml_common.save(spec_path, spec)
    return {"ok": True, "experiment": x}


def close_iteration(spec, root, spec_path, outcome, reason, do_restore=False, dry_run=False, adopted=None,
                    none_passed=False):
    e = _open_entry(spec)
    if not e:
        return {"ok": False, "problems": ["No iteration is open."]}
    if adopted is not None or none_passed:
        if not e.get("experiments"):
            return {"ok": False, "problems": ["--adopted and --none-passed close a round, and this iteration has no "
                                              "experiments."]}
        if adopted is not None and (outcome != "kept" or _experiment(e, adopted) is None):
            return {"ok": False, "problems": [f"--adopted needs --outcome kept and an experiment of this round "
                                              f"({', '.join(str(x['id']) for x in e['experiments'])})."]}
        if none_passed and outcome != "rolled_back":
            return {"ok": False, "problems": ["--none-passed needs --outcome rolled_back."]}
        do_restore = do_restore or none_passed   # nothing to restore: the project never changed
    if outcome not in OUTCOMES:
        return {"ok": False, "problems": [f"outcome must be one of {', '.join(OUTCOMES)}"]}
    if not str(reason or "").strip():
        return {"ok": False, "problems": ["Give the reason for the outcome."]}
    if outcome == "rolled_back" and not do_restore:
        return {"ok": False, "problems": ["Rolling back restores the snapshot over the current files. Ask the user, then "
                                          "rerun with --restore."]}
    _, left = remaining(spec)
    end = _champion_state(spec)
    end["folds"] = _hash_obj(((spec.get("split") or {}).get("cv") or {}).get("folds"))
    carried = (e.get("carried_forward") or {}).get("name")
    lb = {r.get("model"): r for r in (spec.get("training") or {}).get("leaderboard") or []}
    end["carried_forward_cv"] = (lb.get(carried) or {}).get("primary_mean") if carried else None
    if end.get("test") is not None and end["test_look"] <= e["start"].get("test_look", 0):
        end["test"] = None   # the evaluation shown is from before the iteration
    bar = bar_result(spec, root, e)
    if adopted is not None:
        x = _experiment(e, adopted)
        x["status"] = "adopted"
        e["adopted"] = x.get("experiment")
        bar = {**(x.get("bar") or {}), **{k: v for k, v in bar.items() if v is not None and k != "status"},
               "status": (x.get("bar") or {}).get("status") or bar.get("status")}
    elif none_passed:
        best = [x for x in e["experiments"] if (x.get("bar") or {}).get("status")]
        bar = {"status": "fail" if best else "incomplete",
               "detail": "No experiment of the round passed the iteration bar: " + "; ".join(
                   f"{x.get('experiment') or x['id']} {(x.get('bar') or {}).get('status', x.get('status'))}"
                   for x in e["experiments"])}
    for x in e.get("experiments") or []:
        if x.get("status") in ("planned", "running", "passed", "failed", "error") and x.get("id") != adopted:
            x["status"] = "discarded" if x.get("status") in ("passed", "failed") else x["status"]
    closed = {**e, "status": "closed", "closed_at": _now(), "outcome": outcome, "reason": reason.strip(), "end": end,
              "not_rerun": left, "bar_result": {k: bar.get(k) for k in ("status", "wins", "folds", "mean_diff", "spread",
                                                                         "detail")}}
    if dry_run:
        return {"ok": True, "dry_run": True, "entry": closed}
    it = _iters(spec)
    it["entries"] = [closed if x.get("id") == e["id"] else x for x in it["entries"]]
    it["current"] = None
    backup, manifest = None, None
    version_state = _version(spec)
    if outcome == "rolled_back" and not none_passed:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(root, ".iterate_backup", stamp)
        snapshot(spec, root, spec_path, backup)
        spec, manifest = restore(root, os.path.join(root, e["snapshot"]), it)
        spec["version"] = version_state   # the version record continues; the snapshot's is older
    _decision(spec, f"Closed iteration {e['id']} ({e.get('version', '')}): {outcome.replace('_', ' ')}.", reason.strip())
    ml_common.log(spec, SKILL, f"closed iteration {e['id']} ({outcome})")
    ml_common.save(spec_path, spec)
    notes, ver = _settle_version(spec, root, spec_path, closed, outcome, reason.strip(), adopted=e.get("adopted"))
    ml_common.save(spec_path, spec)
    notes += _settle_git(spec, root, spec_path, closed, outcome, ver, (manifest or {}).get("copied"))
    return {"ok": True, "entry": closed, "backup": ml_common.relpath(backup, root) if backup else None,
            "notes": notes}


def record_carry(spec, spec_path, cf):
    e = _open_entry(spec)
    e["carried_forward"] = {"name": cf["model"]["name"], "type": cf["model"]["type"], "params": cf["model"]["params"],
                            "inputs": cf["model"].get("inputs"), "usable": cf["usable"], "missing": cf["missing"]}
    ml_common.log(spec, SKILL, f"recorded the carried forward champion for iteration {e['id']}")
    ml_common.save(spec_path, spec)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["diagnose", "open", "remaining", "carry-forward", "compare", "bar", "close",
                                        "record-experiment"])
    ap.add_argument("plan", nargs="?", help="open: the plan file; record-experiment: the experiment id")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--json", action="store_true", help="Print the result as JSON")
    ap.add_argument("--dry-run", action="store_true", help="open, close: show what would happen without changing anything")
    ap.add_argument("--outcome", choices=OUTCOMES, help="close: kept, rolled_back, or stopped")
    ap.add_argument("--reason", help="close: why")
    ap.add_argument("--restore", action="store_true", help="close --outcome rolled_back: restore the snapshot")
    ap.add_argument("--adopted", help="close a round: the id of the experiment adopted into the project")
    ap.add_argument("--none-passed", action="store_true", help="close a round where no experiment passed the bar")
    ap.add_argument("--result", help="record-experiment: the experiment_result.json file")
    ap.add_argument("--status", choices=EXPERIMENT_STATUS, help="record-experiment: set the status directly")
    a = ap.parse_args()
    spec_path = ml_common.spec_path(a.spec)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found. Set the project up with /ml-project-setup first.")
    root = os.path.dirname(spec_path)
    os.chdir(root)
    sys.path.insert(0, root)   # so src.features can be imported
    spec, _ = ml_common.load_or_create(spec_path)

    if a.command == "diagnose":
        d = diagnose(spec, root)
        _say(d, a.json, print_diagnosis(d))

    elif a.command == "open":
        if not a.plan:
            ap.error("give the plan file")
        res = open_iteration(spec, root, spec_path, ml_common.load(a.plan), dry_run=a.dry_run)
        if not res["ok"]:
            print("Nothing was changed. Problems:\n" + "\n".join("  " + p for p in res["problems"]))
            sys.exit(1)
        e = res["entry"]
        if a.json:
            _say(res, True, [])
            return
        if res.get("dry_run"):
            print(f"Dry run. Iteration {e['id']} would save {len(res['files'])} files to {e['snapshot']}/ and rerun: "
                  + " → ".join(e["rerun"]) + ".")
            return
        print(f"Opened iteration {e['id']} as {e.get('version')}: {e['hypothesis']}")
        for n_ in res.get("notes") or []:
            print("  " + n_)
        print(f"Snapshot: {e['snapshot']}/ ({res['copied']} files; large data files not copied: "
              + (", ".join(res["skipped"]) or "none") + ")")
        print("Rerun, in this order: " + " → ".join(f"/{ml_common.STAGE_SKILL[s]}" for s in e["rerun"]))
        if "training" in e["rerun"]:
            print("Before training, run `iterate.py carry-forward` to add the previous champion to the models.")

    elif a.command == "remaining":
        e, left = remaining(spec)
        if not e:
            _say({"open": None}, a.json, ["No iteration is open."])
            return
        _say({"open": e["id"], "remaining": left}, a.json,
             [f"Iteration {e['id']} is open: {e['hypothesis']}",
              "Still to rerun: " + (" → ".join(f"/{ml_common.STAGE_SKILL[s]}" for s in left) if left else
                                    "none. Close it with `iterate.py close`.")])

    elif a.command == "carry-forward":
        cf = carry_forward(spec, root)
        if not cf["ok"]:
            print("\n".join(cf["problems"]))
            sys.exit(1)
        record_carry(spec, spec_path, cf)
        if a.json:
            _say(cf, True, [])
            return
        print("Add this model to the models in the training answers:" if cf["usable"] else
              "The previous champion cannot be carried forward:")
        if cf["usable"]:
            import yaml
            print(yaml.safe_dump([cf["model"]], sort_keys=False, allow_unicode=True, width=110))
        for n in cf["notes"]:
            print("Note: " + n)

    elif a.command == "compare":
        rows = compare(spec, root)
        if a.json:
            _say(rows, True, [])
            return
        if not rows:
            print("No iterations yet.")
            return
        label = (spec.get("training") or {}).get("primary_label") or "metric"
        print(f"| Iteration | Champion | CV {label} | Test {label} | Test look | Same folds as before | "
              "Previous champion on these folds | Iteration bar | Outcome |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            same = "" if r["same_folds_as_previous"] is None else ("yes" if r["same_folds_as_previous"] else "no")
            print(f"| {r['iteration']} | {r['champion']} | {_f(r['cv'])} | {_f(r['test'])} | {r['test_look'] or ''} | "
                  f"{same} | {_f(r['carried_forward_cv'])} | {r.get('bar') or ''} | {r['outcome']} |")
        if any(r["same_folds_as_previous"] is False for r in rows):
            print("\nCV scores on different folds are not like for like. Compare with the previous champion's score "
                  "on the same folds where it was carried forward.")
        if any((r.get("test_look") or 0) > 1 for r in rows):
            print("Test scores from the second look onwards are optimistic.")

    elif a.command == "bar":
        res = bar_result(spec, root)
        res["bar"] = ml_common.iteration_bar(spec)
        lines = [f"Iteration bar: {res['status']}. {res['detail']}",
                 f"The bar: win on more than {res['bar']['min_fold_win_share']:.0%} of the folds, with a mean improvement "
                 f"larger than {res['bar']['spread_multiple']:g} times the spread of the per fold differences "
                 f"(at least {res['bar']['min_folds']} folds). Edit it under iterations.bar in the spec."]
        _say(res, a.json, lines)

    elif a.command == "record-experiment":
        if not a.plan or not (a.result or a.status):
            ap.error("record-experiment needs the experiment id and --result or --status")
        result = {}
        if a.result:
            with open(a.result, encoding="utf-8") as f:
                result = json.load(f)
        if a.status:
            result["status"] = a.status
        res = record_experiment(spec, spec_path, a.plan, result)
        if not res["ok"]:
            print("Nothing was changed. " + " ".join(res["problems"]))
            sys.exit(1)
        x = res["experiment"]
        _say(res, a.json, [f"Experiment {x.get('experiment') or x['id']}: {x['status']}. "
                           f"Iteration bar: {(x.get('bar') or {}).get('status')}. {(x.get('bar') or {}).get('detail') or ''}"])

    else:
        if not a.outcome or not a.reason:
            ap.error("close needs --outcome and --reason")
        res = close_iteration(spec, root, spec_path, a.outcome, a.reason, do_restore=a.restore, dry_run=a.dry_run,
                              adopted=a.adopted, none_passed=a.none_passed)
        if not res["ok"]:
            print("Nothing was changed. " + " ".join(res["problems"]))
            sys.exit(1)
        e = res["entry"]
        if a.json:
            _say(res, True, [])
            return
        print(("Dry run. Would close" if res.get("dry_run") else "Closed") + f" iteration {e['id']} as {a.outcome}.")
        end = e["end"]
        print(f"End state: champion {end.get('champion')}, CV {_f(end.get('cv'))}, test {_f(end.get('test'))}, "
              f"test look {end.get('test_look')}.")
        print(f"Iteration bar: {(e.get('bar_result') or {}).get('status')}. {(e.get('bar_result') or {}).get('detail')}")
        if a.outcome == "kept" and (e.get("bar_result") or {}).get("status") == "fail":
            print("Kept although the bar failed: record why in a decision that names this iteration and the bar, or "
                  "/ml-status will flag it.")
        if e.get("not_rerun"):
            print("Not rerun in this iteration: " + ", ".join(e["not_rerun"]) + ".")
        for x in e.get("experiments") or []:
            print(f"Experiment {x.get('experiment') or x['id']}: {x.get('status')}, bar "
                  f"{(x.get('bar') or {}).get('status')}")
        if res.get("backup"):
            print(f"Restored {e['snapshot']}/. The state before the rollback is in {res['backup']}/. The split and "
                  "feature tables were not in the snapshot: rerun the split and features notebooks to rebuild them.")
        for n_ in res.get("notes") or []:
            print("  " + n_)
        if not res.get("dry_run"):
            print("Next: /ml-reconcile-notes to align the notes, then /ml-status.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
