#!/usr/bin/env python3
"""The driver of an ML project run by an agent: what to do next, the checkpoints, the backlog of experiments, and
the rounds. It writes only the autopilot section of the spec; ml-iterate's iterate.py writes the iterations section,
and workspace.py makes the experiments' worktrees.

Commands (run from the project root with the project's Python; every command prints JSON with --json):
    autopilot.py start [--experiments-per-round K] [--max-rounds N] [--attended]   # record the run's settings
    autopilot.py next                     # the next action, from the spec
    autopilot.py checkpoint write <kind>  # docs/checkpoint.md for business, framing, test_v1 or test_final; stop
    autopilot.py checkpoint clear <kind>  # after the person reviewed it
    autopilot.py backlog add <items.yaml> # candidate experiments: [{stage, change, hypothesis, cost}]
    autopilot.py backlog list
    autopilot.py round open               # take the next K backlog items, open the round, make the workspaces
    autopilot.py round close              # adopt the best experiment that passed, discard the rest, close the round
    autopilot.py stop --reason "..."      # stop iterating early (the data cannot support more, say)
    autopilot.py limits                   # rounds used, failures in a row, and whether another round is allowed
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import git_ops  # noqa: E402
import ml_common  # noqa: E402
import workspace  # noqa: E402

SKILL = "ml-autopilot"
# The stages the agent runs without a person, in order. Checkpoints sit before the business and ML framing are
# confirmed and before every opening of the test set.
UNATTENDED = {"discovery", "eda", "split", "features", "baseline", "training"}


def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _ap(spec):
    ap = spec.get("autopilot")
    if not isinstance(ap, dict):
        ap = spec["autopilot"] = {}
    ap.setdefault("checkpoints", [])
    ap.setdefault("backlog", [])
    return ap


def _save(spec_path, spec, action, extra=()):
    """Save the spec and commit it (with any extra paths) on the version branch, like every other skill."""
    ml_common.log(spec, SKILL, action)
    ml_common.save(spec_path, spec)
    if (spec.get("git") or {}).get("auto_commit", True):
        git_ops.commit_paths(os.path.dirname(spec_path), [spec_path, *extra], f"autopilot: {action}", spec,
                             trailers={"Skill": SKILL, "Agent-Session": os.environ.get("ML_AGENT_SESSION")})


# ---------------------------------------------------------------- the rounds and their limits

def rounds(spec):
    return [e for e in ((spec.get("iterations") or {}).get("entries") or []) if e.get("experiments")]


def limits(spec):
    bar = ml_common.iteration_bar(spec)
    settings = ml_common.autopilot_settings(spec)
    closed = [e for e in rounds(spec) if e.get("status") == "closed"]
    fails = 0
    for e in reversed(closed):
        if e.get("outcome") == "kept":
            break
        fails += 1
    max_rounds = int(min(settings.get("max_rounds") or bar["max_iterations"], bar["max_iterations"]))
    stop = (_ap(spec).get("stopped") or {}).get("reason")
    reason = None
    if stop:
        reason = f"stopped: {stop}"
    elif len(closed) >= max_rounds:
        reason = f"{len(closed)} of {max_rounds} rounds used"
    elif fails >= int(bar["max_consecutive_fails"]):
        reason = f"{fails} rounds in a row found nothing that passed the iteration bar"
    return {"rounds": len(closed), "max_rounds": max_rounds, "fails_in_a_row": fails,
            "max_consecutive_fails": int(bar["max_consecutive_fails"]), "another_round": reason is None,
            "reason": reason, "backlog_open": len([b for b in _ap(spec)["backlog"] if b.get("status") == "open"])}


# ---------------------------------------------------------------- what to do next

def _checkpoint(spec, kind):
    """The latest checkpoint of a kind, or None."""
    cps = [c for c in _ap(spec)["checkpoints"] if c.get("kind") == kind]
    return cps[-1] if cps else None


def _drafted(spec, section):
    """A business or framing section that the agent has drafted in full and that waits for review."""
    sec = spec.get(section) or {}
    if sec.get("status") != "draft":
        return False
    _, missing, _ = ml_common.validate(spec)
    return not [m for m in missing if m.startswith(section + ".")]


def _test_looks(spec):
    return [o.get("at") for o in ((spec.get("evaluation") or {}).get("test_opened") or []) if isinstance(o, dict)]


def next_action(spec):
    """{action, ...}. Actions: ask_goal, run_stage, checkpoint, review, open_round, await_experiments,
    close_round, manual_iteration, done."""
    ap = _ap(spec)
    for c in ml_common.open_checkpoints(spec):
        return {"action": "review", "checkpoint": c["kind"], "file": c.get("file"),
                "why": "A checkpoint is waiting for the person's review."}
    if not ml_common.step_done(spec, "setup"):
        return {"action": "run_stage", "stage": "setup", "skill": "ml-project-setup", "attended": True,
                "why": "The project is not set up."}
    if not ml_common.anchors_done(spec) or not ap.get("settings"):
        return {"action": "ask_goal", "skill": "ml-business-framing",
                "why": "The mode, the two anchors, and the run's settings come from the person before the data."}
    step, skill, how = ml_common.next_step(spec)
    if step in ("business", "framing"):
        cp = _checkpoint(spec, step)
        if _drafted(spec, step) and not (cp and cp.get("status") == "cleared"):
            return {"action": "checkpoint", "checkpoint": step,
                    "why": f"The {step} section is drafted; the person reviews it before it is confirmed."}
        return {"action": "run_stage", "stage": step, "skill": skill, "attended": False, "mode": "draft",
                "why": f"Draft the {step} section from the data; it is confirmed only at the checkpoint."}
    if step in UNATTENDED:
        return {"action": "run_stage", "stage": step, "skill": skill, "attended": False, "why": how}
    if step == "evaluation":
        cp = _checkpoint(spec, "test_v1")
        if not (cp and cp.get("status") == "cleared"):
            return {"action": "checkpoint", "checkpoint": "test_v1",
                    "why": "Training is done; opening the test set needs the person's agreement."}
        return {"action": "run_stage", "stage": "evaluation", "skill": "ml-offline-evaluation", "attended": True,
                "why": "The person agreed to open the test set."}
    if step is not None:
        return {"action": "run_stage", "stage": step, "skill": skill, "attended": False, "why": how}

    # Every step is done: the rounds of experiments
    it = spec.get("iterations") or {}
    cur = next((e for e in it.get("entries") or [] if e.get("id") == it.get("current") and e.get("status") == "open"),
               None)
    if cur is not None:
        if not cur.get("experiments"):
            return {"action": "manual_iteration", "iteration": cur["id"],
                    "why": "An iteration opened by hand is open. Finish it with /ml-iterate first."}
        waiting = [x["id"] for x in cur["experiments"] if x.get("status") in ("planned", "running")]
        if waiting:
            return {"action": "await_experiments", "iteration": cur["id"], "experiments": waiting,
                    "why": "Experiments of the round have not reported yet."}
        return {"action": "close_round", "iteration": cur["id"], "why": "Every experiment of the round reported."}
    lim = limits(spec)
    if lim["another_round"]:
        return {"action": "open_round", "limits": lim,
                "why": "Diagnose, add the proposed changes to the backlog, then open a round. When the diagnosis "
                       "finds nothing worth trying, stop with `autopilot.py stop`."}
    kept = [e for e in rounds(spec) if e.get("outcome") == "kept"]
    looks = _test_looks(spec)
    if kept and (not looks or str(kept[-1].get("closed_at")) > str(looks[-1])):
        cp = _checkpoint(spec, "test_final")
        if not (cp and cp.get("status") == "cleared" and str(cp.get("cleared_at")) > str(kept[-1].get("closed_at"))):
            return {"action": "checkpoint", "checkpoint": "test_final", "limits": lim,
                    "why": f"Iterating has finished ({lim['reason']}); the final champion needs a look at the test "
                           "set, which is optimistic now."}
        return {"action": "run_stage", "stage": "evaluation", "skill": "ml-offline-evaluation", "attended": True,
                "why": "The person agreed to open the test set for the final champion."}
    return {"action": "done", "limits": lim,
            "why": f"Iterating has finished ({lim['reason']})." + ("" if kept else " No round was kept, so the "
                                                                    "evaluated champion stands.")}


# ---------------------------------------------------------------- checkpoints

def _fmt(v, width=160):
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False, default=str)
    s = " ".join(str(v).split())
    return (s[: width - 1] + "…") if len(s) > width else s


def _section_rows(spec, section):
    rows = []

    def walk(prefix, val):
        if isinstance(val, dict) and val and prefix.count(".") < 2:
            for k, v in val.items():
                if k in ("mlflow", "history"):
                    continue
                walk(f"{prefix}.{k}", v)
        else:
            rows.append((prefix, val))
    for k, v in (spec.get(section) or {}).items():
        if k not in ("status", "mlflow"):
            walk(f"{section}.{k}", v)
    return rows


def _red_flags(spec, section):
    _, missing, recommended = ml_common.validate(spec)
    flags = [f"Required and empty: {m}" for m in missing if m.startswith(section + ".")]
    flags += [f"Recommended and empty: {m}" for m in recommended if m.startswith(section + ".")]
    flags += [c["detail"] for c in ml_common.consistency_checks(spec)
              if any(str(w).startswith(section) for w in c.get("where") or [])]
    flags += [f"{q.get('id')}: {q.get('question')}" for q in spec.get("open_questions") or []
              if isinstance(q, dict) and q.get("status") == "open" and q.get("stage") in (section, "business")]
    return flags


def checkpoint_text(spec, kind):
    proj = (spec.get("project") or {}).get("name")
    lines = [f"# Checkpoint: {kind.replace('_', ' ')}", "",
             f"Project {proj}. Written by ml-autopilot on {_now()}. The agent has stopped and waits for you. Resume "
             "with `/ml-autopilot`; it goes through this page with you first.", ""]
    if kind in ("business", "framing"):
        lines += [f"## Drafts in the {kind} section", "",
                  "Every value below is a draft the agent proposed from the data. Mark the rows you dispute; the rest "
                  "are accepted as they are.", "", "| Field | Draft |", "|---|---|"]
        lines += [f"| `{k}` | {_fmt(v).replace('|', '/')} |" for k, v in _section_rows(spec, kind)]
        drafts = [a for a in spec.get("assumptions") or [] if isinstance(a, dict)
                  and "autopilot" in str(a.get("source") or "")]
        if drafts:
            lines += ["", "## Assumptions the agent made", ""] + [f"* {_fmt(a.get('assumption') or a.get('text') or a)}"
                                                                  for a in drafts]
        flags = _red_flags(spec, kind)
        lines += ["", "## Red flags", ""] + ([f"* {f}" for f in flags] or ["* None found."])
        lines += ["", "## What resuming does", "",
                  f"The agent goes one question at a time through the rows you dispute and the red flags, reruns "
                  f"{ml_common.STAGE_SKILL.get(kind, kind)} with the reviewed answers, and marks the section "
                  "confirmed. Then it carries on unattended until the next checkpoint."]
    else:
        tr, ev = spec.get("training") or {}, spec.get("evaluation") or {}
        champ = tr.get("champion") or {}
        label = tr.get("primary_label")
        lines += ["## The champion on CV", "",
                  f"Champion {champ.get('name')} ({champ.get('type')}), CV {label} {champ.get(label) if label else ''}.",
                  "", "| Model | CV mean | CV std |", "|---|---|---|"]
        for r in (tr.get("leaderboard") or [])[:10]:
            lines.append(f"| {r.get('model')} | {_fmt(r.get('primary_mean'))} | {_fmt(r.get('primary_std'))} |")
        v1 = tr.get("v1") or {}
        if v1:
            lines += ["", f"V1 bar on CV: {v1.get('overall') or v1.get('status')}."]
        if kind == "test_final":
            lines += ["", "## The rounds of experiments", "", "| Round | Experiment | Change | Bar | Status |",
                      "|---|---|---|---|---|"]
            for e in rounds(spec):
                for x in e.get("experiments") or []:
                    ch = "; ".join(f"{c.get('stage')}: {c.get('change')}" for c in x.get("changes") or [])
                    lines.append(f"| {e['id']} | {x.get('experiment')} | {_fmt(ch, 90)} | "
                                 f"{(x.get('bar') or {}).get('status')} | {x.get('status')} |")
            lines += ["", f"Limits: {limits(spec)['reason']}."]
        looks = len(ev.get("test_opened") or [])
        lines += ["", "## Opening the test set", "",
                  f"The test set has been opened {looks} time(s). " + (
                      "This look is the first, so the score is honest." if looks == 0 else
                      "Another look makes the score optimistic: the choices since the last look were made "
                      "knowing it."), "",
                  "Resuming asks whether to open it. If you say no, the agent stops with the CV result."]
    return "\n".join(lines) + "\n"


def checkpoint_write(root, spec_path, spec, kind):
    if kind not in ml_common.CHECKPOINT_KINDS:
        return {"ok": False, "problems": [f"kind must be one of {', '.join(ml_common.CHECKPOINT_KINDS)}"]}
    docs = ml_common.project_paths(spec, root)["docs"]
    os.makedirs(docs, exist_ok=True)
    path = os.path.join(docs, "checkpoint.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(checkpoint_text(spec, kind))
    ap = _ap(spec)
    for c in ap["checkpoints"]:
        if c.get("kind") == kind and c.get("status") == "open":
            c["status"] = "cleared"
            c["cleared_at"] = _now()
            c["note"] = "replaced by a newer checkpoint"
    ap["checkpoints"].append({"kind": kind, "status": "open", "opened_at": _now(),
                              "file": ml_common.relpath(path, root)})
    _save(spec_path, spec, f"checkpoint {kind}: stopped for review", extra=[path])
    proj = (spec.get("project") or {}).get("name")
    return {"ok": True, "file": path,
            "notification": f"{proj}: the ML agent stopped at the {kind.replace('_', ' ')} checkpoint. "
                            "Review docs/checkpoint.md and resume with /ml-autopilot."}


def checkpoint_clear(spec_path, spec, kind, note=None):
    cps = [c for c in _ap(spec)["checkpoints"] if c.get("kind") == kind and c.get("status") == "open"]
    if not cps:
        return {"ok": False, "problems": [f"No open {kind} checkpoint."]}
    for c in cps:
        c.update({"status": "cleared", "cleared_at": _now(), "note": note})
    _save(spec_path, spec, f"checkpoint {kind}: reviewed and cleared")
    return {"ok": True, "checkpoint": kind}


# ---------------------------------------------------------------- the backlog

def backlog_add(spec_path, spec, items):
    ap = _ap(spec)
    have = {(b.get("stage"), b.get("change")) for b in ap["backlog"]}
    added, problems = [], []
    n = max([int(str(b.get("id", "0")).lstrip("B") or 0) for b in ap["backlog"]], default=0)
    for it in items or []:
        if it.get("stage") not in ml_common.STAGES or not str(it.get("change") or "").strip() \
                or not str(it.get("hypothesis") or "").strip():
            problems.append(f"Each item needs a stage, a change, and a hypothesis: {it}")
            continue
        if (it["stage"], it["change"]) in have:
            continue
        n += 1
        b = {"id": f"B{n}", "stage": it["stage"], "change": str(it["change"]).strip(),
             "hypothesis": str(it["hypothesis"]).strip(), "cost": int(it.get("cost") or 2),
             "source": it.get("source") or "diagnosis", "status": "open", "added_at": _now()}
        ap["backlog"].append(b)
        added.append(b)
    if added:
        _save(spec_path, spec, f"backlog: added {', '.join(b['id'] for b in added)}")
    return {"ok": not problems, "added": added, "problems": problems}


def _pick(spec, k):
    """The next k open items: cheapest first, then the order they were added; at most one per stage, so the
    experiments of a round test different things."""
    items = sorted([b for b in _ap(spec)["backlog"] if b.get("status") == "open"],
                   key=lambda b: (b.get("cost", 2), int(str(b["id"]).lstrip("B") or 0)))
    out, stages = [], set()
    for b in items:
        if b["stage"] in stages:
            continue
        out.append(b)
        stages.add(b["stage"])
        if len(out) == k:
            break
    for b in items:   # fill up when there are fewer stages than k
        if len(out) == k:
            break
        if b not in out:
            out.append(b)
    return out


# ---------------------------------------------------------------- rounds

def round_open(root, spec_path, spec, scratch):
    act = next_action(spec)
    if act["action"] != "open_round":
        return {"ok": False, "problems": [f"A round cannot open now: {act['why']}"]}
    k = int(ml_common.autopilot_settings(spec)["experiments_per_round"])
    picked = _pick(spec, k)
    if not picked:
        return {"ok": False, "problems": ["The backlog has no open items. Diagnose and add some with "
                                          "`autopilot.py backlog add` first."]}
    letters = "abcdefghij"
    plan = {"hypothesis": "Round of " + str(len(picked)) + " experiments: " + "; ".join(b["hypothesis"] for b in picked),
            "diagnosis": [f"{b['id']}: {b['change']}" for b in picked],
            "experiments": [{"id": letters[i], "hypothesis": b["hypothesis"],
                             "changes": [{"stage": b["stage"], "change": b["change"]}], "backlog": b["id"]}
                            for i, b in enumerate(picked)]}
    os.makedirs(scratch, exist_ok=True)
    plan_path = os.path.join(scratch, f"round_plan_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(ml_common.dump_text(plan))
    res = workspace.run_iterate(root, "open", plan_path)
    if res.get("ok") is False:
        return {"ok": False, "problems": res.get("problems"), "plan": plan_path}
    spec = ml_common.load(spec_path)
    entry = workspace.open_round(spec)
    for b in _ap(spec)["backlog"]:
        if b["id"] in {p["id"] for p in picked}:
            b.update({"status": "in_round", "round": entry["id"]})
    _save(spec_path, spec, f"opened round {entry['id']} with {', '.join(p['id'] for p in picked)}")
    created = []
    for x in entry.get("experiments") or []:
        w = workspace.create(root, spec_path, x["id"])
        created.append(w if w.get("ok") else {"id": x["id"], "ok": False, "problems": w.get("problems")})
    return {"ok": all(w.get("ok") for w in created), "round": entry["id"], "version": entry.get("version"),
            "plan": plan_path, "experiments": created}


def round_close(root, spec_path, spec):
    entry = workspace.open_round(spec)
    if not entry or not entry.get("experiments"):
        return {"ok": False, "problems": ["No round is open."]}
    waiting = [x["id"] for x in entry["experiments"] if x.get("status") in ("planned", "running")]
    if waiting:
        return {"ok": False, "problems": [f"Experiments {', '.join(waiting)} have not reported yet. Record them with "
                                          "`iterate.py record-experiment` (status error when an agent failed)."]}
    passed = sorted([x for x in entry["experiments"] if x.get("status") == "passed"],
                    key=lambda x: -((x.get("bar") or {}).get("mean_diff") or 0))
    winner = passed[0] if passed else None
    notes = []
    if winner is not None:   # adopt first: when the merge fails, nothing else of the round has changed yet
        a = workspace.adopt(root, spec_path, winner["id"])
        if not a.get("ok"):
            return {"ok": False, "problems": a.get("problems"), "notes": notes}
        notes += a.get("notes") or []
    for x in entry["experiments"]:
        if winner is not None and x["id"] == winner["id"]:
            continue
        why = ("another experiment of the round did better" if x.get("status") == "passed"
               else f"{x.get('status')}: {(x.get('bar') or {}).get('detail') or ''}".strip())
        d = workspace.discard(root, spec_path, x["id"], why)
        notes += d.get("notes") or d.get("problems") or []
    if winner is not None:
        reason = f"{winner['experiment']} passed the iteration bar: {(winner.get('bar') or {}).get('detail')}"
        res = workspace.run_iterate(root, "close", "--outcome", "kept", "--adopted", str(winner["id"]),
                                    "--reason", reason)
    else:
        reason = "No experiment of the round passed the iteration bar."
        res = workspace.run_iterate(root, "close", "--outcome", "rolled_back", "--none-passed", "--reason", reason)
    if res.get("ok") is False:
        return {"ok": False, "problems": res.get("problems"), "notes": notes}
    notes += res.get("notes") or []
    spec = ml_common.load(spec_path)
    closed = next((e for e in (spec.get("iterations") or {}).get("entries") or [] if e.get("id") == entry["id"]), {})
    tried = {x.get("backlog"): x for x in closed.get("experiments") or []}
    for b in _ap(spec)["backlog"]:
        x = tried.get(b["id"])
        if x is not None and b.get("status") == "in_round":
            b["status"] = "adopted" if x.get("status") == "adopted" else "tried"
            b["result"] = {"experiment": x.get("experiment"), "bar": (x.get("bar") or {}).get("status")}
    _save(spec_path, spec, f"closed round {entry['id']}: " + (f"adopted {winner['experiment']}" if winner else
                                                               "nothing passed"))
    return {"ok": True, "round": entry["id"], "adopted": winner and winner.get("experiment"), "notes": notes,
            "limits": limits(spec)}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["start", "next", "checkpoint", "backlog", "round", "stop", "limits"])
    ap.add_argument("sub", nargs="?", help="checkpoint: write or clear; backlog: add or list; round: open or close")
    ap.add_argument("arg", nargs="?", help="checkpoint: the kind; backlog add: the items file")
    ap.add_argument("--experiments-per-round", type=int)
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--attended", action="store_true", help="start: ask before each unattended stage")
    ap.add_argument("--reason", help="stop: why; checkpoint clear: a note")
    ap.add_argument("--scratch", help="round open: where to write the round plan (default: .worktrees/plans)")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--json", action="store_true", help="accepted for symmetry: the output is always JSON")
    a = ap.parse_args()
    spec_path = ml_common.spec_path(a.spec)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found. Set the project up with /ml-project-setup first.")
    root = os.path.dirname(os.path.abspath(spec_path))
    if ml_common.workspace_info(root):
        sys.exit("This is an experiment workspace. The autopilot runs from the main project.")
    os.chdir(root)
    spec = ml_common.migrate(ml_common.load(spec_path))

    if a.command == "start":
        ap_ = _ap(spec)
        settings = {**ml_common.autopilot_settings(spec)}
        if a.experiments_per_round:
            settings["experiments_per_round"] = max(1, min(a.experiments_per_round, 5))
        if a.max_rounds:
            settings["max_rounds"] = max(1, a.max_rounds)
        settings["unattended"] = not a.attended
        ap_["settings"] = settings
        ap_.setdefault("started_at", _now())
        ap_.pop("stopped", None)
        _save(spec_path, spec, "autopilot settings: " + ", ".join(f"{k} {v}" for k, v in settings.items()))
        res = {"ok": True, "settings": settings}
    elif a.command == "next":
        res = {"ok": True, **next_action(spec)}
    elif a.command == "limits":
        res = {"ok": True, **limits(spec)}
    elif a.command == "stop":
        _ap(spec)["stopped"] = {"at": _now(), "reason": a.reason or "stopped by the agent"}
        _save(spec_path, spec, f"autopilot stopped iterating: {a.reason}")
        res = {"ok": True, "stopped": _ap(spec)["stopped"]}
    elif a.command == "checkpoint":
        if a.sub not in ("write", "clear") or not a.arg:
            ap.error("checkpoint write|clear <kind>")
        res = (checkpoint_write(root, spec_path, spec, a.arg) if a.sub == "write"
               else checkpoint_clear(spec_path, spec, a.arg, a.reason))
    elif a.command == "backlog":
        if a.sub == "add":
            if not a.arg:
                ap.error("backlog add <items.yaml>")
            import yaml
            with open(a.arg, encoding="utf-8") as f:
                items = yaml.safe_load(f)
            res = backlog_add(spec_path, spec, items if isinstance(items, list) else (items or {}).get("items"))
        else:
            res = {"ok": True, "backlog": _ap(spec)["backlog"]}
    else:
        if a.sub == "open":
            res = round_open(root, spec_path, spec, a.scratch or os.path.join(root, workspace.WORKTREES, "plans"))
        elif a.sub == "close":
            res = round_close(root, spec_path, spec)
        else:
            ap.error("round open|close")
    print(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    sys.exit(0 if res.get("ok", True) else 1)


if __name__ == "__main__":
    main()
