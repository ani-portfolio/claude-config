#!/usr/bin/env python3
"""The driver of an ML project run by an agent: what to do next, the checkpoints, the backlog of experiments, and
the rounds. It writes only the autopilot section of the spec; ml-iterate's iterate.py writes the iterations section,
and workspace.py makes the experiments' worktrees.

Commands (run from the project root with the project's Python; every command prints JSON with --json):
    autopilot.py start [--max-parallel K] [--attended]   # record the run's settings
    autopilot.py next                     # the next action, from the spec
    autopilot.py checkpoint write <kind>  # docs/checkpoint.md for business, framing, test_v1 or test_final; stop
    autopilot.py checkpoint clear <kind>  # after the person reviewed it
    autopilot.py backlog add <items.yaml> # candidate experiments: [{stage, change, hypothesis, cost}]
    autopilot.py backlog list
    autopilot.py round open               # open a round on the open backlog items and make the first workspaces
    autopilot.py round launch             # make workspaces for planned experiments, up to max_parallel running
    autopilot.py round advance            # judge the finished phase and plan the next one, or settle the round
    autopilot.py round close              # adopt what the round settled on, discard the rest, close the round
    autopilot.py stop --reason "..."      # stop iterating early (the data cannot support more, say)
    autopilot.py limits                   # rounds used, and whether another round is allowed

A round is forward selection over the backlog, in phases:
    screen   every open item alone, judged against the champion carried forward
    combine  every item that passed, together (skipped when only one passed)
    ablate   the combination without each item in turn; an item goes when taking it out passes the bar
    verify   what is left after the ablation, against the champion again
    greedy   when the combination fails: add the items that passed one at a time, best first, each kept only
             when it passes the bar against the set before it
Items upstream of the split (the split itself, the framing, the data) change the folds, so they cannot be compared
fold by fold with the others: each one gets a round of its own, before the rest.
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
# Backlog items in these stages change the folds, so each one is screened in a round of its own.
UPSTREAM = {"setup", "business", "discovery", "framing", "eda", "split"}
# Stages before training: taking one of their changes out of a combination is scored with the combination's model
# and its tuned settings fixed, so the comparison measures the change and not the luck of tuning.
BEFORE_TRAINING = UPSTREAM | {"features", "baseline"}


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
    """Rounds stop at iterations.bar.max_iterations, when stopped by hand, or when the last round adopted nothing
    and the backlog has nothing left to try (a round tries every open item, so a new diagnosis would only repeat
    it)."""
    max_rounds = int(ml_common.iteration_bar(spec)["max_iterations"])
    closed = [e for e in rounds(spec) if e.get("status") == "closed"]
    open_items = len([b for b in _ap(spec)["backlog"] if b.get("status") == "open"])
    stop = (_ap(spec).get("stopped") or {}).get("reason")
    reason = None
    if stop:
        reason = f"stopped: {stop}"
    elif len(closed) >= max_rounds:
        reason = f"{len(closed)} of {max_rounds} rounds used"
    elif closed and closed[-1].get("outcome") != "kept" and not open_items:
        reason = "the last round adopted nothing, and the backlog has nothing left to try"
    return {"rounds": len(closed), "max_rounds": max_rounds, "another_round": reason is None, "reason": reason,
            "backlog_open": open_items}


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
    """{action, ...}. Actions: ask_goal, run_stage, checkpoint, review, open_round, launch_experiments,
    await_experiments, advance_round, close_round, manual_iteration, done."""
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
        st = _round_state(spec, cur["id"]) or {}
        if st.get("decided"):
            return {"action": "close_round", "iteration": cur["id"], "phase": st.get("phase"),
                    "why": "The round has settled: " + (st.get("log") or ["."])[-1]}
        running = [x["id"] for x in cur["experiments"] if x.get("status") == "running"]
        planned = [x["id"] for x in cur["experiments"] if x.get("status") == "planned"]
        slots = int(ml_common.autopilot_settings(spec)["max_parallel"]) - len(running)
        if planned and slots > 0:
            return {"action": "launch_experiments", "iteration": cur["id"], "phase": st.get("phase") or "screen",
                    "experiments": planned[:slots],
                    "why": f"{len(planned)} experiment(s) of the round wait for a workspace and an agent."}
        if running:
            return {"action": "await_experiments", "iteration": cur["id"], "phase": st.get("phase") or "screen",
                    "experiments": running, "why": "Experiments of the round have not reported yet."}
        return {"action": "advance_round", "iteration": cur["id"], "phase": st.get("phase") or "screen",
                "why": "Every experiment of this phase reported. Judge it and plan the next phase."}
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
            lines += ["", "## The rounds of experiments", "",
                      "| Round | Experiment | Phase | Changes | Bar against the champion | Status |",
                      "|---|---|---|---|---|---|"]
            for e in rounds(spec):
                for x in e.get("experiments") or []:
                    ch = "; ".join(f"{c.get('stage')}: {c.get('change')}" for c in x.get("changes") or [])
                    lines.append(f"| {e['id']} | {x.get('experiment')} | {x.get('phase') or 'screen'} | "
                                 f"{_fmt(ch, 90).replace('|', '/')} | {(x.get('bar') or {}).get('status')} | "
                                 f"{x.get('status')} |")
            lines += ["", "## How each round settled", ""]
            for e in rounds(spec):
                lines += [f"* Round {e['id']}: {line}" for line in (_round_state(spec, e["id"]) or {}).get("log") or []]
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


def _cheapest_first(items):
    return sorted(items, key=lambda b: (b.get("cost", 2), int(str(b["id"]).lstrip("B") or 0)))


def _round_items(spec):
    """The items a round screens: the cheapest open item upstream of the split alone, when there is one, since it
    changes the folds; otherwise every open item."""
    items = _cheapest_first([b for b in _ap(spec)["backlog"] if b.get("status") == "open"])
    upstream = [b for b in items if b["stage"] in UPSTREAM]
    return upstream[:1] if upstream else items


# ---------------------------------------------------------------- rounds

def _round_state(spec, iteration):
    """The autopilot's record of a round: its phase, the items that passed, and how it settled."""
    return next((r for r in _ap(spec).get("rounds") or [] if r.get("iteration") == iteration), None)


def _screen_id(i):
    return "abcdefghijklmnopqrstuvwxyz"[i] if i < 26 else f"s{i + 1}"


def _backlog_ids(x):
    b = x.get("backlog")
    return [b] if isinstance(b, str) else list(b or [])


def _gain(x):
    return (x.get("bar") or {}).get("mean_diff") or 0


def _folds(x, which="candidate"):
    """Per fold CV scores an experiment reported: of the model it is judged by, or of the carried champion."""
    bar = x.get("bar") or {}
    per_fold = bar.get("per_fold") or {}
    if which == "carried":
        return per_fold.get(bar.get("carried"))
    return per_fold.get(x.get("candidate") or bar.get("new") or x.get("champion"))


def _judge(spec, x, ref):
    """The iteration bar for experiment x against experiment ref, on the same folds."""
    return ml_common.iteration_bar_evaluate(_folds(x), _folds(ref),
                                            ml_common.lower_is_better(spec.get("training") or {}),
                                            ml_common.iteration_bar(spec))


def _changes(xs):
    out = []
    for x in xs:
        out += [c for c in x.get("changes") or [] if c not in out]
    return out


def _plan_experiment(id_, phase, xs, hypothesis, compare_to=None, frozen_model=None):
    plan = {"id": id_, "phase": phase, "hypothesis": hypothesis, "changes": _changes(xs),
            "backlog": [b for x in xs for b in _backlog_ids(x)], "refs": [x["branch"] for x in xs if x.get("branch")]}
    if compare_to:
        plan["compare_to"] = compare_to
    if frozen_model:
        plan["frozen_model"] = frozen_model
    return plan


def _launch(root, spec_path):
    """Make workspaces for planned experiments of the open round, so at most max_parallel run at once."""
    spec = ml_common.load(spec_path)
    entry = workspace.open_round(spec)
    exps = (entry or {}).get("experiments") or []
    running = [x for x in exps if x.get("status") == "running"]
    slots = int(ml_common.autopilot_settings(spec)["max_parallel"]) - len(running)
    created = []
    for x in [x for x in exps if x.get("status") == "planned"][:max(slots, 0)]:
        w = workspace.create(root, spec_path, x["id"])
        created.append(w if w.get("ok") else {"id": x["id"], "ok": False, "problems": w.get("problems")})
    return created


def round_open(root, spec_path, spec, scratch):
    act = next_action(spec)
    if act["action"] != "open_round":
        return {"ok": False, "problems": [f"A round cannot open now: {act['why']}"]}
    picked = _round_items(spec)
    if not picked:
        return {"ok": False, "problems": ["The backlog has no open items. Diagnose and add some with "
                                          "`autopilot.py backlog add` first."]}
    plan = {"hypothesis": "Round of " + str(len(picked)) + " changes, each screened alone, then the ones that pass "
                          "combined: " + "; ".join(b["hypothesis"] for b in picked),
            "diagnosis": [f"{b['id']}: {b['change']}" for b in picked],
            "experiments": [{"id": _screen_id(i), "phase": "screen", "hypothesis": b["hypothesis"],
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
    _ap(spec).setdefault("rounds", []).append({"iteration": entry["id"], "version": entry.get("version"),
                                               "phase": "screen", "status": "open", "opened_at": _now(), "log": []})
    _save(spec_path, spec, f"opened round {entry['id']} with {', '.join(p['id'] for p in picked)}")
    created = _launch(root, spec_path)
    return {"ok": all(w.get("ok") for w in created), "round": entry["id"], "version": entry.get("version"),
            "plan": plan_path, "experiments": created,
            "waiting": len(picked) - len(created)}


def round_launch(root, spec_path, spec):
    if not workspace.open_round(spec):
        return {"ok": False, "problems": ["No round is open."]}
    created = _launch(root, spec_path)
    return {"ok": all(w.get("ok") for w in created), "experiments": created}


def _settle(st, final, why):
    st.update({"decided": True, "final": final, "settled_at": _now()})
    st["log"].append(why)


def _advance(spec, entry, st):
    """Judge the phase that just finished and return (experiments to add, notes). Updates the round state st."""
    exps = entry["experiments"]
    by_id = {x["id"]: x for x in exps}
    phase = st.get("phase") or "screen"
    name = lambda i: by_id[i].get("experiment") or i  # noqa: E731
    if phase == "screen":
        passers = sorted([x for x in exps if (x.get("phase") or "screen") == "screen" and x.get("status") == "passed"],
                         key=lambda x: -_gain(x))
        st["passers"] = [x["id"] for x in passers]
        if not passers:
            _settle(st, None, "No change passed the iteration bar on its own.")
            return [], []
        if len(passers) == 1:
            _settle(st, passers[0]["id"], f"Only {name(passers[0]['id'])} passed on its own, so it is adopted alone.")
            return [], []
        st["phase"], st["combo"] = "combine", "comb"
        st["log"].append(f"{len(passers)} changes passed on their own: "
                         + ", ".join(name(x["id"]) for x in passers) + ". They are combined.")
        return [_plan_experiment("comb", "combine", passers,
                                 "Every change that passed on its own, together: "
                                 + "; ".join(x["hypothesis"] for x in passers))], []

    passers = [by_id[i] for i in st.get("passers") or []]
    if phase == "combine":
        combo = by_id[st["combo"]]
        if combo.get("status") == "passed":
            st["phase"], st["ablations"] = "ablate", {}
            st["log"].append(f"The combination {name(combo['id'])} passed the bar against the champion. Each change "
                             "is taken out in turn.")
            plans = []
            for n, p in enumerate(passers, 1):
                rest = [x for x in passers if x["id"] != p["id"]]
                if len(rest) == 1:   # the combination without p is the other change alone, screened already
                    st["ablations"][p["id"]] = rest[0]["id"]
                    continue
                frozen = None
                if all(c.get("stage") in BEFORE_TRAINING for c in p.get("changes") or []):
                    frozen = (combo.get("bar") or {}).get("champion_model")
                aid = f"abl{n}"
                st["ablations"][p["id"]] = aid
                plans.append(_plan_experiment(aid, "ablate", rest,
                                              f"The combination without {name(p['id'])} ({p['hypothesis']}) is "
                                              "better than with it.", compare_to=combo["id"], frozen_model=frozen))
            if not plans:
                return [], []
            return plans, []
        return _start_greedy(st, passers, name,
                             f"The combination {name(combo['id'])} did not pass the bar against the champion "
                             f"({(combo.get('bar') or {}).get('detail') or combo.get('status')}).")

    if phase == "ablate":
        combo = by_id[st["combo"]]
        dropped, verdicts = [], {}
        for p in passers:
            a = by_id.get(st["ablations"].get(p["id"]))
            if a is None or a.get("status") == "error" or not _folds(a):
                verdicts[p["id"]] = "no result, so the change stays"
                continue
            v = _judge(spec, a, combo)
            verdicts[p["id"]] = f"{v['status']}: {v['detail']}"
            if v["status"] == "pass":
                dropped.append(p)
        st["ablation_verdicts"] = verdicts
        st["dropped"] = [p["id"] for p in dropped]
        kept = [p for p in passers if p not in dropped]
        if not dropped:
            _settle(st, combo["id"], f"Taking any change out of {name(combo['id'])} did not help, so the whole "
                                     "combination is adopted.")
            return [], []
        st["log"].append("Taking out " + ", ".join(name(p["id"]) for p in dropped) + " made the combination better, "
                         "so they go.")
        if not kept:
            return _start_greedy(st, passers, name, "Taking out every change helped, so the changes conflict.")
        if len(kept) == 1:
            _settle(st, kept[0]["id"], f"Only {name(kept[0]['id'])} is left, and it passed on its own.")
            return [], []
        st["phase"], st["verify"] = "verify", "ver"
        return [_plan_experiment("ver", "verify", kept, "The changes left after the ablation, together: "
                                 + "; ".join(x["hypothesis"] for x in kept))], []

    if phase == "verify":
        v = by_id[st["verify"]]
        if v.get("status") == "passed":
            _settle(st, v["id"], f"The changes left after the ablation ({name(v['id'])}) passed the bar against the "
                                 "champion.")
        else:
            _settle(st, st["combo"], f"The changes left after the ablation ({name(v['id'])}) did not pass the bar "
                                     f"against the champion, so the full combination {name(st['combo'])} is adopted.")
        return [], []

    if phase == "greedy":
        g = st["greedy"]
        x, cur = by_id[g["trying"]], by_id[g["current"]]
        v = _judge(spec, x, cur) if x.get("status") == "passed" else None
        added = by_id[g["order"][g["next"]]]
        if v and v["status"] == "pass":
            g["current"] = x["id"]
            g["set"].append(added["id"])
            st["log"].append(f"Adding {name(added['id'])} passed the bar against {name(cur['id'])}.")
        else:
            st["log"].append(f"Adding {name(added['id'])} did not help: "
                             + (v["detail"] if v else f"{name(x['id'])} {x.get('status')} against the champion."))
        g["next"] += 1
        if g["next"] < len(g["order"]):
            return _greedy_step(st, by_id, name), []
        _settle(st, g["current"], f"Greedy selection settled on {name(g['current'])}.")
        return [], []
    return [], [f"Unknown phase {phase!r}."]


def _start_greedy(st, passers, name, why):
    st["phase"] = "greedy"
    st["log"].append(why + " The changes are added one at a time instead, best first.")
    order = [p["id"] for p in passers]
    st["greedy"] = {"order": order, "set": [order[0]], "current": order[0], "next": 1, "steps": 0}
    return _greedy_step(st, {p["id"]: p for p in passers}, name), []


def _greedy_step(st, by_id, name):
    g = st["greedy"]
    g["steps"] += 1
    gid = f"g{g['steps']}"
    g["trying"] = gid
    xs = [by_id[i] for i in g["set"] + [g["order"][g["next"]]]]
    nxt = by_id[g["order"][g["next"]]]
    return [_plan_experiment(gid, "greedy", xs, f"Adding {name(nxt['id'])} ({nxt['hypothesis']}) to "
                             f"{', '.join(name(i) for i in g['set'])} helps.", compare_to=g["current"])]


def round_advance(root, spec_path, spec, scratch):
    entry = workspace.open_round(spec)
    if not entry or not entry.get("experiments"):
        return {"ok": False, "problems": ["No round is open."]}
    waiting = [x["id"] for x in entry["experiments"] if x.get("status") in ("planned", "running")]
    if waiting:
        return {"ok": False, "problems": [f"Experiments {', '.join(waiting)} have not reported yet. Record them with "
                                          "`iterate.py record-experiment` (status error when an agent failed)."]}
    st = _round_state(spec, entry["id"])
    if st is None:   # a round opened before rounds had phases
        st = {"iteration": entry["id"], "version": entry.get("version"), "phase": "screen", "status": "open",
              "log": []}
        _ap(spec).setdefault("rounds", []).append(st)
    if st.get("decided"):
        return {"ok": False, "problems": ["The round has settled. Close it with `autopilot.py round close`."]}
    phase = st.get("phase") or "screen"
    plans, problems = _advance(spec, entry, st)
    while not plans and not problems and not st.get("decided"):   # a phase whose results are all reused
        plans, problems = _advance(spec, entry, st)
    if problems:
        return {"ok": False, "problems": problems}
    notes = []
    # Experiments that failed the screen can never be adopted: free their worktrees now
    if phase == "screen":
        for x in entry["experiments"]:
            if (x.get("phase") or "screen") == "screen" and x.get("status") in ("failed", "error"):
                d = workspace.discard(root, spec_path, x["id"], f"{x.get('status')} on its own: "
                                      f"{(x.get('bar') or {}).get('detail') or ''}".strip())
                notes += d.get("notes") or d.get("problems") or []
                st.setdefault("discarded", []).append(x["id"])
    _save(spec_path, spec, f"round {entry['id']}: {phase} judged" + (", settled" if st.get("decided") else
                                                                       f", next {st.get('phase')}"))
    created = []
    if plans:
        os.makedirs(scratch, exist_ok=True)
        plan_path = os.path.join(scratch, f"round_{entry['id']}_{st['phase']}_"
                                          f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(ml_common.dump_text({"experiments": plans}))
        res = workspace.run_iterate(root, "add-experiments", plan_path)
        if res.get("ok") is False:
            return {"ok": False, "problems": res.get("problems"), "plan": plan_path}
        created = _launch(root, spec_path)
    return {"ok": all(w.get("ok") for w in created), "round": entry["id"], "phase": st.get("phase"),
            "decided": bool(st.get("decided")), "final": st.get("final"), "log": st["log"],
            "added": [p["id"] for p in plans], "experiments": created, "notes": notes}


def round_close(root, spec_path, spec):
    entry = workspace.open_round(spec)
    if not entry or not entry.get("experiments"):
        return {"ok": False, "problems": ["No round is open."]}
    st = _round_state(spec, entry["id"]) or {}
    if not st.get("decided"):
        return {"ok": False, "problems": ["The round has not settled yet. Run `autopilot.py round advance` until "
                                          "it has."]}
    by_id = {x["id"]: x for x in entry["experiments"]}
    final = by_id.get(st.get("final")) if st.get("final") else None
    notes = []
    if final is not None:   # adopt first: when the merge fails, nothing else of the round has changed yet
        a = workspace.adopt(root, spec_path, final["id"])
        if not a.get("ok"):
            return {"ok": False, "problems": a.get("problems"), "notes": notes}
        notes += a.get("notes") or []
    for x in entry["experiments"]:
        if (final is not None and x["id"] == final["id"]) or x["id"] in (st.get("discarded") or []):
            continue
        why = (f"the round settled on {final.get('experiment')}" if final is not None and x.get("status") == "passed"
               else f"{x.get('status')}: {(x.get('bar') or {}).get('detail') or ''}".strip())
        d = workspace.discard(root, spec_path, x["id"], why)
        notes += d.get("notes") or d.get("problems") or []
    reason = st["log"][-1]
    if final is not None:
        res = workspace.run_iterate(root, "close", "--outcome", "kept", "--adopted", str(final["id"]),
                                    "--reason", reason)
    else:
        res = workspace.run_iterate(root, "close", "--outcome", "rolled_back", "--none-passed", "--reason", reason)
    if res.get("ok") is False:
        return {"ok": False, "problems": res.get("problems"), "notes": notes}
    notes += res.get("notes") or []
    spec = ml_common.load(spec_path)
    st = _round_state(spec, entry["id"])
    adopted = set(_backlog_ids(final)) if final is not None else set()
    screens = {b: x for x in entry["experiments"] if (x.get("phase") or "screen") == "screen"
               for b in _backlog_ids(x)}
    dropped = {b for i in st.get("dropped") or [] for b in _backlog_ids(by_id[i])}
    for b in _ap(spec)["backlog"]:
        if b.get("status") != "in_round" or b.get("round") != entry["id"]:
            continue
        x = screens.get(b["id"]) or {}
        b["status"] = "adopted" if b["id"] in adopted else "tried"
        b["result"] = {"experiment": x.get("experiment"), "bar": (x.get("bar") or {}).get("status"),
                       "note": ("adopted" if b["id"] in adopted else
                                "passed on its own, but taking it out made the combination better" if b["id"] in dropped
                                else "passed on its own, but did not add to the others" if x.get("status") == "passed"
                                else "did not pass on its own")}
    st.update({"status": "closed", "closed_at": _now()})
    _save(spec_path, spec, f"closed round {entry['id']}: " + (f"adopted {final['experiment']}" if final is not None
                                                               else "nothing passed"))
    return {"ok": True, "round": entry["id"], "adopted": final and final.get("experiment"),
            "changes": final and final.get("changes"), "log": st["log"], "notes": notes, "limits": limits(spec)}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["start", "next", "checkpoint", "backlog", "round", "stop", "limits"])
    ap.add_argument("sub", nargs="?", help="checkpoint: write or clear; backlog: add or list; "
                                           "round: open, launch, advance, or close")
    ap.add_argument("arg", nargs="?", help="checkpoint: the kind; backlog add: the items file")
    ap.add_argument("--max-parallel", type=int, help=f"start: experiments run at once (1 to {ml_common.MAX_PARALLEL})")
    ap.add_argument("--attended", action="store_true", help="start: ask before each unattended stage")
    ap.add_argument("--reason", help="stop: why; checkpoint clear: a note")
    ap.add_argument("--scratch", help="round open, advance: where to write the plans (default: .worktrees/plans)")
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
        if a.max_parallel:
            settings["max_parallel"] = max(1, min(a.max_parallel, ml_common.MAX_PARALLEL))
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
        scratch = a.scratch or os.path.join(root, workspace.WORKTREES, "plans")
        if a.sub == "open":
            res = round_open(root, spec_path, spec, scratch)
        elif a.sub == "launch":
            res = round_launch(root, spec_path, spec)
        elif a.sub == "advance":
            res = round_advance(root, spec_path, spec, scratch)
        elif a.sub == "close":
            res = round_close(root, spec_path, spec)
        else:
            ap.error("round open|launch|advance|close")
    print(json.dumps(res, indent=1, default=str, ensure_ascii=False))
    sys.exit(0 if res.get("ok", True) else 1)


if __name__ == "__main__":
    main()
