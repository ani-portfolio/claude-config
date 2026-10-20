"""Design the end to end ML system: read what the earlier stages decided, check the answers against it, and write
the system section of ml_spec.yaml and docs/system_design.md.

Commands, run at the project root with the project's Python:

  design.py --context [--spec PATH]                    the facts the interview drafts from, and the entry rule (JSON)
  design.py ANSWERS.yaml --check [--spec PATH]         blocks and warnings for these answers; writes nothing
  design.py ANSWERS.yaml [--overwrite] [--spec PATH]   write the system section and docs/system_design.md

--json (from nb_common.cli) prints one JSON object with the result. Exit codes: 0 done, 1 blocked or failed,
2 docs/system_design.md was edited by hand since the skill wrote it (rerun with --overwrite to replace it).

The answers file has the keys in templates/system_answers.example.yaml: status, platform, entry, components (the
twelve in COMPONENTS, each with needed, tool, settings, and reason), reasoning, warnings_accepted, risks,
assumptions, open_questions, and decisions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402
import system_doc  # noqa: E402

SKILL = "ml-system-design"
STAGE = "system"
DEFAULTS_PATH = os.path.join(HERE, "..", "reference", "defaults.yaml")
COMPONENTS = ["ingestion", "data_validation", "features", "training", "registry", "inference", "serving",
              "label_feedback", "monitoring", "failure_handling", "orchestration", "testing"]
# The parts drafted from the champion. They go out of date when the champion changes; the rest is infrastructure.
CHAMPION_DEPENDENT = ["features", "registry", "monitoring", "failure_handling"]
FREQUENCY_RANK = {"hourly": 1, "daily": 2, "weekly": 3, "monthly": 4}
FREQUENCIES = tuple(FREQUENCY_RANK) + ("on_request",)
PLATFORM_KINDS = ("default", "platform", "neutral")
ENTRY_OUTCOMES = ("go", "no_go_build", "override")
SUMMARY_JSON = os.path.join("docs", "ml_system_summary", "system_summary.json")


def load_defaults():
    return ml_common.load(DEFAULTS_PATH)


def stack_for(platform, defaults=None):
    """The defaults stack for a platform choice: default, neutral, or the named platform (None when not listed)."""
    defaults = defaults or load_defaults()
    platform = platform or {}
    key = platform.get("name") if platform.get("kind") == "platform" else platform.get("kind")
    return (defaults.get("stacks") or {}).get(str(key or "").lower())


def doc_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def doc_path(spec, root):
    return os.path.join(ml_common.project_paths(spec, root)["docs"], "system_design.md")


def _ts(x):
    try:
        return _dt.datetime.fromisoformat(str(x)) if x else None
    except ValueError:
        return None


# ---------------------------------------------------------------- the entry rule

def entry_state(spec):
    """Whether the design may start: {"rule", "decision", "mode", "answer_actions", "detail"}. rule is go (proceed),
    exploratory_no_go (proceed only if the matching answer still builds the system), refuse, or unclear (the
    decision text could not be read; ask the person what was decided)."""
    dec = ml_common.evaluation_decision(spec)
    mode = ml_common.business_mode(spec)
    actions = (spec.get("business") or {}).get("answer_actions") or []
    out = {"decision": dec, "mode": mode, "answer_actions": actions if mode == "exploratory" else []}
    if not (spec.get("evaluation") or {}):
        return {**out, "rule": "refuse", "detail": "The offline evaluation has not run. Run ml-offline-evaluation first."}
    if dec is None:
        return {**out, "rule": "refuse", "detail": "No go, iterate, or no go decision is recorded on the evaluation. "
                "Record it in the evaluation notebook's decisions cell and run it, then design the system."}
    if dec["outcome"] == "go":
        return {**out, "rule": "go", "detail": f"The evaluation decision is go: {dec['decision']}"}
    if dec["outcome"] is None:
        return {**out, "rule": "unclear", "detail": f"The evaluation decision could not be read as go, iterate, or no "
                f"go: {dec['decision']!r}. Ask the person what was decided."}
    if mode == "exploratory":
        return {**out, "rule": "exploratory_no_go", "detail": f"The evaluation decision is {dec['outcome'].replace('_', ' ')}. "
                "In an exploratory project, proceed only if the answer that matches the result still says to build "
                "the system."}
    return {**out, "rule": "refuse", "detail": f"The evaluation decision is {dec['outcome'].replace('_', ' ')}, so there "
            "is no model to design a system around. Iterate with ml-iterate, or override with a written reason."}


# ---------------------------------------------------------------- context for the interview

def stale_parts(spec):
    """Champion dependent components of an existing design that no longer match the current champion."""
    sysd = spec.get("system") or {}
    if not sysd:
        return []
    champ = ((spec.get("training") or {}).get("champion") or {}).get("name")
    eval_t = _ts((spec.get("evaluation") or {}).get("run_at"))
    sys_t = _ts(sysd.get("run_at"))
    changed = (sysd.get("champion") and champ and sysd["champion"] != champ) or (eval_t and sys_t and eval_t > sys_t)
    return list(CHAMPION_DEPENDENT) if changed else []


def context(spec, root):
    fr = spec.get("framing") or {}
    tr = spec.get("training") or {}
    ev = spec.get("evaluation") or {}
    champ = tr.get("champion") or {}
    pt = fr.get("prediction_time") or {}
    summary = os.path.join(root, SUMMARY_JSON)
    return {
        "entry": entry_state(spec),
        "business": {k: (spec.get("business") or {}).get(k) for k in
                     ("mode", "learning_question", "business_goal", "decision_supported", "users", "output_use",
                      "constraints")},
        "framing": {"task_type": fr.get("task_type"), "target": (fr.get("target") or {}).get("name"),
                    "unit_of_prediction": fr.get("unit_of_prediction"), "entity_id_cols": fr.get("entity_id_cols"),
                    "prediction_time": pt, "label_delay": (fr.get("target") or {}).get("label_delay"),
                    "primary_metric": ml_common.primary_metric(spec),
                    "v1_bar": (fr.get("success") or {}).get("v1_bar")},
        "data_sources": [s.get("path") for s in (spec.get("data") or {}).get("sources") or [] if isinstance(s, dict)],
        "features": {"inputs": [i.get("name") for i in (spec.get("features") or {}).get("inputs") or []
                                if isinstance(i, dict)],
                     "code": (spec.get("features") or {}).get("code")},
        "champion": {k: champ.get(k) for k in ("name", "type", "std", "path", "run_id", "vs_reference_pct")},
        "primary_label": tr.get("primary_label") or ev.get("primary_label"),
        "top_features": tr.get("top_features"),
        "evaluation": {"metrics": ev.get("metrics"), "comparisons": ev.get("comparisons"),
                       "score_psi": ev.get("score_psi"), "target_psi": ev.get("target_psi"), "v1": ev.get("v1"),
                       "best_non_ml": (spec.get("baseline") or {}).get("best_non_ml")},
        "tracking": {k: (spec.get("tracking") or {}).get(k) for k in ("uri", "experiment")},
        "existing_system_summary": SUMMARY_JSON if os.path.exists(summary) else None,
        "previous_design": {k: (spec.get("system") or {}).get(k) for k in ("status", "platform", "champion", "run_at")}
        if spec.get("system") else None,
        "out_of_date": stale_parts(spec),
        "champion_dependent": CHAMPION_DEPENDENT,
        "defaults": load_defaults(),
    }


# ---------------------------------------------------------------- checks

def _settings(answers, comp):
    return ((answers.get("components") or {}).get(comp) or {}).get("settings") or {}


def _needed(answers, comp):
    return ((answers.get("components") or {}).get(comp) or {}).get("needed") is not False


def check(spec, answers, defaults=None):
    """Returns (blocks, warnings), each a list of {"id", "detail"}. Blocks are contradictions with the spec or
    broken answers, and stop the build. Warnings are judgement calls; each needs an entry in warnings_accepted."""
    blocks, warns = [], []
    B = lambda i, d: blocks.append({"id": i, "detail": d})  # noqa: E731
    W = lambda i, d: warns.append({"id": i, "detail": d})  # noqa: E731
    status = answers.get("status") or "draft"
    comps = answers.get("components") or {}
    fr = spec.get("framing") or {}
    platform = answers.get("platform") or {}

    # Shape of the answers
    if status not in ("draft", "confirmed"):
        B("status", f"status is {status!r}; use draft or confirmed.")
    if platform.get("kind") not in PLATFORM_KINDS:
        B("platform", f"platform.kind is {platform.get('kind')!r}; use one of {', '.join(PLATFORM_KINDS)}.")
    elif platform.get("kind") == "platform" and not platform.get("name"):
        B("platform", "platform.kind is platform, but platform.name is empty.")
    unknown = sorted(set(comps) - set(COMPONENTS))
    if unknown:
        B("components", f"Unknown components: {', '.join(unknown)}. Use: {', '.join(COMPONENTS)}.")
    missing = [c for c in COMPONENTS if c not in comps]
    if missing and status == "confirmed":
        B("components", f"A confirmed design needs every component; missing: {', '.join(missing)}.")
    for c in COMPONENTS:
        e = comps.get(c)
        if e is None:
            continue
        if not isinstance(e, dict):
            B("components", f"components.{c} must be a mapping with needed, tool, settings, and reason.")
        elif e.get("needed") is False and not str(e.get("reason") or "").strip():
            B("not needed", f"{c} is marked not needed without a reason.")
        elif e.get("needed") is not False and not e.get("tool") and status == "confirmed":
            B("tool", f"{c} is needed but has no tool.")

    # The entry rule
    entry = answers.get("entry") or {}
    state = entry_state(spec)
    outcome = entry.get("outcome")
    if outcome not in ENTRY_OUTCOMES:
        B("entry", f"entry.outcome is {outcome!r}; use one of {', '.join(ENTRY_OUTCOMES)}.")
    elif outcome == "override":
        if not str(entry.get("reason") or "").strip():
            B("entry", "An override of the entry rule needs a written reason in entry.reason.")
    elif state["rule"] in ("refuse", "unclear"):
        B("entry", state["detail"] + " To go ahead anyway, set entry.outcome to override with a reason.")
    elif outcome == "go" and state["rule"] != "go":
        B("entry", f"entry.outcome is go, but {state['detail'][0].lower() + state['detail'][1:]}")
    elif outcome == "no_go_build":
        texts = [str(a.get("answer") or "") for a in state["answer_actions"] if isinstance(a, dict)]
        if state["rule"] != "exploratory_no_go":
            B("entry", "entry.outcome no_go_build is only for an exploratory project whose evaluation was not a go.")
        elif str(entry.get("answer") or "") not in texts:
            B("entry", "entry.answer must be the business answer_actions entry that matches the result, word for "
              "word, and its action must say to build the system.")

    # The gate against the framing and the evaluation
    reg = _settings(answers, "registry")
    gate = reg.get("gate") or {}
    if _needed(answers, "registry") and comps.get("registry"):
        primary = ml_common.primary_metric(spec)
        if gate.get("metric") and primary:
            gk = ml_common.parse_metric(gate["metric"])[0].get("key")
            fk = ml_common.parse_metric(primary)[0].get("key")
            if gk and fk and gk != fk:
                B("gate metric", f"The gate compares {gate['metric']}, but the primary metric is {primary}. "
                  "The gate must judge models on the metric V1 was judged on.")
        elif status == "confirmed":
            B("gate metric", "registry.settings.gate.metric is empty.")
        std = ((spec.get("training") or {}).get("champion") or {}).get("std")
        margin = gate.get("margin")
        if margin is not None:
            if not isinstance(margin, (int, float)):
                B("gate margin", f"registry.settings.gate.margin must be a number in the metric's units, not {margin!r}.")
            elif isinstance(std, (int, float)) and abs(margin) < std:
                B("gate margin", f"The gate margin ({margin:g}) is smaller than the champion's CV spread (std {std:g}), "
                  "so the gate would swap champions on noise. Use a margin of at least the CV std.")
        elif status == "confirmed":
            B("gate margin", "registry.settings.gate.margin is empty.")
        if gate.get("promotion") not in (None, "auto", "manual"):
            B("gate promotion", f"registry.settings.gate.promotion is {gate.get('promotion')!r}; use auto or manual.")
        if gate.get("promotion") == "auto":
            W("auto promotion", "Challengers are promoted without a person looking. Safe only when the gate and the "
              "monitoring are trusted.")
        stack = stack_for(platform, defaults)
        if stack and stack.get("has_registry") and not reg.get("decides_champion"):
            B("registry", f"{platform.get('name') or platform.get('kind')} has its own model registry, so "
              "registry.settings.decides_champion must say which registry decides the champion (ask every time).")

    # Features served at prediction time
    inputs = [i.get("name") for i in (spec.get("features") or {}).get("inputs") or [] if isinstance(i, dict)]
    if _needed(answers, "features") and comps.get("features") and inputs:
        served = _settings(answers, "features").get("served_inputs")
        if served is None:
            if status == "confirmed":
                B("served inputs", "features.settings.served_inputs is empty: list how every model input is "
                  "computed at serving time.")
        else:
            gone = sorted(set(inputs) - set(served or []))
            if gone:
                B("served inputs", "These champion inputs are not computed at serving time, so the served model "
                  "would see different data than it was trained on: " + ", ".join(gone[:15])
                  + ("." if len(gone) <= 15 else f", and {len(gone) - 15} more."))

    # Schedules
    ing = _settings(answers, "ingestion")
    inf = _settings(answers, "inference")
    for comp, s, key in (("ingestion", ing, "frequency"), ("inference", inf, "schedule")):
        if s.get(key) is not None and s.get(key) not in FREQUENCIES:
            B("schedule", f"{comp}.settings.{key} is {s.get(key)!r}; use one of {', '.join(FREQUENCIES)}.")
    if inf.get("mode") not in (None, "batch", "online"):
        B("inference mode", f"inference.settings.mode is {inf.get('mode')!r}; use batch or online.")
    if inf.get("mode") == "batch" and inf.get("schedule") in FREQUENCY_RANK and ing.get("frequency") in FREQUENCY_RANK \
            and FREQUENCY_RANK[inf["schedule"]] < FREQUENCY_RANK[ing["frequency"]]:
        B("schedule", f"Batch inference runs {inf['schedule']}, but new data only arrives {ing['frequency']}, so most "
          "runs would score unchanged data.")
    cadence = str(((fr.get("prediction_time") or {}).get("cadence")) or "").lower()
    if inf.get("mode") == "online" and re.search(r"daily|weekly|monthly|batch", cadence):
        W("online for a batch cadence", f"Inference is online, but the framing's cadence is {cadence!r}. A batch job "
          "that precomputes the predictions would be simpler and cheaper.")

    # Judgement calls
    fh = _settings(answers, "failure_handling")
    if not _needed(answers, "failure_handling") or (comps.get("failure_handling") and not fh.get("fallback")):
        W("no fallback", "Nothing is served when the model or its features fail. Serving the best simple baseline is "
          "the usual fallback.")
    if _needed(answers, "monitoring") and comps.get("monitoring") and not _needed(answers, "label_feedback"):
        W("monitoring without labels", "Monitoring cannot measure live performance without label feedback, and the "
          "gate has no live scores to compare.")
    if comps.get("testing") and not _needed(answers, "testing"):
        W("no tests", "No tests guard the system, so a change to the feature code can break serving silently.")

    accepted = {w.get("check") for w in answers.get("warnings_accepted") or []
                if isinstance(w, dict) and str(w.get("reason") or "").strip()}
    for w in warns:
        w["accepted"] = w["id"] in accepted
    return blocks, warns


# ---------------------------------------------------------------- build

def build(spec_path, answers, overwrite=False):
    spec = ml_common.load(spec_path)
    root = os.path.dirname(os.path.abspath(spec_path))
    blocks, warns = check(spec, answers)
    open_warns = [w for w in warns if not w["accepted"]]
    if blocks or open_warns:
        return {"ok": False, "blocks": blocks, "warnings": warns}
    out_path = doc_path(spec, root)
    old = spec.get("system") or {}
    if os.path.exists(out_path) and old.get("doc_hash") and not overwrite:
        with open(out_path, encoding="utf-8") as f:
            if doc_hash(f.read()) != old["doc_hash"]:
                return {"ok": False, "edited": ml_common.relpath(out_path, root)}

    champ = ((spec.get("training") or {}).get("champion") or {}).get("name")
    entry = answers.get("entry") or {}
    values = {
        "status": answers.get("status") or "draft",
        "platform": answers.get("platform"),
        "entry": {**entry, "evaluation_decision": (ml_common.evaluation_decision(spec) or {}).get("decision")},
        "champion": champ,
        "components": {c: (answers.get("components") or {}).get(c) for c in COMPONENTS
                       if (answers.get("components") or {}).get(c) is not None},
        "champion_dependent": CHAMPION_DEPENDENT,
        "warnings_accepted": answers.get("warnings_accepted") or [],
        "risks": answers.get("risks") or [],
        "doc": ml_common.relpath(out_path, root),
        "run_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
    }
    decisions = list(answers.get("decisions") or [])
    if entry.get("outcome") == "override":
        decisions.append({"decision": "Designed the system although the evaluation decision did not allow it.",
                          "reason": entry.get("reason")})
    for w in answers.get("warnings_accepted") or []:
        if isinstance(w, dict) and w.get("check"):
            decisions.append({"decision": f"Accepted the system design warning: {w['check']}.", "reason": w.get("reason")})
    nb_common.write_stage(spec_path, STAGE, values, SKILL, answers.get("assumptions") or [],
                          answers.get("open_questions") or [], decisions, replace=True)

    text = system_doc.render(ml_common.load(spec_path), answers.get("reasoning") or {})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    spec = ml_common.load(spec_path)
    spec[STAGE]["doc_hash"] = doc_hash(text)
    ml_common.log(spec, SKILL, f"wrote {values['doc']}")
    ml_common.save(spec_path, spec)
    return {"ok": True, "blocks": [], "warnings": warns, "doc": values["doc"], "status": values["status"]}


def _print_findings(blocks, warns):
    for b in blocks:
        print(f"BLOCK ({b['id']}): {b['detail']}")
    for w in warns:
        print(f"WARNING ({w['id']}, {'accepted' if w['accepted'] else 'needs a reason'}): {w['detail']}")
    if not blocks and not warns:
        print("No blocks or warnings.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("answers", nargs="?", help="answers YAML file")
    ap.add_argument("--context", action="store_true", help="print the facts the interview drafts from")
    ap.add_argument("--check", action="store_true", help="check the answers and write nothing")
    ap.add_argument("--overwrite", action="store_true", help="replace a hand edited docs/system_design.md")
    ap.add_argument("--spec", help="path to ml_spec.yaml")
    a = ap.parse_args()
    spec_path = ml_common.spec_path(a.spec)
    if not os.path.exists(spec_path):
        sys.exit(f"No {ml_common.SPEC_NAME} found. Run ml-project-setup first.")
    spec = ml_common.migrate(ml_common.load(spec_path))
    root = os.path.dirname(os.path.abspath(spec_path))
    if a.context:
        print(json.dumps(context(spec, root), indent=2, default=str, ensure_ascii=False))
        return
    if not a.answers:
        sys.exit("Give an answers file, or --context.")
    answers = ml_common.load(a.answers) or {}
    if a.check:
        blocks, warns = check(spec, answers)
        _print_findings(blocks, warns)
        sys.exit(1 if blocks else 0)
    result = build(spec_path, answers, overwrite=a.overwrite)
    if result.get("edited"):
        print(f"{result['edited']} was edited by hand after the skill wrote it. Move the edits into the answers "
              "(reasoning), or rerun with --overwrite to replace it.")
        sys.exit(2)
    if not result["ok"]:
        print("Nothing was written.")
        _print_findings(result["blocks"], result["warnings"])
        sys.exit(1)
    _print_findings([], result["warnings"])
    print(f"Wrote the system section ({result['status']}) and {result['doc']}.")


if __name__ == "__main__":
    nb_common.cli(main, STAGE, SKILL)
