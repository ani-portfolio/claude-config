#!/usr/bin/env python3
"""Shared code for the ML project skills: project root, folder layout, kernel naming, and the
shared spec file (ml_spec.yaml).

The canonical copy lives in claude-skills/_ml_shared/ml_common.py. Every ML skill keeps an identical copy
in its own scripts folder so it works on its own. Edit the canonical copy, then run
`python3 tools/sync_ml_shared.py` from the claude-skills folder to update the copies.

Uses only the standard library. PyYAML is used when installed (the project .venv has it);
without it the spec is written as JSON, which is also valid YAML.

Commands (run with the project's Python from the project root, for example .venv/bin/python):
    ml_common.py root                          # the project root and how it was found
    ml_common.py path                          # the spec path
    ml_common.py paths                         # the project folders from the spec, and whether they exist
    ml_common.py init --name churn_v1 [--description "..."]
    ml_common.py show [--section framing]
    ml_common.py status                        # step progress, open questions, and the next skill to run
    ml_common.py next                          # just the next step and skill
    ml_common.py gate                          # exit 1 with reasons when modelling should not start yet
    ml_common.py validate                      # structure errors and required fields still empty
    ml_common.py merge business answers.yaml --skill ml-business-framing   # overwrite keys in a section
    ml_common.py fill project values.json --skill ml-project-setup       # set only keys that are empty
    ml_common.py columns data/raw/train.csv    # column names and types only, never values
    ml_common.py notebook-name eda --suffix train   # 04_eda_train.ipynb
    ml_common.py need scikit-learn optuna      # exit 1 with the uv add command when a package is not installed

Pass --spec PATH to any command to use a spec somewhere else.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import platform
import re
import sys

SPEC_VERSION = 1
SPEC_NAME = "ml_spec.yaml"

# The workflow, in order. Each stage owns one spec section and has one skill. Stage numbers prefix the stage's
# notebooks.
WORKFLOW = [
    # (stage, skill, notebook number, short description)
    ("setup", "ml-project-setup", 0, "project folder, .venv, kernel, and spec"),
    ("business", "ml-business-framing", 1, "business goal, decision, current approach, success, constraints"),
    ("discovery", "ml-data-discovery", 2, "sources, grain, time coverage, joins, data dictionary, framing hints"),
    ("framing", "ml-problem-framing", 3, "target, unit of prediction, prediction time, window, metric"),
    ("eda", "ml-eda-report", 4, "target analysis, leakage, and checks against the framing"),
    ("split", "ml-train-test-split", 5, "train, validation, and test sets"),
    ("features", "ml-feature-engineering", 6, "feature pipeline fitted on training data only"),
    ("baseline", "ml-baseline-model", 7, "dummy, heuristic, and simple model baselines"),
    ("training", "ml-model-training", 8, "main model with cross validation"),
    ("evaluation", "ml-offline-evaluation", 9, "test results, segments, calibration, model card"),
]
# The base name of each stage's notebook, after its number: 01_business_framing.ipynb, 04_eda_<dataset>.ipynb.
NOTEBOOK_BASE = {
    "setup": "scratch",
    "business": "business_framing",
    "discovery": "data_discovery",
    "framing": "problem_framing",
    "eda": "eda",
    "split": "split",
    "features": "features",
    "baseline": "baseline",
    "training": "training",
    "evaluation": "evaluation",
}
STAGES = [w[0] for w in WORKFLOW]
STAGE_NUMBER = {w[0]: w[2] for w in WORKFLOW if w[2] is not None}
STAGE_SKILL = {w[0]: w[1] for w in WORKFLOW}
# Helper skills that keep a notebook of their own but are not workflow steps.
STAGE_NUMBER["review"] = 10
NOTEBOOK_BASE["review"] = "review"
STAGE_SKILL["review"] = "ml-review"

# The steps a project goes through, in order: one step per stage. Kept separate from WORKFLOW so a stage can
# later run in more than one step.
STEPS = [(stage, skill, "", what) for stage, skill, _, what in WORKFLOW]
EDA_MODES = ("discovery", "targeted")

# Folders relative to the project root. Stored in the spec under paths, so a project can override them.
DEFAULT_PATHS = {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "notebooks": "notebooks",
    "src": "src",
    "models": "models",
    "docs": "docs",
}

BUSINESS_TODO = "The business problem has not been framed yet. Run the ml-business-framing skill."
FRAMING_TODO = "The ML problem has not been framed yet. Run the ml-problem-framing skill before building features."
PREV_FRAMING_TODO = "The ML problem has not been framed yet. Run the ml-framing skill before building features."
OLD_FRAMING_TODO = "The problem has not been framed yet. Run the problem-framing skill before building features."
OLD_BUSINESS_TODO = "The business problem has not been framed yet. Run the business-framing skill."
BUSINESS_STATUS = {"not_started", "draft", "confirmed"}
ML_NEEDED = {"yes", "no", "unsure"}
TASK_TYPES = {"binary_classification", "multiclass_classification", "regression", "ranking", "forecasting", "other"}
FRAMING_STATUS = {"not_started", "draft", "confirmed"}

# Packages that only some stages need, so new projects do not install them up front. A stage checks for them with
# `ml_common.py need <package>` and asks the user before running `uv add`. Package name: import name.
IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml", "optuna": "optuna", "statsmodels": "statsmodels",
                "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------- names

def slugify(name):
    return re.sub(r"[^a-z0-9._-]+", "-", str(name).lower()).strip("-") or "project"


def file_slug(name):
    """A safe part of a file name, for example a dataset name in 04_eda_customers.ipynb."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "data"


def kernel_name(project_name):
    """The Jupyter kernel name for a project, for example 'martech_adtech_data'."""
    return slugify(project_name)


def kernel_display(project_name, python_version=None):
    """How editors show the project kernel, for example 'martech_adtech_data (.venv, Python 3.11.6)'."""
    return f"{project_name} (.venv, Python {python_version or platform.python_version()})"


def notebook_name(stage, suffix=None):
    """Numbered notebook name for a workflow stage, in workflow order: notebook_name('business') gives
    '01_business_framing.ipynb', and notebook_name('eda', 'customers') gives '04_eda_customers.ipynb'."""
    n = STAGE_NUMBER.get(stage)
    stem = NOTEBOOK_BASE.get(stage, stage)
    base = f"{n:02d}_{stem}" if n is not None else stem
    return base + (f"_{file_slug(suffix)}" if suffix else "") + ".ipynb"


def doc_name(stage, suffix=None):
    """Document name for a stage, for example doc_name('eda', 'customers') gives 'eda_customers.md'."""
    return stage + (f"_{file_slug(suffix)}" if suffix else "") + ".md"


# ---------------------------------------------------------------- project root

def _walk_up(start):
    d = os.path.abspath(start)
    home = os.path.expanduser("~")
    while True:
        yield d
        if d == home or os.path.dirname(d) == d:
            return
        d = os.path.dirname(d)


def find_project_root(start="."):
    """Return (root, reason). The root is the nearest folder with ml_spec.yaml, else the nearest with .venv,
    else the nearest with .git. The search stops at the home folder. Returns (None, reason) when none is found."""
    for marker, why in ((SPEC_NAME, "has ml_spec.yaml"), (".venv", "has .venv"), (".git", "has .git")):
        for d in _walk_up(start):
            if os.path.exists(os.path.join(d, marker)):
                return d, why
    return None, "no folder with ml_spec.yaml, .venv, or .git between here and the home folder"


def spec_path(explicit=None, start="."):
    """The spec path: the explicit path, else <project root>/ml_spec.yaml, else ./ml_spec.yaml."""
    if explicit:
        return os.path.abspath(explicit)
    root, _ = find_project_root(start)
    return os.path.join(root or os.path.abspath(start), SPEC_NAME)


def project_python(root):
    """The project's own interpreter, or None when the project has no .venv."""
    for rel in ("bin/python", "Scripts/python.exe"):
        p = os.path.join(root, ".venv", rel)
        if os.path.exists(p):
            return p
    return None


def project_paths(spec, root):
    """Absolute folder paths for the project, from spec['paths'] with defaults for missing keys."""
    paths = dict(DEFAULT_PATHS)
    paths.update({k: v for k, v in ((spec or {}).get("paths") or {}).items() if v})
    return {k: (v if os.path.isabs(v) else os.path.normpath(os.path.join(root, v))) for k, v in paths.items()}


def relpath(path, root):
    """path relative to root when it is inside root, else unchanged."""
    if not path:
        return path
    r = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return path if r.startswith("..") else r


# ---------------------------------------------------------------- spec layout

def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def template(name=None, description=None):
    """An empty spec. Every key is present so people can see what is expected."""
    now = _now()
    return {
        "spec_version": SPEC_VERSION,
        "project": {
            "name": name,
            "description": description,
            "owner": None,
            "python": None,
            "kernel": None,
            "kernel_display": None,
            "created": now,
            "updated": now,
        },
        "paths": dict(DEFAULT_PATHS),
        "business": {
            "status": "not_started",  # not_started, draft, confirmed
            "business_goal": None,
            "business_kpi": None,
            "decision_supported": None,
            "users": None,
            "capacity": None,
            "current_approach": None,
            "outcome": None,  # what should be predicted, in business terms
            "output_use": None,  # ranked_list, threshold_decision, probability, quantity, other
            "v1_bar_intent": None,
            "ml_needed": None,  # yes, no, unsure
            "ml_needed_reason": None,
            "constraints": {
                "latency": None,
                "interpretability": None,
                "fairness": None,
                "data_access": None,
                "other": [],
            },
            "out_of_scope": [],
            "risks": [],  # each: {problem, resolution}, found when the framing was checked
        },
        "discovery": {},
        "framing": {
            "status": "not_started",
            "task_type": None,
            "target": {
                "name": None,
                "definition": None,
                "positive_class": None,
                "outcome_window": None,
                "exists_in_data": None,
                "construction": None,
                "label_delay": None,
            },
            "unit_of_prediction": None,
            "entity_id_cols": [],
            "population": None,
            "prediction_time": {
                "description": None,
                "column": None,
                "feature_cutoff": None,
                "cadence": None,
                "horizon": None,
            },
            "success": {
                "primary_metric": None,
                "secondary_metrics": [],
                "v1_bar": None,
                "v1_criteria": [],  # the V1 bar as checks a script can apply; see V1_KINDS
                "guardrails": [],
            },
        },
        "data": {
            "sources": [],
            "date_col": None,
            "id_cols": [],
            "excluded_cols": [],
        },
        "eda": {},
        "notebooks": {},  # stage: {path, decisions_hash, run_at}, written when a stage notebook runs
        "split": {},
        "features": {},
        "baseline": {},
        "training": {},
        "evaluation": {},
        "iterations": {"current": None, "entries": []},  # written by the ml-iterate helper skill
        "assumptions": [],
        "open_questions": [],
        "decisions": [],
        "history": [],
    }


REQUIRED_BUSINESS = ["business_goal", "decision_supported", "outcome"]
RECOMMENDED_BUSINESS = ["business_kpi", "users", "current_approach", "output_use", "v1_bar_intent", "ml_needed"]
REQUIRED_FRAMING = [
    "task_type",
    "target.definition",
    "unit_of_prediction",
    "prediction_time.description",
    "success.primary_metric",
    "success.v1_bar",
]
RECOMMENDED_FRAMING = [
    "success.v1_criteria",
    "target.outcome_window",
    "target.exists_in_data",
    "entity_id_cols",
    "population",
    "prediction_time.feature_cutoff",
    "prediction_time.cadence",
]

# Fields that moved from framing to business when the framing was split into ml-business-framing and ml-problem-framing.
_MOVED_TO_BUSINESS = {
    "business_goal": "business_goal",
    "decision_supported": "decision_supported",
    "users": "users",
    "current_approach": "current_approach",
    "constraints": "constraints",
    "out_of_scope": "out_of_scope",
}


def migrate(spec):
    """Bring a spec written by older skill versions up to date, in place, and return it.

    Specs from before the business section existed kept the business fields in framing. They move to business,
    which takes the framing status. Nothing is deleted except the moved keys.
    """
    if not isinstance(spec, dict):
        return spec
    bus0 = spec.get("business")
    if isinstance(bus0, dict) and isinstance(bus0.get("ml_needed"), bool):
        # YAML reads an unquoted yes or no as true or false.
        bus0["ml_needed"] = "yes" if bus0["ml_needed"] else "no"
    fr = spec.get("framing")
    if isinstance(fr, dict) and any(k in fr for k in _MOVED_TO_BUSINESS):
        bus = dict(spec.get("business") or {})
        for old, new in _MOVED_TO_BUSINESS.items():
            if old in fr:
                v = fr.pop(old)
                if _empty(bus.get(new)) and not _empty(v):
                    bus[new] = v
        succ = fr.get("success") or {}
        if "business_kpi" in succ:
            v = succ.pop("business_kpi")
            if _empty(bus.get("business_kpi")) and not _empty(v):
                bus["business_kpi"] = v
        if _empty(bus.get("outcome")) and not _empty((fr.get("target") or {}).get("definition")):
            bus["outcome"] = fr["target"]["definition"]
        if _empty(bus.get("status")) or bus.get("status") == "not_started":
            st = fr.get("status") or "not_started"
            bus["status"] = st if st in BUSINESS_STATUS else "draft"
        spec["business"] = bus
    return spec


# ---------------------------------------------------------------- reading and writing

def load(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return {}
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            sys.exit(f"{path} is YAML, and PyYAML is not installed in {sys.executable}. Run this with the "
                     f"project's .venv/bin/python, or install it with: {sys.executable} -m pip install pyyaml")
    if data is None:
        return {}
    if not isinstance(data, dict):
        sys.exit(f"{path} does not hold a mapping at the top level, so it is not a valid spec.")
    return data


if yaml is not None:
    class _Dumper(yaml.SafeDumper):
        pass

    def _str_rep(dumper, s):
        return dumper.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)

    _Dumper.add_representer(str, _str_rep)

    def _indent_lists(self, flow=False, indentless=False):  # lists indented under their key
        return yaml.SafeDumper.increase_indent(self, flow, False)

    _Dumper.increase_indent = _indent_lists
else:  # pragma: no cover
    _Dumper = None


HEADER = (
    "# Shared spec for this ML project. Each skill reads it first and writes only its own section.\n"
    "# You can edit it by hand. Keep the keys, and use null for anything not known yet.\n"
    "# Field meanings: reference/spec_fields.md in any of the ML project skills.\n"
)


def dump_text(data):
    if yaml is not None:
        return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, default_flow_style=False,
                         width=110)
    return json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"


def _auto_resolve(data):
    """Close open questions that the spec itself now answers."""
    done = {
        FRAMING_TODO: (((data.get("framing") or {}).get("status") or "not_started") != "not_started",
                       "The ML framing was started with the ml-problem-framing skill."),
        BUSINESS_TODO: (((data.get("business") or {}).get("status") or "not_started") != "not_started",
                        "The business framing was started with the ml-business-framing skill."),
    }
    done[OLD_FRAMING_TODO] = done[BUSINESS_TODO]
    done[PREV_FRAMING_TODO] = done[FRAMING_TODO]
    done[OLD_BUSINESS_TODO] = done[BUSINESS_TODO]
    for q in data.get("open_questions") or []:
        if isinstance(q, dict) and q.get("status") == "open" and q.get("question") in done:
            ok, answer = done[q["question"]]
            if ok:
                q["status"] = "resolved"
                q["answer"] = answer


def save(path, data):
    _auto_resolve(data)
    data = _clean(data)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if yaml is not None:
            f.write(HEADER)
        f.write(dump_text(data))
    os.replace(tmp, path)
    return path


def _clean(x):
    """Turn numpy and pandas values into plain Python so YAML stays readable."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            x = x.item()
        except Exception:
            return str(x)
    if isinstance(x, float):
        return None if x != x else round(x, 6)
    if isinstance(x, (_dt.date, _dt.datetime)):
        return x.isoformat()
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


# ---------------------------------------------------------------- updates

def deep_merge(base, new):
    """Merge new into base. Dicts merge key by key, everything else (lists included) is replaced."""
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _empty(v):
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0)


def fill_missing(base, new):
    """Like deep_merge, but only sets keys that are missing or empty in base."""
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = fill_missing(out[k], v)
        elif _empty(out.get(k)) and not _empty(v):
            out[k] = copy.deepcopy(v)
    return out


def load_or_create(path, name=None):
    """Return (spec, created). Keys added to the template since the spec was written are filled in,
    without touching existing values."""
    if path and os.path.exists(path):
        spec = migrate(load(path))
        merged = deep_merge(template((spec.get("project") or {}).get("name") or name), spec)
        if not spec.get("paths"):
            # An older spec without paths: leave them empty, so ml-project-setup can record the real layout.
            # project_paths() applies the defaults when reading.
            merged["paths"] = {}
        return merged, False
    return template(name), True


def add_open_questions(spec, questions, stage, raised_by):
    """Add questions not already listed (same text). Each item is a string, which uses the given stage,
    or a (question, stage) pair. Returns the number added."""
    spec["open_questions"] = list(spec.get("open_questions") or [])
    existing = {q.get("question") for q in spec["open_questions"] if isinstance(q, dict)}
    ids = [int(str(q.get("id"))[1:]) for q in spec["open_questions"]
           if isinstance(q, dict) and re.fullmatch(r"Q\d+", str(q.get("id", "")))]
    n, added = max(ids, default=0), 0
    for item in questions:
        q, st = item if isinstance(item, (tuple, list)) else (item, stage)
        if q in existing:
            continue
        n += 1
        spec["open_questions"].append({"id": f"Q{n}", "question": q, "stage": st, "raised_by": raised_by,
                                       "status": "open"})
        existing.add(q)
        added += 1
    return added


def log(spec, skill, action):
    spec.setdefault("history", []).append({"at": _now(), "skill": skill, "action": action})
    spec.setdefault("project", {})["updated"] = _now()


def update_section(path, section, data, skill, replace=True, open_questions=None, question_stage=None,
                   extra=None, action=None, name=None):
    """Write one section of the spec and log it. Creates the spec when it does not exist.

    replace=True overwrites the section, which suits generated sections such as eda.
    extra is merged into other sections, for example {"data": {"sources": [...]}}.
    Returns (path, created).
    """
    spec, created = load_or_create(path, name=name)
    spec[section] = data if replace else deep_merge(spec.get(section) or {}, data)
    for k, v in (extra or {}).items():
        spec[k] = deep_merge(spec.get(k) or {}, v) if isinstance(v, dict) else v
    if open_questions:
        add_open_questions(spec, open_questions, question_stage or section, skill)
    log(spec, skill, action or (("created spec and wrote " if created else "updated ") + section))
    save(path, spec)
    return path, created


# ---------------------------------------------------------------- reading the spec

def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def validate(spec):
    """Return (errors, missing_required, missing_recommended)."""
    errors = []
    if not isinstance(spec, dict):
        return ["The spec is not a mapping."], [], []
    v = spec.get("spec_version")
    if v is None:
        errors.append("spec_version is missing.")
    elif v != SPEC_VERSION:
        errors.append(f"spec_version is {v}, but these skills expect {SPEC_VERSION}.")
    fr = spec.get("framing") or {}
    if not isinstance(fr, dict):
        errors.append("framing must be a mapping.")
        fr = {}
    st = fr.get("status")
    if st not in FRAMING_STATUS:
        errors.append(f"framing.status is {st!r}. Use one of: {', '.join(sorted(FRAMING_STATUS))}.")
    tt = fr.get("task_type")
    if tt is not None and tt not in TASK_TYPES:
        errors.append(f"framing.task_type is {tt!r}. Use one of: {', '.join(sorted(TASK_TYPES))}.")
    for key in ("entity_id_cols", "out_of_scope"):
        if fr.get(key) is not None and not isinstance(fr.get(key), list):
            errors.append(f"framing.{key} must be a list.")
    for key in ("assumptions", "open_questions", "decisions", "history"):
        if spec.get(key) is not None and not isinstance(spec.get(key), list):
            errors.append(f"{key} must be a list.")
    paths = spec.get("paths")
    if paths is not None and not isinstance(paths, dict):
        errors.append("paths must be a mapping.")
    data = spec.get("data") or {}
    for key in ("sources", "id_cols", "excluded_cols"):
        if data.get(key) is not None and not isinstance(data.get(key), list):
            errors.append(f"data.{key} must be a list.")
    for i, e in enumerate(data.get("excluded_cols") or []):
        if not (isinstance(e, dict) and e.get("column")):
            errors.append(f"data.excluded_cols[{i}] needs a column and a reason.")
    bus = spec.get("business") or {}
    if not isinstance(bus, dict):
        errors.append("business must be a mapping.")
        bus = {}
    bst = bus.get("status")
    if bst is not None and bst not in BUSINESS_STATUS:
        errors.append(f"business.status is {bst!r}. Use one of: {', '.join(sorted(BUSINESS_STATUS))}.")
    if bus.get("ml_needed") is not None and bus.get("ml_needed") not in ML_NEEDED:
        errors.append(f"business.ml_needed is {bus.get('ml_needed')!r}. Use one of: {', '.join(sorted(ML_NEEDED))}.")
    errors += [f"framing.success.v1_criteria: {p}" for p in v1_problems(((fr.get("success") or {}).get("v1_criteria")))]
    leftovers = [k for k in _MOVED_TO_BUSINESS if k in fr]
    if leftovers:
        errors.append("framing still holds business fields (" + ", ".join(leftovers) + "). They belong in business. "
                      "Any write through ml_common.py moves them.")
    bus_req = [f"business.{k}" for k in REQUIRED_BUSINESS if _empty(_get(bus, k))]
    bus_rec = [f"business.{k}" for k in RECOMMENDED_BUSINESS if _empty(_get(bus, k))]
    fr_req = [f"framing.{k}" for k in REQUIRED_FRAMING if _empty(_get(fr, k))]
    fr_rec = [f"framing.{k}" for k in RECOMMENDED_FRAMING if _empty(_get(fr, k))]
    if bst == "confirmed" and bus_req:
        errors.append("business.status is confirmed, but required fields are empty: " + ", ".join(bus_req) + ".")
    if st == "confirmed" and fr_req:
        errors.append("framing.status is confirmed, but required fields are empty: " + ", ".join(fr_req) + ".")
    return errors, bus_req + fr_req, bus_rec + fr_rec


def eda_mode(spec):
    """'discovery', 'targeted', or None when EDA has not run. Specs written before modes existed count as
    targeted when the EDA had a target."""
    eda = (spec or {}).get("eda") or {}
    if not eda:
        return None
    return eda.get("mode") or ("targeted" if eda.get("target") else "discovery")


def step_done(spec, step):
    bus_status = ((spec.get("business") or {}).get("status")) or "not_started"
    fr_status = ((spec.get("framing") or {}).get("status")) or "not_started"
    if step == "setup":
        return bool((spec.get("project") or {}).get("kernel"))
    if step == "business":
        return bus_status == "confirmed"
    if step == "discovery":
        # Skipped when the ML framing was confirmed directly, for example when the data was already well known.
        return bool(spec.get("discovery")) or fr_status == "confirmed"
    if step == "framing":
        return fr_status == "confirmed"
    if step == "eda":
        return eda_mode(spec) == "targeted" and fr_status == "confirmed"
    return bool(spec.get(step))


def stage_done(spec, stage):
    return step_done(spec, stage)


def next_step(spec):
    """The first step that is not done, as (step, skill, how), or (None, None, None) when all are done."""
    for step, skill, how, _ in STEPS:
        if not step_done(spec, step):
            return step, skill, how
    return None, None, None


def next_stage(spec):
    """The next step as (step, skill). See next_step for how to run it."""
    step, skill, _ = next_step(spec)
    return step, skill


def modelling_blockers(spec):
    """Reasons a modelling stage (split and later) should not start yet. Empty when it can."""
    out = []
    if (spec.get("business") or {}).get("ml_needed") == "no":
        out.append("the business framing concluded that ML is not needed")
    if not step_done(spec, "framing"):
        out.append("the ML framing is not confirmed yet: run ml-problem-framing")
    if not step_done(spec, "eda"):
        out.append("the targeted EDA has not run on the confirmed target: run ml-eda-report")
    return out


def framing_defaults(spec):
    """Settings that later skills take from the framing instead of asking again."""
    fr = (spec or {}).get("framing") or {}
    data = (spec or {}).get("data") or {}
    tgt = fr.get("target") or {}
    tt = fr.get("task_type")
    target_type = {"binary_classification": "categorical", "multiclass_classification": "categorical",
                   "regression": "continuous", "forecasting": "continuous"}.get(tt)
    ids = list(dict.fromkeys([*(fr.get("entity_id_cols") or []), *(data.get("id_cols") or [])]))
    return {
        "status": fr.get("status"),
        "target": tgt.get("name") if tgt.get("exists_in_data") is not False else None,
        "target_type": target_type,
        "task_type": tt,
        "date_col": data.get("date_col") or (fr.get("prediction_time") or {}).get("column"),
        "id_cols": ids,
        "excluded_cols": [e.get("column") for e in (data.get("excluded_cols") or []) if isinstance(e, dict)],
    }


# ---------------------------------------------------------------- consistency between stages

def days_in(text):
    """Days in a phrase such as '60 days after the scoring date', '8 weeks', or 28. None when there is none."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)", str(text), re.I)
    if not m:
        return None
    return float(m.group(1)) * {"day": 1, "week": 7, "month": 30.4, "year": 365}[m.group(2).lower()]


def parse_metric(text, date_col=None):
    """A metric key and its settings from a phrase such as 'Recall at the top 500 customers per week' or 'PR AUC'.
    Returns ({"key", "k", "k_share", "k_per"}, note). Used by the modelling skills and the consistency checks."""
    t = str(text or "").lower()
    m = {"key": None, "k": None, "k_share": None, "k_per": None}
    rules = [(r"recall\s*(at|@)", "recall_at_k"), (r"precision\s*(at|@)", "precision_at_k"), (r"lift\s*(at|@)", "lift_at_k"),
             (r"pr auc|pr-auc|average precision|auprc", "pr_auc"), (r"auc|roc", "roc_auc"),
             (r"log ?loss|cross entropy", "log_loss"), (r"brier", "brier"), (r"rmse|root mean squared", "rmse"),
             (r"wape|weighted absolute percentage", "wape"), (r"mape|mean absolute percentage", "mape"),
             (r"\bmae\b|mean absolute error", "mae"), (r"\br2\b|r squared|r-squared", "r2")]
    for pattern, key in rules:
        if re.search(pattern, t):
            m["key"] = key
            break
    if m["key"] and m["key"].endswith("_at_k"):
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%", t)
        num = re.search(r"(\d[\d,]*)", t)
        if pct:
            m["k_share"] = float(pct.group(1)) / 100
        elif num:
            m["k"] = int(num.group(1).replace(",", ""))
        if re.search(r"\bper\s+(week|day|month|scoring|date|run|batch)", t) and date_col:
            m["k_per"] = date_col
    note = None if m["key"] else f"Could not read a metric from {text!r}. Set the metric key by hand."
    return m, note


# ---------------------------------------------------------------- the V1 bar as checks

# framing.success.v1_criteria: the V1 bar in a form a script (or an agent) can apply. Each criterion is a mapping
# with an id, a kind, and on (cv or test: the scores it is judged on), plus the fields of its kind:
#   beats_model         model (a model or baseline name), margin (default 0): the champion's primary metric is
#                       better than that model's by at least the margin
#   pct_over_reference  pct: the champion is at least pct % better than the baseline reference
#   threshold           metric (default the primary metric), op (<, <=, >, >=), value
#   check               function: a function in the project's src/v1_checks.py, given a context, that returns
#                       (passed, detail); for guardrails such as "every media effect is non negative"
#   manual              text: judged by a person; a bar with a manual criterion is never met by a script alone
V1_KINDS = {"beats_model": ["model"], "pct_over_reference": ["pct"], "threshold": ["op", "value"],
            "check": ["function"], "manual": ["text"]}
V1_ON = ("cv", "test")
LOWER_IS_BETTER = {"rmse", "mae", "mape", "wape", "log_loss", "brier"}


def v1_problems(criteria):
    """Problems with the structure of v1_criteria, in plain sentences."""
    if criteria in (None, []):
        return []
    if not isinstance(criteria, list):
        return ["must be a list of criteria."]
    out, ids = [], set()
    for i, c in enumerate(criteria):
        if not isinstance(c, dict):
            out.append(f"item {i} is not a mapping.")
            continue
        cid, kind = c.get("id") or f"item {i}", c.get("kind")
        if kind not in V1_KINDS:
            out.append(f"{cid}: kind {kind!r} is not one of {', '.join(V1_KINDS)}.")
            continue
        if c.get("on", "cv") not in V1_ON:
            out.append(f"{cid}: on must be cv or test.")
        missing = [k for k in V1_KINDS[kind] if c.get(k) in (None, "")]
        if missing:
            out.append(f"{cid}: {kind} needs " + ", ".join(missing) + ".")
        if kind == "threshold" and c.get("op") not in (None, "<", "<=", ">", ">="):
            out.append(f"{cid}: op must be <, <=, > or >=.")
        if cid in ids:
            out.append(f"{cid}: the id is used twice.")
        ids.add(cid)
    return out


def _better(a, b, lower, margin=0.0):
    return (a <= b - margin) if lower else (a >= b + margin)


def v1_evaluate(spec, on, scores, champion, reference=None, checks=None, primary=None):
    """Apply the V1 criteria judged on `on` (cv or test).

    scores: {model or baseline name: {metric label: value}}, with the primary metric (training.primary_label, such
    as mape or "recall at 500 per week") for every model.
    checks: {function name: (passed or None, detail)}, already computed for the check criteria.
    Returns {"on", "status", "criteria": [{id, kind, status, detail}]}. status is met, not_met, incomplete (a
    criterion could not be judged), needs_human (a manual criterion), or not_set (no criteria for this on). It is
    never not_met because of missing information."""
    tr, bl = (spec or {}).get("training") or {}, (spec or {}).get("baseline") or {}
    primary = primary or tr.get("primary_label") or bl.get("primary_label") or ""
    lower = primary in LOWER_IS_BETTER
    reference = reference or ((bl.get("reference") or {}).get("name"))
    crit = [c for c in (((spec or {}).get("framing") or {}).get("success") or {}).get("v1_criteria") or []
            if isinstance(c, dict) and c.get("on", "cv") == on]
    mine = (scores or {}).get(champion) or {}
    out = []
    for c in crit:
        cid, kind = c.get("id") or c.get("kind"), c.get("kind")
        status, detail = "unavailable", ""
        if kind == "beats_model":
            other = (scores or {}).get(c.get("model")) or {}
            a, b, m = mine.get(primary), other.get(primary), float(c.get("margin") or 0)
            if a is None or b is None:
                detail = f"No {on} {primary} for {'the champion' if a is None else c.get('model')}."
            else:
                status = "pass" if _better(a, b, lower, m) else "fail"
                detail = f"{champion} {a:.4f} against {c.get('model')} {b:.4f}" + (f" (margin {m:g})" if m else "") + "."
        elif kind == "pct_over_reference":
            a, b = mine.get(primary), ((scores or {}).get(reference) or {}).get(primary)
            if a is None or b in (None, 0):
                detail = f"No {on} {primary} for the champion or the reference {reference}."
            else:
                gain = (b - a) / abs(b) * 100 if lower else (a - b) / abs(b) * 100
                status = "pass" if gain >= float(c["pct"]) else "fail"
                detail = f"{gain:+.1f}% against {reference}, needs {float(c['pct']):g}%."
        elif kind == "threshold":
            key = c.get("metric") or primary
            a = mine.get(key)
            if a is None:
                detail = f"No {on} {key} for the champion."
            else:
                v = float(c["value"])
                ok = {"<": a < v, "<=": a <= v, ">": a > v, ">=": a >= v}[c["op"]]
                status = "pass" if ok else "fail"
                detail = f"{key} {a:.4f} {c['op']} {v:g}."
        elif kind == "check":
            res = (checks or {}).get(c.get("function"))
            if res is None:
                detail = f"The check {c.get('function')} was not run."
            else:
                passed, detail = res
                status = "unavailable" if passed is None else ("pass" if passed else "fail")
        elif kind == "manual":
            status, detail = "manual", str(c.get("text") or "")
        out.append({"id": cid, "kind": kind, "status": status, "detail": detail})
    if not out:
        overall = "not_set"
    elif any(r["status"] == "fail" for r in out):
        overall = "not_met"
    elif any(r["status"] == "unavailable" for r in out):
        overall = "incomplete"
    elif any(r["status"] == "manual" for r in out):
        overall = "needs_human"
    else:
        overall = "met"
    return {"on": on, "status": overall, "criteria": out}


def v1_summary(result):
    """One line for a printout or a report."""
    if not result or result.get("status") == "not_set":
        return "no V1 criteria for these scores; compare the V1 bar by hand"
    parts = [f"{r['id']} {r['status']}" for r in result["criteria"]]
    return f"{result['status']} on {result['on']} (" + "; ".join(parts) + ")"


# Words in a V1 bar or a decision that name a kind of model.
MODEL_WORDS = {
    "logistic": ("logistic",), "linear regression": ("ridge", "linear"), "ridge": ("ridge",), "lasso": ("lasso",),
    "decision tree": ("tree",), "random forest": ("forest",), "gradient boost": ("gbm",), "xgboost": ("gbm",),
    "lightgbm": ("gbm",), "catboost": ("gbm",), "neural": ("neural",),
}


def consistency_checks(spec):
    """Settings that disagree between stages, found by fixed rules. Returns a list of
    {"check", "detail", "where": [spec fields]}. Notes in free text are read by the ml-reconcile-notes skill."""
    out = []
    fr, data, eda = spec.get("framing") or {}, spec.get("data") or {}, spec.get("eda") or {}
    sp, fe, bl, bus = (spec.get(k) or {} for k in ("split", "features", "baseline", "business"))
    tgt, pt, success = fr.get("target") or {}, fr.get("prediction_time") or {}, fr.get("success") or {}

    def add(check, detail, *where):
        out.append({"check": check, "detail": detail, "where": list(where)})

    # The same setting in several stages
    targets = {"framing.target.name": tgt.get("name") if tgt.get("exists_in_data") is not False else None,
               "eda.target.column": (eda.get("target") or {}).get("column"), "split.target": sp.get("target"),
               "features.config.target": (fe.get("config") or {}).get("target")}
    seen = {k: v for k, v in targets.items() if v}
    if len(set(seen.values())) > 1:
        add("target", "The target differs between stages: " + ", ".join(f"{k} is {v}" for k, v in seen.items()) + ".", *seen)
    dates = {"framing.prediction_time.column": pt.get("column"), "data.date_col": data.get("date_col"),
             "split.date_col": sp.get("date_col"), "features.config.date_col": (fe.get("config") or {}).get("date_col")}
    seen = {k: v for k, v in dates.items() if v}
    if len(set(seen.values())) > 1:
        add("date column", "The date column differs between stages: " + ", ".join(f"{k} is {v}" for k, v in seen.items()) + ".", *seen)
    ents = fr.get("entity_id_cols") or []
    if sp.get("strategy") == "group" and sp.get("group_col") and ents and sp["group_col"] not in ents:
        add("entity", f"The split groups by {sp['group_col']}, but the framing's entity is {', '.join(ents)}.",
            "split.group_col", "framing.entity_id_cols")

    # Columns excluded in the framing but used as features
    excluded = {e.get("column") for e in data.get("excluded_cols") or [] if isinstance(e, dict)}
    used = {i.get("name") for i in fe.get("inputs") or []} | {i.get("source") for i in fe.get("inputs") or []}
    clash = sorted(c for c in excluded & used if c)
    if clash:
        add("excluded column used", f"Excluded in the framing but used by the features: {', '.join(clash)}.",
            "data.excluded_cols", "features.inputs")

    # The primary metric
    if bl.get("metrics") and success.get("primary_metric"):
        fm, _ = parse_metric(success["primary_metric"], sp.get("date_col"))
        bm = (bl["metrics"] or {}).get("primary") or {}
        if fm.get("key") and bm.get("key") and fm["key"] != bm.get("key"):
            add("primary metric", f"The framing's primary metric is {fm['key']} ({success['primary_metric']}), but the "
                "baseline uses " + str(bm.get("key")) + ".", "framing.success.primary_metric", "baseline.metrics")
        elif fm.get("k") and bm.get("k") and fm["k"] != bm["k"]:
            notes = " ".join(map(str, (spec.get("assumptions") or []) + [d.get("decision", "") + " " + str(d.get("reason", ""))
                                                                       for d in spec.get("decisions") or [] if isinstance(d, dict)]))
            if not (str(fm["k"]) in notes and str(bm["k"]) in notes):
                add("metric k", f"The framing measures the top {fm['k']:,}, the baseline the top {bm['k']:,}, and no "
                    "assumption or decision says why.", "framing.success.primary_metric", "baseline.metrics")

    # A V1 bar or business intent that names a model that is not built
    if bl.get("models"):
        built = {str(m.get("type")) for m in bl["models"]} | {str(m.get("name")).lower() for m in bl["models"]}
        for field, text in (("framing.success.v1_bar", success.get("v1_bar")), ("business.v1_bar_intent", bus.get("v1_bar_intent"))):
            low = str(text or "").lower()
            if "baseline" not in low:
                continue
            for word, types in MODEL_WORDS.items():
                if word in low and not any(t in built for t in types):
                    add("V1 bar", f"{field} says {text!r}, but there is no {word} baseline (the baselines are "
                        + ", ".join(m.get("name") for m in bl["models"]) + ").", field, "baseline.models")
        ref = (bl.get("reference") or {}).get("name")
        if ref and ref not in {m.get("name") for m in bl["models"]}:
            add("reference", f"The baseline reference {ref} is not one of the baselines.", "baseline.reference")

    # V1 criteria that name a model or baseline that was never built
    known = {m.get("name") for m in (bl.get("models") or [])} | {m.get("name") for m in ((spec.get("training") or {}).get("models") or [])}
    if spec.get("training"):
        for c in success.get("v1_criteria") or []:
            if isinstance(c, dict) and c.get("kind") == "beats_model" and c.get("model") not in known:
                add("V1 criteria", f"V1 criterion {c.get('id')} compares with {c.get('model')}, which is not a built "
                    "model or baseline (" + ", ".join(sorted(n for n in known if n)) + ").",
                    "framing.success.v1_criteria", "training.models")

    # Time settings
    if sp.get("strategy") == "time" and fr.get("task_type") != "forecasting":
        need = max(days_in(tgt.get("outcome_window")) or 0, days_in(tgt.get("label_delay")) or 0)
        if need and (sp.get("gap_days") or 0) < need:
            add("gap", f"The split's gap is {sp.get('gap_days') or 0} days, shorter than the outcome window or label "
                f"delay ({need:g} days).", "split.gap_days", "framing.target")
    horizon = days_in(pt.get("horizon"))
    if fr.get("task_type") == "forecasting" and horizon and fe.get("config"):
        if (fe["config"].get("min_shift_days") or 0) < horizon:
            add("horizon", f"The features allow lags of {fe['config'].get('min_shift_days') or 0} days, shorter than the "
                f"{horizon:g} day horizon.", "features.config.min_shift_days", "framing.prediction_time.horizon")

    # Business conclusion against later work
    if bus.get("ml_needed") == "no" and any(spec.get(s) for s in ("split", "features", "baseline", "training")):
        add("ML needed", "The business framing concluded ML is not needed, but modelling stages have run.",
            "business.ml_needed")

    # Open questions for stages that are finished
    order = [s[0] for s in STEPS]
    nxt = next_step(spec)[0]
    nxt_i = order.index(nxt) if nxt in order else len(order)
    for q in spec.get("open_questions") or []:
        if isinstance(q, dict) and q.get("status") == "open" and q.get("stage") in order and order.index(q["stage"]) < nxt_i:
            add("open question", f"{q.get('id')} is still open, but its stage ({q['stage']}) is finished: {q.get('question')}",
                f"open_questions.{q.get('id')}")
    return out


# ---------------------------------------------------------------- packages

def missing_packages(packages):
    """The packages from the list that this Python cannot import, by package name (scikit-learn, not sklearn)."""
    import importlib.util
    return [p for p in packages if importlib.util.find_spec(IMPORT_NAMES.get(p, p.replace("-", "_"))) is None]


# ---------------------------------------------------------------- CLI

def _columns(path, sheet=None):
    import pandas as pd
    ext = path.lower()
    if ext.endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
            sch = pq.read_schema(path)
            print(f"{pq.ParquetFile(path).metadata.num_rows:,} rows and {len(sch.names):,} columns")
            for f in sch:
                print(f"  {f.name:40} {f.type}")
            return
        except ImportError:
            df = pd.read_parquet(path)
    elif ext.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet or 0, nrows=2000)
    elif ext.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", nrows=2000, low_memory=False)
    elif ext.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, nrows=2000)
    elif ext.endswith(".json"):
        df = pd.read_json(path)
    elif ext.endswith(".feather"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path, nrows=2000, low_memory=False)
    print(f"{df.shape[1]:,} columns (types read from the first {len(df):,} rows)")
    for c in df.columns:
        print(f"  {str(c)[:40]:40} {df[c].dtype}")


def _print_status(spec, path):
    root = os.path.dirname(path)
    print(f"Spec: {path}")
    pr = spec.get("project") or {}
    print(f"Project: {pr.get('name')}" + (f": {pr.get('description')}" if pr.get("description") else ""))
    print("Steps:")
    for i, (step, skill, how, what) in enumerate(STEPS, 1):
        label = "done" if step_done(spec, step) else "not started"
        if step == "discovery" and label == "done" and not spec.get("discovery"):
            label = "skipped"
        elif step in ("business", "framing") and label == "not started" and \
                ((spec.get(step) or {}).get("status")) == "draft":
            label = "draft"
        print(f"  {i:2}. {step:16} {label:12} {skill}" + (f" ({how})" if how else "") + f": {what}")
    oq = [q for q in spec.get("open_questions") or [] if isinstance(q, dict) and q.get("status") == "open"]
    print(f"Open questions: {len(oq)}")
    for q in oq:
        print(f"  {q.get('id')} [{q.get('stage')}] {q.get('question')}")
    step, skill, how = next_step(spec)
    print("Next: " + (f"run the {skill} skill" + (f", {how}" if how else "") + f" ({step})." if step
                      else "every step is done."))
    py = project_python(root)
    print(f"Project Python: {py or 'none (.venv missing, run ml-project-setup)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["root", "path", "paths", "init", "show", "status", "validate", "merge",
                                        "fill", "columns", "notebook-name", "next", "gate", "need"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--spec", help=f"Spec path (default: {SPEC_NAME} at the project root)")
    ap.add_argument("--name", help="Project name for init")
    ap.add_argument("--description", help="One sentence project description for init")
    ap.add_argument("--section", help="Section to print with show")
    ap.add_argument("--suffix", help="Suffix for notebook-name, such as a dataset name")
    ap.add_argument("--skill", default="manual", help="Skill name recorded in history")
    ap.add_argument("--sheet")
    a = ap.parse_args()

    if a.command == "columns":
        if not a.args:
            ap.error("columns needs a data file path")
        _columns(a.args[0], a.sheet)
        return
    if a.command == "notebook-name":
        if not a.args:
            ap.error("notebook-name needs a stage, one of: " + ", ".join(STAGE_NUMBER))
        print(notebook_name(a.args[0], a.suffix))
        return
    if a.command == "need":
        if not a.args:
            ap.error("need takes one or more package names, such as scikit-learn")
        missing = missing_packages(a.args)
        if missing:
            print(f"Not installed in {sys.executable}: {', '.join(missing)}. Ask the user, then run: "
                  f"uv add {' '.join(missing)}")
            sys.exit(1)
        print("Installed: " + ", ".join(a.args))
        return
    if a.command == "root":
        root, why = find_project_root()
        print(f"{root} ({why})" if root else f"No project root found: {why}. Run the ml-project-setup skill.")
        return

    path = spec_path(a.spec)
    exists = os.path.exists(path)

    if a.command == "path":
        print(path if exists else f"No spec yet. It would be created at {path}")
        return
    if a.command == "init":
        if exists:
            sys.exit(f"A spec already exists at {path}. Use merge or fill to change it.")
        spec = template(a.name or os.path.basename(os.path.dirname(path)), a.description)
        log(spec, a.skill if a.skill != "manual" else "ml_common", "created spec")
        save(path, spec)
        print(f"Created {path}" + ("" if yaml else " (as JSON, because PyYAML is not installed in this Python)"))
        return
    if not exists:
        sys.exit(f"No spec at {path}. Create one with: {os.path.basename(sys.executable)} "
                 f"{os.path.abspath(__file__)} init --name <project>")
    spec = migrate(load(path))

    if a.command == "next":
        step, skill, how = next_step(spec)
        print(f"{step} {skill}" + (f" ({how})" if how else "") if step else "done")
        return
    if a.command == "gate":
        blockers = modelling_blockers(spec)
        print("Ready for modelling." if not blockers else "Not ready for modelling: " + "; ".join(blockers) + ".")
        sys.exit(1 if blockers else 0)
    if a.command == "paths":
        root = os.path.dirname(path)
        for k, v in project_paths(spec, root).items():
            print(f"{k:15} {relpath(v, root):25} {'exists' if os.path.isdir(v) else 'missing'}")
    elif a.command == "show":
        part = spec.get(a.section) if a.section else spec
        if a.section and part is None:
            sys.exit(f"The spec has no section '{a.section}'.")
        sys.stdout.write(dump_text(part))
    elif a.command == "status":
        _print_status(spec, path)
    elif a.command == "validate":
        errors, req, rec = validate(spec)
        print(f"Spec: {path}")
        print("Structure: " + ("no problems" if not errors else f"{len(errors)} problem(s)"))
        for e in errors:
            print(f"  ERROR {e}")
        print("Required fields still empty: " + (", ".join(req) if req else "none"))
        print("Recommended fields still empty: " + (", ".join(rec) if rec else "none"))
        sys.exit(1 if errors else 0)
    elif a.command in ("merge", "fill"):
        if len(a.args) != 2:
            ap.error(f"{a.command} needs a section name (or root) and a YAML or JSON file")
        section, src = a.args
        new = load(src)
        spec = load_or_create(path)[0]
        fn = deep_merge if a.command == "merge" else fill_missing
        if section == "root":
            spec = fn(spec, new)
        else:
            spec[section] = fn(spec.get(section) or {}, new)
        log(spec, a.skill, f"{a.command} {section} from {os.path.basename(src)}")
        save(path, spec)
        errors, req, _ = validate(spec)
        print(f"Updated {path}")
        for e in errors:
            print(f"  ERROR {e}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # output piped into head or similar
        sys.stderr.close()
