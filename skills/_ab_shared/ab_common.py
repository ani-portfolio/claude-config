#!/usr/bin/env python3
"""Shared code for the A/B testing skills: project root, folder layout, kernel naming, and the shared spec file
(ab_spec.yaml).

The canonical copy lives in skills/_ab_shared/ab_common.py. Every A/B testing skill keeps an identical copy
in its own scripts folder so it works on its own. Edit the canonical copy, then run
`python3 tools/sync_ab_shared.py --fix` from the skills folder to update the copies.

It follows ml_common.py from the ML project skills: the same function names, so nb_common.py works with either
library, and the same project layout, so an A/B test can live in a folder set up by ml-project-setup.

Uses only the standard library. PyYAML is used when installed (the project .venv has it); without it the spec is
written as JSON, which is also valid YAML.

Commands (run with the project's Python from the project root, for example .venv/bin/python):
    ab_common.py root                          # the project root and how it was found
    ab_common.py path                          # the spec path
    ab_common.py paths                         # the project folders from the spec, and whether they exist
    ab_common.py init --name promo_test [--description "..."]
    ab_common.py show [--section framing]
    ab_common.py status                        # step progress, open questions, and the next skill to run
    ab_common.py next                          # just the next step and skill
    ab_common.py gate                          # exit 1 with reasons when the analysis should not start yet
    ab_common.py validate                      # structure errors and required fields still empty
    ab_common.py merge framing answers.yaml --skill ab-test-framing   # overwrite keys in a section
    ab_common.py fill project values.json --skill ab-project-setup    # set only keys that are empty
    ab_common.py columns data/raw/test.csv     # column names and types only, never values
    ab_common.py notebook-name analysis        # 04_analysis.ipynb
    ab_common.py need statsmodels              # exit 1 with the uv add command when a package is not installed

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
SPEC_NAME = "ab_spec.yaml"
OTHER_SPECS = ("ml_spec.yaml",)  # a project set up by ml-project-setup is also a project root

# The workflow, in order. Each stage owns one spec section and has one skill. Stage numbers prefix the stage's
# notebooks.
WORKFLOW = [
    # (stage, skill, notebook number, short description)
    ("setup", "ab-project-setup", 0, "project folder, .venv, kernel, and spec"),
    ("framing", "ab-test-framing", 1, "hypothesis, decision, unit, variants and planned split, metrics, ship rule"),
    ("design", "ab-test-design", 2, "alpha, power, minimum detectable effect, sample size, duration"),
    ("validation", "ab-data-validation", 3, "column mapping, sample ratio, assignment integrity, balance, timing"),
    ("analysis", "ab-analysis", 4, "lift with confidence intervals, significance, guardrails"),
    ("segments", "ab-segment-analysis", 5, "planned segments, heterogeneity tests, false discovery control"),
    ("readout", "ab-readout", 6, "result, risks, and recommendation for the decision owner"),
]
NOTEBOOK_BASE = {
    "setup": "scratch",
    "framing": "test_framing",
    "design": "design",
    "validation": "validation",
    "analysis": "analysis",
    "segments": "segments",
    "readout": "readout",
}
STAGES = [w[0] for w in WORKFLOW]
STAGE_NUMBER = {w[0]: w[2] for w in WORKFLOW if w[2] is not None}
STAGE_SKILL = {w[0]: w[1] for w in WORKFLOW}
STEPS = [(stage, skill, "", what) for stage, skill, _, what in WORKFLOW]
OPTIONAL_STEPS = {"design", "segments"}  # can be skipped; the status shows them as skipped

DEFAULT_PATHS = {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "notebooks": "notebooks",
    "src": "src",
    "models": "models",
    "docs": "docs",
}

FRAMING_STATUS = {"not_started", "draft", "confirmed"}
DESIGN_STATUS = {"not_started", "draft", "confirmed", "skipped"}
TEST_STATES = {"planned", "running", "finished"}
METRIC_TYPES = {"binary", "continuous", "count"}
DIRECTIONS = {"increase", "decrease"}
ROLES = {"control", "treatment"}
VALIDATION_RESULTS = {"passed", "warning", "failed"}
SRM_ALPHA = 0.001  # the usual threshold for a sample ratio mismatch

IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml", "statsmodels": "statsmodels"}

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------- names

def slugify(name):
    return re.sub(r"[^a-z0-9._-]+", "-", str(name).lower()).strip("-") or "project"


def file_slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") or "data"


def kernel_name(project_name):
    """The Jupyter kernel name for a project. The same rule as ml_common, so both workflows share one kernel."""
    return slugify(project_name)


def kernel_display(project_name, python_version=None):
    return f"{project_name} (.venv, Python {python_version or platform.python_version()})"


def notebook_name(stage, suffix=None):
    """Numbered notebook name for a stage: notebook_name('analysis') gives '04_analysis.ipynb'. A/B notebooks start
    with ab_ so they sit apart from the ML notebooks when both workflows share a project."""
    n = STAGE_NUMBER.get(stage)
    stem = NOTEBOOK_BASE.get(stage, stage)
    if stage == "setup":
        return "00_scratch.ipynb"
    base = f"ab{n:02d}_{stem}" if n is not None else stem
    return base + (f"_{file_slug(suffix)}" if suffix else "") + ".ipynb"


def doc_name(stage, suffix=None):
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
    """Return (root, reason). The root is the nearest folder with ab_spec.yaml, else ml_spec.yaml, else .venv, else
    .git. The search stops at the home folder."""
    markers = [(SPEC_NAME, f"has {SPEC_NAME}")] + [(m, f"has {m}") for m in OTHER_SPECS] + \
              [(".venv", "has .venv"), (".git", "has .git")]
    for marker, why in markers:
        for d in _walk_up(start):
            if os.path.exists(os.path.join(d, marker)):
                return d, why
    return None, f"no folder with {SPEC_NAME}, .venv, or .git between here and the home folder"


def spec_path(explicit=None, start="."):
    if explicit:
        return os.path.abspath(explicit)
    root, _ = find_project_root(start)
    return os.path.join(root or os.path.abspath(start), SPEC_NAME)


def project_python(root):
    for rel in ("bin/python", "Scripts/python.exe"):
        p = os.path.join(root, ".venv", rel)
        if os.path.exists(p):
            return p
    return None


def project_paths(spec, root):
    paths = dict(DEFAULT_PATHS)
    paths.update({k: v for k, v in ((spec or {}).get("paths") or {}).items() if v})
    return {k: (v if os.path.isabs(v) else os.path.normpath(os.path.join(root, v))) for k, v in paths.items()}


def relpath(path, root):
    if not path:
        return path
    r = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return path if r.startswith("..") else r


# ---------------------------------------------------------------- spec layout

def _now():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _metric():
    return {"name": None, "type": None, "direction": None, "definition": None}


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
        "framing": {
            "status": "not_started",  # not_started, draft, confirmed
            "test_name": None,
            "test_state": None,  # planned, running, finished
            "hypothesis": None,  # because <change>, <metric> will <move>, for <population>
            "change": None,  # what the treatment does differently
            "decision": None,  # what will be done with the result
            "owner": None,  # who makes the decision
            "unit": None,  # the unit of randomisation, for example user
            "population": None,
            "variants": [],  # each: {name, role: control or treatment, planned_share}
            "primary_metric": _metric(),
            "secondary_metrics": [],  # each like primary_metric
            "guardrail_metrics": [],  # each like primary_metric, plus max_harm
            "mde": None,  # the smallest effect worth acting on, in business terms or as a number
            "ship_rule": None,
            "out_of_scope": [],
            "risks": [],  # each: {problem, resolution}
        },
        "design": {
            "status": "not_started",  # not_started, draft, confirmed, skipped
        },
        "data": {
            "source": None,
            "unit_col": None,
            "group_col": None,
            "variant_values": {},  # variant name: value in group_col
            "metric_cols": {},  # metric name: column
            "date_col": None,
            "pre_treatment_cols": [],  # recorded before assignment, safe for balance checks, CUPED, and segments
            "post_treatment_cols": [],  # recorded after assignment, never used to adjust or segment
        },
        "validation": {},
        "notebooks": {},
        "analysis": {},
        "segments": {},
        "readout": {},
        "assumptions": [],
        "open_questions": [],
        "decisions": [],
        "history": [],
    }


REQUIRED_FRAMING = ["hypothesis", "decision", "unit", "variants", "primary_metric.name"]
RECOMMENDED_FRAMING = ["test_state", "change", "owner", "population", "primary_metric.type",
                       "primary_metric.direction", "mde", "ship_rule", "guardrail_metrics"]


def migrate(spec):
    """Bring a spec written by older skill versions up to date, in place. Nothing to migrate yet."""
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
                     f"project's .venv/bin/python.")
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

    def _indent_lists(self, flow=False, indentless=False):
        return yaml.SafeDumper.increase_indent(self, flow, False)

    _Dumper.increase_indent = _indent_lists
else:  # pragma: no cover
    _Dumper = None


HEADER = (
    "# Shared spec for this A/B test. Each skill reads it first and writes only its own section.\n"
    "# You can edit it by hand. Keep the keys, and use null for anything not known yet.\n"
    "# Field meanings: reference/spec_fields.md in any of the A/B testing skills.\n"
)


def dump_text(data):
    if yaml is not None:
        return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, default_flow_style=False,
                         width=110)
    return json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"


def save(path, data):
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
        return None if x != x else round(x, 8)
    if isinstance(x, (_dt.date, _dt.datetime)):
        return x.isoformat()
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


# ---------------------------------------------------------------- updates

def deep_merge(base, new):
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
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = fill_missing(out[k], v)
        elif _empty(out.get(k)) and not _empty(v):
            out[k] = copy.deepcopy(v)
    return out


def load_or_create(path, name=None):
    if path and os.path.exists(path):
        spec = migrate(load(path))
        return deep_merge(template((spec.get("project") or {}).get("name") or name), spec), False
    return template(name), True


def add_open_questions(spec, questions, stage, raised_by):
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


def _check_metric(errors, where, m, guardrail=False):
    if not isinstance(m, dict):
        errors.append(f"{where} must be a mapping with name, type, direction, and definition.")
        return
    if m.get("type") is not None and m.get("type") not in METRIC_TYPES:
        errors.append(f"{where}.type is {m.get('type')!r}. Use one of: {', '.join(sorted(METRIC_TYPES))}.")
    if m.get("direction") is not None and m.get("direction") not in DIRECTIONS:
        errors.append(f"{where}.direction is {m.get('direction')!r}. Use increase or decrease.")


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
    if fr.get("test_state") is not None and fr["test_state"] not in TEST_STATES:
        errors.append(f"framing.test_state is {fr['test_state']!r}. Use planned, running, or finished.")
    variants = fr.get("variants") or []
    if not isinstance(variants, list):
        errors.append("framing.variants must be a list.")
        variants = []
    roles = []
    for i, var in enumerate(variants):
        if not isinstance(var, dict) or not var.get("name"):
            errors.append(f"framing.variants[{i}] needs a name and a role.")
            continue
        if var.get("role") not in ROLES:
            errors.append(f"framing.variants[{i}].role is {var.get('role')!r}. Use control or treatment.")
        roles.append(var.get("role"))
    if variants and roles.count("control") != 1:
        errors.append(f"framing.variants needs exactly one control, it has {roles.count('control')}.")
    shares = [var.get("planned_share") for var in variants if isinstance(var, dict)]
    if shares and all(isinstance(s, (int, float)) for s in shares) and abs(sum(shares) - 1) > 0.001:
        errors.append(f"framing.variants planned shares add up to {sum(shares):g}, not 1.")
    _check_metric(errors, "framing.primary_metric", fr.get("primary_metric") or {})
    for key in ("secondary_metrics", "guardrail_metrics"):
        ms = fr.get(key)
        if ms is not None and not isinstance(ms, list):
            errors.append(f"framing.{key} must be a list.")
            continue
        for i, m in enumerate(ms or []):
            _check_metric(errors, f"framing.{key}[{i}]", m)
    ds = (spec.get("design") or {}).get("status")
    if ds is not None and ds not in DESIGN_STATUS:
        errors.append(f"design.status is {ds!r}. Use one of: {', '.join(sorted(DESIGN_STATUS))}.")
    res = (spec.get("validation") or {}).get("result")
    if res is not None and res not in VALIDATION_RESULTS:
        errors.append(f"validation.result is {res!r}. Use passed, warning, or failed.")
    data = spec.get("data") or {}
    for key in ("pre_treatment_cols", "post_treatment_cols"):
        if data.get(key) is not None and not isinstance(data.get(key), list):
            errors.append(f"data.{key} must be a list.")
    both = set(data.get("pre_treatment_cols") or []) & set(data.get("post_treatment_cols") or [])
    if both:
        errors.append("These columns are listed as both before and after assignment: " + ", ".join(sorted(both)) + ".")
    for key in ("assumptions", "open_questions", "decisions", "history"):
        if spec.get(key) is not None and not isinstance(spec.get(key), list):
            errors.append(f"{key} must be a list.")
    req = [f"framing.{k}" for k in REQUIRED_FRAMING if _empty(_get(fr, k))]
    rec = [f"framing.{k}" for k in RECOMMENDED_FRAMING if _empty(_get(fr, k))]
    if st == "confirmed" and req:
        errors.append("framing.status is confirmed, but required fields are empty: " + ", ".join(req) + ".")
    return errors, req, rec


def control_variant(spec):
    """The name of the control variant, or None."""
    for v in ((spec or {}).get("framing") or {}).get("variants") or []:
        if isinstance(v, dict) and v.get("role") == "control":
            return v.get("name")
    return None


def test_defaults(spec):
    """Settings that later skills take from earlier sections instead of asking again."""
    fr = (spec or {}).get("framing") or {}
    de = (spec or {}).get("design") or {}
    data = (spec or {}).get("data") or {}
    variants = [v for v in fr.get("variants") or [] if isinstance(v, dict)]
    metrics = [("primary", fr.get("primary_metric") or {})]
    metrics += [("secondary", m) for m in fr.get("secondary_metrics") or [] if isinstance(m, dict)]
    metrics += [("guardrail", m) for m in fr.get("guardrail_metrics") or [] if isinstance(m, dict)]
    return {
        "status": fr.get("status"),
        "variants": variants,
        "control": control_variant(spec),
        "treatments": [v.get("name") for v in variants if v.get("role") == "treatment"],
        "planned_shares": {v.get("name"): v.get("planned_share") for v in variants},
        "metrics": [dict(m, role=r) for r, m in metrics if m.get("name")],
        "alpha": de.get("alpha") or 0.05,
        "sides": de.get("sides") or "two",
        "mde_abs": de.get("mde_abs"),
        "unit_col": data.get("unit_col"),
        "group_col": data.get("group_col"),
        "variant_values": data.get("variant_values") or {},
        "metric_cols": data.get("metric_cols") or {},
        "pre_treatment_cols": data.get("pre_treatment_cols") or [],
        "post_treatment_cols": data.get("post_treatment_cols") or [],
    }


def step_state(spec, step):
    """'done', 'draft', 'skipped', or 'not started'."""
    sec = spec.get(step) or {}
    if step == "setup":
        return "done" if (spec.get("project") or {}).get("kernel") else "not started"
    if step == "framing":
        return {"confirmed": "done", "draft": "draft"}.get(sec.get("status"), "not started")
    if step == "design":
        return {"confirmed": "done", "draft": "draft", "skipped": "skipped"}.get(sec.get("status"), "not started")
    if step == "validation":
        return "done" if sec.get("result") else "not started"
    if step == "analysis":
        return "done" if sec.get("primary") else "not started"
    if step == "segments":
        return {"skipped": "skipped"}.get(sec.get("status"), "done" if sec.get("results") else "not started")
    if step == "readout":
        return {"final": "done", "draft": "draft"}.get(sec.get("status"), "not started")
    return "done" if sec else "not started"


def step_done(spec, step):
    return step_state(spec, step) in ("done", "skipped")


def stage_done(spec, stage):
    return step_done(spec, stage)


def next_step(spec):
    """The first step that is not done, as (step, skill, how). Optional steps (design, segments) are offered in order
    but do not hold back later steps once a later step has started."""
    for i, (step, skill, how, _) in enumerate(STEPS):
        if step_done(spec, step):
            continue
        if step in OPTIONAL_STEPS and any(step_state(spec, s) != "not started" for s, *_ in STEPS[i + 1:]):
            continue
        return step, skill, how
    return None, None, None


def next_stage(spec):
    step, skill, _ = next_step(spec)
    return step, skill


def analysis_blockers(spec):
    """Reasons the analysis (and the segments and readout after it) should not start yet. Empty when it can."""
    out = []
    if not step_done(spec, "framing"):
        out.append("the test framing is not confirmed yet: run ab-test-framing")
    val = spec.get("validation") or {}
    if not val.get("result"):
        out.append("the data has not been validated: run ab-data-validation")
    elif val.get("result") == "failed" and not val.get("override"):
        out.append("the data validation failed (" + "; ".join(val.get("failures") or ["see the validation notebook"])
                   + "), and no override with a reason was recorded")
    return out


# ---------------------------------------------------------------- consistency between stages

def consistency_checks(spec):
    """Settings that disagree between stages, found by fixed rules. Returns a list of
    {"check", "detail", "where": [spec fields]}."""
    out = []
    fr, de, data = spec.get("framing") or {}, spec.get("design") or {}, spec.get("data") or {}
    val, an, seg = spec.get("validation") or {}, spec.get("analysis") or {}, spec.get("segments") or {}

    def add(check, detail, *where):
        out.append({"check": check, "detail": detail, "where": list(where)})

    planned = {v.get("name"): v.get("planned_share") for v in fr.get("variants") or [] if isinstance(v, dict)}
    srm_exp = ((val.get("srm") or {}).get("expected_shares")) or {}
    for name, share in srm_exp.items():
        p = planned.get(name)
        if isinstance(p, (int, float)) and isinstance(share, (int, float)) and abs(p - share) > 0.001:
            add("planned split", f"The framing plans {p:g} for {name}, but the sample ratio check used {share:g}.",
                "framing.variants", "validation.srm.expected_shares")
    pm = (fr.get("primary_metric") or {}).get("name")
    am = (an.get("primary") or {}).get("metric")
    if pm and am and pm != am:
        add("primary metric", f"The framing's primary metric is {pm}, but the analysis reports {am} as primary.",
            "framing.primary_metric", "analysis.primary.metric")
    for key in ("alpha", "sides"):
        if de.get(key) is not None and an.get(key) is not None and de[key] != an[key]:
            add(key, f"The design uses {key} {de[key]}, but the analysis uses {an[key]}.", f"design.{key}", f"analysis.{key}")
    if de.get("mde_abs") is not None and an.get("mde_abs") is not None and de["mde_abs"] != an["mde_abs"]:
        add("MDE", f"The design's minimum detectable effect is {de['mde_abs']}, but the analysis judges practical "
            f"significance against {an['mde_abs']}.", "design.mde_abs", "analysis.mde_abs")
    post = set(data.get("post_treatment_cols") or [])
    used = {s.get("column") for s in seg.get("results") or [] if isinstance(s, dict)}
    bad = sorted(c for c in used & post if c)
    if bad and not seg.get("post_treatment_ok"):
        add("post-treatment segment", "Segments use columns recorded after assignment: " + ", ".join(bad) + ".",
            "segments.results", "data.post_treatment_cols")
    if an.get("cuped_col") and an["cuped_col"] in post:
        add("post-treatment CUPED", f"CUPED adjusts with {an['cuped_col']}, which is recorded after assignment.",
            "analysis.cuped_col", "data.post_treatment_cols")
    ro = spec.get("readout") or {}
    if ro.get("result_label") and (an.get("primary") or {}).get("result_label") and \
            ro["result_label"] != an["primary"]["result_label"]:
        add("readout result", f"The readout says {ro['result_label']!r}, but the analysis says "
            f"{an['primary']['result_label']!r}.", "readout.result_label", "analysis.primary.result_label")

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
    import importlib.util
    return [p for p in packages if importlib.util.find_spec(IMPORT_NAMES.get(p, p.replace("-", "_"))) is None]


# ---------------------------------------------------------------- CLI

def _columns(path, sheet=None):
    import pandas as pd
    ext = path.lower()
    if ext.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    elif ext.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet or 0, nrows=2000)
    elif ext.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", nrows=2000, low_memory=False)
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
        print(f"  {i:2}. {step:12} {step_state(spec, step):12} {skill}: {what}")
    oq = [q for q in spec.get("open_questions") or [] if isinstance(q, dict) and q.get("status") == "open"]
    print(f"Open questions: {len(oq)}")
    for q in oq:
        print(f"  {q.get('id')} [{q.get('stage')}] {q.get('question')}")
    step, skill, _ = next_step(spec)
    print("Next: " + (f"run the {skill} skill ({step})." if step else "every step is done."))
    print(f"Project Python: {project_python(root) or 'none (.venv missing, run ab-project-setup)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["root", "path", "paths", "init", "show", "status", "validate", "merge",
                                        "fill", "columns", "notebook-name", "next", "gate", "need"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--spec", help=f"Spec path (default: {SPEC_NAME} at the project root)")
    ap.add_argument("--name", help="Project name for init")
    ap.add_argument("--description", help="One sentence description for init")
    ap.add_argument("--section", help="Section to print with show")
    ap.add_argument("--suffix", help="Suffix for notebook-name")
    ap.add_argument("--skill", default="manual", help="Skill name recorded in history")
    ap.add_argument("--sheet")
    a = ap.parse_args()

    if a.command == "need":
        if not a.args:
            ap.error("need takes one or more package names, such as statsmodels")
        missing = missing_packages(a.args)
        if missing:
            print(f"Not installed in {sys.executable}: {', '.join(missing)}. Ask the user, then run: "
                  f"uv add {' '.join(missing)}")
            sys.exit(1)
        print("Installed: " + ", ".join(a.args))
        return
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
    if a.command == "root":
        root, why = find_project_root()
        print(f"{root} ({why})" if root else f"No project root found: {why}. Run the ab-project-setup skill.")
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
        log(spec, a.skill if a.skill != "manual" else "ab_common", "created spec")
        save(path, spec)
        print(f"Created {path}" + ("" if yaml else " (as JSON, because PyYAML is not installed in this Python)"))
        return
    if not exists:
        sys.exit(f"No spec at {path}. Run the ab-project-setup skill, or create one with: "
                 f"{os.path.basename(sys.executable)} {os.path.abspath(__file__)} init --name <test>")
    spec = migrate(load(path))

    if a.command == "next":
        step, skill, _ = next_step(spec)
        print(f"{step} {skill}" if step else "done")
        return
    if a.command == "gate":
        blockers = analysis_blockers(spec)
        print("Ready for analysis." if not blockers else "Not ready for analysis: " + "; ".join(blockers) + ".")
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
        errors, _, _ = validate(spec)
        print(f"Updated {path}")
        for e in errors:
            print(f"  ERROR {e}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
