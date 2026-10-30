"""The deployment layout of an ML project: one package per pipeline under src/, after ml-system-design.

The pipelines follow the feature, training, and inference split (plus a monitoring pipeline for label feedback,
monitoring, and the gate reports). Each pipeline is created only when the design needs one of its components.
Like the rest of ml-project-setup, it creates only what is missing and never overwrites:

  src/<pipeline>/__init__.py, jobs/, tests/unit/, tests/component/   (feature_pipeline also gets expectations/
                                                                       when the design validates data)
  tests/system/, tests/unit/test_pipeline_boundaries.py               (no pipeline imports another)
  conftest.py                                                         (marks tests by their level folder)
  pyproject.toml   only missing items: tests left out of the wheel, src in pytest's testpaths, importlib imports,
                   the level markers, and pytest in the dev group. A conflicting value is reported, not changed.
  CLAUDE.md        a "Deployment layout" section, replaced only with --update-claude-md
  ml_spec.yaml     paths.pipelines, so the build skills find the packages

Uses the standard library only, like setup_project.py.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "templates"
LEVELS = ("unit", "component", "system")
MARKERS = {
    "unit": "unit: fast tests of one function, no network or disk outside tmp_path",
    "component": "component: tests of one part of the system (feature store, model registry, API) with its real "
                 "dependencies",
    "system": "system: end to end tests of the running system, from raw data to a served prediction",
}
IMPORT_MODE = "--import-mode=importlib"
HEADING = "## Deployment layout"

# key, package, the design components that call for it, and what it does
PIPELINES = [
    ("feature", "feature_pipeline", ("ingestion", "data_validation", "features"),
     "Feature pipeline: reads raw data and writes features and labels to the feature store."),
    ("training", "training_pipeline", ("training", "registry"),
     "Training pipeline: reads point in time correct training data from the feature store and writes a trained "
     "model to the model registry."),
    ("inference", "inference_pipeline", ("inference", "serving", "failure_handling"),
     "Inference pipeline: reads new features and a trained model and writes predictions to their consumer."),
    ("monitoring", "monitoring_pipeline", ("label_feedback", "monitoring"),
     "Monitoring pipeline: joins actuals to the logged predictions and writes scores, drift, and gate reports."),
]


# ---------------------------------------------------------------- what the design needs

def design_ready(spec):
    """None when the system design is confirmed, else why the layout cannot be added yet."""
    system = (spec or {}).get("system") or {}
    if system.get("status") != "confirmed" or not system.get("components"):
        return ("The system design is not confirmed yet, so there is no deployment layout to set up. Run "
                "/ml-system-design first.")
    return None


def _needed(components, name):
    c = components.get(name) or {}
    return bool(c) and c.get("needed", True) is not False


def pipelines(spec):
    """The pipelines the confirmed design needs, as [(key, package, title, [(component, tool)])]."""
    comps = ((spec or {}).get("system") or {}).get("components") or {}
    out = []
    for key, package, names, title in PIPELINES:
        used = [(n, (comps.get(n) or {}).get("tool")) for n in names if _needed(comps, n)]
        if used:
            out.append((key, package, title, used))
    return out


# ---------------------------------------------------------------- the files

def init_text(title, used):
    lines = [f'"""{title}', "", "From the system design (docs/system_design.md):"]
    lines += [f"* {name}: {tool}" for name, tool in used if tool]
    return "\n".join(lines) + '\n"""\n'


def boundary_test(src, packages):
    text = (TEMPLATES / "test_pipeline_boundaries.py").read_text(encoding="utf-8")
    return text.replace("__SRC__", src).replace("__PIPELINES__", repr(tuple(packages)))


def claude_section(src, found):
    lines = [HEADING,
             "The production system designed in `docs/system_design.md`, one deep package per pipeline, set up by "
             "ml-project-setup.",
             *[f"* `{src}/{package}/`: {title.split(': ', 1)[1]}" for _, package, title, _ in found],
             f"* Each pipeline holds its code, `jobs/` (the entry scripts and job settings the platform runs), and "
             f"`tests/` (left out of the wheel). The modules at the top of `{src}/` are shared by notebooks and "
             "every pipeline.",
             f"* Pipelines import the shared modules and never each other, and the shared modules import no "
             "pipeline. They communicate only through the feature store and the model registry. "
             "`tests/unit/test_pipeline_boundaries.py` enforces this.",
             "* Tests are split by level in folders named `unit/`, `component/`, and `system/`, in `tests/` and in "
             "each pipeline's `tests/`. The root `conftest.py` marks every test by its level folder, so "
             "`.venv/bin/python -m pytest -m unit` runs one level across the project."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- pyproject.toml, edited as text

def _table(text, name):
    """(start, end) of a table's body, or None. The body runs to the next table header."""
    m = re.search(rf"^\[{re.escape(name)}\][ \t]*$", text, re.M)
    if not m:
        return None
    nxt = re.search(r"^\[", text[m.end():], re.M)
    return m.end(), (m.end() + nxt.start() if nxt else len(text))


def _value(body, key):
    """(start, end, value) of `key = value` in a table body; the value may be a list over several lines."""
    m = re.search(rf"^{re.escape(key)}[ \t]*=[ \t]*", body, re.M)
    if not m:
        return None
    i, depth, quote = m.end(), 0, None
    while i < len(body):
        ch = body[i]
        if quote:
            quote = None if ch == quote and body[i - 1] != "\\" else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "\n" and depth <= 0:
            break
        i += 1
    raw = body[m.end():i].split(" #")[0].strip()
    try:
        return m.start(), i, ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return m.start(), i, raw


def _render(key, items, multiline):
    if multiline:
        return f"{key} = [\n" + "".join(f'    "{x}",\n' for x in items) + "]"
    return f"{key} = [" + ", ".join(f'"{x}"' for x in items) + "]"


def _ensure_list(text, table, key, wanted, problems, note=None, contains=None):
    """Add the missing items of a list setting. A value that is not a list is a conflict and stays as it is."""
    t = _table(text, table)
    if t is None:
        return text, []
    body = text[t[0]:t[1]]
    found = _value(body, key)
    if found is None:
        line = (f"# {note}\n" if note else "") + _render(key, wanted, False) + "\n"
        body = body.rstrip("\n") + "\n" + line + ("\n" if t[1] < len(text) else "")
        return text[:t[0]] + body + text[t[1]:], [f"{key} in [{table}]"]
    start, end, value = found
    if not isinstance(value, list):
        problems.append(f"pyproject.toml: [{table}] {key} is {body[start:end].split('=', 1)[1].strip()}, not a list, "
                        f"so {', '.join(wanted)} could not be added. Ask the user, then change it by hand.")
        return text, []
    has = contains or (lambda items, w: w in items)
    missing = [w for w in wanted if not has(value, w)]
    if not missing:
        return text, []
    new = _render(key, value + missing, "\n" in body[start:end])
    return text[:t[0]] + body[:start] + new + body[end:] + text[t[1]:], [f"{key} in [{table}]"]


def update_pyproject(text, src, problems):
    """The text with the missing deployment settings added, and what was added. On any conflict, the text is
    returned unchanged and the conflict is added to problems."""
    conflicts = []
    out, added = text, []
    if _table(out, "tool.hatch.build.targets.wheel") is None:
        conflicts.append("pyproject.toml has no [tool.hatch.build.targets.wheel] table, so the pipeline tests could "
                         "not be left out of the wheel. Ask the user, then add it by hand.")
    out, a = _ensure_list(out, "tool.hatch.build.targets.wheel", "exclude", [f"{src}/*/tests"], conflicts,
                          note="Pipeline tests stay in the repo; the wheel leaves them out")
    added += a
    if _table(out, "tool.pytest.ini_options") is None:
        out = out.rstrip("\n") + "\n\n[tool.pytest.ini_options]\n"
        added.append("[tool.pytest.ini_options]")
    out, a = _ensure_list(out, "tool.pytest.ini_options", "testpaths", ["tests", src], conflicts)
    added += a
    t = _table(out, "tool.pytest.ini_options")
    body = out[t[0]:t[1]]
    found = _value(body, "addopts")
    if found is None:
        line = "# Test files in different pipelines may share a name\n" + f'addopts = "{IMPORT_MODE}"\n'
        out = out[:t[0]] + body.rstrip("\n") + "\n" + line + ("\n" if t[1] < len(out) else "") + out[t[1]:]
        added.append("addopts in [tool.pytest.ini_options]")
    elif IMPORT_MODE not in str(found[2]):
        conflicts.append(f"pyproject.toml: [tool.pytest.ini_options] addopts is {found[2]!r}, so {IMPORT_MODE} "
                         "could not be added (test files in different pipelines may share a name). Ask the user, "
                         "then change it by hand.")
    out, a = _ensure_list(out, "tool.pytest.ini_options", "markers", [MARKERS[l] for l in LEVELS], conflicts,
                          contains=lambda items, w: any(str(x).startswith(w.split(":")[0] + ":") for x in items))
    added += a
    if conflicts:
        problems.extend(conflicts)
        return text, []
    return out, added


def has_dev_pytest(text):
    t = _table(text, "dependency-groups")
    found = _value(text[t[0]:t[1]], "dev") if t else None
    return bool(found) and isinstance(found[2], list) and any(re.match(r"pytest\b", str(x)) for x in found[2])


# ---------------------------------------------------------------- the step

def add(root, spec, src, rep, write_new, update_claude_md=False, uv_add_dev=None, fill_spec=None):
    """Create the deployment layout in root. rep is setup_project's Report; write_new(path, text, rep, label)
    creates a file only when it is missing; uv_add_dev(package) adds a dev dependency; fill_spec(values) fills empty
    spec fields. Returns the files and folders to commit."""
    why = design_ready(spec)
    if why:
        rep.problem(why)
        return []
    root = Path(root)
    found = pipelines(spec)
    for key, package, title, used in found:
        base = f"{src}/{package}"
        write_new(root / base / "__init__.py", init_text(title, used), rep, f"{base}/__init__.py")
        folders = ["jobs", "tests/unit", "tests/component"]
        if key == "feature" and any(n == "data_validation" for n, _ in used):
            folders.append("expectations")
        for f in folders:
            if not any((root / base / f).glob("*")):
                write_new(root / base / f / ".gitkeep", "", rep, f"{base}/{f}/.gitkeep")
    if not any((root / "tests" / "system").glob("*")):
        write_new(root / "tests" / "system" / ".gitkeep", "", rep, "tests/system/.gitkeep")
    write_new(root / "tests" / "unit" / "test_pipeline_boundaries.py",
              boundary_test(src, [p for _, p, _, _ in found]), rep, "tests/unit/test_pipeline_boundaries.py")
    write_new(root / "conftest.py", (TEMPLATES / "conftest.py").read_text(encoding="utf-8"), rep, "conftest.py")
    if (root / "tests" / "conftest.py").exists():
        rep.kept("tests/conftest.py also exists; the root conftest.py now marks every test by its level folder, so "
                 "the user can remove tests/conftest.py if it does the same")

    pp = root / "pyproject.toml"
    if pp.exists():
        text = pp.read_text(encoding="utf-8")
        new, added = update_pyproject(text, src, rep.problems)
        if added:
            if not rep.dry_run:
                pp.write_text(new, encoding="utf-8")
            rep.did("add to pyproject.toml: " + ", ".join(added))
        if not has_dev_pytest(new):
            if rep.dry_run or not uv_add_dev:
                rep.did("add pytest to the dev dependency group (uv add --dev pytest)")
            else:
                uv_add_dev("pytest")
    else:
        rep.problem("There is no pyproject.toml, so the deployment settings for the wheel and pytest were not added.")

    claude_md(root / "CLAUDE.md", claude_section(src, found), rep, update_claude_md)

    want = {key: f"{src}/{package}" for key, package, _, _ in found}
    have = ((spec.get("paths") or {}).get("pipelines")) or {}
    if have and have != want:
        rep.kept(f"paths.pipelines in the spec ({have}) differs from the design's pipelines ({want}); it was left as "
                 "is")
    elif not have and fill_spec:
        if not rep.dry_run:
            fill_spec({"paths": {"pipelines": want}})
        rep.did("record the pipeline packages in paths.pipelines of the spec")
    return [src, "tests", "conftest.py", "pyproject.toml", "uv.lock", "CLAUDE.md"]


def claude_md(path, section, rep, update):
    if not path.exists():
        rep.problem("There is no CLAUDE.md, so the Deployment layout section was not added.")
        return
    text = path.read_text(encoding="utf-8")
    if HEADING not in text:
        if not rep.dry_run:
            path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
        rep.did(f"append the {HEADING[3:]} section to CLAUDE.md")
        return
    start = text.index(HEADING)
    nxt = text.find("\n## ", start + 3)
    end = len(text) if nxt == -1 else nxt + 1
    if text[start:end].rstrip() + "\n" == section:
        return
    if update:
        if not rep.dry_run:
            rest = text[end:]
            path.write_text(text[:start] + section + ("\n" + rest if rest.strip() else ""), encoding="utf-8")
        rep.did(f"replace the {HEADING[3:]} section of CLAUDE.md with the current layout")
    else:
        rep.problem(f"The {HEADING[3:]} section of CLAUDE.md differs from the current layout. It was left as is. "
                    "Ask the user, then rerun with --update-claude-md to replace that section only.")


def status_line(root, spec, src):
    """One line for setup_project.py --status."""
    if design_ready(spec):
        return "deployment layout: not yet (the system design is not confirmed)"
    root = Path(root)
    missing = [f"{src}/{p}" for _, p, _, _ in pipelines(spec) if not (root / src / p / "__init__.py").exists()]
    missing += [f for f in ("conftest.py", "tests/unit/test_pipeline_boundaries.py") if not (root / f).exists()]
    if not missing:
        return "deployment layout: yes"
    return "deployment layout: missing " + ", ".join(missing) + " (add it with --add-deployment)"
