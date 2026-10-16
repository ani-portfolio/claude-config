#!/usr/bin/env python3
"""Build, run, and protect the stage notebooks of the ML project skills and the A/B testing skills.

Every workflow skill writes one numbered notebook (01_business_framing.ipynb, 02_data_discovery.ipynb, and so
on). The notebook is the working document: markdown for the reasoning, assumptions, and open questions, and
code cells for the decisions and checks. Decision cells are tagged "decision", so they can be found and
compared. Running the notebook writes its decisions to the spec and records the run under
spec["notebooks"][stage], with a hash of the decision cells, so the status skill can tell when a notebook was
edited after it last ran.

The spec library is ab_common (ab_spec.yaml) when it sits next to this file, as in the A/B testing skills, and
ml_common (ml_spec.yaml) otherwise. The canonical copy lives in claude-skills/_ml_shared/nb_common.py, copied
into each skill by tools/sync_ml_shared.py and tools/sync_ab_shared.py.

Commands:
    nb_common.py state <notebook>          # absent, unchanged, or edited (since the skill wrote it)
    nb_common.py next-version <notebook>   # the next free name, such as 02_data_discovery_v2.ipynb
    nb_common.py run <notebook>            # run an existing notebook as it is, for example after editing a decision cell
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import pprint
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The spec library: ab_common in the A/B testing skills, ml_common in the ML project skills. The rest of this file
# calls it ml_common, so the ML behaviour is unchanged.
if os.path.exists(os.path.join(HERE, "ab_common.py")):
    import ab_common as ml_common  # noqa: E402
else:
    import ml_common  # noqa: E402
SPEC_NAME = ml_common.SPEC_NAME
SPEC_LIB = ml_common.__name__

META_KEY = "ml_skills"


# ---------------------------------------------------------------- cells

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text, decision=False):
    meta = {"tags": ["decision"]} if decision else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def lit(value, indent=0, width=100):
    """A Python literal for a value, formatted to be easy to read and edit in a notebook cell: short values on
    one line, longer dicts and lists with one item per line."""
    one = repr(value)
    if not isinstance(value, (dict, list)) or not value or len(one) + indent <= width:
        return one
    pad, inner = " " * indent, " " * (indent + 4)
    if isinstance(value, dict):
        items = [f"{inner}{k!r}: {lit(v, indent + 4, width)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f",\n{pad}}}"
    items = [f"{inner}{lit(v, indent + 4, width)}" for v in value]
    return "[\n" + ",\n".join(items) + f",\n{pad}]"


def setup_cells(stage, title, intro, scripts_dir, notebook_name):
    """The title cell and the setup cell every stage notebook starts with."""
    return [
        md(f"# {title}\n\n{intro}"),
        md(f"## Setup\n\nFinds the project root, loads the shared spec `{SPEC_NAME}`, and checks the kernel. "
           "Run this cell first."),
        code(f'''
import os
import sys
from pathlib import Path

# The project root is the closest folder with {SPEC_NAME}. Everything runs from there.
here = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in here if (p / "{SPEC_NAME}").is_file()), Path.cwd())
os.chdir(ROOT)

# Shared code from the {'A/B testing' if SPEC_LIB == 'ab_common' else 'ML project'} skills
sys.path.insert(0, r"{scripts_dir}")
import {SPEC_LIB}
import nb_common
try:  # MLflow tracking, in the ML project skills: keeps a copy of each chart for the stage's run
    import tracking
    tracking.capture_figures()
except ImportError:
    tracking = None

SPEC_PATH = str(ROOT / "{SPEC_NAME}")
spec = {SPEC_LIB}.migrate({SPEC_LIB}.load(SPEC_PATH)) if os.path.exists(SPEC_PATH) else {SPEC_LIB}.template()
PATHS = {SPEC_LIB}.project_paths(spec, str(ROOT))
NOTEBOOK = os.path.join(PATHS["notebooks"], "{notebook_name}")
STAGE = "{stage}"

print("Project root:", ROOT)
print("Python:", sys.executable)
if (ROOT / ".venv").is_dir() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    print("Warning: this kernel is not the project's .venv. Choose 'Select Kernel' and pick the project kernel.")
'''),
    ]


def tracking_cells(skill):
    """The cells that log the stage's run to MLflow, placed after the cell that writes the spec."""
    return [
        md("## Log to MLflow\n\nLogs this run to the project's experiment on the shared MLflow server: the settings, "
           "the metrics, the tables, and the charts above, read from what this notebook just wrote to the spec. The "
           "run is marked as the current one for this stage. When the server is not reachable, nothing is logged "
           "and the notebook still finishes."),
        code(f'''
MLFLOW_RUN = tracking.log_stage(SPEC_PATH, STAGE, NOTEBOOK, "{skill}") if tracking else None
'''),
    ]


def load_table(path, sheet=None, **kw):
    """Load a data file into a DataFrame based on its extension (the same rules as the EDA engine)."""
    import pandas as pd
    p = str(path)
    low = p.lower()
    if low.endswith((".csv", ".txt", ".csv.gz", ".csv.zip", ".csv.bz2")):
        return pd.read_csv(p, low_memory=False, **kw)
    if low.endswith(".tsv"):
        return pd.read_csv(p, sep="\t", low_memory=False, **kw)
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(p, **kw)
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(p, sheet_name=sheet or 0, **kw)
    if low.endswith(".jsonl"):
        return pd.read_json(p, lines=True, **kw)
    if low.endswith(".json"):
        return pd.read_json(p, **kw)
    if low.endswith(".feather"):
        return pd.read_feather(p, **kw)
    raise ValueError(f"Unsupported file type: {p}")


# ---------------------------------------------------------------- hashes and edit detection

def _sources(nb, decision_only=False):
    out = []
    for c in nb.get("cells", []):
        if decision_only and "decision" not in (c.get("metadata", {}).get("tags") or []):
            continue
        src = c.get("source", "")
        out.append(src if isinstance(src, str) else "".join(src))
    return out


def _hash(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:16]


def source_hash(nb):
    return _hash(_sources(nb))


def decisions_hash(nb):
    return _hash(_sources(nb, decision_only=True))


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def state(path):
    """'absent', 'unchanged' (as the skill wrote it), or 'edited' (a person changed a cell since)."""
    if not os.path.exists(path):
        return "absent"
    try:
        nb = read(path)
    except (OSError, ValueError):
        return "edited"
    meta = nb.get("metadata", {}).get(META_KEY) or {}
    if not meta.get("source_hash"):
        return "edited"
    return "unchanged" if meta["source_hash"] == source_hash(nb) else "edited"


def next_version(path):
    stem, ext = os.path.splitext(path)
    stem = re.sub(r"_v\d+$", "", stem)
    n = 2
    while os.path.exists(f"{stem}_v{n}{ext}"):
        n += 1
    return f"{stem}_v{n}{ext}"


# ---------------------------------------------------------------- kernels

def _env_label():
    ver = platform.python_version()
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return f"{os.path.basename(sys.prefix)} ({ver})"
    return f"Python {ver}"


def find_kernel():
    """(name, display_name) of a registered Jupyter kernel that runs this exact interpreter, or None."""
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        specs = KernelSpecManager().get_all_specs()
    except Exception:
        return None
    me = os.path.normpath(os.path.abspath(sys.executable))
    me_dir = os.path.dirname(me)
    for name, info in specs.items():
        argv = (info.get("spec") or {}).get("argv") or []
        if not argv:
            continue
        exe = os.path.normpath(os.path.abspath(argv[0]))
        if exe == me or (os.path.dirname(exe) == me_dir and os.path.basename(exe).startswith("python")):
            return name, info["spec"].get("display_name", name)
    return None


def kernel_for(spec):
    """The kernel to name in the notebook metadata: the registered kernel for this interpreter, else the
    project kernel from the spec, else a generic one."""
    found = find_kernel()
    if found:
        return found
    pr = (spec or {}).get("project") or {}
    if pr.get("kernel"):
        return pr["kernel"], pr.get("kernel_display") or pr["kernel"]
    return "python3", _env_label()


def _kernel_manager(kdir):
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    spec_dir = os.path.join(kdir, "ml-skills-python")
    os.makedirs(spec_dir, exist_ok=True)
    with open(os.path.join(spec_dir, "kernel.json"), "w") as f:
        json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": _env_label(), "language": "python"}, f)
    return KernelManager(kernel_name="ml-skills-python", kernel_spec_manager=KernelSpecManager(kernel_dirs=[kdir]))


# ---------------------------------------------------------------- building and saving

def build(cells, spec, skill, stage):
    kname, kdisplay = kernel_for(spec)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": kdisplay, "language": "python", "name": kname},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    nb["metadata"][META_KEY] = {"skill": skill, "stage": stage,
                                "generated_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
                                "source_hash": source_hash(nb)}
    return nb


def _write(path, nb, executed=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if executed:
            import nbformat
            nbformat.write(nb, f)
        else:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")


def _normalise(nb):
    """Cell sources as single strings. Jupyter saves them as lists of lines, which nbclient does not accept."""
    for c in nb.get("cells", []):
        if isinstance(c.get("source"), list):
            c["source"] = "".join(c["source"])


_NOTEBOOK_LINE = re.compile(r'^(NOTEBOOK = os\.path\.join\(PATHS\["notebooks"\], )"[^"]*"\)', re.M)


def _point_to(nb, path):
    """Make the setup cell's NOTEBOOK name match the file the notebook is saved as, for example after
    --new-version, so the run is recorded against the right file."""
    name = os.path.basename(path)
    meta = nb.get("metadata", {}).get(META_KEY)
    was_unchanged = bool(meta and meta.get("source_hash") == source_hash(nb))
    changed = False
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code" and "NOTEBOOK = " in c.get("source", ""):
            new = _NOTEBOOK_LINE.sub(lambda m: f'{m.group(1)}"{name}")', c["source"])
            changed |= new != c["source"]
            c["source"] = new
    if changed and was_unchanged:  # a person's edits stay edits
        meta["source_hash"] = source_hash(nb)


def save_and_run(nb, path, mode="ask", execute=True, timeout=1800):
    """Write the notebook and run it once so outputs are filled in. Returns a dict with the final path,
    whether it ran, and any error.

    mode: 'ask' refuses to replace a notebook a person has edited (returns status 'edited'), 'overwrite'
    replaces it, 'new-version' writes the next free _v2, _v3 name instead.
    """
    st = state(path)
    if st == "edited" and mode == "ask":
        return {"status": "edited", "path": path, "next_version": next_version(path)}
    if mode == "new-version" and st != "absent":
        path = next_version(path)
    _normalise(nb)
    _point_to(nb, path)
    # Written before running, so the notebook's own record cell can hash its decision cells from disk.
    _write(path, nb)
    if not execute:
        return {"status": "written", "path": path, "executed": False, "error": None}
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as e:
        return {"status": "written", "path": path, "executed": False,
                "error": f"{e.name} is not installed, so the notebook was saved without running it"}
    node = nbformat.from_dict(nb)
    executed, err = False, None
    started = _dt.datetime.now().replace(microsecond=0).isoformat()
    try:
        with tempfile.TemporaryDirectory() as kdir:
            NotebookClient(node, timeout=timeout, km=_kernel_manager(kdir),
                           resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}}).execute()
        executed = True
    except Exception as e:  # keep the outputs of the cells that did run, so the error is visible in the notebook
        lines = [re.sub(r"\x1b\[[0-9;]*m", "", x) for x in str(e).strip().splitlines() if x.strip()]
        err = f"{type(e).__name__}: " + (lines[-1] if lines else "")
    _write(path, node, executed=True)
    if executed:
        _attach_to_run(nb, path, started)
    return {"status": "written", "path": path, "executed": executed, "error": err}


def _attach_to_run(nb, path, since):
    """Attach the executed notebook to the MLflow run its last cells logged, when the stage logs one."""
    try:
        import tracking
    except ImportError:
        return
    stage = ((nb.get("metadata") or {}).get(META_KEY) or {}).get("stage")
    spec_file = ml_common.spec_path(start=os.path.dirname(os.path.abspath(path)))
    if stage and os.path.exists(spec_file):
        tracking.attach_notebook(path, stage, spec_file, since)


# ---------------------------------------------------------------- called from inside a notebook

def record_run(spec_path, stage, notebook_path, skill):
    """Record that a stage notebook ran, with a hash of its decision cells as saved on disk."""
    try:
        h = decisions_hash(read(notebook_path))
    except (OSError, ValueError):
        h = None
    spec, _ = ml_common.load_or_create(spec_path)
    nbs = dict(spec.get("notebooks") or {})
    nbs[stage] = {"path": ml_common.relpath(notebook_path, os.path.dirname(os.path.abspath(spec_path))),
                  "decisions_hash": h, "run_at": _dt.datetime.now().replace(microsecond=0).isoformat()}
    spec["notebooks"] = nbs
    ml_common.log(spec, skill, f"ran {os.path.basename(notebook_path)}")
    ml_common.save(spec_path, spec)


def write_stage(spec_path, section, values, skill, assumptions=(), open_questions=(), decisions=(),
                question_stage=None, replace=False):
    """Write a stage's decisions from a notebook to the spec.

    values are merged into the section (keys not in values are kept), or replace it when replace is True, which
    suits sections that are computed from the data on every run, such as split. assumptions, open questions, and
    decisions are appended when their text is not there yet, because other skills add to those lists too.
    open_questions items are dicts with question and stage, or plain strings. Returns the validation result.
    """
    spec, _ = ml_common.load_or_create(spec_path)
    spec[section] = values if replace else ml_common.deep_merge(spec.get(section) or {}, values)
    have = set(spec.get("assumptions") or [])
    spec["assumptions"] = list(spec.get("assumptions") or []) + [a for a in assumptions if a not in have]
    today = _dt.date.today().isoformat()
    have_d = {d.get("decision") for d in spec.get("decisions") or [] if isinstance(d, dict)}
    spec["decisions"] = list(spec.get("decisions") or []) + [
        {"date": today, "stage": section, "decision": d["decision"], "reason": d.get("reason")}
        for d in decisions if isinstance(d, dict) and d.get("decision") not in have_d]
    qs = [(q["question"], q.get("stage") or question_stage or section) if isinstance(q, dict) else q
          for q in open_questions]
    ml_common.add_open_questions(spec, qs, question_stage or section, skill)
    ml_common.log(spec, skill, f"wrote {section} from its notebook")
    ml_common.save(spec_path, spec)
    errors, req, rec = ml_common.validate(spec)
    return errors, [r for r in req if r.startswith(section + ".")], [r for r in rec if r.startswith(section + ".")]


def resolve_questions(spec_path, resolved, skill):
    """Mark open questions as resolved: resolved is a list of {"id": "Q3", "answer": "..."}. Returns the IDs that
    were not found."""
    spec, _ = ml_common.load_or_create(spec_path)
    by_id = {q.get("id"): q for q in spec.get("open_questions") or [] if isinstance(q, dict)}
    missing = []
    for r in resolved or []:
        q = by_id.get(r.get("id"))
        if q is None:
            missing.append(r.get("id"))
            continue
        q["status"], q["answer"] = "resolved", r.get("answer")
    if resolved:
        ml_common.log(spec, skill, "resolved " + ", ".join(str(r.get("id")) for r in resolved))
        ml_common.save(spec_path, spec)
    return missing


def show_result(errors, required, recommended, status):
    """Print the validation result of write_stage in plain sentences."""
    if errors:
        print("Problems in the spec:")
        for e in errors:
            print("  " + e)
    if required:
        print("Required fields still empty: " + ", ".join(required))
        if status == "confirmed":
            print("The status is confirmed, but required fields are empty. Fill them in, or set the status to draft.")
    if recommended:
        print("Recommended fields still empty: " + ", ".join(recommended))
    if not errors and not required:
        print(f"Written to {SPEC_NAME} with status {status}.")


def edited_since_run(spec, root, stage):
    """True when the decision cells of a stage notebook on disk differ from the last run recorded in the spec."""
    rec = ((spec or {}).get("notebooks") or {}).get(stage) or {}
    if not rec.get("path") or not rec.get("decisions_hash"):
        return False
    p = os.path.join(root, rec["path"])
    if not os.path.exists(p):
        return False
    try:
        return decisions_hash(read(p)) != rec["decisions_hash"]
    except (OSError, ValueError):
        return False


def edit_cell(path, cell_id, old, new):
    """Replace text in one cell of a notebook: old must appear exactly once in that cell. The skill hash in the
    notebook metadata is kept, so the change counts as a person's edit and no skill replaces it without asking.
    Returns the cell's type ("markdown", "code", or "decision")."""
    nb = read(path)
    cell = next((c for c in nb.get("cells", []) if c.get("id") == cell_id), None)
    if cell is None:
        raise ValueError(f"{path}: no cell with id {cell_id!r}")
    src = cell.get("source", "")
    src = src if isinstance(src, str) else "".join(src)
    n = src.count(old)
    if n != 1:
        raise ValueError(f"{path} {cell_id}: the text to replace appears {n} times, it must appear once: {old[:80]!r}")
    cell["source"] = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    if cell.get("cell_type") == "code" and "decision" in (cell.get("metadata", {}).get("tags") or []):
        return "decision"
    return cell.get("cell_type")


def run_existing(path, timeout=1800):
    """Run a notebook that already exists, keeping its cells as they are (for example after a decision cell
    was edited), and save the outputs. The hash of the cells as the skill wrote them is kept, so a person's
    edits still count as edits and a skill will not replace the notebook without asking."""
    nb = read(path)
    nb.setdefault("metadata", {}).setdefault(META_KEY, {})["rerun_at"] = _dt.datetime.now().replace(microsecond=0).isoformat()
    _normalise(nb)
    return save_and_run(nb, path, mode="overwrite", timeout=timeout)


# ---------------------------------------------------------------- command line wrapper for the stage scripts

# Flags that only print facts. A run with one of them writes no notebook, so its answers are not saved.
INFO_FLAGS = {"--suggest", "--check", "--list-columns", "--suggest-locations", "--kernel-status", "--list-extras",
              "--status", "--dry-run", "--no-run", "--help", "-h"}
EXIT_REASONS = {0: "ok", 1: "failed", 2: "edited_notebook", 3: "edited_code"}


def _answers_files(argv):
    """Answer files named on the command line: positional .yaml or .yml files, and the value after --answers."""
    out = []
    for i, a in enumerate(argv):
        if a == "--answers" and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif a.endswith((".yaml", ".yml")) and not a.startswith("-") and (i == 0 or argv[i - 1] != "--spec"):
            out.append(a)
    return [p for p in dict.fromkeys(out) if os.path.isfile(p) and os.path.basename(p) != SPEC_NAME]


def save_answers(root, stage, skill, argv, answers_paths=()):
    """Keep the settings a stage notebook was built from in answers/<NN>_<stage>.yaml at the project root, so the
    stage can be rebuilt (by a person, ml-iterate, or an agent) without anyone answering questions again. The
    notebook's decision cells stay the record of what the stage decided; the next build replaces this file."""
    number = getattr(ml_common, "STAGE_NUMBER", {}).get(stage)
    name = (f"{number:02d}_" if isinstance(number, int) else "") + f"{stage}.yaml"
    folder = os.path.join(root, "answers")
    os.makedirs(folder, exist_ok=True)
    answers = {}
    for p in answers_paths:
        try:
            loaded = ml_common.load(p)
        except Exception:  # noqa: BLE001
            loaded = None
        if isinstance(loaded, dict):
            answers.update(loaded)
    record = {"skill": skill, "stage": stage, "saved_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
              "command": [os.path.basename(sys.argv[0])] + list(argv), "answers": answers or None}
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Settings the {stage} notebook was last built from, written by {skill}. Rebuild with the same\n"
                "# script and this file. The notebook's decision cells are the record; the next build replaces this.\n")
        f.write(ml_common.dump_text(record))
    return path


def _summary(section, stage):
    """The few fields of a stage section a caller needs to decide what to do next."""
    if not isinstance(section, dict):
        return None
    keep = {"status", "run_at", "mlflow", "strategy", "n_inputs", "reference", "best_non_ml", "primary_label",
            "champion", "data_version", "v1", "decision"}
    out = {k: v for k, v in section.items() if k in keep}
    if stage == "evaluation":
        out["metrics"] = section.get("metrics")
        out["times_test_opened"] = len(section.get("test_opened") or [])
    if stage == "split":
        out["folds"] = ((section.get("cv") or {}).get("folds"))
    if stage == "training":
        out["leaderboard"] = section.get("leaderboard")
    probs = (section.get("checks") or {}).get("problems") if isinstance(section.get("checks"), dict) else None
    out["problems"] = probs if probs is not None else section.get("problems")
    return out


def cli(main_fn, stage, skill):
    """Run a stage script's main(). Adds two things every stage script shares:

    --json   the usual text goes to stderr, and stdout gets one JSON object: stage, skill, exit_code, exit_reason
             (ok, failed, edited_notebook, edited_code), notebook, spec, and a summary of the stage's section.
    answers  after a successful build, the answer files and command line are kept in answers/<NN>_<stage>.yaml.
    """
    argv = list(sys.argv[1:])
    as_json = "--json" in argv
    if as_json:
        sys.argv.remove("--json")
        argv.remove("--json")
    real_stdout = sys.stdout
    if as_json:
        sys.stdout = sys.stderr
    code, error = 0, None
    try:
        main_fn()
    except SystemExit as e:
        if isinstance(e.code, int) or e.code is None:
            code = e.code or 0
        else:
            print(e.code)
            code = 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code, error = 1, f"{type(e).__name__}: {e}"
    finally:
        sys.stdout = real_stdout
    spec_arg = argv[argv.index("--spec") + 1] if "--spec" in argv and argv.index("--spec") + 1 < len(argv) else None
    try:
        spec_path = ml_common.spec_path(spec_arg)
        spec = ml_common.load(spec_path) if os.path.exists(spec_path) else {}
    except Exception:  # noqa: BLE001
        spec_path, spec = None, {}
    building = not (set(argv) & INFO_FLAGS)
    saved = None
    if code == 0 and building and spec_path and os.path.exists(spec_path):
        try:
            saved = save_answers(os.path.dirname(spec_path), stage, skill, argv, _answers_files(argv))
        except Exception as e:  # noqa: BLE001  keeping the answers is a convenience
            print(f"Could not keep the answers: {e}", file=sys.stderr)
    if as_json:
        root = os.path.dirname(spec_path) if spec_path else None
        print(json.dumps({"stage": stage, "skill": skill, "exit_code": code,
                          "exit_reason": EXIT_REASONS.get(code, "failed"), "error": error,
                          "notebook": ((spec.get("notebooks") or {}).get(stage) or {}).get("path"),
                          "spec": spec_path, "answers": ml_common.relpath(saved, root) if saved and root else None,
                          "summary": _summary(spec.get(stage), stage) if building else None},
                         default=str, ensure_ascii=False))
    sys.exit(code)


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("state", "next-version", "run"):
        sys.exit(__doc__)
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "run":
        res = run_existing(path)
        print(f"Ran {path}: " + ("yes" if res["executed"] else f"no ({res['error']})"))
        sys.exit(0 if res["executed"] else 1)
    print(state(path) if cmd == "state" else next_version(path))


if __name__ == "__main__":
    main()
