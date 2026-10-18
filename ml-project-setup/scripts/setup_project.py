#!/usr/bin/env python3
"""Set up a Python ML project folder, the first stage of the ML project skills.

Creates (only what is missing, never overwriting):
  .venv/                         virtual environment made by uv (never deleted or recreated)
  pyproject.toml, uv.lock        dependencies (base packages plus any extras), managed with `uv add`
  data/raw/ data/processed/      input data (read only) and data written by the skills
  notebooks/ src/ models/ docs/ answers/  project folders
  notebooks/00_scratch.ipynb     starter notebook bound to the project kernel
  ml_spec.yaml                   shared spec read and written by every ML project skill
  CLAUDE.md README.md .gitignore
  .vscode/settings.json          points VS Code at .venv (merged into an existing file)

Then installs everything with `uv sync` (the project itself is installed in editable mode, so
`from src.x import y` works from any folder), registers .venv as a named Jupyter kernel, and checks
the result.

A project that already uses requirements.txt keeps it: packages are installed with
`uv pip install -r requirements.txt`, unless --to-pyproject is passed to move it to pyproject.toml.

Needs uv (https://docs.astral.sh/uv/). Uses the standard library only, so any python3 can run it.
The spec is written with the project's .venv Python, so it is YAML.
"""
import argparse
import json
import tempfile
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_common  # noqa: E402  shared with the other ML project skills

BASE_PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "ipykernel",
    "pyyaml",      # ml_spec.yaml
    "nbformat",    # notebooks written by the skills
    "nbclient",    # running those notebooks so outputs are filled in
    "jinja2",      # formatted tables in notebooks (pandas Styler)
]

# Packages that only later stages need are extras, so a project that never models does not install them. The
# modelling skills check for them with `ml_common.py need` and ask before running `uv add`.
EXTRA_GROUPS = {
    "sklearn": ["scikit-learn"],   # split, features, baseline, training, evaluation
    "tuning": ["optuna"],          # tuning in ml-model-training
    "boosting": ["lightgbm", "xgboost", "catboost"],
    "timeseries": ["statsmodels"],
    "tracking": ["mlflow"],
    "sql": ["duckdb"],
    "dev": ["pytest", "ruff"],
}

MIN_PY = (3, 9)
DEFAULT_PYTHON = "3.11"
UV_INSTALL_HELP = ("Install uv with `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`, "
                   "then open a new terminal and rerun.")
DEFAULT_DESCRIPTION = "Describe the goal of this project in one or two sentences."


# ----------------------------------------------------------------------------- helpers

class Report:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.lines = []
        self.problems = []

    def did(self, what):
        self.lines.append(("WOULD " if self.dry_run else "") + what)

    def kept(self, what):
        self.lines.append(f"KEPT {what}")

    def problem(self, what):
        self.problems.append(what)


def slugify(name):
    return ml_common.slugify(name)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def python_version(exe):
    try:
        r = run([str(exe), "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"], timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def find_uv():
    """The uv executable, also looking in the usual install folders that may not be on PATH yet."""
    found = shutil.which("uv")
    if found:
        return found
    for p in ("~/.local/bin/uv", "~/.cargo/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv"):
        p = os.path.expanduser(p)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def uv_version(uv):
    r = run([uv, "--version"])
    return r.stdout.strip() if r.returncode == 0 else None


def package_name(name):
    """A valid Python package distribution name for pyproject.toml."""
    n = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return n if n and n[0].isalnum() else f"project-{n or 'ml'}"


def write_new(path, text, rep, label=None):
    """Write a file only if it does not exist yet."""
    label = label or path.name
    if path.exists():
        rep.kept(f"{label} (already exists, not changed)")
        return False
    if not rep.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    rep.did(f"create {label}")
    return True


# ----------------------------------------------------------------------------- templates

# The workflow section in CLAUDE.md. ab-project-setup replaces both with the A/B testing versions.
WORKFLOW_HEADING = "## ML workflow"
SPEC_FILE = ml_common.SPEC_NAME
WORKFLOW_SECTION = """## ML workflow
This project uses the ML project skills. `ml_spec.yaml` at the project root is the shared record of every
decision: read it before starting any modelling work, and never contradict it without asking the user.
* Steps, in order: ml-project-setup, ml-business-framing (the mode and two anchors, before the data),
  ml-data-discovery, ml-business-framing again (the rest, drafted from the data), ml-problem-framing, ml-eda-report,
  ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, ml-offline-evaluation.
* Each step writes one numbered notebook in `notebooks/`: `01_business_framing.ipynb`, `02_data_discovery.ipynb`,
  `03_problem_framing.ipynb`, `04_eda_<dataset>.ipynb`, `05_split.ipynb`, `06_features.ipynb`, `07_baseline.ipynb`,
  `08_training.ipynb`, `09_evaluation.ipynb`. The notebook is the record of that step.
* Cells marked *Decision* hold the answers. To change a decision, edit that cell and run the whole notebook,
  which writes the answers to `ml_spec.yaml`. Do not edit the spec sections those notebooks own by hand.
* The data dictionary is `docs/data_dictionary.yaml`. Check a column's meaning and timing there before using it.
* `answers/` keeps the settings each stage notebook was last built from, so a stage can be rebuilt without
  asking again. Commit it with the notebooks.
* No modelling (split onwards) before the ML framing is confirmed and the targeted EDA has run.
* See progress and the next skill to run with `/ml-status`.
* Each skill writes only its own section of the spec. Open questions in the spec name the stage that needs them.
* Runs of the modelling stages (split onwards, and reviews) are logged to MLflow, in this project's experiment on
  the shared server (`tracking` in `ml_spec.yaml` has the address and the link). The spec decides; MLflow records.
  Start the server with `python3 ~/Projects/claude-skills/tools/mlflow_server.py start` when it is not running.
"""


def env_lines(dep_mode):
    """How packages are managed, for CLAUDE.md."""
    if dep_mode == "requirements":
        return ("* Install packages with `uv pip install --python .venv/bin/python <package>` and add each one to "
                "`requirements.txt`.")
    return ("* Packages are managed with uv. Add one with `uv add <package>` (or `uv add --dev <package>` for tools such "
            "as pytest), which updates `pyproject.toml` and `uv.lock` and installs it. Never `pip install` into `.venv` "
            "directly, because `uv sync` would not know about it.\n"
            "* After pulling changes, run `uv sync --inexact` to match `.venv` to `uv.lock`.")


def claude_md(name, desc, pyver, kernel, display, paths, dep_mode="pyproject"):
    return f"""# {name}

{desc}

## Environment
* Python {pyver} lives in `.venv/`. Run code with `.venv/bin/python` (or `uv run python`), never a global `python3`.
{env_lines(dep_mode)}
* Never delete or recreate `.venv/`. If it looks broken, stop and tell the user.
* The Jupyter kernel for this project is `{kernel}`, shown in editors as "{display}". Notebooks in `notebooks/` use this kernel.
* `from src.<module> import <name>` works from notebooks, scripts, and tests in any folder, because the project is installed in `.venv`.

## Layout
* `{paths['data_raw']}/`: input data. Treat it as read only. Not committed to git.
* `{paths['data_processed']}/`: data written by the project, such as split sets and feature tables. Not committed to git.
* `{paths['notebooks']}/`: numbered stage notebooks, plus `00_scratch.ipynb` for free exploration.
* `{paths['src']}/`: reusable code. Move logic out of notebooks into here once it is used twice.
* `{paths['models']}/`: fitted pipelines and models. Not committed to git.
* `{paths['docs']}/`: the business framing summary (`business_framing.md`), the data dictionary (`data_dictionary.yaml`), and later the model card.
* `ml_spec.yaml`: the shared project spec. Committed to git.
* `{'requirements.txt' if dep_mode == 'requirements' else 'pyproject.toml'}`: direct dependencies of the project.{'' if dep_mode == 'requirements' else ' `uv.lock` pins every version. Commit both.'}

{WORKFLOW_SECTION}
## Conventions
* Build paths from the project root, not from the current folder. The folders above are also listed under `paths` in `ml_spec.yaml`.
* Keep notebooks short. Functions belong in `src/`.
"""


def readme_md(name, desc, pyver, kernel, display, packages, paths, dep_mode="pyproject"):
    if dep_mode == "requirements":
        install = (f"uv venv .venv --python {pyver}\n"
                   "uv pip install --python .venv/bin/python -r requirements.txt")
        deps = "Dependencies are listed in `requirements.txt`."
    else:
        install = "uv sync"
        deps = ("Dependencies are listed in `pyproject.toml`, and `uv.lock` pins every version, so `uv sync` rebuilds "
                "the same environment. Add a package with `uv add <package>`.")
    return f"""# {name}

{desc}

## Setup

The environment uses Python {pyver} and is managed with [uv](https://docs.astral.sh/uv/). To rebuild it on a new machine:

```bash
{install}
.venv/bin/python -m ipykernel install --user --name {kernel} --display-name "{display}"
```

In VS Code, open this folder on its own (File, Open Folder) so it picks up `.venv` from `.vscode/settings.json`. In a notebook, choose the "{display}" kernel if it is not already selected.

{deps}

## Project structure

```
ml_spec.yaml       shared project spec: framing, data findings, and decisions
{paths['data_raw']}/{' ' * max(1, 15 - len(paths['data_raw']))}input data, read only (not committed)
{paths['data_processed']}/{' ' * max(1, 15 - len(paths['data_processed']))}data written by the project (not committed)
{paths['notebooks']}/{' ' * max(1, 15 - len(paths['notebooks']))}numbered stage notebooks
{paths['src']}/{' ' * max(1, 15 - len(paths['src']))}reusable code, importable as `src`
{paths['models']}/{' ' * max(1, 15 - len(paths['models']))}fitted models (not committed)
{paths['docs']}/{' ' * max(1, 15 - len(paths['docs']))}business framing summary, data dictionary, model card
```

## Workflow

The project follows the ML project skills: ml-project-setup, ml-business-framing (the mode and two anchors),
ml-data-discovery, ml-business-framing again (the rest, drafted from the data), ml-problem-framing, ml-eda-report,
and then ml-train-test-split, ml-feature-engineering, ml-baseline-model,
ml-model-training, and ml-offline-evaluation. Each step writes one numbered notebook (`01_business_framing.ipynb`
to `09_evaluation.ipynb`) that holds its reasoning, decisions, and checks. Running a notebook records its
decisions in `ml_spec.yaml`. Run `/ml-status` in Claude Code to see progress.

## Data

Describe each data source here: where it came from, when it was pulled, and what one row means.
"""


def gitignore(paths):
    """The .gitignore, with the data and model folders from this project's paths."""
    keep = []
    for key in ("data_raw", "data_processed", "models"):
        d = paths[key].rstrip("/")
        keep += [f"{d}/*", f"!{d}/.gitkeep"] if key != "data_raw" or d != "data" else ["data/*", "!data/processed/"]
    body = "\n".join(dict.fromkeys(keep))
    return f"""# Environment
.venv/
.venv.old-*/
__pycache__/
*.pyc
.env
*.tmp
*.egg-info/

# Data and models (the folders are kept, their contents are not committed)
{body}

# Notebooks and tools
.ipynb_checkpoints/
.ruff_cache/
.pytest_cache/
.DS_Store

# Copies the skills keep outside git (git history already holds every version)
iterations/
.iterate_backup/
.reconcile_backup/
mlflow_export/
.claude/settings.local.json
"""


def starter_notebook(name, kernel, display, pyver):
    check = (
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "# Find the project root: the closest folder with ml_spec.yaml, else the closest with .venv\n"
        "here = [Path.cwd(), *Path.cwd().parents]\n"
        "ROOT = next((p for p in here if (p / \"ml_spec.yaml\").is_file()), None) or next(p for p in here if (p / \".venv\").is_dir())\n"
        "\n"
        "print(\"Python:\", sys.executable)\n"
        "assert Path(sys.prefix).resolve() == (ROOT / \".venv\").resolve(), (\n"
        f"    \"Wrong kernel. Choose 'Select Kernel' and pick '{display}'.\"\n"
        ")\n"
        "\n"
        "# Project folders, from the paths section of ml_spec.yaml\n"
        "import yaml\n"
        "spec_file = ROOT / \"ml_spec.yaml\"\n"
        "spec = (yaml.safe_load(spec_file.read_text()) or {}) if spec_file.is_file() else {}\n"
        "PATHS = {**" + repr(ml_common.DEFAULT_PATHS) + ", **(spec.get(\"paths\") or {})}\n"
        "DATA_RAW = ROOT / PATHS[\"data_raw\"]\n"
        "DATA_PROCESSED = ROOT / PATHS[\"data_processed\"]\n"
        "print(\"Raw data:\", DATA_RAW)"
    )
    imports = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "pd.set_option(\"display.max_columns\", 100)\n"
        "sns.set_theme(style=\"whitegrid\")"
    )

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": src.splitlines(keepends=True)}

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                f"# {name}: scratch\n", "\n",
                "Free exploration. The first cell checks the kernel and sets the project folders. Work that should be "
                "kept belongs in the numbered stage notebooks made by the ML project skills.\n"]},
            code(check),
            code(imports),
        ],
        "metadata": {
            "kernelspec": {"display_name": display, "language": "python", "name": kernel},
            "language_info": {"name": "python", "version": pyver},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i}"
    return json.dumps(nb, indent=1) + "\n"


# ----------------------------------------------------------------------------- steps

def has_flat_data(root):
    """True when data/ holds data files directly, as in projects made before the raw and processed layout."""
    d = root / "data"
    if not d.is_dir():
        return False
    return any(p.is_file() and not p.name.startswith(".") for p in d.iterdir())


def spec_value(root, key):
    """A top level section of an existing ml_spec.yaml, or None. Reads YAML with the .venv Python when this
    Python has no PyYAML."""
    spec_file = root / ml_common.SPEC_NAME
    if not spec_file.exists():
        return None
    try:
        return ml_common.load(str(spec_file)).get(key)
    except SystemExit:
        vpy = venv_python(root)
        if not vpy.exists():
            return None
        r = run([str(vpy), "-c", "import json, sys, yaml; d = yaml.safe_load(open(sys.argv[1])) or {}; "
                 "print(json.dumps(d.get(sys.argv[2]), default=str))", str(spec_file), key])
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def project_paths(root, rep):
    """The folder layout for this project. Starts from the spec when it has paths, then the defaults.
    An older project with data files directly in data/ keeps them there: data_raw is recorded as data."""
    paths = dict(ml_common.DEFAULT_PATHS)
    spec_file = root / ml_common.SPEC_NAME
    saved = spec_value(root, "paths") or {}
    if any(saved.values()):
        paths.update({k: v for k, v in saved.items() if v})
        return paths
    if has_flat_data(root) and not (root / "data" / "raw").exists():
        paths["data_raw"] = "data"
        rep.kept("data/ holds data files directly, so it stays the raw data folder (paths.data_raw: data in the spec)")
    return paths


def step_folders(root, paths, rep):
    for key in ["data_raw", "data_processed", "notebooks", "src", "models", "docs"]:
        d = paths[key]
        p = root / d
        if p.is_dir():
            rep.kept(f"{d}/")
        else:
            if not rep.dry_run:
                p.mkdir(parents=True)
            rep.did(f"create {d}/")
    if not (root / "answers").is_dir():   # the settings each stage notebook was built from (nb_common.save_answers)
        if not rep.dry_run:
            (root / "answers").mkdir(parents=True)
        rep.did("create answers/")
    write_new(root / paths["src"] / "__init__.py", "", rep, f"{paths['src']}/__init__.py")
    for key in ["data_raw", "data_processed", "models"]:
        if paths[key] != "data":
            write_new(root / paths[key] / ".gitkeep", "", rep, f"{paths[key]}/.gitkeep")


def venv_python(root):
    return root / ".venv" / "bin" / "python"


def step_venv(root, uv, python_request, rep, rebuild=False):
    """Create .venv with uv, or reuse it if it works. Never delete it: a rebuild renames the old one."""
    vpy = venv_python(root)
    if rebuild and (root / ".venv").exists():
        from datetime import datetime
        old = root / f".venv.old-{datetime.now():%Y%m%d-%H%M%S}"
        if not rep.dry_run:
            (root / ".venv").rename(old)
        rep.did(f"move the old .venv to {old.name} (delete it yourself once the new one works)")
    if (root / ".venv").exists() and not (rebuild and rep.dry_run):
        ver = python_version(vpy)
        if ver:
            rep.kept(f".venv (Python {ver})")
            return ver
        rep.problem(
            ".venv exists but its Python does not run (the base interpreter may have been "
            "upgraded or removed). It was left untouched, and the docs, notebook, and kernel were skipped. "
            "Ask the user, then rerun with --rebuild-venv, which moves it aside and builds a new one."
        )
        return None
    if not rep.dry_run:
        r = run([uv, "venv", str(root / ".venv"), "--python", python_request], cwd=str(root), env=uv_env())
        if r.returncode != 0:
            sys.exit(f"ERROR: uv could not create .venv with Python {python_request}:\n{r.stderr}")
        ver = python_version(vpy)
        if not ver or tuple(int(x) for x in ver.split(".")[:2]) < MIN_PY:
            sys.exit(f"ERROR: .venv was created with Python {ver}, but {MIN_PY[0]}.{MIN_PY[1]} or newer is needed.")
        rep.did(f"create .venv with uv (Python {ver})")
        return ver
    rep.did(f"create .venv with uv (Python {python_request})")
    return python_request


def uv_env():
    """Environment for uv commands: drop an activated environment so uv always uses the project's .venv."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env


def dep_mode(root, to_pyproject=False):
    """'pyproject' for new projects and projects with pyproject.toml. 'requirements' for a project that only has
    requirements.txt, unless --to-pyproject was passed."""
    if (root / "pyproject.toml").exists():
        return "pyproject"
    if (root / "requirements.txt").exists() and not to_pyproject:
        return "requirements"
    return "pyproject"


def req_name(spec):
    """The normalized package name of a requirement such as 'pandas>=2.0'."""
    return re.split(r"[<>=!~\[; @]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def pyproject_deps(path):
    """Return (has_project_table, set of package names in dependencies, optional dependencies, and dev group)."""
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:  # Python before 3.11: approximate with the quoted requirement strings in the file
        has = re.search(r"^\[project\]", text, re.M) is not None
        return has, {req_name(m) for m in re.findall(r'"([A-Za-z0-9][A-Za-z0-9_.\-]*(?:\[[^\]]*\])?[^"]*)"', text)}
    d = tomllib.loads(text)
    proj = d.get("project")
    if not isinstance(proj, dict):
        return False, set()
    names = {req_name(x) for x in proj.get("dependencies", []) if isinstance(x, str)}
    for group in (proj.get("optional-dependencies") or {}).values():
        names |= {req_name(x) for x in group if isinstance(x, str)}
    for group in (d.get("dependency-groups") or {}).values():
        names |= {req_name(x) for x in group if isinstance(x, str)}
    return True, names


def pyproject_text(name, description, pyver):
    major_minor = ".".join(str(pyver).split(".")[:2])
    desc = (description or "").replace("\\", "\\\\").replace('"', '\\"')
    return f"""[project]
name = "{package_name(name)}"
version = "0.1.0"
description = "{desc}"
requires-python = ">={major_minor}"
dependencies = []

# The project is installed into .venv in editable mode, so `from src.x import y` works from any folder.
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
"""


def step_dependencies(root, uv, mode, extras, rep, name, description, pyver, no_install, to_pyproject):
    """Record the wanted packages. Returns the list of wanted package names.

    pyproject mode: create pyproject.toml when missing, then `uv add` only the packages it does not list yet,
    so existing version constraints are never changed. requirements mode: append missing packages to
    requirements.txt, as before.
    """
    wanted = list(BASE_PACKAGES)
    for g in extras:
        if g != "dev":
            wanted += [p for p in EXTRA_GROUPS[g] if p not in wanted]
    dev = list(EXTRA_GROUPS["dev"]) if "dev" in extras else []

    if mode == "requirements":
        req = root / "requirements.txt"
        lines = req.read_text(encoding="utf-8").splitlines()
        have = {req_name(l) for l in lines if l.strip() and not l.strip().startswith(("#", "-"))}
        missing = [p for p in wanted + dev if p.lower() not in have]
        if missing:
            if not rep.dry_run:
                text = req.read_text(encoding="utf-8")
                if text and not text.endswith("\n"):
                    text += "\n"
                req.write_text(text + "\n".join(missing) + "\n", encoding="utf-8")
            rep.did(f"add to requirements.txt: {', '.join(missing)}")
        else:
            rep.kept("requirements.txt (already lists every package)")
        rep.kept("requirements.txt is still the dependency file. Rerun with --to-pyproject to move to pyproject.toml.")
        return wanted + dev

    pyfile = root / "pyproject.toml"
    created = not pyfile.exists()
    if created:
        if not rep.dry_run:
            pyfile.write_text(pyproject_text(name, description, pyver), encoding="utf-8")
        rep.did("create pyproject.toml")
    else:
        rep.kept("pyproject.toml (existing entries and version constraints are not changed)")
    lock_flag = ["--frozen"] if no_install else ["--no-sync"]

    req = root / "requirements.txt"
    if to_pyproject and req.exists():
        if rep.dry_run:
            rep.did("add the packages in requirements.txt to pyproject.toml with uv add -r")
        else:
            r = run([uv, "add", *lock_flag, "-r", str(req)], cwd=str(root), env=uv_env())
            if r.returncode != 0:
                rep.problem("uv add -r requirements.txt failed:\n" + (r.stderr or r.stdout)[-2000:])
                return wanted + dev
            rep.did("add the packages in requirements.txt to pyproject.toml (requirements.txt is kept; "
                    "delete it once you have checked pyproject.toml)")

    if rep.dry_run and created:
        have, has_project = set(), True
    else:
        has_project, have = pyproject_deps(pyfile)
    if not has_project:
        rep.problem("pyproject.toml has no [project] table (it may be a Poetry project or hold only tool "
                    "settings), so uv cannot add packages to it. It was left as is. Add a [project] table, or set up "
                    "the packages by hand.")
        return wanted + dev
    for group, pkgs in (("", wanted), ("--dev", dev)):
        missing = [p for p in pkgs if p.lower() not in have]
        if not missing:
            continue
        label = "dev group" if group else "dependencies"
        if rep.dry_run:
            rep.did(f"uv add to {label}: {', '.join(missing)}")
            continue
        r = run([uv, "add", *lock_flag, *([group] if group else []), *missing], cwd=str(root), env=uv_env())
        if r.returncode != 0:
            rep.problem(f"uv add failed for {', '.join(missing)}:\n" + (r.stderr or r.stdout)[-2000:])
        else:
            rep.did(f"add to pyproject.toml {label}: {', '.join(missing)}")
    if not any(p.lower() not in have for p in wanted + dev) and not created:
        rep.kept("pyproject.toml already lists every base package")
    return wanted + dev


def step_install(root, uv, mode, rep):
    if mode == "requirements":
        if rep.dry_run:
            rep.did("uv pip install -r requirements.txt")
            return
        r = run([uv, "pip", "install", "--python", str(venv_python(root)), "-r", str(root / "requirements.txt")],
                cwd=str(root), env=uv_env())
        what = "install packages from requirements.txt with uv"
    else:
        if rep.dry_run:
            rep.did("uv sync --inexact (installs the project in editable mode and writes uv.lock)")
            return
        # --inexact keeps packages that were installed by hand and are not in uv.lock.
        r = run([uv, "sync", "--inexact"], cwd=str(root), env=uv_env())
        what = "install with uv sync --inexact (uv.lock is up to date)"
    if r.returncode != 0:
        rep.problem("installing packages failed:\n" + (r.stderr or r.stdout)[-2000:])
    else:
        rep.did(what)


def step_src_importable(root, rep):
    """Make sure `import src` works from any folder. An editable install already does this. Otherwise (a
    requirements.txt project, or a pyproject.toml without a build system) add a .pth file to .venv."""
    if rep.dry_run:
        return
    r = run([str(venv_python(root)), "-c", "import src"], cwd="/")
    if r.returncode == 0:
        rep.kept("src is importable from any folder")
        return
    step_import_path(root, rep)


def step_import_path(root, rep):
    """Put the project root on the venv's import path with a .pth file, so `import src` works anywhere."""
    if rep.dry_run:
        rep.did("add project root to the .venv import path")
        return
    r = run([str(venv_python(root)), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
    if r.returncode != 0:
        rep.problem("could not find the .venv site-packages folder")
        return
    pth = Path(r.stdout.strip()) / "_project_root.pth"
    want = str(root) + "\n"
    if pth.exists() and pth.read_text() == want:
        rep.kept("project root on the .venv import path")
        return
    pth.write_text(want)
    rep.did("add project root to the .venv import path")


def kernel_name(name):
    return ml_common.kernel_name(name)


def step_kernel(root, name, pyver, rep):
    kernel = kernel_name(name)
    display = ml_common.kernel_display(name, pyver)
    if rep.dry_run:
        rep.did(f"register Jupyter kernel '{kernel}'")
        return kernel, display
    vpy = venv_python(root)
    r = run([str(vpy), "-m", "ipykernel", "install", "--user", "--name", kernel, "--display-name", display])
    if r.returncode != 0:
        rep.problem("kernel registration failed:\n" + r.stderr[-1500:])
        return kernel, display
    rep.did(f"register Jupyter kernel '{kernel}' shown as \"{display}\"")
    return kernel, display


def step_vscode(root, rep, fix=False):
    """Merge interpreter settings into .vscode/settings.json without removing anything."""
    path = root / ".vscode" / "settings.json"
    wanted = {
        "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
        "python.terminal.activateEnvironment": True,
        "jupyter.notebookFileRoot": "${workspaceFolder}",
    }
    current = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rep.problem(".vscode/settings.json has comments or invalid JSON, so it was not changed. "
                        "Add these keys by hand: " + json.dumps(wanted))
            return
    key = "python.defaultInterpreterPath"
    if fix and key in current and current[key] != wanted[key]:
        rep.did(f"change {key} from {current[key]!r} to the project .venv")
        current[key] = wanted[key]
        if not rep.dry_run:
            path.write_text(json.dumps(current, indent=4) + "\n", encoding="utf-8")
    if key in current and current[key] != wanted[key]:
        rep.problem(f".vscode/settings.json sets {key} to {current[key]!r}, not this project's .venv. "
                    "It was left as is. Ask the user, then rerun with --fix-vscode to point it at .venv.")
    added = {k: v for k, v in wanted.items() if k not in current}
    if not added:
        rep.kept(".vscode/settings.json (already has the interpreter settings)")
        return
    current.update(added)
    existed = path.exists()
    if not rep.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=4) + "\n", encoding="utf-8")
    rep.did(("update" if existed else "create") + f" .vscode/settings.json ({', '.join(added)})")


def step_verify(root, kernel, rep):
    if rep.dry_run:
        return
    vpy = venv_python(root)
    code = (
        "import importlib, sys\n"
        "bad = []\n"
        "for m in ['pandas','numpy','scipy','matplotlib','seaborn','pyarrow','ipykernel','yaml','nbformat','nbclient','jinja2','src']:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: bad.append(f'{m}: {e}')\n"
        "print('\\n'.join(bad))\n"
    )
    r = run([str(vpy), "-c", code], cwd="/")  # cwd=/ proves `import src` works from any folder
    missing = r.stdout.strip()
    if r.returncode != 0 or missing:
        rep.problem("these imports failed in .venv:\n" + (missing or r.stderr))
    r = run([str(vpy), "-m", "jupyter", "kernelspec", "list", "--json"])
    try:
        specs = json.loads(r.stdout)["kernelspecs"]
        argv0 = specs[kernel]["spec"]["argv"][0]
        if os.path.normpath(argv0) != os.path.normpath(str(vpy)):
            rep.problem(f"kernel '{kernel}' points at {argv0}, not this project's .venv")
    except (ValueError, KeyError):
        rep.problem(f"kernel '{kernel}' is not registered")


def step_claude_md_workflow(root, rep, update=False):
    """An existing CLAUDE.md is never rewritten. If it lacks the ML workflow section, report it, and append
    the section only with --update-claude-md."""
    path = root / "CLAUDE.md"
    if not path.exists() or (rep.dry_run and not path.exists()):
        return
    text = path.read_text(encoding="utf-8")
    if WORKFLOW_HEADING in text:
        start = text.index(WORKFLOW_HEADING)
        nxt = text.find("\n## ", start + 5)
        end = len(text) if nxt == -1 else nxt + 1
        current = text[start:end].rstrip() + "\n"
        if current == WORKFLOW_SECTION:
            return
        if update:
            if not rep.dry_run:
                path.write_text(text[:start] + WORKFLOW_SECTION + ("\n" + text[end:] if text[end:].strip() else ""),
                                encoding="utf-8")
            rep.did(f"replace the {WORKFLOW_HEADING[3:]} section of CLAUDE.md with the current version")
        else:
            rep.problem(f"The {WORKFLOW_HEADING[3:]} section of CLAUDE.md is from an older version of the skills. It "
                        "was left as is. Ask the user, then rerun with --update-claude-md to replace that section only.")
        return
    if update:
        if not rep.dry_run:
            path.write_text(text.rstrip() + "\n\n" + WORKFLOW_SECTION, encoding="utf-8")
        rep.did(f"append the {WORKFLOW_HEADING[3:]} section to CLAUDE.md")
    else:
        rep.problem(f"CLAUDE.md has no {WORKFLOW_HEADING[3:]} section, so Claude sessions in this project will not "
                    f"know about {SPEC_FILE}. It was left as is. Ask the user, then rerun with --update-claude-md to "
                    "append it.")


def step_spec(root, name, description, pyver, kernel, display, paths, rep):
    """Create ml_spec.yaml, or fill empty project fields and paths in an existing one. Never overwrites values.
    Runs ml_common.py with the project's Python, which has PyYAML, so the spec is written as YAML."""
    spec_file = root / ml_common.SPEC_NAME
    existed = spec_file.exists()
    if rep.dry_run:
        rep.did(("fill empty project fields and paths in " if existed else "create ") + ml_common.SPEC_NAME)
        return
    has_yaml = lambda exe: run([exe, "-c", "import yaml"]).returncode == 0
    py = str(venv_python(root))
    if not has_yaml(py):
        py = sys.executable
    as_json = not has_yaml(py)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_common.py")
    if not existed:
        r = run([py, script, "init", "--spec", str(spec_file), "--name", name, "--skill", "ml-project-setup"]
                + (["--description", description] if description else []))
        if r.returncode != 0:
            rep.problem("could not create ml_spec.yaml:\n" + (r.stderr or r.stdout)[-1500:])
            return
    values = {"project": {"name": name, "description": description, "python": pyver, "kernel": kernel,
                          "kernel_display": display},
              "paths": paths,
              # V1 is the first pass; ml-iterate opens V2, V3, ... Each version has its own git branch, ml/v<N>.
              "version": {"current": "V1", "branch": "ml/v1", "history": []},
              "git": {"auto_commit": True}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(values, f)
        tmp = f.name
    try:
        # A new spec takes this project's paths as they are. An existing spec keeps its values, and only
        # empty fields are filled.
        r = run([py, script, "fill" if existed else "merge", "root", tmp, "--spec", str(spec_file),
                 "--skill", "ml-project-setup"])
    finally:
        os.unlink(tmp)
    if r.returncode != 0:
        rep.problem("could not update ml_spec.yaml:\n" + (r.stderr or r.stdout)[-1500:])
        return
    note = " (as JSON, because .venv has no PyYAML yet; it turns into YAML on the next write by a skill)" if as_json else ""
    rep.did(("fill empty project fields and paths in " if existed else "create ") + ml_common.SPEC_NAME + note)


def step_tracking(root, rep):
    """Find or create the project's MLflow experiment on the shared server and write the spec's tracking section.
    A server that is not running is not a problem for the setup: the stages log once it is up."""
    if rep.dry_run:
        rep.did("set up MLflow tracking (the project's experiment on the shared server)")
        return
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking.py")
    r = run([str(venv_python(root)), script, "setup"], cwd=str(root))
    out = (r.stdout or "").strip()
    if r.returncode == 0:
        rep.did(out.splitlines()[0] if out else "set up MLflow tracking")
    else:
        rep.kept("MLflow tracking not set up yet: " + (out or (r.stderr or "")[-300:]).replace("MLflow: ", "")
                 + " Run `setup_project.py <folder> --add-tracking` once it is up.")


def skills_tools():
    """The tools folder of the claude-skills folder this skill lives in (through the ~/.claude/skills link)."""
    p = Path(os.path.realpath(__file__)).parents[2] / "tools"
    return p if p.is_dir() else None


def step_tracing(root, rep):
    """Trace Claude Code sessions in this project into the project's MLflow experiment, and add the SessionStart
    hook that gives every command the session id (ML_AGENT_SESSION), so runs and commits link to their trace."""
    if rep.dry_run:
        rep.did("trace Claude Code sessions in this project into its MLflow experiment")
        return
    tools = skills_tools()
    hook = f'python3 "{tools / "session_env.py"}"' if tools and (tools / "session_env.py").exists() else None
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking.py")
    r = run([str(venv_python(root)), script, "tracing"] + ([hook] if hook else []), cwd=str(root))
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode == 0:
        rep.did(out.splitlines()[-1] if out else "set up Claude Code tracing")
    else:
        rep.kept("Claude Code tracing not set up yet: " + out[-300:] + " Run `setup_project.py <folder> --add-tracking` "
                 "once the MLflow server is up.")


SETUP_FILES = [".gitignore", "CLAUDE.md", "README.md", "pyproject.toml", "uv.lock", "requirements.txt",
               ml_common.SPEC_NAME, "notebooks", "src", "docs", ".vscode/settings.json", ".claude/settings.json",
               "data/raw/.gitkeep", "data/processed/.gitkeep", "models/.gitkeep"]


def step_git(root, rep):
    """A git repository with an empty first commit on main and the work on the version branch (ml/v1). Nothing is
    ever committed to main: the skills commit each stage build on the version branch, and a person merges."""
    if rep.dry_run:
        rep.did("create a git repository (empty first commit on main) and commit the set up on branch ml/v1")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import git_ops
    spec = {}
    try:
        spec = ml_common.load(str(root / SPEC_FILE))
    except (SystemExit, Exception):  # noqa: BLE001  no PyYAML here: the version defaults to V1
        pass
    new_repo = not git_ops.is_repo(str(root))
    ok, msg = git_ops.init_repo(str(root), spec)
    if not ok:
        rep.problem(msg)
        return
    rep.did(msg)
    paths = ["."] if new_repo else [p for p in SETUP_FILES if (root / p).exists()]
    sha = git_ops.commit_paths(str(root), paths, "V1 setup: project set up by ml-project-setup", spec,
                               trailers={"Version": "V1", "Stage": "setup", "Skill": "ml-project-setup"})
    rep.did(f"commit the set up on {git_ops.branch(str(root))} ({sha[:8]})" if sha else "KEPT git: nothing new to commit")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="project folder (created if missing)")
    ap.add_argument("--name", help="project name for docs and the kernel (default: folder name)")
    ap.add_argument("--description", default=DEFAULT_DESCRIPTION)
    ap.add_argument("--python", help=f"Python version or interpreter path for a new .venv, passed to uv (default: the "
                                     f"spec's project.python, else {DEFAULT_PYTHON}). uv downloads it if needed.")
    ap.add_argument("--extras", default="", help="comma separated extra groups: " + ", ".join(EXTRA_GROUPS))
    ap.add_argument("--no-install", action="store_true",
                    help="skip installing packages (files, folders, .venv, and pyproject.toml entries only)")
    ap.add_argument("--to-pyproject", action="store_true",
                    help="in a project that uses requirements.txt, create pyproject.toml from it (ask the user first)")
    ap.add_argument("--no-notebook", action="store_true", help="skip the starter notebook")
    ap.add_argument("--rebuild-venv", action="store_true",
                    help="move an existing .venv to .venv.old-<time> and build a new one (ask the user first)")
    ap.add_argument("--fix-vscode", action="store_true",
                    help="replace a python.defaultInterpreterPath that points elsewhere (ask the user first)")
    ap.add_argument("--update-claude-md", action="store_true",
                    help="append the ML workflow section to an existing CLAUDE.md that lacks it (ask the user first)")
    ap.add_argument("--no-tracking", action="store_true",
                    help="do not add mlflow or set up the project's MLflow experiment")
    ap.add_argument("--no-git", action="store_true", help="do not create a git repository or commit the set up")
    ap.add_argument("--no-tracing", action="store_true", help="do not trace Claude Code sessions into MLflow")
    ap.add_argument("--add-git", action="store_true",
                    help="in an existing project, only create the git repository and commit on the version branch")
    ap.add_argument("--add-tracking", action="store_true",
                    help="in an existing project, only add mlflow and set up its MLflow experiment")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen and change nothing")
    ap.add_argument("--status", action="store_true", help="report the state of an existing project and exit")
    ap.add_argument("--list-extras", action="store_true", help="print the extra package groups and exit")
    a = ap.parse_args()

    if a.list_extras:
        print("Base packages: " + ", ".join(BASE_PACKAGES))
        for g, pkgs in EXTRA_GROUPS.items():
            print(f"{g}: {', '.join(pkgs)}" + (" (added to the dev dependency group)" if g == "dev" else ""))
        return 0

    if not a.path:
        ap.error("the project folder path is required")
    root = Path(os.path.expanduser(a.path)).absolute()
    name = a.name or ((spec_value(root, "project") or {}).get("name") if root.exists() else None) or root.name
    extras = [e.strip() for e in a.extras.split(",") if e.strip()]
    if not a.no_tracking and "tracking" not in extras:
        extras.append("tracking")   # every project logs its modelling runs to the shared MLflow server
    unknown = [e for e in extras if e not in EXTRA_GROUPS]
    if unknown:
        sys.exit(f"ERROR: unknown extra group(s): {', '.join(unknown)}. Choose from {', '.join(EXTRA_GROUPS)}.")

    if a.status:
        return status(root, name)

    rep = Report(a.dry_run)
    if not root.exists():
        if not a.dry_run:
            root.mkdir(parents=True)
        rep.did(f"create {root}")

    uv = find_uv()
    if not uv:
        if not a.dry_run:
            sys.exit("ERROR: uv is not installed, and this skill uses it to create .venv and install packages. "
                     + UV_INSTALL_HELP)
        rep.problem("uv is not installed. " + UV_INSTALL_HELP)
        uv = "uv"
    python_request = a.python or ((spec_value(root, "project") or {}).get("python") if root.exists() else None) \
        or DEFAULT_PYTHON
    description = (a.description if a.description != DEFAULT_DESCRIPTION else None) or \
        ((spec_value(root, "project") or {}).get("description") if root.exists() else None)
    if a.add_git:
        if not (root / SPEC_FILE).exists():
            sys.exit(f"ERROR: --add-git is for an existing project with {SPEC_FILE}. Run the full setup.")
        gi = root / ".gitignore"
        if gi.exists() and "iterations/" not in gi.read_text(encoding="utf-8"):
            if not a.dry_run:
                with open(gi, "a", encoding="utf-8") as f:
                    f.write("\n# Copies the skills keep outside git\niterations/\n.iterate_backup/\n.reconcile_backup/\n"
                            "mlflow_export/\n.claude/settings.local.json\n")
            rep.did("add the skills' snapshot folders to .gitignore")
        step_git(root, rep)
        pyver = python_version(str(venv_python(root))) if venv_python(root).exists() else None
        return finish(root, kernel_name(name), ml_common.kernel_display(name, pyver), pyver, rep, a.dry_run)
    if a.add_tracking:
        if not (root / SPEC_FILE).exists() or not venv_python(root).exists():
            sys.exit(f"ERROR: --add-tracking is for an existing project with {SPEC_FILE} and .venv. Run the full setup.")
        mode = dep_mode(root)
        pyver = python_version(str(venv_python(root)))
        step_dependencies(root, uv, mode, ["tracking"], rep, name, description, pyver, a.no_install, False)
        if not a.no_install:
            step_install(root, uv, mode, rep)
        step_claude_md_workflow(root, rep, True)
        step_tracking(root, rep)
        if not a.no_tracing:
            step_tracing(root, rep)
        if not a.no_git:
            step_git(root, rep)
        return finish(root, kernel_name(name), ml_common.kernel_display(name, pyver), pyver, rep, a.dry_run)
    paths = project_paths(root, rep)
    step_folders(root, paths, rep)
    pyver = step_venv(root, uv, str(python_request), rep, a.rebuild_venv)
    mode = dep_mode(root, a.to_pyproject)
    kernel, display = kernel_name(name), ml_common.kernel_display(name, pyver)
    packages = []
    if pyver:  # .venv works
        packages = step_dependencies(root, uv, mode, extras, rep, name, description, pyver, a.no_install,
                                     a.to_pyproject)
        if not a.no_install:
            step_install(root, uv, mode, rep)
            step_src_importable(root, rep)
        if not a.no_install or a.dry_run:
            kernel, display = step_kernel(root, name, pyver, rep)
    step_vscode(root, rep, a.fix_vscode)

    if not pyver:
        return finish(root, kernel, display, pyver, rep, a.dry_run)

    write_new(root / "CLAUDE.md", claude_md(name, a.description, pyver or "3", kernel, display, paths, mode), rep)
    step_claude_md_workflow(root, rep, a.update_claude_md)
    write_new(root / "README.md", readme_md(name, a.description, pyver or "3", kernel, display, packages, paths, mode),
              rep)
    write_new(root / ".gitignore", gitignore(paths), rep)
    if not a.no_notebook:
        nb = f"{paths['notebooks']}/00_scratch.ipynb"
        if (root / paths["notebooks"] / "01_exploration.ipynb").exists():
            rep.kept(f"{paths['notebooks']}/01_exploration.ipynb (older starter notebook, 00_scratch.ipynb not added)")
        else:
            write_new(root / nb, starter_notebook(name, kernel, display, pyver or "3"), rep, nb)
    step_spec(root, name, description, pyver, kernel, display, paths, rep)
    if not a.no_tracking and not a.no_install:
        step_tracking(root, rep)
        if not a.no_tracing:
            step_tracing(root, rep)

    if not a.no_install:
        step_verify(root, kernel, rep)
    if not a.no_git:
        step_git(root, rep)   # last, so the commit holds everything the set up wrote
    return finish(root, kernel, display, pyver, rep, a.dry_run)


def finish(root, kernel, display, pyver, rep, dry_run):
    print(f"Project: {root}")
    print(f"Interpreter: {venv_python(root)}" + (f" (Python {pyver})" if pyver else ""))
    uv = find_uv()
    print(f"uv: {uv_version(uv) if uv else 'not installed'}")
    if pyver:
        print(f"Kernel: '{kernel}' shown as \"{display}\"")
    print("Changes:")
    for l in rep.lines:
        print(f"  {l}")
    if rep.problems:
        print("Problems:")
        for p in rep.problems:
            print("  " + p.replace("\n", "\n    "))
        return 1
    print("Result: OK" + (" (dry run, nothing changed)" if dry_run else ""))
    return 0


def status(root, name):
    kernel = kernel_name(name)
    vpy = venv_python(root)
    print(f"Project: {root} ({'exists' if root.exists() else 'missing'})")
    rep = Report(dry_run=True)
    paths = project_paths(root, rep) if root.exists() else dict(ml_common.DEFAULT_PATHS)
    items = [paths[k] for k in ["data_raw", "data_processed", "notebooks", "src", "models", "docs"]]
    for item in items + ["ml_spec.yaml", "pyproject.toml", "uv.lock", "requirements.txt", "CLAUDE.md", "README.md",
                         ".gitignore", ".vscode/settings.json"]:
        print(f"  {item}: {'yes' if (root / item).exists() else 'no'}")
    print(f"  dependency file: {'pyproject.toml' if dep_mode(root) == 'pyproject' else 'requirements.txt'}")
    uv = find_uv()
    print(f"  uv: {uv_version(uv) if uv else 'not installed. ' + UV_INSTALL_HELP}")
    cm = root / "CLAUDE.md"
    if cm.exists():
        print(f"  CLAUDE.md {WORKFLOW_HEADING[3:]} section: {'yes' if WORKFLOW_HEADING in cm.read_text(encoding='utf-8') else 'no'}")
    for line in rep.lines:
        print(f"  note: {line.replace('KEPT ', '')}")
    ver = python_version(vpy) if vpy.exists() else None
    if (root / ".venv").exists():
        print(f"  .venv: {'works, Python ' + ver if ver else 'BROKEN (python does not run)'}")
    else:
        print("  .venv: no")
    if ver:
        r = run([str(vpy), "-m", "jupyter", "kernelspec", "list", "--json"])
        try:
            spec = json.loads(r.stdout)["kernelspecs"].get(kernel)
            if not spec:
                print(f"  kernel '{kernel}': not registered")
            else:
                ok = os.path.normpath(spec["spec"]["argv"][0]) == os.path.normpath(str(vpy))
                print(f"  kernel '{kernel}': {'registered, points at .venv' if ok else 'registered but points elsewhere: ' + spec['spec']['argv'][0]}")
        except ValueError:
            print(f"  kernel '{kernel}': unknown (ipykernel is not installed in .venv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
