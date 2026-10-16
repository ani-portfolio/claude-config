"""Build (and optionally run) an EDA notebook for a data file.

The notebook imports eda_core from this skill, so every number matches the
markdown report. It is written as plain notebook JSON, and run with nbclient
when nbclient and ipykernel are installed, so outputs are filled in.
"""

import hashlib
import json
import os
import re
import platform
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eda_core import SECTION_ORDER  # noqa: E402
import ml_common  # noqa: E402

# Set by build_notebook from the spec's project.name, so the kernel name matches the one ml-project-setup registered.
PROJECT_NAME = None


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


def _loader(var, path, sheet):
    ext = path.lower()
    if ext.endswith((".parquet", ".pq")):
        return f'{var} = pd.read_parquet(r"{path}")'
    if ext.endswith((".xlsx", ".xlsm", ".xls")):
        return f'{var} = pd.read_excel(r"{path}", sheet_name={sheet!r} if {sheet!r} else 0)'
    if ext.endswith(".tsv"):
        return f'{var} = pd.read_csv(r"{path}", sep="\\t", low_memory=False)'
    if ext.endswith(".jsonl"):
        return f'{var} = pd.read_json(r"{path}", lines=True)'
    if ext.endswith(".json"):
        return f'{var} = pd.read_json(r"{path}")'
    if ext.endswith(".feather"):
        return f'{var} = pd.read_feather(r"{path}")'
    return f'{var} = pd.read_csv(r"{path}", low_memory=False)'


def _env_label():
    """A kernel name like VS Code shows it, for example '.venv (3.11.6)'."""
    ver = platform.python_version()
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return f"{os.path.basename(sys.prefix)} ({ver})"
    return f"Python {ver}"


def _project_name():
    """The project name: the spec's project.name when known, else the folder that holds the environment,
    for example 'martech_adtech_data' for /Users/ani/Projects/martech_adtech_data/.venv."""
    if PROJECT_NAME:
        return PROJECT_NAME
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return os.path.basename(os.path.dirname(os.path.abspath(sys.prefix)))
    return "python"


def _kernel_name():
    """Same naming rule as ml-project-setup (ml_common.kernel_name)."""
    return ml_common.kernel_name(_project_name())


def find_kernel():
    """Return (name, display_name) of a registered Jupyter kernel that runs this exact interpreter, or None.

    Paths are compared without resolving symlinks, because every venv's python is a symlink to the same
    base Python, and resolving would match the wrong project's environment.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        specs = KernelSpecManager().get_all_specs()
    except Exception:
        return None
    me = os.path.normpath(os.path.abspath(sys.executable))
    me_dir = os.path.dirname(me)
    for name, info in specs.items():
        spec = info.get("spec", {})
        argv = spec.get("argv") or []
        if not argv:
            continue
        exe = os.path.normpath(os.path.abspath(argv[0]))
        if exe == me or (os.path.dirname(exe) == me_dir and os.path.basename(exe).startswith("python")):
            return name, spec.get("display_name", name)
    return None


def register_kernel():
    """Register this interpreter as a Jupyter kernel for the current user. Returns (name, display_name)."""
    import subprocess
    name = _kernel_name()
    display = ml_common.kernel_display(_project_name()) if sys.prefix != sys.base_prefix else _env_label()
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", name,
                    "--display-name", display], check=True, capture_output=True, text=True)
    return name, display


def kernel_status():
    found = find_kernel()
    print(f"Interpreter: {os.path.abspath(sys.executable)}")
    if found:
        print(f"Registered Jupyter kernel: yes, name '{found[0]}', shown as '{found[1]}'")
    else:
        print("Registered Jupyter kernel: no. Pass --register-kernel to register it as "
              f"'{_kernel_name()}'.")
    return found


def _kernel_manager(kdir):
    """A kernel that runs on this exact interpreter, so the notebook is executed with the same packages
    that run_eda.py was started with, whatever kernels are registered elsewhere on the machine."""
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager
    spec_dir = os.path.join(kdir, "ml-eda-report-python")
    os.makedirs(spec_dir, exist_ok=True)
    with open(os.path.join(spec_dir, "kernel.json"), "w") as f:
        json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": _env_label(), "language": "python"}, f)
    ksm = KernelSpecManager(kernel_dirs=[kdir])
    return KernelManager(kernel_name="ml-eda-report-python", kernel_spec_manager=ksm)


def build_notebook(nb_path, data_path, mode=None, target=None, date_col=None, id_cols=None, target_type=None,
                   register=False,
                   corr_threshold=0.9, cramers_threshold=0.8,
                   sheet=None, name=None, sample_rows=200_000,
                   max_plots=24, execute=True, timeout=900, spec_path=None, project_name=None, log_scale=False):
    global PROJECT_NAME
    if project_name:
        PROJECT_NAME = project_name
    nb_path = os.path.abspath(nb_path)
    data_path = os.path.abspath(data_path)
    name = name or os.path.basename(data_path).split(".")[0]
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)

    mode = mode or ("targeted" if target else "discovery")
    settings = [f"* **Data:** `{data_path}`",
                "* **Mode:** " + ("targeted, with the target below" if mode == "targeted" else
                                  "discovery, without a target. The summary lists hints for the ML framing.")]
    if target:
        settings.append(f"* **Target column:** `{target}`" + (f" ({target_type})" if target_type else ""))
    if date_col:
        settings.append(f"* **Date column:** `{date_col}`")
    if id_cols:
        settings.append("* **ID columns:** " + ", ".join(f"`{c}`" for c in id_cols))
    if spec_path:
        settings.append(f"* **Shared spec:** `{spec_path}`. The summary checks the data against its framing, and the "
                        "last cell writes the results to its `eda` section.")

    kernel = find_kernel()
    if kernel is None and register:
        kernel = register_kernel()
    kernel_name, env_label = kernel if kernel else ("python3", _env_label())
    cells = [
        _md(f"""
# Exploratory Data Analysis{": Discovery" if mode == "discovery" else ""}: {name}

{chr(10).join(settings)}

The first rows of the data are shown once, right after loading. The report sections below show summary
statistics only, and values of identifier, free text, and personal data columns are never shown in them.
Every section can be rerun on its own after you change the data.
"""),
        _md("## Setup"),
        _code(f"""
import importlib.util
import os
import sys

# This notebook was built for: {env_label} ({os.path.abspath(sys.executable)})
# If a package is missing below, the notebook is running on a different Python. In VS Code or Jupyter,
# choose "Select Kernel" and pick {env_label}, then restart the kernel and run all cells.
print("Kernel Python:", sys.executable)
missing = [m for m in ("pandas", "numpy", "matplotlib", "seaborn") if importlib.util.find_spec(m) is None]
if missing:
    raise ModuleNotFoundError(
        f"This kernel ({{sys.executable}}) is missing: {{', '.join(missing)}}. "
        "Select the kernel for {env_label} ({os.path.abspath(sys.executable)}), or install them into this kernel with "
        f"%pip install {{' '.join(missing)}}"
    )

import pandas as pd

# The EDA engine lives in the ml-eda-report skill folder.
sys.path.insert(0, r"{HERE}")
from eda_core import EDAReport
""" + (f"""
# Shared project spec. The EDA reads the framing from it and writes its results back in the last cell.
from ml_common import load as load_spec
SPEC_PATH = r"{spec_path}"
spec = load_spec(SPEC_PATH) if os.path.exists(SPEC_PATH) else None
""" if spec_path else "")),
        _code(_loader("df", data_path, sheet) + '\nprint(f"Loaded {len(df):,} rows and {df.shape[1]:,} columns")'),
        _code("df.head()"),
        _md("## Full report\n\nThe next cell runs every check once. Each section below then displays its own results."),
        _code(f"""
eda = EDAReport(
    df,
    target={target!r},
    target_type={target_type!r},
    corr_threshold={corr_threshold},
    cramers_threshold={cramers_threshold},
    date_col={date_col!r},
    id_cols={list(id_cols) if id_cols else None!r},
    name={name!r},
    sample_rows={sample_rows},
    max_plots={max_plots},
    log_scale={bool(log_scale)},
    mode={(mode or ("targeted" if target else "discovery"))!r},{chr(10) + "    spec=spec," if spec_path else ""}
).run()
print(f"{{len(eda.issues)}} issues flagged and {{eda.figure_count}} plots created")
"""),
    ]
    for key, title, intro in SECTION_ORDER:
        cells.append(_md(f"## {title}\n\n{intro}"))
        if key == "relationships":
            cells.append(_code(
                "# Change the thresholds to list more or fewer pairs. The heatmaps are not recomputed.\n"
                f'eda.show("relationships", header=False, corr_threshold={corr_threshold}, cramers_threshold={cramers_threshold})'))
        elif key == "target":
            cells.append(_code(
                "# top_n: number of features to list and plot. min_association: smallest absolute association with the target.\n"
                "# min_mutual_info: smallest mutual information. All three can be None. Nothing is recomputed.\n"
                'eda.show("target", header=False, top_n=None, min_association=None, min_mutual_info=None)'))
        elif key == "distributions":
            cells.append(_code(
                "# top_n: how many numeric and how many categorical columns to plot. None plots up to max_plots columns.\n"
                "# With a target, the top columns are the ones most associated with it, so try top_n=10.\n"
                "# log_scale: True plots strongly skewed numeric columns on a log axis, False on their original scale.\n"
                f'eda.show("distributions", header=False, top_n=None, log_scale={bool(log_scale)})'))
        else:
            cells.append(_code(f'eda.show("{key}", header=False)'))
    if spec_path:
        cells.append(_md("## Update the shared spec\n\nWrites the results above to the `eda` section of the spec. "
                         "Rerun this cell after changing the data or the settings."))
        cells.append(_code(f"""
path, created = eda.write_spec(SPEC_PATH, source=r"{data_path}", report=r"{nb_path}")
print(("Created" if created else "Updated"), path)
"""))
    cells.append(_md("## Save a markdown copy (optional)\n\nUncomment to also write the report as markdown with PNG plots."))
    cells.append(_code('# eda.to_markdown("docs")  # writes docs/eda_<name>.md and docs/eda_plots/<name>/'))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": env_label, "language": "python", "name": kernel_name},
            # Older VS Code versions pick the interpreter from this hash of its path.
            "vscode": {"interpreter": {"hash": hashlib.sha256(os.path.abspath(sys.executable).encode()).hexdigest()}},
            "language_info": {"name": "python", "version": platform.python_version()},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    # The same metadata the other ML project skills write, so an edited notebook is not replaced without asking.
    import nb_common
    nb["metadata"][nb_common.META_KEY] = {"skill": "ml-eda-report", "stage": "eda",
                                          "source_hash": nb_common.source_hash(nb)}

    executed, err = False, None
    if execute:
        try:
            import nbformat
            from nbclient import NotebookClient

            node = nbformat.from_dict(nb)
            with tempfile.TemporaryDirectory() as kdir:
                NotebookClient(node, timeout=timeout, km=_kernel_manager(kdir),
                               resources={"metadata": {"path": os.path.dirname(nb_path)}}).execute()
            nb = node
            executed = True
        except ImportError as e:
            err = f"{e.name} is not installed, so the notebook was saved without outputs"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e).strip().splitlines()[-1] if str(e).strip() else ''}"
    with open(nb_path, "w", encoding="utf-8") as f:
        if executed:
            import nbformat
            nbformat.write(nb, f)
        else:
            json.dump(nb, f, indent=1)
    return nb_path, executed, err, kernel
