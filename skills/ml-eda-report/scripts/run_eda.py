#!/usr/bin/env python3
"""Run the EDA report on a data file.

Examples:
    python3 run_eda.py data/train.csv
    python3 run_eda.py data/train.parquet --target churned --date-col signup_date --id-cols customer_id
    python3 run_eda.py data/train.csv --format ipynb --notebook notebooks/eda_train.ipynb
    python3 run_eda.py --check
    python3 run_eda.py data/train.csv --list-columns
    python3 run_eda.py data/train.csv --target ltv_30d --target-type continuous
    python3 run_eda.py data/train.csv --spec ml_spec.yaml     # read the framing, write the eda section

Markdown output (default): <docs>/eda_<file name>.md and <docs>/eda_plots/<file name>/*.png
Notebook output: --notebook, or <notebooks>/04_eda_<file name>.ipynb, executed so outputs are filled in.
<docs> and <notebooks> are the folders under paths in ml_spec.yaml (default docs and notebooks).
Run from the project root with the project's Python: .venv/bin/python.

Shared spec: by default the script uses ml_spec.yaml at the project root (see ml_common.py).
It takes the target, target type, date column, and ID columns from the spec when they are not given on
the command line, checks the data against the framing, and writes the results to the spec's eda section.
The spec is created at ./ml_spec.yaml when none exists. Pass --no-spec to turn this off.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REQUIRED = ["pandas", "numpy", "matplotlib", "seaborn"]
OPTIONAL = {
    "sklearn": "mutual information in the target section (package name: scikit-learn)",
    "pyarrow": "reading parquet and feather files",
    "openpyxl": "reading xlsx files",
    "nbformat": "notebook output",
    "nbclient": "running the notebook so outputs are filled in",
    "ipykernel": "running the notebook so outputs are filled in",
    "jinja2": "formatted tables in the notebook (plain tables are shown without it)",
    "yaml": "writing ml_spec.yaml as YAML instead of JSON (package name: pyyaml)",
}


def install_command(packages):
    """The command that adds packages the way this project manages them: uv add for a pyproject.toml project,
    uv pip install for a requirements.txt project, else pip."""
    import shutil
    from ml_common import find_project_root
    root = find_project_root()[0]
    pkgs = " ".join(packages)
    if root and os.path.exists(os.path.join(root, "pyproject.toml")) and shutil.which("uv"):
        return f"uv add {pkgs}   (run in {root}; updates pyproject.toml and uv.lock)"
    if root and os.path.exists(os.path.join(root, "requirements.txt")) and shutil.which("uv"):
        return f"uv pip install --python {sys.executable} {pkgs}   (then add them to requirements.txt)"
    return f"{sys.executable} -m pip install {pkgs}"


def check_env(verbose=True):
    missing_req, missing_opt = [], []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except Exception:
            missing_req.append(m)
    for m in OPTIONAL:
        try:
            importlib.import_module(m)
        except Exception:
            missing_opt.append(m)
    if verbose:
        print(f"Python: {sys.executable}")
        print("Required packages: " + ("all installed" if not missing_req else "missing " + ", ".join(missing_req)))
        if missing_opt:
            print("Optional packages not installed:")
            for m in missing_opt:
                print(f"  {m}: needed for {OPTIONAL[m]}")
        pip = [{"sklearn": "scikit-learn", "yaml": "pyyaml"}.get(m, m) for m in missing_req + missing_opt]
        if pip:
            print("Install with: " + install_command(pip))
    return missing_req, missing_opt


def column_names(path, sheet=None):
    """Column names without loading the whole file where the format allows it."""
    from eda_core import load_data
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
            return set(pq.read_schema(path).names)
        except ImportError:
            pass
    if low.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".csv.zip", ".csv.bz2", ".xlsx", ".xlsm", ".xls", ".jsonl")):
        return set(load_data(path, sheet=sheet, nrows=50).columns)
    return set(load_data(path, sheet=sheet).columns)


def resolve_spec(a):
    """Return (spec_path, spec_dict or None). spec_path is None when --no-spec is set."""
    from ml_common import spec_path, load
    if a.no_spec:
        return None, None
    path = spec_path(a.spec)
    return path, (load(path) if os.path.exists(path) else None)


def default_mode(a, spec):
    """targeted when a target was given or the framing is confirmed, else discovery."""
    status = (((spec or {}).get("framing") or {}).get("status")) or "not_started"
    return "targeted" if (a.target or status == "confirmed") else "discovery"


def apply_spec_defaults(a, spec, columns):
    """Fill settings the user did not give from the spec. Returns lines describing what was taken.
    The target is taken only in targeted mode."""
    from ml_common import framing_defaults
    d = framing_defaults(spec)
    used = []
    if a.mode == "targeted" and not a.target and d["target"] and d["target"] in columns:
        a.target = d["target"]
        used.append(f"target `{a.target}`")
        if not a.target_type and d["target_type"]:
            a.target_type = d["target_type"]
            used.append(f"target type {a.target_type} (task type {d['task_type']})")
    if not a.date_col and d["date_col"] and d["date_col"] in columns:
        a.date_col = d["date_col"]
        used.append(f"date column `{a.date_col}`")
    if not a.id_cols:
        ids = [c for c in d["id_cols"] if c in columns]
        if ids:
            a.id_cols = ",".join(ids)
            used.append("ID columns " + ", ".join(f"`{c}`" for c in ids))
    return used, d


def print_spec_info(path, spec, columns):
    """Show what the spec already settles, so the setup questions can skip it."""
    from ml_common import framing_defaults
    print()
    if path is None:
        print("Spec: turned off with --no-spec")
        return
    if spec is None:
        print(f"Spec: none yet. One will be created at {path} after the run.")
        return
    d = framing_defaults(spec)
    print(f"Spec: {path} (framing status: {d['status']})")
    def show(label, val, check=True):
        if not val:
            print(f"  {label}: not set")
        elif check and isinstance(val, str):
            print(f"  {label}: {val}" + ("" if val in columns else "  (NOT in the data)"))
        else:
            print(f"  {label}: {val}")
    show("Target", d["target"])
    show("Target type", d["target_type"], check=False)
    show("Task type", d["task_type"], check=False)
    show("Date column", d["date_col"])
    ids = d["id_cols"]
    print("  ID columns: " + (", ".join(c + ("" if c in columns else " (NOT in the data)") for c in ids) if ids else "not set"))
    if d["excluded_cols"]:
        print("  Excluded columns: " + ", ".join(d["excluded_cols"]))


def list_columns(path, sheet=None, named_target=None):
    """Column names, types, and counts only, plus ranked target candidates. Never prints values."""
    import pandas as pd
    from eda_core import load_data, match_column, suggest_targets
    df = load_data(path, sheet=sheet)
    print(f"{len(df):,} rows and {df.shape[1]:,} columns")
    print()
    print(f"{'column':40} {'dtype':12} {'unique':>10} {'missing %':>9}")
    for c in df.columns:
        y = df[c]
        try:
            nun = f"{y.nunique():,}"
        except TypeError:
            nun = "n/a"
        print(f"{str(c)[:40]:40} {str(y.dtype)[:12]:12} {nun:>10} {y.isna().mean() * 100:>9.1f}")
    print()
    if named_target:
        exact, close = match_column(df, named_target)
        if exact:
            print(f"Named target '{named_target}': found in the data.")
        elif close:
            print(f"Named target '{named_target}': not found. Closest column names: " + ", ".join(close))
        else:
            print(f"Named target '{named_target}': not found, and no similar column names.")
        print()
    print("Likely target columns, best first (inferred from names and data properties):")
    for i, r in enumerate([r for r in suggest_targets(df, top=5) if r["score"] > 0], 1):
        print(f"  {i}. {r['column']}  score {r['score']:g}  suggested type {r['suggested_type']}  "
              f"({r['dtype']}, {r['unique']:,} unique, {r['missing_pct']:.1f}% missing)  because: {', '.join(r['reasons'])}")
    return list(df.columns)


NOTEBOOK_DIRS = ("notebooks", "notebook", "nbs", "experimentation", "experiments", "experiment", "analysis",
                 "exploration", "eda")
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".ipynb_checkpoints", "site-packages"}


def suggest_locations(data_path, spec=None, mode="targeted"):
    """Print candidate save locations for the markdown report and the notebook, recommended first.

    In a project set up with ml-project-setup, the recommended places are the spec's notebooks and docs folders,
    with the standard names 04_eda_<dataset>.ipynb and eda_<dataset>.md. Other notebook or experimentation
    folders, and the folder next to the data, follow.
    """
    from ml_common import find_project_root, project_paths, notebook_name, doc_name
    root, why = find_project_root()
    root = root or os.path.abspath(".")
    data_dir = os.path.dirname(os.path.abspath(data_path))
    stem = os.path.basename(data_path).split(".")[0]
    paths = project_paths(spec or {}, root)
    suffix = stem if mode == "targeted" else f"discovery_{stem}"
    nb_file, md_file = notebook_name("eda", suffix), doc_name("eda", suffix)
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        if os.path.relpath(dirpath, root).count(os.sep) >= 2:
            dirnames[:] = []
        for d in dirnames:
            if d.lower() in NOTEBOOK_DIRS:
                found.append(os.path.join(dirpath, d))
    found.sort(key=lambda d: (d.count(os.sep), d))
    rel = lambda p: os.path.relpath(p, root) if p.startswith(root) else p
    print(f"Project root: {root} ({why})")
    print(f"Data folder: {rel(data_dir)}")
    print("Notebook or experimentation folders found: " + (", ".join(rel(d) for d in found) if found else "none"))
    print()
    nb_dirs = [paths["notebooks"]] + [d for d in found if d != paths["notebooks"]] + [data_dir]
    md_dirs = [paths["docs"]] + [d for d in found if d != paths["docs"]] + [data_dir]
    for label, dirs, fname, std in (("Notebook (.ipynb)", nb_dirs, nb_file, paths["notebooks"]),
                                    ("Markdown (.md)", md_dirs, md_file, paths["docs"])):
        print(f"{label} locations, recommended first:")
        seen = []
        for d in dirs:
            p = os.path.join(d, fname)
            if p not in seen:
                seen.append(p)
        for i, p in enumerate(seen[:4], 1):
            notes = []
            if os.path.dirname(p) == std:
                notes.append("standard project folder" + ("" if os.path.isdir(std) else ", will be created"))
            if os.path.exists(p):
                notes.append("file exists, will be overwritten")
            print(f"  {i}. {rel(p)}" + (f" ({'; '.join(notes)})" if notes else ""))
        print()


def pd_counts(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", help="Path to a csv, tsv, parquet, xlsx, json, jsonl, or feather file")
    ap.add_argument("--target", help="Target column name")
    ap.add_argument("--target-type", choices=["continuous", "categorical"],
                    help="Treat the target as continuous (regression) or categorical (classification)")
    ap.add_argument("--date-col", help="Main date column name")
    ap.add_argument("--corr-threshold", type=float, default=0.9,
                    help="List numeric pairs with |Spearman| at or above this value (default 0.9)")
    ap.add_argument("--cramers-threshold", type=float, default=0.8,
                    help="List categorical pairs with Cramér's V at or above this value (default 0.8)")
    ap.add_argument("--id-cols", help="Comma separated ID column names")
    ap.add_argument("--sheet", help="Excel sheet name")
    ap.add_argument("--name", help="Dataset name used in titles (default: file name)")
    ap.add_argument("--format", choices=["md", "ipynb"],
                    help="ipynb (default inside an ML project, where the notebook is the record) or md")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited after the skill wrote it")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead of replacing one")
    ap.add_argument("--out", help="Output folder for the markdown report (default: the spec's docs folder)")
    ap.add_argument("--notebook", help="Notebook path for --format ipynb")
    ap.add_argument("--sample-rows", type=int, default=200_000)
    ap.add_argument("--max-plots", type=int, default=24)
    ap.add_argument("--log-scale", action="store_true",
                    help="Plot strongly skewed numeric columns on a log axis (default: original scale)")
    ap.add_argument("--no-execute", action="store_true", help="Write the notebook without running it")
    ap.add_argument("--check", action="store_true", help="Check installed packages and exit")
    ap.add_argument("--kernel-status", action="store_true",
                    help="Print whether this interpreter is registered as a Jupyter kernel, then exit")
    ap.add_argument("--register-kernel", action="store_true",
                    help="Register this interpreter as a Jupyter kernel for the user if it is not registered yet")
    ap.add_argument("--suggest-locations", action="store_true",
                    help="Print suggested save locations for the report and notebook, then exit")
    ap.add_argument("--report", help="Full path of the markdown report, such as notebooks/eda_train.md. "
                                     "Plots go in an eda_plots folder next to it. Overrides --out.")
    ap.add_argument("--list-columns", action="store_true",
                    help="Print column names, dtypes, unique counts, and ranked target candidates, then exit. "
                         "With --target, also check that the named column exists. No values are printed.")
    ap.add_argument("--spec", help="Path of the shared ml_spec.yaml (default: found from the current folder, "
                                   "or created as ./ml_spec.yaml)")
    ap.add_argument("--no-spec", action="store_true", help="Do not read or write the shared spec")
    ap.add_argument("--mode", choices=["discovery", "targeted"],
                    help="discovery: no target, profile the data and write hints for the framing. targeted: analyse the "
                         "target. Default: targeted when a target is given or the spec's framing is confirmed, "
                         "else discovery")
    a = ap.parse_args()

    if a.kernel_status:
        from make_notebook import kernel_status
        kernel_status()
        return
    if a.check:
        req, _ = check_env()
        sys.exit(1 if req else 0)
    if not a.data:
        ap.error("a data file path is required")
    req, _ = check_env(verbose=False)
    if req:
        check_env()
        sys.exit(1)
    if a.target_type and not a.target:
        ap.error("--target-type needs --target")
    if not os.path.exists(a.data):
        sys.exit(f"File not found: {a.data}")

    spec_path, spec = resolve_spec(a)
    a.mode = a.mode or default_mode(a, spec)
    if a.mode == "discovery" and a.target:
        ap.error("discovery mode runs without a target. Drop --target, or pass --mode targeted.")
    if a.suggest_locations:
        suggest_locations(a.data, spec, a.mode)
        return
    if a.list_columns:
        cols = list_columns(a.data, a.sheet, named_target=a.target)
        print_spec_info(spec_path, spec, set(cols))
        print(f"Mode: {a.mode}" + (" (no target; the framing is not confirmed yet)" if a.mode == "discovery" else ""))
        return

    spec_used = []
    if spec is not None:
        cols = column_names(a.data, a.sheet)
        spec_used, _ = apply_spec_defaults(a, spec, cols)
        if spec_used:
            print("From the spec: " + "; ".join(spec_used) + ".")
    if a.mode == "targeted" and not a.target:
        ap.error("targeted mode needs a target. Pass --target, confirm the framing with ml-problem-framing, or use "
                 "--mode discovery.")
    print(f"Mode: {a.mode}")

    stem = os.path.basename(a.data).split(".")[0]
    name = a.name or stem
    suffix = stem if a.mode == "targeted" else f"discovery_{stem}"
    # Discovery and targeted reports keep separate plot folders, so one run does not remove the other's plots.
    from ml_common import file_slug
    plot_dir_name = f"eda_plots/{file_slug(suffix)}"
    id_cols = [c.strip() for c in a.id_cols.split(",")] if a.id_cols else None

    from ml_common import find_project_root, project_paths, notebook_name, doc_name
    proj_paths = project_paths(spec or {}, find_project_root()[0] or os.path.abspath("."))
    if a.format is None:
        a.format = "ipynb" if spec_path else "md"
    if a.format == "ipynb":
        from make_notebook import build_notebook
        import nb_common
        nb_path = a.notebook or os.path.join(proj_paths["notebooks"], notebook_name("eda", suffix))
        st = nb_common.state(nb_path)
        if st == "edited" and not (a.overwrite or a.new_version):
            print(f"Not written: {nb_path} was edited after the skill wrote it. Rerun with --overwrite to replace it, "
                  f"or --new-version to write {os.path.basename(nb_common.next_version(nb_path))}.")
            sys.exit(2)
        if a.new_version and st != "absent":
            nb_path = nb_common.next_version(nb_path)
        path, executed, err, kernel = build_notebook(
            nb_path, data_path=a.data, mode=a.mode, target=a.target, target_type=a.target_type, date_col=a.date_col, id_cols=id_cols,
            corr_threshold=a.corr_threshold, cramers_threshold=a.cramers_threshold, sheet=a.sheet, name=name,
            sample_rows=a.sample_rows, max_plots=a.max_plots, log_scale=a.log_scale, execute=not a.no_execute, register=a.register_kernel,
            spec_path=os.path.abspath(spec_path) if spec_path else None,
            project_name=((spec or {}).get("project") or {}).get("name"))
        print(f"Wrote notebook: {path}")
        print("Executed: yes" if executed else f"Executed: no ({err})")
        if spec_path:
            print(f"Spec: {spec_path} " + ("was updated by the notebook's last cell." if executed else
                                           "will be updated when the notebook's last cell is run."))
        if kernel:
            print(f"Kernel: '{kernel[1]}' (registered name '{kernel[0]}'). Editors that read the notebook's kernel name select it automatically.")
        else:
            print("Kernel: this environment is not registered as a Jupyter kernel, so the editor may pick another "
                  "Python. Rerun with --register-kernel, or select the kernel by hand.")
        return

    import matplotlib
    matplotlib.use("Agg")
    from eda_core import EDAReport, load_data

    print(f"Loading {a.data} ...")
    df = load_data(a.data, sheet=a.sheet)
    print(f"Loaded {len(df):,} rows and {df.shape[1]:,} columns. Running checks ...")
    eda = EDAReport(df, target=a.target, target_type=a.target_type, corr_threshold=a.corr_threshold,
                    cramers_threshold=a.cramers_threshold, date_col=a.date_col, id_cols=id_cols, name=name, sample_rows=a.sample_rows,
                    max_plots=a.max_plots, log_scale=a.log_scale, spec=spec, mode=a.mode).run()
    from eda_core import slug
    if a.report:
        out_dir = os.path.dirname(a.report) or "."
        path = eda.to_markdown(out_dir, report_name=os.path.basename(a.report), plot_dir_name=plot_dir_name,
                               source=a.data)
    else:
        out_dir = a.out or proj_paths["docs"]
        path = eda.to_markdown(out_dir, report_name=doc_name("eda", suffix), plot_dir_name=plot_dir_name,
                               source=a.data)
    plot_dir = os.path.join(out_dir, plot_dir_name)
    sev = {}
    for g in eda.issue_groups:
        sev[g["severity"]] = sev.get(g["severity"], 0) + 1
    print(f"Wrote report: {path}")
    if spec_path:
        sp, created = eda.write_spec(spec_path, source=a.data, report=path)
        print(f"Spec: {'created' if created else 'updated'} {sp}")
        if eda.mode == "discovery":
            h = eda.framing_hints()
            print("Framing hints: " + ", ".join([
                f"{len(h['candidate_targets'])} candidate targets",
                "grain " + (" + ".join(h["grain"]["columns"]) if h["grain"]["columns"] else "unclear"),
                f"{len(h['entity_id_candidates'])} possible entity IDs",
                f"history {h['history_days']:,} days" if h["history_days"] is not None else "no date column",
                f"{len(h['post_outcome_suspects'])} columns possibly recorded after the outcome"]))
        if eda.framing_checks:
            st = pd_counts([c["status"] for c in eda.framing_checks])
            print("Framing checks: " + ", ".join(f"{v} {k}" for k, v in st.items()))
            for c in eda.framing_checks:
                if c["status"] in ("fail", "warn", "skipped"):
                    print(f"  [{c['status']}] {c['check']}: {c['detail']}")
    print(f"Plots: {eda.figure_count} in {plot_dir}")
    print(f"Issue types: {sev.get('High', 0)} High, {sev.get('Medium', 0)} Medium, {sev.get('Low', 0)} Low")
    skipped = [s.title for s in eda.sections.values() if s.skipped]
    if skipped:
        print("Sections skipped: " + "; ".join(skipped))
    for g in eda.issue_groups[:12]:
        unit = "check" if g["area"] == "Framing" else "column"
        cols = (f" ({g['count']} {unit}" + ("s" if g["count"] > 1 else "") + ")") if g["count"] else ""
        print(f"  [{g['severity']}] {g['finding']}{cols}")


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "eda", "ml-eda-report")
