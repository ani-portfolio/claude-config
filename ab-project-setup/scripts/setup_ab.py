#!/usr/bin/env python3
"""Set up a project folder for the A/B testing skills, the first stage of that workflow.

Runs the setup of the ML project skills (ml-project-setup/scripts/setup_project.py) with the A/B testing versions of
the spec, the CLAUDE.md workflow section, the README, and the starter notebook. Everything else is the same: uv,
.venv, pyproject.toml and uv.lock, the folders, the Jupyter kernel, and the VS Code settings. It only creates what
is missing and never overwrites, so it can run in a folder that ml-project-setup already set up: that adds
ab_spec.yaml next to ml_spec.yaml, and both workflows share the folders, .venv, and kernel.

Creates (only what is missing):
  .venv/, pyproject.toml, uv.lock    the base packages of the ML setup plus statsmodels
  data/raw/ data/processed/ notebooks/ src/ models/ docs/
  notebooks/00_scratch.ipynb         starter notebook bound to the project kernel
  ab_spec.yaml                       shared spec read and written by every A/B testing skill
  CLAUDE.md README.md .gitignore .vscode/settings.json

Takes the same options as setup_project.py (--description, --extras, --python, --dry-run, --status,
--update-claude-md, --fix-vscode, --rebuild-venv, --no-install, --no-notebook, --to-pyproject). Needs uv.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common  # noqa: E402

SKILL = "ab-project-setup"


def find_setup_project():
    """The ML setup script, which lives in the ml-project-setup skill next to this one."""
    here_real = os.path.realpath(HERE)
    for base in (os.path.dirname(os.path.dirname(here_real)), os.path.expanduser("~/.claude/skills")):
        p = os.path.join(base, "ml-project-setup", "scripts", "setup_project.py")
        if os.path.isfile(p):
            return p
    sys.exit("ERROR: ab-project-setup uses the setup script of the ml-project-setup skill, which was not found next "
             "to this skill or in ~/.claude/skills. Install ml-project-setup, then rerun.")


SP_PATH = find_setup_project()
sys.path.insert(1, os.path.dirname(SP_PATH))
import setup_project as sp  # noqa: E402

AB_HEADING = "## A/B testing workflow"
AB_SECTION = """## A/B testing workflow
This project uses the A/B testing skills. `ab_spec.yaml` at the project root is the shared record of the test:
read it before analysing anything, and never contradict it without asking the user.
* Steps, in order: ab-project-setup, ab-test-framing, ab-test-design (optional for a finished test),
  ab-data-validation, ab-analysis, ab-segment-analysis (optional), ab-readout.
* Each step writes one numbered notebook in `notebooks/`: `ab01_test_framing.ipynb`, `ab02_design.ipynb`,
  `ab03_validation.ipynb`, `ab04_analysis.ipynb`, `ab05_segments.ipynb`, `ab06_readout.ipynb`.
* Cells marked *Decision* hold the answers. To change a decision, edit that cell and run the whole notebook, which
  writes the answers to `ab_spec.yaml`. Do not edit the spec sections those notebooks own by hand.
* The primary metric, the planned split, and the ship rule are fixed in the framing before results are seen. A
  change after seeing results is recorded in `decisions` with the reason.
* No analysis before the framing is confirmed and the data validation has run without a failure, such as a sample
  ratio mismatch.
* Only columns recorded before assignment (`data.pre_treatment_cols`) are used for balance checks, CUPED, or
  segments.
* See progress and the next skill to run with `/ab-status`.
* Each step's run is logged to MLflow, in this test's experiment on the shared server (`tracking` in
  `ab_spec.yaml` has the address and the link). The spec decides; MLflow records. Start the server with
  `python3 ~/Projects/claude-skills/tools/mlflow_server.py start` when it is not running.
"""

# ---------------------------------------------------------------- the A/B versions of the setup templates

sp.__doc__ = __doc__
sp.WORKFLOW_HEADING = AB_HEADING
sp.WORKFLOW_SECTION = AB_SECTION
sp.SPEC_FILE = ab_common.SPEC_NAME
sp.BASE_PACKAGES = sp.BASE_PACKAGES + ["statsmodels"]

_claude_md, _readme_md, _starter = sp.claude_md, sp.readme_md, sp.starter_notebook


def _swap(text, pairs):
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def claude_md(*args, **kw):
    return _swap(_claude_md(*args, **kw), [
        ("* `ml_spec.yaml`: the shared project spec. Committed to git.",
         "* `ab_spec.yaml`: the shared spec of the A/B test. Committed to git."),
        ("the business framing summary (`business_framing.md`), the data dictionary (`data_dictionary.yaml`), and "
         "later the model card.", "the test plan (`test_plan.md`) and the readout for stakeholders (`ab_readout.md`)."),
        ("listed under `paths` in `ml_spec.yaml`", "listed under `paths` in `ab_spec.yaml`"),
    ])


def readme_md(*args, **kw):
    text = _swap(_readme_md(*args, **kw), [
        ("ml_spec.yaml       shared project spec: framing, data findings, and decisions",
         "ab_spec.yaml       shared test spec: framing, design, validation, results, and decisions"),
        ("business framing summary, data dictionary, model card", "test plan and readout"),
    ])
    start, end = text.find("## Workflow\n"), text.find("\n## Data")
    if start != -1 and end != -1:
        text = text[:start] + (
            "## Workflow\n\n"
            "The project follows the A/B testing skills: ab-project-setup, ab-test-framing, ab-test-design,\n"
            "ab-data-validation, ab-analysis, ab-segment-analysis, and ab-readout. Each step writes one numbered\n"
            "notebook (`ab01_test_framing.ipynb` to `ab06_readout.ipynb`) that holds its reasoning, decisions, and\n"
            "checks. Running a notebook records its decisions in `ab_spec.yaml`. Run `/ab-status` in Claude Code to\n"
            "see progress.\n") + text[end:]
    return text


def starter_notebook(*args, **kw):
    return _swap(_starter(*args, **kw), [("ml_spec.yaml", "ab_spec.yaml"),
                                         ("made by the ML project skills", "made by the A/B testing skills")])


sp.claude_md, sp.readme_md, sp.starter_notebook = claude_md, readme_md, starter_notebook


def _read_spec(path, root):
    try:
        return ab_common.load(str(path))
    except SystemExit:  # this Python has no PyYAML: read it with the project's .venv Python
        vpy = sp.venv_python(root)
        if not vpy.exists():
            return {}
        r = sp.run([str(vpy), "-c", "import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1])) or {}, "
                    "default=str))", str(path)])
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}


def spec_value(root, key):
    """A top level section of ab_spec.yaml, else of ml_spec.yaml (a folder that ml-project-setup set up), else None.
    So an A/B test in an ML project takes the same name, Python, and paths."""
    for name in (ab_common.SPEC_NAME, *ab_common.OTHER_SPECS):
        f = root / name
        if f.exists():
            v = (_read_spec(f, root) or {}).get(key)
            if v:
                return v
    return None


def step_spec(root, name, description, pyver, kernel, display, paths, rep):
    """Create ab_spec.yaml, or fill empty project fields and paths in an existing one. Never overwrites values."""
    spec_file = root / ab_common.SPEC_NAME
    existed = spec_file.exists()
    label = ("fill empty project fields and paths in " if existed else "create ") + ab_common.SPEC_NAME
    if rep.dry_run:
        rep.did(label)
        return
    has_yaml = lambda exe: sp.run([exe, "-c", "import yaml"]).returncode == 0
    py = str(sp.venv_python(root))
    if not has_yaml(py):
        py = sys.executable
    as_json = not has_yaml(py)
    script = os.path.join(HERE, "ab_common.py")
    if not existed:
        r = sp.run([py, script, "init", "--spec", str(spec_file), "--name", name, "--skill", SKILL]
                   + (["--description", description] if description else []))
        if r.returncode != 0:
            rep.problem(f"could not create {ab_common.SPEC_NAME}:\n" + (r.stderr or r.stdout)[-1500:])
            return
    values = {"project": {"name": name, "description": description, "python": pyver, "kernel": kernel,
                          "kernel_display": display}, "paths": paths}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(values, f)
        tmp = f.name
    try:
        r = sp.run([py, script, "fill" if existed else "merge", "root", tmp, "--spec", str(spec_file), "--skill", SKILL])
    finally:
        os.unlink(tmp)
    if r.returncode != 0:
        rep.problem(f"could not update {ab_common.SPEC_NAME}:\n" + (r.stderr or r.stdout)[-1500:])
        return
    rep.did(label + (" (as JSON, because .venv has no PyYAML yet)" if as_json else ""))


_status = sp.status


def status(root, name):
    code = _status(root, name)
    print(f"  {ab_common.SPEC_NAME}: {'yes' if (root / ab_common.SPEC_NAME).exists() else 'no'}")
    return code


def step_tracking(root, rep):
    """The A/B test's MLflow experiment, ab/<project>, set up with this skill's tracking.py (which reads ab_spec.yaml)."""
    if rep.dry_run:
        rep.did("set up MLflow tracking (the test's experiment on the shared server)")
        return
    r = sp.run([str(sp.venv_python(root)), os.path.join(HERE, "tracking.py"), "setup"], cwd=str(root))
    out = (r.stdout or "").strip()
    if r.returncode == 0:
        rep.did(out.splitlines()[0] if out else "set up MLflow tracking")
    else:
        rep.kept("MLflow tracking not set up yet: " + (out or (r.stderr or "")[-300:]).replace("MLflow: ", "")
                 + " Run `setup_ab.py <folder> --add-tracking` once it is up.")


sp.spec_value, sp.step_spec, sp.status, sp.step_tracking = spec_value, step_spec, status, step_tracking

if __name__ == "__main__":
    sys.exit(sp.main())
