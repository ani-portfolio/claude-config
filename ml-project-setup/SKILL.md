---
name: ml-project-setup
description: Set up a new or existing Python ML project folder, the first stage of the ML project skills. Uses uv to create a .venv and manage packages in pyproject.toml and uv.lock, creates the standard folders (data/raw, data/processed, notebooks, src, models, docs), the shared ml_spec.yaml that the later skills read and write, CLAUDE.md with the ML workflow, README.md, and .gitignore, sets up the project's MLflow experiment on the shared tracking server, traces Claude Code sessions into it, makes the folder a git repository with work on the branch ml/v1 (never main), and registers the .venv as a named Jupyter kernel with VS Code settings so editors always pick this project's Python. Also repairs projects where VS Code or Jupyter picks the wrong kernel. Use when asked to set up, scaffold, start, or initialize a project, or to fix a project's environment or kernel.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Project Setup

Create a project folder with its own Python environment and the shared project spec, and make sure VS Code and Jupyter select that environment every time.

This is the first step of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, ml-eda-report, then ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, ml-offline-evaluation, and ml-system-design. Every stage reads `ml_spec.yaml` first and writes only its own section. The conventions shared by all of them are in `ML_CONVENTIONS.md` in the claude-skills folder.

## Ground rules

1. **Never delete or overwrite.** The script only creates what is missing. Existing files are kept as they are, and an existing `.venv` is reused. A broken `.venv` is only replaced after the user agrees, and even then it is moved aside, not deleted.
2. **Use the scripts.** All setup goes through `scripts/setup_project.py`, and spec changes go through `scripts/ml_common.py`, so every project gets the same layout, kernel naming, and spec as the other skills.
3. **Ask before risky fixes.** Changing `.vscode/settings.json`, rebuilding `.venv`, or appending to an existing `CLAUDE.md` needs the user's agreement first.
4. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Which Python runs the scripts

`setup_project.py` uses only the standard library, so run it with `python3` (the project may not have a `.venv` yet). After setup, every ML project skill runs its scripts with the project's own `.venv/bin/python` from the project root.

## uv

The script uses [uv](https://docs.astral.sh/uv/) to create `.venv`, record packages in `pyproject.toml` and `uv.lock`, and install them. It looks for `uv` on the PATH and in `~/.local/bin`, `~/.cargo/bin`, and the Homebrew folders. If uv is missing, the script stops before changing anything. Tell the user to install it with `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`, and do not install it yourself.

## What the script creates

| Item | Details |
| --- | --- |
| `.venv/` | Built by `uv venv` with Python 3.11 by default (`--python` picks another version or interpreter). uv downloads that Python if it is not installed. |
| `pyproject.toml` | Project name, description, and Python version, with the base packages added by `uv add`: pandas, numpy, scipy, matplotlib, seaborn, pyarrow, ipykernel, pyyaml, nbformat, nbclient, jinja2, plus any extra groups. `uv add` records each package with a lower bound such as `pandas>=3.0.6`. The `dev` extra group goes in the `dev` dependency group. |
| `uv.lock` | Exact versions of every package, written by `uv sync`. Commit it with `pyproject.toml`, so `uv sync` rebuilds the same environment on another machine. |
| `data/raw/`, `data/processed/` | Input data (read only) and data written by the skills. Contents are ignored by git. |
| `notebooks/`, `src/`, `models/`, `docs/` | Stage notebooks, reusable code (`src/__init__.py` makes it importable), fitted models (ignored by git), and documents. |
| `notebooks/00_scratch.ipynb` | Starter notebook bound to the project kernel. Its first cell finds the project root, stops with a clear message if the wrong kernel is running, and sets `DATA_RAW` and `DATA_PROCESSED` from the spec. |
| `ml_spec.yaml` | The shared spec, with the project name, description, Python version, kernel, and folder paths filled in. In an existing spec, only empty fields are filled. |
| `CLAUDE.md`, `README.md`, `.gitignore` | Filled in with the project name, description, Python version, kernel, layout, and the ML workflow. |
| `.vscode/settings.json` | Points VS Code at `.venv` and runs notebooks from the project root. Merged into an existing file. |
| Jupyter kernel | Registered under the project name (for example `martech_adtech_data`), shown as "`<name>` (.venv, Python 3.11.x)". |
| MLflow tracking | `mlflow` is added to the dependencies, and the project gets its own experiment, `ml/<project>`, on the shared MLflow server. The address, the experiment, and the UI link are written to the `tracking` section of the spec. Skip with `--no-tracking`. |
| Claude Code tracing | `mlflow autolog claude` writes `.claude/settings.json` and `.claude/settings.local.json`, so every Claude Code session in the project is traced into the project's experiment. A SessionStart hook (`tools/session_env.py` in the claude-skills folder) puts the session id in `ML_AGENT_SESSION`, and runs and commits are tagged with it. Skipped with `--no-tracking`. |
| Git repository | A new repository gets one empty commit on main ("Initialize repository"), then the branch `ml/v1` with a commit of the set up files. In an existing repository, only the set up files are committed, on `ml/v1` when main is checked out, or on the branch already checked out otherwise. The spec gets `version: {current: V1, branch: ml/v1}` and `git: {auto_commit: true}`. Skip with `--no-git`. |
| Editable install | `uv sync` installs the project itself in editable mode (hatchling build), so `from src.utils import x` works from notebooks, scripts, and tests in any folder, and new modules under `src/` work without reinstalling. |

**Existing dependency files.**

* **`pyproject.toml` exists:** it is kept. The script runs `uv add` only for base packages it does not list yet, so existing entries and version pins are never changed. If it has no `[project]` table (for example a Poetry project), the script adds nothing and reports it. If it has no build system, the project is not installed, and the script puts the project root on the `.venv` import path instead so `import src` still works.
* **Only `requirements.txt` exists:** it stays the dependency file. Missing base packages are added to it and installed with `uv pip install -r requirements.txt`. With `--to-pyproject`, the script creates `pyproject.toml` from it (`uv add -r requirements.txt`) and keeps `requirements.txt` for the user to delete after checking.
* **`uv sync --inexact`** is used, so packages that were installed into `.venv` by hand are not removed.

**Older projects.** If `data/` already holds data files directly, they stay where they are: the script does not create `data/raw/`, and records `data_raw: data` under `paths` in the spec so the other skills look there. An existing `notebooks/01_exploration.ipynb` is kept, and `00_scratch.ipynb` is not added next to it.

Extra package groups (pass with `--extras`, comma separated): `sklearn` (scikit-learn, needed from ml-train-test-split onwards), `tuning` (optuna, for ml-model-training), `boosting` (lightgbm, xgboost, catboost), `timeseries` (statsmodels), `tracking` (mlflow, always added unless `--no-tracking`), `sql` (duckdb), `dev` (pytest, ruff, in the dev dependency group). Run `python3 ${CLAUDE_SKILL_DIR}/scripts/setup_project.py --list-extras` to print them.

## Step 1: Work out the target folder

* If the user names a path or a project name, use it. A bare name means a new folder inside the current working directory, unless the user says otherwise.
* If the user says "this project" or "fix the kernel here", find the project root with `python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root`. It looks for the nearest folder with `ml_spec.yaml`, then `.venv`, then `.git`. If none is found, use the current working directory.
* If the folder already exists, run the status check first and tell the user in one or two sentences what is already there and what is missing:

  ```bash
  python3 ${CLAUDE_SKILL_DIR}/scripts/setup_project.py <path> --status
  ```

## Step 2: Ask the setup questions

Ask one question at a time, one AskUserQuestion call per question, and wait for each answer before asking the next, leaving out anything the request or an existing `ml_spec.yaml` already answers:

1. **Description.** "What is this project about, in one sentence?" This goes into the spec, CLAUDE.md, and README.md, and ml-business-framing uses it as the first draft of the business goal. Offer two or three guesses from the folder name and context, and the user can type their own.
2. **Move to pyproject.toml**, only when the status check shows the project uses `requirements.txt`. Question: "This project uses requirements.txt. Should I move it to pyproject.toml and uv.lock?" Options: "Yes, create pyproject.toml" (Recommended), described as "requirements.txt is kept until you delete it", and "No, keep requirements.txt". If yes, add `--to-pyproject`.
3. **Extra packages** (multiSelect). "Which extra packages should I install on top of the base set?" Options: the three groups that best fit the description, plus "None, base set only". Mark the best fitting group (Recommended). `sklearn` is needed by every modelling step, and `boosting` is usually right for tabular ML projects. A project that will not build models needs neither, and the modelling skills ask before adding them later.

If AskUserQuestion is unavailable or the user is not present, use the placeholder description and no extras, and say so at the top of the reply.

## MLflow tracking

Every project logs the runs of its modelling stages (split onwards, and reviews) to one shared MLflow server on this computer, at `http://127.0.0.1:5000`. The server keeps its data in `~/.mlflow/`, outside every project, so the experimentation agent and every project use the same store. The spec stays the record of decisions; MLflow records each run so it can be browsed and compared.

The server is set up once per computer, not per project. Projects get only the client, `mlflow-skinny` (pinned to the server's major version, `>=3.17,<4`), added to the `tracking` extra. The full `mlflow` package belongs to the server alone: in a project it pins packages such as `pyarrow` below their latest versions, and uv then quietly falls back to a very old MLflow that does not import. A project that already lists `mlflow` keeps it.

On macOS, the AirPlay Receiver also listens on port 5000. If `status` says the server is not running but something answers on port 5000 with `Server: AirTunes`, ask the user to turn off AirPlay Receiver (System Settings, General, AirDrop and Handoff), or to set `MLFLOW_PORT` and `MLFLOW_TRACKING_URI` to another port.

Check the server with:

```bash
python3 ~/Projects/claude-skills/tools/mlflow_server.py status
```

When it is not running, tell the user the one time setup and let them run it, because it adds a login item to their computer:

```bash
uv tool install mlflow
python3 ~/Projects/claude-skills/tools/mlflow_server.py install-agent   # macOS: starts now and at every login
```

`mlflow_server.py start` runs it until logout instead. A server that is down does not fail the setup: the script reports that tracking is not set up yet, and `--add-tracking` sets it up later.

**Older projects.** `setup_project.py <folder> --add-tracking` only adds `mlflow`, updates the ML workflow section of CLAUDE.md, and creates the experiment. Ask the user first, because it changes `pyproject.toml` and CLAUDE.md. Stage notebooks get their MLflow cell the next time their skill rebuilds them.

## Versions and git

Every project is a git repository, and every pass at the model is a version: V1 is the first pass, and each iteration opened with **ml-iterate** builds the next one (V2, V3). Each version has its own branch, `ml/v1`, `ml/v2`. **Nothing is ever committed to main**: main holds only the empty first commit, and the user merges a version into main when they want to.

After setup, every successful stage build commits the stage's files on the version branch (the spec, the stage notebook, the kept answers, `src`, and `docs`), with the message `V1 <stage>: built by <skill>`. Data, models, and `.venv` are ignored. `git.auto_commit: false` in the spec turns the commits off.

Commits use the user's git identity. When `git config user.name` or `user.email` is not set, the script commits nothing and reports a problem: give the user the two `git config --global` commands it printed, and rerun with `--add-git` after they set them.

**Older projects.** `setup_project.py <folder> --add-git` makes the folder a repository (or moves an existing one off main) and commits the set up files. `--add-tracking` also adds the Claude Code tracing and the git set up. Ask the user first.

## Step 3: Preview, then run

Run a dry run first and check that it creates only what you expect:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/setup_project.py <path> --description "<sentence>" [--extras a,b] --dry-run
```

Then run it for real, with the same options. Resolving and installing packages takes one to three minutes, so use a timeout of at least 10 minutes:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/setup_project.py <path> --description "<sentence>" [--extras a,b]
```

Other options: `--name` sets a project name that differs from the folder name, `--python 3.12` (or a path) picks the Python for a new `.venv`, `--no-install` creates files, folders, `.venv`, and the `pyproject.toml` entries without resolving or installing packages, and `--no-notebook` skips the starter notebook, `--no-tracking` leaves out MLflow and the Claude Code tracing, and `--no-git` leaves out the git repository.

## Step 4: Handle problems

The script exits with code 1 and a "Problems" list when something needs the user. Never fix these silently:

* **`.vscode/settings.json` points at another Python.** This is the usual cause of the wrong kernel being picked. Ask the user, then rerun with `--fix-vscode`.
* **`.vscode/settings.json` has comments or is not valid JSON.** Show the user the keys the script printed, and offer to add them with the Edit tool.
* **`.venv` is broken.** Ask the user, then rerun with `--rebuild-venv`. The old environment is moved to `.venv.old-<time>`, and the user can delete it once the new one works. Packages that were installed by hand and are not in `pyproject.toml` (or `requirements.txt`) will not come back, so ask the user whether any should be added first with `uv add`.
* **`CLAUDE.md` has no ML workflow section.** This happens in projects set up before the ML project skills. Ask the user, then rerun with `--update-claude-md`, which appends the section and changes nothing else.
* **`CLAUDE.md` has an older ML workflow section.** The skills changed since the project was set up, for example the notebook names. Ask the user, then rerun with `--update-claude-md`, which replaces that section only.
* **MLflow tracking not set up yet.** The shared server was not running. This is not an error. Give the user the server setup from the "MLflow tracking" section, then rerun with `--add-tracking`.
* **git has no user name or email.** Nothing was committed. Give the user the two `git config --global` commands from the problem, and rerun with `--add-git` once they are set.
* **MLflow tracking says the client is not installed, or `import mlflow` fails** (for example on `google.protobuf`). The project probably lists the full `mlflow` and uv resolved an old version. Ask the user, then replace it: `uv remove mlflow` and `uv add "mlflow-skinny>=3.17,<4"`, and rerun with `--add-tracking`.
* **The mlflow command is not installed**, so Claude Code sessions are not traced. Tracking of runs still works. Give the user `uv tool install mlflow`, then rerun with `--add-tracking`.
* **uv is not installed.** The script stopped before changing anything. Give the user the install command from the "uv" section, and rerun after they install it.
* **`pyproject.toml` has no `[project]` table.** Tell the user the base packages were not added, and offer to add a `[project]` table with the Edit tool.
* **`uv add` or installing failed.** Show the error. Common causes are no network, a version pin in `pyproject.toml` that conflicts with a base package, or a package with no build for this Python version.

## Step 5: Update the spec

The script already wrote `ml_spec.yaml`. Check it with the project's Python from the project root:

```bash
.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
```

If the script printed that the spec was written as JSON (packages were not installed), say so. It becomes YAML the next time a skill writes it with the project's Python.

## Step 6: Report back

Reply with these parts, in this order:

1. **Files written:** the project path, the git branch and the set up commit, the Python version, the dependency file (`pyproject.toml` with `uv.lock`, or `requirements.txt`), and the kernel name and display name, then what was created and what already existed and was kept, in a short list.
2. **Summary:** one or two sentences on the state of the project.
3. **Problems:** anything from Step 4 that is still open, one sentence each, or say there are none.
4. **Open questions:** none are expected at this stage. Say so only if the spec lists some.
5. **Next step:** say which branch the work is on (`ml/v1`) and that each stage build is committed there, open the folder on its own in VS Code (File, Open Folder) so it reads `.vscode/settings.json`, run the first cell of the starter notebook to confirm the kernel, put the input data in `data/raw/`, and then run the **ml-business-framing** skill. It asks only business questions, so it can run before the data is in place.
