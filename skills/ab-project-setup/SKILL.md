---
name: ab-project-setup
description: Set up a project folder for an A/B test, the first step of the A/B testing skills. Uses uv to create a .venv with pandas, scipy, and statsmodels in pyproject.toml and uv.lock, the standard folders (data/raw, data/processed, notebooks, src, models, docs), the shared ab_spec.yaml that the later A/B skills read and write, CLAUDE.md with the A/B testing workflow, README.md, and .gitignore, and registers the .venv as a named Jupyter kernel with VS Code settings. Works in an empty folder or in one that ml-project-setup already set up, where it only adds what is missing. Use when asked to set up, start, or initialise an A/B test, experiment, or campaign test analysis.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# A/B Test Project Setup

Create a project folder with its own Python environment and the shared A/B test spec, and make sure VS Code and Jupyter select that environment every time.

This is the first step of the A/B testing skills: **ab-project-setup**, ab-test-framing, ab-test-design, ab-data-validation, ab-analysis, ab-segment-analysis, and ab-readout. Every step reads `ab_spec.yaml` first and writes only its own section. `/ab-status` shows where a test stands. The shared conventions are in `AB_CONVENTIONS.md` in the skills folder.

## Ground rules

1. **Never delete or overwrite.** The script only creates what is missing. Existing files are kept, and an existing `.venv` is reused. A broken `.venv` is only replaced after the user agrees, and even then it is moved aside, not deleted.
2. **Use the script.** All setup goes through `scripts/setup_ab.py`. It runs the setup of the **ml-project-setup** skill with the A/B testing spec and documents, so both workflows lay out a project the same way. ml-project-setup must be installed next to this skill.
3. **Ask before risky fixes.** Changing `.vscode/settings.json`, rebuilding `.venv`, or appending to an existing `CLAUDE.md` needs the user's agreement first.
4. **Standard practice first.** When you recommend or offer options for A/B test design or analysis, start from the industry standard pattern, then adapt it to this project's requirements. State both parts explicitly in the question or reply: the standard pattern, and whether the recommendation follows it or departs from it, why, and what a departure gives up. If there is no clear standard, or practice varies by scale or platform, say so instead of presenting one approach as the standard. Check platform specific claims against current docs.
5. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Which Python runs the script

`setup_ab.py` uses only the standard library, so run it with `python3` (the project may not have a `.venv` yet). After setup, every A/B testing skill runs its scripts with the project's own `.venv/bin/python` from the project root.

## uv

The script uses [uv](https://docs.astral.sh/uv/) to create `.venv` and manage packages. If uv is missing, the script stops before changing anything. Tell the user to install it with `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`, and do not install it yourself.

## What the script creates

| Item | Details |
| --- | --- |
| `.venv/`, `pyproject.toml`, `uv.lock` | Python 3.11 by default. The base packages of ml-project-setup (pandas, numpy, scipy, matplotlib, seaborn, pyarrow, ipykernel, pyyaml, nbformat, nbclient, jinja2) plus statsmodels. Extra groups with `--extras` as in ml-project-setup. |
| Folders | `data/raw/` (read only input), `data/processed/`, `notebooks/`, `src/`, `models/`, `docs/`. |
| `notebooks/00_scratch.ipynb` | Starter notebook bound to the project kernel. Not added when one exists. |
| `ab_spec.yaml` | The shared spec, with the project name, description, Python version, kernel, and folder paths filled in. In an existing spec, only empty fields are filled. |
| `CLAUDE.md`, `README.md`, `.gitignore` | With the A/B testing workflow. An existing `CLAUDE.md` is not changed; see Step 4. |
| MLflow tracking | `mlflow` is added, and the test gets its own experiment, `ab/<project>`, on the shared MLflow server (see ml-project-setup, "MLflow tracking", for the one time server setup). The address and the link go to the `tracking` section of `ab_spec.yaml`. Every step's notebook then logs a run there. Skip with `--no-tracking`; add it to an existing project with `--add-tracking` (ask first: it changes `pyproject.toml` and CLAUDE.md). |
| `.vscode/settings.json`, Jupyter kernel | As in ml-project-setup. The kernel is named after the project, so an ML and an A/B workflow in one folder share it. |

**A folder that ml-project-setup already set up** keeps everything it has. The script adds `ab_spec.yaml`, installs statsmodels, and takes the project name, Python, and paths from `ml_spec.yaml`. The A/B notebooks start with `ab` (`ab01_test_framing.ipynb`), so they sit apart from the ML notebooks.

## Step 1: Work out the target folder

* If the user names a path or a project name, use it. A bare name means a new folder inside the current working directory.
* If the user says "this project", find the root with `python3 ${CLAUDE_SKILL_DIR}/scripts/ab_common.py root`. It looks for the nearest folder with `ab_spec.yaml`, then `ml_spec.yaml`, then `.venv`, then `.git`.
* If the folder exists, run the status check first and tell the user in one or two sentences what is there and what is missing:

  ```bash
  python3 ${CLAUDE_SKILL_DIR}/scripts/setup_ab.py <path> --status
  ```

## Step 2: Ask the setup question

Ask with one AskUserQuestion call, leaving out anything the request or an existing spec answers:

1. **Description.** "What is this test about, in one sentence?" It goes into the spec, CLAUDE.md, and README.md, and ab-test-framing uses it as the first draft of the hypothesis. Offer two or three guesses from the folder name, the data file names, and the conversation.

Extra package groups are rarely needed for A/B tests. Offer them only if the user mentions modelling.

If AskUserQuestion is unavailable or the user is not present, use the folder name as the description, and say so at the top of the reply.

## Step 3: Preview, then run

Run a dry run first and check that it creates only what you expect:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/setup_ab.py <path> --description "<sentence>" --dry-run
```

Then run it for real, with the same options and a timeout of at least 10 minutes:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/setup_ab.py <path> --description "<sentence>"
```

## Step 4: Handle problems

The script exits with code 1 and a "Problems" list when something needs the user. Never fix these silently:

* **`CLAUDE.md` has no A/B testing workflow section.** Usual in a folder set up by ml-project-setup. Ask the user, then rerun with `--update-claude-md`, which appends the section and changes nothing else.
* **`.vscode/settings.json` points at another Python.** Ask, then rerun with `--fix-vscode`.
* **`.vscode/settings.json` is not valid JSON.** Show the keys the script printed, and offer to add them with the Edit tool.
* **`.venv` is broken.** Ask, then rerun with `--rebuild-venv`. The old one is moved to `.venv.old-<time>`.
* **uv is not installed** or **installing failed.** Show the message and the install command.
* **MLflow tracking not set up yet.** The shared server was not running. This is not an error. Tell the user how to start it (`python3 ~/Projects/claude-config/skills/tools/mlflow_server.py install-agent`, after `uv tool install mlflow`), then rerun with `--add-tracking`.
* **ml-project-setup was not found.** Tell the user this skill needs it installed next to it in `~/.claude/skills`.

## Step 5: Check the spec

```bash
.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/ab_common.py status
```

## Step 6: Report back

Reply with these parts, in this order:

1. **Files written:** the project path, the Python version, the dependency files, and the kernel name, then what was created and what already existed and was kept, in a short list.
2. **Summary:** one or two sentences on the state of the project.
3. **Problems:** anything from Step 4 that is still open, one sentence each, or say there are none.
4. **Open questions:** none are expected at this stage.
5. **Next step:** put the test data in `data/raw/`, open the folder on its own in VS Code, and run **ab-test-framing**. It asks only about the test and the decision, so it can run before the data is in place.
