---
name: code-map
description: Build an interactive map of a Python repo's src/ folder for getting oriented before a code review. Shows the groups (pipelines or subpackages) and, for ML systems, the feature store and registry names each one reads and writes; drills into files and the imports between them with the names used; lists every caller of each shared file by function; and gives a guided tour from each entry point built from module docstrings. Clicking a file opens it in VS Code. The map is built by a standard library script, so refreshing it costs no tokens. Use when asked to map, visualize, or get oriented in a codebase, or to see which modules call which helpers.
argument-hint: "[repo path, default: the current folder]"
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*)
---

# Code Map

Run the map script and point the user to the page. The script does all the work; this skill only sets a repo up once and reports what the script prints. Do not read the source files, describe the map, or comment on the code: that would cost tokens on every run, and the user asked for a tool, not a review.

## Step 1: Find the repo

The repo is `$ARGUMENTS` when given, otherwise the current working directory. Call it `REPO`. If `REPO/src` does not exist, say that the map covers only `src/` and stop.

## Step 2: Set up once

If `REPO/.code_map/config.json` exists, skip to Step 3.

Otherwise run:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/code_map.py detect REPO
```

It prints `pipelines` (packages named `*_pipeline`), `assets` (constants that read or write calls use, such as feature groups and models), and `suggest_ml`. Ask one AskUserQuestion: whether to show the ML extras (store and registry arrows, tours from `jobs/`). Recommend the option that matches `suggest_ml`, and put the found pipelines and assets in the descriptions. Then run:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/code_map.py setup REPO --ml    # or --no-ml
```

Setup writes `.code_map/config.json` and a "Code map" task in `.vscode/tasks.json`, and adds both to `.git/info/exclude`, so nothing shows in `git status`. When the repo already tracks `.vscode/tasks.json`, setup leaves it alone and says so: tell the user to refresh the map with `/code-map` instead of the task.

## Step 3: Build the map

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/code_map.py REPO --open
```

The first line is the page path; each later line starting with `warning:` is an assumption the repo does not meet.

## Step 4: Reply

Keep it short:

1. The page path, opened in the default browser. Clicking a file there opens it in VS Code (the browser asks once to allow `vscode://` links).
2. The warnings, one line each, as printed.
3. After a first setup: to refresh later, run the **Code map** task (Terminal > Run Task) without Claude, or run `/code-map` again. To switch the ML extras, run setup again with the other flag.

## Files in this skill

* `scripts/code_map.py`: the scanner, the model, the page renderer, and the `detect` and `setup` commands (Python standard library only). `--json` also writes `model.json` next to the page.
* `scripts/page.html`: the page template; the model and Cytoscape.js are inlined into it.
* `scripts/cytoscape.min.js`: Cytoscape.js 3.30.2, pinned.
* Tests: `tests/test_code_map.py` in the claude-skills folder, run with `python3 tests/test_code_map.py`, on the fixture repos in `tests/fixtures/code_map/`.
