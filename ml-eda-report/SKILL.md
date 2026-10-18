---
name: ml-eda-report
description: Run exploratory data analysis on a pandas DataFrame or a data file (csv, parquet, xlsx, json) with compact numeric and categorical profile tables (everything in df.info() and df.describe(include="all") plus missing values and per column flags), missing value patterns, duplicates and keys, dates, correlations, and target analysis with leakage checks, with seaborn plots in 3 column grids. In the ML project skills it is step 6: it takes the confirmed target from the shared ml_spec.yaml (written by ml-problem-framing), analyses it, checks for leakage and checks the data against the framing, and writes its findings to the spec's eda section. On its own it can also profile any file with or without a target. In an ML project it writes notebooks/04_eda_<dataset>.ipynb, which runs the analysis in cells the user can check and rerun. On its own it asks whether to write a Jupyter notebook or a markdown report and where to save it (suggesting notebooks or experimentation folders, or next to the data). Use when asked to explore, profile, summarize, or run EDA on a dataset or DataFrame.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)
---

# EDA Report

Profile a dataset with summary statistics and plots, and rank the data quality issues found.

This is step 6 of the ML project skills: ml-project-setup, ml-business-framing (anchors only), ml-data-discovery, ml-business-framing (full framing), ml-problem-framing, **ml-eda-report**, then ml-train-test-split, ml-feature-engineering, ml-baseline-model, ml-model-training, and ml-offline-evaluation. By this step, ml-data-discovery has described the data and drafted the data dictionary, and ml-problem-framing has confirmed the target. ml-eda-report takes the target, its type, and the date and ID columns from the spec, analyses the target, checks for leakage, and checks the data against the framing. Its findings go to the spec's `eda` section, which the modelling skills read. The shared conventions are in `ML_CONVENTIONS.md` in the claude-skills folder.

The skill also works on any data file outside such a project, with or without a target. A run without a target ("discovery mode") profiles the data only.

## Ground rules

These apply for the whole task.

1. **Summary statistics only, with one exception.** The only place raw rows appear is the `df.head()` cell that the generated notebook runs right after loading the data, which the user asked for. Apart from that, never print raw rows. Do not run `df.head()`, `df.sample()`, `cat`, `head`, or similar commands on the data yourself, do not read notebook outputs that show rows, and do not paste data values into your reply. Reading column names is fine. The report sections hide values of identifier, free text, and personal data columns. Markdown reports do not include `df.head()`, so no raw rows are written into them.
2. **Never change the data.** The report only reads data. Suggest fixes, but do not apply them to the user's files.
3. **Use the scripts.** All analysis goes through `scripts/run_eda.py` or `scripts/eda_core.py`, and spec changes go through `scripts/ml_common.py`, so results are consistent between runs, between markdown and notebook output, and with the other skills.
4. **Writing style.** Use simple language and full sentences. Use plain headings. Do not use dashes as punctuation.

## Step 1: Work out the input and the output

**Input.** There are two cases:

* **A file path.** The user names a data file (csv, tsv, txt, parquet, xlsx, xls, json, jsonl, or feather). Use the file path workflow.
* **A DataFrame in a notebook.** The user is working in a Jupyter notebook where the DataFrame is already loaded, for example "run EDA on df in analysis.ipynb". Use the notebook DataFrame workflow.

**Output format.** In an ML project (a folder with `ml_spec.yaml`), the output is a notebook, because each step of the workflow is recorded in its numbered notebook. Do not ask about the format there. Use markdown only when the request asks for it (markdown, `.md`, or a report file). Outside a project, if the request mentions a notebook, Jupyter, or `.ipynb`, use a notebook; if it says markdown, use markdown; if it says neither, ask in Step 3. The notebook DataFrame workflow always adds a cell to the user's notebook, so it needs no format question.

**Save location.** If the request gives a path or folder, use it. In an ML project, the notebook goes to its standard place, `notebooks/04_eda_<dataset>.ipynb` (the folder comes from `paths` in the spec), which is the script's default, so do not ask. A run without a target is saved as `04_eda_discovery_<dataset>.ipynb`. Outside a project, ask in Step 3.

**Mode.** Decide it from the spec (the **Spec** block printed in Step 3 shows the framing status), or from the request when the user names a mode:

* **Framing `confirmed`:** targeted mode. The target and its type come from the spec, so do not ask about them.
* **Framing `draft` or `not_started` in an ML project:** the target is not settled yet. Say so in one sentence, and recommend **ml-data-discovery** (if the `discovery` section is empty) and then **ml-problem-framing** instead, because they describe the data and settle the target. If the user still wants EDA now, ask whether to profile without a target (discovery mode) or to analyse a target they name (Step 3, `--mode targeted`), and say the framing should still be confirmed with ml-problem-framing afterwards.
* **No spec, or the user passed `--no-spec`:** the skill is being used on its own. Ask for the target as in Step 3. Choosing "No target" runs discovery mode.

The script applies the same rule when `--mode` is not given: targeted when a target is given or the framing is confirmed, discovery otherwise.

**Optional settings.** Take these from the user's request when given:

* `date_col`: the main date column. Other date columns are detected automatically.
* `id_cols`: ID columns. Identifier columns are also detected from their names.

The format, the save location, and the target are settled in Step 3, by asking the user, unless the spec already settles the target.

**Shared spec.** The scripts use `ml_spec.yaml` at the project root. If the user names another spec, pass `--spec PATH`. If none exists, the run creates one at the project root with only the `eda` section filled in and an open question to run `ml-problem-framing`. Pass `--no-spec` only when the user asks not to touch the spec.

## Step 2: Find the project and the Python, and check packages

Work from the project root. Find it with:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/ml_common.py root
```

The root is the nearest folder with `ml_spec.yaml`, then `.venv`, then `.git`. `cd` there. If none is found, stay in the current folder (usually the folder with the data). Use the project's own Python, `.venv/bin/python`, because it has the right packages and its notebooks run on the project kernel. If there is no `.venv`, try `venv/bin/python`, then `python3`, and mention that the **ml-project-setup** skill would create one. Call the chosen interpreter `PY`.

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/run_eda.py --check
```

If a required package (pandas, numpy, matplotlib, seaborn) is missing, show the user the install command that `--check` prints and ask before installing anything. In a project made with ml-project-setup, that command is `uv add <package>`, which records the package in `pyproject.toml` and `uv.lock`. Optional packages only switch off small parts of the report: scikit-learn (mutual information), pyarrow (parquet), openpyxl (xlsx), nbformat, nbclient, and ipykernel (notebook output), and pyyaml (the spec is written as JSON without it). A project made with ml-project-setup has all of them.

## Step 3: Ask the setup questions

Settle the output format, the save location, and (only when Step 1 says so) the target before running anything. Ask with AskUserQuestion one question at a time, one question per call, and wait for each answer before asking the next. The user answers at most four short questions. Leave out any question the request already answers.

**Gather the facts first.** Run both commands with the interpreter from Step 2. Neither prints data values.

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/run_eda.py <data file> --list-columns [--target <name the user gave>]
PY ${CLAUDE_SKILL_DIR}/scripts/run_eda.py <data file> --suggest-locations
```

* `--list-columns` prints every column's dtype, unique count, and missing %, then up to five likely target columns ranked by a score, each with a suggested type and the reasons for its score. The score uses the column name (for example `target`, `label`, `is_*`, `churn`, `revenue`), the data (binary, a few classes, or continuous numbers), the position (first or last column), and penalties for IDs, dates, free text, personal data, constant columns, and mostly missing columns. With `--target`, it also checks that the named column exists and prints the closest matches if not. Combine the ranking with anything the user said about the goal, such as "predict clicks".
* `--list-columns` also ends with a **Spec** block: the spec path, the framing status, and the target, target type, task type, date column, ID columns, and excluded columns the spec already gives, followed by the mode the script would use. A value marked "NOT in the data" does not match any column.
* `--suggest-locations` looks in the project (the current working directory, two levels deep, skipping `.venv`, `.git`, and similar folders) for `notebooks/`, `experimentation/`, `experiments/`, `analysis/`, and similar folders, and prints save paths for each format, recommended first. A notebook or experimentation folder is recommended when one exists, otherwise the folder next to the data file. For markdown it also lists `docs/`. It marks paths where a file already exists.

For a DataFrame in a notebook, skip both commands. Read the notebook's code (not its outputs) for the column names and hints such as `y = df["label"]` or a `train_test_split` call. If the columns are not visible, ask for the target name in plain text.

**Questions** (in this order, one AskUserQuestion call each):

1. **Output format**, only outside an ML project, and only when the request did not say. Question: "Should the EDA be a Jupyter notebook or a markdown report?" Options: "Jupyter notebook (.ipynb)", described as "Runs top to bottom, each section can be rerun, raw rows shown once with df.head()", and "Markdown report (.md)", described as "A single file with tables and PNG plots, no raw rows". Put markdown first and mark it (Recommended), unless the user's wording points to working interactively.
2. **Target column**, only when Step 1 says to ask (the skill is used on its own, or the user wants target analysis before the framing is confirmed). Ask even when the inference seems certain or the user already named a column, unless the user said to skip target analysis. If the spec names a target that is not in the data, say so in the question.
   * Question: "Which column is the target? I inferred `<best candidate>` because <its reasons in a few words>."
   * Options: the best candidate first, marked (Recommended), then up to two other candidates, then **always** a final option labelled "No target", described as "Skip target analysis. The report covers the data only, with no class balance, target relationships, or leakage checks." AskUserQuestion allows at most four options, so the "No target" option is never dropped: if there is no room, show fewer candidate columns instead. Include it even when the user named a target or the inference looks certain. In each candidate's description, give the dtype, unique count, and the main reason, for example "int64, 2 unique values, named like a target, first column". The user can also type another column name.
   * If no column scores above zero in `--list-columns`, still ask, with "No target" first and marked (Recommended), followed by the top one or two columns.
   * If the user picks "No target", run without `--target`, skip the target type question, and leave out `target` and `target_type` in the notebook DataFrame workflow. The Target Analysis section then says it was skipped and how to turn it on later.
   * If the user named a target and it exists, make it the recommended option. If it does not exist exactly, say so in the question and offer the closest matches the script printed.
   * If the user types a name that is not a column, rerun `--list-columns --target <name>` and ask again with the closest matches.

3. **Save location**, only outside an ML project, and only when the request did not give one. Question: "Where should I save `<file name from --suggest-locations>`?" Use the paths `--suggest-locations` printed for the chosen format, recommended first and marked (Recommended), up to four options. The first is the standard project folder from the spec. Say in the description when a folder is the standard project folder, another notebook or experimentation folder, when it is next to the data, and when a file already exists there and would be overwritten. The user can type another path. For markdown, the plots go in an `eda_plots/<file name>/` folder next to the report.
4. **Target type**, only when a target was chosen, the request did not already say the type, and the target did not come from a confirmed framing. Ask even when the type seems obvious, because a 0 or 1 column or an integer score can be either.
   * Question: "Is `<column>` continuous or categorical?"
   * Options: "Continuous (regression)" and "Categorical (classification)". Put the type suggested by `--list-columns` first and mark it (Recommended). Mention the dtype and number of unique values in the descriptions, for example "float64 with 29,288 unique values".

Settings the user does not give on the command line are taken from the spec when it has them: target and target type (from `framing.target.name` and `framing.task_type`), date column (`data.date_col`, then `framing.prediction_time.column`), and ID columns (`framing.entity_id_cols` and `data.id_cols`). The script prints a "From the spec:" line listing what it took. Anything given on the command line wins over the spec.

Pass the answers on the command line: `--format ipynb --notebook <path>` or `--report <path>` for markdown, `--target <column> --target-type continuous|categorical` when a target was chosen, and `--mode discovery|targeted` when it differs from the default. Run `--suggest-locations` with the same `--mode` so the file names match. In the notebook DataFrame workflow, pass `target=` and `target_type=` in `eda_report(...)`.

If AskUserQuestion is unavailable or the user is not present, use a notebook in its standard place inside an ML project, and markdown in the recommended location otherwise. Use the mode from Step 1 (discovery unless the framing is confirmed). Say so at the top of the reply, and list the inferred target columns with their suggested types and reasons so the user can rerun.

**What the type changes.** A continuous target goes in the numeric profile table and the distribution plots, uses Spearman and Eta for feature relationships, and shows 2D histograms against numeric features. A categorical target goes in the categorical profile table, gets class balance and an imbalance check, uses Spearman (two classes), Eta, or Cramér's V, and shows box plots by class. A continuous target that is not stored as numbers is rejected with a clear message.

## Step 4a: File path workflow, markdown output

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/run_eda.py <data file> --report <chosen path>.md [--mode discovery|targeted] \
    [--target COL --target-type continuous|categorical] [--date-col COL] [--id-cols A,B] \
    [--corr-threshold 0.9] [--cramers-threshold 0.8] \
    [--sheet NAME] [--sample-rows 200000] [--max-plots 24] [--log-scale]
```

The script prints the mode it used, writes the `eda` section of `ml_spec.yaml`, and prints the framing hints (discovery mode) or the framing check results (targeted mode). The report is written to the chosen path, and its PNG plots go in `eda_plots/<file name>/` in the same folder, so reports for several datasets can sit side by side. Missing folders are created. If a report already exists at that path, it is replaced, which is why the location question says so.

## Step 4b: File path workflow, notebook output

1. **Make sure the editor will pick the right kernel.** VS Code and Jupyter match a notebook to a kernel by its registered kernel name. If the project's environment is not registered, the editor falls back to whatever Python it used last, which may belong to another project. Check with:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/run_eda.py --kernel-status
   ```

   If it prints "Registered Jupyter kernel: no", ask with AskUserQuestion (as its own question, after the Step 3 questions): "Register this project's Python environment as a Jupyter kernel so VS Code and Jupyter select it automatically? This adds one small kernel file in your user Jupyter folder." Options: "Yes, register it" (Recommended) and "No, I will select the kernel myself". If the user says yes, add `--register-kernel` to the command below. Never register a kernel without asking.
2. Run:

   ```bash
   PY ${CLAUDE_SKILL_DIR}/scripts/run_eda.py <data file> [--format ipynb --notebook <chosen path>.ipynb] [--register-kernel] [same options as above, without --report]
   ```

   In an ML project, leave out `--format` and `--notebook`: the notebook is the default, at `notebooks/04_eda_<dataset>.ipynb`.

   **If it exits with code 2,** the user edited that notebook after the skill last wrote it. Do not overwrite it silently. Ask with AskUserQuestion: "Replace it" (rerun with `--overwrite`), "Write a new version" (rerun with `--new-version`, which writes `04_eda_<dataset>_v2.ipynb`), or "Keep my edits" (run it as it is with `PY ${CLAUDE_SKILL_DIR}/scripts/nb_common.py run <notebook>`).

3. Always run this with the project's own interpreter from Step 2 (for example `.venv/bin/python`), not a global `python3`. The notebook is executed on that exact interpreter, the notebook metadata names its registered kernel (for example `martech_adtech_data`) when there is one, and its first cell prints the kernel's Python path and stops with a clear message if pandas, numpy, matplotlib, or seaborn are missing. In the reply, give the kernel line the script prints. If the kernel is not registered, tell the user to choose "Select Kernel" and pick the interpreter path the script used.
4. The script runs the notebook once so the outputs and plots are filled in. The notebook's last section, "Update the shared spec", writes the `eda` section of `ml_spec.yaml`, so running it also updates the spec. If it prints `Executed: no`, the notebook was saved without outputs and the spec was not updated. Tell the user the reason it printed, and that they can open the notebook and choose Run All.

The notebook loads the data, runs `df.head()` in the next cell, then runs all checks once and has one cell per report section. Each section can be rerun on its own. There is no separate `df.info()`, `df.describe()`, or `df.isna()` cell, because the Column Profiles tables already cover them. Never add a plain `df.describe(include="all")`, because that prints the most common value of every text column, including emails and IDs.

## Step 4c: Notebook DataFrame workflow

Claude cannot run code inside the user's live notebook kernel, so add a cell for the user to run.

1. Read the notebook to find the DataFrame's variable name. Do not read or print cell outputs that contain raw rows.
2. Add one new code cell at the end of the notebook, or after the cell the user points to, with NotebookEdit:

   ```python
   import os, sys
   sys.path.insert(0, os.path.expanduser("~/.claude/skills/ml-eda-report/scripts"))
   from eda_core import eda_report

   eda = eda_report(
       df,                      # the user's DataFrame
       target=None,             # for example "churned". None runs discovery mode
       mode=None,               # "discovery" or "targeted"; default: targeted when target is set
       target_type=None,        # "continuous" or "categorical", as confirmed with the user
       date_col=None,           # for example "signup_date"
       id_cols=None,            # for example ["customer_id"]
       name="df",               # used in the report title and file name
       out_dir=None,            # set to "docs" to also write docs/eda_<name>.md (path from the notebook's folder)
       spec_path="ml_spec.yaml", # shared spec: framing checks in, eda section out. None to skip
   )
   ```

   Fill in the variable name and any settings the user gave. Use the spec's path relative to the notebook's folder, or an absolute path, and fill `target`, `target_type`, `date_col`, and `id_cols` from the spec when it gives them, since this function does not read them from the spec by itself. Leave `target` as None while the framing is not confirmed, following the mode rule in Step 1. Use the real path of this skill's `scripts` folder (`${CLAUDE_SKILL_DIR}/scripts`) if it is not under `~/.claude/skills/ml-eda-report`.
3. Tell the user to run the cell. Every section appears inline. They can show one section again later with `eda.show("missing")`. For the relationships section, `eda.show("relationships", corr_threshold=0.7, cramers_threshold=0.5)` lists the pairs above new thresholds without recomputing anything, and `eda.correlated_pairs(0.7)` or `eda.associated_pairs(0.5)` return those pairs as DataFrames. For the distributions section, `eda.show("distributions", top_n=10)` plots the top 10 numeric and the top 10 categorical columns, ranked by association with the target when there is one, and `eda.show("distributions", log_scale=True)` redraws strongly skewed numeric columns on a log axis (the default is their original scale). For the target section, `eda.show("target", top_n=10, min_association=0.1, min_mutual_info=0.01)` lists and plots at most 10 features, and keeps only those with an absolute association of at least 0.1 and mutual information of at least 0.01 (each limit is optional). Every plot title shows the feature's association and mutual information. The section names are `overview`, `profiles`, `missing`, `distributions`, `relationships`, `target`, and `summary`.

If the user is not using a notebook file but still wants in-memory use, give them the same code to paste.

## Step 5: Update the spec

The run already wrote the `eda` section (in the notebook workflows, the notebook's last cell does it). Check the result from the project root:

```bash
PY ${CLAUDE_SKILL_DIR}/scripts/ml_common.py status
```

Read the generated markdown report (or the summary printed by the script for a notebook), and the `eda` section and open questions of the spec. Look at two or three of the most important plots with the Read tool to confirm they render properly. Do not change `framing` or `data.excluded_cols` yourself. If the checks suggest a change there, put it to the user, and record any agreed change in `decisions`.

## Step 6: Report back

Reply with these parts, in this order. Do not paste the whole report, and do not include data values beyond what the report itself shows.

1. **Files written:** the report or notebook path, the plots folder, and whether the spec was created or updated.
2. **Summary:** three to five sentences on the data (size, grain, main column types, and overall quality), and which settings came from the spec.
3. **Problems:** the High severity issue types, each in one sentence. In targeted mode, then every Fail and Check result from the framing checks in one sentence each (or one sentence saying the data matches the framing). Mention sections that were skipped and how to turn them on.
4. **Framing hints** (discovery mode only): the candidate targets, the grain, the possible entity IDs, the history length, and the columns that may be recorded after the outcome, one line each.
5. **Open questions:** the new open questions the spec now lists, one line each, with the stage that needs the answer.
6. **Next step:** in discovery mode inside an ML project, run **ml-data-discovery** and **ml-problem-framing** to settle the target, then run this skill again. In targeted mode, if a framing check failed, run **ml-problem-framing** again to fix the framing, or fix the data. Otherwise resolve the open questions for the split stage and run the **ml-train-test-split** skill (once it exists), which reads the grain, the entity IDs, and the date column from the spec.

## What the report covers

The report is built to avoid repeating the same numbers. Each fact appears in one place, and the summary groups issues by type instead of listing one row per column.

1. **Overview:** rows, columns, memory, sampling, column counts by role, duplicate rows, the grain, and type problems, in a few sentences.
2. **Column Profiles:** one table for numeric columns (dtype, count, missing %, unique, mean, std, min, P1, P5, P25, median, P75, P95, P99, max, skew, zero %, negative %, outlier %, flags), then one table for categorical columns (dtype, count, missing %, unique, most common value, its count and share, rare categories, flags). Date, free text, and identifier or personal data columns get small tables of their own when the data has them. The numeric table is sorted by missing %, then outlier %, and the categorical table by missing %, then unique %, both lowest first, with the target row kept at the top. Together these cover everything in `df.info()` and `df.describe(include="all")`. The Flags column lists every issue found for that column, such as "highly skewed", "high cardinality", or "mostly missing".
3. **Missing Values:** a short summary, a bar chart of columns with missing values, disguised missing values, a heatmap of columns missing together with a table of the strongest pairs and a note on how the correlation is calculated, a structural missingness part (groups of columns that go missing together, a verdict for each group, the target rate when a group is missing and when it is present, and the rule in another column that explains it, if any), and missing values over time when there is a date column.
4. **Distributions:** one grid of numeric histograms with box plots, one grid of top values for categorical columns, and rows over time. Grids have 3 columns and up to 24 panels (`--max-plots`). Numeric columns are plotted on their original scale with all of their values. `--log-scale` (or `log_scale=True`) plots strongly skewed columns on a log axis instead.
5. **Relationships:** first a table of numeric pairs with |Spearman| at or above `corr_threshold` (default 0.9) and a table of categorical pairs with Cramér's V at or above `cramers_threshold` (default 0.8), or one sentence when no pair qualifies. Then one figure with the Spearman heatmap and the Cramér's V heatmap side by side, then date ordering checks. Set the thresholds with `--corr-threshold` and `--cramers-threshold`, or in the notebook with `eda.show("relationships", header=False, corr_threshold=..., cramers_threshold=...)`. Thresholds the user asks for in the request go on the command line.
6. **Target Analysis:** class balance or task type in one or two sentences, one table of feature relationships with the target (with mutual information), leakage warnings, and one grid of the strongest features against the target.
7. **Data Quality Summary:** in discovery mode, it starts with hints for the ML framing (candidate targets with the smaller class share for binary ones, the grain, possible entity IDs with rows per value, date columns with their range, and columns whose names suggest they are recorded after the outcome). In targeted mode with a framing, it starts with a table of checks against the framing (target column exists, target type matches the task type, class balance against the primary metric, entity ID columns and rows per entity, grain against the unit of prediction, prediction time column, excluded columns, and leakage suspects not yet excluded). A failed check is also listed as a High issue. Then one row per issue type with the affected columns, ranked by severity, a single line of checks that found nothing, and next steps.

All plots use seaborn with one consistent style, applied only inside the report so the user's own matplotlib settings are not changed.

See `reference/checks.md` for every threshold and how each check works.

## What the spec receives

The `eda` section of `ml_spec.yaml` is replaced on every run. It holds the mode, the source and report paths, rows and columns, the detected grain, duplicate rows, the main date range, column names grouped by role, target statistics (with class balance for up to ten classes), the ten features most associated with the target, leakage suspects, suggested exclusions (identifiers, free text, personal data, and strong leakage suspects), the grouped issues, the framing checks (targeted mode), and the framing hints (both modes). It never holds raw rows or values other than the target's class labels and the date range. The run also records the data file in `data.sources`, fills `data.date_col` and `data.id_cols` when they were empty and the user gave them, and adds open questions for failed checks and unresolved leakage suspects (marked with the stage that needs the answer). It never changes `framing` or `data.excluded_cols`. Suggest exclusions to the user instead.

The spec code is `scripts/ml_common.py`, the notebook helpers are in `scripts/nb_common.py`, and field meanings are in `reference/spec_fields.md`. All three are copies of the canonical files in `claude-skills/_ml_shared/`, kept in sync by `tools/sync_ml_shared.py`.

## Large data

Above `--sample-rows` rows (default 200,000), counts, missing values, duplicates, and summary statistics still use every row, while plots, correlations, associations, and mutual information use a random sample. The report says so at the top. About one million rows with twenty columns takes roughly twenty seconds.
