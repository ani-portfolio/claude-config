# Conventions for the ML Project Skills

These rules apply to every skill that takes an ML project from an empty folder to an evaluated V1 model. Follow them when changing an existing skill or writing a new one, so the skills behave like one workflow rather than separate tools. `_ml-skill-template/` is a starting point for a new skill.

## Naming and scope

1. Every skill in the ML project workflow is named with the `ml-` prefix, so typing `/ml-` lists them all. The folder name and the `name` in the SKILL.md frontmatter are the same.
2. These conventions, `_ml_shared/`, `_ml-skill-template/`, and `tools/sync_ml_shared.py` apply only to `ml-` skills that are part of the workflow (the ones listed in `_ml_shared/skills.txt`). Skills for other kinds of work keep their own names and rules.
3. **Helper skills** support the workflow without being a step in it. `ml-status` reads the spec and reports each step's state, out of date results, blocking questions, notes that may disagree, and the next skill. It never writes anything. `ml-reconcile-notes` finds notes that disagree across stages and, with the user's approval, fixes each one in the notebook that owns it and runs the notebooks again. `ml-review` audits the project: scripted checks plus an independent reviewer (a fresh agent that did not build the work). It writes only `notebooks/10_review.ipynb`, `review_findings.parquet`, and the `review` section, and names the skill that should fix each finding. `ml-autopilot` drives the whole workflow as an agent (see "The autopilot" below); it writes only the `autopilot` section, `docs/checkpoint.md`, and the experiment worktrees. `ml-iterate` plans and tracks an iteration after the evaluation: it records the hypothesis and the changes by stage in the `iterations` section, snapshots the starting state to `iterations/iter_<N>/`, hands off to each stage skill in rerun order, carries the old champion forward so it is compared on the same folds, and guards another look at the test set. It writes only the `iterations` section, the snapshot folders, and a decision when an iteration opens or closes. `ml_common.consistency_checks` holds the fixed checks both helpers use; add a rule there when a new stage adds a setting that another stage repeats. Helper skills that read the spec are listed in `skills.txt` after the workflow steps, so they get the shared files too.
4. `ml-system-summary` also uses the prefix because it is about ML systems, but it is not part of the workflow: it does not read or write `ml_spec.yaml` and is not listed in `skills.txt`.
5. **The A/B testing skills** (`ab-` prefix) are a separate workflow with their own spec, `ab_spec.yaml`, and conventions, `AB_CONVENTIONS.md`. They share `nb_common.py` and the project layout with these skills. When ml-business-framing concludes that ML is not needed because the question is whether a change worked, it points to them.

## The workflow

| Step | Skill | Main output |
|---|---|---|
| 1. setup | ml-project-setup | Project folder, `.venv`, `pyproject.toml` and `uv.lock`, kernel, `ml_spec.yaml`, `notebooks/00_scratch.ipynb` |
| 2. business (anchors only) | ml-business-framing | `notebooks/01_business_framing.ipynb` as a draft, with `business.mode` and the two anchors |
| 3. discovery | ml-data-discovery | `notebooks/02_data_discovery.ipynb`, the `discovery` section, and `docs/data_dictionary.yaml` |
| 4. business (full framing) | ml-business-framing | `notebooks/01_business_framing.ipynb`, the `business` section, and the summary `docs/business_framing.md` |
| 5. framing | ml-problem-framing | `notebooks/03_problem_framing.ipynb`, the `framing` section, and the `data` settings |
| 6. eda | ml-eda-report | `notebooks/04_eda_<dataset>.ipynb` and the `eda` section, with target analysis and framing checks |
| 7. split | ml-train-test-split | `notebooks/05_split.ipynb`, the `split` section, and `train.parquet`, `test.parquet`, and `cv_folds.parquet` in `data/processed/` |
| 8. features | ml-feature-engineering | `notebooks/06_features.ipynb`, the `features` section, `src/features.py` and `src/custom_features.py`, and `features_train.parquet` and `features_test.parquet` in `data/processed/` |
| 9. baseline | ml-baseline-model | `notebooks/07_baseline.ipynb`, the `baseline` section, and the results and out of fold predictions as parquet |
| 10. training | ml-model-training | `notebooks/08_training.ipynb`, the `training` section, the run log and out of fold predictions as parquet, and the champion in `models/` |
| 11. evaluation | ml-offline-evaluation | `notebooks/09_evaluation.ipynb`, the `evaluation` section, and `docs/model_card.md` |
| 12. system | ml-system-design | `docs/system_design.md` (no notebook), the `system` section, and `answers/11_system.yaml` |

The order follows what each question needs: the mode and the two business anchors (the goal and the decision, or for an exploratory project the learning question and what each answer leads to) need no data and are what the data cannot tell, so they come first and keep the framing tied to a real decision; the rest of the business framing is then drafted from what discovery found, so its questions carry suggestions; the exact target, unit, and prediction time need the data to be understood first; and target analysis needs a confirmed target. The steps are defined once, in `WORKFLOW` and `STEPS` in `_ml_shared/ml_common.py`. `WORKFLOW` lists the stages (one section and one notebook each); `STEPS` lists the steps, where the business stage appears twice, as `business_anchors` and `business`. A new step is added there first. `/ml-status` shows progress, out of date results, and the next step. Underneath, `ml_common.py status` and `ml_common.py next` give the same progress from the command line. ml-data-discovery can be skipped when the data is already well known: ml-problem-framing then works from the column names, and the status shows the step as skipped.

**The modelling gate.** Steps 7 to 11 start by running `ml_common.py gate`. It exits with code 1 while the ML framing is not confirmed, the targeted EDA has not run on the confirmed target, or the business framing concluded that ML is not needed. The skill then names the step to run first and stops, unless the user explicitly chooses to continue, which it records in `decisions`. This keeps models from being built on an unconfirmed target.

**Project modes.** `business.mode` is `standard` (a clear business goal and KPI) or `exploratory` (an experiment). An exploratory project replaces the goal, decision, and current approach with `learning_question` and `answer_actions`, and its V1 bar is a margin over the best simple baseline (`pct_over_reference`, 10% by default), tied to an answer action. A spec without a mode is standard. Skills that read the business section check the mode with `ml_common.business_mode`.

**The iteration bar.** A new version (V2, V3, ...) is kept only when its champion passes `iterations.bar` against the champion carried forward, on the same folds: it wins on more than half the folds, and its mean improvement is larger than the spread of the per fold differences (`ml_common.iteration_bar_evaluate`). The bar also records the most rounds of iterating without a person (`max_iterations`), which a driver agent enforces.

**Discovery before targeting.** ml-data-discovery describes structure, coverage, joins, and the meaning and timing of columns only. Feature to target relationships wait for ml-eda-report, so the target and the split are chosen before anyone studies which features predict the target.

**The data dictionary.** `docs/data_dictionary.yaml` is the record of what each column means and when it is filled in. It is written by `02_data_discovery.ipynb`. Corrections go in that notebook's `dictionary_answers` cell, and running the notebook applies them, so they are kept when discovery runs again. Later skills check a column's `timing` there before using it as a feature, and never use columns marked `possibly_after_outcome` or `unknown` without the user's confirmation.

## The shared spec

1. `ml_spec.yaml` sits at the project root. Every skill reads the whole file before asking anything, and skips questions the spec already answers.
2. Each skill writes only its own section, and only through `ml_common.py` (`update_section`, `merge`, `fill`) or `nb_common.write_stage` from its notebook, so the layout stays consistent and every write is logged in `history`.
3. `assumptions`, `open_questions`, and `decisions` are shared lists. Append to them, never replace them. Each open question names the stage that needs the answer. A skill marks a question `resolved`, with an `answer`, when its work settles it.
4. A skill may suggest changing a section it does not own, but it asks the user first and records the change in `decisions`.
5. The spec never holds raw rows or data values. Column names, counts, shares, and date ranges are fine.
6. Field meanings are in `_ml_shared/spec_fields.md`. Add new fields there in the same change that adds them to the template in `ml_common.py`.

## Project root and paths

1. The project root is the nearest folder with `ml_spec.yaml`, then `.venv`, then `.git` (`ml_common.find_project_root`). Every skill starts by finding it and works from there.
2. Folders come from `paths` in the spec (`ml_common.project_paths`), never from hard coded strings. The defaults are `data/raw`, `data/processed`, `notebooks`, `src`, `models`, and `docs`. An older project can point `data_raw` at `data`.
3. Every data file a notebook writes is parquet: split sets, feature tables, run logs, predictions, and evaluation results. Documents are the only exception (the business summary, the data dictionary, the model card, and the system design), and fitted models are saved in `models/` as joblib files.
4. Run logs (such as `training_runs.parquet`) are only appended to. A stage that can be run again to add work (such as adding a model) reads earlier results back from its log instead of recomputing them.
5. `data_raw` is read only. Skills write data only to `data_processed`, models only to `models`, notebooks only to `notebooks`, and the few documents (the business summary, the data dictionary, the model card, the system design) only to `docs`.

## Python and permissions

1. Scripts run with the project's own `.venv/bin/python` from the project root. Only ml-project-setup runs with `python3`, because the `.venv` may not exist yet. If a project has no `.venv`, a skill falls back to `python3` and suggests ml-project-setup.
2. Every skill's frontmatter allows both forms: `allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*), Bash(.venv/bin/python ${CLAUDE_SKILL_DIR}/scripts/*)`.
3. Packages are managed with uv. Projects record dependencies in `pyproject.toml` and exact versions in `uv.lock`, and the project itself is installed in editable mode so `import src` works anywhere. A skill that needs a package the project lacks runs `ml_common.py need <package>`, shows the user the `uv add <package>` command it prints, and asks before running it. It never runs `pip install` into `.venv`, because `uv sync` would not know about the package. Older projects that still use `requirements.txt` get `uv pip install` instead, until they move with `setup_project.py --to-pyproject`.
4. A package every stage needs goes in `BASE_PACKAGES` in `ml-project-setup/scripts/setup_project.py`. A package only later stages need goes in an extra group (`sklearn` for scikit-learn from ml-train-test-split onwards, `tuning` for optuna), so a project that stops before modelling does not install it. Those stages check for it with `ml_common.py need` in Step 1.

## Notebooks

The notebook is the record of each step. A person should be able to open it, read why each decision was made, check it against the data, change a decision, and run it again.

1. Every step writes one numbered notebook, except ml-system-design (see rule 12): `ml_common.notebook_name(stage, suffix)` gives `01_business_framing.ipynb`, `02_data_discovery.ipynb`, `03_problem_framing.ipynb`, `04_eda_<dataset>.ipynb`, `05_split.ipynb`, and so on to `09_evaluation.ipynb`. `00_scratch.ipynb` is for free exploration.
2. A notebook starts with a markdown title and a short note on how to use it, then the setup cell from `nb_common.setup_cells`, which finds the project root, checks the kernel, and loads the spec as `spec`.
3. Markdown cells hold the reasoning, the risks found, the assumptions, and the open questions, in plain sentences.
4. Code cells that hold answers are marked *Decision* (a `# Decision` first line and the `decision` cell tag). They hold readable Python values, one setting per line, so a person can change them. Every other code cell computes from the data or the spec, so a person can check each claim, such as the grain, the date coverage, or the class balance.
5. A notebook loads what the previous stage saved (from `data_processed` and `models`) and saves what the next stage needs. It ends with a cell that checks the decisions, writes them to the spec with `nb_common.write_stage`, and records the run with `nb_common.record_run`. Running the notebook is therefore what updates the spec, and the spec always matches the notebook that last ran.
6. Skills build notebooks from scripts (`make_notebook.py` or the stage's main script) and run them once with `nb_common.save_and_run`, so outputs are filled in. If running fails, the notebook is still saved with the outputs so far, and the reply says which cell failed.
7. **Edits are protected.** The notebook metadata keeps a hash of the cells as the skill wrote them. When a person has edited the notebook since, the script does not replace it: it exits with code 2. The skill then asks the user to choose one of three options: replace it (`--overwrite`), write the next version such as `03_problem_framing_v2.ipynb` (`--new-version`), or keep the edits and run it as it is (`nb_common.py run <notebook>`).
8. `/ml-status` compares each notebook's decision cells with those recorded at its last run, and says when a notebook was edited and not run again.
9. Notebook metadata names the project kernel (`ml_common.kernel_name(project.name)`), the same name ml-project-setup registered.
10. Code that later stages and production must share lives in the project's `src/`, not in the notebooks. `src/features.py` is written by ml-feature-engineering from its template and protected like a notebook: the skill refuses to replace it once edited. `src/custom_features.py` belongs to the user. Learned steps are only fitted inside the CV folds, by the stage that trains a model.
11. There are no separate markdown reports. The only documents in `docs` are the business framing summary, which the business notebook writes for people who do not read notebooks, the data dictionary, and later the model card and the system design.
12. **ml-system-design writes no notebook.** A design has no data to check, so its record is `docs/system_design.md`, generated by `scripts/design.py` from the `system` section and the reasoning in `answers/11_system.yaml`. A hand edit of the document is protected like an edited notebook (exit code 2). To change a decision, edit the answers file and build again. The autopilot never runs it, and `ml-iterate` does not rerun it; `/ml-status` marks it out of date when the champion or the evaluation changes.

## Experiment tracking

Every project logs the runs of its modelling stages to MLflow, so runs can be browsed and compared across time, models, and projects.

1. **One shared server, one store.** `tools/mlflow_server.py` runs MLflow (the full package, as a uv tool outside every project) at `http://127.0.0.1:5000`, with its data in `~/.mlflow/` (`mlflow.db` and `artifacts/`). Every project and the experimentation agent log there. The address comes from `MLFLOW_TRACKING_URI`, then the spec's `tracking.uri`, then that default, so a team server can replace it with one setting.
2. **Projects install the client only.** The `tracking` extra is `mlflow-skinny>=3.17,<4`, never the full `mlflow`, whose pins (such as `pyarrow<26`) make uv fall back to an old MLflow that does not import.
3. **The spec decides; MLflow records.** Decisions live in the spec and the notebooks, and the parquet run log stays the cache that stops models from retraining. MLflow is a view on top: nothing reads a decision back from it.
4. **Tracking never blocks the work.** When `mlflow` is not installed or the server is down, the notebook prints one line and finishes. ml-status and ml-review report the gap.
5. **One experiment per project**, `ml/<project>` (A/B projects: `ab/<project>`, see `AB_CONVENTIONS.md`), created by ml-project-setup and tagged with the project folder. **One run per stage notebook run**, from split onwards and for reviews, tagged with the project, stage, skill, notebook, and data version. The newest run of each stage has the tag `current = true`. Baselines, models, and evaluated models are child runs.
6. **What is logged** is read from the stage's spec section after the notebook writes it (`tracking.log_stage`, in the cells from `nb_common.tracking_cells`): settings as parameters, results as metrics, small tables as CSV, every chart the notebook showed, the stage's spec section, and the executed notebook. Training logs each model's best trial only (every trial stays in the run log), and registers the champion as `<project>-champion` with the alias `champion` and the version alias `champion-v1`, `champion-v2`, and so on. The run id goes to `<stage>.mlflow` in the spec.
7. **Archiving a project.** `tracking.py export` copies the project's runs and artifacts into `mlflow_export/` in the project.
8. MLflow's usage reporting to its makers is turned off (`MLFLOW_DISABLE_TELEMETRY`).

## Versions and git

Every project is a git repository, and every pass at the model is a version.

1. **Versions.** The first pass through the workflow is V1. Each iteration opened with ml-iterate builds the next version: iteration 1 builds V2, iteration 2 builds V3. `spec.version` holds the current version, its branch, and the history of earlier versions.
2. **Nothing is committed to main.** ml-project-setup makes the repository with one empty commit on main ("Initialize repository") and does all work on the branch `ml/v1`. Each version has its own branch, `ml/v2`, `ml/v3`, branched from the version before it. A person merges into main when they want to; the skills never do. When a stage build finds main checked out, it moves to the version branch before it commits.
3. **Automatic commits.** After every successful stage build, `nb_common.cli` commits the stage's files: the spec, the stage notebook, the kept answers, `src`, and `docs`. Only these paths are committed, so a person's other changes stay out. Data, models, `.venv`, snapshots, and MLflow exports are ignored by `.gitignore`. Each commit message is `<version> <stage>: built by <skill>`, with trailers for the version, stage, skill, MLflow run, and Claude Code session. `git.auto_commit: false` in the spec turns this off.
4. **Opening and closing a version.** ml-iterate tags the version it leaves (`v1`), branches `ml/v2`, and creates the version's MLflow parent run. Closing as kept or stopped promotes the version in MLflow and tags `v2`. Rolling back restores the snapshot, rejects the version in MLflow, tags `v2-rolled-back`, and goes back to the previous branch, where it commits the spec with the record of the rollback.
5. **The git identity** is the person's own (`git config user.name` and `user.email`). Without one, nothing is committed and setup says how to set it.
6. `_ml_shared/git_ops.py` holds the git helpers; `git_ops.py status` prints the branch, the last commit, and uncommitted files.

## Agent observability

Claude Code sessions in a project are traced into the project's own MLflow experiment, next to its runs.

1. **Traces.** ml-project-setup runs `mlflow autolog claude` for the project. It writes `.claude/settings.json` (the tracking address and experiment id) and `.claude/settings.local.json` (MLflow's Claude Code tracing plugin). Every session in the project then appears as a trace in the experiment's Traces tab.
2. **Linking runs to sessions.** The setup also adds a SessionStart hook, `tools/session_env.py`, that puts the session id in `ML_AGENT_SESSION` for every command of the session. Stage runs are tagged `agent_session` with it, and commits carry an `Agent-Session` trailer, so a run, its commit, and the trace of the session that made it can be found from each other.
3. **Versions in MLflow.** Every stage run is tagged with `version`, `git_branch`, and `git_commit`, and logs the kept answers as an artifact. From V2 on, the runs of a version are children of the version's parent run and are tagged `candidate` until the version closes. The alias `champion` moves only when a version is kept.
4. **Testing.** `tests/run_regression.py` runs the modelling stages on the fixtures in `tests/fixtures` against a throwaway MLflow store, and with `--iterate` also tests a kept V2 and a rolled back V3. Run it after changing the shared files.

## SKILL.md structure

Every skill uses the same order, so they read alike:

1. Frontmatter: `name`, a `description` that says what it does, which stage it is, and when to use it, and `allowed-tools` as above.
2. A one paragraph purpose, then the workflow sentence and a pointer to this file.
3. **Ground rules:** never invent answers, never change input data, no raw rows, use the scripts, and the writing style rule.
4. **Step 1: Find the project and the spec.** Run `ml_common.py root` and `status`, and read the sections this skill depends on. Modelling skills (split onwards) also run `ml_common.py gate`.
5. The skill's own steps. Ask one question at a time: exactly one question per message or AskUserQuestion call, then wait for the answer before asking the next. Never send a numbered list of questions. Lead with proposed answers marked (Recommended), push back on a vague answer with one follow up before moving on, and skip anything the spec or an earlier answer already settles. Each skill says what happens when AskUserQuestion is unavailable: take the recommended answers, write them as drafts or assumptions, and say so at the top of the reply.
6. **Build and run the notebook**, which updates the spec, second to last.
7. **Report back**, last, with these parts in this order: Files written, Summary, Problems, Open questions, and Next step (which names the next skill).

## Running without a person

The skills are built so an agent (or ml-iterate) can drive them, not only a person in a conversation.

1. **Every stage script is non-interactive.** It takes an answers file and options, never prompts, and returns an exit code: 0 done, 1 failed, 2 the notebook was edited by hand, 3 `src/features.py` was edited by hand. Each script's entry point is `nb_common.cli(main, stage, skill)`, so `--json` sends the usual text to stderr and prints one JSON result on stdout (`exit_reason`, the notebook, and a summary of the stage's section).
2. **The answers are kept.** After a successful build, `nb_common.cli` saves the answers and the command line in `answers/<NN>_<stage>.yaml`, so a stage can be rebuilt without anyone answering questions again.
3. **The V1 bar is checkable.** `framing.success.v1_criteria` holds the bar as checks (`ml_common.v1_evaluate`); project guardrails are functions in the user's `src/v1_checks.py` (`model_core.run_v1_checks`). No script parses the `v1_bar` text.
4. **The saved champion loads anywhere.** `model_core.save_model` stores skill estimators (such as `NonNegRidge`) by value, so `joblib.load` needs only scikit-learn and the project's `src`.
5. **Evaluation checks itself before the test set is opened.** Every model's CV score must reproduce through the evaluation's scoring path (`eval_core.self_check`); otherwise the notebook stops.
6. **Experiments never take over the project in MLflow.** While `ML_ITERATE_EXPERIMENT` is set, or the project root has `.ml_workspace.json`, `tracking.log_stage` logs stage runs as candidates under the experiment's parent run, never as `current`, and the champion it registers gets the alias `candidate-<experiment>`, never `champion` or the version's alias. Only `tracking.promote` moves `current` and the aliases; `tracking.reject` only tags. The spec still decides; MLflow records.
7. **Autopilot mode.** A stage skill called with the argument `autopilot`, or run in an experiment workspace (`ml_common.autopilot_mode`), asks nothing: it takes its recommended answers, records them as assumptions with `source: autopilot draft`, and never confirms a framing. The spec keeps assumptions as plain text, so `nb_common.write_stage` stores such a mapping as text that starts with `Autopilot draft: ` (`ml_common.assumption_text`), which is how the checkpoint page finds them.

### The autopilot

`ml-autopilot` is the driver: it loops on `autopilot.py next`, calls each stage skill in autopilot mode, and stops at checkpoints (`business`, `framing`, and before every opening of the test set) with `docs/checkpoint.md` and a push notification. The person confirms the framings and agrees to every look at the test set; nothing else needs them.

After the first evaluation it runs rounds of experiments. A round is an ml-iterate iteration whose experiments do forward selection over `autopilot.backlog`: every open item is screened alone, the ones that pass the iteration bar are combined, each is taken out of the combination again and goes only when that passes the bar, and when the combination fails the items are added one at a time instead (greedy). Items upstream of the split change the folds, so each gets a round of its own.

1. `autopilot.py round open` opens the iteration (snapshot, version branch `ml/v2`, the version's MLflow parent run), then `workspace.py create` makes one git worktree per experiment on its own branch (`exp/v2-a`), at most `autopilot.settings.max_parallel` at a time. The worktree links `.venv` and `data/raw`, copies `data/processed`, `models`, and the snapshot, and gets the experiment's MLflow parent run (`V2-a`, a child of `V2`).
2. One `ml-experimenter` agent per experiment (`agents/ml-experimenter.md`, linked into `~/.claude/agents/`) makes its changes and reruns the changed stages in its worktree, carries the old champion forward, applies the iteration bar on CV, and writes `experiment_result.json` with the per fold scores. It never opens the test set.
3. `autopilot.py round advance` judges each phase from those per fold scores and adds the next phase's experiments (the combination, its ablations, a verify, or greedy steps) until the round settles on one experiment, or on none.
4. `autopilot.py round close` adopts the experiment the round settled on: it merges its branch into the version branch, copies its processed data and models back, and closes the iteration as kept, which promotes its runs in MLflow. Every other experiment is rejected in MLflow and its worktree removed; its branch and runs stay for the record. When none passed, the round closes as rolled back and the project is unchanged.

Rounds stop at `iterations.bar.max_iterations`, when a round adopts nothing and the backlog has nothing left to try, or when the diagnosis finds nothing worth trying. A final look at the test set then needs the person again.

## Writing style

Simple language and full sentences, including in lists. Plain, objective headings. No dashes as punctuation. This applies to SKILL.md files, notebook text, generated documents, and replies.

## Shared files

The files in `_ml_shared/` are copied into every skill listed in `_ml_shared/skills.txt`:

| Canonical file | Copy in each skill |
|---|---|
| `_ml_shared/ml_common.py` | `scripts/ml_common.py` |
| `_ml_shared/nb_common.py` (building, running, and protecting notebooks; also copied into the A/B testing skills, where it uses `ab_common.py`) | `scripts/nb_common.py` |
| `_ml_shared/spec_fields.md` | `reference/spec_fields.md` |
| `_ml_shared/split_core.py` (the split engine, and `cv_splits` for reading the folds back) | `scripts/split_core.py`, only in ml-train-test-split and the modelling skills after it (`SPLIT_SKILLS` in `tools/sync_ml_shared.py`) |
| `_ml_shared/model_core.py` (metrics, model definitions, and cross validation) | `scripts/model_core.py`, only in ml-baseline-model, ml-model-training, and ml-offline-evaluation (`MODEL_SKILLS` in `tools/sync_ml_shared.py`) |
| `_ml_shared/tracking.py` (MLflow tracking: the address, the project's experiment, and one run per stage notebook) | `scripts/tracking.py` |
| `_ml_shared/eda_core.py` (the EDA engine) | `scripts/eda_core.py`, only in ml-eda-report and ml-data-discovery (`EDA_SKILLS` in `tools/sync_ml_shared.py`) |

Edit only the canonical files, then run `python3 tools/sync_ml_shared.py --fix` from the claude-skills folder. After editing `nb_common.py`, also run `python3 tools/sync_ab_shared.py --fix`. Without `--fix`, it reports copies that differ and exits with code 1, which makes it usable as a check before committing.

## Adding a new skill

1. Copy `_ml-skill-template/` to a folder named after the skill in the workflow table, and fill in the placeholders.
2. Add the folder name to `_ml_shared/skills.txt`, then run `python3 tools/sync_ml_shared.py --fix`.
3. Add its section fields to the template in `_ml_shared/ml_common.py` and to `_ml_shared/spec_fields.md`, if it needs more than an empty mapping.
4. Link it into Claude Code: `ln -s ~/Projects/claude-skills/<skill> ~/.claude/skills/<skill>`.
5. Test it on a fresh project made with ml-project-setup, after running the earlier stages.
