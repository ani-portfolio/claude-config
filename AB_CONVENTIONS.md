# Conventions for the A/B Testing Skills

These rules apply to every skill that takes an A/B test from an empty folder to a readout for the decision owner. They follow `ML_CONVENTIONS.md` wherever the two workflows overlap, so a person who knows one knows the other. Follow them when changing an A/B skill or writing a new one. `_ab-skill-template/` is a starting point for a new skill.

## Naming and scope

1. Every skill in the A/B testing workflow is named with the `ab-` prefix, so typing `/ab-` lists them all. The folder name and the `name` in the SKILL.md frontmatter are the same.
2. These conventions, `_ab_shared/`, `_ab-skill-template/`, and `tools/sync_ab_shared.py` apply to the skills listed in `_ab_shared/skills.txt`.
3. **Helper skills** support the workflow without being a step. `ab-status` reports each step's state and the next skill, and never writes anything. Planned helpers: `ab-reconcile-notes` (align notes that disagree between steps), `ab-sequential-monitor` (safe looks at a running test), `ab-bayesian-analysis` (an alternative readout), and `ab-review` (a read only audit). `ab_common.consistency_checks` holds the fixed checks between steps.

## The workflow

| Step | Skill | Main output |
|---|---|---|
| 0. setup | ab-project-setup | Project folder, `.venv`, `pyproject.toml` and `uv.lock`, kernel, `ab_spec.yaml`, `notebooks/00_scratch.ipynb` |
| 1. framing | ab-test-framing | `notebooks/ab01_test_framing.ipynb`, the `framing` section, and `docs/test_plan.md` |
| 2. design (optional for finished tests) | ab-test-design | `notebooks/ab02_design.ipynb` and the `design` section |
| 3. validation | ab-data-validation | `notebooks/ab03_validation.ipynb`, the `data` and `validation` sections, and `data/processed/ab_units.parquet` |
| 4. analysis | ab-analysis | `notebooks/ab04_analysis.ipynb`, the `analysis` section, and `data/processed/ab_results.parquet` |
| 5. segments (optional) | ab-segment-analysis | `notebooks/ab05_segments.ipynb` and the `segments` section |
| 6. readout | ab-readout | `notebooks/ab06_readout.ipynb`, the `readout` section, and `docs/ab_readout.md` |

The order follows what each question needs: the hypothesis, metrics, and ship rule are agreed before anyone sees a result; the design fixes the error rates and the minimum effect; the data is trusted before any effect is estimated; and the readout only restates what the analysis found. The steps are defined once, in `WORKFLOW` in `_ab_shared/ab_common.py`. The optional steps are in `OPTIONAL_STEPS`: they are offered in order, but they stop being the next step once a later step has started, and `--skip --reason` marks them skipped.

**The analysis gate.** The analysis, segments, and readout start by running `ab_common.py gate`. It exits with code 1 while the framing is not confirmed, the data has not been validated, or the validation failed without an override. The skill then names the step to run first and stops, unless the user explicitly chooses to continue, which it records in `decisions`.

**Results come last.** No skill before ab-analysis compares a metric between variants. ab-data-validation looks at counts and the observed split, labels, column types, missing values, value ranges, duplicates, metrics pooled over all variants, and columns recorded before assignment only. A choice changed after the results were seen (the primary metric, the sidedness, alpha, the ship rule) is recorded in `decisions` with the reason, and the readout shows it.

**The planned split.** The sample ratio check compares the data against `framing.variants[].planned_share`. It is never filled from the observed counts. If it is not known, the check is skipped with a warning and an open question.

**Column timing.** `data.pre_treatment_cols` and `data.post_treatment_cols` record which columns were recorded before and after assignment. Only columns recorded before assignment are used for balance checks, CUPED, and segments. A column with unknown timing counts as recorded after assignment. ab-segment-analysis refuses columns recorded after assignment unless the user chooses otherwise with a reason, and labels the result descriptive.

## The shared spec

The rules of `ML_CONVENTIONS.md` apply, with `ab_spec.yaml` and `ab_common.py` in place of `ml_spec.yaml` and `ml_common.py`:

1. `ab_spec.yaml` sits at the project root. Every skill reads it before asking anything and skips questions it already answers.
2. Each skill writes only its own section, through `ab_common.py` (`update_section`, `merge`, `fill`) or `nb_common.write_stage` from its notebook. ab-data-validation writes two sections: `data` and `validation`.
3. `assumptions`, `open_questions`, and `decisions` are shared lists. Append, never replace. Each open question names the stage that needs the answer, and a skill marks it resolved when its work settles it (`resolved_questions` in the answers).
4. The spec never holds raw rows. Counts, rates, test statistics, and the labels of the group column are fine.
5. Field meanings are in `_ab_shared/spec_fields.md`. Add new fields there in the same change that adds them to `template()` in `ab_common.py`.

**Sharing a folder with the ML workflow.** An A/B test can live in a folder set up by ml-project-setup. `ab_common.find_project_root` looks for `ab_spec.yaml`, then `ml_spec.yaml`, then `.venv`, then `.git`. The two specs are separate files, the folders and the kernel are shared, and the A/B notebooks start with `ab` (`ab01_test_framing.ipynb`) so they never collide with the ML notebooks.

## Project root, paths, and Python

The same as `ML_CONVENTIONS.md`: folders come from `paths` in the spec (`ab_common.project_paths`), data files written by notebooks are parquet, `data_raw` is read only, scripts run with the project's `.venv/bin/python` (only ab-project-setup runs with `python3`), and packages are managed with uv. A skill that needs a package the project lacks runs `ab_common.py need <package>`, shows the user the `uv add` command it prints, and runs it only after they agree.

ab-project-setup runs `ml-project-setup/scripts/setup_project.py` with the A/B versions of the spec, the CLAUDE.md section (`## A/B testing workflow`), the README, and the starter notebook, and adds statsmodels. So ml-project-setup must be installed next to it.

## Statistics

All statistics live in `_ab_shared/ab_core.py`, which uses numpy, scipy, and pandas only. Notebooks call it and never reimplement a test. Defaults:

1. Binary metrics use the pooled two proportion z test for the p value and the unpooled standard error for the interval. Continuous and count metrics use Welch's t test. Relative lift intervals use the delta method. Intervals are two sided at `1 - alpha`, even for one sided tests.
2. The sample ratio check is a chi-square test at p below 0.001.
3. Secondary metrics, and the primary metric across several treatments, use Holm adjusted p values. Segment heterogeneity across columns uses Benjamini-Hochberg.
4. Every result gets a plain label from `ab_core.label_result`: significant and practically meaningful, significant but not clearing the minimum effect, inconclusive, no meaningful effect, or harm. A result that is not significant is never called "no effect" unless its interval rules out the minimum effect.
5. New functions are checked against statsmodels or a published example before they are used. `ab_core.fmt_p` formats p values for reports.

## Notebooks

The notebook rules of `ML_CONVENTIONS.md` apply unchanged: one numbered notebook per step (`ab_common.notebook_name(stage)`), the setup cell from `nb_common.setup_cells` (which loads `ab_spec.yaml` when `ab_common.py` sits next to `nb_common.py`), markdown for reasoning, *Decision* cells for answers, check cells that compute from the data, a last cell that writes the spec and records the run, and edit protection with exit code 2 and the three choices (`--overwrite`, `--new-version`, or `nb_common.py run`). The only documents in `docs` are the test plan and the readout, both written from the spec by the notebooks.

## SKILL.md structure

The same order as `ML_CONVENTIONS.md`: frontmatter (`name`, a `description` that says what it does, which step it is, and when to use it, and `allowed-tools`), purpose and workflow sentence, **Ground rules**, **Step 1: Find the project and the spec** (plus `gate` for the analysis, segments, and readout), the skill's own steps, asking one question at a time (see `ML_CONVENTIONS.md`), and what happens without AskUserQuestion, build and run the notebook, and **Report back** with Files written, Summary, Problems, Open questions, and Next step.

## Writing style

Simple language and full sentences, including in lists. Plain, objective headings. No dashes as punctuation. This applies to SKILL.md files, notebook text, generated documents, and replies.

## Experiment tracking

The rules of `ML_CONVENTIONS.md` ("Experiment tracking") apply: one shared MLflow server, the spec decides and MLflow records, and tracking never blocks the work. Each A/B project has one experiment, `ab/<project>`, made by ab-project-setup. Every step from the framing to the readout logs one run, read from the section its notebook just wrote, with the charts, the spec section, and the executed notebook:

* **framing:** the plan before any result, with the hypothesis, the ship rule, and `preregistered = true` once confirmed. The run time is the record that the plan came first.
* **design:** alpha, power, sidedness, the minimum effect, and the sample size and duration.
* **validation:** the result, the sample ratio test, unit counts, the column mapping, and the balance table.
* **analysis:** the primary result as metrics, and one child run per comparison (primary, secondary, guardrail, for each treatment) with its effect, interval, p values, and label.
* **segments:** the heterogeneity test per column, and the effect per segment as a table.
* **readout:** the recommendation, the ship rule, the business impact, and the primary result again, so tests can be compared from their readouts.

No raw rows are logged. `ab-status` shows the experiment link and the steps with no logged run.

## Shared files

| Canonical file | Copy in each skill |
|---|---|
| `_ab_shared/ab_common.py` | `scripts/ab_common.py` |
| `_ml_shared/nb_common.py` (shared with the ML skills) | `scripts/nb_common.py` |
| `_ab_shared/spec_fields.md` | `reference/spec_fields.md` |
| `_ml_shared/tracking.py` (MLflow tracking, shared with the ML skills; it uses `ab_common` when it sits next to it) | `scripts/tracking.py` |
| `_ab_shared/ab_core.py` (the statistics) | `scripts/ab_core.py`, only in the skills in `STATS_SKILLS` in `tools/sync_ab_shared.py` |

Edit only the canonical files, then run `python3 tools/sync_ab_shared.py --fix` from the claude-skills folder. After editing `_ml_shared/nb_common.py` or `_ml_shared/tracking.py`, run both `tools/sync_ml_shared.py --fix` and `tools/sync_ab_shared.py --fix`, and check one notebook from each workflow.

## Adding a new skill

1. Copy `_ab-skill-template/` to a folder named after the skill, and fill in the placeholders.
2. Add the folder name to `_ab_shared/skills.txt` (and to `STATS_SKILLS` if it computes statistics), then run `python3 tools/sync_ab_shared.py --fix`.
3. For a new step, add it to `WORKFLOW`, `NOTEBOOK_BASE`, `step_state`, and `template()` in `_ab_shared/ab_common.py`, to `_ab_shared/spec_fields.md`, and to `CHAIN` and `expected_outputs` in `ab-status/scripts/status.py`.
4. Link it into Claude Code: `ln -s ~/Projects/claude-skills/<skill> ~/.claude/skills/<skill>`.
5. Test it on a fresh project made with ab-project-setup, after running the earlier steps.
