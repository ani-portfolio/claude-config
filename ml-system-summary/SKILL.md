---
name: ml-system-summary
description: Scan an ML system codebase and write a markdown summary of exactly what the code does (problem framing, data, features, feature store, training, registry, deployment, experimentation, inference, outputs, monitoring), using code as the source of truth, citing files for every claim, and listing every discrepancy between documentation and code, plus an Excalidraw system diagram exported to PNG and embedded in the markdown. Use when asked to summarize, document, explain, audit, or map an ML or data science codebase or pipeline.
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*) Bash(node ${CLAUDE_SKILL_DIR}/scripts/*) Bash(bash ${CLAUDE_SKILL_DIR}/scripts/setup.sh)
---

# ML System Summary

Produce an evidence based summary of an ML system codebase and a system diagram in a fixed layout.

Arguments (optional): `$ARGUMENTS` may name the repository path and an output folder. Defaults: the repository is the current working directory, and the output folder is `docs/ml_system_summary/` inside it.

## Outputs

Write these four files to the output folder:

1. `ml_system_summary.md`: the full written summary, following `templates/summary_template.md`.
2. `system_summary.json`: the structured content for the diagram, following `reference/schema.md`.
3. `system_diagram.excalidraw`: the editable diagram, built by the script.
4. `system_diagram.png`: the rendered diagram. The scene is embedded, so the PNG can be dragged into Excalidraw and edited.

## Ground rules

These rules apply for the whole task.

1. **Read only.** Never modify, move, or delete files in the repository. Write only inside the output folder.
2. **Static analysis only.** Do not run training, inference, notebooks, or project scripts. Do not connect to databases, feature stores, or cloud services.
3. **Code is the source of truth.** Describe what the code actually does, not what the documentation says it does. See "Source of truth" below for what counts as code and how to handle documentation.
4. **Evidence for every claim.** Every statement about the system cites a code file path, and a function, class, or config key where possible, for example `src/train/train.py` (`train_model`). A documentation file alone is never enough evidence for a Confirmed or Inferred status.
5. **Status for every section.** Mark each section as one of:
   * **Confirmed**: directly shown in code that runs, or in config that the code loads.
   * **Inferred**: suggested by code but not fully shown, such as a function name, a partial implementation, or a dependency that is installed but whose use is unclear. Say what the inference rests on.
   * **Documented only**: described in documentation, but no supporting code was found. Report it, but never present it as how the system works.
   * **Not found**: no evidence in code or documentation. Write "Not found in this codebase." Never describe a typical setup to fill the gap.
6. **Every discrepancy is reported.** Whenever documentation and code disagree, describe the system as the code has it, and add a row to the Documentation and Code Discrepancies section.
7. **No sensitive content.** Never copy secrets, credentials, connection strings, tokens, customer data, or sample records into the outputs. Name the config key instead of its value.

## Source of truth

**Code (authoritative):** source files reachable from an entry point, config files that the code loads, pipeline and DAG definitions, Dockerfiles, CI workflows, and infrastructure files such as Terraform, Helm, and Kubernetes manifests.

**Documentation (claims to verify):** README files, anything under `docs/`, design documents, docstrings, code comments, markdown cells in notebooks, and commit messages.

**Weaker code (report with care):**
* Notebooks often hold experiments that never reached production. Treat their code as Inferred unless a production entry point uses it.
* Dead code that no entry point reaches, commented out code, and unused config keys do not describe the running system. Mention them only when they explain a discrepancy.
* Tests show intended behaviour, not proof of production use. Cite them as supporting evidence only.

**Effective values:** when a value is set in several places, report the value the code actually uses. Follow the override order, for example a hard coded default, then a config file, then an environment variable, then a command line argument. Cite where the effective value comes from, and record a discrepancy if the documentation states a different value.

## Workflow

### Step 1: Check the export tools

Run `ls ${CLAUDE_SKILL_DIR}/scripts/node_modules >/dev/null 2>&1 || bash ${CLAUDE_SKILL_DIR}/scripts/setup.sh`. If setup fails, continue. The markdown and the `.excalidraw` file can still be produced, and Step 6 explains the fallback.

### Step 2: Inventory the repository

1. List tracked files with `git ls-files` (or Glob if it is not a git repository). Ignore `node_modules`, virtual environments, build output, and data files.
2. Skim README files and `docs/` only for orientation, such as which folders matter and what the entry points are called. Do not take any fact from them yet.
3. Read the code in this order, because each layer explains the next:
   1. Dependency files: `requirements*.txt`, `pyproject.toml`, `setup.py`, `environment.yml`, `package.json`. Libraries reveal the stack quickly.
   2. Container and CI files: `Dockerfile`, `docker-compose*`, `.github/workflows/`, `Jenkinsfile`, `cloudbuild.yaml`, `Makefile`. These show what actually gets built and run.
   3. Orchestration and infrastructure: DAG files, pipeline definitions, Terraform, Helm charts, Kubernetes manifests.
   4. Entry points (`main`, `__main__`, CLI commands, DAG tasks, API routes), then the code they call. Trace the call path, so you know which code is live and which is dead.
   5. Configuration that the code loads: YAML, TOML, JSON, Hydra or OmegaConf configs, `params.yaml`, and environment variables the code reads. Resolve effective values as described in "Source of truth".
   6. Notebooks last, and only to fill gaps.
4. Grep for the keyword signals in `reference/signals.md` to find where each section lives. A keyword hit is a lead, not evidence. Open the file and confirm before citing it.
5. For very large repositories (several hundred source files or more), you may delegate keyword sweeps to an Explore subagent. Read the key files yourself before writing claims about them.

### Step 2b: Check documentation against code

After the code is understood, read the documentation carefully: README files, `docs/`, design documents, and the docstrings and comments in the key files.

1. List each concrete claim the documentation makes about the system, such as a model type, label definition, metric, threshold, schedule, feature list, data source, deployment method, or fallback.
2. Check each claim against the code and give it one result:
   * **Matches**: the code does what the documentation says. No row is needed.
   * **Conflicts**: the code does something different. Add a row to the discrepancies table.
   * **Not implemented**: the documentation describes something with no supporting code. Add a row, and use the Documented only status for that item in the relevant section.
   * **Undocumented**: important behaviour in the code that the documentation never mentions, such as a filter on the training data, a clipping step, or a silent fallback. Add a row when a reader of the documentation would be misled without it.
3. Rate each row as High, Medium, or Low impact. High means a reader would misunderstand the model's behaviour, its outputs, or its risks, such as a different label window, a different model type, or a missing guardrail. Low means a naming or wording difference.
4. Keep the documentation's own words short in the table, at most one sentence, and always cite the file and line or heading.

### Step 3: Fill each section

Work through the sections in `templates/summary_template.md`. Points that deserve extra care:

1. **Label and splits.** Find exactly how the target is built (aggregation window, filters, censoring) and how data is split. Flag any leakage risk, such as random splits on time series data or features computed after the prediction time.
2. **Training and serving skew.** Compare the feature code used in training with the feature code used at inference. Note whether they share one implementation or duplicate logic, and list any differences.
3. **Fallback and guardrails.** Look for default values, try and except blocks around prediction calls, clipping or capping of outputs, and baseline models.
4. **Business framing.** Business KPI, guardrail metrics, and assumptions are often only in docs or comments. Mark them Documented only when that is their only source, and Not found when nothing mentions them. Business goals are the one area where documentation is often the only source, so report them, but label them clearly.
5. **Feedback loops.** Record a retraining trigger only when code or config shows one (drift alert wired to a retrain job, scheduled retraining). Record a behaviour feedback loop when predictions drive an action that changes future training data, and say whether the code accounts for it (holdouts, exploration, propensity logging).

### Step 4: Write `system_summary.json`

Follow `reference/schema.md`. Keep diagram bullets short, at most six per component, about ten words each. The markdown holds the detail. Bullets describe what the code does. Use `"status": "documented"` for components that only the documentation describes, and `"status": "missing"` for components that were not found. Set `discrepancy_count` to the number of High impact rows. The JSON and the markdown must agree.

### Step 5: Build the diagram

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/build_excalidraw.py <out>/system_summary.json <out>/system_diagram.excalidraw
```

### Step 6: Export the PNG

```bash
node ${CLAUDE_SKILL_DIR}/scripts/export_png.mjs <out>/system_diagram.excalidraw <out>/system_diagram.png
```

If export fails, rerun setup once. If it still fails, skip the PNG, replace the image line in the markdown with a link to the `.excalidraw` file plus the note "Open this file at excalidraw.com to view the diagram.", and tell the user what failed.

### Step 7: Review the diagram

Open the PNG with the Read tool and look at it. Fix any problem by shortening bullets in the JSON and rebuilding:

1. Text runs outside a box or overlaps another element.
2. A box is so tall that the diagram becomes hard to read.
3. The diagram disagrees with the markdown.

### Step 8: Write the markdown

Fill `templates/summary_template.md` and save it as `ml_system_summary.md`. Embed the diagram right after the summary card with `![System diagram](system_diagram.png)` and link the editable source below it.

### Step 9: Final check

Before finishing, confirm that:

1. Every section from the template is present, in order.
2. Every Confirmed or Inferred section cites at least one code file that exists. Spot check a few citations by opening the files.
3. No statement in the summary rests only on documentation unless it is labelled Documented only.
4. Every Not found section says so plainly and does not describe a typical setup.
5. Every conflict found in Step 2b appears in the Documentation and Code Discrepancies section, and the matching section of the summary describes the code's behaviour.
6. The summary card, the JSON, and the diagram agree.
7. Gaps and Open Questions lists everything that could not be confirmed.

Then tell the user where the files are, give a three to five sentence overview of the system, list the High impact discrepancies, and list the most important gaps.

## Writing style

1. Use simple language and full, concise sentences, including in bullet points.
2. Use plain, objective headings. Do not use dramatic or promotional wording.
3. Do not use dashes as punctuation in the prose.
4. Prefer specific names (functions, tables, metrics, thresholds) over general descriptions.

## Files in this skill

* `templates/summary_template.md`: section order and fields for the markdown.
* `reference/schema.md`: the JSON format the diagram builder reads.
* `reference/signals.md`: keywords that point to each section.
* `scripts/build_excalidraw.py`: builds the diagram in a fixed layout (Python standard library only).
* `scripts/export_png.mjs`: renders the PNG with Excalidraw's own exporter in a headless browser and embeds the scene.
* `scripts/setup.sh`: one time install of the export dependencies.
* `examples/`: a complete example JSON, diagram, and PNG for an LTV bidding system.
