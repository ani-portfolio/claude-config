#!/usr/bin/env python3
"""Write notebooks/03_problem_framing.ipynb from the ML framing answers, and run it once.

The notebook is the working document for the ML problem framing. It shows the business framing and the
discovery findings it builds on, holds the decisions in editable cells tagged "decision" (target, unit of
prediction, prediction time, success, columns to exclude), and checks them against the data in short pandas
cells: the target's class balance, whether the unit of prediction is unique, the prediction time column, how
much labelled history the outcome window leaves, and whether risky columns are excluded. Running it writes the
framing and data sections of ml_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml                 # build and run; refuses to replace an edited notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write 03_problem_framing_v2.ipynb instead
    make_notebook.py --from-spec                  # rebuild from the framing already in the spec

answers.yaml (see templates/framing_answers.example.yaml): status, framing, data, reasoning, checks, risks,
assumptions, open_questions, decisions, resolved_questions.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ml-problem-framing"
STAGE = "framing"

SECTIONS = [
    ("target", "Target", ["task_type", "target"],
     "The exact rule for the outcome, with its window. When `exists_in_data` is false, `construction` says how to "
     "build it. task_type is one of binary_classification, multiclass_classification, regression, ranking, "
     "forecasting, or other."),
    ("unit", "Unit of Prediction and Population", ["unit_of_prediction", "entity_id_cols", "population"],
     "What one row represents when a prediction is made, the columns that identify the entity, and who is scored."),
    ("prediction_time", "Prediction Time", ["prediction_time"],
     "When the prediction is made, the column that marks that moment, and the latest data a feature may use."),
    ("success", "Success Criteria", ["success"],
     "The offline metric that decides whether V1 is good enough, the bar as a number against a reference, and "
     "metrics that must not get worse. `v1_bar` is the bar in words, for people. `v1_criteria` is the same bar as "
     "checks a script can apply, each with an `id`, a `kind`, and `on` (cv or test): `beats_model` (`model`, "
     "optional `margin`), `pct_over_reference` (`pct`), `threshold` (`metric`, `op`, `value`), `check` (`function` "
     "in `src/v1_checks.py`, for project guardrails), or `manual` (`text`, judged by a person). Model training "
     "judges the cv criteria and offline evaluation the test criteria."),
]


def _default_data_path(spec, root):
    disc = spec.get("discovery") or {}
    main = disc.get("main_source")
    for s in disc.get("sources") or []:
        if s.get("name") == main:
            return s.get("path")
    for s in (spec.get("data") or {}).get("sources") or []:
        if isinstance(s, dict) and s.get("path"):
            return s["path"]
    return None


def _context(spec):
    """Markdown describing the business framing and discovery findings this framing builds on."""
    b, d = spec.get("business") or {}, spec.get("discovery") or {}
    lines = []
    if b and b.get("mode") == "exploratory":
        acts = "; ".join(f"{a.get('answer')} then {a.get('action')}" for a in b.get("answer_actions") or []
                         if isinstance(a, dict))
        lines += ["* **Mode:** exploratory, so V1 is compared with the best simple baseline, not a current approach",
                  f"* **Learning question:** {b.get('learning_question') or 'not set'}",
                  f"* **What each answer leads to:** {acts or 'not set'}",
                  f"* **Outcome in business terms:** {b.get('outcome') or 'not set'}",
                  f"* **Output use:** {b.get('output_use') or 'not set'}. **V1 bar intent:** {b.get('v1_bar_intent') or 'not set'}"]
    elif b:
        lines += [f"* **Goal:** {b.get('business_goal') or 'not set'}",
                  f"* **Decision:** {b.get('decision_supported') or 'not set'}"
                  + (f" (capacity: {b['capacity']})" if b.get("capacity") else ""),
                  f"* **Outcome in business terms:** {b.get('outcome') or 'not set'}",
                  f"* **Output use:** {b.get('output_use') or 'not set'}. **V1 bar intent:** {b.get('v1_bar_intent') or 'not set'}",
                  f"* **Current approach:** {b.get('current_approach') or 'not set'}"]
    if d:
        main = next((s for s in d.get("sources") or [] if s.get("name") == d.get("main_source")), {})
        g = (main.get("grain") or {}).get("columns") or []
        lines += [f"* **Main source:** {d.get('main_source')}, {main.get('rows', 0):,} rows, grain "
                  + (" + ".join(f"`{c}`" for c in g) if g else "unclear")
                  + (f", {main['history_days']:,} days of history in `{main.get('main_date')}`" if main.get("history_days") else "")]
        cands = (d.get("framing_hints") or {}).get("candidate_targets") or []
        if cands:
            lines.append("* **Candidate targets from discovery:** " + ", ".join(f"`{c['column']}`" for c in cands))
    return "\n".join(lines) or "The business framing and data discovery have not run yet."


def build(answers, spec, root, scripts_dir, nb_name):
    fr = dict(answers.get("framing") or {})
    fr.pop("status", None)
    data = dict(answers.get("data") or {})
    reasoning = answers.get("reasoning") or {}
    status = answers.get("status") or "draft"
    checks = answers.get("checks") or {}
    data_path = checks.get("data_path") or _default_data_path(spec, root)

    cells = nb_common.setup_cells(
        STAGE, "ML Problem Framing",
        "The exact ML problem: what is predicted, for which unit, at what moment, and how success is measured, "
        "built on the business framing (`01_business_framing.ipynb`) and the data discovery "
        "(`02_data_discovery.ipynb`).\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the framing. Edit them, then choose Run All. "
        "The *Checks Against the Data* cells test the framing on the data, and the last cells write it to "
        "`ml_spec.yaml`. Raw rows are never shown here.",
        scripts_dir, nb_name)
    cells.append(code('''
import re

import pandas as pd
from IPython.display import display

from nb_common import load_table as load_data
'''))
    cells.append(md("## What This Builds On\n\n" + _context(spec) + "\n\nThe cell below prints the same from the spec, "
                    "so it stays current when the earlier notebooks change."))
    cells.append(code('''
business, discovery = spec.get("business") or {}, spec.get("discovery") or {}
keys = (("mode", "learning_question", "answer_actions") if business.get("mode") == "exploratory" else
        ("business_goal", "decision_supported", "capacity", "current_approach"))
for key in keys + ("outcome", "output_use", "v1_bar_intent"):
    print(f"{key}: {business.get(key)}")
print(f"main source: {discovery.get('main_source')}")
print("candidate targets: " + ", ".join(c["column"] for c in (discovery.get("framing_hints") or {}).get("candidate_targets", [])))
'''))
    cells.append(code("framing = {}  # filled in by the decision cells below"))
    for key, heading, fields, hint in SECTIONS:
        text = f"## {heading}\n\n" + ((reasoning.get(key) or "").strip() + "\n\n" if reasoning.get(key) else "") + f"*{hint}*"
        cells.append(md(text))
        body = ",\n".join(f"    {f!r}: {lit(fr.get(f), indent=4)}" for f in fields)
        cells.append(code(f"# Decision\nframing.update({{\n{body},\n}})", decision=True))

    cells.append(md("## Data Settings and Excluded Columns\n\n" + ((reasoning.get("exclusions") or "").strip() + "\n\n"
                    if reasoning.get("exclusions") else "") + "*The main date column, other ID columns, and the columns "
                    "that must never be features, each with a reason: recorded after the outcome, personal data, an "
                    "identifier, or timing unknown.*"))
    cells.append(code(
        "# Decision\n"
        f"data_settings = {{\n"
        f"    'date_col': {data.get('date_col')!r},\n"
        f"    'id_cols': {lit(data.get('id_cols') or [], indent=4)},\n"
        f"    'excluded_cols': {lit(data.get('excluded_cols') or [], indent=4)},\n"
        "}", decision=True))

    cells.append(md("## Checks Against the Data\n\nEach check loads only what it needs and prints counts, never rows. "
                    "Change `DATA_PATH` if the framing is built on another file, and `TEST_DAYS` to the length of "
                    "the test period you expect to hold out."))
    cells.append(code(
        "# Decision\n"
        f"DATA_PATH = {data_path!r}\n"
        f"TEST_DAYS = {int(checks.get('test_days') or 90)}", decision=True))
    cells.append(code('''
df = load_data(DATA_PATH)
target = framing.get("target") or {}
pt = framing.get("prediction_time") or {}
ids = [c for c in framing.get("entity_id_cols") or []]
print(f"{DATA_PATH}: {len(df):,} rows and {df.shape[1]} columns")
'''))
    cells.append(md("### 1. Target\n\nThe target column exists, and its class balance (for classification) or "
                    "spread (for regression)."))
    cells.append(code('''
name = target.get("name")
if target.get("exists_in_data") is False:
    print(f"The target is built, not a column: {target.get('construction') or 'no construction rule yet'}")
elif not name or name not in df.columns:
    print(f"Problem: the target column {name!r} is not in the data.")
elif "classification" in str(framing.get("task_type")):
    shares = df[name].value_counts(normalize=True, dropna=False)
    print(f"{name}: {df[name].nunique()} classes. Shares: {shares.round(4).to_dict()}")
    print(f"Missing targets: {df[name].isna().mean():.2%}")
    if framing.get("task_type") == "binary_classification" and df[name].nunique() != 2:
        print("Problem: binary classification needs exactly two classes.")
else:
    print(df[name].describe().round(3).to_string())
'''))
    cells.append(md("### 2. Unit of Prediction\n\nRows should be unique on the entity ID columns plus the prediction "
                    "time column. If they are not, either aggregate them or the unit is finer than the framing says."))
    cells.append(code('''
key = [c for c in ids + [pt.get("column")] if c]
missing_cols = [c for c in key if c not in df.columns]
if not key:
    print("No entity ID or prediction time column is set yet.")
elif missing_cols:
    print(f"Problem: not in the data: {missing_cols}")
else:
    dup = df.duplicated(subset=key, keep=False).sum()
    print(f"Rows sharing their {' + '.join(key)} with another row: {dup:,} of {len(df):,}")
    for c in ids:
        print(f"{c}: {df[c].nunique():,} values, {len(df) / max(df[c].nunique(), 1):.1f} rows each on average")
'''))
    cells.append(md("### 3. Prediction Time and History\n\nThe prediction time column's range, and how much history "
                    "has complete labels once the outcome window (or the label delay, if longer) is taken off, and the "
                    "test period is held out."))
    cells.append(code('''
def to_days(text):
    """Days in a phrase such as '60 days after the scoring date' or '8 weeks'. None when there is no number."""
    m = re.search(r"(\\d+(?:\\.\\d+)?)\\s*(day|week|month|year)", str(text or ""), re.I)
    if not m:
        return None
    return float(m.group(1)) * {"day": 1, "week": 7, "month": 30.4, "year": 365}[m.group(2).lower()]


col = pt.get("column")
if not col or col not in df.columns:
    print(f"Problem: the prediction time column {col!r} is not set or not in the data.")
else:
    d = pd.to_datetime(df[col], errors="coerce", format="mixed")
    history = (d.max() - d.min()).days
    window, delay = to_days(target.get("outcome_window")), to_days(target.get("label_delay"))
    # The label is known once the outcome window has passed, or later when there is a label delay. The delay is
    # counted from the prediction, so it includes the window: take the larger of the two.
    labelled = history - max(window or 0, delay or 0)
    print(f"{col}: {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}, {history:,} days. Unreadable dates: {d.isna().mean():.1%}")
    print(f"Outcome window: {window or 'not set'} days. Label delay: {delay or 'not set'} days.")
    print(f"Labelled history: about {labelled:,.0f} days. After a {TEST_DAYS} day test period: about {labelled - TEST_DAYS:,.0f} days for training.")
    if labelled - TEST_DAYS < 90:
        print("Problem: less than about three months is left for training. Consider a shorter window or test period, or more history.")
'''))
    cells.append(md("### 4. Excluded and Risky Columns\n\nExcluded columns should exist. Columns the data dictionary "
                    "marks as possibly recorded after the outcome, or with unknown timing, should be excluded unless "
                    "their timing has been confirmed."))
    cells.append(code('''
excluded = {e["column"] for e in data_settings.get("excluded_cols") or []}
not_found = sorted(excluded - set(df.columns))
print(f"Excluded columns: {len(excluded)}. Not in this file: {not_found or 'none'}")

dict_path = os.path.join(PATHS["docs"], "data_dictionary.yaml")
if os.path.exists(dict_path):
    dictionary = ml_common.load(dict_path)
    main_cols = next((b["columns"] for b in dictionary["sources"].values()
                      if os.path.normpath(b["path"]) == os.path.normpath(DATA_PATH)), [])
    risky = [e["column"] for e in main_cols
             if e["timing"] in ("possibly_after_outcome", "unknown") and e["column"] not in excluded
             and e["column"] != target.get("name")]
    print(f"Kept but possibly recorded after the outcome, or with unknown timing: {risky or 'none'}")
else:
    print("No data dictionary yet. Run 02_data_discovery.ipynb first.")
'''))
    cells.append(md("### 5. Metric Against the Class Balance\n\nAccuracy is misleading when one class is rare."))
    cells.append(code('''
metric = str((framing.get("success") or {}).get("primary_metric") or "")
name = target.get("name")
if name in df.columns and "classification" in str(framing.get("task_type")):
    smallest = df[name].value_counts(normalize=True).min()
    print(f"Smallest class: {smallest:.1%}. Primary metric: {metric or 'not set'}")
    if smallest < 0.05 and "accuracy" in metric.lower():
        print("Problem: the positive class is rare, so prefer PR AUC or recall at k over accuracy.")
else:
    print(f"Primary metric: {metric or 'not set'}")
'''))
    cells.append(md("## Risks, Assumptions, Open Questions, and Decisions\n\nRisks found when the framing was "
                    "checked, and how each was resolved. `resolved_questions` closes open questions in the spec that "
                    "this framing answers, by ID."))
    cells.append(code(
        "# Decision\n"
        f"risks = {lit(list(answers.get('risks') or []), indent=0)}\n\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []), indent=0)}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []), indent=0)}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []), indent=0)}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []), indent=0)}", decision=True))
    cells.append(md("## Status\n\n`confirmed` once you agree with the framing and the checks above show no problem "
                    "you have not accepted. A confirmed framing, followed by `/ml-eda-report`, unlocks modelling."))
    cells.append(code(f"# Decision\nSTATUS = {status!r}", decision=True))
    cells.append(md("## Write to the Spec\n\nWrites the `framing` and `data` sections of `ml_spec.yaml`, closes the "
                    "resolved questions, and records this run."))
    cells.append(code(f'''
assert STATUS in ("draft", "confirmed"), "STATUS must be draft or confirmed"
for e in data_settings.get("excluded_cols") or []:
    assert e.get("column") and e.get("reason"), "every excluded column needs a column and a reason"

result = nb_common.write_stage(SPEC_PATH, "framing", {{**framing, "risks": risks, "status": STATUS}}, "{SKILL}",
                               assumptions, open_questions, decisions)
nb_common.write_stage(SPEC_PATH, "data", data_settings, "{SKILL}")
not_found = nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
if not_found:
    print("These open question IDs were not found: " + ", ".join(map(str, not_found)))
nb_common.show_result(*result, STATUS)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
'''))
    cells.append(md("## Next Step\n\nRun `/ml-eda-report`. It takes the target, date, and ID columns from the spec, "
                    "analyses the target, checks for leakage, and checks the data against this framing. Check progress "
                    "with `/ml-status`."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def answers_from_spec(spec):
    fr = dict(spec.get("framing") or {})
    return {"status": fr.pop("status", "draft"), "risks": fr.pop("risks", []), "framing": fr,
            "data": {k: (spec.get("data") or {}).get(k) for k in ("date_col", "id_cols", "excluded_cols")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--from-spec", action="store_true", help="Build from the framing already in the spec")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()

    spec_path = ml_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ml_common.load_or_create(spec_path)
    if a.from_spec:
        answers = answers_from_spec(spec)
    elif a.answers:
        answers = ml_common.load(a.answers)
    else:
        ap.error("give an answers file or --from-spec")
    if not os.path.exists(spec_path):
        ml_common.save(spec_path, spec)
    nb_name = ml_common.notebook_name(STAGE)
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    nb = build(answers, spec, root, HERE, nb_name)
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    if res["status"] == "edited":
        print(f"Not written: {ml_common.relpath(path, root)} was edited after the skill wrote it. Rerun with "
              f"--overwrite to replace it, or --new-version to write {os.path.basename(res['next_version'])}.")
        sys.exit(2)
    print(f"Wrote notebook: {ml_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if not res["executed"]:
        print("Spec: not updated. Open the notebook, fix the cell that failed, and choose Run All.")
        sys.exit(1 if res["error"] else 0)
    fr = ml_common.load(spec_path).get("framing") or {}
    print(f"Spec: framing written, status {fr.get('status')}.")
    crit = (fr.get("success") or {}).get("v1_criteria") or []
    for p in ml_common.v1_problems(crit):
        print(f"V1 criteria problem: {p}")
    if any(isinstance(c, dict) and c.get("kind") == "check" for c in crit):
        target = os.path.join(ml_common.project_paths(spec, root)["src"], "v1_checks.py")
        if not os.path.exists(target):   # created once from the template, then it belongs to the user
            import shutil
            shutil.copy(os.path.join(HERE, "..", "templates", "v1_checks.py"), target)
            print(f"Created {ml_common.relpath(target, root)} from the template. Write the check functions there.")
        missing = [c["function"] for c in crit if isinstance(c, dict) and c.get("kind") == "check"
                   and f"def {c.get('function')}(" not in open(target).read()]
        if missing:
            print("V1 check functions still to write in src/v1_checks.py: " + ", ".join(missing))
    print("Check results (also in the notebook):")
    import json
    nb_out = nb_common.read(res["path"])
    for c in nb_out["cells"]:
        if c["cell_type"] == "code" and "".join(c["source"]).lstrip().startswith(("name = target", "key = [", "def to_days", "excluded = {", "metric = ")):
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    for line in "".join(o["text"]).strip().splitlines():
                        print("  " + line)


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "framing", "ml-problem-framing")
