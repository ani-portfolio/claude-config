#!/usr/bin/env python3
"""Write notebooks/05_split.ipynb from the split answers, and run it once.

The notebook is the record of the split. Markdown cells say why the strategy, the test period, the gap, and the
CV folds were chosen. Decision cells hold the data path, the target, and the split settings, which can be
edited. The other cells make the split with split_core.py and check it in short pandas code: the size and
class balance of each set, the CV folds, a timeline of the folds, and leakage checks (copies of test rows in
training, entities on both sides, and the gap). Running it writes train.parquet, test.parquet, cv_folds.parquet
(and gap.parquet for a time split) to the processed data folder, and the split section of ml_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py --suggest                    # facts and recommended settings as an answers YAML, no data values
    make_notebook.py answers.yaml                 # build and run; refuses to replace an edited notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write 05_split_v2.ipynb instead

answers.yaml (see templates/split_answers.example.yaml): data_path, target, task_type, target_code, split,
reasoning, risks, assumptions, open_questions, decisions, resolved_questions.
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

SKILL = "ml-train-test-split"
STAGE = "split"

SETTING_HELP = (
    "*Settings:* `strategy` is `random`, `group`, or `time`. `test_size` is the share held out for the random and "
    "group strategies. `test_days` is the length of the test period, and `gap_days` the days left out before it "
    "(and before each validation window), for the time strategy. `stratify` keeps the class balance equal across "
    "sets. `group_col` keeps every row of an entity on one side (group), or is used to report overlap (time). "
    "`cv` sets the folds: `n_folds`, and for time `valid_days` (length of each validation window), `window` "
    "(`expanding` uses all earlier data, `sliding` only the last `train_days`). `duplicates` is `keep_together`, "
    "`drop`, or `ignore`. `missing_target` is `drop` or `error`.")

STRATEGY_TEXT = {
    "random": "Rows are assigned to train and test at random. Cross validation on the training set uses k folds.",
    "group": "All rows of an entity land on the same side, in the test split and in every CV fold, so the model "
             "is never scored on an entity it trained on.",
    "time": "The latest period is the test set, as in production, where the model predicts the future from the "
            "past. Cross validation uses rolling backtest folds: each fold trains on the data before its "
            "validation window, with the same gap, and the windows move forward in time.",
}


def default_data_path(spec):
    eda = spec.get("eda") or {}
    if eda.get("source"):
        return eda["source"]
    disc = spec.get("discovery") or {}
    for s in disc.get("sources") or []:
        if s.get("name") == disc.get("main_source"):
            return s.get("path")
    for s in (spec.get("data") or {}).get("sources") or []:
        if isinstance(s, dict) and s.get("path"):
            return s["path"]
    return None


def _context(spec):
    fr = spec.get("framing") or {}
    tgt = fr.get("target") or {}
    pt = fr.get("prediction_time") or {}
    eda = spec.get("eda") or {}
    lines = [
        f"* **Task:** {fr.get('task_type') or 'not set'}. **Target:** `{tgt.get('name')}`, {tgt.get('definition') or 'no definition yet'}",
        f"* **Unit of prediction:** {fr.get('unit_of_prediction') or 'not set'}. **Entity:** "
        + (", ".join(f"`{c}`" for c in fr.get("entity_id_cols") or []) or "none"),
        f"* **Prediction time:** {pt.get('description') or 'not set'}"
        + (f", column `{pt['column']}`" if pt.get("column") else "")
        + (f", horizon {pt['horizon']}" if pt.get("horizon") else ""),
        f"* **Outcome window:** {tgt.get('outcome_window') or 'not set'}. **Label delay:** {tgt.get('label_delay') or 'not set'}",
    ]
    if eda:
        g = (eda.get("grain") or {}).get("columns") or []
        lines.append(f"* **From the EDA:** grain " + (" + ".join(f"`{c}`" for c in g) if g else "unclear")
                     + f", {eda.get('duplicate_rows') or 0:,} exact duplicate rows")
    return "\n".join(lines)


def build(answers, spec, scripts_dir, nb_name):
    fr = spec.get("framing") or {}
    split = answers.get("split") or {}
    reasoning = answers.get("reasoning") or {}
    target = answers.get("target") or (fr.get("target") or {}).get("name")
    task = answers.get("task_type") or fr.get("task_type")
    data_path = answers.get("data_path") or default_data_path(spec)
    target_code = (answers.get("target_code") or "").rstrip()
    strategy = split.get("strategy", "random")

    cells = nb_common.setup_cells(
        STAGE, "Train, Validation, and Test Split",
        "How the data is divided into the final test set, which is used once at the end, and the cross validation "
        "folds on the training set, which later stages use to compare and tune models.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the data, the target, and the split "
        "settings. Edit them, then choose Run All. The checks below show whether the split is sound, and the last "
        "cells save the sets as parquet files and write the `split` section of `ml_spec.yaml`. Raw rows are never "
        "shown here.",
        scripts_dir, nb_name)
    cells.append(code('''
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import split_core
from nb_common import load_table as load_data
'''))
    cells.append(md("## What This Builds On\n\n" + _context(spec) + "\n\nThe cell below prints the same from the spec, "
                    "and whether modelling may start (the framing is confirmed and the targeted EDA has run)."))
    cells.append(code('''
framing = spec.get("framing") or {}
target_spec = framing.get("target") or {}
print(f"task: {framing.get('task_type')}, target: {target_spec.get('name')}, entity: {framing.get('entity_id_cols')}")
print(f"prediction time: {(framing.get('prediction_time') or {}).get('column')}, outcome window: "
      f"{target_spec.get('outcome_window')}, label delay: {target_spec.get('label_delay')}")
blockers = ml_common.modelling_blockers(spec)
print("Modelling may start." if not blockers else "Not ready for modelling: " + "; ".join(blockers))
'''))

    body = target_code or '    """The target is a column in the data, so nothing is built here."""\n    return df'
    cells.append(md("## Data and Target\n\n" + ((reasoning.get("target") or "").strip() + "\n\n" if reasoning.get("target") else "")
                    + "*`build_target` builds the target when it is not a column yet, following "
                    "`framing.target.construction`. It must only use information known at the outcome, and it runs "
                    "before the split, so every set gets the same rule.*"))
    cells.append(code(
        "# Decision\n"
        f"DATA_PATH = {data_path!r}\n"
        f"TARGET = {target!r}\n"
        f"TASK_TYPE = {task!r}\n\n\n"
        f"def build_target(df):\n{body}\n", decision=True))
    cells.append(code('''
df = build_target(load_data(DATA_PATH))
print(f"{DATA_PATH}: {len(df):,} rows and {df.shape[1]} columns")
assert TARGET in df.columns, f"The target {TARGET!r} is not in the data. Fill in build_target above."
'''))

    text = "## Split Settings\n\n" + STRATEGY_TEXT.get(strategy, "") + "\n\n"
    for key in ("strategy", "test", "cv", "duplicates"):
        if reasoning.get(key):
            text += reasoning[key].strip() + "\n\n"
    cells.append(md(text + SETTING_HELP))
    cells.append(code(f"# Decision\nSPLIT = {lit(split, indent=0)}", decision=True))

    cells.append(md("## Prepare the Data\n\nAdds a `row_id` column (the row's position in the original file, so every "
                    "row can be traced back), drops rows without a target, and finds exact duplicate rows."))
    cells.append(code('''
SPLIT = split_core.settings(SPLIT)
data, prep, duplicate_key = split_core.prepare(df, TARGET, SPLIT)
print(f"Rows in: {prep['rows_in']:,}. Rows without a target: {prep['missing_target']:,} (dropped).")
print(f"Exact duplicate rows: {prep['duplicate_rows']:,}. Handling: {SPLIT['duplicates']}"
      + (f", {prep['duplicates_dropped']:,} dropped" if prep["duplicates_dropped"] else ""))
print(f"Rows to split: {prep['rows_out']:,}")
'''))
    cells.append(md("## Make the Split"))
    cells.append(code('''
result = split_core.make_split(data, TARGET, TASK_TYPE, SPLIT, duplicate_key)
train, test, gap, folds = result["train"], result["test"], result["gap"], result["folds"]
print(f"Strategy: {SPLIT['strategy']}. CV method: {result['cv_method']}, {len(folds)} folds.")
print(f"Train: {len(train):,} rows. Test: {len(test):,} rows." + (f" Gap: {len(gap):,} rows." if len(gap) else ""))
if result["info"].get("unreadable_dates"):
    print(f"Rows dropped because the date could not be read: {result['info']['unreadable_dates']:,}")
'''))

    cells.append(md("## Checks\n\n### Size and Balance of Each Set\n\nEvery set should have the same class balance "
                    "(or a similar target mean). For a time split, the dates show the order of the sets."))
    cells.append(code('''
classification = split_core.is_classification(TASK_TYPE, data[TARGET])
date_col = SPLIT.get("date_col")
parts = {"train": train, "gap": gap, "test": test} if len(gap) else {"train": train, "test": test}
rows = []
for name, part in parts.items():
    row = {"set": name, "rows": len(part), "share": round(len(part) / len(data), 4)}
    if classification:
        row.update({f"share of {k}": v for k, v in part[TARGET].value_counts(normalize=True).round(4).items()})
    else:
        row.update({"target mean": round(part[TARGET].mean(), 4), "target std": round(part[TARGET].std(), 4)})
    if date_col and date_col in part:
        d = pd.to_datetime(part[date_col], errors="coerce", format="mixed")
        row.update({"first date": f"{d.min():%Y-%m-%d}", "last date": f"{d.max():%Y-%m-%d}"})
    rows.append(row)
display(pd.DataFrame(rows))
'''))
    cells.append(md("### Cross Validation Folds\n\nEach fold validates on its own rows and trains on the rest of the "
                    "training set. For a time split, a fold trains only on dates before its validation window."))
    cells.append(code('''
display(folds)
'''))
    cells.append(md("### Timeline\n\nFor a time split: what each fold trains and validates on, and the final split. "
                    "Grey is training, blue is validation or test, and the white space between them is the gap."))
    cells.append(code('''
if SPLIT["strategy"] == "time":
    import matplotlib.dates as mdates
    span = lambda a, b: (mdates.date2num(pd.Timestamp(a)), (pd.Timestamp(b) - pd.Timestamp(a)).days + 1)
    train_dates = pd.to_datetime(train[date_col], format="mixed")
    test_dates = pd.to_datetime(test[date_col], format="mixed")
    bars = [(f"fold {int(f.fold)}", span(f.train_start, f.train_end), span(f.valid_start, f.valid_end)) for f in folds.itertuples()]
    bars.append(("final", span(train_dates.min(), train_dates.max()), span(test_dates.min(), test_dates.max())))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(bars) + 1))
    for i, (label, tr, va) in enumerate(bars):
        ax.broken_barh([tr], (i - 0.35, 0.7), color="#b8b8b8")
        ax.broken_barh([va], (i - 0.35, 0.7), color="#3b75af")
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.set_title("Training (grey) and validation or test (blue) periods")
    fig.autofmt_xdate()
    plt.show()
else:
    print("Not a time split, so there is no timeline. Folds are drawn at random" + (" by group." if SPLIT["strategy"] == "group" else "."))
'''))
    cells.append(md("### Leakage Checks\n\n1. No test row should have an exact copy in training.\n"
                    "2. For a group split, no entity may be on both sides. For a time split, the same entities "
                    "appear before and after the cutoff, which is expected.\n"
                    "3. For a time split, the gap between training and test, and inside every fold, must be at "
                    "least `gap_days`."))
    cells.append(code('''
problems = []
content_cols = [c for c in train.columns if c not in (prep["row_id_col"], split_core.FOLD)]
train_hashes = set(pd.util.hash_pandas_object(train[content_cols], index=False))
copies = int(pd.util.hash_pandas_object(test[content_cols], index=False).isin(train_hashes).sum())
print(f"1. Test rows with an exact copy in training: {copies:,}")
if copies:
    problems.append(f"{copies:,} test rows have an exact copy in training.")

entity = SPLIT.get("group_col")
if entity and entity in train:
    both = set(train[entity].dropna()) & set(test[entity].dropna())
    print(f"2. Values of {entity} in both training and test: {len(both):,} of {test[entity].nunique():,} in test")
    if SPLIT["strategy"] == "group" and both:
        problems.append(f"{len(both):,} values of {entity} are in both training and test.")
else:
    print("2. No entity column set, so entity overlap was not checked.")

if SPLIT["strategy"] == "time":
    gap_days = float(SPLIT["gap_days"] or 0)
    actual = (pd.to_datetime(test[date_col], format="mixed").min() - pd.to_datetime(train[date_col], format="mixed").max()).days
    print(f"3. Days between the last training date and the first test date: {actual} (gap_days: {gap_days:g})")
    if actual < gap_days:
        problems.append(f"The gap before the test set is {actual} days, less than gap_days.")
    for f in folds.itertuples():
        g = (pd.Timestamp(f.valid_start) - pd.Timestamp(f.train_end)).days
        if g < gap_days:
            problems.append(f"Fold {f.fold}: the gap is {g} days, less than gap_days.")
else:
    print("3. Not a time split, so there is no gap to check.")

print()
print("Problems: " + (" ".join(problems) if problems else "none found."))
'''))
    cells.append(md("## Save the Sets\n\nEvery file is parquet, in the processed data folder. `train.parquet` keeps "
                    "a `fold` column: the CV fold the row validates, or -1. Later stages read the folds with "
                    "`split_core.cv_splits(train, spec['split'])`."))
    cells.append(code('''
files, text_cols = split_core.save(result, PATHS["data_processed"])
for name, path in files.items():
    print(f"{name}: {ml_common.relpath(path, str(ROOT))} ({os.path.getsize(path) / 1e6:.1f} MB)")
if text_cols:
    print("Columns that mix types were saved as text: " + ", ".join(text_cols))
'''))
    cells.append(md("## Risks, Assumptions, Open Questions, and Decisions\n\n`resolved_questions` closes open questions "
                    "in the spec that this split answers, by ID."))
    cells.append(code(
        "# Decision\n"
        f"risks = {lit(list(answers.get('risks') or []), indent=0)}\n\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []), indent=0)}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []), indent=0)}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []), indent=0)}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []), indent=0)}", decision=True))
    cells.append(md("## Write to the Spec\n\nReplaces the `split` section of `ml_spec.yaml` with this run, and records "
                    "the run."))
    cells.append(code(f'''
dates = lambda part: pd.to_datetime(part[date_col], format="mixed") if date_col and len(part) else None
split_section = split_core.plain({{
    "strategy": SPLIT["strategy"],
    "data_source": DATA_PATH,
    "target": TARGET,
    "task_type": TASK_TYPE,
    "row_id_col": prep["row_id_col"],
    "date_col": date_col,
    "group_col": SPLIT.get("group_col"),
    "stratify": bool(SPLIT.get("stratify")) and classification and SPLIT["strategy"] != "time",
    "seed": SPLIT.get("seed"),
    "test": {{
        "rows": len(test),
        "share": round(len(test) / len(data), 4),
        "size": SPLIT.get("test_size") if SPLIT["strategy"] != "time" else None,
        "days": SPLIT.get("test_days") if SPLIT["strategy"] == "time" else None,
        "start": dates(test).min() if SPLIT["strategy"] == "time" else None,
        "end": dates(test).max() if SPLIT["strategy"] == "time" else None,
    }},
    "gap_days": SPLIT.get("gap_days") if SPLIT["strategy"] == "time" else None,
    "cv": {{
        "method": result["cv_method"],
        "n_folds": len(folds),
        "window": SPLIT["cv"].get("window") if SPLIT["strategy"] == "time" else None,
        "valid_days": (SPLIT["cv"].get("valid_days") or SPLIT.get("test_days")) if SPLIT["strategy"] == "time" else None,
        "train_days": SPLIT["cv"].get("train_days") if SPLIT["strategy"] == "time" else None,
        "folds": folds.to_dict("records"),
    }},
    "duplicates": {{"handling": SPLIT["duplicates"], "rows": prep["duplicate_rows"], "dropped": prep["duplicates_dropped"]}},
    "dropped": {{"missing_target": prep["missing_target"], "unreadable_dates": result["info"].get("unreadable_dates", 0)}},
    "rows": {{"train": len(train), "test": len(test), "gap": len(gap)}},
    "files": {{k: ml_common.relpath(p, str(ROOT)) for k, p in files.items()}},
    "checks": {{"problems": problems, "test_rows_with_copy_in_train": copies}},
    "run_at": dt.datetime.now().replace(microsecond=0).isoformat(),
}})
nb_common.write_stage(SPEC_PATH, "split", split_section, "{SKILL}", assumptions, open_questions, decisions,
                      replace=True)
not_found = nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
if not_found:
    print("These open question IDs were not found: " + ", ".join(map(str, not_found)))
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
print("Wrote the split section of ml_spec.yaml." + (" Problems: " + " ".join(problems) if problems else ""))
'''))
    cells += nb_common.tracking_cells(SKILL)
    cells.append(md("## Next Step\n\nRun `/ml-feature-engineering`. It loads these files, fits every learned step "
                    "(imputers, encoders, scalers) inside the CV folds on training rows only, and for a time split "
                    "builds lag features on the full timeline before cutting by date. The test set stays untouched "
                    "until the final evaluation. Check progress with `/ml-status`."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def suggest(spec, root):
    """Print the facts the questions need, and a recommended answers YAML. Column names and counts only."""
    import pandas as pd  # noqa: F401
    import split_core
    fr = spec.get("framing") or {}
    tgt = fr.get("target") or {}
    data_path = default_data_path(spec)
    blockers = ml_common.modelling_blockers(spec)
    print("Gate: " + ("open" if not blockers else "closed: " + "; ".join(blockers)))
    if not data_path or not os.path.exists(os.path.join(root, data_path)):
        print(f"Data file: not found ({data_path!r}). Set data_path in the answers.")
        df = None
    else:
        df = nb_common.load_table(os.path.join(root, data_path))
        print(f"Data file: {data_path}, {len(df):,} rows, {df.shape[1]} columns")
    target = tgt.get("name")
    print(f"Task: {fr.get('task_type')}. Target: {target}"
          + ("" if df is None or target in df.columns else " (NOT in the data: needs target_code)"))
    if df is not None and target in df.columns and "classification" in str(fr.get("task_type")):
        print("Class shares: " + ", ".join(f"{k}: {v:.1%}" for k, v in df[target].value_counts(normalize=True).items()))
    s, reasons = split_core.recommend(spec, df)
    if df is not None and s.get("date_col") in getattr(df, "columns", []):
        d = pd.to_datetime(df[s["date_col"]], errors="coerce", format="mixed")
        print(f"Dates in {s['date_col']}: {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}, {(d.max() - d.min()).days} days, "
              f"{d.dt.to_period('M').nunique()} months")
    for c in fr.get("entity_id_cols") or []:
        if df is not None and c in df.columns:
            print(f"Entity {c}: {df[c].nunique():,} values, {len(df) / max(df[c].nunique(), 1):.1f} rows each")
    if df is not None:
        print(f"Exact duplicate rows: {int(df.duplicated().sum()):,}")
    print("\nReasons:")
    for r in reasons:
        print("  " + r)
    if s["strategy"] == "time":
        s.pop("test_size", None)
        s.pop("stratify", None)
    else:
        for k in ("test_days", "gap_days", "date_col"):
            s.pop(k, None)
        s["cv"] = {"n_folds": s["cv"]["n_folds"]}
        if s["strategy"] == "random":
            s.pop("group_col", None)
    out = {"data_path": data_path, "target": target, "task_type": fr.get("task_type"), "target_code": None,
           "split": s, "reasoning": {"strategy": " ".join(reasons)}}
    print("\nRecommended answers (edit, then pass the file to make_notebook.py):\n")
    import yaml
    print(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--suggest", action="store_true", help="Print facts and recommended answers, then stop")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()

    spec_path = ml_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found. Run /ml-project-setup and the framing skills first.")
    spec = ml_common.migrate(ml_common.load(spec_path))
    if a.suggest:
        os.chdir(root)
        suggest(spec, root)
        return
    if not a.answers:
        ap.error("give an answers file, or --suggest")
    answers = ml_common.load(a.answers)
    nb_name = ml_common.notebook_name(STAGE)
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    nb = build(answers, spec, HERE, nb_name)
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
    sp = ml_common.load(spec_path).get("split") or {}
    cv = sp.get("cv") or {}
    print(f"Spec: split written. Strategy {sp.get('strategy')}, CV {cv.get('method')} with {cv.get('n_folds')} folds.")
    r = sp.get("rows") or {}
    print(f"Rows: train {r.get('train', 0):,}, test {r.get('test', 0):,}" + (f", gap {r['gap']:,}" if r.get("gap") else ""))
    t = sp.get("test") or {}
    if t.get("start"):
        print(f"Test period: {t['start']} to {t['end']}, gap {sp.get('gap_days')} days")
    for f in cv.get("folds") or []:
        extra = f", train {f['train_start']} to {f['train_end']}, valid {f['valid_start']} to {f['valid_end']}" if f.get("train_start") else ""
        print(f"  fold {f['fold']}: {f['n_train']:,} train, {f['n_valid']:,} valid{extra}")
    print("Files: " + ", ".join((sp.get("files") or {}).values()))
    probs = (sp.get("checks") or {}).get("problems") or []
    print("Problems: " + (" ".join(probs) if probs else "none found"))


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "split", "ml-train-test-split")
