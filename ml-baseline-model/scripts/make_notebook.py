#!/usr/bin/env python3
"""Write notebooks/07_baseline.ipynb from the baseline answers, and run it once.

Baselines are the non ML references every model must beat: the current business rule, the base rate or mean, the
best single feature, a segment average, and for forecasting the naive, seasonal naive, and moving average
forecasts. They show what is possible without a model. Every ML model, simple or complex, is built in
ml-model-training. Every baseline is scored on the same CV folds as the models will be. The test set is not used.

The notebook holds the metrics and the baselines in decision cells, and shows the results per baseline and per
fold, a chart, and sanity checks. Running it writes baseline_cv_results.parquet and
baseline_oof_predictions.parquet, and the baseline section of ml_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py --suggest                    # the metric, and the non ML options for this task
    make_notebook.py answers.yaml                 # build and run; refuses to replace an edited notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write 07_baseline_v2.ipynb instead
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

SKILL = "ml-baseline-model"
STAGE = "baseline"


# ---------------------------------------------------------------- options

def _strongest(spec, numeric_only=True):
    fe = spec.get("features") or {}
    kinds = {i["name"]: i.get("kind") for i in fe.get("inputs") or []}
    for r in ((fe.get("checks") or {}).get("strongest") or []):
        if not numeric_only or kinds.get(r["feature"]) == "numeric":
            return r["feature"], r.get("score")
    return None, None


def options(spec, train=None):
    """Non ML baseline options for this task, each with name, kind, type, params, description, recommended."""
    import model_core
    fr, fe, sp = spec.get("framing") or {}, spec.get("features") or {}, spec.get("split") or {}
    bus = spec.get("business") or {}
    family = model_core.task_family(fr.get("task_type"))
    forecasting = fr.get("task_type") == "forecasting"
    cfg = fe.get("config") or {}
    target = sp.get("target")
    cats = [c for c in cfg.get("categorical") or []][:2]
    non_ml = []
    if bus.get("current_approach") and not forecasting:
        non_ml.append({"name": "current_rule", "kind": "non_ml", "type": "rule", "params": {"expression": None},
                       "description": f"Today's approach as a rule on the columns: {bus['current_approach']}",
                       "recommended": True})
    if forecasting:
        lags = sorted([(d, l["column"]) for l in cfg.get("lags") or [] for d in l["days"] if l["column"] == target])
        h = cfg.get("min_shift_days") or 0
        if lags:
            d, col = lags[0]
            non_ml.append({"name": "naive", "kind": "non_ml", "type": "column",
                           "params": {"column": f"{col}_lag_{d}d"},
                           "description": f"Tomorrow looks like {d} days ago: the value {d} days earlier (the horizon is {h} days).",
                           "recommended": True})
            weekly = [x for x in lags if x[0] % 7 == 0 and x[0] != d] or [x for x in lags if x[0] >= 364]
            if weekly:
                d2, col2 = weekly[-1] if weekly[-1][0] >= 364 else weekly[0]
                non_ml.append({"name": "seasonal_naive", "kind": "non_ml", "type": "column",
                               "params": {"column": f"{col2}_lag_{d2}d"},
                               "description": f"The value on the same day of the season, {d2} days earlier.",
                               "recommended": d % 7 != 0})
        for r in cfg.get("rolling") or []:
            if r["column"] == target and "mean" in (r.get("stats") or ["mean"]):
                w, sh = max(r["windows"]), int(r.get("shift_days") or 0)
                non_ml.append({"name": "moving_average", "kind": "non_ml", "type": "column",
                               "params": {"column": f"{target}_roll_{w}d_mean" + (f"_shift_{sh}d" if sh else "")},
                               "description": f"The average of the last {w} days, ending {sh} days before the forecast.",
                               "recommended": True})
                break
        series = (fr.get("entity_id_cols") or [None])[0]
        if series:
            non_ml.append({"name": "series_mean", "kind": "non_ml", "type": "segment", "params": {"by": [series]},
                           "description": f"The average of each series ({series}) over the training period.",
                           "recommended": False})
    else:
        non_ml.append({"name": "base_rate" if family == "binary" else "mean", "kind": "non_ml", "type": "prior",
                       "params": {}, "description": "Predict the average for every row. The floor every model must clear.",
                       "recommended": True})
        if family == "binary":
            col, score = _strongest(spec)
            if col:
                direction = "higher"
                if train is not None and target in train and col in train:
                    from sklearn.metrics import roc_auc_score
                    x = train[col].fillna(train[col].min() - 1)
                    direction = "higher" if roc_auc_score(train[target], x) >= 0.5 else "lower"
                non_ml.append({"name": f"best_feature_{col}", "kind": "non_ml", "type": "column",
                               "params": {"column": col, "direction": direction},
                               "description": f"Rank by {col} alone ({direction} first), the strongest single feature "
                                              f"(score {score}).", "recommended": not non_ml or non_ml[0]["type"] != "rule"})
        if cats:
            non_ml.append({"name": "segment_" + "_".join(cats), "kind": "non_ml", "type": "segment",
                           "params": {"by": cats},
                           "description": f"The average target of each {' and '.join(cats)} segment in the training rows.",
                           "recommended": False})
    return non_ml


def default_metrics(spec):
    import model_core
    fr, sp = spec.get("framing") or {}, spec.get("split") or {}
    family = model_core.task_family(fr.get("task_type"))
    m, note = model_core.parse_metric((fr.get("success") or {}).get("primary_metric"), sp.get("date_col"))
    if not m["key"]:
        if not (fr.get("success") or {}).get("primary_metric"):
            note = "The framing has no primary metric, so one is proposed from the task type."
        m = {"key": "pr_auc" if family == "binary" else "wape" if fr.get("task_type") == "forecasting" else "rmse"}
    m = {k: v for k, v in m.items() if v is not None}
    secondary = (["pr_auc", "roc_auc", "log_loss"] if family == "binary"
                 else ["wape", "mae", "rmse", "bias"] if fr.get("task_type") == "forecasting" else ["rmse", "mae", "r2"])
    return {"primary": m, "secondary": [s for s in secondary if s != m["key"]]}, note


def suggest(spec, root):
    import pandas as pd
    import yaml
    blockers = ml_common.modelling_blockers(spec)
    print("Gate: " + ("open" if not blockers else "closed: " + "; ".join(blockers)))
    fe = spec.get("features") or {}
    if not fe.get("files"):
        print("No features yet. Run /ml-feature-engineering first.")
        return
    fr, bus = spec.get("framing") or {}, spec.get("business") or {}
    train = pd.read_parquet(os.path.join(root, fe["files"]["train"]))
    print(f"Task: {fr.get('task_type')}. Target: {(spec.get('split') or {}).get('target')}. Training rows: {len(train):,}. "
          f"Folds: {((spec.get('split') or {}).get('cv') or {}).get('n_folds')}")
    print(f"Current approach: {bus.get('current_approach')}")
    print(f"Primary metric in the framing: {(fr.get('success') or {}).get('primary_metric')}. V1 bar: {(fr.get('success') or {}).get('v1_bar')}")
    metrics, note = default_metrics(spec)
    if note:
        print(note)
    print(f"Proposed primary metric: {metrics['primary']}")
    pm = metrics["primary"]
    if pm.get("k_per") and pm.get("k") and pm["k_per"] in train:
        size = train.groupby(pm["k_per"]).size().median()
        print(f"Rows per {pm['k_per']}: about {size:,.0f}. k = {pm['k']:,}"
              + (" is not smaller than that, so every row would be picked: use k_share instead." if pm["k"] >= size else "."))
    non_ml = options(spec, train)
    print("\nNon ML options (the skill asks which to run):\n")
    print(yaml.safe_dump(non_ml, sort_keys=False, allow_unicode=True, width=110))
    print("Recommended answers (edit, then pass the file to make_notebook.py):\n")
    chosen = [dict(o) for o in non_ml if o["recommended"]]
    for o in chosen:
        o["reason"] = o.pop("description")
        o.pop("recommended")
    ref = next((o["name"] for o in chosen if o["type"] == "rule"), None)
    out = {"metrics": metrics, "baselines": chosen, "reference": ref, "reasoning": {}, "risks": [], "assumptions": [],
           "open_questions": [], "decisions": [], "resolved_questions": []}
    print(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))


# ---------------------------------------------------------------- notebook

def build(answers, spec, scripts_dir, nb_name):
    reasoning = answers.get("reasoning") or {}
    baselines = []
    for b in answers.get("baselines") or []:
        b = dict(b)
        b.pop("recommended", None)
        if "description" in b and "reason" not in b:
            b["reason"] = b.pop("description")
        b.pop("description", None)
        baselines.append(b)
    cells = nb_common.setup_cells(
        STAGE, "Baseline Models",
        "The non ML references every model must beat, such as today's business rule. They show what is possible "
        "without a model. Every ML model, from simple to complex, is built in `08_training.ipynb`. Every baseline "
        "is scored on the same CV folds that model training will use. The test set is not used here.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the metrics and the baselines. Edit them, "
        "then choose Run All. Raw rows are never shown here.",
        scripts_dir, nb_name)
    cells.append(code('''
import datetime as dt
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

import model_core
import split_core

sys.path.insert(0, str(ROOT))   # the project's own src package
import src.features as F
importlib.reload(F)
'''))
    cells.append(md("## What This Builds On"))
    cells.append(code('''
framing, business, split, features = (spec.get(k) or {} for k in ("framing", "business", "split", "features"))
TASK_TYPE, TARGET = framing.get("task_type"), split.get("target")
CONFIG = F.load_config(spec)
INPUTS = F.feature_columns(CONFIG)
success = framing.get("success") or {}
print(f"Task: {TASK_TYPE}. Target: {TARGET}. Inputs: {len(INPUTS)}")
print(f"Current approach: {business.get('current_approach')}")
print(f"Primary metric in the framing: {success.get('primary_metric')}")
print(f"V1 bar: {success.get('v1_bar')}")
blockers = ml_common.modelling_blockers(spec)
print("Modelling may start." if not blockers else "Not ready for modelling: " + "; ".join(blockers))
'''))
    cells.append(md("## Load the Training Rows and Folds\n\nThe feature table, joined with the other columns of the "
                    "split's training rows so a business rule can use them. Columns excluded in the framing, and "
                    "features the leakage check left out, are not allowed in any baseline."))
    cells.append(code('''
row_id = split.get("row_id_col") or "row_id"
feat = pd.read_parquet(features["files"]["train"])
raw = pd.read_parquet(split["files"]["train"])
extra = [c for c in raw.columns if c not in feat.columns]
data = feat.merge(raw[[row_id] + extra], on=row_id, how="left").reset_index(drop=True)
folds = split_core.cv_splits(data, split)
EXCLUDED = {e["column"]: e.get("reason") for e in (spec.get("data") or {}).get("excluded_cols") or []}
EXCLUDED.update({c: "left out by the leakage check" for c in (features.get("checks") or {}).get("flagged") or []})
print(f"Training rows: {len(data):,}. Folds: {len(folds)}, validation rows per fold: {[len(v) for _, v in folds]}")
if "classification" in str(TASK_TYPE):
    print(f"Base rate of {TARGET}: {data[TARGET].mean():.4f}")
'''))
    cells.append(md("## Metrics\n\n" + ((reasoning.get("metrics") or "").strip() + "\n\n" if reasoning.get("metrics") else "")
                    + "*`primary` decides which baseline is the reference. Keys: roc_auc, pr_auc, log_loss, brier, "
                    "recall_at_k, precision_at_k, lift_at_k for classification; rmse, mae, mape, wape, r2, bias for "
                    "regression and forecasting. The \"at k\" metrics take `k` (rows) or `k_share` (a share of rows), "
                    "and `k_per` (a column such as the scoring date, to apply k within each date).*"))
    cells.append(code(f"# Decision\nMETRICS = {lit(answers.get('metrics') or {}, indent=0)}", decision=True))
    text = "## Baselines\n\n"
    for key in ("baselines", "non_ml"):
        if reasoning.get(key):
            text += reasoning[key].strip() + "\n\n"
    text += ("*Each baseline has a `name`, `type`, `params`, and `reason`. Types: `prior` (the average), `rule` (a "
             "pandas expression, such as `days_since_last_order > 30`), `column` (one column as the score or "
             "forecast; `direction` lower or higher), `segment` (the average per segment, `by`, learned on the "
             "fold's training rows). ML models belong in model training.*")
    cells.append(md(text))
    ref = answers.get("reference")
    if ref is None:
        ref = next((b["name"] for b in baselines if b.get("type") == "rule"), None)
    cells.append(code(f"# Decision\nBASELINES = {lit(baselines, indent=0)}", decision=True))
    cells.append(md("### Reference\n\nThe baseline later models are measured against, usually today's approach, "
                    "because the V1 bar is stated against it. `None` uses the best non ML baseline."))
    cells.append(code(f"# Decision\nREFERENCE = {ref!r}", decision=True))

    cells.append(md("## Score Every Baseline on the CV Folds"))
    cells.append(code('''
problems = [p for b in BASELINES for p in model_core.check_model(b, data, INPUTS, EXCLUDED)]
problems += [f"{b['name']} is an ML model ({b['type']}); ML models are built in model training."
             for b in BASELINES if b.get("type") not in model_core.NON_ML_TYPES]
for b in BASELINES:
    b.setdefault("kind", "non_ml")
assert not problems, " ".join(problems)
per_fold, oof = model_core.cross_validate(BASELINES, data, INPUTS, TARGET, folds, METRICS, TASK_TYPE, CONFIG)
summary = model_core.summarise(per_fold, METRICS)
PRIMARY = model_core.metric_label(METRICS["primary"])
display(summary)
'''))
    cells.append(md("### Per Fold\n\nThe primary metric in each fold. A baseline whose score moves a lot between folds "
                    "is less reliable as a reference."))
    cells.append(code('''
display(per_fold.pivot(index="model", columns="fold", values=PRIMARY).round(4))
fig, ax = plt.subplots(figsize=(8, 0.45 * len(summary) + 1.2))
order = summary["model"].tolist()[::-1]
for i, name in enumerate(order):
    vals = per_fold.loc[per_fold.model == name, PRIMARY]
    ax.barh(i, vals.mean(), color="#3b75af" if name == REFERENCE else "#b8b8b8")
    ax.plot(vals, [i] * len(vals), "k.", markersize=6)
ax.set_yticks(range(len(order)), order)
ax.set_xlabel(PRIMARY + " (bar: mean, dots: folds)")
ax.set_title("Baselines" + (f": the reference, {REFERENCE}, in blue" if REFERENCE else ""))
plt.show()
'''))
    cells.append(md("## Checks\n\n1. A baseline score that is too good to be true usually means a column leaks the "
                    "outcome.\n2. When another non ML baseline beats the reference, a simpler change to today's "
                    "approach may already help, and a model must beat that too.\n3. Scores should be stable across "
                    "folds.\n4. The \"at k\" metric must pick fewer rows than each group holds, or every baseline "
                    "scores the same."))
    cells.append(code('''
pm = model_core.all_metrics(METRICS)[0]
higher = model_core.HIGHER_IS_BETTER.get(pm["key"]) is not False
best_non_ml = summary.head(1)
reference = summary[summary.model == REFERENCE].head(1) if REFERENCE else best_non_ml
assert len(reference), f"REFERENCE {REFERENCE!r} is not one of the baselines."
checks = []
if "classification" in str(TASK_TYPE) and "roc_auc" in summary and summary["roc_auc"].max() > 0.97:
    checks.append(f"An AUC of {summary['roc_auc'].max():.3f} is very high for a baseline. Check the columns it uses for leakage.")
r = reference[PRIMARY].iloc[0]
print(f"Reference: {reference.model.iloc[0]}, {PRIMARY} {r:.4f}")
gain = None
if best_non_ml.model.iloc[0] != reference.model.iloc[0]:
    b = best_non_ml[PRIMARY].iloc[0]
    gain = ((b - r) if higher else (r - b)) / abs(r) * 100 if r else None
    print(f"Best baseline: {best_non_ml.model.iloc[0]}, {PRIMARY} {b:.4f}"
          + (f", {gain:.1f}% better than the reference" if gain is not None else "")
          + ". A simpler change to today's approach may already help, and a model must beat this too.")
std_col = f"{PRIMARY} std"
unstable = summary[(summary[std_col] > 0.2 * summary[PRIMARY].abs()) & summary[std_col].notna()]["model"].tolist()
if unstable:
    checks.append("Scores vary a lot between folds for: " + ", ".join(unstable) + ".")
if pm.get("k_per") and pm.get("k"):
    sizes = pd.concat([data.iloc[v] for _, v in folds]).groupby(pm["k_per"]).size()
    if pm["k"] >= sizes.median():
        checks.append(f"k = {pm['k']:,} is not smaller than the typical {pm['k_per']} group ({sizes.median():,.0f} rows), "
                      "so the at k metric picks every row. Use k_share instead.")
print("Problems: " + (" ".join(checks) if checks else "none found."))
'''))
    cells.append(md("## Save the Results\n\nParquet files in the processed data folder: the metrics per baseline and "
                    "fold, and the out of fold predictions (one row per training row, fold, and baseline), which "
                    "model training and error analysis compare against."))
    cells.append(code('''
os.makedirs(PATHS["data_processed"], exist_ok=True)
files = {"cv_results": os.path.join(PATHS["data_processed"], "baseline_cv_results.parquet"),
         "oof_predictions": os.path.join(PATHS["data_processed"], "baseline_oof_predictions.parquet")}
per_fold.to_parquet(files["cv_results"], index=False)
keys = [c for c in dict.fromkeys([row_id, split.get("date_col"), TARGET]) if c and c in data]
oof_out = oof.join(data[keys], on="position").drop(columns="position")
oof_out.to_parquet(files["oof_predictions"], index=False)
files = {k: ml_common.relpath(v, str(ROOT)) for k, v in files.items()}
print(files)
'''))
    cells.append(md("## Risks, Assumptions, Open Questions, and Decisions"))
    cells.append(code(
        "# Decision\n"
        f"risks = {lit(list(answers.get('risks') or []), indent=0)}\n\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []), indent=0)}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []), indent=0)}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []), indent=0)}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []), indent=0)}", decision=True))
    cells.append(md("## Write to the Spec\n\nReplaces the `baseline` section of `ml_spec.yaml`. Model training reads "
                    "the metrics, the reference, and the best baseline from there."))
    cells.append(code(f'''
pick = lambda frame: {{"name": frame.model.iloc[0], PRIMARY: float(frame[PRIMARY].iloc[0])}} if len(frame) else None
baseline_section = split_core.plain({{
    "metrics": METRICS,
    "primary_label": PRIMARY,
    "models": BASELINES,
    "results": summary.to_dict("records"),
    "reference": pick(reference),
    "best_non_ml": pick(best_non_ml),
    "best_vs_reference_pct": round(gain, 1) if gain is not None else None,
    "files": files,
    "checks": {{"problems": checks}},
    "run_at": dt.datetime.now().replace(microsecond=0).isoformat(),
}})
nb_common.write_stage(SPEC_PATH, "baseline", baseline_section, "{SKILL}", assumptions, open_questions, decisions,
                      replace=True)
not_found = nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
if not_found:
    print("These open question IDs were not found: " + ", ".join(map(str, not_found)))
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
print("Wrote the baseline section of ml_spec.yaml.")
'''))
    cells += nb_common.tracking_cells(SKILL)
    cells.append(md("## Next Step\n\nRun `/ml-model-training`. It builds the ML models, from simple (logistic or "
                    "ridge regression) to complex (tuned gradient boosting), on the same folds and metrics, and "
                    "compares every one with the reference and the best baseline above. Check progress with "
                    "`/ml-status`."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--suggest", action="store_true", help="Print the metric and the options, then stop")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()

    spec_path = ml_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found. Run /ml-project-setup and the earlier skills first.")
    os.chdir(root)
    sys.path.insert(0, root)
    spec = ml_common.migrate(ml_common.load(spec_path))
    if a.suggest:
        suggest(spec, root)
        return
    if not a.answers:
        ap.error("give an answers file, or --suggest")
    if not (spec.get("features") or {}).get("files"):
        sys.exit("No features yet. Run /ml-feature-engineering first.")
    answers = ml_common.load(a.answers)
    import model_core
    ml = [b.get("name") for b in answers.get("baselines") or [] if b.get("type") not in model_core.NON_ML_TYPES]
    if ml:
        sys.exit(f"{', '.join(ml)}: only non ML baselines are built here. ML models, simple and complex, are built "
                 "with /ml-model-training.")
    missing = [b.get("name") for b in answers.get("baselines") or []
               if b.get("type") == "rule" and not (b.get("params") or {}).get("expression")]
    if missing:
        sys.exit(f"Write the rule expression for {', '.join(missing)} first (params.expression).")
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
    b = ml_common.load(spec_path).get("baseline") or {}
    label = b.get("primary_label")
    print(f"Spec: baseline written. Primary metric: {label}")
    for r in b.get("results") or []:
        std = r.get(f"{label} std")
        print(f"  {r['model']}: {r[label]}" + (f" (std {std})" if std is not None else ""))
    ref, nm = b.get("reference") or {}, b.get("best_non_ml") or {}
    print(f"Reference: {ref.get('name')} {ref.get(label)}")
    if nm.get("name") != ref.get("name"):
        print(f"Best baseline: {nm.get('name')} {nm.get(label)}"
              + (f", {b['best_vs_reference_pct']:.1f}% better than the reference" if b.get("best_vs_reference_pct") is not None else ""))
    print("Files: " + ", ".join((b.get("files") or {}).values()))
    print("Problems: " + (" ".join((b.get("checks") or {}).get("problems") or []) or "none found"))


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "baseline", "ml-baseline-model")
