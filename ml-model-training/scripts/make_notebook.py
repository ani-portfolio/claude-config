#!/usr/bin/env python3
"""Write notebooks/08_training.ipynb, or add models to it, and run it.

Every ML model of the project is built here, from simple (logistic or ridge regression) to complex (tuned gradient
boosting). Each model's default settings are scored on the CV folds, then Optuna tries more settings. Every trial
is a row in training_runs.parquet, which is only appended to. A tuning run that is already in the log (same model,
settings, tuning budget, metrics, and features and split) is read back instead of trained again, so adding a model
later only trains the new one.

The notebook compares every model with the baselines, checks the results (leakage, overfitting, stability, the V1
bar), picks the champion by the best mean CV score, refits it on the whole training set, and saves it to models/.
The test set is not used.

Usage (from the project root, with the project's Python):
    make_notebook.py --suggest                     # facts, packages, timing, models already built, a recommendation
    make_notebook.py answers.yaml                  # build the notebook, or add the models in answers to it, and run
    make_notebook.py answers.yaml --replace-model  # a model with the same name replaces the one in the notebook
    make_notebook.py --rebuild answers.yaml --overwrite   # write the notebook from scratch (asks when it was edited)
    make_notebook.py --check                       # which model packages are installed
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ml-model-training"
STAGE = "training"
TUNING = {"none": {"n_trials": 1}, "quick": {"n_trials": 20, "timeout_minutes": 10},
          "thorough": {"n_trials": 60, "timeout_minutes": 30}}
PACKAGES = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost", "optuna": "optuna"}


# ---------------------------------------------------------------- facts and suggestion

def installed():
    import importlib.util
    return {name: importlib.util.find_spec(mod) is not None for name, mod in PACKAGES.items()}


def models_in_notebook(path):
    """The MODELS list in an existing notebook, or None."""
    if not os.path.exists(path):
        return None
    cell = _models_cell(nb_common.read(path))
    return cell[1] if cell else None


def _models_cell(nb):
    """(cell, models) for the decision cell that assigns MODELS."""
    for c in nb.get("cells", []):
        src = c.get("source", "")
        src = src if isinstance(src, str) else "".join(src)
        if c.get("cell_type") != "code" or "MODELS" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise ValueError(f"The MODELS cell is not valid Python: {e}")
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODELS" for t in node.targets):
                return c, ast.literal_eval(node.value)
    return None


def suggest(spec, root):
    import numpy as np
    import pandas as pd
    import yaml
    sys.path.insert(0, root)
    import model_core
    import split_core
    blockers = ml_common.modelling_blockers(spec)
    print("Gate: " + ("open" if not blockers else "closed: " + "; ".join(blockers)))
    fe, bl, sp, fr = (spec.get(k) or {} for k in ("features", "baseline", "split", "framing"))
    if not fe.get("files"):
        print("No features yet. Run /ml-feature-engineering first.")
        return
    if not bl.get("metrics"):
        print("No baselines yet. Run /ml-baseline-model first, so the models have a reference and metrics.")
        return
    from src.features import feature_columns, load_config
    cfg = load_config(spec)
    data = pd.read_parquet(os.path.join(root, fe["files"]["train"]))
    folds = split_core.cv_splits(data, sp)
    family = model_core.task_family(fr.get("task_type"))
    target = sp.get("target")
    label = bl.get("primary_label")
    print(f"Task: {fr.get('task_type')}. Training rows: {len(data):,}. Inputs: {len(feature_columns(cfg))}. Folds: {len(folds)}")
    if family == "binary":
        share = data[target].mean()
        print(f"Positive share: {share:.2%}" + (" (rare: class weights are an option)" if share < 0.1 else ""))
    ref, best = bl.get("reference") or {}, bl.get("best_non_ml") or {}
    print(f"Metric: {label}. Reference: {ref.get('name')} {ref.get(label)}. Best baseline: {best.get('name')} {best.get(label)}")
    print(f"V1 bar: {(fr.get('success') or {}).get('v1_bar')}")
    cons = ((spec.get("business") or {}).get("constraints") or {})
    if cons.get("interpretability"):
        print(f"Interpretability: {cons['interpretability']}")
    pk = installed()
    print("Packages: " + ", ".join(f"{k} {'installed' if v else 'not installed'}" for k, v in pk.items()))
    if not pk["optuna"]:
        print("Optuna is needed for tuning: uv add optuna")
    # One default fit, to estimate the time of a trial (all folds)
    tr, va = folds[-1]
    m = {"name": "timing", "type": "hist_gbm"}
    start = time.time()
    model_core.fit_predict(m, data.iloc[tr], data.iloc[va], feature_columns(cfg), target, family, cfg)
    per_fit = time.time() - start
    per_trial = per_fit * len(folds)
    mins = lambda n: f"{max(1, round(n * per_trial / 60))} minute" + ("s" if round(n * per_trial / 60) > 1 else "")
    print(f"One gradient boosting fit takes about {per_fit:.1f} seconds, so a trial (all folds) about {per_trial:.0f} "
          f"seconds. Quick tuning (20 trials) takes about {mins(20)} per model and thorough tuning (60 trials) about "
          f"{mins(60)}, within limits of 10 and 30 minutes. Boosting libraries and random forests can take longer.")
    nb_path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], ml_common.notebook_name(STAGE))
    existing = models_in_notebook(nb_path)
    if existing:
        print("Models already in the notebook: " + ", ".join(f"{x['name']} ({x['type']})" for x in existing))
    runs_path = os.path.join(ml_common.project_paths(spec, root)["data_processed"], "training_runs.parquet")
    if os.path.exists(runs_path):
        runs = model_core.RunLog(runs_path).load()
        v = model_core.data_version(spec)
        stale = sorted(set(runs.loc[runs.data_version != v, "model"]) - set(runs.loc[runs.data_version == v, "model"]))
        if stale:
            print("Models trained on older features or split (stale): " + ", ".join(stale))
    linear = "logistic" if family == "binary" else "ridge"
    boost = next((t for t in ("lightgbm", "xgboost", "catboost") if pk.get(t)), "hist_gbm")
    names = {x["name"] for x in existing or []}
    rec = [m for m in (
        {"name": linear, "type": linear, "params": {}, "class_weight": None, "tuning": dict(TUNING["quick"]),
         "reason": "The simple model first: it shows whether the features carry signal, and it is easy to explain."},
        {"name": boost, "type": boost, "params": {}, "class_weight": None, "tuning": dict(TUNING["quick"]),
         "reason": "Gradient boosting usually gives the best results on tables like this one."},
    ) if m["name"] not in names]
    print("\nModel types: " + ", ".join(t for t in model_core.ML_MODELS if t != "gbm")
          + ". Tuning presets: none (default settings), quick (20 trials, 10 minutes), thorough (60 trials, 30 minutes).")
    print("\nRecommended answers (edit, then pass the file to make_notebook.py):\n")
    print(yaml.safe_dump({"models": rec, "training": {"seed": 42, "selection": "best_mean"}, "reasoning": {},
                          "risks": [], "assumptions": [], "open_questions": [], "decisions": [], "resolved_questions": []},
                         sort_keys=False, allow_unicode=True, width=110))


# ---------------------------------------------------------------- notebook

def _clean(models):
    out = []
    for m in models or []:
        m = {k: m.get(k) for k in ("name", "type", "params", "class_weight", "tuning", "inputs", "reason")}
        m["params"] = m["params"] or {}
        if not m["inputs"]:
            m.pop("inputs")
        if isinstance(m.get("tuning"), str):
            m["tuning"] = TUNING[m["tuning"]]
        m["tuning"] = m["tuning"] or TUNING["none"]
        out.append(m)
    return out


def build(answers, spec, scripts_dir, nb_name):
    reasoning = answers.get("reasoning") or {}
    models = _clean(answers.get("models"))
    training = {"seed": 42, "selection": "best_mean", **(answers.get("training") or {})}
    cells = nb_common.setup_cells(
        STAGE, "Model Training",
        "Every ML model of the project, from simple to complex, on the same CV folds and metrics as the baselines. "
        "Each model's default settings are scored first, then Optuna tries more settings. Every trial is logged to "
        "`training_runs.parquet`; a tuning run already in the log is read back instead of trained again. The champion "
        "is the model with the best mean CV score; it is refitted on the whole training set and saved for "
        "`/ml-offline-evaluation`. The test set is not used here.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the models and the training settings. To add a "
        "model, add it to `MODELS` (or run `/ml-model-training` again) and choose Run All: only new models are trained. "
        "Raw rows are never shown here.",
        scripts_dir, nb_name)
    cells.append(code('''
import datetime as dt
import importlib
import json

import joblib
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
framing, business, split, features, baseline = (spec.get(k) or {} for k in ("framing", "business", "split", "features", "baseline"))
TASK_TYPE, TARGET = framing.get("task_type"), split.get("target")
FAMILY = model_core.task_family(TASK_TYPE)
CONFIG = F.load_config(spec)
INPUTS = F.feature_columns(CONFIG)
METRICS = baseline["metrics"]
PRIMARY = baseline["primary_label"]
HIGHER = model_core.HIGHER_IS_BETTER.get(model_core.all_metrics(METRICS)[0]["key"]) is not False
VERSION = model_core.data_version(spec)
reference, best_baseline = baseline.get("reference") or {}, baseline.get("best_non_ml") or {}
print(f"Task: {TASK_TYPE}. Target: {TARGET}. Inputs: {len(INPUTS)}. Metric: {PRIMARY} ({'higher' if HIGHER else 'lower'} is better)")
print(f"Reference: {reference.get('name')} {reference.get(PRIMARY)}. Best baseline: {best_baseline.get('name')} {best_baseline.get(PRIMARY)}")
print(f"V1 bar: {(framing.get('success') or {}).get('v1_bar')}")
print(f"Features and split version: {VERSION}")
blockers = ml_common.modelling_blockers(spec)
print("Modelling may start." if not blockers else "Not ready for modelling: " + "; ".join(blockers))
'''))
    cells.append(md("## Load the Training Rows and Folds"))
    cells.append(code('''
data = pd.read_parquet(features["files"]["train"]).reset_index(drop=True)
folds = split_core.cv_splits(data, split)
row_id = split.get("row_id_col") or "row_id"
print(f"Training rows: {len(data):,}. Folds: {len(folds)}, validation rows per fold: {[len(v) for _, v in folds]}")
'''))
    text = "## Models\n\n"
    for key in ("models", "tuning"):
        if reasoning.get(key):
            text += reasoning[key].strip() + "\n\n"
    text += ("*Each model has a `name` (unique), a `type` (logistic, ridge, tree, random_forest, hist_gbm, lightgbm, "
             "xgboost, catboost), `params` (fixed settings, never tuned), `class_weight` (`balanced` or None), `tuning` "
             "(`n_trials`, 1 for the default settings only, and `timeout_minutes`), optional `inputs` (a subset of the "
             "feature inputs, when the model must use only some of them), and a `reason`. To add a model, "
             "append it here and choose Run All; models already trained with the same settings are read back from the "
             "run log.*")
    cells.append(md(text))
    cells.append(code(f"# Decision\nMODELS = {lit(models, indent=0)}", decision=True))
    cells.append(md("## Training Settings\n\n*`seed` makes the tuning repeatable. `selection` is `best_mean`: the "
                    "champion is the model with the best mean CV score on the primary metric.*"))
    cells.append(code(f"# Decision\nTRAINING = {lit(training, indent=0)}", decision=True))

    cells.append(md("## Train and Tune\n\nNew tuning runs are trained and appended to the run log. Runs already in "
                    "the log are read back."))
    cells.append(code('''
names = [m["name"] for m in MODELS]
assert len(names) == len(set(names)), "Every model needs a unique name."
for m in MODELS:
    assert m["type"] in model_core.ML_MODELS, f"{m['name']}: unknown type {m['type']}"
    assert model_core.available(m["type"]), f"{m['name']}: the package for {m['type']} is not installed (uv add {model_core.ML_MODELS[m['type']][1]})"
    model_core.model_config(m, CONFIG)   # checks that a model's inputs subset is in the feature set
RUNS = os.path.join(PATHS["data_processed"], "training_runs.parquet")
OOF = os.path.join(PATHS["data_processed"], "training_oof_predictions.parquet")
log = model_core.RunLog(RUNS)
keys = {}
for m in MODELS:
    key = model_core.study_key(m, METRICS, VERSION)
    keys[m["name"]] = key
    if log.has_study(key):
        print(f"{m['name']}: read back from the run log (study {key})")
        continue
    records, oof = model_core.tune(m, data, INPUTS, TARGET, folds, METRICS, TASK_TYPE, CONFIG, seed=TRAINING["seed"])
    log.append(m, key, VERSION, records)
    oof = oof.assign(model=m["name"], study=key, **{row_id: data[row_id].values[oof["position"]]}).drop(columns="position")
    old = pd.read_parquet(OOF) if os.path.exists(OOF) else pd.DataFrame()
    keep = old[old["model"] != m["name"]] if len(old) else old
    pd.concat([keep, oof], ignore_index=True).to_parquet(OOF, index=False)
    print(f"{m['name']}: {len(records)} trials logged")
'''))
    cells.append(md("## Leaderboard\n\nThe best trial of every model, with the baselines from `07_baseline.ipynb`, on "
                    "the same folds and metric. `vs reference` is the improvement over the reference in %, positive "
                    "when better."))
    cells.append(code('''
runs = log.load()
runs = runs[runs["study"].isin(keys.values())]
pick = (lambda g: g.loc[g["primary_mean"].idxmax()]) if HIGHER else (lambda g: g.loc[g["primary_mean"].idxmin()])
best = runs.groupby("model", sort=False).apply(pick, include_groups=False).reset_index()
best["trials"] = runs.groupby("model", sort=False).size().reindex(best["model"]).values
best["minutes"] = (runs.groupby("model", sort=False)["seconds"].sum() / 60).round(1).reindex(best["model"]).values
board = best[["model", "type", "primary_mean", "primary_std", "trials", "minutes"]].assign(kind="ml")
bl_runs = pd.read_parquet(baseline["files"]["cv_results"])
bl_runs = bl_runs[bl_runs["model"].isin([b["name"] for b in baseline.get("models") or []])]   # current baselines only
bl = bl_runs.groupby("model", sort=False)[PRIMARY].agg(["mean", "std"]).reset_index()
bl["model"] = [n if n not in keys else f"{n} (baseline)" for n in bl["model"]]
bl = bl.rename(columns={"mean": "primary_mean", "std": "primary_std"}).assign(kind="baseline", type="non ML")
board = pd.concat([board, bl], ignore_index=True)
ref = reference.get(PRIMARY)
if reference.get("name") in set(bl_runs["model"]):   # the exact value, not the rounded one in the spec
    ref = float(bl_runs.loc[bl_runs["model"] == reference["name"], PRIMARY].mean())
board["vs reference %"] = (((board.primary_mean - ref) if HIGHER else (ref - board.primary_mean)) / abs(ref) * 100).round(1) if ref else None
board = board.sort_values("primary_mean", ascending=not HIGHER).reset_index(drop=True)
display(board.rename(columns={"primary_mean": PRIMARY, "primary_std": "std"}))

fig, ax = plt.subplots(figsize=(8, 0.45 * len(board) + 1.2))
order = board.iloc[::-1]
for i, r in enumerate(order.itertuples()):
    ax.barh(i, r.primary_mean, color="#3b75af" if r.kind == "ml" else "#b8b8b8")
    if r.kind == "ml":
        vals = json.loads(best.set_index("model").loc[r.model, "folds"])
        ax.plot(vals, [i] * len(vals), "k.", markersize=6)
if ref:
    ax.axvline(ref, color="#c0504d", linestyle="--", linewidth=1)
ax.set_yticks(range(len(order)), order["model"])
ax.set_xlabel(PRIMARY + " (bar: mean of the best trial, dots: folds; dashed: the reference)")
ax.set_title("Models in blue, baselines in grey")
plt.show()
'''))
    cells.append(md("### Tuning History\n\nThe best score so far after each trial. A curve that is still rising at the "
                    "end means more trials could help."))
    cells.append(code('''
fig, ax = plt.subplots(figsize=(8, 3.5))
for name, g in runs.groupby("model", sort=False):
    g = g.sort_values("trial")
    so_far = g["primary_mean"].cummax() if HIGHER else g["primary_mean"].cummin()
    ax.plot(g["trial"], so_far, marker=".", label=name)
ax.set_xlabel("trial")
ax.set_ylabel(PRIMARY)
ax.legend()
plt.show()
'''))
    cells.append(md("## Champion\n\nThe model with the best mean CV score. It is refitted on every training row (the gap "
                    "and test rows are not used) and saved to the models folder for `/ml-offline-evaluation`."))
    cells.append(code('''
ml_board = board[board.kind == "ml"]
champ_row = ml_board.iloc[0]
champ_model = next(m for m in MODELS if m["name"] == champ_row["model"])
champ_trial = best.set_index("model").loc[champ_row["model"]]
champion = {**champ_model, "params": json.loads(champ_trial["params"])}
print(f"Champion: {champion['name']} ({champion['type']}), {PRIMARY} {champ_row['primary_mean']:.4f} "
      f"(std {champ_row['primary_std']:.4f}), trial {int(champ_trial['trial'])}")
print("Settings: " + json.dumps(champion["params"]))
pipe = model_core.make_estimator(champion, FAMILY, CONFIG, data[TARGET])
pipe.fit(data[INPUTS], data[TARGET])
os.makedirs(PATHS["models"], exist_ok=True)
MODEL_PATH = os.path.join(PATHS["models"], f"champion_{champion['name']}.joblib")
SERIALIZATION = model_core.save_model(pipe, MODEL_PATH)   # loads with joblib.load, without the skills
print(f"Saved {ml_common.relpath(MODEL_PATH, str(ROOT))} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
'''))
    cells.append(md("## Checks\n\n1. **Leakage:** a score that is too good to be true usually means a feature leaks the "
                    "outcome.\n2. **Against the baselines:** the champion should beat the reference and the best "
                    "baseline, and meet the V1 bar.\n3. **Overfitting:** the champion's score on its own training rows "
                    "against its CV score.\n4. **Stability:** how much the score moves between folds."))
    cells.append(code('''
import re
problems = []
champ_metrics = json.loads(champ_trial["metrics"])
if FAMILY == "binary" and champ_metrics.get("roc_auc", 0) > 0.97:
    problems.append(f"The champion's AUC is {champ_metrics['roc_auc']:.3f}, which is suspiciously high. Check the features for leakage.")
gain = champ_row["vs reference %"]
bb = best_baseline.get(PRIMARY)
beats = lambda a, b: a > b if HIGHER else a < b
if ref is not None and not beats(champ_row["primary_mean"], ref):
    problems.append("The champion does not beat the reference.")
if bb is not None and not beats(champ_row["primary_mean"], bb):
    problems.append(f"The champion does not beat the best baseline, {best_baseline.get('name')}.")
# The V1 bar, judged on CV from framing.success.v1_criteria (the text bar is shown for people, never parsed)
cv_scores = {str(n).replace(" (baseline)", ""): {PRIMARY: float(v)} for n, v in zip(board["model"], board["primary_mean"])}
v1_ctx = model_core.V1Context(spec, str(ROOT), "cv", champion, data, INPUTS, TARGET, FAMILY, CONFIG, folds, pipeline=pipe)
v1_cv = ml_common.v1_evaluate(spec, "cv", cv_scores, champion["name"], reference.get("name"),
                              model_core.run_v1_checks(v1_ctx), primary=PRIMARY)
print(f"V1 bar: {(framing.get('success') or {}).get('v1_bar') or 'not set'}")
print("V1 criteria on CV: " + ml_common.v1_summary(v1_cv))
for r in v1_cv["criteria"]:
    print(f"  {r['id']} ({r['kind']}): {r['status']}. {r['detail']}")
if v1_cv["status"] == "not_met":
    problems.append("The V1 bar is not met on CV: " + ", ".join(r["id"] for r in v1_cv["criteria"] if r["status"] == "fail") + ".")
elif v1_cv["status"] == "incomplete":
    problems.append("Some V1 criteria could not be judged: " + "; ".join(r["detail"] for r in v1_cv["criteria"] if r["status"] == "unavailable"))
v1_met = {"met": True, "not_met": False}.get(v1_cv["status"])

# Overfitting: score on the training rows of each fold against the validation rows
train_scores = []
for tr, va in folds:
    sample = tr if len(tr) <= 50_000 else np.random.default_rng(0).choice(tr, 50_000, replace=False)
    pred = model_core.fit_predict(champion, data.iloc[tr], data.iloc[sample], INPUTS, TARGET, FAMILY, CONFIG)
    m0 = model_core.all_metrics(METRICS)[0]
    groups = data.iloc[sample][m0["k_per"]].values if m0.get("k_per") else None
    train_scores.append(model_core.score_metric(m0, data.iloc[sample][TARGET].values, pred, groups,
                                                probabilistic=champion["type"] in model_core.PROBABILISTIC))
train_mean, cv_mean = float(np.mean(train_scores)), float(champ_row["primary_mean"])
print(f"Overfitting: {PRIMARY} on the training rows {train_mean:.4f}, on the validation rows {cv_mean:.4f}")
if CONFIG.get("high_cardinality"):
    print("Note: TargetEncoder columns are encoded with their own targets when a model scores its training rows, so "
          "the training score above is too high and is not used as an overfitting signal.")
elif cv_mean and abs(train_mean - cv_mean) / abs(cv_mean) > 0.3:
    problems.append(f"The champion scores {train_mean:.3f} on its training rows and {cv_mean:.3f} in CV: it may overfit. "
                    "Try stronger regularisation or fewer, simpler settings.")
if champ_row["primary_std"] and abs(champ_row["primary_std"]) > 0.2 * abs(cv_mean):
    problems.append("The champion's score varies a lot between folds.")
print("Problems: " + (" ".join(problems) if problems else "none found."))
'''))
    cells.append(md("### Feature Importance\n\nPermutation importance of the champion on the validation rows of the last "
                    "fold: how much the score drops when one input is shuffled. A single input that dominates, or an "
                    "input that should not matter, deserves a second look."))
    cells.append(code('''
from sklearn.inspection import permutation_importance
tr, va = folds[-1]
fold_pipe = model_core.make_estimator(champion, FAMILY, CONFIG, data.iloc[tr][TARGET]).fit(data.iloc[tr][INPUTS], data.iloc[tr][TARGET])
va_rows = va if len(va) <= 20_000 else np.random.default_rng(0).choice(va, 20_000, replace=False)
scoring = "roc_auc" if FAMILY == "binary" else "neg_mean_absolute_error"
imp = permutation_importance(fold_pipe, data.iloc[va_rows][INPUTS], data.iloc[va_rows][TARGET], scoring=scoring,
                             n_repeats=3, random_state=0)
importance = pd.DataFrame({"feature": INPUTS, "importance": imp.importances_mean, "std": imp.importances_std})
importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
print(f"Score used: {scoring}")
display(importance.head(15).round(4))
'''))
    cells.append(md("### Calibration\n\nFor classification: the champion's predicted probabilities against the observed "
                    "rate, in ten groups of its out of fold predictions. Points on the diagonal mean the probabilities "
                    "can be read as chances. Class weights push them away from it."))
    cells.append(code('''
if FAMILY == "binary":
    oof_all = pd.read_parquet(OOF)
    co = oof_all[oof_all["model"] == champion["name"]].merge(data[[row_id, TARGET]], on=row_id)
    co["group"] = pd.qcut(co["prediction"].rank(method="first"), 10, labels=False)
    cal = co.groupby("group").agg(predicted=("prediction", "mean"), observed=(TARGET, "mean"), rows=(TARGET, "size"))
    display(cal.round(4))
    fig, ax = plt.subplots(figsize=(4, 4))
    lim = max(cal.predicted.max(), cal.observed.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#b8b8b8", linewidth=1)
    ax.plot(cal.predicted, cal.observed, marker="o", color="#3b75af")
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_title("Calibration of the champion")
    plt.show()
else:
    print("Not a classification task, so there is no calibration check.")
'''))
    cells.append(md("## Risks, Assumptions, Open Questions, and Decisions"))
    cells.append(code(
        "# Decision\n"
        f"risks = {lit(list(answers.get('risks') or []), indent=0)}\n\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []), indent=0)}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []), indent=0)}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []), indent=0)}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []), indent=0)}", decision=True))
    cells.append(md("## Write to the Spec\n\nReplaces the `training` section of `ml_spec.yaml`. Offline evaluation "
                    "reads the champion from there."))
    cells.append(code(f'''
training_section = split_core.plain({{
    "models": MODELS,
    "settings": TRAINING,
    "data_version": VERSION,
    "primary_label": PRIMARY,
    "leaderboard": board.to_dict("records"),
    "champion": {{
        "name": champion["name"], "type": champion["type"], "params": champion["params"],
        "class_weight": champion.get("class_weight"), "inputs": champion.get("inputs"), "run_id": champ_trial["run_id"],
        PRIMARY: float(champ_row["primary_mean"]), "std": float(champ_row["primary_std"]),
        "vs_reference_pct": None if gain is None or pd.isna(gain) else float(gain),
        "beats_best_baseline": None if bb is None else bool(beats(champ_row["primary_mean"], bb)),
        "v1_bar_met": v1_met,
        "v1": v1_cv,
        "path": ml_common.relpath(MODEL_PATH, str(ROOT)),
        "serialization": SERIALIZATION,
    }},
    "top_features": importance.head(10)[["feature", "importance"]].round(5).to_dict("records"),
    "files": {{"runs": ml_common.relpath(RUNS, str(ROOT)), "oof_predictions": ml_common.relpath(OOF, str(ROOT)),
               "champion_model": ml_common.relpath(MODEL_PATH, str(ROOT))}},
    "checks": {{"problems": problems, "train_score": train_mean, "cv_score": cv_mean}},
    "run_at": dt.datetime.now().replace(microsecond=0).isoformat(),
}})
nb_common.write_stage(SPEC_PATH, "training", training_section, "{SKILL}", assumptions, open_questions, decisions,
                      replace=True)
not_found = nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
if not_found:
    print("These open question IDs were not found: " + ", ".join(map(str, not_found)))
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
print(f"Wrote the training section of ml_spec.yaml. Champion: {{champion['name']}}.")
'''))
    cells += nb_common.tracking_cells(SKILL)
    cells.append(md("## Next Step\n\nTo try another model, run `/ml-model-training` again or add it to `MODELS` above. "
                    "When the champion is good enough, run `/ml-offline-evaluation`, which scores it and the baselines "
                    "on the test set once. Check progress with `/ml-status`."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def add_models(path, new_models, replace=False):
    """Append models to the MODELS cell of an existing notebook. Returns the names added."""
    import json
    nb = nb_common.read(path)
    was_unchanged = nb_common.state(path) == "unchanged"
    found = _models_cell(nb)
    if not found:
        raise ValueError("The notebook has no MODELS cell. Rebuild it with --rebuild.")
    cell, models = found
    names = {m["name"] for m in models}
    added = []
    for m in _clean(new_models):
        if m["name"] in names:
            if not replace:
                raise ValueError(f"A model named {m['name']!r} is already in the notebook. Give it another name, or use "
                                 "--replace-model.")
            models = [m if x["name"] == m["name"] else x for x in models]
        else:
            models.append(m)
        added.append(m["name"])
    cell["source"] = f"# Decision\nMODELS = {lit(models, indent=0)}"
    meta = nb.setdefault("metadata", {}).setdefault(nb_common.META_KEY, {})
    if was_unchanged:   # a change made by the skill, not by a person
        meta["source_hash"] = nb_common.source_hash(nb)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return added


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--suggest", action="store_true", help="Print facts and a recommendation, then stop")
    ap.add_argument("--check", action="store_true", help="Print which model packages are installed")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--rebuild", action="store_true", help="Write the notebook from scratch instead of adding models")
    ap.add_argument("--replace-model", action="store_true", help="A model with the same name replaces the old one")
    ap.add_argument("--overwrite", action="store_true", help="With --rebuild: replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="With --rebuild: write 08_training_v2.ipynb instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()
    if a.check:
        print(", ".join(f"{k}: {'installed' if v else 'missing'}" for k, v in installed().items()))
        return
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
    if not (spec.get("baseline") or {}).get("metrics"):
        sys.exit("No baselines yet. Run /ml-baseline-model first.")
    answers = ml_common.load(a.answers)
    import model_core
    family = model_core.task_family((spec.get("framing") or {}).get("task_type"))
    for m in _clean(answers.get("models")):
        if m["type"] not in model_core.ML_MODELS:
            sys.exit(f"{m['name']}: unknown model type {m['type']!r}. Use one of: {', '.join(model_core.ML_MODELS)}")
        if model_core.ML_MODELS[m["type"]][2 if family == "binary" else 3] is None:
            sys.exit(f"{m['name']}: {m['type']} does not fit a {family} task. Use "
                     f"{'logistic' if family == 'binary' else 'ridge'} instead.")
        if not model_core.available(m["type"]):
            sys.exit(f"{m['name']}: {model_core.ML_MODELS[m['type']][1]} is not installed. Ask the user, then run: "
                     f"uv add {model_core.ML_MODELS[m['type']][1]}")
        if (m["tuning"].get("n_trials") or 1) > 1 and not installed()["optuna"]:
            sys.exit("Tuning needs optuna, which is not installed. Ask the user, then run: uv add optuna")
    nb_name = ml_common.notebook_name(STAGE)
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    if os.path.exists(path) and not a.rebuild:
        try:
            added = add_models(path, answers.get("models"), replace=a.replace_model)
        except ValueError as e:
            sys.exit(f"Not changed: {e}")
        print(f"Added to {ml_common.relpath(path, root)}: {', '.join(added)}")
        if a.no_run:
            return
        res = nb_common.run_existing(path)
    else:
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
    t = ml_common.load(spec_path).get("training") or {}
    label = t.get("primary_label")
    print(f"Spec: training written. Metric: {label}")
    for r in t.get("leaderboard") or []:
        print(f"  {r['model']} ({r['kind']}, {r['type']}): {r['primary_mean']:.4f}"
              + (f" (std {r['primary_std']:.4f})" if r.get("primary_std") == r.get("primary_std") and r.get("primary_std") is not None else "")
              + (f", {r['vs reference %'] + 0.0:+.1f}% vs reference" if r.get("vs reference %") is not None else "")
              + (f", {int(r['trials'])} trials, {r['minutes']} min" if r["kind"] == "ml" else ""))
    c = t.get("champion") or {}
    print(f"Champion: {c.get('name')} ({c.get('type')}), saved to {c.get('path')}. V1 bar on CV: "
          + ml_common.v1_summary(c.get("v1")))
    print("Top features: " + ", ".join(f["feature"] for f in (t.get("top_features") or [])[:5]))
    print("Problems: " + (" ".join((t.get("checks") or {}).get("problems") or []) or "none found"))


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "training", "ml-model-training")
