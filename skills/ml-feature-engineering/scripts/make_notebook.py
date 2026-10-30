#!/usr/bin/env python3
"""Write notebooks/06_features.ipynb from the feature answers, install the project's feature code, and run it once.

The feature code lives in the project, so every later stage (and batch scoring) uses exactly the same features:
    src/features.py         build_features (row and history steps) and make_preprocessor (learned steps, unfitted)
    src/custom_features.py  hand written features, created once and never changed by the skill

The notebook is the record. Decision cells hold the feature settings and the excluded columns. The other cells
build the features on the full timeline, then check them: a table of every input, single feature strength on the
CV folds (a feature whose AUC alone is 0.9 or more is excluded until it is confirmed), drift between training
and test, history windows, and a fit of both preprocessors inside one fold. Running it writes
features_train.parquet and features_test.parquet and the features section of ml_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py --suggest                    # facts and a recommended answers YAML, no data values
    make_notebook.py answers.yaml                 # install the code, build and run the notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write 06_features_v2.ipynb instead
    make_notebook.py answers.yaml --replace-code  # replace src/features.py even though it was edited
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ml-feature-engineering"
STAGE = "features"
TEMPLATES = os.path.join(SKILL_DIR, "templates")
TEXT_HINTS = ("text", "comment", "description", "note", "message", "review", "title", "body")


# ---------------------------------------------------------------- project code

def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def install_code(spec, root, replace=False):
    """Copy src/features.py (when absent, unchanged since the skill wrote it, or replace=True) and create
    src/custom_features.py once. Returns (info, problem)."""
    src = ml_common.project_paths(spec, root)["src"]
    os.makedirs(src, exist_ok=True)
    init = os.path.join(src, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    target = os.path.join(src, "features.py")
    template = os.path.join(TEMPLATES, "features.py")
    recorded = (((spec.get("features") or {}).get("code") or {}).get("features_hash"))
    state = "created"
    if os.path.exists(target):
        current = _sha(target)
        if current == _sha(template):
            state = "unchanged"
        elif current == recorded or replace:
            state = "updated"
        else:
            return None, (f"{ml_common.relpath(target, root)} was edited after the skill wrote it. Put hand written "
                          "features in src/custom_features.py instead, or rerun with --replace-code to replace it.")
    if state != "unchanged":
        shutil.copyfile(template, target)
    custom = os.path.join(src, "custom_features.py")
    custom_state = "kept"
    if not os.path.exists(custom):
        shutil.copyfile(os.path.join(TEMPLATES, "custom_features.py"), custom)
        custom_state = "created"
    return {"features": ml_common.relpath(target, root), "custom": ml_common.relpath(custom, root),
            "features_hash": _sha(target), "state": state, "custom_state": custom_state}, None


# ---------------------------------------------------------------- suggestion

def _dictionary(spec, root):
    path = os.path.join(ml_common.project_paths(spec, root)["docs"], "data_dictionary.yaml")
    if not os.path.exists(path):
        return {}
    d = ml_common.load(path) or {}
    src = os.path.normpath(str((spec.get("split") or {}).get("data_source") or ""))
    for block in (d.get("sources") or {}).values():
        if os.path.normpath(str(block.get("path"))) == src:
            return {e["column"]: e for e in block.get("columns") or []}
    return {}


def suggest_answers(spec, root, train):
    """A recommended answers dict, with a reason for every column. Uses names, types, counts, and the dictionary."""
    import pandas as pd
    import split_core
    fr, data, sp = spec.get("framing") or {}, spec.get("data") or {}, spec.get("split") or {}
    tgt = (fr.get("target") or {})
    task = fr.get("task_type") or sp.get("task_type")
    target = sp.get("target") or tgt.get("name")
    date_col = sp.get("date_col") or data.get("date_col") or (fr.get("prediction_time") or {}).get("column")
    entities = list(fr.get("entity_id_cols") or [])
    excluded = {e["column"]: e.get("reason") for e in data.get("excluded_cols") or [] if isinstance(e, dict)}
    ids = set(data.get("id_cols") or [])
    dic = _dictionary(spec, root)
    horizon = split_core.to_days((fr.get("prediction_time") or {}).get("horizon"))
    delay = max(split_core.to_days(tgt.get("label_delay")) or 0, split_core.to_days(tgt.get("outcome_window")) or 0)

    cfg = {"numeric": [], "categorical": [], "high_cardinality": [], "date_parts": {}, "days_since": [], "log": [],
           "ratios": [], "lags": [], "rolling": [], "custom": True, "date_col": date_col, "target": target,
           "min_shift_days": int(horizon or 0), "target_min_shift_days": int(max(horizon or 0, delay)),
           "rare_min_frequency": 0.01, "exclude": []}
    reasons, drop = {}, {}
    skip = {sp.get("row_id_col") or "row_id", "fold", "_set"}
    n = len(train)
    for col in train.columns:
        if col in skip:
            continue
        s = train[col]
        e = dic.get(col) or {}
        timing = e.get("timing")
        if col == target:
            continue
        if col in excluded:
            drop[col] = f"excluded in the ML framing: {excluded[col]}"
            continue
        if col in entities and task != "forecasting":
            drop[col] = "entity ID: identifies the row, so it is not a feature (it keeps the split grouped)"
            continue
        if col in ids or timing == "identifier":
            drop[col] = "identifier"
            continue
        if timing in ("possibly_after_outcome", "unknown"):
            drop[col] = f"the data dictionary says its timing is {timing.replace('_', ' ')}; confirm it before use"
            continue
        if s.nunique(dropna=True) <= 1:
            drop[col] = "constant in the training rows"
            continue
        is_date = pd.api.types.is_datetime64_any_dtype(s) or timing == "event_time"
        if not is_date and not pd.api.types.is_numeric_dtype(s) and any(k in col.lower() for k in ("date", "_dt", "_at", "time")):
            parsed = pd.to_datetime(s.dropna().head(500), errors="coerce", format="mixed")
            is_date = parsed.notna().mean() > 0.9
        if col == date_col:
            d = pd.to_datetime(s, errors="coerce", format="mixed").dropna().drop_duplicates().sort_values()
            step = d.diff().dt.days.median() if len(d) > 2 else None
            long_history = (d.max() - d.min()).days > 400
            # Month and week only help when every month has been seen in training at least once, which needs more
            # than a year of history. Otherwise test months would take values the model never saw.
            parts = (["dayofweek"] if step and step < 7 else []) + (["month"] if long_history else [])
            if long_history and step and step < 28:
                parts.append("week")
            if parts:
                cfg["date_parts"][col] = parts
                reasons.update({f"{col}_{p}": f"calendar part of the prediction date `{col}`" for p in parts})
                drop[col] = "the prediction date itself; its calendar parts are used instead"
            else:
                drop[col] = ("the prediction date itself; with under 400 days of history, month or week would take "
                             "values in the test period that training never saw")
            continue
        if is_date:
            if date_col:
                cfg["days_since"].append({"column": col, "reference": date_col})
                reasons[f"days_since_{col}"] = f"days from `{col}` to the prediction date"
                drop[col] = "a date; the days from it to the prediction date are used instead"
            else:
                cfg["date_parts"][col] = ["year", "month", "dayofweek"]
                reasons.update({f"{col}_{p}": f"calendar part of `{col}`" for p in ["year", "month", "dayofweek"]})
                drop[col] = "a date; its calendar parts are used instead"
            continue
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            cfg["numeric"].append(col)
            reasons[col] = "numeric" + (f", timing {timing.replace('_', ' ')}" if timing else "")
            x = pd.to_numeric(s, errors="coerce")
            if x.min() >= 0 and x.nunique() > 20 and x.skew() > 2:
                cfg["log"].append(col)
                reasons[f"log_{col}"] = f"log of `{col}`, which is strongly skewed (helps linear models)"
            continue
        text = s.dropna().astype(str)
        uniq = s.nunique(dropna=True)
        if any(k in col.lower() for k in TEXT_HINTS) and text.str.len().mean() > 30 and uniq > 0.5 * n:
            drop[col] = "free text; needs its own text features, not in V1"
            continue
        if uniq <= 50:
            cfg["categorical"].append(col)
            reasons[col] = f"categorical, {uniq} values"
        else:
            cfg["high_cardinality"].append(col)
            reasons[col] = f"high cardinality, {uniq:,} values: TargetEncoder"
    if task == "forecasting" and target and date_col and horizon:
        by = entities[0] if entities else None
        h = int(horizon)
        lags = [h, h + 7]
        d = pd.to_datetime(train[date_col], errors="coerce", format="mixed")
        if (d.max() - d.min()).days > 400 + h:
            lags.append(364 if h <= 364 else h)
        cfg["lags"].append({"column": target, "days": sorted(set(lags)), "by": by})
        cfg["rolling"].append({"column": target, "windows": [7, 28], "stats": ["mean"], "shift_days": h, "by": by})
        for x in lags:
            reasons[f"{target}_lag_{x}d"] = f"{target} {x} days earlier (at least the {h} day horizon)"
        for w in (7, 28):
            reasons[f"{target}_roll_{w}d_mean_shift_{h}d"] = f"mean {target} over {w} days, ending {h} days earlier"
        if by and by in drop:
            drop.pop(by)
        if by in cfg["numeric"]:
            cfg["numeric"].remove(by)
        if by and by not in cfg["categorical"] + cfg["high_cardinality"]:
            (cfg["categorical"] if train[by].nunique() <= 50 else cfg["high_cardinality"]).append(by)
            reasons[by] = "the series ID, so the model can learn each series' level"
    return {"config": cfg, "excluded": drop, "feature_reasons": reasons, "leakage_threshold": 0.9,
            "confirmed_safe": [], "reasoning": {}, "risks": [], "assumptions": [], "open_questions": [],
            "decisions": [], "resolved_questions": []}


def suggest(spec, root):
    import pandas as pd
    blockers = ml_common.modelling_blockers(spec)
    print("Gate: " + ("open" if not blockers else "closed: " + "; ".join(blockers)))
    sp = spec.get("split") or {}
    if not sp.get("files"):
        print("No split yet. Run /ml-train-test-split first.")
        return
    train = pd.read_parquet(os.path.join(root, sp["files"]["train"]))
    print(f"Training rows: {len(train):,}, columns: {train.shape[1]}. Split: {sp.get('strategy')}, "
          f"CV {((sp.get('cv') or {}).get('method'))}. Task: {(spec.get('framing') or {}).get('task_type')}")
    ans = suggest_answers(spec, root, train)
    c = ans["config"]
    print(f"Numeric: {len(c['numeric'])}, categorical: {len(c['categorical'])}, high cardinality: "
          f"{len(c['high_cardinality'])}, made by steps: {len(ans['feature_reasons']) - len(c['numeric']) - len(c['categorical']) - len(c['high_cardinality'])}")
    print("Excluded:")
    for col, why in ans["excluded"].items():
        print(f"  {col}: {why}")
    import yaml
    print("\nRecommended answers (edit, then pass the file to make_notebook.py):\n")
    print(yaml.safe_dump(ans, sort_keys=False, allow_unicode=True, width=110))


# ---------------------------------------------------------------- notebook

def build(answers, spec, scripts_dir, nb_name):
    cfg = answers.get("config") or {}
    reasoning = answers.get("reasoning") or {}
    cells = nb_common.setup_cells(
        STAGE, "Feature Engineering",
        "Which columns become model inputs, how they are built, and how they are prepared for each kind of model.\n\n"
        "* **Row and history steps** (calendar parts, days since, logs, ratios, lags, rolling windows) learn nothing "
        "from the data. They are built once on the full timeline by `src.features.build_features`.\n"
        "* **Learned steps** (imputing, encoding, scaling, TargetEncoder) are an unfitted pipeline from "
        "`src.features.make_preprocessor`. Later stages fit it on training rows only, inside each CV fold.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the feature settings. Edit them, then choose "
        "Run All. Hand written features go in `src/custom_features.py`. Raw rows are never shown here.",
        scripts_dir, nb_name)
    cells.append(code('''
import datetime as dt
import importlib
import time

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import roc_auc_score

import split_core

sys.path.insert(0, str(ROOT))   # the project's own src package
import src.features as F
importlib.reload(F)
'''))
    cells.append(md("## What This Builds On"))
    cells.append(code('''
framing, split = spec.get("framing") or {}, spec.get("split") or {}
TASK_TYPE, TARGET = framing.get("task_type"), split.get("target")
print(f"task: {TASK_TYPE}, target: {TARGET}, split: {split.get('strategy')}, CV: {(split.get('cv') or {}).get('method')} "
      f"with {(split.get('cv') or {}).get('n_folds')} folds")
blockers = ml_common.modelling_blockers(spec)
print("Modelling may start." if not blockers else "Not ready for modelling: " + "; ".join(blockers))
'''))
    cells.append(md("## Load the Split Sets\n\nTraining, gap (time split only), and test rows are stacked in one table, "
                    "so history features of the test rows can see the rows before them. Only training and test rows "
                    "are saved at the end."))
    cells.append(code('''
sets = split_core.load_sets(spec, str(ROOT))
parts = [sets["train"].assign(_set="train"), sets["test"].assign(_set="test")]
if "gap" in sets:
    parts.insert(1, sets["gap"].assign(_set="gap"))
full = pd.concat(parts, ignore_index=True)
print({name: len(part) for name, part in sets.items() if name != "folds"})
'''))
    text = "## Feature Settings\n\n"
    for key in ("inputs", "steps", "history", "encoding"):
        if reasoning.get(key):
            text += reasoning[key].strip() + "\n\n"
    text += ("*`numeric`, `categorical` (up to about 50 values: one hot for linear models, ordinal codes for trees), and "
             "`high_cardinality` (TargetEncoder) list the input columns. `date_parts`, `days_since`, `log`, `ratios`, "
             "`lags`, and `rolling` make new columns. A lag or rolling window must look back at least "
             "`min_shift_days` (the forecast horizon), and lags of the target at least `target_min_shift_days`. "
             "`exclude` leaves columns out of the inputs. `custom: True` also runs `src/custom_features.py`.*")
    cells.append(md(text))
    cells.append(code(f"# Decision\nCONFIG = {lit(cfg, indent=0)}", decision=True))
    cells.append(md("## Columns Left Out\n\nEvery column of the data that is not an input, with the reason. "
                    "Move a column from here into the settings above to use it."))
    cells.append(code(f"# Decision\nEXCLUDED = {lit(answers.get('excluded') or {}, indent=0)}", decision=True))
    cells.append(md("## Why Each Feature Is Used\n\nOne line per input, shown in the feature table and kept in the spec."))
    cells.append(code(f"# Decision\nFEATURE_REASONS = {lit(answers.get('feature_reasons') or {}, indent=0)}", decision=True))

    cells.append(md("## Build the Features"))
    cells.append(code('''
listed = list(CONFIG["numeric"]) + list(CONFIG["categorical"]) + list(CONFIG["high_cardinality"])
twice = sorted({c for c in listed if listed.count(c) > 1})
assert not twice, f"These columns are in more than one of numeric, categorical, and high_cardinality: {twice}"
start = time.time()
built = F.build_features(full, CONFIG)
inputs = F.feature_columns(CONFIG)
missing_cols = [c for c in inputs if c not in built.columns]
assert not missing_cols, f"Inputs that were not built or are not in the data: {missing_cols}"
clash = sorted(set(inputs) & set(EXCLUDED))
assert not clash, f"These columns are both inputs and left out: {clash}"
train_f = built[built["_set"] == "train"].reset_index(drop=True)
test_f = built[built["_set"] == "test"].reset_index(drop=True)
print(f"Built {len(inputs)} inputs in {time.time() - start:.1f} seconds. Training rows: {len(train_f):,}, test rows: {len(test_f):,}")
'''))
    cells.append(md("## Feature Table\n\nEvery input with how it was made, its type, missing values, and distinct "
                    "values in the training rows."))
    cells.append(code('''
made = F.made_columns(CONFIG)
def kind_of(c):
    return "categorical" if c in CONFIG["categorical"] else "high cardinality" if c in CONFIG["high_cardinality"] else "numeric"
table = pd.DataFrame([{
    "feature": c, "kind": kind_of(c), "step": made.get(c, ("column", c))[0], "source": made.get(c, ("column", c))[1],
    "missing %": round(train_f[c].isna().mean() * 100, 1), "distinct": train_f[c].nunique(),
    "reason": FEATURE_REASONS.get(c, ""),
} for c in inputs])
display(table)
'''))

    cells.append(md("## Checks\n\n### Single Feature Strength\n\nEach feature on its own, scored on the CV folds: for "
                    "classification the AUC of the feature alone (the larger of AUC and 1 minus AUC, so 0.5 is no "
                    "signal and 1.0 is perfect), for regression and forecasting the absolute Spearman correlation. "
                    "Categorical features are first replaced by the target rate of their category in the fold's "
                    "training rows, so an ID cannot score well by memorising rows. A real feature rarely reaches 0.9 "
                    "alone; a score that high usually means it was recorded after the outcome. Flagged features are "
                    "left out unless listed in `CONFIRMED_SAFE`, with the reason in the decisions below."))
    cells.append(code(
        "# Decision\n"
        f"LEAKAGE_THRESHOLD = {float(answers.get('leakage_threshold') or 0.9)}\n"
        f"CONFIRMED_SAFE = {lit(list(answers.get('confirmed_safe') or []), indent=0)}", decision=True))
    cells.append(code('''
folds = split_core.cv_splits(train_f, split)
classification = "classification" in str(TASK_TYPE)
y_all = train_f[TARGET]
rng = np.random.default_rng(0)
keep = np.sort(rng.choice(len(train_f), 300_000, replace=False)) if len(train_f) > 300_000 else np.arange(len(train_f))
kept = np.zeros(len(train_f), dtype=bool); kept[keep] = True   # large data: score on a sample of 300,000 rows


def fold_score(x, y, train_idx, valid_idx, categorical):
    if categorical:   # target rate of each category, learned on the fold's training rows only
        key = x.astype("object").where(x.notna(), "missing").astype(str)
        rates = pd.Series(y.iloc[train_idx].values, index=key.iloc[train_idx].values).groupby(level=0).mean()
        v = key.iloc[valid_idx].map(rates).fillna(y.iloc[train_idx].mean())
    else:
        v = pd.to_numeric(x.iloc[valid_idx], errors="coerce")
        v = v.fillna(v.min() - 1 if v.notna().any() else 0)   # missing values rank lowest
    yv = y.iloc[valid_idx]
    if classification:
        classes = sorted(yv.unique())
        if len(classes) < 2:
            return np.nan
        pos = classes[-1:] if len(classes) == 2 else classes
        aucs = [roc_auc_score(yv == c, v) for c in pos]
        return max(max(a, 1 - a) for a in aucs)
    return abs(pd.Series(v.values).corr(pd.Series(yv.values), method="spearman"))


rows = []
for c in inputs:
    cat = c in CONFIG["categorical"] or c in CONFIG["high_cardinality"]
    scores = [fold_score(train_f[c], y_all, tr[kept[tr]], va[kept[va]], cat) for tr, va in folds]
    rows.append({"feature": c, "score": round(float(np.nanmean(scores)), 4), "measure": "AUC" if classification else "|Spearman|"})
strength = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
strength["flagged"] = strength["score"] >= LEAKAGE_THRESHOLD
strength["confirmed safe"] = strength["feature"].isin(CONFIRMED_SAFE)
flagged = strength.loc[strength["flagged"] & ~strength["confirmed safe"], "feature"].tolist()
display(strength.head(15))
print("Left out by this check: " + (", ".join(flagged) if flagged else "none"))
FINAL = {**CONFIG, "exclude": sorted(set(CONFIG.get("exclude") or []) | set(flagged))}
final_inputs = F.feature_columns(FINAL)
print(f"Inputs after the check: {len(final_inputs)}")
'''))
    cells.append(md("### Drift Between Training and Test\n\nThe population stability index (PSI) of each feature, "
                    "training rows against test rows. Under 0.1 is stable, 0.1 to 0.25 is a moderate shift, and over "
                    "0.25 is a large shift worth understanding. For a time split some drift is expected (for example "
                    "calendar parts); a large shift in an ordinary feature can mean its meaning changed."))
    cells.append(code('''
def psi(a, b, categorical):
    if categorical:
        a, b = a.astype("object").fillna("missing").astype(str), b.astype("object").fillna("missing").astype(str)
        top = a.value_counts().index[:20]
        a, b = a.where(a.isin(top), "other"), b.where(b.isin(top), "other")
        p, q = a.value_counts(normalize=True), b.value_counts(normalize=True)
    else:
        a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        edges = np.unique(np.nanquantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        p = pd.cut(a, edges).value_counts(normalize=True, dropna=False)
        q = pd.cut(b, edges).value_counts(normalize=True, dropna=False)
    p, q = p.align(q, fill_value=0)
    p, q = p.clip(lower=1e-4), q.clip(lower=1e-4)
    return float(((p - q) * np.log(p / q)).sum())


drift = pd.DataFrame([{"feature": c, "psi": round(psi(train_f[c], test_f[c], kind_of(c) != "numeric"), 3)}
                      for c in final_inputs]).sort_values("psi", ascending=False).reset_index(drop=True)
drift["shift"] = pd.cut(drift["psi"], [-1, 0.1, 0.25, np.inf], labels=["stable", "moderate", "large"])
display(drift.head(10))
'''))
    cells.append(md("### History Windows\n\nEvery lag and rolling window, how far back it looks, and the minimum it must "
                    "look back. Missing values at the start of the history are expected; many more missing values "
                    "in test than in training would mean the test rows cannot see their past."))
    cells.append(code('''
history = []
for lag in CONFIG.get("lags") or []:
    for d in lag["days"]:
        history.append((f"{lag['column']}_lag_{d}d", lag["column"], d))
for r in CONFIG.get("rolling") or []:
    for w in r["windows"]:
        for s in r.get("stats") or ["mean"]:
            sh = int(r.get("shift_days") or 0)
            history.append((f"{r['column']}_roll_{w}d_{s}" + (f"_shift_{sh}d" if sh else ""), r["column"], sh))
if history:
    display(pd.DataFrame([{
        "feature": name, "looks back (days)": back,
        "minimum": max(CONFIG.get("min_shift_days") or 0, (CONFIG.get("target_min_shift_days") or 0) if col == TARGET else 0),
        "missing % train": round(train_f[name].isna().mean() * 100, 1), "missing % test": round(test_f[name].isna().mean() * 100, 1),
    } for name, col, back in history]))
else:
    print("No lags or rolling windows.")
'''))
    cells.append(md("### Fit Inside One Fold\n\nBoth preprocessors are fitted on the training rows of the first CV fold "
                    "only, then applied to its validation rows, as the modelling stages will do. The linear version "
                    "must have no missing values left; the tree version may, because tree models handle them."))
    cells.append(code('''
tr, va = folds[0]
fit_check = []
for kind in ("linear", "tree"):
    start = time.time()
    pre = F.make_preprocessor(kind, FINAL)
    pre.fit(train_f.loc[tr, final_inputs], train_f.loc[tr, TARGET])
    out = pre.transform(train_f.loc[va, final_inputs])
    fit_check.append({"preprocessor": kind, "inputs": len(final_inputs), "outputs": out.shape[1],
                      "missing after": int(out.isna().sum().sum()), "seconds": round(time.time() - start, 1)})
fit_check = pd.DataFrame(fit_check)
display(fit_check)
problems = []
if int(fit_check.loc[fit_check.preprocessor == "linear", "missing after"].iloc[0]):
    problems.append("The linear preprocessor leaves missing values.")
shifted = drift.loc[drift["shift"] == "large", "feature"].tolist()
calendar = [c for c in shifted if made.get(c, ("",))[0] == "date_part"]
target_history = [c for c in shifted if made.get(c, ("", ""))[0] in ("lag", "rolling") and made[c][1] == TARGET]
large = [c for c in shifted if c not in calendar and c not in target_history]
if calendar:
    print("Calendar parts shift between training and test, which a time split always causes: " + ", ".join(calendar)
          + ". Check that the test period's values (for example its months) also appear in training.")
if target_history:
    print("Lags and windows of the target shift when the level of the series changes over time, which is expected: "
          + ", ".join(target_history) + ".")
if large:
    problems.append("Large drift between training and test: " + ", ".join(large) + ".")
print("Problems: " + (" ".join(problems) if problems else "none found."))
'''))
    cells.append(md("## Save the Feature Tables\n\nParquet files in the processed data folder, with the keys (row ID, "
                    "fold, date, entity), the target, and every input before the learned steps."))
    cells.append(code('''
keys = [c for c in dict.fromkeys([split.get("row_id_col"), split_core.FOLD, split.get("date_col"), split.get("group_col"), TARGET])
        if c and c in train_f.columns and c not in final_inputs]
files = {}
for name, frame in (("train", train_f), ("test", test_f)):
    path = os.path.join(PATHS["data_processed"], f"features_{name}.parquet")
    frame[[c for c in keys if c in frame.columns] + final_inputs].to_parquet(path, index=False)
    files[name] = ml_common.relpath(path, str(ROOT))
    print(f"{files[name]}: {len(frame):,} rows, {os.path.getsize(path) / 1e6:.1f} MB")
'''))
    cells.append(md("## Risks, Assumptions, Open Questions, and Decisions\n\nRecord a decision with the reason for "
                    "every feature in `CONFIRMED_SAFE`. `resolved_questions` closes open questions by ID."))
    cells.append(code(
        "# Decision\n"
        f"risks = {lit(list(answers.get('risks') or []), indent=0)}\n\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []), indent=0)}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []), indent=0)}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []), indent=0)}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []), indent=0)}", decision=True))
    cells.append(md("## Write to the Spec\n\nReplaces the `features` section of `ml_spec.yaml` with this run. Later "
                    "stages read the config from there with `src.features.load_config`."))
    cells.append(code(f'''
code_info = (spec.get("features") or {{}}).get("code") or {{}}
features_section = split_core.plain({{
    "config": FINAL,
    "inputs": [{{"name": r.feature, "kind": r.kind, "step": r.step, "source": r.source, "reason": r.reason}}
               for r in table[table.feature.isin(final_inputs)].itertuples()],
    "excluded": [{{"column": c, "reason": r}} for c, r in EXCLUDED.items()]
                + [{{"column": c, "reason": f"single feature score {{strength.set_index('feature').score[c]}} reached the leakage threshold"}} for c in flagged],
    "n_inputs": len(final_inputs),
    "outputs": {{r.preprocessor: r.outputs for r in fit_check.itertuples()}},
    "files": files,
    "code": {{"features": code_info.get("features", "src/features.py"), "custom": code_info.get("custom", "src/custom_features.py"),
             "features_hash": code_info.get("features_hash")}},
    "checks": {{
        "leakage_threshold": LEAKAGE_THRESHOLD,
        "strongest": strength.head(10).to_dict("records"),
        "flagged": flagged,
        "confirmed_safe": CONFIRMED_SAFE,
        "drift_large": large,
        "problems": problems,
    }},
    "run_at": dt.datetime.now().replace(microsecond=0).isoformat(),
}})
nb_common.write_stage(SPEC_PATH, "features", features_section, "{SKILL}", assumptions, open_questions, decisions,
                      replace=True)
not_found = nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
if not_found:
    print("These open question IDs were not found: " + ", ".join(map(str, not_found)))
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
print(f"Wrote the features section of ml_spec.yaml: {{len(final_inputs)}} inputs.")
'''))
    cells += nb_common.tracking_cells(SKILL)
    cells.append(md("## Next Step\n\nRun `/ml-baseline-model`. It scores the current business rule and simple models "
                    "on the same CV folds, using `make_preprocessor(\"linear\", config)` fitted inside each fold. "
                    "Check progress with `/ml-status`."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--suggest", action="store_true", help="Print facts and recommended answers, then stop")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--replace-code", action="store_true", help="Replace src/features.py even if it was edited")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()

    spec_path = ml_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    if not os.path.exists(spec_path):
        sys.exit("No ml_spec.yaml found. Run /ml-project-setup and the earlier skills first.")
    os.chdir(root)
    spec = ml_common.migrate(ml_common.load(spec_path))
    if a.suggest:
        suggest(spec, root)
        return
    if not a.answers:
        ap.error("give an answers file, or --suggest")
    if not (spec.get("split") or {}).get("files"):
        sys.exit("No split yet. Run /ml-train-test-split first.")
    answers = ml_common.load(a.answers)
    nb_name = ml_common.notebook_name(STAGE)
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    if nb_common.state(path) == "edited" and not (a.overwrite or a.new_version):
        print(f"Not written: {ml_common.relpath(path, root)} was edited after the skill wrote it. Rerun with "
              f"--overwrite to replace it, or --new-version to write {os.path.basename(nb_common.next_version(path))}.")
        sys.exit(2)
    info, problem = install_code(spec, root, replace=a.replace_code)
    if problem:
        print("Not written: " + problem)
        sys.exit(3)
    spec.setdefault("features", {})
    spec["features"] = {**(spec.get("features") or {}), "code": {k: info[k] for k in ("features", "custom", "features_hash")}}
    ml_common.save(spec_path, spec)
    print(f"Code: {info['features']} {info['state']}, {info['custom']} {info['custom_state']}")
    nb = build(answers, spec, HERE, nb_name)
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    print(f"Wrote notebook: {ml_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if not res["executed"]:
        print("Spec: not updated. Open the notebook, fix the cell that failed, and choose Run All.")
        sys.exit(1 if res["error"] else 0)
    fe = ml_common.load(spec_path).get("features") or {}
    ch = fe.get("checks") or {}
    print(f"Spec: features written. Inputs: {fe.get('n_inputs')}. Outputs after the learned steps: {fe.get('outputs')}")
    print("Strongest single features: " + ", ".join(f"{r['feature']} {r['score']}" for r in (ch.get("strongest") or [])[:5]))
    print("Left out by the leakage check: " + (", ".join(ch.get("flagged") or []) or "none"))
    print("Large drift: " + (", ".join(ch.get("drift_large") or []) or "none"))
    print("Files: " + ", ".join((fe.get("files") or {}).values()))
    print("Problems: " + (" ".join(ch.get("problems") or []) or "none found"))


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "features", "ml-feature-engineering")
