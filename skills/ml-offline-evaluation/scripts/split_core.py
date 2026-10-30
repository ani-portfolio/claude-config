#!/usr/bin/env python3
"""Split engine for the ML project skills: the final test set, cross validation folds on the training set, and
the helpers later stages use to read them back.

Three strategies:
    random   rows are assigned at random, stratified by the target for classification. CV: stratified k fold.
    group    every row of an entity (for example a customer) lands on the same side. CV: group k fold.
    time     the latest period is the test set, with a gap before it, and CV uses rolling backtest folds: each
             fold trains on the past and validates on the next window, with the same gap. Used for forecasting,
             and for any problem where the model predicts the future from the past.

Exact duplicate rows are kept on the same side by default (random and group), so a copy of a test row is never
in training. Every file written is parquet:
    <data_processed>/train.parquet   training rows, with a "fold" column (the CV fold the row validates, -1 if none)
    <data_processed>/test.parquet    the final test set, used once at the end
    <data_processed>/gap.parquet     time strategy only: rows between training and test, kept as feature history
    <data_processed>/cv_folds.parquet  one row per fold: sizes, and date ranges for the time strategy

Later stages call cv_splits(train, spec["split"]) to get the (train, validation) row positions of each fold,
and load_sets(spec, root) to load the files.

The canonical copy lives in claude-skills/_ml_shared/split_core.py, copied into the modelling skills by
tools/sync_ml_shared.py.
"""

from __future__ import annotations

import math
import os
import re

import numpy as np
import pandas as pd

ROW_ID = "row_id"
FOLD = "fold"
STRATEGIES = ("random", "group", "time")
DUPLICATES = ("keep_together", "drop", "ignore")

DEFAULTS = {
    "strategy": "random",
    "test_size": 0.2,            # random and group: share of rows (or entities) in the test set
    "test_days": None,           # time: length of the test period in days
    "gap_days": 0,               # time: days left out between training and the test, and inside each fold
    "stratify": True,            # random and group: keep the class balance the same in every set
    "group_col": None,           # group: the entity column. Time: only used to report entity overlap
    "date_col": None,            # time: the column that orders the rows, usually the prediction time
    "cv": {"n_folds": 5, "window": "expanding", "valid_days": None, "train_days": None},
    "duplicates": "keep_together",
    "missing_target": "drop",
    "seed": 42,
}


# ---------------------------------------------------------------- settings

def to_days(text):
    """Days in a phrase such as '60 days after the scoring date', '8 weeks', or 28. None when there is no number."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)", str(text), re.I)
    if not m:
        return None
    return float(m.group(1)) * {"day": 1, "week": 7, "month": 30.4, "year": 365}[m.group(2).lower()]


def settings(user=None):
    """The defaults overlaid with the user's settings."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (user or {}).items():
        if k == "cv":
            out["cv"].update(v or {})
        else:
            out[k] = v
    if out["strategy"] not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}, not {out['strategy']!r}")
    if out["duplicates"] not in DUPLICATES:
        raise ValueError(f"duplicates must be one of {', '.join(DUPLICATES)}")
    return out


def is_classification(task_type, y=None):
    if task_type:
        return "classification" in str(task_type)
    return y is not None and y.nunique(dropna=True) <= 20


def recommend(spec, df=None):
    """Recommended split settings from the spec (and the data, when given), with the reason for each choice.
    Returns (settings, reasons)."""
    fr = spec.get("framing") or {}
    data = spec.get("data") or {}
    eda = spec.get("eda") or {}
    tgt = fr.get("target") or {}
    task = fr.get("task_type")
    pt = fr.get("prediction_time") or {}
    date_col = data.get("date_col") or pt.get("column")
    ids = list(fr.get("entity_id_cols") or [])
    s = settings()
    reasons = []
    if df is not None and date_col and date_col not in df.columns:
        reasons.append(f"The date column `{date_col}` from the spec is not in the data, so it was not used.")
        date_col = None
    history = None
    if df is not None and date_col:
        d = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        history = (d.max() - d.min()).days if d.notna().any() else None

    if task == "forecasting" or date_col:
        s["strategy"] = "time"
        s["date_col"] = date_col
        s["group_col"] = ids[0] if ids else None
        if task == "forecasting":
            horizon = to_days(pt.get("horizon")) or 28
            s["gap_days"] = 0
            s["test_days"] = int(horizon)
            s["cv"].update({"n_folds": 3, "valid_days": int(horizon), "window": "expanding"})
            reasons.append("Forecasting: the test set is the latest period, one forecast horizon long, and CV uses "
                           "rolling backtest folds of the same length. No gap is needed, because feature engineering "
                           f"shifts lags by at least the horizon ({int(horizon)} days).")
        else:
            window, delay = to_days(tgt.get("outcome_window")), to_days(tgt.get("label_delay"))
            gap = int(math.ceil(max(window or 0, delay or 0)))
            test_days = 90
            if history:
                test_days = int(max(14, min(90, round(history * 0.2))))
            s["gap_days"] = gap
            s["test_days"] = test_days
            s["cv"].update({"n_folds": 3, "valid_days": test_days, "window": "expanding"})
            reasons.append(f"The prediction is made at a point in time (`{date_col}`), so the test set is the latest "
                           f"{test_days} days, as in production, and CV uses rolling backtest folds.")
            if gap:
                reasons.append(f"A gap of {gap} days (the outcome window or label delay, whichever is longer) keeps "
                               "training labels from looking into the test period.")
        if history and s["test_days"]:
            # The first backtest fold should still train on at least 40% of the training period. Use fewer folds,
            # then shorter validation windows (not under 28 days, or the horizon for forecasting), until it does.
            gap_d, cv = s["gap_days"], s["cv"]
            available = history - s["test_days"] - gap_d
            floor = cv["valid_days"] if task == "forecasting" else 28
            first_train = lambda n, v: available - n * (v + gap_d)
            while cv["n_folds"] > 2 and first_train(cv["n_folds"], cv["valid_days"]) < 0.4 * available:
                cv["n_folds"] -= 1
            if first_train(cv["n_folds"], cv["valid_days"]) < 0.4 * available:
                v = int((0.6 * available) / cv["n_folds"] - gap_d)
                cv["valid_days"] = max(floor, v)
            ft = first_train(cv["n_folds"], cv["valid_days"])
            reasons.append(f"With {history} days of history, {cv['n_folds']} folds of {cv['valid_days']} days leave "
                           f"about {max(ft, 0):.0f} days of training data in the first fold.")
            if ft < 0.25 * available:
                reasons.append("Warning: the history is short for this test period and gap, so the first fold has "
                               "little training data. Consider a shorter test period, or a single validation fold.")
    elif ids and df is not None and ids[0] in df.columns and df[ids[0]].duplicated().any():
        s["strategy"] = "group"
        s["group_col"] = ids[0]
        reasons.append(f"`{ids[0]}` appears in more than one row, so all rows of an entity stay on one side. "
                       "Otherwise the model could memorise entities instead of learning patterns.")
    else:
        s["strategy"] = "random"
        reasons.append("There is no time order or repeated entity to respect, so rows are assigned at random.")
    if s["strategy"] != "time":
        s["stratify"] = is_classification(task)
        if s["stratify"]:
            reasons.append("Classification: every set keeps the same class balance (stratified).")
    dups = eda.get("duplicate_rows") or 0
    if dups and s["strategy"] != "time":
        reasons.append(f"EDA found {dups:,} exact duplicate rows. They are kept together on one side, so a copy "
                       "of a test row is never in training.")
    return s, reasons


# ---------------------------------------------------------------- preparing the data

def prepare(df, target, split_settings):
    """Add the row ID, handle missing targets and exact duplicates. Returns (df, info, duplicate_key).

    duplicate_key is a Series that is equal for exact duplicate rows (used to keep them together), or None."""
    s = settings(split_settings)
    info = {"rows_in": int(len(df))}
    df = df.copy()
    row_id = ROW_ID if ROW_ID not in df.columns else "_" + ROW_ID
    df.insert(0, row_id, np.arange(len(df)))
    info["row_id_col"] = row_id
    if target not in df.columns:
        raise ValueError(f"The target column {target!r} is not in the data. Build it in the Target cell first.")
    miss = int(df[target].isna().sum())
    info["missing_target"] = miss
    if miss:
        if s["missing_target"] != "drop":
            raise ValueError(f"{miss:,} rows have no target. Set missing_target to 'drop', or fill them first.")
        df = df[df[target].notna()]
    content = df.drop(columns=[row_id])
    key = pd.Series(pd.util.hash_pandas_object(content, index=False).values, index=df.index)
    dup = int(key.duplicated().sum())
    info["duplicate_rows"] = dup
    info["duplicates_dropped"] = 0
    if dup and s["duplicates"] == "drop":
        keep = ~key.duplicated()
        df, key = df[keep], key[keep]
        info["duplicates_dropped"] = dup
    info["rows_out"] = int(len(df))
    use_key = key if dup and s["duplicates"] == "keep_together" else None
    return df.reset_index(drop=True), info, (use_key.reset_index(drop=True) if use_key is not None else None)


# ---------------------------------------------------------------- making the split

def _stratify_ok(y, n_splits):
    counts = y.value_counts()
    return len(counts) >= 2 and counts.min() >= n_splits


def _test_mask(df, y, groups, s, classification):
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
    idx = np.arange(len(df))
    strat = s["stratify"] and classification
    if groups is None:
        n = max(2, round(1 / s["test_size"]))
        _, test_idx = train_test_split(idx, test_size=s["test_size"], random_state=s["seed"],
                                       stratify=y if strat and _stratify_ok(y, n) else None)
    elif strat and _stratify_ok(y, max(2, round(1 / s["test_size"]))):
        sgk = StratifiedGroupKFold(n_splits=max(2, round(1 / s["test_size"])), shuffle=True, random_state=s["seed"])
        _, test_idx = next(sgk.split(idx, y, groups))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=s["test_size"], random_state=s["seed"])
        _, test_idx = next(gss.split(idx, y, groups))
    mask = np.zeros(len(df), dtype=bool)
    mask[test_idx] = True
    return mask


def _kfold(train, y, groups, s, classification):
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
    k = int(s["cv"]["n_folds"])
    strat = s["stratify"] and classification and _stratify_ok(y, k)
    if groups is None:
        method = "stratified_kfold" if strat else "kfold"
        splitter = (StratifiedKFold if strat else KFold)(n_splits=k, shuffle=True, random_state=s["seed"])
        parts = splitter.split(np.zeros(len(train)), y if strat else None)
    else:
        method = "stratified_group_kfold" if strat else "group_kfold"
        if strat:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s["seed"])
            parts = splitter.split(np.zeros(len(train)), y, groups)
        else:
            parts = GroupKFold(n_splits=k).split(np.zeros(len(train)), y, groups)
    fold = np.full(len(train), -1)
    for i, (_, v) in enumerate(parts):
        fold[v] = i
    return fold, method


def _rows_by_fold(train, y, classification):
    rows = []
    for f in sorted(set(train[FOLD]) - {-1}):
        v = train[FOLD] == f
        r = {"fold": int(f), "n_train": int((~v).sum()), "n_valid": int(v.sum())}
        if classification:
            r["valid_class_shares"] = {str(k): round(float(x), 4) for k, x in y[v].value_counts(normalize=True).items()}
        rows.append(r)
    return pd.DataFrame(rows)


def make_split(df, target, task_type, split_settings, duplicate_key=None):
    """Split prepared data (see prepare). Returns a dict with train, test, gap (DataFrames), folds (one row per CV
    fold), cv_method, and info."""
    s = settings(split_settings)
    classification = is_classification(task_type, df[target])
    info = {}
    if s["strategy"] == "time":
        return _time_split(df, target, s, classification)
    if s["strategy"] == "group":
        col = s["group_col"]
        if not col or col not in df.columns:
            raise ValueError(f"The group strategy needs group_col set to a column in the data, not {col!r}.")
        groups = df[col].astype(str)
        if duplicate_key is not None:  # rows that are exact copies share an entity already, unless the entity is missing
            groups = groups.where(df[col].notna(), "dup_" + duplicate_key.astype(str))
    elif duplicate_key is not None:
        return _random_keep_duplicates(df, target, s, classification, duplicate_key)
    else:
        groups = None
    y = df[target]
    test_mask = _test_mask(df, y, groups, s, classification)
    train, test = df[~test_mask].copy(), df[test_mask].copy()
    g_train = groups[~test_mask].values if groups is not None else None
    train[FOLD], method = _kfold(train, train[target], g_train, s, classification)
    info["test_share"] = round(float(test_mask.mean()), 4)
    folds = _rows_by_fold(train, train[target], classification)
    return {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True),
            "gap": df.iloc[0:0].copy(), "folds": folds, "cv_method": method, "info": info}


def _random_keep_duplicates(df, target, s, classification, key):
    """Random split where exact duplicates follow their first copy: split the distinct rows, then give every copy
    the same set and fold. Much faster than a group split with one group per row."""
    work = df.assign(_dup_key=key.values)
    first = ~work["_dup_key"].duplicated()
    task = "binary_classification" if classification else "regression"
    res = make_split(work[first].reset_index(drop=True), target, task, {**s, "duplicates": "ignore"})
    is_test = work["_dup_key"].isin(set(res["test"]["_dup_key"]))
    fold_of = dict(zip(res["train"]["_dup_key"], res["train"][FOLD]))
    train, test = work[~is_test].copy(), work[is_test].copy()
    train[FOLD] = train["_dup_key"].map(fold_of).astype(int)
    train, test = train.drop(columns="_dup_key"), test.drop(columns="_dup_key")
    return {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True), "gap": df.iloc[0:0].copy(),
            "folds": _rows_by_fold(train, train[target], classification), "cv_method": res["cv_method"],
            "info": {"test_share": round(float(is_test.mean()), 4)}}


def _fmt(ts):
    return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.isoformat()


def _time_split(df, target, s, classification):
    col = s["date_col"]
    if not col or col not in df.columns:
        raise ValueError(f"The time strategy needs date_col set to a column in the data, not {col!r}.")
    if not s["test_days"]:
        raise ValueError("The time strategy needs test_days, the length of the test period in days.")
    d = pd.to_datetime(df[col], errors="coerce", format="mixed")
    info = {"unreadable_dates": int(d.isna().sum())}
    df, d = df[d.notna()].copy(), d[d.notna()]
    gap = pd.Timedelta(days=float(s["gap_days"] or 0))
    test_from = d.max() - pd.Timedelta(days=float(s["test_days"]))   # test: dates after this
    train_to = test_from - gap                                       # train: dates up to and including this
    is_test, is_train = d > test_from, d <= train_to
    train, test, gap_rows = df[is_train].copy(), df[is_test].copy(), df[~is_test & ~is_train].copy()
    dt = d[is_train]

    cv = s["cv"]
    k = int(cv["n_folds"])
    valid = pd.Timedelta(days=float(cv.get("valid_days") or s["test_days"]))
    train_len = pd.Timedelta(days=float(cv["train_days"])) if cv.get("window") == "sliding" and cv.get("train_days") else None
    fold = np.full(len(train), -1)
    rows = []
    last = dt.max()
    for i in range(k):
        v_to = last - valid * (k - 1 - i)
        v_from = v_to - valid                       # validation: v_from < date <= v_to
        t_to = v_from - gap                         # training: date <= t_to
        tr = (dt <= t_to) & ((dt > t_to - train_len) if train_len is not None else True)
        va = (dt > v_from) & (dt <= v_to)
        if not tr.any() or not va.any():
            raise ValueError(f"Fold {i} has {int(tr.sum())} training and {int(va.sum())} validation rows. Use fewer "
                             "folds, shorter validation windows, or a shorter gap.")
        fold[va.values] = i
        r = {"fold": i, "n_train": int(tr.sum()), "n_valid": int(va.sum()),
             "train_start": _fmt(dt[tr].min()), "train_end": _fmt(dt[tr].max()),
             "valid_start": _fmt(dt[va].min()), "valid_end": _fmt(dt[va].max())}
        if classification:
            r["valid_class_shares"] = {str(a): round(float(b), 4) for a, b in
                                       train.loc[va.values, target].value_counts(normalize=True).items()}
        rows.append(r)
    train[FOLD] = fold
    info.update({"test_from_exclusive": test_from.strftime("%Y-%m-%d"), "train_to": train_to.strftime("%Y-%m-%d")})
    return {"train": train.reset_index(drop=True), "test": test.reset_index(drop=True),
            "gap": gap_rows.reset_index(drop=True), "folds": pd.DataFrame(rows),
            "cv_method": "time_backtest", "info": info}


# ---------------------------------------------------------------- saving and reading back

def _to_parquet(df, path):
    """Write parquet. Object columns that mix types (which parquet cannot store) are written as text, and their
    names are returned."""
    try:
        df.to_parquet(path, index=False)
        return []
    except Exception:  # pyarrow raises ArrowTypeError or ArrowInvalid for mixed object columns
        fixed = df.copy()
        changed = []
        for c in fixed.columns:
            if fixed[c].dtype == object:
                kinds = fixed[c].dropna().map(type).nunique()
                if kinds > 1:
                    fixed[c] = fixed[c].where(fixed[c].isna(), fixed[c].astype(str))
                    changed.append(c)
        fixed.to_parquet(path, index=False)
        return changed


def save(result, out_dir):
    """Write train, test, gap (when not empty), and cv_folds as parquet. Returns ({name: path}, columns written
    as text)."""
    os.makedirs(out_dir, exist_ok=True)
    paths, text_cols = {}, set()
    for name in ("train", "test", "gap"):
        frame = result[name]
        p = os.path.join(out_dir, f"{name}.parquet")
        if name == "gap" and frame.empty:
            if os.path.exists(p):
                os.remove(p)   # an older run's gap file would be misleading
            continue
        text_cols.update(_to_parquet(frame, p))
        paths[name] = p
    folds = result["folds"].copy()
    if "valid_class_shares" in folds:
        folds["valid_class_shares"] = folds["valid_class_shares"].map(str)
    p = os.path.join(out_dir, "cv_folds.parquet")
    folds.to_parquet(p, index=False)
    paths["folds"] = p
    return paths, sorted(text_cols)


def plain(obj):
    """The same value with numpy and pandas scalars turned into plain Python, so it can be written to the spec."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return _fmt(obj)
    return obj


def load_sets(spec, root="."):
    """Load the split files named in spec['split']['files'] as a dict of DataFrames (train, test, gap, folds)."""
    files = (spec.get("split") or {}).get("files") or {}
    return {k: pd.read_parquet(os.path.join(root, p)) for k, p in files.items() if p}


def cv_splits(train, split_section):
    """The CV folds of a training set as a list of (train_positions, validation_positions), in fold order.

    For random and group splits, a row validates the fold in its "fold" column and trains in every other fold.
    For the time strategy, each fold trains on the dates in its training range and validates on its validation
    range, as recorded in split_section['cv']['folds'], so rows after a fold's validation window are never used to
    train it."""
    cv = split_section.get("cv") or {}
    out = []
    if cv.get("method") == "time_backtest":
        d = pd.to_datetime(train[split_section["date_col"]], errors="coerce", format="mixed")
        for f in cv.get("folds") or []:
            tr = (d >= pd.Timestamp(f["train_start"])) & (d <= pd.Timestamp(f["train_end"]))
            va = (d >= pd.Timestamp(f["valid_start"])) & (d <= pd.Timestamp(f["valid_end"]))
            out.append((np.flatnonzero(tr.values), np.flatnonzero(va.values)))
        return out
    fold = train[FOLD].values
    for f in sorted(set(fold) - {-1}):
        out.append((np.flatnonzero(fold != f), np.flatnonzero(fold == f)))
    return out
