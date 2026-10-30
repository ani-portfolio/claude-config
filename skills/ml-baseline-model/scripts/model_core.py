#!/usr/bin/env python3
"""Model engine for the ML project skills: metrics, baseline and model definitions, and cross validation on the
folds from the split.

Metrics are named by key. Binary classification: roc_auc, pr_auc, log_loss, brier, recall_at_k, precision_at_k,
lift_at_k. Regression and forecasting: rmse, mae, mape, wape, r2, bias. The "at k" metrics take k (rows per group)
or k_share (a share of rows per group), and k_per (a column such as the scoring date, so k is applied within each
date, as when 500 customers are contacted every week). Without k_per, k applies to all rows at once.

A model is a dict: {"name", "kind" ("non_ml" or "ml"), "type", "params", "reason"}, and for ML models an
optional "inputs" list: a subset of the feature inputs that model uses (for example a model without the media
features). Without it, the model uses every input. Types:
    prior      the average of the target in the fold's training rows (base rate, or mean)
    rule       a pandas expression on the row, for example "days_since_last_order > 30" (the current business rule)
    column     one column used as the score or the prediction, for example the best single feature, or a lag of the
               target as a naive forecast ("sales_lag_28d"); params.direction is "higher" (default) or "lower"
    segment    the target average per segment (params.by: columns), learned on the fold's training rows
    logistic   logistic regression on the linear preprocessor (classification)
    ridge      ridge regression on the linear preprocessor (regression and forecasting)
    ridge_nonneg  ridge regression whose coefficients on the inputs in params.nonneg are kept at zero or above, the
               others free (for example media effects in a marketing mix model); regression and forecasting
    tree       a shallow decision tree (params.max_depth, default 4) on the tree preprocessor
    gbm        gradient boosting with default settings (HistGradientBoosting) on the tree preprocessor

cross_validate(models, data, inputs, target, folds, metrics, task, config) fits every model inside every fold and
returns the per fold metrics and the out of fold predictions. Nothing here touches the test set.

The canonical copy lives in skills/_ml_shared/model_core.py, copied into the modelling skills by
tools/sync_ml_shared.py.
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np
import pandas as pd

BINARY_METRICS = ("roc_auc", "pr_auc", "log_loss", "brier", "recall_at_k", "precision_at_k", "lift_at_k")
REGRESSION_METRICS = ("rmse", "mae", "mape", "wape", "r2", "bias")
HIGHER_IS_BETTER = {"roc_auc": True, "pr_auc": True, "log_loss": False, "brier": False, "recall_at_k": True,
                    "precision_at_k": True, "lift_at_k": True, "rmse": False, "mae": False, "mape": False,
                    "wape": False, "r2": True, "bias": None}
NON_ML_TYPES = ("prior", "rule", "column", "segment")


def task_family(task_type):
    t = str(task_type or "")
    if "multiclass" in t:
        return "multiclass"
    if "classification" in t:
        return "binary"
    return "regression"


# ---------------------------------------------------------------- metrics

def parse_metric(text, date_col=None):
    """A metric key and its settings from a phrase. See ml_common.parse_metric."""
    import ml_common
    return ml_common.parse_metric(text, date_col)


def _top_k_mask(score, k, k_share, groups, rng):
    """True for the k highest scores (within each group when groups is given). Ties are broken at random."""
    s = pd.Series(np.asarray(score, dtype=float)).fillna(-np.inf) + rng.random(len(score)) * 1e-9
    if groups is None:
        n = int(k) if k else int(np.ceil(k_share * len(s)))
        return s.rank(ascending=False, method="first").le(n).values
    g = pd.Series(np.asarray(groups))
    ranks = s.groupby(g.values).rank(ascending=False, method="first")
    if k:
        return ranks.le(k).values
    sizes = g.map(g.value_counts())
    return (ranks <= np.ceil(k_share * sizes)).values


def top_k_weights(score, k, k_share, groups=None):
    """The chance that each row is among the k highest scores (within each group when groups is given), with ties
    at the cut off picked at random: 1 above the cut off, 0 below, and the share of free places for tied rows."""
    s = pd.Series(np.asarray(score, dtype=float)).fillna(-np.inf)
    g = pd.Series(np.zeros(len(s)) if groups is None else np.asarray(groups))
    by = s.groupby(g.values)
    lo, hi = by.rank(ascending=False, method="min").values, by.rank(ascending=False, method="max").values
    size = g.map(g.value_counts()).values
    n = np.minimum(np.full(len(s), float(k)) if k else np.ceil(k_share * size), size)
    w = np.where(hi <= n, 1.0, np.where(lo > n, 0.0, (n - lo + 1) / (hi - lo + 1)))
    return w


def score_metric(spec, y, score, groups=None, probabilistic=True, seed=0):
    """One metric. spec is a key or a dict with key, k, k_share, k_per. Returns a float, or NaN when the metric does
    not apply (for example log loss of a score that is not a probability)."""
    from sklearn import metrics as skm
    m = {"key": spec} if isinstance(spec, str) else dict(spec)
    key = m["key"]
    y = np.asarray(y, dtype=float)
    p = np.asarray(score, dtype=float)
    ok = ~np.isnan(p) if key in REGRESSION_METRICS else np.ones(len(p), dtype=bool)
    if key == "roc_auc":
        return float(skm.roc_auc_score(y, np.nan_to_num(p, nan=np.nanmin(p) - 1))) if len(set(y)) > 1 else np.nan
    if key == "pr_auc":
        return float(skm.average_precision_score(y, np.nan_to_num(p, nan=np.nanmin(p) - 1))) if y.sum() else np.nan
    if key in ("log_loss", "brier"):
        if not probabilistic:
            return np.nan
        q = np.clip(np.nan_to_num(p, nan=np.nanmean(p)), 1e-6, 1 - 1e-6)
        return float(skm.log_loss(y, q, labels=[0, 1]) if key == "log_loss" else skm.brier_score_loss(y, q))
    if key in ("recall_at_k", "precision_at_k", "lift_at_k"):
        if not m.get("k") and not m.get("k_share"):
            raise ValueError(f"{key} needs k or k_share.")
        # Rows tied at the cut off are counted by their chance of being picked, so a rule or a constant with many
        # ties gets its expected value instead of one lucky or unlucky random draw.
        w = top_k_weights(p, m.get("k"), m.get("k_share"), groups)
        hits = float((w * y).sum())
        if key == "recall_at_k":
            return hits / y.sum() if y.sum() else np.nan
        prec = hits / w.sum() if w.sum() else np.nan
        return prec if key == "precision_at_k" else prec / y.mean() if y.mean() else np.nan
    yt, yp = y[ok], p[ok]
    if not len(yt):
        return np.nan
    if key == "rmse":
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    if key == "mae":
        return float(np.mean(np.abs(yt - yp)))
    if key == "mape":
        nz = yt != 0
        return float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz]))) if nz.any() else np.nan
    if key == "wape":
        return float(np.sum(np.abs(yt - yp)) / np.sum(np.abs(yt))) if np.sum(np.abs(yt)) else np.nan
    if key == "r2":
        return float(skm.r2_score(yt, yp))
    if key == "bias":
        return float(np.mean(yp - yt))
    raise ValueError(f"Unknown metric {key!r}")


def metric_label(m):
    m = {"key": m} if isinstance(m, str) else m
    key = m["key"]
    if key.endswith("_at_k"):
        k = f"{m['k']:,}" if m.get("k") else f"{m['k_share']:.0%}"
        return f"{key.replace('_at_k', '')} at {k}" + (f" per {m['k_per']}" if m.get("k_per") else "")
    return key


def all_metrics(metrics):
    """The primary metric first, then the secondary ones, as dicts."""
    out = [metrics["primary"]] + list(metrics.get("secondary") or [])
    return [{"key": m} if isinstance(m, str) else m for m in out]


# ---------------------------------------------------------------- models

class NonNegRidge:
    """Ridge regression, minimising ||y - Xb - c||^2 + alpha ||b||^2 as sklearn's Ridge does, with the coefficients
    of the columns named in nonneg bounded at zero or above. The intercept and every other coefficient are free.
    Solved exactly as bounded least squares (scipy.optimize.lsq_linear). X must be a DataFrame when nonneg is set."""

    _estimator_type = "regressor"

    def __init__(self, alpha=1.0, nonneg=()):
        self.alpha = alpha
        self.nonneg = nonneg

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "nonneg": self.nonneg}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_tags__(self):
        from sklearn.base import BaseEstimator, RegressorMixin

        class _Tmp(RegressorMixin, BaseEstimator):
            pass
        return _Tmp().__sklearn_tags__()

    def fit(self, X, y):
        from scipy.optimize import lsq_linear
        cols = list(X.columns) if hasattr(X, "columns") else None
        Xa, ya = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, k = Xa.shape
        lower = np.full(k, -np.inf)
        if self.nonneg:
            if cols is None:
                raise ValueError("NonNegRidge needs a DataFrame to find the columns named in nonneg.")
            missing = [c for c in self.nonneg if c not in cols]
            if missing:
                raise ValueError(f"nonneg columns not in the inputs: {', '.join(missing)}")
            lower[[cols.index(c) for c in self.nonneg]] = 0.0
        xm, ym = Xa.mean(axis=0), ya.mean()
        A = np.vstack([Xa - xm, np.sqrt(float(self.alpha)) * np.eye(k)])
        b = np.concatenate([ya - ym, np.zeros(k)])
        res = lsq_linear(A, b, bounds=(lower, np.full(k, np.inf)))
        coef = res.x.copy()
        coef[(res.active_mask != 0) & np.isfinite(lower)] = 0.0   # held at the bound: exactly zero, not 1e-31
        self.coef_ = coef
        self.intercept_ = float(ym - xm @ res.x)
        self.n_features_in_ = k
        if cols is not None:
            self.feature_names_in_ = np.array(cols, dtype=object)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


def fold_pipelines(model, data, folds, inputs, target, family, config):
    """The model fitted on the training rows of each CV fold, in fold order."""
    out = []
    for tr, _ in folds:
        train = data.iloc[tr]
        out.append(make_estimator(model, family, config, train[target]).fit(train[inputs], train[target]))
    return out


def coefficients(pipe, raw_units=False):
    """The coefficients of a fitted linear pipeline by input name, or None for other models. With raw_units, they are
    divided by the scaler's spread, so each is the change in the prediction per unit of the input."""
    est, prep = pipe.named_steps.get("model"), pipe.named_steps.get("prep")
    coef = getattr(est, "coef_", None)
    if coef is None:
        return None
    coef = np.ravel(coef)
    try:
        names = list(prep.get_feature_names_out())
    except Exception:  # noqa: BLE001
        names = list(getattr(est, "feature_names_in_", range(len(coef))))
    s = pd.Series(coef, index=names[:len(coef)])
    if raw_units:
        try:
            num = prep.named_transformers_["numeric"]
            scale = num.named_steps["scale"].scale_
            cols = list(num.get_feature_names_out())[:len(scale)]
            s.loc[cols] = s.loc[cols].values / scale
        except Exception:  # noqa: BLE001  no scaler: the coefficients are already in raw units
            pass
    return s


class V1Context:
    """What a check function in the project's src/v1_checks.py receives. The fitted champion and the per fold fits
    are made only when a check asks for them."""

    def __init__(self, spec, root, on, model, data, inputs, target, family, config, folds, pipeline=None, test=None):
        self.spec, self.root, self.on, self.model = spec, root, on, model
        self.data, self.inputs, self.target, self.family, self.config, self.folds = data, inputs, target, family, config, folds
        self.test = test
        self._pipeline, self._folds = pipeline, None

    @property
    def pipeline(self):
        """The champion fitted on every training row."""
        if self._pipeline is None:
            self._pipeline = make_estimator(self.model, self.family, self.config, self.data[self.target]).fit(
                self.data[self.inputs], self.data[self.target])
        return self._pipeline

    @property
    def fold_pipelines(self):
        """The champion fitted on the training rows of each CV fold."""
        if self._folds is None:
            self._folds = fold_pipelines(self.model, self.data, self.folds, self.inputs, self.target, self.family,
                                         self.config)
        return self._folds

    def coefficients(self, pipe=None, raw_units=False):
        return coefficients(pipe if pipe is not None else self.pipeline, raw_units)


def run_v1_checks(ctx):
    """Run the check criteria of the V1 bar judged on ctx.on. Returns {function: (passed or None, detail)}: None
    when the function is missing or fails, so the bar is reported incomplete, never failed, for that reason."""
    import importlib
    import os

    import ml_common
    crit = [c for c in ((ctx.spec.get("framing") or {}).get("success") or {}).get("v1_criteria") or []
            if isinstance(c, dict) and c.get("kind") == "check" and ml_common.criterion_on(c) == ctx.on]
    if not crit:
        return {}
    if ctx.root not in sys.path:
        sys.path.insert(0, ctx.root)
    try:
        if not os.path.exists(os.path.join(ctx.root, "src", "v1_checks.py")):
            raise ImportError("src/v1_checks.py does not exist")
        mod = importlib.reload(importlib.import_module("src.v1_checks")) if "src.v1_checks" in sys.modules \
            else importlib.import_module("src.v1_checks")
    except Exception as e:  # noqa: BLE001
        return {c["function"]: (None, f"Could not import src/v1_checks.py: {e}") for c in crit}
    out = {}
    for c in crit:
        fn = getattr(mod, c["function"], None)
        if fn is None:
            out[c["function"]] = (None, f"{c['function']} is not defined in src/v1_checks.py.")
            continue
        try:
            res = fn(ctx, **(c.get("params") or {}))
            passed, detail = (res if isinstance(res, tuple) else (bool(res), ""))
            out[c["function"]] = (None if passed is None else bool(passed), str(detail))
        except Exception as e:  # noqa: BLE001
            out[c["function"]] = (None, f"{c['function']} failed: {type(e).__name__}: {e}")
    return out


def save_model(pipe, path):
    """Save a fitted pipeline so it loads with joblib.load anywhere the project's src and scikit-learn are
    installed, without the skills. Estimators defined here (such as NonNegRidge) are stored by value with
    cloudpickle, a dependency of joblib. Returns the serialization name recorded in the spec."""
    import cloudpickle
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    with open(path, "wb") as f:
        cloudpickle.dump(pipe, f)
    return "cloudpickle_by_value"


def _preprocessor(kind, config):
    from src.features import make_preprocessor
    return make_preprocessor(kind, config)


# ML model types: (preprocessor kind, package to import, classifier, regressor, default settings)
ML_MODELS = {
    "logistic": ("linear", "sklearn", "sklearn.linear_model.LogisticRegression", None, {"max_iter": 2000, "C": 1.0}),
    "ridge": ("linear", "sklearn", None, "sklearn.linear_model.Ridge", {"alpha": 1.0}),
    "ridge_nonneg": ("linear", "scipy", None, "model_core.NonNegRidge", {"alpha": 1.0, "nonneg": []}),
    "tree": ("tree", "sklearn", "sklearn.tree.DecisionTreeClassifier", "sklearn.tree.DecisionTreeRegressor",
             {"max_depth": 4, "min_samples_leaf": 50, "random_state": 0}),
    "random_forest": ("tree", "sklearn", "sklearn.ensemble.RandomForestClassifier", "sklearn.ensemble.RandomForestRegressor",
                      {"n_estimators": 300, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": -1, "random_state": 0}),
    "hist_gbm": ("tree", "sklearn", "sklearn.ensemble.HistGradientBoostingClassifier",
                 "sklearn.ensemble.HistGradientBoostingRegressor", {"random_state": 0}),
    "lightgbm": ("tree", "lightgbm", "lightgbm.LGBMClassifier", "lightgbm.LGBMRegressor",
                 {"n_estimators": 300, "learning_rate": 0.05, "verbose": -1, "random_state": 0, "n_jobs": -1}),
    "xgboost": ("tree", "xgboost", "xgboost.XGBClassifier", "xgboost.XGBRegressor",
                {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "verbosity": 0, "random_state": 0, "n_jobs": -1}),
    "catboost": ("tree", "catboost", "catboost.CatBoostClassifier", "catboost.CatBoostRegressor",
                 {"iterations": 300, "learning_rate": 0.05, "verbose": 0, "random_seed": 0, "allow_writing_files": False}),
}
ML_MODELS["gbm"] = ML_MODELS["hist_gbm"]   # the name used by earlier baselines
ML_TYPES = tuple(ML_MODELS)
PROBABILISTIC = {"prior", "segment"} | {t for t, v in ML_MODELS.items() if v[2]}


def available(model_type):
    """True when the package a model type needs is installed."""
    import importlib.util
    pkg = ML_MODELS[model_type][1]
    return importlib.util.find_spec(pkg) is not None


def _class(path):
    import importlib
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def make_estimator(model, family, config, y=None):
    """An unfitted sklearn pipeline (preprocessor and model) for an ML model. params override the defaults.
    model["class_weight"] = "balanced" weights the classes by their frequency (classification only)."""
    from sklearn.pipeline import Pipeline
    t = model["type"]
    if t not in ML_MODELS:
        raise ValueError(f"{t!r} is not an ML model type. Use one of: {', '.join(ML_MODELS)}")
    kind, pkg, clf, reg, defaults = ML_MODELS[t]
    path = clf if family == "binary" else reg
    if path is None:
        raise ValueError(f"{t} is for {'regression' if family == 'binary' else 'classification'}; use "
                         f"{'logistic' if family == 'binary' else 'ridge'} instead.")
    kw = {**defaults, **dict(model.get("params") or {})}
    config = model_config(model, config)
    if model.get("class_weight") == "balanced" and family == "binary":
        if t in ("logistic", "tree", "random_forest", "hist_gbm", "gbm", "lightgbm"):
            kw["class_weight"] = "balanced"
        elif t == "xgboost" and y is not None:
            kw["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
        elif t == "catboost":
            kw["auto_class_weights"] = "Balanced"
    pipe = Pipeline([("prep", _preprocessor(kind, config)), ("model", _class(path)(**kw))])
    if t == "ridge_nonneg":   # the model needs the column names to know which coefficients to bound
        pipe.named_steps["prep"].set_output(transform="pandas")
    return pipe


def model_config(model, config):
    """The feature config for one model: with model["inputs"], every other input is excluded, so the preprocessor
    passes only that subset to the model. Without it, the config is returned as it is."""
    subset = model.get("inputs")
    if not subset:
        return config
    from src.features import feature_columns
    every = feature_columns(config)
    missing = [c for c in subset if c not in every]
    if missing:
        raise ValueError(f"{model.get('name')}: inputs not in the feature set: {', '.join(missing)}")
    drop = [c for c in every if c not in set(subset)]
    return {**config, "exclude": sorted(set(config.get("exclude") or []) | set(drop))}


def fit_predict(model, train, valid, inputs, target, family, config):
    """Fit one model on train and predict valid. Returns predictions as a float array: the probability of the
    positive class for binary classification, the value for regression."""
    t, p = model["type"], dict(model.get("params") or {})
    y = train[target]
    if t == "prior":
        return np.full(len(valid), float(y.mean()))
    if t == "rule":
        out = valid.eval(p["expression"])
        return pd.Series(out, index=valid.index).astype(float).values
    if t == "column":
        v = pd.to_numeric(valid[p["column"]], errors="coerce").astype(float)
        if family == "regression":
            fallback = pd.to_numeric(train[p["column"]], errors="coerce").mean()
            v = v.fillna(fallback if pd.notna(fallback) else y.mean())
        sign = -1.0 if p.get("direction") == "lower" else 1.0
        return (sign * v).values
    if t == "segment":
        by = list(p["by"])
        key = lambda d: d[by].astype("object").where(d[by].notna(), "missing").astype(str).agg("|".join, axis=1)
        rates = y.groupby(key(train).values).mean()
        return key(valid).map(rates).fillna(float(y.mean())).astype(float).values
    est = make_estimator(model, family, config, y)
    est.fit(train[inputs], y)
    if family == "binary":
        return est.predict_proba(valid[inputs])[:, 1]
    return est.predict(valid[inputs]).astype(float)


def check_model(model, data, inputs, excluded):
    """Problems with a model definition, in plain sentences."""
    t, p = model.get("type"), model.get("params") or {}
    out = []
    if t not in NON_ML_TYPES + ML_TYPES:
        return [f"{model.get('name')}: unknown type {t!r}."]
    used = []
    if t == "rule":
        if not p.get("expression"):
            return [f"{model.get('name')}: the rule has no expression yet."]
        used = [c for c in data.columns if re.search(rf"\b{re.escape(c)}\b", p["expression"])]
    elif t == "column":
        used = [p.get("column")]
    elif t == "segment":
        used = list(p.get("by") or [])
    for c in used:
        if c not in data.columns:
            out.append(f"{model['name']}: column {c!r} is not in the data.")
        elif c in excluded:
            out.append(f"{model['name']}: column {c!r} is excluded ({excluded[c]}), so a baseline must not use it either.")
    return out


def cross_validate(models, data, inputs, target, folds, metrics, task_type, config, group_col=None, log=print):
    """Fit and score every model in every fold. Returns (per_fold DataFrame, out of fold predictions DataFrame).
    group_col is the column the "at k" metrics group by (k_per)."""
    family = task_family(task_type)
    if family == "multiclass":
        raise ValueError("Multiclass baselines are not supported yet. Frame the task as binary, or ask for this to be added.")
    mets = all_metrics(metrics)
    rows, preds = [], []
    for model in models:
        start = time.time()
        for f, (tr, va) in enumerate(folds):
            train, valid = data.iloc[tr], data.iloc[va]
            p = fit_predict(model, train, valid, inputs, target, family, config)
            row = {"model": model["name"], "kind": model.get("kind"), "type": model["type"], "fold": f,
                   "n_train": len(tr), "n_valid": len(va)}
            for m in mets:
                groups = valid[m["k_per"]].values if m.get("k_per") else None
                row[metric_label(m)] = score_metric(m, valid[target].values, p, groups,
                                                    probabilistic=model["type"] in PROBABILISTIC or family == "regression")
            rows.append(row)
            preds.append(pd.DataFrame({"position": va, "fold": f, "model": model["name"], "prediction": p}))
        log(f"{model['name']}: {len(folds)} folds in {time.time() - start:.1f} seconds")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarise(per_fold, metrics):
    """Mean and standard deviation of every metric per model, best first by the primary metric."""
    labels = [metric_label(m) for m in all_metrics(metrics)]
    primary = labels[0]
    g = per_fold.groupby(["model", "kind", "type"], sort=False)
    out = g[labels].mean().round(4)
    std = g[primary].std().round(4).rename(f"{primary} std")
    out = out.join(std).reset_index()
    asc = HIGHER_IS_BETTER.get(all_metrics(metrics)[0]["key"]) is False
    return out.sort_values(primary, ascending=asc).reset_index(drop=True)


# ---------------------------------------------------------------- tuning and the run log

def _space(model_type, trial):
    """Settings Optuna tries for each model type."""
    f, i, c = trial.suggest_float, trial.suggest_int, trial.suggest_categorical
    if model_type == "logistic":
        return {"C": f("C", 1e-3, 10.0, log=True)}
    if model_type == "ridge":
        return {"alpha": f("alpha", 1e-3, 100.0, log=True)}
    if model_type == "ridge_nonneg":
        return {"alpha": f("alpha", 1e-3, 1000.0, log=True)}
    if model_type == "tree":
        return {"max_depth": i("max_depth", 2, 12), "min_samples_leaf": i("min_samples_leaf", 5, 200, log=True)}
    if model_type == "random_forest":
        return {"n_estimators": i("n_estimators", 200, 600, step=100), "max_depth": c("max_depth", [None, 6, 10, 16]),
                "min_samples_leaf": i("min_samples_leaf", 1, 50, log=True),
                "max_features": c("max_features", ["sqrt", 0.3, 0.5, 0.8])}
    if model_type in ("hist_gbm", "gbm"):
        return {"learning_rate": f("learning_rate", 0.01, 0.3, log=True), "max_iter": i("max_iter", 100, 600, step=50),
                "max_leaf_nodes": i("max_leaf_nodes", 15, 127, log=True),
                "min_samples_leaf": i("min_samples_leaf", 10, 200, log=True),
                "l2_regularization": f("l2_regularization", 1e-6, 10.0, log=True)}
    if model_type == "lightgbm":
        return {"learning_rate": f("learning_rate", 0.01, 0.3, log=True), "n_estimators": i("n_estimators", 100, 1000, step=50),
                "num_leaves": i("num_leaves", 15, 255, log=True), "min_child_samples": i("min_child_samples", 10, 200, log=True),
                "subsample": f("subsample", 0.5, 1.0), "subsample_freq": 1, "colsample_bytree": f("colsample_bytree", 0.5, 1.0),
                "reg_lambda": f("reg_lambda", 1e-6, 10.0, log=True)}
    if model_type == "xgboost":
        return {"learning_rate": f("learning_rate", 0.01, 0.3, log=True), "n_estimators": i("n_estimators", 100, 1000, step=50),
                "max_depth": i("max_depth", 3, 10), "min_child_weight": f("min_child_weight", 1.0, 20.0, log=True),
                "subsample": f("subsample", 0.5, 1.0), "colsample_bytree": f("colsample_bytree", 0.5, 1.0),
                "reg_lambda": f("reg_lambda", 1e-6, 10.0, log=True)}
    if model_type == "catboost":
        return {"learning_rate": f("learning_rate", 0.01, 0.3, log=True), "iterations": i("iterations", 200, 1000, step=100),
                "depth": i("depth", 4, 10), "l2_leaf_reg": f("l2_leaf_reg", 1.0, 10.0, log=True)}
    raise ValueError(f"No search space for {model_type!r}")


def data_version(spec):
    """A short hash of the features and split the runs were trained on. Runs with another version are stale."""
    import hashlib
    import json
    fe, sp = spec.get("features") or {}, spec.get("split") or {}
    key = json.dumps([fe.get("config"), fe.get("run_at"), sp.get("run_at"), (sp.get("cv") or {}).get("folds")],
                     sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def study_key(model, metrics, version):
    """Identifies one tuning run of one model: its type, fixed settings, tuning budget, metrics, data version, and
    its inputs when it uses a subset."""
    import hashlib
    import json
    parts = [model.get("type"), model.get("params"), model.get("class_weight"), model.get("tuning"), metrics, version]
    if model.get("inputs"):   # only when set, so the keys of earlier runs without a subset stay the same
        parts.append(sorted(model["inputs"]))
    key = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def tune(model, data, inputs, target, folds, metrics, task_type, config, seed=42, log=print):
    """Score the default settings, then let Optuna try more settings on the CV folds. model["tuning"] is
    {"n_trials": 20, "timeout_minutes": 10}; n_trials 1 (or no tuning) scores the defaults only. Settings in
    model["params"] are fixed and never tuned. Returns (trial records, out of fold predictions of the best trial)."""
    import json
    tuning = model.get("tuning") or {}
    n_trials = int(tuning.get("n_trials") or 1)
    timeout = float(tuning.get("timeout_minutes") or 0) * 60 or None
    mets = all_metrics(metrics)
    primary = metric_label(mets[0])
    maximise = HIGHER_IS_BETTER.get(mets[0]["key"]) is not False
    fixed = dict(model.get("params") or {})
    records, oofs = [], {}

    def evaluate(params, number):
        m = {**model, "params": {**params, **fixed}}
        start = time.time()
        per_fold, oof = cross_validate([m], data, inputs, target, folds, metrics, task_type, config, log=lambda *_: None)
        labels = [metric_label(x) for x in mets]
        rec = {"trial": number, "params": json.dumps({**params, **fixed}, sort_keys=True, default=str),
               "primary_label": primary, "primary_mean": float(per_fold[primary].mean()),
               "primary_std": float(per_fold[primary].std()) if len(per_fold) > 1 else 0.0,
               "metrics": json.dumps({k: float(per_fold[k].mean()) for k in labels}),
               "folds": json.dumps([float(x) for x in per_fold[primary]]), "seconds": round(time.time() - start, 1)}
        records.append(rec)
        oofs[number] = oof
        log(f"{model['name']} trial {number}: {primary} {rec['primary_mean']:.4f} ({rec['seconds']}s)")
        return rec["primary_mean"]

    evaluate({}, 0)
    if n_trials > 1:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize" if maximise else "minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda tr: evaluate(_space(model["type"], tr), tr.number + 1), n_trials=n_trials - 1,
                       timeout=timeout, catch=(ValueError,))
    best = max(records, key=lambda r: r["primary_mean"] if maximise else -r["primary_mean"])
    return records, oofs[best["trial"]]


class RunLog:
    """The run log: one row per trial, in a parquet file that is only appended to. A tuning run (a study) with the
    same key is not repeated: its trials are read back instead."""

    COLUMNS = ["run_id", "study", "model", "type", "class_weight", "trial", "params", "primary_label", "primary_mean",
               "primary_std", "metrics", "folds", "seconds", "data_version", "run_at"]

    def __init__(self, path):
        self.path = path

    def load(self):
        import os
        if os.path.exists(self.path):
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=self.COLUMNS)

    def has_study(self, key):
        df = self.load()
        return bool(len(df)) and (df["study"] == key).any()

    def append(self, model, key, version, records):
        import datetime as _dt
        now = _dt.datetime.now().replace(microsecond=0).isoformat()
        rows = pd.DataFrame([{"run_id": f"{key}-{r['trial']}", "study": key, "model": model["name"], "type": model["type"],
                              "class_weight": model.get("class_weight"), "data_version": version, "run_at": now, **r}
                             for r in records])
        df = pd.concat([self.load(), rows], ignore_index=True).drop_duplicates("run_id", keep="last")
        df[self.COLUMNS].to_parquet(self.path, index=False)
        return rows
