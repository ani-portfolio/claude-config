#!/usr/bin/env python3
"""Evaluation engine for ml-offline-evaluation: score every model on the test set once, with grouped bootstrap
intervals, curves, an operating point, slices, and drift. Used by notebooks/09_evaluation.ipynb.

Nothing here chooses or tunes anything on the test set. The champion was fixed in model training, the operating
threshold (when the output is a yes or no decision) is chosen on the training out of fold predictions, and the
test set only measures.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import model_core


# ---------------------------------------------------------------- data and models

def load_rows(spec, which):
    """The feature table rows of the train or test set, joined with the other columns of the split's rows."""
    sp, fe = spec["split"], spec["features"]
    row_id = sp.get("row_id_col") or "row_id"
    feat = pd.read_parquet(fe["files"][which])
    raw = pd.read_parquet(sp["files"][which])
    extra = [c for c in raw.columns if c not in feat.columns]
    return feat.merge(raw[[row_id] + extra], on=row_id, how="left").reset_index(drop=True)


def models_to_score(spec, runs):
    """Every model to score: the champion, the other ML models (best trial on the current data), and the baselines.
    Returns a list of dicts with name, kind (champion, ml, baseline), type, params, class_weight, and inputs
    when the model uses a subset of the inputs."""
    tr, bl = spec.get("training") or {}, spec.get("baseline") or {}
    champ = (tr.get("champion") or {}).get("name")
    version = tr.get("data_version")
    higher = model_core.HIGHER_IS_BETTER.get(model_core.all_metrics(bl["metrics"])[0]["key"]) is not False
    out = []
    names = {m["name"] for m in tr.get("models") or []}
    cur = runs[(runs["data_version"] == version) & runs["model"].isin(names)]
    for name, g in cur.groupby("model", sort=False):
        best = g.loc[g["primary_mean"].idxmax() if higher else g["primary_mean"].idxmin()]
        model = next(m for m in tr["models"] if m["name"] == name)
        out.append({"name": name, "kind": "champion" if name == champ else "ml", "type": model["type"],
                    "params": json.loads(best["params"]), "class_weight": model.get("class_weight"),
                    "cv": float(best["primary_mean"]),
                    # A model trained on a subset of the inputs must be refitted on that same subset.
                    **({"inputs": list(model["inputs"])} if model.get("inputs") else {})})
    for b in bl.get("models") or []:
        cv = next((r[bl["primary_label"]] for r in bl.get("results") or [] if r.get("model") == b["name"]), None)
        out.append({**b, "kind": "baseline", "cv": cv})
    order = {"champion": 0, "ml": 1, "baseline": 2}
    return sorted(out, key=lambda m: order[m["kind"]])


# Model types whose fits involve randomness beyond a fixed seed; their recomputed CV may differ slightly.
NOISY_TYPES = {"random_forest", "lightgbm", "xgboost", "catboost"}


def self_check(models, train, folds, inputs, target, task_type, config, metrics, champion_path=None, log=print):
    """Check the scoring paths before the test set is opened, so a bug in how evaluation refits a model cannot
    reach the test results unnoticed. Every non champion model is scored again on the CV folds through the exact
    definition evaluation will use, and must reproduce the CV score recorded for it. The saved champion must give
    the same predictions on the training rows as a fresh fit of its definition. Returns a DataFrame with one row
    per model and an ok column; nothing here uses the test rows."""
    import joblib
    family = model_core.task_family(task_type)
    rows = []
    primary = model_core.metric_label(model_core.all_metrics(metrics)[0])
    for m in models:
        if m["kind"] == "champion":
            if not champion_path or not os.path.exists(champion_path):
                rows.append({"model": m["name"], "kind": m["kind"], "check": "saved model against a fresh fit",
                             "recorded": None, "recomputed": None, "rel_diff": None, "ok": None})
                continue
            saved = joblib.load(champion_path)
            fresh = model_core.make_estimator(m, family, config, train[target]).fit(train[inputs], train[target])
            a = saved.predict_proba(train[inputs])[:, 1] if family == "binary" else saved.predict(train[inputs])
            b = fresh.predict_proba(train[inputs])[:, 1] if family == "binary" else fresh.predict(train[inputs])
            diff = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) / (np.mean(np.abs(b)) or 1))
            tol = 1e-3 if m["type"] in NOISY_TYPES else 1e-6
            rows.append({"model": m["name"], "kind": m["kind"], "check": "saved model against a fresh fit",
                         "recorded": None, "recomputed": None, "rel_diff": diff, "ok": diff <= tol})
            continue
        if m.get("cv") is None:
            continue
        per_fold, _ = model_core.cross_validate([m], train, inputs, target, folds, metrics, task_type, config,
                                                log=lambda *_: None)
        got, want = float(per_fold[primary].mean()), float(m["cv"])
        diff = abs(got - want) / (abs(want) or 1)
        tol = 1e-3 if m["type"] in NOISY_TYPES else 1e-6
        # Baseline scores are recorded rounded to 4 decimals in the baseline section; the run log keeps full precision.
        ok = diff <= tol or (m["kind"] == "baseline" and abs(round(got, 4) - want) < 1e-9)
        rows.append({"model": m["name"], "kind": m["kind"], "check": f"CV {primary} reproduced", "recorded": want,
                     "recomputed": got, "rel_diff": diff, "ok": ok})
    out = pd.DataFrame(rows)
    bad = out[out["ok"] == False]  # noqa: E712
    log("Self-check: " + ("every scoring path reproduces its recorded result." if bad.empty else
                          "these do not reproduce their recorded result: " + ", ".join(bad["model"]) + "."))
    return out


def predict_all(models, train, test, inputs, target, family, config, champion_path=None, log=print):
    """Predictions on the test rows for every model. The champion is the saved pipeline; the other models are fitted
    on every training row with their chosen settings. Returns {name: array}."""
    import joblib
    preds = {}
    for m in models:
        if m["kind"] == "champion" and champion_path and os.path.exists(champion_path):
            pipe = joblib.load(champion_path)
            X = test[inputs]
            preds[m["name"]] = pipe.predict_proba(X)[:, 1] if family == "binary" else pipe.predict(X).astype(float)
        else:
            preds[m["name"]] = model_core.fit_predict(m, train, test, inputs, target, family, config)
        log(f"{m['name']} ({m['kind']}): scored")
    return preds


# ---------------------------------------------------------------- bootstrap

def _clusters(test, cluster_col):
    if cluster_col and cluster_col in test:
        codes, _ = pd.factorize(test[cluster_col].astype(str))
        return codes
    return np.arange(len(test))


def bootstrap(fn, test, cluster_col=None, n=200, seed=0):
    """Percentile interval (2.5%, 97.5%) of fn(rows, draw) over n resamples of whole clusters (dates, entities, or
    rows). fn takes an index array into test, and a draw array saying which resampled cluster each row came from (so
    a date drawn twice counts as two dates, not one date twice the size), and returns a number."""
    codes = _clusters(test, cluster_col)
    groups = pd.Series(np.arange(len(test))).groupby(codes).apply(np.asarray).tolist()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        draw = np.concatenate([np.full(len(groups[i]), j) for j, i in enumerate(pick)])
        v = fn(idx, draw)
        if v == v:   # not NaN
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metric_fn(m, test, target, pred, model_type):
    """fn(idx) for one metric of one model, for bootstrap."""
    y = test[target].values
    groups = test[m["k_per"]].values if m.get("k_per") else None
    prob = model_type in model_core.PROBABILISTIC

    def fn(idx, draw=None):
        g = None
        if groups is not None:
            g = groups[idx] if draw is None else (pd.Series(groups[idx]).astype(str) + "#" + pd.Series(draw).astype(str)).values
        return model_core.score_metric(m, y[idx], pred[idx], g, probabilistic=prob)
    return fn


# ---------------------------------------------------------------- operating point and curves

def operating_mask(spec, test, pred, oof=None, oof_target=None):
    """Which test rows the model acts on: the top k (per group) when the primary metric is "at k", otherwise rows
    above a threshold chosen on the training out of fold predictions (best F1). Returns (mask, description)."""
    m = model_core.all_metrics(spec["baseline"]["metrics"])[0]
    if m["key"].endswith("_at_k"):
        groups = test[m["k_per"]].values if m.get("k_per") else None
        mask = model_core._top_k_mask(pred, m.get("k"), m.get("k_share"), groups, np.random.default_rng(0))
        return mask, f"top {m.get('k') or format(m.get('k_share'), '.0%')}" + (f" per {m['k_per']}" if m.get("k_per") else "")
    if oof is None:
        thr = 0.5
        how = "0.5 (no out of fold predictions to choose from)"
    else:
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(oof_target, oof)
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        thr = float(t[np.nanargmax(f1[:-1])]) if len(t) else 0.5
        how = f"{thr:.4f}, the best F1 on the training out of fold predictions"
    return pred >= thr, f"score at or above {how}"


def gain_curve(y, pred, points=100):
    """Share of all positives found against the share of rows acted on, best scores first. Rows with the same score
    are taken together, so tied rows count by their average (a constant score gives the diagonal)."""
    y = np.asarray(y, dtype=float)
    pred = np.nan_to_num(np.asarray(pred, dtype=float), nan=-np.inf)
    d = pd.DataFrame({"p": pred, "y": y}).groupby("p", sort=True)["y"].agg(["size", "sum"]).iloc[::-1]
    rows = np.concatenate([[0], d["size"].cumsum().values]) / len(y)
    found = np.concatenate([[0], d["sum"].cumsum().values]) / max(y.sum(), 1)
    share = np.linspace(0, 1, points + 1)
    return share, np.interp(share, rows, found)


# ---------------------------------------------------------------- slices

def propose_slices(spec, train, test):
    """Slice columns from the feature settings, importance, the date, entities, missing values, and protected
    attributes named in the business constraints."""
    cfg = (spec.get("features") or {}).get("config") or {}
    sp, tr = spec["split"], spec.get("training") or {}
    slices = {"categorical": [c for c in cfg.get("categorical") or [] if c in test][:4], "numeric_bins": {},
              "time": None, "new_entity": None, "missing": [], "protected": [], "min_rows": 100, "min_positives": 30}
    top = [f["feature"] for f in tr.get("top_features") or []]
    numeric = [c for c in top if c in (cfg.get("numeric") or []) and c in test and pd.api.types.is_numeric_dtype(test[c])]
    for c in numeric[:3]:
        slices["numeric_bins"][c] = 4
    date = sp.get("date_col")
    if date and date in test:
        d = pd.to_datetime(test[date], errors="coerce", format="mixed")
        slices["time"] = {"column": date, "period": "W" if (d.max() - d.min()).days <= 120 else "M"}
    ent = sp.get("group_col") or next(iter((spec.get("framing") or {}).get("entity_id_cols") or []), None)
    if ent and ent in test and ent in train:
        slices["new_entity"] = ent
    for c in top[:10]:
        if c in test and test[c].isna().mean() > 0.05:
            slices["missing"].append(c)
    fair = str(((spec.get("business") or {}).get("constraints") or {}).get("fairness") or "").lower()
    slices["protected"] = [c for c in test.columns if len(c) > 2 and c.lower() in fair][:3]
    return slices


def slice_columns(test, train, slices):
    """{slice dimension: Series of slice labels} for the test rows."""
    out = {}
    for c in list(slices.get("categorical") or []) + list(slices.get("protected") or []):
        if c in test:
            out[c] = test[c].astype("object").where(test[c].notna(), "missing").astype(str)
    for c, bins in (slices.get("numeric_bins") or {}).items():
        if c in test:
            edges = np.unique(np.nanquantile(pd.to_numeric(train[c], errors="coerce"), np.linspace(0, 1, int(bins) + 1)))
            if len(edges) > 2:
                edges[0], edges[-1] = -np.inf, np.inf
                out[f"{c} (quantiles of training)"] = pd.cut(pd.to_numeric(test[c], errors="coerce"), edges).astype(str)
    t = slices.get("time")
    if t and t.get("column") in test:
        d = pd.to_datetime(test[t["column"]], errors="coerce", format="mixed")
        out[f"{t['column']} by {'week' if t.get('period') == 'W' else 'month'}"] = d.dt.to_period(t.get("period", "M")).astype(str)
    e = slices.get("new_entity")
    if e and e in test and e in train:
        out[f"{e} seen in training"] = np.where(test[e].isin(set(train[e])), "known", "new")
    for c in slices.get("missing") or []:
        if c in test:
            out[f"{c} missing"] = np.where(test[c].isna(), "missing", "present")
    return out


def slice_table(spec, test, train, target, preds, champion, reference, slices, act, n_boot=100, seed=0):
    """One row per slice: rows, positives or target mean, the champion's and the reference's primary metric, the
    difference with an interval, and flags. For "at k" metrics, the top k is chosen on the whole test set (as in
    production) and the slice shows recall or precision among its own rows."""
    m = model_core.all_metrics(spec["baseline"]["metrics"])[0]
    family = model_core.task_family((spec.get("framing") or {}).get("task_type"))
    y = test[target].values
    at_k = m["key"].endswith("_at_k")
    rows = []

    def score(name, idx):
        if at_k:
            sel, yy = act[name][idx], y[idx]
            if m["key"] == "recall_at_k":
                return yy[sel].sum() / yy.sum() if yy.sum() else np.nan
            prec = yy[sel].mean() if sel.any() else np.nan
            return prec if m["key"] == "precision_at_k" else prec / y.mean()
        return model_core.score_metric(m["key"], y[idx], preds[name][idx], probabilistic=True)

    overall = score(champion, np.arange(len(test)))
    for dim, labels in slice_columns(test, train, slices).items():
        labels = pd.Series(np.asarray(labels))
        for value, idx in labels.groupby(labels.values).groups.items():
            idx = np.asarray(idx)
            npos = int(y[idx].sum()) if family == "binary" else None
            small = len(idx) < slices.get("min_rows", 100) or (family == "binary" and npos < slices.get("min_positives", 30))
            c, r = score(champion, idx), score(reference, idx)
            sub = test.iloc[idx]
            lo, hi = (np.nan, np.nan) if small else bootstrap(
                lambda j, draw=None: score(champion, idx[j]) - score(reference, idx[j]), sub, None, n_boot, seed)
            higher = model_core.HIGHER_IS_BETTER.get(m["key"]) is not False
            diff = c - r if higher else r - c
            rows.append({"slice": dim, "value": value, "rows": len(idx),
                         ("positive share" if family == "binary" else "target mean"): round(float(y[idx].mean()), 4),
                         "champion": round(float(c), 4) if c == c else None, "reference": round(float(r), 4) if r == r else None,
                         "champion better by": round(float(diff), 4) if diff == diff else None,
                         "interval": None if lo != lo else f"{(lo if higher else -hi):.3f} to {(hi if higher else -lo):.3f}",
                         **({"champion acts on": round(float(act[champion][idx].mean()), 3),
                             "reference acts on": round(float(act[reference][idx].mean()), 3)} if at_k else {}),
                         "too small": small,
                         # Worse only when the whole interval of the difference is below zero
                         "worse than reference": bool(not small and lo == lo and (hi < 0 if higher else lo > 0)),
                         # For "at k" metrics a slice's share is driven by where the capacity goes, so it is not
                         # compared with the overall score
                         "far below overall": bool(not at_k and not small and c == c and overall and
                                                   ((c < 0.8 * overall) if higher else (c > 1.25 * overall)))})
    return pd.DataFrame(rows), overall


def psi(a, b, bins=10):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    edges = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(a, edges)[0] / len(a)
    q = np.histogram(b, edges)[0] / len(b)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(((p - q) * np.log(p / q)).sum())


# ---------------------------------------------------------------- model card

def model_card(spec, ev):
    """docs/model_card.md from the spec, after evaluation. Evidence only: the decision is the user's."""
    pr, bus, fr, sp, fe, tr = (spec.get(k) or {} for k in ("project", "business", "framing", "split", "features", "training"))
    tgt, champ = fr.get("target") or {}, tr.get("champion") or {}
    label = ev["primary_label"]
    L = [f"# Model Card: {pr.get('name')}", "",
         f"**Evaluated:** {ev['run_at'][:10]}  ",
         "**Source:** `notebooks/09_evaluation.ipynb`, written from `ml_spec.yaml`. Rerun the notebook to update it.", "",
         "## Purpose", "", f"{bus.get('business_goal') or ''} {bus.get('decision_supported') or ''}".strip(), "",
         "## Model", "",
         f"* **Champion:** {champ.get('name')} ({champ.get('type')}), chosen by its mean CV score in model training.",
         f"* **Predicts:** {tgt.get('definition') or tgt.get('name')}",
         f"* **Unit and timing:** {fr.get('unit_of_prediction')}; {(fr.get('prediction_time') or {}).get('description')}",
         f"* **Inputs:** {fe.get('n_inputs')} features. Excluded: "
         + (", ".join(e["column"] for e in fe.get("excluded") or []) or "none") + ".",
         f"* **Saved model:** `{champ.get('path')}`", "",
         "## Data", "",
         f"* **Split:** {sp.get('strategy')}; test set of {(sp.get('test') or {}).get('rows', 0):,} rows"
         + (f", {(sp.get('test') or {}).get('start')} to {(sp.get('test') or {}).get('end')}" if (sp.get('test') or {}).get("start") else "")
         + (f", with a {sp.get('gap_days')} day gap" if sp.get("gap_days") else "") + ".", "",
         "## Results on the Test Set", "",
         *([f"**The test set has been opened {len(ev.get('test_opened') or [])} times.** Scores after the first "
             "look are optimistic, because earlier test results shaped the changes since.", ""]
           if len(ev.get("test_opened") or []) > 1 else []),
         f"Primary metric: {label}. Intervals are 95% bootstrap intervals"
         + (f", resampling whole {ev['cluster']} values." if ev.get("cluster") else "."), "",
         "| Model | Kind | Test | 95% interval | CV |", "|---|---|---|---|---|"]
    for r in ev["metrics"]:
        L.append(f"| {r['model']}{' (champion)' if r['kind'] == 'champion' else ''} | {r['kind']} | {r['test']:.4f} | "
                 f"{r['low']:.4f} to {r['high']:.4f} | " + (f"{r['cv']:.4f}" if r.get("cv") is not None else "") + " |")
    if ev.get("comparisons"):
        L += ["", "## Comparisons", "", "Paired bootstrap differences; positive is better, also for error metrics.", "",
              "| Comparison | Difference | 95% interval | Relative |", "|---|---|---|---|"]
        L += [f"| {c['comparison']} | {c['difference']:+.4f} | {c['low']:+.4f} to {c['high']:+.4f} | "
              + (f"{c['relative %']:+.1f}%" if c.get("relative %") is not None else "") + " |" for c in ev["comparisons"]]
    if ev.get("operating_point"):
        L += ["", f"Operating point: {ev['operating_point']}."]
    L += ["", "## Evidence", "", "| Question | Answer |", "|---|---|"]
    L += [f"| {e['question']} | {e['answer']} |" for e in ev["evidence"]]
    weak = [s for s in ev["slices"] if s.get("worse than reference") or s.get("far below overall")]
    L += ["", "## Slices", ""]
    L += ([f"* {s['slice']} = {s['value']}: champion {s['champion']}, reference {s['reference']}"
           + (" (worse than the reference)" if s.get("worse than reference") else " (far below the overall score)")
           + (f"; the champion acts on {s['champion acts on']:.0%} of its rows, the reference on {s['reference acts on']:.0%}"
              if s.get("champion acts on") is not None else "") for s in weak]
          or ["No slice large enough to judge is worse than the reference or far below the overall score."])
    L += ["", "## Limitations and Assumptions", ""] + ([f"* {a}" for a in spec.get("assumptions") or []] or ["None recorded."])
    oq = [q for q in spec.get("open_questions") or [] if isinstance(q, dict) and q.get("status") == "open"]
    L += ["", "## Open Questions", ""] + ([f"* {q.get('id')}: {q.get('question')}" for q in oq] or ["None."])
    made = [d for d in spec.get("decisions") or [] if isinstance(d, dict) and d.get("stage") == "evaluation"
            and not d.get("superseded_by")]
    L += ["", "## Decision", ""]
    L += ([f"* {d.get('date')}: {d.get('decision')} {d.get('reason') or ''}".rstrip() for d in made] or
          ["Not recorded yet. The evidence above is for the model owner to decide go, iterate, or no go; record the "
           "decision with its reason in the decisions cell of the evaluation notebook."])
    return "\n".join(L) + "\n"


CARD_MARK = "<!-- written by ml-offline-evaluation. If you edit this card, the next run keeps it and writes model_card_new.md: "


def write_model_card(path, text):
    """Write the card, unless the existing card was edited since the skill wrote it. Then write model_card_new.md
    next to it and leave the edited card alone. Returns the path written."""
    import hashlib
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    if os.path.exists(path):
        old = open(path, encoding="utf-8").read()
        body, _, tail = old.partition(CARD_MARK)
        stored = tail.split(" ")[0] if tail else None
        intact = (stored is not None and old == body + CARD_MARK + stored + " -->\n"
                  and hashlib.sha256(body.encode()).hexdigest()[:16] == stored)
        if not intact:
            path = path.replace(".md", "_new.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + CARD_MARK + digest + " -->\n")
    return path
