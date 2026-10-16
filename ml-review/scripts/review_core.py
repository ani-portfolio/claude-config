"""Scripted checks for the ml-review skill.

Every check reads the spec, the saved parquet files, the notebooks, and the project's src code. Nothing is changed.
Each finding is a dict with a stable id, so a finding can be tracked as open, fixed, or accepted across reviews:

    id, stage, category, severity (blocker, warning, note), title, evidence, fix, source ("script")

Severity:
    blocker  the results cannot be trusted until this is fixed (leakage, wrong data, missing files).
    warning  this could change a decision, or makes a result weaker than it looks.
    note     worth knowing; no action needed unless it matters for this project.

Command line (run with the project's Python from the project root):
    review_core.py stages            the stages that are done, and what changed since the last review
    review_core.py check [stages]    print the scripted findings as YAML
    review_core.py brief [stages]    print the brief for the independent reviewer
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ml_common  # noqa: E402
import nb_common  # noqa: E402

REVIEW_STAGES = ["framing", "eda", "split", "features", "baseline", "training", "evaluation"]
STAGE_LABEL = {"project": "the whole project", "framing": "ML framing", "eda": "EDA", "split": "split",
               "features": "features", "baseline": "baseline", "training": "training", "evaluation": "evaluation"}
SEVERITY_ORDER = {"blocker": 0, "warning": 1, "note": 2}
CATEGORIES = ("leakage", "consistency", "results too good", "test set use", "statistical strength",
              "business fit", "reproducibility", "pipeline state")
FINDING_KEYS = ("id", "stage", "category", "severity", "title", "evidence", "fix", "source")


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:60]


def _ts(text):
    try:
        return _dt.datetime.fromisoformat(str(text)[:19])
    except (TypeError, ValueError):
        return None


def _read(root, rel, columns=None):
    p = os.path.join(root, rel) if rel else None
    if not p or not os.path.exists(p):
        return None
    try:
        return pd.read_parquet(p, columns=columns)
    except Exception:  # noqa: BLE001  a file that cannot be read is reported by the caller
        return None


def done_stages(spec):
    """The reviewable stages that have run."""
    return [s for s in REVIEW_STAGES if ml_common.step_done(spec, s)]


class Findings:
    def __init__(self, stages):
        self.stages = set(stages) | {"project"}
        self.items = []

    def add(self, stage, check, subject, category, severity, title, evidence, fix):
        if stage not in self.stages:
            return
        fid = f"{stage}.{check}" + (f".{_slug(subject)}" if subject else "")
        if any(f["id"] == fid for f in self.items):
            return
        self.items.append({"id": fid, "stage": stage, "category": category, "severity": severity, "title": title,
                           "evidence": evidence, "fix": fix, "source": "script"})

    def sorted(self):
        order = {s: i for i, s in enumerate(["project"] + REVIEW_STAGES)}
        return sorted(self.items, key=lambda f: (SEVERITY_ORDER[f["severity"]], order.get(f["stage"], 99), f["id"]))


# ---------------------------------------------------------------- checks by area

def _stage_of_field(field):
    head = str(field).split(".")[0]
    return {"business": "framing", "data": "framing", "open_questions": "project"}.get(head, head) \
        if head not in REVIEW_STAGES else head


CONSISTENCY_SEVERITY = {"target": "blocker", "date column": "blocker", "excluded column used": "blocker",
                        "gap": "blocker", "horizon": "blocker", "primary metric": "warning", "metric k": "warning",
                        "V1 bar": "warning", "reference": "warning", "entity": "warning", "ML needed": "warning",
                        "open question": "note"}


def check_consistency(spec, F):
    for c in ml_common.consistency_checks(spec):
        stages = [_stage_of_field(w) for w in c["where"]]
        later = [s for s in REVIEW_STAGES if s in stages]
        stage = later[-1] if later else "project"
        sev = CONSISTENCY_SEVERITY.get(c["check"], "warning")
        cat = "leakage" if c["check"] in ("gap", "horizon", "excluded column used") else \
            "business fit" if c["check"] in ("V1 bar", "metric k", "ML needed") else "consistency"
        if c["check"] == "open question":
            qid = c["where"][0].split(".")[-1]
            F.add(stage, "open_question", qid, "business fit", sev, f"Question {qid} is still open after its stage finished",
                  c["detail"], "Answer it in the notebook of its stage, or with /ml-reconcile-notes.")
            continue
        F.add(stage, _slug(c["check"]), "", cat, sev, f"Stages disagree: {c['check']}", c["detail"],
              "Align the notes with /ml-reconcile-notes, or rerun the stage that owns the setting.")


def check_pipeline(spec, root, F):
    runs = spec.get("notebooks") or {}
    chain = ["framing", "split", "features", "baseline", "training", "evaluation"]
    for prev, stage in zip(chain, chain[1:]):
        before, after = _ts((runs.get(prev) or {}).get("run_at")), _ts((runs.get(stage) or {}).get("run_at"))
        if before and after and before > after and spec.get(stage):
            F.add(stage, "stale", "", "pipeline state", "warning", f"The {stage} results are older than the {prev} step",
                  f"The {prev} notebook ran at {before:%Y-%m-%d %H:%M}, after the {stage} notebook ({after:%Y-%m-%d %H:%M}). "
                  f"The {stage} results may use old settings.", f"Rerun /{ml_common.STAGE_SKILL[stage]}.")
    for stage, rec in runs.items():
        if stage in REVIEW_STAGES and nb_common.edited_since_run(spec, root, stage):
            F.add(stage, "edited", "", "pipeline state", "warning", "A notebook was edited after it last ran",
                  f"The decision cells of {rec.get('path')} differ from the run recorded in the spec, so the spec may "
                  "not match the notebook.", f"Open {rec.get('path')} and choose Run All.")
    for stage in REVIEW_STAGES:
        for name, rel in ((spec.get(stage) or {}).get("files") or {}).items():
            if rel and not os.path.exists(os.path.join(root, rel)):
                F.add(stage, "missing_file", name, "pipeline state", "blocker", f"A saved file is missing: {rel}",
                      f"The spec lists {rel} as the {stage} stage's {name} file, but it does not exist.",
                      f"Rerun /{ml_common.STAGE_SKILL[stage]}.")
    tr = spec.get("training") or {}
    if tr.get("data_version") and spec.get("features"):
        try:
            import model_core
            now = model_core.data_version(spec)
        except Exception:  # noqa: BLE001
            now = None
        if now and now != tr["data_version"]:
            F.add("training", "data_version", "", "pipeline state", "warning",
                  "The champion was trained on older features or an older split",
                  f"The training section records data version {tr['data_version']}, but the current features and split "
                  f"give {now}.", "Rerun /ml-model-training (models already logged on the new data are read back).")
    ev = spec.get("evaluation") or {}
    champ = (tr.get("champion") or {}).get("name")
    if ev.get("champion") and champ and ev["champion"] != champ:
        F.add("evaluation", "champion_changed", "", "test set use", "warning",
              "The evaluation scored a different champion",
              f"The evaluation scored {ev['champion']}, but the current champion is {champ}.",
              "Rerun /ml-offline-evaluation, and note that the test set was already opened.")


def check_split(spec, root, F):
    sp = spec.get("split") or {}
    if not sp.get("files"):
        return
    fr = spec.get("framing") or {}
    target, date = sp.get("target"), sp.get("date_col")
    train, test = _read(root, sp["files"].get("train")), _read(root, sp["files"].get("test"))
    if train is None or test is None:
        return
    if target in test and target in train and sp.get("task_type", "").endswith("classification"):
        pos = int((pd.to_numeric(test[target], errors="coerce") == 1).sum()) if test[target].nunique() <= 2 else None
        if pos is not None and pos < 100:
            F.add("split", "few_positives", "", "statistical strength", "warning",
                  "The test set has few positive rows",
                  f"The test set has {pos} positive rows, so test metrics will have wide intervals.",
                  "Read the test results with their intervals, or consider a longer test period with /ml-train-test-split.")
    if sp.get("strategy") == "time" and date in train and date in test:
        tmax = pd.to_datetime(train[date], errors="coerce", format="mixed").max()
        smin = pd.to_datetime(test[date], errors="coerce", format="mixed").min()
        if pd.notna(tmax) and pd.notna(smin):
            gap = (smin - tmax).days
            if gap <= 0:
                F.add("split", "time_overlap", "", "leakage", "blocker", "Training and test periods overlap",
                      f"The last training date is {tmax.date()} and the first test date is {smin.date()}.",
                      "Rerun /ml-train-test-split.")
            need = max(ml_common.days_in((fr.get("target") or {}).get("outcome_window")) or 0,
                       ml_common.days_in((fr.get("target") or {}).get("label_delay")) or 0)
            if fr.get("task_type") != "forecasting" and need and 0 < gap < need:
                F.add("split", "short_gap", "", "leakage", "blocker", "The gap between training and test is too short",
                      f"The test set starts {gap} days after the last training date, but the outcome window or label "
                      f"delay is {need:g} days, so late training labels overlap the test period.",
                      "Rerun /ml-train-test-split with a gap at least as long as the outcome window.")
    ents = [c for c in fr.get("entity_id_cols") or [] if c in train and c in test]
    if sp.get("strategy") in ("random", "group") and ents:
        e = sp.get("group_col") if sp.get("group_col") in ents else ents[0]
        shared = set(train[e].dropna().unique()) & set(test[e].dropna().unique())
        share = len(shared) / max(test[e].nunique(), 1)
        if shared and (sp.get("strategy") == "group" or share > 0.05):
            F.add("split", "entity_overlap", e, "leakage", "blocker" if sp.get("strategy") == "group" else "warning",
                  f"The same {e} values are in training and test",
                  f"{len(shared):,} of {test[e].nunique():,} test {e} values ({share:.0%}) also appear in training. A "
                  "model can learn the entity rather than the pattern, so the test score may be optimistic.",
                  f"Use a group split on {e} with /ml-train-test-split, or accept it if new entities are never scored.")
    cfg = (spec.get("features") or {}).get("config") or {}
    cols = [c for c in (cfg.get("numeric") or []) + (cfg.get("categorical") or []) + (cfg.get("high_cardinality") or [])
            if c in train and c in test]
    if cols and sp.get("strategy") != "time":
        key = lambda d: pd.util.hash_pandas_object(d[cols + ([target] if target in d else [])].astype(str), index=False)
        dup = key(test).isin(set(key(train))).mean()
        if dup > 0.01:
            F.add("split", "duplicates", "", "leakage", "warning", "Test rows have exact copies in training",
                  f"{dup:.1%} of test rows have the same inputs and target as a training row, which flatters the test score.",
                  "Check for duplicated records, and drop or group them in /ml-train-test-split.")


SUSPICIOUS = [
    (r"\.shift\(\s*-", "a negative shift reads values from later rows (the future)"),
    (r"\.(fit|fit_transform)\(", "fitting here learns from every row, including test rows"),
    (r"\.transform\(\s*['\"](mean|median|sum|std|max|min|count)['\"]\s*\)",
     "a group statistic over all rows includes later rows and test rows"),
    (r"\.rolling\(", "a rolling window needs rows sorted by date within each entity"),
    (r"\.(expanding|cumsum|cummax|cummin)\(", "a running total needs rows sorted by date and must exclude the current outcome"),
    (r"\.(mean|median)\(\)", "a statistic over the whole column learns from test rows"),
]


def scan_code(path, target, excluded):
    """Lines in user code that need a second look, as (line number, line, reasons, uses the target)."""
    out = []
    if not os.path.exists(path):
        return out
    names = [n for n in [target] + list(excluded) if n]
    for i, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        s = line.split("#", 1)[0]
        if not s.strip():
            continue
        whys = [f"uses {n}, which is the target or an excluded column" for n in names
                if re.search(r"['\"]" + re.escape(n) + r"['\"]", s)]
        uses_target = bool(whys)
        whys += [why for pat, why in SUSPICIOUS if re.search(pat, s)]
        if whys:
            out.append((i, line.strip(), "; ".join(whys), uses_target))
    return out


def check_features(spec, root, F):
    fe, sp = spec.get("features") or {}, spec.get("split") or {}
    if not fe:
        return
    cfg = fe.get("config") or {}
    target = sp.get("target") or cfg.get("target")
    excluded = [e.get("column") for e in (spec.get("data") or {}).get("excluded_cols") or [] if isinstance(e, dict)]
    used = set((cfg.get("numeric") or []) + (cfg.get("categorical") or []) + (cfg.get("high_cardinality") or []))
    # Lags and rolling windows of the target are shifted on purpose; only a direct use is a leak
    used |= {i.get("source") for i in fe.get("inputs") or [] if i.get("step") in (None, "column", "log", "ratio")}
    if target and target in used:
        F.add("features", "target_input", "", "leakage", "blocker", "The target is used as a feature",
              f"{target} is in the feature lists.", "Remove it in /ml-feature-engineering.")
    for c in sorted(set(excluded) & used):
        F.add("features", "excluded_input", c, "leakage", "blocker", f"An excluded column is used: {c}",
              f"{c} was excluded in the framing but is a feature input.", "Remove it in /ml-feature-engineering.")
    chk = fe.get("checks") or {}
    safe = set(chk.get("confirmed_safe") or [])
    for s in chk.get("strongest") or []:
        name, score = s.get("feature"), s.get("score")
        if score is None:
            continue
        strong = score >= max(chk.get("leakage_threshold") or 0.9, 0.9) if s.get("measure") == "AUC" else abs(score) >= 0.9
        if s.get("flagged") and name not in safe:
            F.add("features", "flagged", name, "leakage", "blocker", f"A feature alone nearly predicts the target: {name}",
                  f"{name} has a single feature {s.get('measure')} of {score:.3f} on the CV folds and is not confirmed safe.",
                  "Confirm when the column is recorded; exclude it in /ml-feature-engineering or mark it CONFIRMED_SAFE "
                  "with the reason.")
        elif strong and name in safe:
            F.add("features", "strong_confirmed", name, "leakage", "note", f"A very strong feature was confirmed safe: {name}",
                  f"{name} has a single feature {s.get('measure')} of {score:.3f}. It was confirmed safe; the reason "
                  "should be written down.", "Check that a decision records why it is known before the prediction time.")
    code = fe.get("code") or {}
    fpath = os.path.join(root, code["features"]) if code.get("features") else None
    if fpath and os.path.exists(fpath) and code.get("features_hash"):
        cur = hashlib.sha256(open(fpath, "rb").read()).hexdigest()[:16]
        if cur != code["features_hash"]:
            F.add("features", "code_changed", "features.py", "pipeline state", "warning",
                  "src/features.py changed after the features ran",
                  "The file's hash differs from the one recorded when the features notebook last ran.",
                  "Rerun /ml-feature-engineering and the later stages.")
    cpath = os.path.join(root, code["custom"]) if code.get("custom") else None
    if cpath and os.path.exists(cpath):
        if cfg.get("custom") and _ts(fe.get("run_at")) and \
                _dt.datetime.fromtimestamp(os.path.getmtime(cpath)) > _ts(fe.get("run_at")):
            F.add("features", "custom_changed", "", "pipeline state", "warning",
                  "src/custom_features.py changed after the features ran",
                  "The file was saved after the features notebook last ran, so the saved features may be out of date.",
                  "Rerun /ml-feature-engineering and the later stages.")
        if cfg.get("custom"):
            for line_no, line, why, uses_target in scan_code(cpath, target, excluded):
                F.add("features", "custom_code", hashlib.sha256(line.encode()).hexdigest()[:8], "leakage", "warning" if uses_target else "note",
                      f"Hand written feature code needs a check (line {line_no})",
                      f"`{line[:100]}`: {why}. The custom features run on training, gap, and test rows together.",
                      "Confirm the line uses only past rows of the same entity; otherwise move the learned step into "
                      "the preprocessor.")
    for d in chk.get("drift_large") or []:
        name = d.get("feature") if isinstance(d, dict) else d
        F.add("features", "drift", name, "statistical strength", "note", f"A feature drifts between training and test: {name}",
              "The features notebook found a large drift (PSI) for this input.",
              "Check whether the meaning of the column changed over time.")


def _family(spec):
    t = (spec.get("framing") or {}).get("task_type") or (spec.get("split") or {}).get("task_type") or ""
    return "binary" if "classification" in t else "regression"


def check_training(spec, root, F):
    tr, bl = spec.get("training") or {}, spec.get("baseline") or {}
    champ = tr.get("champion") or {}
    if not champ:
        return
    label = tr.get("primary_label") or bl.get("primary_label")
    key = ((bl.get("metrics") or {}).get("primary") or {}).get("key")
    try:
        import model_core
        higher = model_core.HIGHER_IS_BETTER.get(key) is not False
    except Exception:  # noqa: BLE001
        higher = True
    score, std = champ.get(label), champ.get("std")
    best = (bl.get("best_non_ml") or {}).get(label)
    if champ.get("beats_best_baseline") is False:
        F.add("training", "below_baseline", "", "business fit", "warning", "The champion does not beat the best baseline",
              f"The champion {champ.get('name')} scores {score} against {best} for {(bl.get('best_non_ml') or {}).get('name')}.",
              "Improve the features or models, or consider the simple rule.")
    if score is not None and best is not None and std:
        gain = (score - best) if higher else (best - score)
        if 0 < gain < std:
            F.add("training", "gain_in_noise", "", "statistical strength", "warning",
                  "The gain over the best baseline is smaller than the CV spread",
                  f"The champion beats the best baseline by {gain:.4f}, but its score varies by {std:.4f} between folds.",
                  "Read the test interval in /ml-offline-evaluation before relying on the gain.")
    pct = champ.get("vs_reference_pct")
    if pct is not None and pct > 200:
        F.add("training", "huge_gain", "", "results too good", "warning", "The champion beats the reference by an unusual margin",
              f"The champion is {pct:.0f}% better than the reference. Such a gap often means the reference is weak or a "
              "feature leaks.", "Check the strongest features and confirm the reference is today's real process.")
    chk = tr.get("checks") or {}
    ts, cs = chk.get("train_score"), chk.get("cv_score")
    target_encoded = bool((((spec.get("features") or {}).get("config") or {}).get("high_cardinality")))
    if ts is not None and cs and abs(cs) > 0 and higher and (ts - cs) / abs(cs) > 0.25 and not target_encoded:
        F.add("training", "overfit", "", "statistical strength", "warning", "The champion scores much better on training rows",
              f"Training score {ts:.4f} against CV {cs:.4f}.", "Use stronger regularisation or fewer features in /ml-model-training.")
    oof = _read(root, (tr.get("files") or {}).get("oof_predictions"))
    sp = spec.get("split") or {}
    train = _read(root, (sp.get("files") or {}).get("train"))
    rid = sp.get("row_id_col") or "row_id"
    if oof is not None and train is not None and rid in oof and rid in train and sp.get("target") in train:
        o = oof[oof["model"] == champ.get("name")].merge(train[[rid, sp["target"]]], on=rid)
        y, p = o[sp["target"]].astype(float).values, o["prediction"].astype(float).values
        ok = ~np.isnan(p)
        if ok.sum() > 50:
            from sklearn.metrics import r2_score, roc_auc_score
            if _family(spec) == "binary" and len(np.unique(y[ok])) == 2:
                auc = roc_auc_score(y[ok], p[ok])
                if auc >= 0.95:
                    F.add("training", "too_good_auc", "", "results too good", "warning", "The out of fold AUC is very high",
                          f"The champion's out of fold ROC AUC is {auc:.3f}. Values this high are rare for real "
                          "business problems and often come from a leak.",
                          "Check the strongest features and the timing of each input.")
            elif _family(spec) != "binary":
                r2 = r2_score(y[ok], p[ok])
                if r2 >= 0.98:
                    F.add("training", "too_good_r2", "", "results too good", "warning", "The out of fold R2 is very high",
                          f"The champion's out of fold R2 is {r2:.3f}.", "Check the lags and the feature timing.")
    for p in chk.get("problems") or []:
        F.add("training", "problem", str(p)[:40], "statistical strength", "note", "Model training reported a problem",
              str(p), "Read the checks in notebooks/08_training.ipynb.")


def check_evaluation(spec, F):
    ev, fr = spec.get("evaluation") or {}, spec.get("framing") or {}
    if not ev:
        return
    sc = (ev.get("checks") or {}).get("self_check")
    if sc == "off":
        F.add("evaluation", "self_check_off", "", "reproducibility", "warning", "The scoring self-check was turned off",
              "The evaluation did not confirm that each model's CV score reproduces through its scoring path before the "
              "test set was opened, so a scoring bug could have reached the test results.",
              "Set SELF_CHECK = True in 09_evaluation.ipynb and run it again, unless the models are too slow to refit.")
    elif isinstance(sc, list):
        bad = [r.get("model") for r in sc if r.get("ok") is False]
        if bad:
            F.add("evaluation", "self_check_failed", "", "reproducibility", "blocker",
                  "Some models do not reproduce their CV score in evaluation",
                  "These models score differently through the evaluation's scoring path than in training: "
                  + ", ".join(map(str, bad)) + ". Their test scores are not trustworthy.",
                  "Fix the scoring code (eval_core.models_to_score or model_core), then run the evaluation again.")
    elif ev.get("metrics"):
        F.add("evaluation", "self_check_missing", "", "reproducibility", "note", "The evaluation has no self-check record",
              "It ran before the scoring self-check existed.", "Run /ml-offline-evaluation again to record it.")
    rows = {r["model"]: r for r in ev.get("metrics") or []}
    champ = rows.get(ev.get("champion")) or {}
    test, cv = champ.get("test"), champ.get("cv")
    if test is not None and cv:
        rel = (test - cv) / abs(cv)
        key = (((spec.get("baseline") or {}).get("metrics") or {}).get("primary") or {}).get("key")
        try:
            import model_core
            higher = model_core.HIGHER_IS_BETTER.get(key) is not False
        except Exception:  # noqa: BLE001
            higher = True
        better = rel if higher else -rel
        if better > 0.15:
            F.add("evaluation", "test_above_cv", "", "results too good", "warning", "The test score is well above the CV score",
                  f"Test {test:.4f} against CV {cv:.4f}. A test set that is easier than the folds, or a leak that only "
                  "helps the test rows, can cause this.", "Compare the test period with the CV folds, and check the slices.")
        elif better < -0.15:
            F.add("evaluation", "test_below_cv", "", "statistical strength", "warning", "The test score is well below the CV score",
                  f"Test {test:.4f} against CV {cv:.4f}. The tuning may have fitted the folds, or the data changed.",
                  "Check the score drift and the metric over the test period.")
    for c in ev.get("comparisons") or []:
        if c.get("low") is not None and c.get("high") is not None and c["low"] <= 0 <= c["high"]:
            F.add("evaluation", "interval_crosses_zero", c.get("comparison"), "statistical strength", "warning",
                  f"The gain is not clear: {c.get('comparison')}",
                  f"The difference is {c['difference']:+.4f}, but its 95% interval runs from {c['low']:+.4f} to "
                  f"{c['high']:+.4f}, which includes zero.", "Treat the gain as uncertain when deciding.")
    bar = str((fr.get("success") or {}).get("v1_bar") or "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", bar)
    ref = next((c for c in ev.get("comparisons") or [] if c.get("comparison", "").endswith(
        "against " + str(((spec.get("baseline") or {}).get("reference") or {}).get("name")))), None)
    if m and ref and ref.get("relative %") is not None:
        need = float(m.group(1))
        ref_score = ref["difference"] / (ref["relative %"] / 100) if ref["relative %"] else None
        low_pct = ref["low"] / ref_score * 100 if ref_score else None
        if ref["relative %"] < need:
            F.add("evaluation", "v1_bar_missed", "", "business fit", "blocker", "The V1 bar is not met on the test set",
                  f"The V1 bar asks for {need:g}% over the reference; the champion is {ref['relative %']:+.1f}% on test.",
                  "Iterate on features or models, or revisit the bar with the business owner.")
        elif low_pct is not None and low_pct < need:
            F.add("evaluation", "v1_bar_uncertain", "", "statistical strength", "warning",
                  "The V1 bar is met, but not across the whole interval",
                  f"The champion is {ref['relative %']:+.1f}% over the reference, but the low end of the interval is "
                  f"about {low_pct:+.1f}%, below the {need:g}% bar.", "Say so when presenting the result.")
    opened = ev.get("test_opened") or []
    if len({(o.get("champion"), o.get("data_version")) for o in opened}) > 1:
        F.add("evaluation", "test_reused", "", "test set use", "warning", "The test set was used with more than one model or data version",
              f"The test set was opened {len(opened)} times, for " + ", ".join(sorted({str(o.get('champion')) for o in opened}))
              + ". If any change followed earlier test results, the latest test score is optimistic.",
              "Record why each change was made. For a fully honest estimate, score a fresh holdout period.")
    psi = ev.get("score_psi")
    if psi is not None and psi >= 0.25 and ev.get("target_psi") is None:
        F.add("evaluation", "drift", "", "statistical strength", "warning", "The champion's scores shifted on the test set",
              f"Score drift (PSI) of {psi:.2f} between the out of fold and test scores.",
              "Find the inputs that changed most before relying on the model.")
    for s in ev.get("slices") or []:
        if s.get("worse than reference") and not s.get("too small") and not (
                s.get("champion acts on") is not None and s["champion acts on"] < s.get("reference acts on", 0)):
            F.add("evaluation", "weak_slice", f"{s.get('slice')}={s.get('value')}", "business fit", "note",
                  f"The champion is worse than the reference in a slice: {s.get('slice')} = {s.get('value')}",
                  f"Champion {s.get('champion')}, reference {s.get('reference')}, interval {s.get('interval')}.",
                  "Decide whether this group needs its own rule or a closer look.")
    made = [d for d in spec.get("decisions") or [] if isinstance(d, dict) and d.get("stage") == "evaluation"]
    if not made:
        F.add("evaluation", "no_decision", "", "business fit", "note", "No go or no go decision is recorded",
              "The evaluation ran, but no decision with stage evaluation is in the spec.",
              "Record the decision and its reason in the evaluation notebook.")


def check_reproducibility(spec, root, F):
    if not os.path.exists(os.path.join(root, "uv.lock")):
        F.add("project", "no_lock", "", "reproducibility", "note", "There is no uv.lock file",
              "Package versions are not pinned, so the results may not reproduce later.", "Run uv lock in the project.")
    sp, tr = spec.get("split") or {}, spec.get("training") or {}
    if sp and sp.get("seed") is None:
        F.add("split", "no_seed", "", "reproducibility", "note", "The split has no seed",
              "Without a seed the split changes on every run.", "Set a seed in /ml-train-test-split.")
    if tr and (tr.get("settings") or {}).get("seed") is None:
        F.add("training", "no_seed", "", "reproducibility", "note", "Training has no seed",
              "Without a seed the tuning changes on every run.", "Set a seed in /ml-model-training.")


def check_tracking(spec, F):
    """The MLflow runs agree with the spec: every finished stage has a run, and the registered champion is the
    spec's champion."""
    tr = spec.get("tracking") or {}
    if not tr.get("experiment_id"):
        return
    for stage in REVIEW_STAGES:
        if stage in ("split", "features", "baseline", "training", "evaluation") and spec.get(stage) \
                and not (spec.get(stage) or {}).get("mlflow"):
            F.add(stage, "no_mlflow_run", "", "reproducibility", "note", "The last run of this stage is not in MLflow",
                  f"The {stage} section has no MLflow run, usually because the server was down when the notebook ran.",
                  f"Start the server and run the {stage} notebook again (Run All).")
    champ = (spec.get("training") or {}).get("champion") or {}
    if not champ or not (spec.get("training") or {}).get("mlflow"):
        return
    try:
        import tracking
        mlflow, uri = tracking.ready(spec)
        if mlflow is None:
            return
        from mlflow.tracking import MlflowClient
        name = tr.get("registered_model") or tracking.registered_model_name(spec)
        mv = MlflowClient().get_model_version_by_alias(name, "champion")
    except Exception:  # noqa: BLE001  no registry entry or no server: nothing to compare
        return
    want = (str(champ.get("run_id")), str((spec.get("training") or {}).get("data_version")))
    have = (mv.tags.get("run_log_id"), mv.tags.get("data_version"))
    if want != have:
        F.add("training", "registry_champion", "", "consistency", "warning",
              "The registered champion is not the spec's champion",
              f"{name} version {mv.version} (alias champion) comes from run {have[0]} on data version {have[1]}, but "
              f"the spec's champion is run {want[0]} on data version {want[1]}.",
              "Run the training notebook again with the server up, so the champion is registered again.")


def run_checks(spec, root, stages=None):
    """Every scripted check for the given stages (default: every stage that is done). Returns a list of findings."""
    stages = [s for s in (stages or done_stages(spec)) if s in REVIEW_STAGES]
    F = Findings(stages)
    check_consistency(spec, F)
    check_pipeline(spec, root, F)
    check_split(spec, root, F)
    check_features(spec, root, F)
    check_training(spec, root, F)
    check_evaluation(spec, F)
    check_reproducibility(spec, root, F)
    check_tracking(spec, F)
    return F.sorted()


# ---------------------------------------------------------------- findings history

def finding_id(f):
    """The id of a finding; reviewer findings get one from their stage, category, and title."""
    if f.get("id"):
        return str(f["id"])
    return f"{f.get('stage')}.reviewer.{_slug(f.get('category'))}.{_slug(f.get('title'))[:40]}"


def merge_history(previous, current, reviewed_stages, accepted, run_at):
    """The findings table after this review.

    previous: the earlier findings table (or None). current: this review's findings. A finding open earlier and not
    found now is fixed, but only when its stage was reviewed this time. accepted: {id: reason}."""
    cols = list(FINDING_KEYS) + ["status", "accepted_reason", "first_seen", "last_seen"]
    prev = previous.copy() if previous is not None and len(previous) else pd.DataFrame(columns=cols)
    prev = prev.set_index("id", drop=False) if len(prev) else prev
    rows = []
    seen = set()
    for f in current:
        fid = finding_id(f)
        seen.add(fid)
        old = prev.loc[fid] if len(prev) and fid in prev.index else None
        row = {k: f.get(k) for k in FINDING_KEYS}
        row["id"] = fid
        row["first_seen"] = old["first_seen"] if old is not None else run_at
        row["last_seen"] = run_at
        if fid in accepted:
            row["status"], row["accepted_reason"] = "accepted", str(accepted[fid])
        else:
            row["status"], row["accepted_reason"] = "open", None
        rows.append(row)
    for _, old in (prev.iterrows() if len(prev) else []):
        if old["id"] in seen:
            continue
        row = old.to_dict()
        if old["stage"] in reviewed_stages or old["stage"] == "project":
            if old["status"] in ("open", "accepted"):
                row["status"] = "fixed" if old["source"] == "script" else "not repeated"
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    out["severity_rank"] = out["severity"].map(SEVERITY_ORDER)
    status_rank = {"open": 0, "accepted": 1, "not repeated": 2, "fixed": 3}
    out = out.assign(_s=out["status"].map(status_rank)).sort_values(["_s", "severity_rank", "stage", "id"])
    return out.drop(columns=["_s", "severity_rank"]).reset_index(drop=True)


# ---------------------------------------------------------------- brief for the reviewer

def files_to_read(spec, root, stages):
    """The files the independent reviewer should read for these stages."""
    runs = spec.get("notebooks") or {}
    out = [ml_common.SPEC_NAME]
    eda = spec.get("eda") or {}
    for s in ["business", "discovery"] + [x for x in REVIEW_STAGES if x in stages]:
        if s == "eda" and eda.get("report"):
            out.append(eda["report"])
        elif (runs.get(s) or {}).get("path"):
            out.append(runs[s]["path"])
    code = (spec.get("features") or {}).get("code") or {}
    if "features" in stages:
        out += [code[k] for k in ("features", "custom") if code.get(k)]
    for doc in ("business_framing.md", "data_dictionary.yaml", "model_card.md"):
        p = os.path.join(ml_common.project_paths(spec, root)["docs"], doc)
        if os.path.exists(p):
            out.append(ml_common.relpath(p, root))
    return [p for p in dict.fromkeys(out) if os.path.exists(os.path.join(root, p))]


def brief(spec, root, stages, findings):
    """Plain text context for the independent reviewer."""
    import yaml
    pr, fr = spec.get("project") or {}, spec.get("framing") or {}
    lines = [f"Project: {pr.get('name')} at {root}",
             f"Task: {fr.get('task_type')}; target {((fr.get('target') or {}).get('name'))}; "
             f"unit: {fr.get('unit_of_prediction')}; prediction time: {(fr.get('prediction_time') or {}).get('description')}",
             f"Stages to review: {', '.join(stages)}",
             f"Project Python: {ml_common.relpath(ml_common.project_python(root) or '.venv/bin/python', root)}",
             "", "Files to read:"] + [f"  {p}" for p in files_to_read(spec, root, stages)]
    data = {k: (spec.get(k) or {}).get("files") for k in ("split", "features", "baseline", "training", "evaluation")
            if (spec.get(k) or {}).get("files")}
    if data:
        lines += ["", "Saved data (parquet; read only, never print raw rows):",
                  yaml.safe_dump(data, sort_keys=False, width=120).rstrip()]
    lines += ["", f"Scripted findings ({len(findings)}), already reported; do not repeat them:"]
    lines += [f"  [{f['severity']}] {f['id']}: {f['title']}" for f in findings] or ["  none"]
    return "\n".join(lines)


def main():
    import yaml
    args = sys.argv[1:]
    cmd = args[0] if args else "check"
    root = ml_common.find_project_root()[0] or "."
    os.chdir(root)
    sys.path.insert(0, root)
    spec = ml_common.migrate(ml_common.load(ml_common.spec_path()))
    stages = [a for a in args[1:] if a in REVIEW_STAGES] or None
    if cmd == "stages":
        done = done_stages(spec)
        print("Stages done: " + (", ".join(done) or "none"))
        rv = spec.get("review") or {}
        if rv.get("run_at"):
            print(f"Last review: {rv['run_at']} of {', '.join(rv.get('stages') or [])}; open blockers "
                  f"{rv.get('open', {}).get('blocker', 0)}, warnings {rv.get('open', {}).get('warning', 0)}.")
            last = _ts(rv["run_at"])
            changed = [s for s, r in (spec.get("notebooks") or {}).items()
                       if s in REVIEW_STAGES and _ts(r.get("run_at")) and last and _ts(r.get("run_at")) > last]
            print("Ran again since the last review: " + (", ".join(changed) or "none"))
        else:
            print("Last review: none")
        return
    findings = run_checks(spec, root, stages)
    if cmd == "brief":
        print(brief(spec, root, stages or done_stages(spec), findings))
    else:
        print(yaml.safe_dump(findings, sort_keys=False, allow_unicode=True, width=120))


if __name__ == "__main__":
    main()
