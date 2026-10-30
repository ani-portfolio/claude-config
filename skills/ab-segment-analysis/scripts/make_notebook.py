#!/usr/bin/env python3
"""Write notebooks/ab05_segments.ipynb from the segment answers, and run it once.

The notebook estimates the effect of each treatment on the primary metric (or another metric) within the segments of
each planned column, tests whether the effects differ between segments with Cochran's Q, and controls the false
discovery rate across columns with Benjamini-Hochberg. Only columns recorded before assignment are allowed, unless
the user chose otherwise with a reason. Running it writes the segments section of ab_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml [--overwrite | --new-version] [--no-run]
    make_notebook.py --skip --reason "No segments were planned in the framing."

answers.yaml keys: segment_cols, metric (default: the primary), bins (quantile bins for numeric columns),
min_units, allow_post_treatment (with a reason in decisions), reasoning, assumptions, open_questions, decisions.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ab-segment-analysis"
STAGE = "segments"


def build(answers, spec, scripts_dir, nb_name):
    a = answers
    reasoning = a.get("reasoning") or {}
    r = lambda k: (reasoning[k].strip() + "\n\n") if reasoning.get(k) else ""
    pm = (spec.get("framing") or {}).get("primary_metric") or {}
    cells = nb_common.setup_cells(
        STAGE, "Segment Analysis",
        "Whether the effect differs between groups of units, for segments planned before the results were seen. With "
        "many segments, some will look different by chance, so the question per column is whether the effects differ "
        "more than chance allows (Cochran's Q), corrected across columns (Benjamini-Hochberg). A segment's own p value "
        "is shown, but it is not a finding on its own.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the columns and settings. Edit them, then "
        "choose Run All.",
        scripts_dir, nb_name)
    cells.append(code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ab_core
from IPython.display import display
d = ab_common.test_defaults(spec)
CONTROL, TREATMENTS = d["control"], d["treatments"]
blockers = ab_common.analysis_blockers(spec)
if blockers:
    print("Warning, the analysis is not ready: " + "; ".join(blockers) + ".")
'''))
    cells.append(md("## Segments\n\n" + r("segments") +
                    "*`SEGMENT_COLS` must be columns recorded before assignment (`data.pre_treatment_cols`). A column the "
                    "treatment can change (for example the number of ads seen) splits units by an outcome of the "
                    "treatment, so its segments are no longer comparable; `ALLOW_POST_TREATMENT = True` permits it only "
                    "with a reason in `decisions`, and the result is labelled descriptive. Numeric columns with more "
                    "than `BINS` values are cut into that many quantile bins. Segments with fewer than `MIN_UNITS` units "
                    "in a variant are shown but left out of the test.*"))
    cells.append(code(
        "# Decision\n"
        f"SEGMENT_COLS = {lit(list(a.get('segment_cols') or []))}\n"
        f"METRIC = {(a.get('metric') or pm.get('name'))!r}\n"
        f"BINS = {a.get('bins', 4)!r}\n"
        f"MIN_UNITS = {a.get('min_units', 100)!r}\n"
        f"ALLOW_POST_TREATMENT = {bool(a.get('allow_post_treatment'))!r}", decision=True))
    cells.append(md("## Load and Check the Columns"))
    cells.append(code('''
df = pd.read_parquet(ROOT / spec["validation"]["files"]["units"])
m = next(x for x in d["metrics"] if x["name"] == METRIC)
col = d["metric_cols"][METRIC]
df[col] = df[col].astype(float)
post = [c for c in SEGMENT_COLS if c not in d["pre_treatment_cols"]]
if post and not ALLOW_POST_TREATMENT:
    raise ValueError("These columns are not recorded before assignment: " + ", ".join(post) +
                     ". Remove them, or set ALLOW_POST_TREATMENT = True with a reason in decisions.")
if post:
    print("Descriptive only, because these columns were recorded after assignment: " + ", ".join(post))
missing = [c for c in SEGMENT_COLS if c not in df.columns]
assert not missing, f"Not in the unit table: {missing}. Add them to the timing cells of ab03_validation and rerun it."
for c in SEGMENT_COLS:
    if df[c].dtype.kind in "biuf" and df[c].nunique() > BINS:
        df[c] = pd.qcut(df[c], BINS, duplicates="drop").astype(str)
print(f"{len(df):,} units, metric {METRIC} ({col}), {len(SEGMENT_COLS)} segment column(s).")
'''))
    cells.append(md("## Effects by Segment"))
    cells.append(code('''
results, seg_rows = [], []
for t in TREATMENTS:
    for c in SEGMENT_COLS:
        h = ab_core.heterogeneity(df, "ab_variant", col, c, CONTROL, t, m.get("type") or "continuous",
                                  d["alpha"], MIN_UNITS)
        results.append({"column": c, "treatment": t, "q": h["q"], "df": h["df"], "p_value": h["p_value"],
                        "post_treatment": c in post, "segments": h["segments"]})
        seg_rows += [{"column": c, "treatment": t, **s} for s in h["segments"]]
pvals = [x["p_value"] for x in results if x["p_value"] is not None]
adj = iter(ab_core.benjamini_hochberg(pvals)) if pvals else iter([])
for x in results:
    x["p_bh"] = next(adj) if x["p_value"] is not None else None
    x["differs"] = bool(x["p_bh"] is not None and x["p_bh"] < d["alpha"] and not x["post_treatment"])
display(pd.DataFrame(results).drop(columns="segments"))
seg_df = pd.DataFrame(seg_rows)
display(seg_df)
'''))
    cells.append(md("## Chart\n\nThe absolute effect in each segment with its interval. The dashed line is no effect."))
    cells.append(code('''
ok = seg_df[~seg_df.too_small] if "too_small" in seg_df else seg_df
for (c, t), part in ok.groupby(["column", "treatment"]):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(part) + 1.2))
    y = np.arange(len(part))
    lo = part["diff"] - part.diff_ci.str[0]
    hi = part.diff_ci.str[1] - part["diff"]
    ax.errorbar(part["diff"], y, xerr=[lo, hi], fmt="o", capsize=3)
    ax.axvline(0, ls="--", color="grey"); ax.set_yticks(y, part.segment)
    ax.set_title(f"{METRIC}: {t} against {CONTROL}, by {c}"); plt.tight_layout(); plt.show()
'''))
    cells.append(md("## Assumptions, Open Questions, and Decisions"))
    cells.append(code(
        "# Decision\n"
        f"assumptions = {lit(list(a.get('assumptions') or []))}\n\n"
        f"open_questions = {lit(list(a.get('open_questions') or []))}\n\n"
        f"decisions = {lit(list(a.get('decisions') or []))}", decision=True))
    cells.append(md("## Write to the Spec"))
    cells.append(code(f'''
assert not post or decisions, "Record in decisions why columns recorded after assignment are used."
segments = {{"status": "done", "metric": METRIC, "results": results, "post_treatment_ok": bool(post),
            "run_at": pd.Timestamp.now().floor("s").isoformat()}}
errors, _, _ = nb_common.write_stage(SPEC_PATH, "segments", segments, "{SKILL}", assumptions, open_questions, decisions,
                                     replace=True)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
for x in results:
    diff = x["p_bh"] is not None and x["p_bh"] < d["alpha"]
    print(f"{{x['column']}} ({{x['treatment']}}): " + ("effects differ between segments" if diff else
          "no clear difference between segments") + f", p after correction {{ab_core.fmt_p(x['p_bh'])}}"
          + (" (descriptive only, because the column was recorded after assignment)" if x["post_treatment"] else "") + ".")
print("Spec problems:", errors or "none")
'''))
    cells.append(md("## Next Step\n\nRun `/ab-readout`. A segment where the effect differs is a hypothesis for the next "
                    "test, not a reason to ship to that segment alone."))
    cells += nb_common.tracking_cells(SKILL)
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?")
    ap.add_argument("--skip", action="store_true", help="mark the segments as skipped, without a notebook")
    ap.add_argument("--reason")
    ap.add_argument("--spec")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--new-version", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    spec_path = ab_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ab_common.load_or_create(spec_path)
    if a.skip:
        if not a.reason:
            ap.error("--skip needs --reason")
        ab_common.update_section(spec_path, "segments", {"status": "skipped", "reason": a.reason}, SKILL,
                                 action="skipped the segments: " + a.reason)
        print(f"Segments marked as skipped: {a.reason}")
        return
    if not ((spec.get("validation") or {}).get("files") or {}).get("units"):
        sys.exit("The validation has not saved the unit table yet. Run ab-data-validation first.")
    if not a.answers:
        ap.error("give an answers file, or --skip with --reason")
    answers = ab_common.load(a.answers)
    if not answers.get("segment_cols"):
        sys.exit("No segment columns given. Use --skip with a reason if no segments were planned.")
    pre = ab_common.test_defaults(spec)["pre_treatment_cols"]
    post = [c for c in answers["segment_cols"] if c not in pre]
    if post and not answers.get("allow_post_treatment"):
        sys.exit("Not recorded before assignment: " + ", ".join(post) + ". Segmenting by them compares units the "
                 "treatment has already sorted. Remove them, or ask the user and set allow_post_treatment: true with "
                 "the reason in decisions.")
    if post and not answers.get("decisions"):
        sys.exit("allow_post_treatment is set, but decisions does not say why. Add the reason.")
    nb_name = ab_common.notebook_name(STAGE)
    path = os.path.join(ab_common.project_paths(spec, root)["notebooks"], nb_name)
    nb = build(answers, spec, HERE, nb_name)
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    if res["status"] == "edited":
        print(f"Not written: {ab_common.relpath(path, root)} was edited after the skill wrote it. "
              f"Rerun with --overwrite to replace it, or --new-version to write {os.path.basename(res['next_version'])}.")
        sys.exit(2)
    print(f"Wrote notebook: {ab_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if not res["executed"]:
        sys.exit(1)
    for x in (ab_common.load(spec_path).get("segments") or {}).get("results") or []:
        pb = x.get("p_bh")
        alpha = ab_common.test_defaults(ab_common.load(spec_path))["alpha"]
        print(f"{x['column']} ({x['treatment']}): {'differs' if pb is not None and pb < alpha else 'no clear difference'}, "
              f"corrected p {'below 0.0001' if pb is not None and pb < 1e-4 else pb}"
              + (" (descriptive, recorded after assignment)" if x.get("post_treatment") else ""))


if __name__ == "__main__":
    main()
