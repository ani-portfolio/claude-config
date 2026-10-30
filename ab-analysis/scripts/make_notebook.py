#!/usr/bin/env python3
"""Write notebooks/ab04_analysis.ipynb from the analysis answers, and run it once.

The notebook estimates the effect of each treatment on every metric in the framing: the absolute and relative lift
with confidence intervals, the p value, a Holm correction for secondary metrics (and for the primary when there are
several treatments), guardrail checks, an optional CUPED adjustment with a column recorded before assignment, a
bootstrap check for continuous metrics, the effect the test could detect, and a plain label for each result. Running
it saves data/processed/ab_results.parquet and writes the analysis section of ab_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml [--overwrite | --new-version] [--no-run]
    make_notebook.py --defaults [--overwrite]     # take every setting from the spec

answers.yaml keys: alpha, sides, mde_abs, cuped_col, n_boot, drop_crossed_units, aggregate, reasoning, assumptions,
open_questions, decisions, resolved_questions.
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

SKILL = "ab-analysis"
STAGE = "analysis"


def defaults(spec):
    d = ab_common.test_defaults(spec)
    return {"alpha": d["alpha"], "sides": d["sides"], "mde_abs": d["mde_abs"], "cuped_col": None, "n_boot": 1000,
            "drop_crossed_units": True, "aggregate": {"binary": "max", "continuous": "sum", "count": "sum"}}


def build(answers, spec, scripts_dir, nb_name):
    s = {**defaults(spec), **{k: v for k, v in answers.items() if k in defaults(spec) and v is not None}}
    if "mde_abs" in answers:
        s["mde_abs"] = answers["mde_abs"]
    reasoning = answers.get("reasoning") or {}
    r = lambda k: (reasoning[k].strip() + "\n\n") if reasoning.get(k) else ""
    cells = nb_common.setup_cells(
        STAGE, "Analysis",
        "The effect of each treatment on every metric in the test framing, with confidence intervals and a plain "
        "label for each result. The metrics, variants, and ship rule come from `ab01_test_framing.ipynb`, and the "
        "column mapping from `ab03_validation.ipynb`.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the settings. Edit them, then choose Run All. "
        "Changing a setting after seeing the results (for example switching the primary metric or the sidedness) "
        "must be recorded in `decisions` with the reason.",
        scripts_dir, nb_name)
    cells.append(code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ab_core
from IPython.display import display
pd.set_option("display.float_format", lambda x: f"{x:,.5f}")

blockers = ab_common.analysis_blockers(spec)
if blockers:
    print("Warning, the analysis is not ready: " + "; ".join(blockers) + ".")
'''))
    cells.append(md("## Settings\n\n" + r("settings") +
                    "*`ALPHA` and `SIDES` should match the design. `MDE_ABS` is the smallest absolute change worth acting "
                    "on, in the units of the primary metric (0.005 is half a percentage point); None skips the practical "
                    "significance check. `CUPED_COL` must be a column recorded before assignment, or None. `N_BOOT` is the "
                    "number of bootstrap resamples for continuous metrics (0 skips the check).*"))
    cells.append(code(
        "# Decision\n"
        f"ALPHA = {s['alpha']!r}\n"
        f"SIDES = {s['sides']!r}  # two or one\n"
        f"MDE_ABS = {s['mde_abs']!r}\n"
        f"CUPED_COL = {s['cuped_col']!r}\n"
        f"N_BOOT = {s['n_boot']!r}\n"
        f"DROP_CROSSED_UNITS = {s['drop_crossed_units']!r}  # drop units seen in more than one variant\n"
        f"AGGREGATE = {lit(s['aggregate'])}  # how repeated rows of one unit are combined, by metric type",
        decision=True))
    cells.append(md("## Load the Units\n\nOne row per unit is required, because the unit is what was randomised. "
                    "Repeated rows are combined with `AGGREGATE`, and units seen in more than one variant are dropped."))
    cells.append(code('''
d = ab_common.test_defaults(spec)
U, CONTROL, TREATMENTS = d["unit_col"], d["control"], d["treatments"]
assert d["post_treatment_cols"] is not None
if CUPED_COL:
    assert CUPED_COL in d["pre_treatment_cols"], f"CUPED_COL {CUPED_COL} is not listed as recorded before assignment."
df = pd.read_parquet(ROOT / spec["validation"]["files"]["units"])
n0 = len(df)
crossed = df.groupby(U)["ab_variant"].nunique()
crossed = crossed[crossed > 1].index
if DROP_CROSSED_UNITS and len(crossed):
    df = df[~df[U].isin(crossed)]
METRICS = [dict(m, column=d["metric_cols"].get(m["name"])) for m in d["metrics"]]
METRICS = [m for m in METRICS if m["column"] in df.columns]
if df[U].duplicated().any():
    agg = {m["column"]: AGGREGATE.get(m.get("type") or "continuous", "sum") for m in METRICS}
    agg.update({c: "first" for c in df.columns if c not in agg and c != U})
    df = df.groupby(U, as_index=False).agg(agg)
for m in METRICS:
    df[m["column"]] = df[m["column"]].astype(float)
print(f"{n0:,} rows, {len(crossed):,} units in more than one variant" + (" (dropped)" if DROP_CROSSED_UNITS else "")
      + f", {len(df):,} units analysed.")
display(df.groupby("ab_variant")[U].count().rename("units").to_frame())
'''))
    cells.append(md("## Variance Reduction\n\nCUPED subtracts the part of each metric that a column recorded before "
                    "assignment predicts, which narrows the intervals without biasing the effect. It is skipped when "
                    "`CUPED_COL` is None."))
    cells.append(code('''
cuped = {}
if CUPED_COL:
    x = df[CUPED_COL].astype(float).fillna(df[CUPED_COL].astype(float).mean())
    for m in METRICS:
        adj, theta, removed = ab_core.cuped(df[m["column"]], x)
        df[m["column"] + "__cuped"] = adj
        cuped[m["name"]] = {"theta": theta, "variance_removed": removed}
    display(pd.DataFrame(cuped).T)
else:
    print("No CUPED: no column recorded before assignment was chosen" +
          (" (none is listed in the spec)." if not d["pre_treatment_cols"] else "."))
'''))
    cells.append(md("## Effects\n\nFor binary metrics, a two proportion z test; for continuous and count metrics, "
                    "Welch's t test. Intervals are two sided at `1 - ALPHA`. The relative lift interval uses the delta "
                    "method. Secondary metrics get Holm adjusted p values, and so does the primary when there are several "
                    "treatments. A guardrail is breached when it moves significantly in the harmful direction."))
    cells.append(code('''
rows = []
for t in TREATMENTS:
    for m in METRICS:
        col = m["column"] + ("__cuped" if CUPED_COL else "")
        if m.get("type") == "binary" and not CUPED_COL:
            res = ab_core.compare(df, "ab_variant", col, CONTROL, t, "binary", ALPHA, SIDES, m.get("direction"))
        else:
            res = ab_core.compare(df, "ab_variant", col, CONTROL, t, "continuous", ALPHA, SIDES, m.get("direction"))
        rows.append({"treatment": t, "metric": m["name"], "role": m["role"], "type": m.get("type"),
                     "direction": m.get("direction"), **res})
res_df = pd.DataFrame(rows)
res_df["p_holm"] = np.nan
for role in ("secondary", "primary"):
    for t in TREATMENTS:
        idx = res_df.index[(res_df.role == role) & (res_df.treatment == t)] if role == "secondary" else \\
              res_df.index[res_df.role == "primary"]
        if len(idx):
            res_df.loc[idx, "p_holm"] = ab_core.holm(res_df.loc[idx, "p_value"].tolist())
        if role == "primary":
            break
res_df["p_used"] = np.where(res_df.role == "guardrail", res_df.p_value, res_df.p_holm)
res_df["significant"] = res_df.p_used < ALPHA
harmful = np.where(res_df.direction == "decrease", res_df["diff"] > 0, res_df["diff"] < 0)
res_df["breached"] = (res_df.role == "guardrail") & harmful & (res_df.p_value < ALPHA)
res_df["result_label"] = [ab_core.label_result({**r, "p_value": r["p_used"]}, MDE_ABS if r["role"] == "primary" else None,
                                               r["direction"], ALPHA) for r in res_df.to_dict("records")]
show = ["treatment", "metric", "role", "control_mean", "treatment_mean", "diff", "diff_ci", "rel_lift", "rel_ci",
        "p_value", "p_holm", "result_label"]
display(res_df[show])
'''))
    cells.append(md("## Chart\n\nThe relative lift of each metric with its interval. The dashed line is no effect."))
    cells.append(code('''
fig, ax = plt.subplots(figsize=(8, 0.6 * len(res_df) + 1.5))
labels = [f"{r.metric} ({r.treatment})" for r in res_df.itertuples()]
y = np.arange(len(res_df))
lo = res_df.rel_lift - res_df.rel_ci.str[0]
hi = res_df.rel_ci.str[1] - res_df.rel_lift
ax.errorbar(res_df.rel_lift, y, xerr=[lo, hi], fmt="o", capsize=4)
ax.axvline(0, ls="--", color="grey")
ax.set_yticks(y, labels); ax.set_xlabel(f"relative lift with {1 - ALPHA:.0%} interval")
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
plt.tight_layout(); plt.show()
'''))
    cells.append(md("## Checks\n\nA bootstrap interval for continuous metrics, as a check on the normal approximation, "
                    "and the smallest effect this sample could detect with 80% power. A result that is not significant "
                    "only rules out effects larger than the detectable effect."))
    cells.append(code('''
checks = []
for t in TREATMENTS:
    for m in METRICS:
        c = df.loc[df.ab_variant == CONTROL, m["column"]]
        tt = df.loc[df.ab_variant == t, m["column"]]
        if m.get("type") == "binary":
            detect = ab_core.mde_proportions(c.mean(), len(c), len(tt), ALPHA, 0.8, SIDES)
            boot = None
        else:
            detect = ab_core.mde_means(c.std(ddof=1), len(c), len(tt), ALPHA, 0.8, SIDES)
            boot = ab_core.bootstrap(c, tt, ALPHA, N_BOOT)["diff_ci"] if N_BOOT else None
        checks.append({"treatment": t, "metric": m["name"], "detectable_effect_abs": detect,
                       "detectable_effect_rel": detect / c.mean() if detect and c.mean() else None, "bootstrap_diff_ci": boot})
display(pd.DataFrame(checks))
'''))
    cells.append(md("## Assumptions, Open Questions, and Decisions\n\n" + r("notes") +
                    "`resolved_questions` closes earlier open questions this analysis settles, as `{\"id\": \"Q1\", \"answer\": \"...\"}`."))
    cells.append(code(
        "# Decision\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []))}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []))}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []))}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []))}", decision=True))
    cells.append(md("## Save and Write to the Spec\n\nSaves every comparison to `data/processed/ab_results.parquet` and "
                    "writes the `analysis` section of `ab_spec.yaml`."))
    cells.append(code(f'''
out = res_df.copy()
for col in ("diff_ci", "rel_ci"):
    out[col + "_low"], out[col + "_high"] = out[col].str[0], out[col].str[1]
results_path = os.path.join(PATHS["data_processed"], "ab_results.parquet")
out.drop(columns=["diff_ci", "rel_ci"]).to_parquet(results_path, index=False)

def entry(r):
    keep = ["metric", "treatment", "method", "control_mean", "treatment_mean", "n_control", "n_treatment", "diff",
            "diff_ci", "rel_lift", "rel_ci", "p_value", "p_holm", "significant", "result_label"]
    e = {{k: r[k] for k in keep}}
    if r["role"] == "guardrail":
        e["breached"] = bool(r["breached"])
    return e

recs = res_df.to_dict("records")
primary = [entry(x) for x in recs if x["role"] == "primary"]
analysis = {{"alpha": ALPHA, "sides": SIDES, "mde_abs": MDE_ABS, "cuped_col": CUPED_COL,
            "units_analysed": int(len(df)), "units_dropped_crossed": int(len(crossed)) if DROP_CROSSED_UNITS else 0,
            "primary": primary[0] if len(primary) == 1 else {{**primary[0], "all_treatments": primary}},
            "secondary": [entry(x) for x in recs if x["role"] == "secondary"],
            "guardrails": [entry(x) for x in recs if x["role"] == "guardrail"],
            "checks": checks, "files": {{"results": ab_common.relpath(results_path, str(ROOT))}},
            "run_at": pd.Timestamp.now().floor("s").isoformat()}}
errors, _, _ = nb_common.write_stage(SPEC_PATH, "analysis", analysis, "{SKILL}", assumptions, open_questions, decisions,
                                     replace=True)
nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
p = analysis["primary"]
print(f"Primary: {{p['metric']}}, {{p['treatment']}} against {{CONTROL}}: {{p['control_mean']:.4%}} to {{p['treatment_mean']:.4%}}, "
      f"lift {{p['rel_lift']:+.2%}} ({{p['rel_ci'][0]:+.2%}} to {{p['rel_ci'][1]:+.2%}}), p {{ab_core.fmt_p(p['p_value'])}}. {{p['result_label'].capitalize()}}.")
print("Spec problems:", errors or "none")
'''))
    cells.append(md("## Next Step\n\nRun `/ab-segment-analysis` if segments were planned in the framing, otherwise "
                    "`/ab-readout`. Check progress with `/ab-status`."))
    cells += nb_common.tracking_cells(SKILL)
    return nb_common.build(cells, spec, SKILL, STAGE)


def _fmt_p(p):
    return "below 0.0001" if p is not None and p < 1e-4 else f"= {p:.3g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?")
    ap.add_argument("--defaults", action="store_true", help="take every setting from the spec")
    ap.add_argument("--spec")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--new-version", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    spec_path = ab_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    if not os.path.exists(spec_path):
        sys.exit(f"No {ab_common.SPEC_NAME} at {root}. Run ab-project-setup first.")
    spec, _ = ab_common.load_or_create(spec_path)
    if not ((spec.get("validation") or {}).get("files") or {}).get("units"):
        sys.exit("The validation has not saved the unit table yet. Run ab-data-validation first.")
    if a.answers:
        answers = ab_common.load(a.answers)
    elif a.defaults:
        answers = {}
    else:
        ap.error("give an answers file or --defaults")
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
    an = ab_common.load(spec_path).get("analysis") or {}
    p = an.get("primary") or {}
    if p:
        print(f"Primary: {p.get('metric')} ({p.get('treatment')}): {p.get('control_mean'):.4%} to {p.get('treatment_mean'):.4%}, "
              f"relative lift {p.get('rel_lift'):+.2%} [{p['rel_ci'][0]:+.2%}, {p['rel_ci'][1]:+.2%}], p {_fmt_p(p.get('p_value'))}. "
              f"{p.get('result_label')}.")
    for g in an.get("guardrails") or []:
        print(f"Guardrail {g['metric']}: " + ("BREACHED" if g.get("breached") else "not breached"))


if __name__ == "__main__":
    main()
