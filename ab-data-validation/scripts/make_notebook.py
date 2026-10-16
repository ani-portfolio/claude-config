#!/usr/bin/env python3
"""Write notebooks/ab03_validation.ipynb from the validation answers, and run it once.

The notebook maps the data onto the framing (unit, group, variants, metrics, dates), records which columns were
recorded before and after assignment, gives an overview of the data (column types, missing values, value ranges,
duplicate rows, and each metric pooled over all variants), and checks whether the data can be trusted: the sample
ratio against the
planned split, units in more than one variant, duplicate units, unmapped labels, missing metrics by variant,
balance on columns recorded before assignment, and coverage over time. Running it writes the data and validation
sections of ab_spec.yaml and saves the unit level table data/processed/ab_units.parquet for the analysis.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml [--overwrite | --new-version] [--no-run]

answers.yaml: start from the output of inspect_data.py --yaml, confirmed with the user. Keys: data, ignored_cols,
unknown_timing_cols, planned_shares, override, resolved_questions, reasoning, assumptions, open_questions, decisions.
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

SKILL = "ab-data-validation"
STAGE = "validation"


def _reason(answers, key):
    r = (answers.get("reasoning") or {}).get(key)
    return (r.strip() + "\n\n") if r else ""


def build(answers, spec, scripts_dir, nb_name):
    data = dict(answers.get("data") or {})
    d = ab_common.test_defaults(spec)
    cells = nb_common.setup_cells(
        STAGE, "Data Validation",
        "Can this test's data be trusted? This notebook maps the data onto the test framing and checks it before any "
        "effect is estimated. A failed check blocks the analysis until its cause is found, or until an override with "
        "a reason is recorded below.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the column mapping, the timing of each column, "
        "and the planned split. Edit them, then choose Run All. Metrics are shown pooled over all variants only. No "
        "group is compared on a metric here: that is `ab04_analysis.ipynb`.",
        scripts_dir, nb_name)
    cells.append(code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ab_core
from IPython.display import display
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
'''))
    cells.append(md("## Column Mapping\n\n" + _reason(answers, "mapping") +
                    "*`variant_values` maps each variant in the framing to its label in the group column. `metric_cols` "
                    "maps each metric in the framing to a column. Index columns left over from an export are ignored.*"))
    cells.append(code(f"# Decision\nDATA = {lit({k: data.get(k) for k in ('source', 'unit_col', 'group_col', 'variant_values', 'metric_cols', 'date_col')})}\n"
                      f"IGNORED_COLS = {lit(list(answers.get('ignored_cols') or []))}", decision=True))
    cells.append(md("## When Each Column Was Recorded\n\n" + _reason(answers, "timing") +
                    "*Columns recorded before assignment cannot be changed by the treatment, so they are safe for balance "
                    "checks, CUPED, and segments. Columns recorded after assignment (exposure counts, activity during the "
                    "test) can be changed by the treatment: adjusting for them or splitting by them biases the effect. "
                    "Columns with unknown timing are treated as recorded after assignment until someone confirms otherwise.*"))
    cells.append(code(
        "# Decision\n"
        f"PRE_TREATMENT_COLS = {lit(list(data.get('pre_treatment_cols') or []))}\n"
        f"POST_TREATMENT_COLS = {lit(list(data.get('post_treatment_cols') or []))}\n"
        f"UNKNOWN_TIMING_COLS = {lit(list(answers.get('unknown_timing_cols') or []))}", decision=True))
    planned = answers.get("planned_shares") or d["planned_shares"]
    cells.append(md("## Planned Split\n\n" + _reason(answers, "split") +
                    "*The share of units each variant was meant to get, from the framing. The sample ratio check compares "
                    "the data against it. Use None for a share that is not known; the check is then skipped with a warning. "
                    "Never fill it from the data.*"))
    cells.append(code(f"# Decision\nPLANNED_SHARES = {lit(planned)}", decision=True))

    cells.append(md("## Load the Data"))
    cells.append(code('''
df = nb_common.load_table(ROOT / DATA["source"])
label_to_variant = {str(v): k for k, v in (DATA["variant_values"] or {}).items()}
df["ab_variant"] = df[DATA["group_col"]].astype(str).map(label_to_variant)
U = DATA["unit_col"]
CONTROL = next(v["name"] for v in spec["framing"]["variants"] if v.get("role") == "control")
TREATMENTS = [v["name"] for v in spec["framing"]["variants"] if v.get("role") == "treatment"]
failures, warnings = [], []
print(f"{len(df):,} rows and {df.shape[1] - 1} columns from {DATA['source']}. Control: {CONTROL}. Treatments: {', '.join(TREATMENTS)}.")
'''))

    cells.append(md("## Data Overview\n\nEach column's type, missing values, distinct values, and range, then each "
                    "metric pooled over all variants. A pooled value says nothing about which variant did better, so it "
                    "is safe to see before the checks are settled. With an unequal split it is close to the largest "
                    "variant's value, so it is not a baseline for the minimum effect."))
    cells.append(code('''
cols = [c for c in df.columns if c not in IGNORED_COLS + ["ab_variant"]]
profile = pd.DataFrame({"type": df[cols].dtypes.astype(str), "missing share": df[cols].isna().mean(),
                        "distinct values": df[cols].nunique()})
num = [c for c in cols if df[c].dtype.kind in "biuf"]
profile["min"], profile["max"] = df[num].min(), df[num].max()
display(profile)
print(f"{int(df.duplicated(subset=cols).sum()):,} rows repeat another row exactly.")

overall = []
for name, col in (DATA["metric_cols"] or {}).items():
    if col in df.columns:
        s = pd.to_numeric(df.loc[df["ab_variant"].notna(), col], errors="coerce").astype(float)
        overall.append({"metric": name, "column": col, "units with a value": int(s.notna().sum()),
                        "pooled mean (rate for binary)": s.mean(), "min": s.min(), "max": s.max()})
if overall:
    display(pd.DataFrame(overall))
'''))

    cells.append(md("## Units and Assignment\n\nEvery row should belong to a mapped variant, every unit should appear once, "
                    "and no unit should be in more than one variant. A unit in two variants saw both experiences, so its "
                    "outcome cannot be credited to either."))
    cells.append(code('''
unmapped = df.loc[df["ab_variant"].isna(), DATA["group_col"]].astype(str).value_counts()
per_unit = df.groupby(U)["ab_variant"].nunique()
units = {
    "rows": len(df),
    "distinct_units": int(df[U].nunique()),
    "duplicate_unit_rows": int(df.duplicated(subset=[U]).sum()),
    "units_in_more_than_one_variant": int((per_unit > 1).sum()),
    "rows_with_unmapped_label": int(unmapped.sum()),
    "rows_with_missing_unit": int(df[U].isna().sum()),
}
display(pd.Series(units, name="count").to_frame())
if units["rows_with_unmapped_label"]:
    failures.append(f"{units['rows_with_unmapped_label']:,} rows have a group label that is not mapped to a variant ({', '.join(unmapped.index)}).")
share_crossed = units["units_in_more_than_one_variant"] / max(units["distinct_units"], 1)
if share_crossed > 0.001:
    failures.append(f"{units['units_in_more_than_one_variant']:,} units ({share_crossed:.2%}) appear in more than one variant.")
elif units["units_in_more_than_one_variant"]:
    warnings.append(f"{units['units_in_more_than_one_variant']:,} units appear in more than one variant. They are few, but the analysis should drop them.")
if units["duplicate_unit_rows"] - units["units_in_more_than_one_variant"] > 0:
    warnings.append(f"{units['duplicate_unit_rows']:,} rows repeat a unit. The analysis must aggregate to one row per unit, the unit that was randomised.")
if units["rows_with_missing_unit"]:
    warnings.append(f"{units['rows_with_missing_unit']:,} rows have no unit.")
'''))

    cells.append(md("## Sample Ratio\n\nA chi-square test of the units per variant against the planned split. A p value "
                    "below 0.001 is a sample ratio mismatch: assignment or logging is broken, and the effect estimate "
                    "cannot be trusted until the cause is found. A large test detects even small mismatches, which is "
                    "the point: small imbalances often hide large biases."))
    cells.append(code('''
counts = df.dropna(subset=["ab_variant"]).groupby("ab_variant")[U].nunique().to_dict()
srm = None
if PLANNED_SHARES and all(isinstance(s, (int, float)) for s in PLANNED_SHARES.values()):
    srm = ab_core.srm_test(counts, PLANNED_SHARES)
    display(pd.DataFrame({"units": srm["observed"], "expected": srm["expected"],
                          "observed share": srm["observed_shares"], "planned share": srm["expected_shares"]}))
    print(f"Chi-square {srm['chi2']:.2f} with {srm['df']} degrees of freedom, p {ab_core.fmt_p(srm['p_value'])}.")
    if not srm["passed"]:
        failures.append(f"Sample ratio mismatch: the split differs from the plan (p {ab_core.fmt_p(srm['p_value'])}, the threshold is 0.001).")
    else:
        print("The split matches the plan.")
else:
    observed = pd.Series(counts, name="units")
    display(pd.DataFrame({"units": observed, "observed share": observed / observed.sum()}))
    warnings.append("The planned split is not known, so the sample ratio was not tested. Find out how the test was set up.")
'''))

    cells.append(md("## Metrics\n\nEach metric in the framing should have a column, binary metrics should hold only two "
                    "values, and missing values should be equally common in every variant. Missing outcomes that depend on "
                    "the variant bias the comparison."))
    cells.append(code('''
rows = []
for m in [dict(spec["framing"]["primary_metric"], role="primary")] + \\
         [dict(x, role="secondary") for x in spec["framing"].get("secondary_metrics") or []] + \\
         [dict(x, role="guardrail") for x in spec["framing"].get("guardrail_metrics") or []]:
    col = (DATA["metric_cols"] or {}).get(m["name"])
    if not col or col not in df.columns:
        (failures if m["role"] == "primary" else warnings).append(f"The {m['role']} metric {m['name']} has no column in the data.")
        continue
    miss = df.groupby("ab_variant")[col].apply(lambda s: s.isna().mean())
    values = df[col].dropna().unique()
    ok_type = m.get("type") != "binary" or set(pd.Series(values).astype(float)) <= {0.0, 1.0}
    rows.append({"metric": m["name"], "role": m["role"], "column": col, "type": m.get("type"), "distinct values": len(values),
                 "type ok": ok_type, "largest missing share": miss.max(), "missing gap between variants": miss.max() - miss.min()})
    if not ok_type:
        failures.append(f"{m['name']} is binary in the framing, but {col} holds values other than 0 and 1.")
    if miss.max() - miss.min() > 0.02:
        warnings.append(f"{col} is missing more often in some variants than others (gap {miss.max() - miss.min():.1%}).")
display(pd.DataFrame(rows))
'''))

    cells.append(md("## Balance on Columns Recorded Before Assignment\n\nWith random assignment, columns recorded before "
                    "assignment should look the same in every variant. A standardised mean difference above 0.1, or a "
                    "chi-square p value below 0.01, is flagged."))
    cells.append(code('''
balance = []
if PRE_TREATMENT_COLS:
    for t in TREATMENTS:
        for r in ab_core.balance(df, "ab_variant", PRE_TREATMENT_COLS, CONTROL, t):
            balance.append({"treatment": t, **r})
    display(pd.DataFrame(balance))
    flagged = [f"{b['column']} ({b['treatment']})" for b in balance if b["flag"]]
    if flagged:
        warnings.append("Imbalance before assignment on: " + ", ".join(flagged) + ". Check how units were assigned.")
else:
    print("No columns recorded before assignment, so balance cannot be checked. Random assignment is the only protection, "
          "which is why the sample ratio check matters.")
'''))

    cells.append(md("## Coverage Over Time\n\nUnits per day by variant, when there is a date column. The treatment share "
                    "should be flat; a jump points to a change in assignment or logging partway through."))
    cells.append(code('''
coverage = None
if DATA.get("date_col") and DATA["date_col"] in df.columns:
    day = pd.to_datetime(df[DATA["date_col"]], errors="coerce").dt.floor("D")
    daily = df.assign(day=day).dropna(subset=["day", "ab_variant"]).groupby(["day", "ab_variant"])[U].nunique().unstack(fill_value=0)
    share = daily.div(daily.sum(axis=1), axis=0)
    ax = share.plot(figsize=(9, 3), title="Share of units per variant by day")
    ax.set_ylabel("share"); plt.show()
    coverage = {"first_day": str(daily.index.min().date()), "last_day": str(daily.index.max().date()), "days": int(len(daily)),
                "min_daily_units": int(daily.sum(axis=1).min())}
    print(coverage)
    if len(daily) < 7:
        warnings.append(f"The test ran for {len(daily)} days, less than a full week, so weekday effects may bias it.")
else:
    print("No date column, so coverage over time and novelty effects cannot be checked.")
'''))

    cells.append(md("## Columns Recorded After Assignment\n\nThese are kept out of balance checks, CUPED, and segments. "
                    "They are summarised here only to describe what each variant experienced."))
    cells.append(code('''
post = [c for c in POST_TREATMENT_COLS + UNKNOWN_TIMING_COLS if c in df.columns]
if post:
    num = [c for c in post if df[c].dtype.kind in "biuf"]
    if num:
        display(df.groupby("ab_variant")[num].mean().T)
    print("Kept out of adjustments and segments: " + ", ".join(post) + ".")
else:
    print("None.")
'''))

    cells.append(md("## Override\n\nOnly if a check failed and the user decides to analyse anyway: a dict with `reason` "
                    "and `by`. The reason is recorded in the spec and shown in the readout."))
    cells.append(code(f"# Decision\nOVERRIDE = {lit(answers.get('override'))}", decision=True))
    cells.append(md("## Assumptions, Open Questions, and Decisions\n\n`resolved_questions` closes open questions from "
                    "earlier steps that this validation settles, as `{\"id\": \"Q1\", \"answer\": \"...\"}`."))
    cells.append(code(
        "# Decision\n"
        f"assumptions = {lit(list(answers.get('assumptions') or []))}\n\n"
        f"open_questions = {lit(list(answers.get('open_questions') or []))}\n\n"
        f"decisions = {lit(list(answers.get('decisions') or []))}\n\n"
        f"resolved_questions = {lit(list(answers.get('resolved_questions') or []))}", decision=True))

    cells.append(md("## Result"))
    cells.append(code('''
result = "failed" if failures else "warning" if warnings else "passed"
print(f"Result: {result}.")
for f in failures:
    print("  Failure:", f)
for w in warnings:
    print("  Warning:", w)
if result == "failed":
    print("The analysis is blocked." + (" An override is recorded: " + OVERRIDE["reason"] if OVERRIDE else
          " Find the cause, or record an override with a reason above."))
'''))
    cells.append(md("## Save and Write to the Spec\n\nSaves one row per unit with the variant, the metrics, and the "
                    "columns recorded before assignment to `data/processed/ab_units.parquet`, then writes the `data` and "
                    "`validation` sections of `ab_spec.yaml`."))
    cells.append(code(f'''
keep = [U, "ab_variant"] + [c for c in (DATA["metric_cols"] or {{}}).values() if c in df.columns] + \\
       [c for c in PRE_TREATMENT_COLS + POST_TREATMENT_COLS + UNKNOWN_TIMING_COLS if c in df.columns] + \\
       ([DATA["date_col"]] if DATA.get("date_col") in df.columns else [])
units_path = os.path.join(PATHS["data_processed"], "ab_units.parquet")
os.makedirs(PATHS["data_processed"], exist_ok=True)
df.dropna(subset=["ab_variant"])[list(dict.fromkeys(keep))].to_parquet(units_path, index=False)
print("Saved", units_path)

timing_q = [{{"question": f"Was {{c}} recorded before or after assignment? It is treated as after until confirmed.", "stage": "validation"}}
            for c in UNKNOWN_TIMING_COLS]
nb_common.write_stage(SPEC_PATH, "data", {{**DATA, "pre_treatment_cols": PRE_TREATMENT_COLS,
                                          "post_treatment_cols": POST_TREATMENT_COLS + UNKNOWN_TIMING_COLS}}, "{SKILL}")
validation = {{"result": result, "failures": failures, "warnings": warnings, "override": OVERRIDE, "srm": srm,
              "units": units, "balance": balance, "coverage": coverage,
              "files": {{"units": ab_common.relpath(units_path, str(ROOT))}},
              "run_at": pd.Timestamp.now().floor("s").isoformat()}}
errors, _, _ = nb_common.write_stage(SPEC_PATH, "validation", validation, "{SKILL}", assumptions,
                                     open_questions + timing_q, decisions, replace=True)
missing = nb_common.resolve_questions(SPEC_PATH, resolved_questions, "{SKILL}")
if missing:
    print("Open questions not found:", ", ".join(map(str, missing)))
print("Spec problems:", errors or "none")
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
'''))
    cells.append(md("## Next Step\n\nIf the result is passed or warning (or failed with an override), run `/ab-analysis`. "
                    "Otherwise find the cause of each failure first. Check progress with `/ab-status`."))
    cells += nb_common.tracking_cells(SKILL)
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", help="Answers YAML file")
    ap.add_argument("--spec")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--new-version", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    spec_path = ab_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    if not os.path.exists(spec_path):
        sys.exit(f"No {ab_common.SPEC_NAME} at {root}. Run ab-project-setup and ab-test-framing first.")
    spec, _ = ab_common.load_or_create(spec_path)
    if not ab_common.control_variant(spec):
        sys.exit("The framing has no control variant yet. Run ab-test-framing first.")
    answers = ab_common.load(a.answers)
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
    val = ab_common.load(spec_path).get("validation") or {}
    print(f"Result: {val.get('result')}")
    for f in val.get("failures") or []:
        print(f"  Failure: {f}")
    for w in val.get("warnings") or []:
        print(f"  Warning: {w}")
    srm = val.get("srm") or {}
    if srm:
        print(f"Sample ratio p value: {srm.get('p_value'):.3g} ({'passed' if srm.get('passed') else 'MISMATCH'})")


if __name__ == "__main__":
    main()
