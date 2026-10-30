#!/usr/bin/env python3
"""Write notebooks/ab02_design.ipynb from the design answers, and run it once.

For a planned or running test, the notebook computes the sample size and the run time for the primary metric from
the baseline, the minimum detectable effect, the significance level, the power, and the planned split. For a finished
test, it also reports the power the test had to detect that effect and the smallest effect it could detect, from the
unit counts the validation saved. Running it writes the design section of ab_spec.yaml.

Usage (from the project root, with the project's Python):
    make_notebook.py answers.yaml [--overwrite | --new-version] [--no-run]
    make_notebook.py --skip --reason "The test has already run and no power check is wanted."

answers.yaml keys: status (draft or confirmed), alpha, power, sides, baseline, baseline_sd, mde_abs, mde_rel,
units_per_day, looks (fixed or sequential), variance_reduction, adjust_for_treatments, reasoning, assumptions,
open_questions, decisions.
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

SKILL = "ab-test-design"
STAGE = "design"


def build(answers, spec, scripts_dir, nb_name):
    a = answers
    reasoning = a.get("reasoning") or {}
    r = lambda k: (reasoning[k].strip() + "\n\n") if reasoning.get(k) else ""
    cells = nb_common.setup_cells(
        STAGE, "Test Design",
        "How many units the test needs to detect the smallest effect worth acting on, and how long that takes. For a "
        "test that has already run, how much power it had. The primary metric and the planned split come from "
        "`ab01_test_framing.ipynb`.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the settings. Edit them, then choose Run All. "
        "The analysis uses the same significance level, sidedness, and minimum effect, so set them here before the "
        "results are seen.",
        scripts_dir, nb_name)
    cells.append(code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ab_core
from IPython.display import display
d = ab_common.test_defaults(spec)
pm = spec["framing"]["primary_metric"]
CONTROL, TREATMENTS = d["control"], d["treatments"]
print(f"Primary metric: {pm['name']} ({pm.get('type')}, good direction {pm.get('direction')}). "
      f"Control: {CONTROL}. Treatments: {', '.join(TREATMENTS)}. Test state: {spec['framing'].get('test_state')}.")
'''))
    cells.append(md("## Error Rates\n\n" + r("error_rates") +
                    "*`ALPHA` is the chance of a false positive, `POWER` the chance of detecting a real effect of the "
                    "minimum size. Two sided tests are the default; one sided only when no effect and harm lead to the "
                    "same decision. With several treatments, `ADJUST_FOR_TREATMENTS` divides alpha between them "
                    "(Bonferroni), which matches the Holm correction in the analysis or is slightly conservative.*"))
    cells.append(code(
        "# Decision\n"
        f"ALPHA = {a.get('alpha', 0.05)!r}\n"
        f"POWER = {a.get('power', 0.8)!r}\n"
        f"SIDES = {a.get('sides', 'two')!r}\n"
        f"ADJUST_FOR_TREATMENTS = {a.get('adjust_for_treatments', True)!r}", decision=True))
    cells.append(md("## Baseline and Minimum Detectable Effect\n\n" + r("effect") +
                    "*`BASELINE` is the control rate (binary) or mean (continuous) expected without the change, ideally "
                    "from data before the test. `BASELINE_SD` is the standard deviation for continuous metrics. `MDE_ABS` "
                    "is the smallest absolute change worth acting on, in the units of the metric (0.005 is half a "
                    "percentage point). Give `MDE_REL` instead to state it relative to the baseline (0.1 is 10%). For a "
                    "finished test with no baseline, the control mean from the validated data is used and noted.*"))
    cells.append(code(
        "# Decision\n"
        f"BASELINE = {a.get('baseline')!r}\n"
        f"BASELINE_SD = {a.get('baseline_sd')!r}\n"
        f"MDE_ABS = {a.get('mde_abs')!r}\n"
        f"MDE_REL = {a.get('mde_rel')!r}", decision=True))
    cells.append(md("## Traffic, Looks, and Variance Reduction\n\n" + r("traffic") +
                    "*`UNITS_PER_DAY` is the number of eligible units a day across all variants, for the run time. "
                    "`LOOKS` is fixed (one analysis at the end) or sequential (looks while running, which needs a "
                    "sequential method; a fixed test that is checked repeatedly has a much higher false positive rate). "
                    "`VARIANCE_REDUCTION` is none, cuped with a column recorded before assignment, or stratification.*"))
    cells.append(code(
        "# Decision\n"
        f"UNITS_PER_DAY = {a.get('units_per_day')!r}\n"
        f"LOOKS = {a.get('looks', 'fixed')!r}\n"
        f"VARIANCE_REDUCTION = {a.get('variance_reduction', 'none')!r}", decision=True))

    cells.append(md("## Sample Size\n\nFor the primary metric, with the planned split between the control and each "
                    "treatment."))
    cells.append(code('''
notes = []
shares = d["planned_shares"]
units_path = ((spec.get("validation") or {}).get("files") or {}).get("units")
units = pd.read_parquet(ROOT / units_path) if units_path and (ROOT / units_path).exists() else None
col = d["metric_cols"].get(pm["name"])
if BASELINE is None and units is not None and col in units.columns:
    BASELINE = float(units.loc[units.ab_variant == CONTROL, col].astype(float).mean())
    if pm.get("type") != "binary" and BASELINE_SD is None:
        BASELINE_SD = float(units.loc[units.ab_variant == CONTROL, col].astype(float).std(ddof=1))
    notes.append("The baseline is the control mean from the test data, because no baseline from before the test was given.")
if MDE_ABS is None and MDE_REL is not None and BASELINE:
    MDE_ABS = MDE_REL * BASELINE
if MDE_REL is None and MDE_ABS is not None and BASELINE:
    MDE_REL = MDE_ABS / BASELINE
alpha_each = ALPHA / max(len(TREATMENTS), 1) if ADJUST_FOR_TREATMENTS else ALPHA
sample, rows = None, []
if BASELINE is not None and MDE_ABS:
    for t in TREATMENTS:
        ratio = (shares.get(t) or 1) / (shares.get(CONTROL) or 1)
        if pm.get("type") == "binary":
            ss = ab_core.sample_size_proportions(BASELINE, abs(MDE_ABS), alpha_each, POWER, SIDES, ratio)
        else:
            assert BASELINE_SD, "BASELINE_SD is needed for a continuous or count metric."
            ss = ab_core.sample_size_means(BASELINE_SD, abs(MDE_ABS), alpha_each, POWER, SIDES, ratio)
        rows.append({"treatment": t, **ss})
    n_control = max(x["n_control"] for x in rows)
    n_treat = {x["treatment"]: x["n_treatment"] for x in rows}
    sample = {"n_control": n_control, "n_treatment": n_treat if len(n_treat) > 1 else next(iter(n_treat.values())),
              "total": n_control + sum(n_treat.values())}
    display(pd.DataFrame(rows))
    print(f"Needed in total: {sample['total']:,} units, at alpha {alpha_each:g} per comparison and power {POWER:.0%}.")
else:
    print("Give a baseline and a minimum detectable effect to compute the sample size.")
duration = ab_core.duration_days(sample["total"], UNITS_PER_DAY) if sample and UNITS_PER_DAY else None
if duration:
    print(f"At {UNITS_PER_DAY:,} units a day, that takes {duration} days (rounded up to whole weeks).")
'''))
    cells.append(md("## Power the Test Had\n\nFor a test that has run, from the units per variant in the validated data: "
                    "the power to detect the minimum effect, and the smallest effect detectable with the chosen power. "
                    "This uses only the unit counts and the baseline, not the treatment's result."))
    cells.append(code('''
achieved = []
if units is not None:
    n = units.groupby("ab_variant").size()
    for t in TREATMENTS:
        nc, nt = int(n.get(CONTROL, 0)), int(n.get(t, 0))
        if not (nc and nt and BASELINE is not None):
            continue
        if pm.get("type") == "binary":
            pw = ab_core.power_proportions(BASELINE, abs(MDE_ABS), nc, nt, alpha_each, SIDES) if MDE_ABS else None
            mde = ab_core.mde_proportions(BASELINE, nc, nt, alpha_each, POWER, SIDES)
        else:
            pw = ab_core.power_means(BASELINE_SD, abs(MDE_ABS), nc, nt, alpha_each, SIDES) if MDE_ABS else None
            mde = ab_core.mde_means(BASELINE_SD, nc, nt, alpha_each, POWER, SIDES)
        achieved.append({"treatment": t, "n_control": nc, "n_treatment": nt, "power_at_mde": pw,
                         "detectable_effect_abs": mde, "detectable_effect_rel": mde / BASELINE if mde and BASELINE else None})
    if achieved:
        display(pd.DataFrame(achieved))
        low = [x["treatment"] for x in achieved if x["power_at_mde"] is not None and x["power_at_mde"] < POWER]
        if low:
            notes.append("The test had less power than planned for: " + ", ".join(low) + ". A result that is not significant there is weak evidence of no effect.")
else:
    print("No validated data yet. Run ab-data-validation to see the power the test had.")
'''))
    cells.append(md("## Power Curve\n\nPower against the true effect, for the planned sample (or the actual one for a "
                    "finished test). The dashed line is the chosen power."))
    cells.append(code('''
if BASELINE is not None and (sample or achieved):
    src = achieved[0] if achieved else {"n_control": sample["n_control"], "n_treatment": rows[0]["n_treatment"]}
    top = (abs(MDE_ABS) if MDE_ABS else (src.get("detectable_effect_abs") or BASELINE * 0.1)) * 2.5
    effects = np.linspace(top / 50, top, 50)
    if pm.get("type") == "binary":
        effects = effects[BASELINE + effects < 1]
        pw = [ab_core.power_proportions(BASELINE, e, src["n_control"], src["n_treatment"], alpha_each, SIDES) for e in effects]
    else:
        pw = [ab_core.power_means(BASELINE_SD, e, src["n_control"], src["n_treatment"], alpha_each, SIDES) for e in effects]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(effects / BASELINE if BASELINE else effects, pw)
    ax.axhline(POWER, ls="--", color="grey")
    if MDE_REL:
        ax.axvline(abs(MDE_REL), ls=":", color="grey")
    ax.set_xlabel("true relative effect" if BASELINE else "true absolute effect"); ax.set_ylabel("power")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); plt.tight_layout(); plt.show()
'''))
    if True:
        cells.append(md("## Assumptions, Open Questions, and Decisions"))
        cells.append(code(
            "# Decision\n"
            f"assumptions = {lit(list(a.get('assumptions') or []))}\n\n"
            f"open_questions = {lit(list(a.get('open_questions') or []))}\n\n"
            f"decisions = {lit(list(a.get('decisions') or []))}", decision=True))
    cells.append(md("## Status\n\n`confirmed` once the decision owner agrees with the minimum effect and the run time, "
                    "`draft` until then."))
    cells.append(code(f"# Decision\nSTATUS = {a.get('status') or 'draft'!r}", decision=True))
    cells.append(md("## Write to the Spec"))
    cells.append(code(f'''
assert STATUS in ("draft", "confirmed"), "STATUS must be draft or confirmed"
assert SIDES in ("two", "one") and LOOKS in ("fixed", "sequential")
for n in notes:
    print("Note:", n)
design = {{"status": STATUS, "alpha": ALPHA, "power": POWER, "sides": SIDES, "alpha_per_comparison": alpha_each,
          "baseline": BASELINE, "baseline_sd": BASELINE_SD, "mde_abs": MDE_ABS, "mde_rel": MDE_REL,
          "sample_size": sample, "units_per_day": UNITS_PER_DAY, "duration_days": duration, "looks": LOOKS,
          "variance_reduction": VARIANCE_REDUCTION, "achieved": achieved,
          "achieved_power": achieved[0]["power_at_mde"] if achieved else None,
          "achieved_mde": achieved[0]["detectable_effect_abs"] if achieved else None, "notes": notes}}
result = nb_common.write_stage(SPEC_PATH, "design", design, "{SKILL}", assumptions, open_questions, decisions, replace=True)
nb_common.show_result(*result, STATUS)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
if LOOKS == "sequential":
    print("Looks while running need a sequential method. A fixed horizon analysis of a test that was checked repeatedly overstates significance.")
'''))
    cells.append(md("## Next Step\n\nFor a planned test, launch it and collect the sample before analysing. For a "
                    "finished test, run `/ab-data-validation` if it has not run yet, then `/ab-analysis`. Check progress "
                    "with `/ab-status`."))
    cells += nb_common.tracking_cells(SKILL)
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?")
    ap.add_argument("--skip", action="store_true", help="mark the design as skipped, without a notebook")
    ap.add_argument("--reason", help="why the design is skipped")
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
    if a.skip:
        if not a.reason:
            ap.error("--skip needs --reason")
        ab_common.update_section(spec_path, "design", {"status": "skipped", "reason": a.reason}, SKILL,
                                 action="skipped the design: " + a.reason)
        print(f"Design marked as skipped: {a.reason}")
        return
    if not ab_common.control_variant(spec) or not ((spec.get("framing") or {}).get("primary_metric") or {}).get("name"):
        sys.exit("The framing needs a control variant and a primary metric. Run ab-test-framing first.")
    if not a.answers:
        ap.error("give an answers file, or --skip with --reason")
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
    de = ab_common.load(spec_path).get("design") or {}
    ss = de.get("sample_size") or {}
    print(f"Design: status {de.get('status')}, baseline {de.get('baseline')}, minimum effect {de.get('mde_abs')} "
          f"({de.get('mde_rel')} relative), sample size {ss.get('total')}, duration {de.get('duration_days')} days.")
    if de.get("achieved_power") is not None or de.get("achieved_mde") is not None:
        print(f"Power the test had at the minimum effect: {de.get('achieved_power')}. Smallest detectable effect: "
              f"{de.get('achieved_mde')}.")
    for n in de.get("notes") or []:
        print("Note:", n)


if __name__ == "__main__":
    main()
