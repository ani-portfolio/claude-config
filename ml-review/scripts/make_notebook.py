#!/usr/bin/env python3
"""Write notebooks/10_review.ipynb and run it: an independent review of the project.

The notebook runs the scripted checks for the chosen stages, adds the findings of the independent reviewer (a fresh
agent that did not build the work), merges them with the earlier reviews (a finding that is no longer found is
marked fixed), shows a note about any open blockers, and writes review_findings.parquet and the review section of
ml_spec.yaml. It changes nothing else in the project.

Usage (from the project root, with the project's Python):
    make_notebook.py --suggest                    # stages done, the last review, and the scripted findings
    make_notebook.py answers.yaml                 # build and run; refuses to replace an edited notebook
    make_notebook.py answers.yaml --overwrite     # replace an edited notebook
    make_notebook.py answers.yaml --new-version   # write 10_review_v2.ipynb instead
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
import nb_common  # noqa: E402
from nb_common import code, lit, md  # noqa: E402

SKILL = "ml-review"
STAGE = "review"
FINDINGS_FILE = "review_findings.parquet"
REVIEWER_KEYS = ("stage", "category", "severity", "title", "evidence", "fix")


def previous_findings(spec, root):
    import pandas as pd
    p = os.path.join(ml_common.project_paths(spec, root)["data_processed"], FINDINGS_FILE)
    return pd.read_parquet(p) if os.path.exists(p) else None


def suggest(spec, root):
    import yaml
    import review_core
    done = review_core.done_stages(spec)
    print("Stages done: " + (", ".join(done) or "none"))
    rv = spec.get("review") or {}
    if rv.get("run_at"):
        o = rv.get("open") or {}
        print(f"Last review: {rv['run_at']} of {', '.join(rv.get('stages') or [])}. Open: {o.get('blocker', 0)} blockers, "
              f"{o.get('warning', 0)} warnings, {o.get('note', 0)} notes. Accepted: {rv.get('accepted', 0)}.")
        last = review_core._ts(rv["run_at"])
        changed = [s for s, r in (spec.get("notebooks") or {}).items() if s in review_core.REVIEW_STAGES
                   and review_core._ts(r.get("run_at")) and last and review_core._ts(r.get("run_at")) > last]
        print("Stages that ran again since then: " + (", ".join(changed) or "none"))
    else:
        print("Last review: none")
    prev = previous_findings(spec, root)
    if prev is not None:
        acc = prev[prev.status == "accepted"]
        if len(acc):
            print("Accepted findings (kept accepted unless you remove them):")
            for r in acc.itertuples():
                print(f"  {r.id}: {r.accepted_reason}")
    findings = review_core.run_checks(spec, root, done)
    print(f"\nScripted findings for every stage done ({len(findings)}):")
    for f in findings:
        print(f"  [{f['severity']}] {f['id']}: {f['title']}")
    out = {"stages": done, "accept": {}, "reviewer": {"summary": "", "files_read": []}, "reviewer_findings": []}
    print("\nAnswers skeleton (fill reviewer_findings from the independent reviewer):\n")
    print(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))


def clean_reviewer(findings):
    out, problems = [], []
    for i, f in enumerate(findings or []):
        f = {k: (str(f.get(k)).strip() if f.get(k) is not None else None) for k in REVIEWER_KEYS}
        if f["title"]:
            f["title"] = f["title"].rstrip(".")
        if f["severity"] not in ("blocker", "warning", "note"):
            problems.append(f"reviewer finding {i + 1}: severity must be blocker, warning, or note")
        if not f["title"] or not f["evidence"]:
            problems.append(f"reviewer finding {i + 1}: needs a title and evidence")
        out.append(f)
    return out, problems


def build(answers, spec, root, scripts_dir, nb_name):
    import review_core
    stages = [s for s in answers.get("stages") or review_core.done_stages(spec) if s in review_core.REVIEW_STAGES]
    prev = previous_findings(spec, root)
    accepted = {}
    if prev is not None:
        accepted = {r.id: r.accepted_reason for r in prev[prev.status == "accepted"].itertuples()}
    accepted.update({str(k): str(v) for k, v in (answers.get("accept") or {}).items()})
    reviewer = answers.get("reviewer") or {}
    rfind, _ = clean_reviewer(answers.get("reviewer_findings"))
    summary = (reviewer.get("summary") or "").strip()

    cells = nb_common.setup_cells(
        STAGE, "Project Review",
        "An independent review of the project. Scripted checks look for leakage, settings that disagree between "
        "stages, results that look too good, honest use of the test set, statistical strength, business fit, and "
        "reproducibility. An independent reviewer, a fresh agent that did not build the work, then read the notebooks "
        "and the code and added what the scripts cannot see. The review changes nothing in the project: each finding "
        "names the skill that should fix it.\n\n"
        "**Severity.** A *blocker* means the results cannot be trusted until it is fixed. A *warning* could change a "
        "decision. A *note* is worth knowing.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the stages to review, the accepted findings "
        "(each with a reason), and the reviewer's findings. Edit them, then choose Run All.",
        scripts_dir, nb_name)
    cells.append(code('''
import pandas as pd
from IPython.display import Markdown, display

import review_core
pd.set_option("display.max_colwidth", 120)
'''))
    cells.append(md("## What Is Reviewed\n\n*`STAGES` lists the stages to review. `ACCEPTED` maps a finding id to the "
                    "reason it is accepted as it is; an accepted finding stays in the table but is not counted as open. "
                    "Remove an entry to open the finding again.*"))
    cells.append(code(
        "# Decision\n"
        f"STAGES = {lit(stages)}\n\n"
        f"ACCEPTED = {lit(accepted)}", decision=True))
    cells.append(code('''
done = review_core.done_stages(spec)
skipped = [s for s in done if s not in STAGES]
print("Stages done: " + ", ".join(done))
print("Reviewed now: " + ", ".join(STAGES) + (f". Not reviewed this time: {', '.join(skipped)}" if skipped else ""))
last = (spec.get("review") or {}).get("run_at")
print("Previous review: " + (last or "none"))
'''))
    cells.append(md("## Scripted Checks\n\nThe same checks give the same result every time, so each finding keeps its "
                    "id across reviews."))
    cells.append(code('''
script_findings = review_core.run_checks(spec, str(ROOT), STAGES)
counts = pd.Series([f["severity"] for f in script_findings], dtype="object").value_counts()
print(f"Scripted findings: {len(script_findings)} ("
      + ", ".join(f"{counts.get(s, 0)} {s}s" for s in ("blocker", "warning", "note")) + ")")
if script_findings:
    display(pd.DataFrame(script_findings)[["severity", "stage", "category", "title", "id"]])
'''))
    cells.append(md("## Independent Reviewer\n\n"
                    + (f"**When:** {reviewer.get('at') or 'this run'}. "
                       + (f"**Files read:** {', '.join('`' + f + '`' for f in reviewer.get('files_read') or [])}.\n\n"
                          if reviewer.get("files_read") else "\n\n")
                       + (summary + "\n\n" if summary else ""))
                    + "*`REVIEWER_FINDINGS` holds what the reviewer found beyond the scripted checks. Each has a stage, "
                      "category, severity, title, evidence, and fix. Edit or remove a finding that is wrong, and say why "
                      "in `ACCEPTED` instead when it is right but acceptable.*"))
    cells.append(code("# Decision\n" f"REVIEWER_FINDINGS = {lit(rfind)}", decision=True))
    cells.append(md("## All Findings\n\nThis review's findings merged with the earlier reviews. A scripted finding "
                    "that is no longer found, in a stage reviewed now, is marked *fixed*; a reviewer finding that was "
                    "not raised again is marked *not repeated*."))
    cells.append(code('''
findings_path = os.path.join(PATHS["data_processed"], "review_findings.parquet")
previous = pd.read_parquet(findings_path) if os.path.exists(findings_path) else None
reviewer_findings = [{**f, "source": "reviewer"} for f in REVIEWER_FINDINGS]
now = pd.Timestamp.now().floor("s").isoformat()
findings = review_core.merge_history(previous, script_findings + reviewer_findings, STAGES, ACCEPTED, now)
unknown = [k for k in ACCEPTED if k not in set(findings["id"])]
if unknown:
    print("Accepted ids that match no finding: " + ", ".join(unknown))
open_ = findings[findings.status == "open"]
table = (open_.groupby(["stage", "severity"]).size().unstack(fill_value=0)
         .reindex(columns=["blocker", "warning", "note"], fill_value=0)) if len(open_) else None
if table is not None:
    order = {s: i for i, s in enumerate(["project"] + review_core.REVIEW_STAGES)}
    display(table.loc[sorted(table.index, key=lambda s: order.get(s, 99))])
lines = []
for sev in ("blocker", "warning", "note"):
    part = open_[open_.severity == sev]
    if len(part):
        lines.append(f"### Open {sev}s ({len(part)})")
        for f in part.itertuples():
            carried = "" if f.last_seen == now else f" *Carried over from the review of {str(f.last_seen)[:10]}; its stage was not reviewed now.*"
            lines.append(f"* **{f.title}** ({f.stage}, {f.category}, {f.source}, `{f.id}`){carried}  \\n"
                         f"  {f.evidence}  \\n  *Fix:* {f.fix or 'see the evidence'}")
acc = findings[findings.status == "accepted"]
if len(acc):
    lines.append(f"### Accepted ({len(acc)})")
    lines += [f"* {f.title} (`{f.id}`): {f.accepted_reason}" for f in acc.itertuples()]
closed = findings[findings.status.isin(["fixed", "not repeated"]) & (findings.last_seen == (previous["last_seen"].max() if previous is not None and len(previous) else None))]
if len(closed):
    lines.append(f"### Fixed since the last review ({len(closed)})")
    lines += [f"* {f.title} (`{f.id}`): {f.status}" for f in closed.itertuples()]
display(Markdown("\\n".join(lines) if lines else "No findings."))
'''))
    cells.append(md("## Blockers\n\nA note on what the open blockers mean for the project."))
    cells.append(code('''
blockers = open_[open_.severity == "blocker"]
n_warn = int((open_.severity == "warning").sum())
if len(blockers):
    stages_hit = list(dict.fromkeys(blockers["stage"]))
    note = [f"**{len(blockers)} open blocker{'s' if len(blockers) > 1 else ''}.** The results of the "
            + ", ".join(review_core.STAGE_LABEL.get(s, s) for s in stages_hit) + " stage" + ("s" if len(stages_hit) > 1 else "") + (" and every later stage" if any(s != "evaluation" for s in stages_hit) else "")
            + " should not be relied on, shared, or used to open the test set until each blocker is fixed, or "
              "accepted in `ACCEPTED` with a reason.", ""]
    for f in blockers.itertuples():
        note.append(f"* {f.title}. Fix: {f.fix or 'see the evidence above'}")
    if "evaluation" not in stages_hit and (spec.get("evaluation") or {}).get("test_opened"):
        note += ["", "The test set was already opened, so a test score after the fix is less clean than the first one."]
else:
    note = ["**No open blockers.**" + (f" {n_warn} warning{'s' if n_warn != 1 else ''} to read before deciding."
                                       if n_warn else "")]
display(Markdown("\\n".join(note)))
'''))
    cells.append(md("## Write the Findings\n\nSaves `review_findings.parquet` (every finding of every review, with its "
                    "status) and replaces the `review` section of `ml_spec.yaml`. Nothing else is changed."))
    cells.append(code(f'''
os.makedirs(PATHS["data_processed"], exist_ok=True)
findings.to_parquet(findings_path, index=False)
open_counts = {{s: int((open_.severity == s).sum()) for s in ("blocker", "warning", "note")}}
review_section = {{
    "stages": STAGES,
    "open": open_counts,
    "accepted": int((findings.status == "accepted").sum()),
    "fixed_since_last": int(len(closed)),
    "open_blockers": [{{"id": f.id, "title": f.title, "fix": f.fix}} for f in blockers.itertuples()],
    "reviewer": {{"findings": len(REVIEWER_FINDINGS), "files_read": {lit(reviewer.get("files_read") or [])}}},
    "files": {{"findings": ml_common.relpath(findings_path, str(ROOT))}},
    "run_at": now,
}}
nb_common.write_stage(SPEC_PATH, "review", review_section, "{SKILL}", replace=True)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "{SKILL}")
print(f"Wrote {{ml_common.relpath(findings_path, str(ROOT))}} and the review section of ml_spec.yaml.")
'''))
    cells += nb_common.tracking_cells(SKILL)
    cells.append(md("## Next Step\n\nFix the blockers first, with the skill each one names, then run `/ml-review` again: "
                    "findings that are gone are marked fixed. Notes that disagree between stages are aligned with "
                    "`/ml-reconcile-notes`. `/ml-status` shows the project and the open blockers."))
    return nb_common.build(cells, spec, SKILL, STAGE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", nargs="?", help="Answers YAML file")
    ap.add_argument("--suggest", action="store_true", help="Print the stages, the last review, and the scripted findings")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    a = ap.parse_args()
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
    answers = ml_common.load(a.answers)
    _, problems = clean_reviewer(answers.get("reviewer_findings"))
    if problems:
        sys.exit("Fix the answers file: " + "; ".join(problems))
    nb_name = ml_common.notebook_name(STAGE)
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    nb = build(answers, spec, root, HERE, nb_name)
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
    import pandas as pd
    rv = ml_common.load(spec_path).get("review") or {}
    f = pd.read_parquet(os.path.join(root, rv["files"]["findings"]))
    o = rv.get("open") or {}
    print(f"Reviewed: {', '.join(rv.get('stages') or [])}")
    print(f"Open: {o.get('blocker', 0)} blockers, {o.get('warning', 0)} warnings, {o.get('note', 0)} notes. "
          f"Accepted: {rv.get('accepted', 0)}. Fixed since the last review: {rv.get('fixed_since_last', 0)}.")
    for sev in ("blocker", "warning", "note"):
        part = f[(f.status == "open") & (f.severity == sev)]
        for r in part.itertuples():
            carried = "" if r.last_seen == rv.get("run_at") else f" (carried over from {str(r.last_seen)[:10]}, stage not reviewed now)"
            print(f"  [{sev}] {r.stage} ({r.source}): {r.title}{carried}. Fix: {r.fix}")
    print("Files: " + rv["files"]["findings"])


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "review", "ml-review")
