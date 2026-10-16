#!/usr/bin/env python3
"""Check or update the shared files that every ML project skill keeps a copy of.

The canonical copies live in _ml_shared/. Each skill listed in _ml_shared/skills.txt keeps:
    scripts/ml_common.py        copy of _ml_shared/ml_common.py
    scripts/nb_common.py        copy of _ml_shared/nb_common.py (building and running stage notebooks)
    reference/spec_fields.md    copy of _ml_shared/spec_fields.md
and the skills that profile data (see EDA_SKILLS below) also keep:
    scripts/eda_core.py         copy of _ml_shared/eda_core.py, the EDA engine
and the split skill and the modelling skills after it (see SPLIT_SKILLS below) also keep:
    scripts/split_core.py       copy of _ml_shared/split_core.py, the split engine and cv_splits
and the stages that fit models (see MODEL_SKILLS below) also keep:
    scripts/model_core.py       copy of _ml_shared/model_core.py, metrics, model definitions, cross validation

Run from the claude-skills folder:
    python3 tools/sync_ml_shared.py          # report copies that differ or are missing (exit 1 if any)
    python3 tools/sync_ml_shared.py --fix    # copy the canonical files into every listed skill

Edit only the files in _ml_shared/, then run with --fix.
"""

import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(ROOT, "_ml_shared")
EDA_SKILLS = ["ml-eda-report", "ml-data-discovery"]
# The split skill and every stage after it read the split back with split_core.cv_splits.
SPLIT_SKILLS = ["ml-train-test-split", "ml-feature-engineering", "ml-baseline-model", "ml-model-training",
                "ml-offline-evaluation", "ml-review"]
# The stages that fit and score models share the metrics and model definitions.
MODEL_SKILLS = ["ml-baseline-model", "ml-model-training", "ml-offline-evaluation", "ml-review"]
FILES = {
    # canonical file: (path inside the skill, skills that keep a copy, or None for every skill in skills.txt)
    "ml_common.py": ("scripts/ml_common.py", None),
    "nb_common.py": ("scripts/nb_common.py", None),
    "spec_fields.md": ("reference/spec_fields.md", None),
    "eda_core.py": ("scripts/eda_core.py", EDA_SKILLS),
    "split_core.py": ("scripts/split_core.py", SPLIT_SKILLS),
    "model_core.py": ("scripts/model_core.py", MODEL_SKILLS),
    "tracking.py": ("scripts/tracking.py", None),
}


def skills():
    with open(os.path.join(SHARED, "skills.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="copy the canonical files into every listed skill")
    a = ap.parse_args()
    bad = 0
    for skill in skills():
        sdir = os.path.join(ROOT, skill)
        if not os.path.isfile(os.path.join(sdir, "SKILL.md")):
            print(f"{skill}: no SKILL.md, skipped (check _ml_shared/skills.txt)")
            bad += 1
            continue
        for src_name, (rel, only) in FILES.items():
            if only is not None and skill not in only:
                continue
            src, dst = os.path.join(SHARED, src_name), os.path.join(sdir, rel)
            same = os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)
            if same:
                print(f"{skill}/{rel}: up to date")
                continue
            state = "differs" if os.path.exists(dst) else "missing"
            if a.fix:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"{skill}/{rel}: {state}, updated")
            else:
                print(f"{skill}/{rel}: {state}")
                bad += 1
    if bad and not a.fix:
        print("Run with --fix to copy the canonical files from _ml_shared/.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
