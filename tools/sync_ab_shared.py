#!/usr/bin/env python3
"""Check or update the shared files that every A/B testing skill keeps a copy of.

The canonical copies live in _ab_shared/, except nb_common.py, which the A/B skills share with the ML project
skills and which lives in _ml_shared/. Each skill listed in _ab_shared/skills.txt keeps:
    scripts/ab_common.py        copy of _ab_shared/ab_common.py
    scripts/nb_common.py        copy of _ml_shared/nb_common.py (it uses ab_common when ab_common sits next to it)
    scripts/tracking.py         copy of _ml_shared/tracking.py (MLflow tracking; the same rule)
    reference/spec_fields.md    copy of _ab_shared/spec_fields.md
and the skills that compute statistics (see STATS_SKILLS below) also keep:
    scripts/ab_core.py          copy of _ab_shared/ab_core.py

Run from the claude-skills folder:
    python3 tools/sync_ab_shared.py          # report copies that differ or are missing (exit 1 if any)
    python3 tools/sync_ab_shared.py --fix    # copy the canonical files into every listed skill

Skills listed in skills.txt that do not exist yet are reported and skipped, so the list can name planned skills.
"""

import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(ROOT, "_ab_shared")
STATS_SKILLS = ["ab-test-design", "ab-data-validation", "ab-analysis", "ab-segment-analysis"]
FILES = {
    # canonical file (relative to ROOT): (path inside the skill, skills that keep a copy, or None for every skill)
    "_ab_shared/ab_common.py": ("scripts/ab_common.py", None),
    "_ml_shared/nb_common.py": ("scripts/nb_common.py", None),
    "_ml_shared/tracking.py": ("scripts/tracking.py", None),
    "_ab_shared/spec_fields.md": ("reference/spec_fields.md", None),
    "_ab_shared/ab_core.py": ("scripts/ab_core.py", STATS_SKILLS),
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
            print(f"{skill}: not built yet, skipped")
            continue
        for src_rel, (rel, only) in FILES.items():
            if only is not None and skill not in only:
                continue
            src, dst = os.path.join(ROOT, src_rel), os.path.join(sdir, rel)
            if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
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
        print("Run with --fix to copy the canonical files.")
    sys.exit(1 if bad and not a.fix else 0)


if __name__ == "__main__":
    main()
