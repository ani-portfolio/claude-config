#!/usr/bin/env python3
"""Facts about an A/B test data file, and a recommended answers YAML for make_notebook.py.

Prints the size, each column's type and number of distinct values, guesses for the unit, group, metric, and date
columns, the distinct labels of the group column (so they can be matched to the variants in the framing), and a
first guess at whether each column was recorded before or after assignment. It never prints rows, and it never
compares the metrics between groups: that is the analysis.

Usage (from the project root, with the project's Python):
    inspect_data.py                       # the only data file in the raw data folder
    inspect_data.py data/raw/test.csv     # a named file
    inspect_data.py data/raw/test.csv --yaml answers.yaml   # also write the recommended answers
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common  # noqa: E402
import nb_common  # noqa: E402

DATA_EXT = (".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl", ".feather", ".csv.gz")
GROUP_WORDS = r"group|variant|arm|bucket|treatment|cohort|test|experiment|condition|version"
UNIT_WORDS = r"user|customer|visitor|client|account|member|session|cookie|device|store|id"
DATE_WORDS = r"date|time|day|ts|timestamp|assigned|exposure|created"
# Words that suggest a column was measured during the test, so the treatment may have changed it.
POST_WORDS = r"total|most|count|num|number|clicks?|views?|impressions?|exposures?|ads?|opens?|visits?|sessions?|" \
             r"spend|revenue|orders?|purchases?|after|post|during|seen|shown|duration|time_on"
PRE_WORDS = r"age|gender|country|region|city|signup|tenure|prior|before|pre_|pre |baseline|segment|device|platform|" \
            r"channel|plan|tier|language|registered|cohort"


def find_file(spec, root):
    raw = ab_common.project_paths(spec, root)["data_raw"]
    found = []
    for base in (raw, root):
        try:
            found += [os.path.join(base, f) for f in sorted(os.listdir(base)) if f.lower().endswith(DATA_EXT)]
        except OSError:
            pass
        if found:
            break
    return found


def is_index_col(name, s, n):
    return bool(re.fullmatch(r"unnamed: ?\d+|index|level_0", str(name).lower())) and s.nunique() == n


def guess(df, spec):
    n = len(df)
    cols = list(df.columns)
    nun = {c: int(df[c].nunique(dropna=True)) for c in cols}
    low = {c: str(c).lower() for c in cols}
    index_cols = [c for c in cols if is_index_col(c, df[c], n)]
    usable = [c for c in cols if c not in index_cols]

    unit = [c for c in usable if nun[c] == n and re.search(UNIT_WORDS, low[c])]
    unit = unit or [c for c in usable if nun[c] == n and df[c].dtype.kind in "iuO"]
    group = [c for c in usable if 2 <= nun[c] <= 10 and re.search(GROUP_WORDS, low[c])]
    group = group or [c for c in usable if 2 <= nun[c] <= 5 and df[c].dtype.kind not in "bf"]
    dates = [c for c in usable if df[c].dtype.kind == "M" or (re.search(DATE_WORDS, low[c]) and df[c].dtype.kind == "O"
                                                              and _parses_as_date(df[c]))]

    defaults = ab_common.test_defaults(spec)
    metric_cols = {}
    for m in defaults["metrics"]:
        name = m["name"].lower()
        words = [w for w in re.split(r"\W+", name) if len(w) > 2]
        hits = [c for c in usable if any(w[:6] in low[c] for w in words)]
        if m.get("type") == "binary":
            hits = [c for c in hits if nun[c] <= 2] or [c for c in usable if nun[c] == 2 and df[c].dtype.kind == "b"]
        if hits:
            metric_cols[m["name"]] = hits[0]

    taken = set(index_cols) | set(unit[:1]) | set(group[:1]) | set(metric_cols.values()) | set(dates[:1])
    pre, post, unknown = [], [], []
    for c in usable:
        if c in taken:
            continue
        if re.search(POST_WORDS, low[c]):
            post.append(c)
        elif re.search(PRE_WORDS, low[c]):
            pre.append(c)
        else:
            unknown.append(c)
    return {"index_cols": index_cols, "unit": unit, "group": group, "dates": dates, "metric_cols": metric_cols,
            "pre": pre, "post": post, "unknown": unknown, "nunique": nun}


def _parses_as_date(s):
    import warnings
    import pandas as pd
    try:
        sample = s.dropna().astype(str).head(200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return len(sample) > 0 and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9
    except Exception:
        return False


def match_variants(labels, spec):
    """{variant name: label in the data}, matching names case insensitively and by first letters."""
    out = {}
    for v in ab_common.test_defaults(spec)["variants"]:
        name = str(v.get("name", "")).lower()
        hit = next((l for l in labels if str(l).lower() == name), None) or \
            next((l for l in labels if name and (str(l).lower().startswith(name[:3]) or name.startswith(str(l).lower()[:3]))), None)
        if hit is not None:
            out[v["name"]] = hit
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--spec")
    ap.add_argument("--yaml", help="write the recommended answers to this file")
    a = ap.parse_args()
    spec_path = ab_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ab_common.load_or_create(spec_path)
    path = a.file
    if not path:
        files = find_file(spec, root)
        if len(files) != 1:
            sys.exit("Name the data file. Found: " + (", ".join(ab_common.relpath(f, root) for f in files) or "none"))
        path = files[0]
    df = nb_common.load_table(path)
    g = guess(df, spec)
    rel = ab_common.relpath(os.path.abspath(path), root)
    print(f"File: {rel}")
    print(f"Rows: {len(df):,}. Columns: {df.shape[1]}.")
    print("Columns (type, distinct values):")
    for c in df.columns:
        print(f"  {str(c)[:40]:40} {str(df[c].dtype):10} {g['nunique'][c]:>10,}")
    print(f"Index columns left over from an export (ignored): {', '.join(map(str, g['index_cols'])) or 'none'}")
    print(f"Unit column candidates: {', '.join(map(str, g['unit'])) or 'none found'}")
    print(f"Group column candidates: {', '.join(map(str, g['group'])) or 'none found'}")
    labels = []
    if g["group"]:
        labels = sorted(df[g["group"][0]].dropna().astype(str).unique().tolist())
        print(f"Labels in {g['group'][0]}: {', '.join(labels)}")
    print(f"Date column candidates: {', '.join(map(str, g['dates'])) or 'none found'}")
    print("Metric columns matched to the framing: " + (", ".join(f"{k} = {v}" for k, v in g["metric_cols"].items()) or "none"))
    missing = [m["name"] for m in ab_common.test_defaults(spec)["metrics"] if m["name"] not in g["metric_cols"]]
    if missing:
        print("Metrics in the framing with no column found: " + ", ".join(missing))
    print("First guess at timing (confirm with the user):")
    print(f"  probably recorded after assignment: {', '.join(map(str, g['post'])) or 'none'}")
    print(f"  probably recorded before assignment: {', '.join(map(str, g['pre'])) or 'none'}")
    print(f"  unknown: {', '.join(map(str, g['unknown'])) or 'none'}")

    variant_values = match_variants(labels, spec)
    answers = {
        "data": {"source": rel, "unit_col": (g["unit"] or [None])[0], "group_col": (g["group"] or [None])[0],
                 "variant_values": variant_values, "metric_cols": g["metric_cols"], "date_col": (g["dates"] or [None])[0],
                 "pre_treatment_cols": g["pre"], "post_treatment_cols": g["post"]},
        "unknown_timing_cols": g["unknown"],
        "ignored_cols": g["index_cols"],
        "planned_shares": ab_common.test_defaults(spec)["planned_shares"],
        "override": None,
        "reasoning": {}, "assumptions": [], "open_questions": [], "decisions": [],
    }
    if a.yaml:
        with open(a.yaml, "w", encoding="utf-8") as f:
            f.write(ab_common.dump_text(answers))
        print(f"Recommended answers written to {a.yaml}. Check every guess with the user before building the notebook.")
    else:
        print("\nRecommended answers:\n" + ab_common.dump_text(answers))


if __name__ == "__main__":
    main()
