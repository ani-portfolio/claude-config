#!/usr/bin/env python3
"""Data discovery for the ML project skills: find out what the data is before deciding what to predict.

Builds notebooks/02_data_discovery.ipynb and runs it once. The notebook checks each source's size, grain,
possible entity IDs, coverage over time, and missing values, how the sources join, candidate targets, and
columns that may be recorded after the outcome, each in short readable pandas code. It drafts a data dictionary
from the column names and the data, flags the columns whose meaning or timing it cannot work out, and takes
answers for them in an editable decision cell. Running it writes docs/data_dictionary.yaml and the discovery
section of ml_spec.yaml. Raw rows appear only once, in the notebook's First Rows cell.

Commands (run from the project root with the project's Python):
    discover.py data/raw/customers.csv data/raw/orders.parquet [--main customers] [--names a,b]
    discover.py ... --answers answers.yaml   # put dictionary answers and source descriptions into the notebook
    discover.py ... --overwrite | --new-version   # when the notebook was edited after the skill wrote it
    discover.py --check                      # check packages
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ml_common  # noqa: E402
from eda_core import ID_NAME  # noqa: E402

TIMINGS = {
    "identifier": "Identifies a row or an entity. Not a feature.",
    "static": "An attribute that is set once and rarely changes, such as a region or a sign up date.",
    "event_time": "The time an event happened. A candidate for the prediction time column.",
    "before_prediction": "Summarises history up to a point in time. Usable if that point is at or before the prediction time.",
    "at_event": "Recorded with the event the row describes, so it is known at the event time (an assumption for event tables).",
    "possibly_after_outcome": "May be filled in after the outcome, which would leak it into the features.",
    "unknown": "Not clear from the name or the data when this is filled in.",
}
FLAG_TEXT = {
    "unclear_meaning": "meaning unclear from the name",
    "timing_unknown": "timing unknown",
    "possibly_after_outcome": "may be recorded after the outcome",
    "candidate_target": "looks like an outcome, possibly the target",
    "personal_data": "possible personal data",
    "mostly_missing": "mostly missing",
    "constant": "a single value",
    "free_text": "free text",
}

# Abbreviations often found in column names, expanded when drafting meanings.
ABBREV = {
    "amt": "amount", "txn": "transaction", "trx": "transaction", "tran": "transaction", "trans": "transaction",
    "cnt": "count", "ct": "count", "qty": "quantity", "num": "number", "no": "number", "nbr": "number",
    "dt": "date", "ts": "timestamp", "tm": "time", "yr": "year", "mo": "month", "mth": "month", "wk": "week",
    "pct": "percent", "perc": "percent", "avg": "average", "mean": "average", "med": "median", "std": "standard deviation",
    "min": "minimum", "max": "maximum", "tot": "total", "sum": "total", "cum": "cumulative",
    "cust": "customer", "usr": "user", "acct": "account", "acc": "account", "merch": "merchant", "mer": "merchant",
    "mcc": "merchant category code", "cat": "category", "desc": "description", "addr": "address",
    "prev": "previous", "prv": "previous", "lst": "last", "curr": "current", "cur": "current",
    "bal": "balance", "lmt": "limit", "cr": "credit", "dr": "debit", "auth": "authorization", "decl": "decline",
    "ctry": "country", "cntry": "country", "geo": "geography", "reg": "region", "lang": "language",
    "dob": "date of birth", "id": "ID", "uid": "user ID", "ref": "reference", "src": "source", "dst": "destination",
    "rev": "revenue", "ltv": "lifetime value", "aov": "average order value", "ctr": "click through rate",
    "imp": "impressions", "impr": "impressions", "clk": "clicks", "conv": "conversion", "seg": "segment",
    "flg": "flag", "ind": "indicator", "pos": "point of sale", "ecom": "ecommerce", "fx": "foreign exchange",
    "ccy": "currency", "cnp": "card not present", "cp": "card present", "chgbk": "chargeback", "cb": "chargeback",
}
WINDOW_RE = re.compile(r"^(\d+)(d|day|days|w|wk|week|weeks|m|mo|mon|month|months|y|yr|year|years|h|hr|hours?)$")
WINDOW_UNIT = {"d": "days", "day": "days", "days": "days", "w": "weeks", "wk": "weeks", "week": "weeks",
               "weeks": "weeks", "m": "months", "mo": "months", "mon": "months", "month": "months",
               "months": "months", "y": "years", "yr": "years", "year": "years", "years": "years",
               "h": "hours", "hr": "hours", "hour": "hours", "hours": "hours"}
GENERIC_NAMES = re.compile(r"^(unnamed.*|col\d*|column\d*|field\d*|var\d*|feature_?\d+|f_?\d+|x\d*|v\d+|value|val|"
                           r"data|misc|other|tmp|temp|attr\d*|c\d+)$", re.I)
CRYPTIC_TOKEN = re.compile(r"^([a-z]{1,2}\d+|\d+[a-z]?|[a-f0-9]{8,}|[bcdfghjklmnpqrstvwxz]{3,})$")

HISTORY_HINT = re.compile(r"(since|last|prev|prior|lag|rolling|trailing|_\d+(d|w|m|y)\b|\d+(d|days|w|m)_|avg|mean|"
                          r"count|cnt|num_|n_|total|sum|max|min|std|freq|recency|tenure|lifetime|cum)", re.I)
STATIC_HINT = re.compile(r"(gender|sex|region|country|ctry|state|province|city|zip|postal|plan|segment|tier|type|"
                         r"category|cat$|channel|brand|product|mcc|industry|language|signup|sign_up|created|"
                         r"registration|open(ed)?_(date|dt)|birth|dob|age$|currency|ccy)", re.I)
POST_HINT = re.compile(r"(outcome|result|label|target|actual|after|post_|_post|final|resolved|resolution|settled|"
                       r"settlement|closed|close_(date|dt|reason)|cancel|churn|future|next_|chargeback|chgbk|refund|"
                       r"dispute|fraud_(flag|confirmed|reported)|confirmed|reported|writeoff|write_off|default)", re.I)
EVENT_DATE_HINT = re.compile(r"(date|_dt$|^dt_|time|timestamp|_ts$|^ts_|day$|period|week|month)", re.I)


# ---------------------------------------------------------------- helpers

def tokens(name):
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def draft_meaning(col, role, info, outcome_like=False):
    """Draft a one sentence meaning from the column name and simple facts about the data.
    Returns (meaning or None, how sure: 'name' or 'unclear').

    outcome_like: the column looks like an outcome. A window in its name then describes the outcome window, whose
    direction (before or after a date) the name does not say, so it is worded neutrally."""
    toks = tokens(col)
    if not toks or GENERIC_NAMES.match(str(col).strip()):
        return None, "unclear"
    words, window, known = [], None, 0
    for t in toks:
        m = WINDOW_RE.match(t)
        if m:
            span = f"{m.group(1)} {WINDOW_UNIT.get(m.group(2), m.group(2))}"
            window = (f"(over a {m.group(1)} {WINDOW_UNIT.get(m.group(2), m.group(2)).rstrip('s')} window)"
                      if outcome_like else f"over the last {span}")
            known += 1
            continue
        if t in ABBREV:
            words.append(ABBREV[t])
            known += 1
        elif CRYPTIC_TOKEN.match(t):
            words.append(t)
        else:
            words.append(t)
            if len(t) >= 3 and re.search(r"[aeiouy]", t):
                known += 1
    if known == 0:
        return None, "unclear"
    flag_like = toks[0] in ("is", "has", "had", "was", "did", "will", "flag", "flg", "ind") or toks[-1] in ("flag", "flg", "ind")
    core = " ".join(w for w in words if w not in ("is", "has", "had", "was", "did", "will", "flag", "flg", "ind"))
    if flag_like or role == "boolean":
        text = f"Whether {core}" if core else "A yes or no flag"
    elif role in ("datetime", "datetime_text"):
        text = core[0].upper() + core[1:] if core else "A date"
        if "date" not in core and "time" not in core:
            text += " (a date)"
    elif role == "id":
        text = core[0].upper() + core[1:] if core else "An identifier"
        if "id" not in core.lower():
            text += " (an identifier)"
    else:
        text = core[0].upper() + core[1:]
    if window:
        text += f" {window}"
    extra = []
    if role == "categorical" and info.get("unique") is not None:
        extra.append(f"{info['unique']:,} distinct values")
    if role in ("datetime", "datetime_text") and info.get("min"):
        extra.append(f"from {info['min']} to {info['max']}")
    if extra:
        text += f" ({'; '.join(extra)})"
    return text + ".", "name"


def draft_timing(col, role, is_candidate_target, event_table=False):
    """Guess when a column is filled in, from its name and role. Returns (timing, basis).

    event_table: the source has one row per event with an event time, so plain attributes of the row are assumed
    to be recorded with the event."""
    name = str(col)
    if role == "id" or (ID_NAME.search(name) and role in ("numeric", "categorical")):
        return "identifier", "it is an identifier"
    if role == "pii":
        return "static", "personal data about the entity; it is kept out of the features in any case"
    if is_candidate_target:
        return "possibly_after_outcome", "it looks like an outcome; if it is the target, it is only known after the prediction"
    if POST_HINT.search(name):
        return "possibly_after_outcome", f"the name contains '{POST_HINT.search(name).group(0)}', which usually describes something that happens later"
    if role in ("datetime", "datetime_text"):
        if STATIC_HINT.search(name):
            return "static", "a date set once, such as a sign up or creation date"
        return "event_time", "a date or timestamp of an event"
    if HISTORY_HINT.search(name):
        return "before_prediction", f"the name contains '{HISTORY_HINT.search(name).group(0)}', which usually describes history up to a point in time"
    if STATIC_HINT.search(name):
        return "static", f"the name contains '{STATIC_HINT.search(name).group(0)}', which usually describes a stable attribute"
    if event_table and role in ("numeric", "categorical", "boolean"):
        return "at_event", "assumed: a plain attribute in a table with one row per event is recorded with the event"
    return "unknown", "nothing in the name or the data shows when it is filled in"


def profile_info(eda, col):
    s = eda.df[col]
    info = {"type": str(s.dtype), "role": eda.roles.get(col, "other"),
            "missing_pct": round(float(s.isna().mean() * 100), 2)}
    try:
        info["unique"] = int(s.nunique())
    except TypeError:
        info["unique"] = None
    if col in eda.parsed_dates:
        d = eda.parsed_dates[col].dropna()
        if len(d):
            info["min"], info["max"] = f"{d.min():%Y-%m-%d}", f"{d.max():%Y-%m-%d}"
    return info


def draft_dictionary(eda, source_name, candidate_targets):
    entries = []
    # An event table: unique on a single ID column and has an event time column.
    g = eda.grain
    event_table = g.get("type") == "single" and any(
        draft_timing(c, eda.roles.get(c), False)[0] == "event_time" for c in eda.parsed_dates)
    for c in eda.df.columns:
        info = profile_info(eda, c)
        role = info["role"]
        if role in ("numeric", "categorical") and ID_NAME.search(str(c)):
            role = info["role"] = "id"
        meaning, how = draft_meaning(c, role, info, outcome_like=c in candidate_targets)
        timing, basis = draft_timing(c, role, c in candidate_targets, event_table)
        flags = []
        if how == "unclear":
            flags.append("unclear_meaning")
        if timing == "unknown":
            flags.append("timing_unknown")
        if timing == "possibly_after_outcome":
            flags.append("candidate_target" if c in candidate_targets else "possibly_after_outcome")
        if role == "pii":
            flags.append("personal_data")
        if role == "text":
            flags.append("free_text")
        if info["missing_pct"] >= 50:
            flags.append("mostly_missing")
        if info.get("unique") == 1:
            flags.append("constant")
        entries.append({
            "column": str(c),
            "type": info["type"],
            "role": {"datetime_text": "date stored as text", "pii": "personal data", "id": "identifier",
                     "text": "free text"}.get(role, role),
            "unique": info.get("unique"),
            "missing_pct": info["missing_pct"],
            "meaning": meaning,
            "meaning_source": how,  # name, unclear, reviewed (Claude improved the wording), user (confirmed)
            "timing": timing,
            "timing_basis": basis,
            "flags": flags,
            "needs_review": bool({"unclear_meaning", "timing_unknown", "possibly_after_outcome", "candidate_target"} & set(flags)),
            "notes": None,
        })
    return entries


KEEP_ON_REDRAFT = ("meaning", "meaning_source", "timing", "timing_basis", "needs_review", "notes")


def merge_dictionary(old, new):
    """Keep what a person or Claude already reviewed. Statistics are refreshed, and drafted entries are replaced."""
    by_col = {e.get("column"): e for e in (old or {}).get("columns") or [] if isinstance(e, dict)}
    out = []
    for e in new:
        prev = by_col.get(e["column"])
        if prev and prev.get("meaning_source") in ("reviewed", "user"):
            for k in KEEP_ON_REDRAFT:
                if k in prev:
                    e[k] = prev[k]
            e["flags"] = [f for f in e["flags"] if f not in ("unclear_meaning", "timing_unknown")] \
                if prev.get("meaning_source") == "user" else e["flags"]
        elif prev and prev.get("notes"):
            e["notes"] = prev["notes"]
        out.append(e)
    return out


def time_coverage(eda):
    """Rows per month of the main date column, months with no rows, and freshness."""
    col = eda._primary_date()
    if not col:
        return None
    d = eda.parsed_dates[col].dropna()
    if not len(d):
        return None
    per_month = d.dt.to_period("M").value_counts().sort_index()
    import pandas as pd
    full = pd.period_range(per_month.index.min(), per_month.index.max(), freq="M")
    per_month = per_month.reindex(full, fill_value=0)
    gaps = [str(p) for p, v in per_month.items() if v == 0]
    low = per_month[per_month > 0]
    thin = [str(p) for p, v in low.items() if v < 0.2 * low.median()] if len(low) >= 3 else []
    return {
        "column": str(col), "min": f"{d.min():%Y-%m-%d}", "max": f"{d.max():%Y-%m-%d}",
        "history_days": int((d.max() - d.min()).days),
        "days_since_last": int((pd.Timestamp(_dt.date.today()) - d.max().normalize()).days),
        "months": len(per_month), "months_with_no_rows": gaps, "thin_months": thin,
        "_per_month": per_month,
    }


def join_checks(frames, roles):
    """Check how each pair of sources joins on the columns they share. Counts only."""
    import pandas as pd
    names = list(frames)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = frames[a], frames[b]
            shared = [c for c in A.columns if c in B.columns]
            keys = [c for c in shared if not pd.api.types.is_float_dtype(A[c]) and roles[a].get(c) not in ("text", "other")
                    and roles[a].get(c) != "pii"]
            id_like = [c for c in keys if roles[a].get(c) == "id" or roles[b].get(c) == "id"
                       or re.search(r"(^id$|_id$|^id_|_key$|_no$|number$)", str(c), re.I)]
            options = [[c] for c in (id_like or keys)[:4]]
            if len(id_like) >= 2:
                options.append(id_like[:3])
            for key in options:
                try:
                    ka = A[key].astype(str).agg("|".join, axis=1) if len(key) > 1 else A[key[0]].astype(str)
                    kb = B[key].astype(str).agg("|".join, axis=1) if len(key) > 1 else B[key[0]].astype(str)
                except Exception:
                    continue
                ca, cb = ka.value_counts(), kb.value_counts()
                left_match = float(ka.isin(cb.index).mean() * 100)
                right_match = float(kb.isin(ca.index).mean() * 100)
                joined_rows = float((ka.map(cb).fillna(0)).sum())
                fan = joined_rows / max(int(ka.isin(cb.index).sum()), 1)
                if ca.max() == 1 and cb.max() == 1:
                    kind = "one to one"
                elif ca.max() == 1:
                    kind = f"one {a} row to many {b} rows"
                elif cb.max() == 1:
                    kind = f"many {a} rows to one {b} row"
                else:
                    kind = "many to many"
                notes = []
                if kind == "many to many":
                    notes.append("a join multiplies rows, so aggregate one side to the key first")
                if left_match < 90:
                    notes.append(f"only {left_match:.0f}% of {a} rows find a match")
                if right_match < 90:
                    notes.append(f"only {right_match:.0f}% of {b} rows find a match")
                out.append({"left": a, "right": b, "keys": key, "relationship": kind,
                            "left_match_pct": round(left_match, 1), "right_match_pct": round(right_match, 1),
                            "rows_per_matched_left_row": round(fan, 2),
                            "left_rows_per_key_max": int(ca.max()), "right_rows_per_key_max": int(cb.max()),
                            "notes": notes})
    return out



def dictionary_summary(dictionary):
    out = {"columns": 0, "needs_review": 0, "unclear_meaning": [], "timing_unknown": [],
           "possibly_after_outcome": [], "candidate_targets": [], "personal_data": []}
    for src, block in (dictionary.get("sources") or {}).items():
        for e in block.get("columns") or []:
            out["columns"] += 1
            ref = f"{src}.{e['column']}"
            if e.get("needs_review"):
                out["needs_review"] += 1
            for f in ("unclear_meaning", "timing_unknown", "possibly_after_outcome", "personal_data"):
                if f in (e.get("flags") or []):
                    out[f].append(ref)
            if "candidate_target" in (e.get("flags") or []):
                out["candidate_targets"].append(ref)
    return out


def dictionary_questions(dictionary):
    """At most a few grouped open questions, so the spec does not fill up with one question per column."""
    qs = []
    for src, block in (dictionary.get("sources") or {}).items():
        unclear = [e["column"] for e in block.get("columns") or [] if "unclear_meaning" in (e.get("flags") or [])]
        if unclear:
            qs.append((f"What do these {src} columns mean: " + ", ".join(f"`{c}`" for c in unclear[:15])
                       + (f", and {len(unclear) - 15} more" if len(unclear) > 15 else "") + "? (see notebooks/02_data_discovery.ipynb)",
                       "framing"))
        post = [e["column"] for e in block.get("columns") or [] if "possibly_after_outcome" in (e.get("flags") or [])]
        if post:
            qs.append((f"When are these {src} columns filled in, relative to the outcome: "
                       + ", ".join(f"`{c}`" for c in post[:15]) + "? If after, they must be excluded from the features.",
                       "framing"))
    return qs



DICT_Q_PREFIXES = ("What do these ", "When are these ")


def close_stale_questions(spec_file, current):
    """Resolve earlier dictionary questions from this skill that the current dictionary no longer raises."""
    if not os.path.exists(spec_file):
        return
    spec, _ = ml_common.load_or_create(spec_file)
    texts = {q if isinstance(q, str) else q[0] for q in current}
    changed = False
    for q in spec.get("open_questions") or []:
        if (isinstance(q, dict) and q.get("status") == "open" and q.get("raised_by") in ("ml-data-discovery", "data-discovery")
                and str(q.get("question", "")).startswith(DICT_Q_PREFIXES) and q.get("question") not in texts):
            q["status"] = "resolved"
            q["answer"] = "Answered in the data dictionary."
            changed = True
    if changed:
        ml_common.save(spec_file, spec)


def load_dictionary(path):
    if not os.path.exists(path):
        return None
    return ml_common.load(path)


def previous_answers(path):
    """Answers already confirmed in the dictionary file (meaning_source user or reviewed), in the form of the
    notebook's dictionary_answers cell, so a rebuilt notebook still shows them."""
    out = {}
    for src, block in ((load_dictionary(path) or {}).get("sources") or {}).items():
        for e in block.get("columns") or []:
            if e.get("meaning_source") in ("user", "reviewed"):
                ans = {k: e.get(k) for k in ("meaning", "timing", "notes") if e.get(k)}
                ans["source"] = e["meaning_source"]
                out[f"{src}.{e['column']}"] = ans
    return out


def save_dictionary(path, dictionary):
    header = ("# Data dictionary drafted by the ml-data-discovery skill from column names and the data.\n"
              "# It is written by notebooks/02_data_discovery.ipynb. To change an entry, add it to the dictionary_answers\n"
              "# cell of that notebook and run it again, so the notebook and this file stay the same.\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if ml_common.yaml is not None:
            fh.write(header)
        fh.write(ml_common.dump_text(ml_common._clean(dictionary)))



# ---------------------------------------------------------------- the discovery, used by the notebook

REVIEW_FLAGS = {"unclear_meaning", "timing_unknown", "possibly_after_outcome", "candidate_target"}


class Discovery:
    """Profiles every source with the EDA engine, drafts the data dictionary, applies the answers from the
    notebook, and writes the dictionary file and the discovery section of the spec.

        disc = Discovery(SPEC_PATH, frames, SOURCES, MAIN, date_columns)
        disc.table()                     # the drafted dictionary as a DataFrame
        disc.apply(dictionary_answers)   # answers from the notebook's decision cell
        disc.write(SOURCE_DESCRIPTIONS)  # writes docs/data_dictionary.yaml and the spec
    """

    def __init__(self, spec_path, frames, sources, main, date_columns=None, sample_rows=200_000):
        from eda_core import EDAReport
        import matplotlib
        import matplotlib.pyplot as plt
        self.spec_path = spec_path
        self.root = os.path.dirname(os.path.abspath(spec_path))
        self.spec, _ = ml_common.load_or_create(spec_path)
        self.paths = ml_common.project_paths(self.spec, self.root)
        self.frames, self.sources, self.main = frames, sources, main
        if main not in frames:
            raise ValueError(f"MAIN must be one of: {', '.join(frames)}")
        fr = self.spec.get("framing") or {}
        data = self.spec.get("data") or {}
        id_cols = list(dict.fromkeys([*(fr.get("entity_id_cols") or []), *(data.get("id_cols") or [])]))
        self.reports, self.coverage = {}, {}
        backend = matplotlib.get_backend()
        for name, df in frames.items():
            dcol = (date_columns or {}).get(name)
            dcol = dcol if dcol in df.columns else None
            with plt.ioff():  # the EDA engine draws plots; they are not needed here
                eda = EDAReport(df, name=name, id_cols=[c for c in id_cols if c in df.columns], date_col=dcol,
                                mode="discovery", sample_rows=sample_rows, max_plots=1).run()
            plt.close("all")
            self.reports[name] = eda
            self.coverage[name] = time_coverage(eda)
        matplotlib.use(backend)
        self.dict_path = os.path.join(self.paths["docs"], "data_dictionary.yaml")
        old = load_dictionary(self.dict_path) or {}
        self.dictionary = {"sources": {}}
        for name in frames:
            hints = self.reports[name].framing_hints()
            cands = {c["column"] for c in hints["candidate_targets"]
                     if any("outcome" in r or "target" in r for r in c["reasons"])}
            prev = (old.get("sources") or {}).get(name)
            self.dictionary["sources"][name] = {
                "path": ml_common.relpath(sources[name], self.root),
                "description": (prev or {}).get("description"),
                "columns": merge_dictionary(prev, draft_dictionary(self.reports[name], name, cands)),
            }

    def table(self, needs_review_only=False):
        import pandas as pd
        rows = []
        for src, block in self.dictionary["sources"].items():
            for e in block["columns"]:
                if needs_review_only and not e.get("needs_review"):
                    continue
                rows.append({"source": src, "column": e["column"], "role": e["role"], "meaning": e.get("meaning"),
                             "timing": e["timing"], "flags": ", ".join(FLAG_TEXT.get(f, f) for f in e["flags"]),
                             "review": "yes" if e.get("needs_review") else ""})
        return pd.DataFrame(rows)

    def apply(self, answers):
        """Apply {"source.column": {"meaning", "timing", "notes", "source"}} from the notebook. Returns the
        references that matched no column."""
        unknown = []
        for ref, ans in (answers or {}).items():
            src, _, col = str(ref).partition(".")
            block = self.dictionary["sources"].get(src) or {}
            entry = next((e for e in block.get("columns", []) if e["column"] == col), None)
            if entry is None:
                unknown.append(ref)
                continue
            ans = ans or {}
            if ans.get("timing") and ans["timing"] not in TIMINGS:
                raise ValueError(f"{ref}: timing must be one of {', '.join(TIMINGS)}")
            if ans.get("meaning"):
                entry["meaning"] = ans["meaning"]
            if ans.get("timing"):
                entry["timing"] = ans["timing"]
                entry["timing_basis"] = ans.get("timing_basis") or "given in the notebook"
            if ans.get("notes"):
                entry["notes"] = ans["notes"]
            how = ans.get("source", "user")
            entry["meaning_source"] = how
            flags = [f for f in entry["flags"] if f not in ("unclear_meaning", "timing_unknown")]
            if entry["timing"] != "possibly_after_outcome":
                flags = [f for f in flags if f not in ("possibly_after_outcome", "candidate_target")]
            if not entry.get("meaning"):
                flags.append("unclear_meaning")
            if entry["timing"] == "unknown":
                flags.append("timing_unknown")
            entry["flags"] = flags
            entry["needs_review"] = False if how == "user" else bool(REVIEW_FLAGS & set(flags))
        return unknown

    def write(self, descriptions=None):
        """Write docs/data_dictionary.yaml and the discovery section of the spec. Returns the summary."""
        for name, text in (descriptions or {}).items():
            if name in self.dictionary["sources"] and text:
                self.dictionary["sources"][name]["description"] = text
        save_dictionary(self.dict_path, self.dictionary)
        roles = {n: r.roles for n, r in self.reports.items()}
        sources = []
        for name, eda in self.reports.items():
            df, cov, hints = self.frames[name], self.coverage[name], eda.framing_hints()
            sev = {}
            for g in eda.issue_groups:
                sev[g["severity"]] = sev.get(g["severity"], 0) + 1
            sources.append({
                "name": name, "path": ml_common.relpath(self.sources[name], self.root),
                "description": self.dictionary["sources"][name].get("description"),
                "rows": int(len(df)), "columns": int(df.shape[1]), "duplicate_rows": int(eda.duplicate_rows),
                "grain": eda.grain, "entity_id_candidates": hints["entity_id_candidates"],
                "date_columns": hints["date_columns"], "main_date": hints["main_date"],
                "history_days": cov["history_days"] if cov else None,
                "days_since_last_date": cov["days_since_last"] if cov else None,
                "months_with_no_rows": cov["months_with_no_rows"] if cov else [],
                "thin_months": cov["thin_months"] if cov else [],
                "issues": [{"severity": g["severity"], "finding": g["kind"] if g["count"] else g["finding"],
                            "count": g["count"]} for g in eda.issue_groups[:10]],
                "issue_counts": sev,
            })
        joins = join_checks(self.frames, roles) if len(self.frames) > 1 else []
        summary = {
            "run_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
            "notebook": ml_common.relpath(os.path.join(self.paths["notebooks"], ml_common.notebook_name("discovery")), self.root),
            "dictionary": ml_common.relpath(self.dict_path, self.root),
            "main_source": self.main,
            "sources": sources,
            "joins": joins,
            "dictionary_summary": dictionary_summary(self.dictionary),
            "framing_hints": self.reports[self.main].framing_hints(),
        }
        spec, _ = ml_common.load_or_create(self.spec_path)
        entries = list((spec.get("data") or {}).get("sources") or [])
        for s in sources:
            entry = {"name": s["name"], "path": s["path"], "rows": s["rows"], "columns": s["columns"],
                     "description": s["description"]}
            for i, e in enumerate(entries):
                if isinstance(e, dict) and os.path.normpath(str(e.get("path"))) == os.path.normpath(s["path"]):
                    entries[i] = {**e, **{k: v for k, v in entry.items() if v is not None}}
                    break
            else:
                entries.append(entry)
        qs = dictionary_questions(self.dictionary)
        for j in joins:
            if j["notes"]:
                qs.append((f"The join of {j['left']} and {j['right']} on {' + '.join(j['keys'])} is {j['relationship']}: "
                           + "; ".join(j["notes"]) + ". Is that expected?", "framing"))
        # Only the anchors come before discovery; the rest of the business framing is drafted from these findings
        # next, which ml-status already shows as the next step.
        if not ml_common.anchors_done(spec):
            qs.append((ml_common.BUSINESS_TODO, "business"))
        close_stale_questions(self.spec_path, qs)
        ml_common.update_section(self.spec_path, "discovery", summary, skill="ml-data-discovery", replace=True,
                                 open_questions=qs, question_stage="framing", extra={"data": {"sources": entries}},
                                 action="wrote discovery from its notebook")
        return summary


# ---------------------------------------------------------------- the notebook

def build_notebook(sources, main, spec, scripts_dir, nb_name, answers=None, descriptions=None, date_columns=None):
    import nb_common
    from nb_common import code, lit, md
    cells = nb_common.setup_cells(
        "discovery", "Data Discovery",
        "What the data is, before deciding what to predict: each source's size, grain, IDs, and coverage over "
        "time, how the sources join, candidate targets, and a drafted data dictionary. Every check below is short "
        "pandas code, so you can read it, change it, and rerun it.\n\n"
        "**How to use this notebook.** The cells marked *Decision* hold the choices: which files, the main source, "
        "and the answers for the data dictionary. Edit them, then choose Run All. The last cells write "
        "`docs/data_dictionary.yaml` and the `discovery` section of `ml_spec.yaml`.\n\n"
        "**Privacy.** Raw rows appear only once, in the *First Rows* cell. Everything else is counts and summaries.",
        scripts_dir, nb_name)
    cells.append(code('''
import re

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import discover
from eda_core import load_data, suggest_targets

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
'''))
    cells.append(md("## Sources\n\nThe data files to profile, as `name: path` from the project root, and the main "
                    "source: the one the model will most likely be trained on."))
    cells.append(code(
        "# Decision\n"
        f"SOURCES = {lit(sources, indent=0)}\n\n"
        f"MAIN = {main!r}\n\n"
        "# Optional: the main date column of a source, when the automatic choice below is wrong, as name: column\n"
        f"DATE_COLUMNS = {lit(date_columns or {}, indent=0)}\n\n"
        "# Optional: one sentence on what each source is and where it comes from\n"
        f"SOURCE_DESCRIPTIONS = {lit(descriptions or {}, indent=0)}", decision=True))
    cells.append(code('''
frames = {name: load_data(path) for name, path in SOURCES.items()}
for name, df in frames.items():
    print(f"{name}: {len(df):,} rows and {df.shape[1]} columns from {SOURCES[name]}")
'''))
    cells.append(md("## First Rows\n\nThe only cell that shows raw rows."))
    cells.append(code('''
for name, df in frames.items():
    print(name)
    display(df.head())
'''))
    cells.append(md("## Columns and Types\n\nThe type, missing share, and number of distinct values of every column."))
    cells.append(code('''
for name, df in frames.items():
    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing %": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
    })
    print(f"{name}: {df.shape[1]} columns")
    display(overview)
'''))
    cells.append(md("## Grain\n\nWhat makes a row unique. First single columns, then pairs of columns. Decimal "
                    "measures are skipped, because they pair up with almost anything into a unique combination. A "
                    "grain of *unclear* means one row's meaning has to be confirmed with the data owner."))
    cells.append(code('''
def unique_keys(df, max_cols=10):
    """The first column, or else pair of columns, that is unique for every row (after removing exact duplicates)."""
    data = df.drop_duplicates()
    n = len(data)
    singles = [c for c in data.columns if data[c].notna().all() and data[c].nunique() == n]
    if singles:
        return singles[:1]
    cands = [c for c in data.columns if 1 < data[c].nunique() < n and not pd.api.types.is_float_dtype(data[c])]
    cands = sorted(cands, key=lambda c: -data[c].nunique())[:max_cols]
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if data[[a, b]].notna().all().all() and not data.duplicated(subset=[a, b]).any():
                return [a, b]
    return []


for name, df in frames.items():
    key = unique_keys(df)
    grain = " + ".join(key) if key else "unclear"
    print(f"{name}: {df.duplicated().sum():,} exact duplicate rows. One row per: {grain}")
'''))
    cells.append(md("## Possible Entity IDs\n\nColumns named like IDs, with how many rows each value has. An ID "
                    "with several rows per value (a customer with many orders) means the split later must keep each "
                    "entity on one side, or split by time."))
    cells.append(code('''
ID_LIKE = re.compile(r"(^id$|_id$|^id_|uuid|guid|_key$|^key$|_no$|number$)", re.I)

for name, df in frames.items():
    rows = [{"column": c, "distinct values": df[c].nunique(), "rows per value": round(len(df) / max(df[c].nunique(), 1), 2)}
            for c in df.columns if ID_LIKE.search(str(c))]
    print(name)
    display(pd.DataFrame(rows) if rows else "No ID-like columns.")
'''))
    cells.append(md("## Coverage Over Time\n\nThe main date column of each source, its range, how recent the "
                    "latest row is, and rows per month. A month with no rows is either a real gap or a missing load. "
                    "Set `DATE_COLUMNS` above when the automatic choice is wrong."))
    cells.append(code('''
def main_date(df, name):
    """DATE_COLUMNS when set, else the first column stored as dates, else the first text column that reads as dates."""
    if name in DATE_COLUMNS:
        return DATE_COLUMNS[name]
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            sample = df[c].dropna().astype(str).head(500)
            looks_numeric = pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5
            if len(sample) and not looks_numeric and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9:
                return c
    return None


dates = {name: main_date(df, name) for name, df in frames.items()}
for name, df in frames.items():
    col = dates[name]
    if col is None:
        print(f"{name}: no date column found.")
        continue
    d = pd.to_datetime(df[col], errors="coerce", format="mixed")
    per_month = d.dt.to_period("M").value_counts().sort_index()
    per_month = per_month.reindex(pd.period_range(per_month.index.min(), per_month.index.max(), freq="M"), fill_value=0)
    gaps = [str(p) for p, v in per_month.items() if v == 0]
    age = (pd.Timestamp.today().normalize() - d.max().normalize()).days
    print(f"{name}: {col} runs from {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d} ({(d.max() - d.min()).days:,} days). "
          f"The latest row is {age:,} days old. Months with no rows: {', '.join(gaps) or 'none'}.")
    ax = per_month.plot(kind="bar", figsize=(10, 2.5), color="#2a78d6", title=f"{name}: rows per month of {col}")
    ax.set_xlabel("")
    plt.show()
'''))
    cells.append(md("## Missing Values\n\nThe columns with the most missing values in each source."))
    cells.append(code('''
for name, df in frames.items():
    missing = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    missing = missing[missing > 0]
    print(f"{name}: {len(missing)} of {df.shape[1]} columns have missing values")
    if len(missing):
        display(missing.head(15).to_frame("missing %"))
'''))
    cells.append(md("## How the Sources Join\n\nFor every column two sources share, the share of rows on each side "
                    "that find a match, and the most rows one key value has on each side. More than one row per key "
                    "on both sides means a join multiplies rows, so one side has to be aggregated first."))
    cells.append(code('''
names = list(frames)
if len(names) < 2:
    print("There is one source, so there are no joins to check.")
for i, a in enumerate(names):
    for b in names[i + 1:]:
        shared = [c for c in frames[a].columns
                  if c in frames[b].columns and not pd.api.types.is_float_dtype(frames[a][c])]
        if not shared:
            print(f"{a} and {b} share no column that could be a join key.")
        for key in shared:
            left, right = frames[a][key], frames[b][key]
            print(f"{a} and {b} on {key}: {left.isin(right).mean():.1%} of {a} rows and {right.isin(left).mean():.1%} "
                  f"of {b} rows find a match. Most rows per key: {left.value_counts().max():,} in {a}, "
                  f"{right.value_counts().max():,} in {b}.")
'''))
    cells.append(md("## Candidate Targets\n\nColumns in the main source that look like an outcome, ranked by their "
                    "name and their values, with the class balance of the yes or no ones. The target itself is chosen "
                    "in `03_problem_framing.ipynb`."))
    cells.append(code('''
candidates = pd.DataFrame([c for c in suggest_targets(frames[MAIN], top=5) if c["score"] > 0])
if candidates.empty:
    print("No column looks like a target. It will probably be built from events in a time window.")
else:
    display(candidates[["column", "suggested_type", "unique", "missing_pct", "reasons"]])
    for c in candidates.itertuples():
        if c.unique == 2:
            print(f"{c.column}: class shares {frames[MAIN][c.column].value_counts(normalize=True).round(3).to_dict()}")
'''))
    cells.append(md("## Columns That May Be Recorded After the Outcome\n\nColumns whose names suggest something "
                    "that happens later, such as a refund, a chargeback, a cancellation, or a final status. Each one "
                    "needs a decision before it can be a feature, because it may leak the outcome."))
    cells.append(code('''
AFTER_OUTCOME = discover.POST_HINT  # refund, chargeback, cancel, closed, resolved, settled, final, outcome, ...
for name, df in frames.items():
    found = [c for c in df.columns if AFTER_OUTCOME.search(str(c))]
    print(f"{name}: {', '.join(found) or 'none'}")
'''))
    cells.append(md("## Data Dictionary (Draft)\n\nEvery column with a meaning drafted from its name and the data, "
                    "and its timing: when it is filled in relative to the prediction. Columns whose meaning or timing "
                    "could not be worked out are left blank and flagged for review. The rules are in "
                    "`reference/dictionary_rules.md` of the ml-data-discovery skill. This cell also profiles each "
                    "source with the shared EDA engine, which takes a little longer on large files."))
    cells.append(code('''
disc = discover.Discovery(SPEC_PATH, frames, SOURCES, MAIN, date_columns=dates)
for name, cov in disc.coverage.items():
    if cov and cov["column"] != dates[name]:
        print(f"Note: the EDA engine chose {cov['column']} as the main date of {name}. Set DATE_COLUMNS to override.")
display(disc.table())
'''))
    cells.append(md("### Columns to Review"))
    cells.append(code('''
review = disc.table(needs_review_only=True)
print(f"{len(review)} columns need review.")
display(review)
'''))
    cells.append(md("## Answers for the Dictionary\n\nFill in what each flagged column means and when it is filled "
                    "in, as `\"source.column\": {...}`. Allowed timings: `identifier`, `static`, `event_time`, "
                    "`before_prediction`, `at_event`, `possibly_after_outcome`, `unknown`. Answers marked "
                    "`\"source\": \"user\"` (the default) count as confirmed. Answers are kept when the notebook "
                    "runs again."))
    cells.append(code(f"# Decision\ndictionary_answers = {lit(answers or {}, indent=0)}", decision=True))
    cells.append(md("## Write to the Spec\n\nApplies the answers, writes `docs/data_dictionary.yaml` and the "
                    "`discovery` section of `ml_spec.yaml`, and records this run."))
    cells.append(code('''
unknown = disc.apply(dictionary_answers)
if unknown:
    print("These answers match no column: " + ", ".join(unknown))
summary = disc.write(SOURCE_DESCRIPTIONS)
nb_common.record_run(SPEC_PATH, STAGE, NOTEBOOK, "ml-data-discovery")
ds = summary["dictionary_summary"]
print(f"Wrote {summary['dictionary']} and the discovery section of ml_spec.yaml.")
print(f"Dictionary: {ds['columns']} columns, {ds['needs_review']} still need review.")
display(disc.table(needs_review_only=True))
'''))
    cells.append(md("## Next Step\n\nAnswer the flagged columns above, especially the ones that may be recorded "
                    "after the outcome, and rerun this notebook. Then run `/ml-problem-framing`, which proposes the "
                    "target, unit of prediction, and prediction time from these findings. Check progress with `/ml-status`."))
    return nb_common.build(cells, spec, "ml-data-discovery", "discovery")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="Data files: csv, tsv, parquet, xlsx, json, jsonl, or feather")
    ap.add_argument("--names", help="Comma separated source names, one per file (default: the file names)")
    ap.add_argument("--main", help="The main source, the one the model will most likely be trained on "
                                   "(default: the largest file)")
    ap.add_argument("--answers", help="YAML with dictionary_answers and source_descriptions, put into the notebook")
    ap.add_argument("--spec", help="Spec path (default: ml_spec.yaml at the project root)")
    ap.add_argument("--overwrite", action="store_true", help="Replace a notebook that was edited")
    ap.add_argument("--new-version", action="store_true", help="Write the next _v2, _v3 notebook instead")
    ap.add_argument("--no-run", action="store_true", help="Write the notebook without running it")
    ap.add_argument("--check", action="store_true", help="Check packages and exit")
    a = ap.parse_args()

    if a.check:
        import importlib.util
        missing = [m for m in ("pandas", "numpy", "matplotlib", "seaborn", "yaml", "nbformat", "nbclient", "ipykernel")
                   if importlib.util.find_spec(m) is None]
        print(f"Python: {sys.executable}")
        print("Packages: " + ("all installed" if not missing else "missing " + ", ".join(missing)))
        sys.exit(1 if missing else 0)
    if not a.files:
        ap.error("give one or more data files")
    for f in a.files:
        if not os.path.exists(f):
            sys.exit(f"File not found: {f}")
    names = [n.strip() for n in a.names.split(",")] if a.names else [os.path.basename(f).split(".")[0] for f in a.files]
    if len(names) != len(a.files) or len(set(names)) != len(names):
        ap.error("--names needs one unique name per file")
    main_src = a.main or names[max(range(len(a.files)), key=lambda i: os.path.getsize(a.files[i]))]
    if main_src not in names:
        ap.error(f"--main must be one of: {', '.join(names)}")

    import nb_common
    spec_path = ml_common.spec_path(a.spec)
    root = os.path.dirname(spec_path)
    spec, _ = ml_common.load_or_create(spec_path)
    if not os.path.exists(spec_path):
        ml_common.save(spec_path, spec)
    extra = ml_common.load(a.answers) if a.answers else {}
    sources = {n: ml_common.relpath(os.path.abspath(f), root) for n, f in zip(names, a.files)}
    nb_name = ml_common.notebook_name("discovery")
    path = os.path.join(ml_common.project_paths(spec, root)["notebooks"], nb_name)
    answers = previous_answers(os.path.join(ml_common.project_paths(spec, root)["docs"], "data_dictionary.yaml"))
    answers.update(extra.get("dictionary_answers") or {})
    nb = build_notebook(sources, main_src, spec, HERE, nb_name, answers=answers,
                        descriptions=extra.get("source_descriptions"), date_columns=extra.get("date_columns"))
    mode = "overwrite" if a.overwrite else "new-version" if a.new_version else "ask"
    res = nb_common.save_and_run(nb, path, mode=mode, execute=not a.no_run)
    if res["status"] == "edited":
        print(f"Not written: {ml_common.relpath(path, root)} was edited after the skill wrote it. Rerun with "
              f"--overwrite to replace it, or --new-version to write {os.path.basename(res['next_version'])}. To keep "
              f"the edits, change its decision cells and run it with nb_common.py run.")
        sys.exit(2)
    print(f"Wrote notebook: {ml_common.relpath(res['path'], root)}")
    print("Ran: yes" if res["executed"] else f"Ran: no ({res['error'] or 'not requested'})")
    if not res["executed"]:
        print("Spec: not updated. Open the notebook and choose Run All to write the spec.")
        sys.exit(1 if res["error"] else 0)
    disc = ml_common.load(spec_path).get("discovery") or {}
    ds = disc.get("dictionary_summary") or {}
    print(f"Spec: discovery section written. Dictionary: {disc.get('dictionary')}")
    print(f"Main source: {disc.get('main_source')}")
    for s in disc.get("sources") or []:
        g = s["grain"]
        print(f"  {s['name']}: {s['rows']:,} rows, {s['columns']} columns, grain "
              + (" + ".join(g["columns"]) if g["columns"] else "unclear")
              + (f", {s['history_days']:,} days of history" if s.get("history_days") is not None else ", no dates"))
    for j in disc.get("joins") or []:
        print(f"  join {j['left']} and {j['right']} on {' + '.join(j['keys'])}: {j['relationship']}, "
              f"{j['left_match_pct']}% and {j['right_match_pct']}% matched" + (f" ({'; '.join(j['notes'])})" if j["notes"] else ""))
    print(f"Dictionary: {ds.get('columns', 0):,} columns, {ds.get('needs_review', 0):,} need review "
          f"({len(ds.get('unclear_meaning', []))} unclear meaning, {len(ds.get('timing_unknown', []))} unknown timing, "
          f"{len(ds.get('possibly_after_outcome', []))} possibly after the outcome, "
          f"{len(ds.get('candidate_targets', []))} candidate targets)")


if __name__ == "__main__":
    import nb_common as _nb   # --json output, and the answers kept in answers/<NN>_<stage>.yaml
    _nb.cli(main, "discovery", "ml-data-discovery")
