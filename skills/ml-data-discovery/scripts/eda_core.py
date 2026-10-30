"""Exploratory data analysis engine for the ml-eda-report skill.

Works in two ways:

1. From a script (run_eda.py) that loads a file and writes a markdown report.
2. Inside a Jupyter notebook, on a DataFrame that is already in memory:

       import sys, os
       sys.path.insert(0, os.path.expanduser("~/.claude/skills/ml-eda-report/scripts"))
       from eda_core import eda_report
       eda = eda_report(df, target="label", date_col="created_at")

Privacy rule: the report shows summary statistics only. It never prints raw
rows. Values are shown only for low risk categorical columns, and only when a
value appears at least MIN_VALUE_COUNT times. Identifier, free text, and
columns that look like personal data never have their values shown.

Required packages: pandas, numpy, matplotlib, seaborn.
Optional package: scikit-learn (mutual information).
"""

from __future__ import annotations

import io
import math
import os
import re
import warnings
from dataclasses import dataclass, field

import contextlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.figure import Figure

try:  # optional
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
except Exception:  # pragma: no cover
    mutual_info_classif = mutual_info_regression = None

# ---------------------------------------------------------------- settings

MIN_VALUE_COUNT = 5          # never show a category value seen fewer times than this
NEAR_CONSTANT = 0.99         # share of the most common value that counts as near constant
HIGH_CARDINALITY = 50        # unique values above which a categorical needs special encoding
RARE_SHARE = 0.01            # categories below this share count as rare
HIGH_CORR = 0.9              # absolute Spearman correlation flagged as high
HIGH_CRAMERS_V = 0.8
MISSING_CORR = 0.8           # correlation of missing indicators listed as 'missing together'
MAX_MISSING_PAIRS = 30       # rows shown in the missing together table
MAX_MISSING_COLS = 100       # columns compared in the missing together table
GROUP_CORR = 0.99            # missing indicators correlated at least this much form one missingness group
MAX_MISSING_GROUPS = 30      # rows shown in the missingness groups table
RULE_PURITY = 0.95           # share of rows in a bucket that must be (all) missing or (all) present for a rule
RULE_STRUCTURAL = 0.98       # purity at which a rule counts as explaining the missingness
RULE_MIN_ROWS = 0.01         # a rule must cover at least this share of rows
RULE_SHARE = 0.5             # a rule must account for this share of the missing (or present) rows to explain a group
RULE_LIFT = 0.2              # the missing rate inside the bucket must differ this much from the overall rate
RULE_SAMPLE = 200000         # rows used to search for rules
LEAKAGE_ASSOC = 0.95         # feature to target association treated as possible leakage
MAX_CORR_COLS = 40
MAX_CRAMERS_COLS = 20
MAX_KEY_PAIR_COLS = 10

# Colors: one blue for data, orange for medians and reference lines, grey for Other and missing.
PRIMARY = "#2a78d6"
PRIMARY_LIGHT = "#bcd6f3"
ACCENT = "#eb6834"
MUTED = "#a3a29c"
ALERT = "#e34948"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#2b2b2b"
DPI = 110


def num_label(x):
    """Short, readable axis or bar label for a number."""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 10 or ax == 0:
        return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"
    return f"{x:.2g}"


THOUSANDS = mticker.FuncFormatter(lambda x, _: num_label(x))

DISGUISED_TOKENS = {
    "", " ", "na", "n/a", "nan", "none", "null", "nil", "missing", "unknown", "unk",
    "?", "-", "--", ".", "undefined", "not available", "not applicable", "#n/a", "tbd",
}
NUMERIC_SENTINELS = [-1, -9, -99, -999, -9999, 999, 9999, 99999, 999999]
PLACEHOLDER_DATES = [pd.Timestamp("1900-01-01"), pd.Timestamp("1970-01-01"), pd.Timestamp("9999-12-31")]

PII_NAME = re.compile(
    r"(e.?mail|phone|mobile|cell|ssn|social.?sec|\bsin\b|passport|licen[cs]e|card.?(num|no)|"
    r"\bpan\b|iban|account.?(num|no)|acct|routing|first.?name|last.?name|full.?name|"
    r"surname|^name$|address|street|postal|zip|birth|\bdob\b|ip.?addr|password|token|secret)",
    re.I,
)
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|_key$|^key$|_no$|number$)", re.I)
LEAK_HINTS = re.compile(
    r"(outcome|result|label|target|actual|after|post_|_post|final|resolved|settled|"
    r"closed|future|next_|chargeback|refund)",
    re.I,
)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
LONG_DIGITS_RE = re.compile(r"^\d{12,19}$")
DATE_HINT_RE = re.compile(r"(?:\d[-/.:]\d|\d{4}-\d|[a-z]{3,9}\s+\d|\d\s+[a-z]{3,9})", re.I)


# ---------------------------------------------------------------- helpers

def fmt(x):
    """Format a value for a table cell."""
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "Yes" if x else "No"
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(x)
        return "" if pd.isna(ts) else (ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.strftime("%Y-%m-%d %H:%M"))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "" if np.isnan(x) else str(x)
        if float(x).is_integer() and abs(x) < 1e15:
            return f"{int(x):,}"
        ax = abs(x)
        if ax >= 1e6:
            return f"{x:,.0f}"
        if ax >= 100:
            return f"{x:,.1f}"
        if ax >= 1:
            return f"{x:,.2f}"
        return f"{x:.3g}"
    return str(x)


def rnd(df, digits=4):
    """Round only the float columns, leaving dates and text untouched."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def pct(x, digits=1):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}%"


def slug(name, i=None):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()[:50] or "col"
    return f"{s}_{i}" if i is not None else s


def short(s, n=40):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def nuniq(s, dropna=True):
    """nunique that also works for columns holding lists or dicts."""
    try:
        return s.nunique(dropna=dropna)
    except TypeError:
        return s.astype(str).where(s.notna()).nunique(dropna=dropna)


def is_text_like(s):
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and not pd.api.types.is_datetime64_any_dtype(s)


def _rc():
    rc = {}
    rc.update(sns.axes_style("whitegrid"))
    rc.update(sns.plotting_context("notebook", font_scale=0.85))
    rc.update({
        "axes.edgecolor": "#d0d0cc", "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#ececea", "grid.linewidth": 0.8, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK, "text.color": INK, "xtick.color": "#55554f",
        "ytick.color": "#55554f", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "legend.frameon": False,
    })
    return rc


@contextlib.contextmanager
def style():
    """Apply the report's seaborn style without changing the user's own matplotlib settings."""
    with mpl.rc_context(_rc()):
        yield


def new_fig(w=7.5, h=3.2, ncols=1, nrows=1, grid_axis="both", **kw):
    fig = Figure(figsize=(w, h), dpi=DPI, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    for ax in axes.flat:
        ax.grid(False)
        if grid_axis:
            ax.grid(True, axis=grid_axis)
        ax.set_axisbelow(True)
    return fig, axes


def pct_label(v):
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def hbar(ax, labels, values, colors, fmt=pct_label, fontsize=8):
    """Horizontal bar chart (largest at the top) with value labels, drawn with seaborn."""
    labels = [str(x) for x in labels]
    if len(set(labels)) < len(labels):  # seaborn needs unique labels, and shortened labels can collide
        seen, out = {}, []
        for x in labels:
            seen[x] = seen.get(x, 0) + 1
            out.append(x if seen[x] == 1 else f"{x} ({seen[x]})")
        labels = out
    pal = dict(zip(labels, colors))
    try:
        sns.barplot(x=list(values), y=labels, hue=labels, palette=pal, legend=False, ax=ax, orient="h",
                    width=0.72, saturation=1)
    except TypeError:  # seaborn older than 0.13
        sns.barplot(x=list(values), y=labels, palette=[pal[l] for l in labels], ax=ax, orient="h", saturation=1)
    for cont in ax.containers:
        ax.bar_label(cont, labels=[fmt(v) if callable(fmt) else fmt.format(v) for v in cont.datavalues], padding=3, fontsize=fontsize, color=INK)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)


def heatmap(ax, mat, diverging=True, annot=False):
    """Lower triangle heatmap with seaborn."""
    mat = mat.iloc[1:, :-1]  # drop the diagonal row and column, which are always 1
    k = len(mat)
    mask = np.triu(np.ones((k, k), dtype=bool), 1)
    sns.heatmap(mat.values.astype(float), mask=mask, ax=ax, cmap="vlag" if diverging else "Blues",
                vmin=-1 if diverging else 0, vmax=1, center=0 if diverging else None, square=True,
                linewidths=0.6, linecolor="white", annot=annot, fmt=".2f", annot_kws={"fontsize": 7},
                xticklabels=[short(c, 20) for c in mat.columns], yticklabels=[short(c, 20) for c in mat.index], cbar_kws={"shrink": 0.7})
    ax.tick_params(axis="x", labelrotation=90, labelsize=8, length=0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8, length=0)
    ax.grid(False)


def date_axis(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def parse_dates(s):
    """Parse a text column into datetimes, returning (parsed, share_parsed)."""
    nonnull = s.dropna()
    if nonnull.empty:
        return None, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(s, errors="coerce")
        except Exception:
            parsed = None
        share = parsed[nonnull.index].notna().mean() if parsed is not None else 0.0
        if share < 0.9:
            try:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                share = parsed[nonnull.index].notna().mean()
            except Exception:
                pass
    if parsed is not None and getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed, float(share)


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    n = tab.values.sum()
    expected = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tab.values - expected) ** 2 / expected)
    phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    kc = k - (k - 1) ** 2 / max(n - 1, 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / d)) if d > 0 else np.nan


def correlation_ratio(categories, values):
    """Eta: how much of a numeric variable's variance is explained by a categorical one."""
    df = pd.DataFrame({"c": categories, "v": values}).dropna()
    if df.empty or df["v"].var() == 0:
        return np.nan
    means = df.groupby("c", observed=True)["v"].agg(["mean", "count"])
    grand = df["v"].mean()
    between = (means["count"] * (means["mean"] - grand) ** 2).sum()
    total = ((df["v"] - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else np.nan


def load_data(path, sheet=None, **kw):
    """Load a data file into a DataFrame based on its extension."""
    p = str(path)
    ext = os.path.splitext(p.lower())[1]
    if p.lower().endswith((".csv.gz", ".csv.zip", ".csv.bz2")):
        ext = ".csv"
    if ext in (".csv", ".txt"):
        return pd.read_csv(p, low_memory=False, **kw)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t", low_memory=False, **kw)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(p, **kw)
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0, **kw)
    if ext == ".json":
        return pd.read_json(p, **kw)
    if ext == ".jsonl":
        return pd.read_json(p, lines=True, **kw)
    if ext == ".feather":
        return pd.read_feather(p, **kw)
    raise ValueError(
        f"Unsupported file type '{ext}'. Supported types are csv, tsv, txt, parquet, xlsx, xls, json, jsonl, and feather."
    )



# ---------------------------------------------------------------- report structure

@dataclass
class Section:
    key: str
    title: str
    intro: str
    blocks: list = field(default_factory=list)
    skipped: str | None = None

    def text(self, s):
        self.blocks.append(("text", s))

    def bullets(self, items):
        if items:
            self.blocks.append(("bullets", list(items)))

    def table(self, caption, df, note=None):
        if df is not None and len(df):
            self.blocks.append(("table", caption, df, note))

    def figure(self, caption, fig, stem):
        self.blocks.append(("figure", caption, fig, stem))


SECTION_ORDER = [
    ("overview", "1. Overview", "Size, sampling, column roles, duplicates, the grain of the data, and type problems."),
    ("profiles", "2. Column Profiles",
     "One row per column. Numeric columns first, then categorical columns, then dates and other columns. "
     "Together these tables cover everything df.info() and df.describe(include=\"all\") show, plus a Flags column "
     "that lists the issues found for each column."),
    ("missing", "3. Missing Values", "Which columns have missing values, disguised missing values, which columns are missing together, and whether the missingness looks structural."),
    ("distributions", "4. Distributions", "Histograms for numeric columns, top values for categorical columns, and rows over time."),
    ("relationships", "5. Relationships Between Columns", "Correlation between numeric columns, association between categorical columns, and date ordering."),
    ("target", "6. Target Analysis", "Target balance, feature relationships with the target, and leakage warnings."),
    ("summary", "7. Data Quality Summary and Next Steps", "Checks against the ML framing when a spec is given, issues grouped by type and ranked by severity, checks that found nothing, and next steps."),
]
SEVERITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
MAX_LISTED = 12  # columns listed per issue row before "and N more"


class EDAReport:
    """Run every EDA section on a DataFrame and render it as markdown or in a notebook."""

    def __init__(self, df, target=None, date_col=None, id_cols=None, name="dataset", sample_rows=200_000,
                 max_plots=24, random_state=0, target_type=None, corr_threshold=HIGH_CORR,
                 cramers_threshold=HIGH_CRAMERS_V, spec=None, mode=None, log_scale=False):
        """log_scale: when True, numeric histograms of strongly skewed columns use a log axis (log1p, or signed log
        for columns with negative values). The default, False, plots every column on its original scale.

        spec: the shared ml_spec.yaml as a dict (optional). When its framing has been started, the summary
        section checks the data against the framing, and spec_summary() returns the results for the spec.

        mode: 'discovery' (no target: profile the data and collect hints for the ML framing) or 'targeted'
        (analyse the target and check the data against the framing). Defaults to targeted when a target is given."""
        mode = mode or ("targeted" if target else "discovery")
        if mode not in ("discovery", "targeted"):
            raise ValueError("mode must be 'discovery' or 'targeted'")
        if mode == "discovery" and target:
            raise ValueError("discovery mode runs without a target. Leave target unset, or use mode='targeted'.")
        self.mode = mode
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if target_type not in (None, "continuous", "categorical"):
            raise ValueError("target_type must be 'continuous', 'categorical', or None")
        if target_type and not target:
            raise ValueError("target_type needs a target column")
        self.target_type = target_type
        self.corr_threshold = corr_threshold
        self.cramers_threshold = cramers_threshold
        self.spearman = self.cramers = None
        self.raw = df
        self.df = df
        self.converted = []
        self.name = name
        self.target = target
        self.date_col = date_col
        self.id_cols = list(id_cols or [])
        self.sample_rows = sample_rows
        self.max_plots = max_plots
        self.log_scale = bool(log_scale)
        self.random_state = random_state
        for c in [target, date_col, *self.id_cols]:
            if c is not None and c not in df.columns:
                raise KeyError(f"Column '{c}' is not in the DataFrame")
        self.sampled = len(df) > sample_rows
        self.s = df.sample(sample_rows, random_state=random_state) if self.sampled else df
        self.sections = {}
        self.issues = []
        self.issue_groups = []
        self.flags = {}
        self.passed = []
        self.profile_tables = []
        self.roles = {}
        self.parsed_dates = {}
        self.figure_count = 0
        self._ran = False
        # Results kept for the shared spec (see spec_summary).
        self.spec = spec or {}
        self.grain = {"type": "unclear", "columns": []}
        self.duplicate_rows = 0
        self.target_info = {}
        self.target_assoc = []
        self.leakage = {}
        self.framing_checks = []
        self.framing_questions = []
        self.missing_groups = []
        self.target_rank = []
        self.target_assoc_df = None
        self.target_ctx = None

    # ------------------------------------------------------------ public API

    def run(self):
        for key, title, intro in SECTION_ORDER:
            self.sections[key] = Section(key, title, intro)
        self._infer_roles()
        self._convert_numeric_text()
        # Checks that feed the Flags column run before the profile tables are filled in.
        order = ["overview", "missing", "profiles", "distributions", "relationships", "target"]
        for key in order:
            sec = self.sections[key]
            try:
                with style():
                    getattr(self, f"_sec_{key}")(sec)
            except Exception as e:  # keep going if one section fails
                sec.skipped = f"This section could not be completed because of an error: {type(e).__name__}: {e}"
        for t in self.profile_tables:
            t["Flags"] = t["Column"].map(lambda c: ", ".join(dict.fromkeys(self.flags.get(c, []))))
        self._sec_summary(self.sections["summary"])
        self._ran = True
        return self

    def show(self, key=None, header=True, corr_threshold=None, cramers_threshold=None, top_n=None,
             min_association=None, min_mutual_info=None, log_scale=None):
        """Display one section (or all sections) inline in a Jupyter notebook.

        corr_threshold and cramers_threshold change only the relationships section display: the tables then list
        numeric pairs with |Spearman| at or above corr_threshold and categorical pairs with Cramér's V at or above
        cramers_threshold. The matrices are not recomputed, so changing them is instant.

        top_n changes only the distributions section display: it plots the top_n numeric columns and the top_n
        categorical columns. With a target, "top" means the strongest association with the target (see the target
        analysis section). Without a target, numeric columns are ranked by skew and categorical columns keep their
        order in the data. The default, None, plots up to max_plots columns as in the first run.
        log_scale also changes only the distributions section: True plots strongly skewed numeric columns on a log
        axis, False on their original scale. The default, None, keeps the setting of the first run.

        For the target section, top_n is the number of features listed in the relationships table and plotted
        (default 40 listed and up to 9 plotted). min_association keeps only features whose absolute association with
        the target is at least that value, and min_mutual_info keeps only features whose mutual information is at
        least that value. The filters combine, and nothing is recomputed. Every plot title shows the feature's
        association and mutual information.
        """
        from IPython.display import Image, Markdown, display

        if not self._ran:
            self.run()
        keys = [key] if key else [k for k, _, _ in SECTION_ORDER]
        for k in keys:
            sec = self.sections[k]
            if k == "relationships" and not sec.skipped and (corr_threshold is not None or cramers_threshold is not None):
                sec = Section(sec.key, sec.title, sec.intro)
                with style():
                    self._relationship_blocks(
                        sec, self.corr_threshold if corr_threshold is None else corr_threshold,
                        self.cramers_threshold if cramers_threshold is None else cramers_threshold)
            if k == "distributions" and not sec.skipped and (
                    top_n is not None or (log_scale is not None and bool(log_scale) != self.log_scale)):
                if top_n is not None and (not isinstance(top_n, (int, np.integer)) or top_n < 1):
                    raise ValueError("top_n must be a whole number of at least 1, or None")
                sec = Section(sec.key, sec.title, sec.intro)
                with style():
                    self._sec_distributions(sec, top_n=None if top_n is None else int(top_n), log_scale=log_scale)
            if k == "target" and not sec.skipped and self.target_assoc_df is not None and (
                    top_n is not None or min_association is not None or min_mutual_info is not None):
                sec = self._filtered_target_section(sec, top_n, min_association, min_mutual_info)
            if header:
                display(Markdown(f"### {sec.title}\n\n{sec.intro}"))
            if sec.skipped:
                display(Markdown(f"*{sec.skipped}*"))
                continue
            for b in sec.blocks:
                kind = b[0]
                if kind == "text":
                    display(Markdown(b[1]))
                elif kind == "bullets":
                    display(Markdown("\n".join(f"* {x}" for x in b[1])))
                elif kind == "table":
                    display(Markdown(f"**{b[1]}**" + (f"\n\n{b[3]}" if b[3] else "")))
                    display(self._display_table(b[2]))
                elif kind == "figure":
                    buf = io.BytesIO()
                    b[2].savefig(buf, format="png", dpi=DPI)
                    display(Image(data=buf.getvalue()))

    def to_markdown(self, out_dir="docs", report_name=None, plot_dir_name=None, source=None):
        """Write the report and its plots. Returns the report path.

        By default the report is saved as docs/eda_<name>.md, with plots in docs/eda_plots/<name>/.
        """
        if not self._ran:
            self.run()
        name = slug(self.name)
        report_name = report_name or f"eda_{name}.md"
        plot_dir_name = plot_dir_name or f"eda_plots/{name}"
        os.makedirs(out_dir, exist_ok=True)
        plot_dir = os.path.join(out_dir, plot_dir_name)
        os.makedirs(plot_dir, exist_ok=True)
        for f in os.listdir(plot_dir):  # remove plots left over from an older run
            if f.endswith(".png"):
                os.remove(os.path.join(plot_dir, f))
        L = [f"# Exploratory Data Analysis{': Discovery' if self.mode == 'discovery' else ''}: {self.name}", ""]
        meta = [f"**Rows:** {len(self.df):,}", f"**Columns:** {self.df.shape[1]:,}"]
        if source:
            meta.insert(0, f"**Source:** `{source}`")
        if self.target:
            meta.append(f"**Target column:** `{self.target}`")
        if self.date_col:
            meta.append(f"**Date column:** `{self.date_col}`")
        if self.id_cols:
            meta.append("**ID columns:** " + ", ".join(f"`{c}`" for c in self.id_cols))
        L += ["  \n".join(meta), ""]
        L += ["This report shows summary statistics only. It does not show raw rows. Values of identifier, "
              "free text, and personal data columns are never shown, and category values seen fewer than "
              f"{MIN_VALUE_COUNT} times are grouped as Other.", ""]
        L += ["## Contents", ""]
        for i, (k, t, _) in enumerate(SECTION_ORDER, 1):
            anchor = re.sub(r"[^a-z0-9 ]", "", t.lower()).replace(" ", "-")
            L.append(f"{i}. [{t.split('. ', 1)[1]}](#{anchor})")
        L.append("")
        used = set()
        for k, _, _ in SECTION_ORDER:
            sec = self.sections[k]
            L += [f"## {sec.title}", "", sec.intro, ""]
            if sec.skipped:
                L += [f"*{sec.skipped}*", ""]
                continue
            for b in sec.blocks:
                kind = b[0]
                if kind == "text":
                    L += [b[1], ""]
                elif kind == "bullets":
                    L += [f"* {x}" for x in b[1]] + [""]
                elif kind == "table":
                    L += [f"**{b[1]}**", ""]
                    if b[3]:
                        L += [b[3], ""]
                    L += [self._md_table(b[2]), ""]
                elif kind == "figure":
                    stem = b[3]
                    base, n = stem, 2
                    while stem in used:
                        stem, n = f"{base}_{n}", n + 1
                    used.add(stem)
                    fname = f"{stem}.png"
                    b[2].savefig(os.path.join(plot_dir, fname), dpi=DPI)
                    L += [f"![{b[1]}]({plot_dir_name}/{fname})", ""]
        path = os.path.join(out_dir, report_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L).rstrip() + "\n")
        return path

    # ------------------------------------------------------------ rendering helpers

    def _sampling_note(self):
        if self.sampled:
            return (f"Counts, missing values, duplicates, and summary statistics use every row. Plots, correlations, "
                    f"associations, and mutual information use a random sample of {len(self.s):,} of the "
                    f"{len(self.df):,} rows (random_state={self.random_state}).")
        return "Every statistic and plot uses all rows."

    @staticmethod
    def _md_table(df):
        cols = [str(c) for c in df.columns]
        esc = lambda v: fmt(v).replace("|", "\\|").replace("\n", " ")
        out = ["| " + " | ".join(c.replace("|", "\\|") for c in cols) + " |",
               "|" + "|".join("---" for _ in cols) + "|"]
        for row in df.itertuples(index=False):
            out.append("| " + " | ".join(esc(v) for v in row) + " |")
        return "\n".join(out)

    @staticmethod
    def _display_table(df):
        try:
            return df.style.format(fmt).hide(axis="index")
        except (AttributeError, ImportError):  # pandas Styler needs jinja2; show a plain table without it
            return df.map(fmt) if hasattr(df, "map") else df.applymap(fmt)

    def _issue(self, severity, area, kind, action, column=None, detail=None, message=None):
        """Record an issue. Issues with the same severity, area, and kind are grouped into one row in the summary.

        kind: short name of the issue type, such as "Highly skewed".
        detail: the per column detail shown in the Columns cell, such as "`price` (skew 3.1)".
        message: a full sentence, used for issues that are not about one column.
        """
        if column is not None and detail is None:
            detail = f"`{column}`"
        self.issues.append(dict(severity=severity, section=area, kind=kind, column=column, detail=detail,
                                message=message or (f"{kind}: {detail}" if detail else kind), action=action))

    def _flag(self, c, text):
        self.flags.setdefault(c, []).append(text)

    def _fig(self, sec, caption, fig, stem):
        sec.figure(caption, fig, stem)
        self.figure_count += 1

    # ------------------------------------------------------------ roles

    def _infer_roles(self):
        df = self.df
        probe = df.sample(min(len(df), 50_000), random_state=self.random_state) if len(df) > 50_000 else df
        for c in df.columns:
            s, ps = df[c], probe[c]
            nun = nuniq(s)
            uniq_ratio = nun / max(s.notna().sum(), 1)
            try:
                s.head(100).map(hash)
                hashable = True
            except TypeError:
                hashable = False
            if not hashable:
                role = "other"
            elif c in self.id_cols:
                role = "id"
            elif pd.api.types.is_bool_dtype(s):
                role = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                role = "datetime"
            elif PII_NAME.search(str(c)):
                role = "pii"
            elif pd.api.types.is_numeric_dtype(s):
                role = "id" if ID_NAME.search(str(c)) and uniq_ratio > 0.9 else "numeric"
            elif is_text_like(s):
                vals = ps.dropna().astype(str).head(2000)
                if len(vals) and (vals.str.match(EMAIL_RE).mean() > 0.5 or vals.str.match(LONG_DIGITS_RE).mean() > 0.5):
                    role = "pii"
                elif ID_NAME.search(str(c)) and uniq_ratio > 0.9:
                    role = "id"
                elif len(vals) and (vals.str.len().mean() > 30 or (vals.str.len().mean() > 15 and uniq_ratio > 0.5)):
                    role = "text"
                elif uniq_ratio > 0.95 and nun > 1000:
                    role = "id"
                else:
                    role = "categorical"
            else:
                role = "other"
            self.roles[c] = role
        for c, r in list(self.roles.items()):
            if r == "datetime":
                d = self.df[c]
                if getattr(d.dt, "tz", None) is not None:
                    d = d.dt.tz_convert("UTC").dt.tz_localize(None)
                self.parsed_dates[c] = d
            elif r in ("categorical", "text", "id") and is_text_like(df[c]):
                vals = probe[c].dropna().astype(str).head(2000)
                if len(vals) and vals.str.contains(DATE_HINT_RE).mean() > 0.9:
                    if pd.to_numeric(vals, errors="coerce").notna().mean() < 0.5:
                        parsed, share = parse_dates(df[c])
                        if share >= 0.9:
                            self.parsed_dates[c] = parsed
                            self.roles[c] = "datetime_text"
        if self.date_col and self.date_col not in self.parsed_dates:
            parsed, share = parse_dates(df[self.date_col].astype(str).where(df[self.date_col].notna()))
            if share >= 0.5:
                self.parsed_dates[self.date_col] = parsed
                self.roles[self.date_col] = "datetime_text"
        if self.target:
            self.roles[self.target] = "target"

    def _convert_numeric_text(self):
        """Analyse numbers stored as text (such as '$1,234.50') as numbers. The original frame is not changed."""
        df, probe = self.raw, self.raw.head(50_000)
        found = {}
        for c, r in self.roles.items():
            if r not in ("categorical", "text") or not is_text_like(df[c]):
                continue
            vals = probe[c].dropna().astype(str).str.strip()
            if len(vals) and vals.nunique() > 12:
                cleaned = vals.str.replace(r"[,$€£%\s]", "", regex=True)
                if pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.95:
                    full = df[c].astype(str).str.strip().str.replace(r"[,$€£%\s]", "", regex=True)
                    found[c] = pd.to_numeric(full.where(df[c].notna()), errors="coerce")
        if not found:
            return
        self.df = self.raw.copy()
        for c, v in found.items():
            self.df[c] = v
            self.roles[c] = "numeric"
            self.converted.append(c)
        self.s = self.df.loc[self.s.index]

    def describe_all(self):
        """df.describe(include="all"), transposed, with values hidden where they could expose individual records."""
        if not self.roles:
            self._infer_roles()
        df = self.raw.copy()
        for c, r in self.roles.items():
            if r == "other":
                df[c] = df[c].astype(str).where(self.raw[c].notna())
        d = df.describe(include="all").T
        d.insert(0, "dtype", [str(t) for t in self.raw.dtypes])
        d["unique"] = [nuniq(self.raw[c]) for c in self.raw.columns]
        order = ["dtype", "count", "unique", "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max"]
        d = d[[c for c in order if c in d.columns] + [c for c in d.columns if c not in order]].astype(object)
        show_values = ("categorical", "boolean", "target", "datetime", "datetime_text", "numeric")
        stat_cols = [c for c in d.columns if c not in ("dtype", "count", "unique")]
        for c in d.index:
            if self.roles.get(c) not in show_values and c not in self.converted:
                for k in stat_cols:
                    if pd.notna(d.at[c, k]):
                        d.at[c, k] = "(hidden)"
            elif "top" in d.columns and pd.notna(d.at[c, "top"]):
                if pd.notna(d.at[c, "freq"]) and d.at[c, "freq"] < MIN_VALUE_COUNT:
                    d.at[c, "top"] = "(hidden)"
                else:
                    d.at[c, "top"] = short(d.at[c, "top"], 30)
        d = d.where(pd.notna(d), None)
        return d.reset_index().rename(columns={"index": "Column"})

    def cols(self, *roles):
        return [c for c, r in self.roles.items() if r in roles]

    def _target_numeric_dtype(self):
        y = self.df[self.target]
        return pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)

    def target_kind(self):
        """'continuous' or 'categorical'. Uses target_type when the user gave it, otherwise a simple rule:
        a numeric target with more than 20 unique values is continuous."""
        if not self.target:
            return None
        if self.target_type:
            return self.target_type
        return "continuous" if self._target_numeric_dtype() and self.df[self.target].nunique() > 20 else "categorical"

    def _numeric_cols(self):
        cols = self.cols("numeric")
        t = self.target
        if t and self._target_numeric_dtype() and self.target_kind() == "continuous":
            cols = [t] + cols
        return cols

    def _categorical_cols(self):
        cols = self.cols("categorical", "boolean")
        t = self.target
        if t and t not in self._numeric_cols():
            cols = [t] + cols
        return cols

    def _hashable(self, c):
        try:
            hash(tuple(self.raw[c].head(50).tolist()))
            return True
        except TypeError:
            return False

    # ------------------------------------------------------------ 1. overview

    def _sec_overview(self, sec):
        df, n = self.raw, len(self.raw)
        mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        role_names = {"numeric": "numeric", "categorical": "categorical", "datetime": "date",
                      "datetime_text": "dates stored as text", "boolean": "boolean", "id": "identifier",
                      "text": "free text", "pii": "possible personal data", "target": "target",
                      "other": "other (lists, dicts, or objects)"}
        rc = pd.Series(self.roles).value_counts()
        roles_txt = ", ".join(f"{v:,} {role_names.get(k, k)}" for k, v in rc.items())
        sec.text(f"The data has **{n:,} rows** and **{df.shape[1]:,} columns**, and uses about **{mem:,.1f} MB** "
                 f"of memory. {self._sampling_note()}")
        sec.text(f"**Columns by role:** {roles_txt}. Roles are the skill's best guess at how each column should be "
                 "treated. Identifier, free text, and possible personal data columns are excluded from value level output.")
        if self.converted:
            sec.text("These columns hold numbers stored as text and are analysed as numbers after removing symbols "
                     "such as commas and currency signs: " + ", ".join(f"`{c}`" for c in self.converted) + ".")
        if self.cols("pii"):
            for c in self.cols("pii"):
                self._flag(c, "possible personal data")
                self._issue("Medium", "Overview", "Columns that look like personal data",
                            "Confirm these columns are needed, and mask or drop them before sharing the data.", c)
        self._keys(sec)
        self._types(sec)

    def _keys(self, sec):
        df, n = self.raw, len(self.raw)
        hashable = [c for c in df.columns if self._hashable(c)]
        dup = int(df[hashable].duplicated().sum()) if hashable else 0
        self.duplicate_rows = dup
        lines =[f"**Duplicate rows:** {dup:,} rows ({dup / max(n, 1):.2%}) are exact copies of another row."]
        if dup:
            self._issue("High" if dup / n > 0.01 else "Medium", "Keys and duplicates", "Exact duplicate rows",
                        "Find out whether duplicates are real repeat events or a loading or join error, then drop them if they are errors.",
                        message=f"{dup:,} rows are exact duplicates ({dup / n:.2%}).")
        kd = df[hashable].drop_duplicates() if dup else df
        kn = len(kd)
        single = [c for c in hashable if kd[c].notna().all() and kd[c].nunique() == kn]
        pairs = []
        if not single and kn > 1:
            # Keys are rarely floating point measures, and those pair up with anything into a unique combination.
            cand = sorted([c for c in hashable if 1 < kd[c].nunique() < kn and not pd.api.types.is_float_dtype(kd[c])],
                          key=lambda c: -kd[c].nunique())[:MAX_KEY_PAIR_COLS]
            for i, a in enumerate(cand):
                for b in cand[i + 1:]:
                    sub = kd[[a, b]]
                    if sub.notna().all().all() and not sub.duplicated().any():
                        pairs.append((a, b))
        after = " after removing exact duplicates" if dup else ""
        if single:
            self.grain = {"type": "single", "columns": [single[0]], "other_unique_columns": single[1:8]}
        elif pairs:
            self.grain = {"type": "pair", "columns": list(pairs[0]), "other_unique_pairs": [list(p) for p in pairs[1:8]]}
        if single:
            lines.append(f"**Grain:** one row per `{single[0]}`. Columns that are unique for every row{after}: "
                         + ", ".join(f"`{c}`" for c in single) + ".")
        elif pairs:
            lines.append(f"**Grain:** one row per combination of `{pairs[0][0]}` and `{pairs[0][1]}`. No single column is "
                         f"unique{after}, but these pairs are: " + "; ".join(f"`{a}` + `{b}`" for a, b in pairs[:8]) + ".")
        else:
            lines.append(f"**Grain:** unclear. No single column or pair of columns uniquely identifies every row{after}.")
            self._issue("Low", "Keys and duplicates", "No unique key", "Confirm the intended grain of the data with its owner.",
                        message="No column or pair of columns uniquely identifies every row.")
        for c in self.id_cols:
            d_key = int(df[c].duplicated(keep=False).sum())
            if not d_key:
                lines.append(f"`{c}` is unique for every row.")
                continue
            d_full = int(df[hashable].duplicated(keep=False).sum()) if hashable else 0
            differ = max(d_key - d_full, 0)
            if differ == 0:
                lines.append(f"`{c}` repeats on {d_key:,} rows, but only because of exact duplicate rows.")
                self._issue("Medium", "Keys and duplicates", "ID repeats only in duplicate rows",
                            "Drop exact duplicates, then the ID becomes unique.", c, f"`{c}` ({d_key:,} rows)")
            else:
                lines.append(f"`{c}` repeats on {d_key:,} rows. {differ:,} of those rows share the ID but differ in at "
                             "least one other column, which often points to a join problem upstream.")
                self._issue("High", "Keys and duplicates", "ID column is not unique",
                            "Check whether the data should have one row per ID, and look for a many to many join.",
                            c, f"`{c}` ({differ:,} rows differ)")
            self._flag(c, "repeated IDs")
        sec.bullets(lines)

    def _types(self, sec):
        df = self.raw
        probe = df.sample(min(len(df), 50_000), random_state=self.random_state) if len(df) > 50_000 else df
        rows = []
        for c in df.columns:
            s, role = probe[c], self.roles[c]
            if role == "datetime_text":
                rows.append((c, str(df[c].dtype), "Dates stored as text", "Convert with pd.to_datetime."))
                continue
            if is_text_like(s) and role != "pii":
                vals = s.dropna().astype(str).str.strip()
                if len(vals):
                    cleaned = vals.str.replace(r"[,$€£%\s]", "", regex=True)
                    num_share = pd.to_numeric(cleaned, errors="coerce").notna().mean()
                    low = set(vals.str.lower().unique()[:10])
                    if num_share >= 0.95 and vals.nunique() > 2:
                        rows.append((c, str(df[c].dtype), "Numbers stored as text",
                                     "Strip symbols such as commas or currency signs, then convert with pd.to_numeric."))
                    elif 1 <= len(low) <= 2 and low <= {"y", "n", "yes", "no", "true", "false", "t", "f", "0", "1"}:
                        rows.append((c, str(df[c].dtype), "Boolean values stored as text", "Map to True and False."))
                    types = s.dropna().map(lambda v: type(v).__name__).value_counts()
                    if pd.api.types.is_object_dtype(df[c]) and len(types) > 1:
                        mix = ", ".join(f"{t} {k / types.sum():.0%}" for t, k in types.items())
                        rows.append((c, str(df[c].dtype), f"Mixed Python types ({mix})", "Cast the column to one type."))
            if pd.api.types.is_integer_dtype(df[c]) and role == "id":
                rows.append((c, str(df[c].dtype), "Integer identifier", "Treat as an ID, not as a numeric feature."))
            if pd.api.types.is_numeric_dtype(df[c]) and role == "numeric":
                nun = df[c].nunique()
                if nun == 2 and set(df[c].dropna().unique()) <= {0, 1}:
                    self._flag(c, "binary 0 or 1")
                elif 2 < nun <= 12 and pd.api.types.is_integer_dtype(df[c]):
                    self._flag(c, f"only {nun} distinct values, may be a code")
        if rows:
            sec.text("**Type problems:** " + "; ".join(f"`{c}` ({f[0].lower() + f[1:]})" for c, _, f, _ in rows)
                     + ". Each one is also in the Flags column of the profile tables.")
            for c, _, finding, fix in rows:
                kind = finding.split(" (")[0]
                self._flag(c, kind[0].lower() + kind[1:])
                if kind in ("Numbers stored as text", "Dates stored as text", "Mixed Python types"):
                    self._issue("Medium", "Data types", kind, fix, c)
        else:
            self.passed.append("no columns with an obviously wrong type")

        found_const = False
        for c in df.columns:
            s = df[c]
            if nuniq(s, dropna=False) <= 1:
                found_const = True
                self._flag(c, "constant")
                self._issue("Medium", "Data types", "Constant column", "Drop it, since it carries no information.", c)
            else:
                top = (s if self.roles[c] != "other" else s.astype(str)).value_counts(dropna=False, normalize=True).iloc[0]
                if top >= NEAR_CONSTANT:
                    found_const = True
                    self._flag(c, f"near constant ({top:.1%} one value)")
                    self._issue("Low", "Data types", f"Near constant column (one value is {NEAR_CONSTANT:.0%} or more of rows)",
                                "Drop it, or keep it only if the rare values matter.", c, f"`{c}` ({top:.1%})")
        if not found_const:
            self.passed.append("no constant or near constant columns")

    # ------------------------------------------------------------ 2. profiles

    def _sec_profiles(self, sec):
        df, n = self.df, max(len(self.df), 1)
        num = self._numeric_cols()
        if num:
            rows = []
            for c in num:
                full = df[c]
                s = full.dropna().astype(float)
                dtype = str(self.raw[c].dtype) + (" (numbers as text)" if c in self.converted else "")
                if s.empty:
                    rows.append({"Column": c, "Dtype": dtype, "Count": 0, "Missing %": 100.0})
                    continue
                q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                iqr = q[0.75] - q[0.25]
                out = int(((s < q[0.25] - 1.5 * iqr) | (s > q[0.75] + 1.5 * iqr)).sum()) if iqr > 0 else 0
                skew = s.skew()
                r = {"Column": c, "Dtype": dtype, "Count": len(s), "Missing %": full.isna().mean() * 100,
                     "Unique": s.nunique(), "Mean": s.mean(), "Std": s.std(), "Min": s.min(), "P1": q[0.01],
                     "P5": q[0.05], "P25": q[0.25], "Median": q[0.5], "P75": q[0.75], "P95": q[0.95],
                     "P99": q[0.99], "Max": s.max(), "Skew": skew, "Zero %": (s == 0).mean() * 100,
                     "Negative %": (s < 0).mean() * 100, "Outlier %": out / len(s) * 100}
                rows.append(r)
                if c == self.target:
                    self._flag(c, "target")
                    continue
                if pd.notna(skew) and abs(skew) > 2:
                    self._flag(c, "highly skewed")
                    self._issue("Low", "Numeric columns", "Highly skewed (absolute skew above 2)",
                                "Consider a log or rank transform for linear models.", c, f"`{c}` ({skew:.1f})")
                if r["Outlier %"] > 5:
                    self._flag(c, f"{r['Outlier %']:.0f}% outliers")
                    self._issue("Low", "Numeric columns", "Many outliers (over 5% of rows by the IQR rule)",
                                "Check whether extreme values are real before capping them.", c, f"`{c}` ({r['Outlier %']:.1f}%)")
                if 0 < r["Negative %"] < 2:
                    self._flag(c, "few negatives")
                    self._issue("Low", "Numeric columns", "Small share of negative values (under 2%)",
                                "Check whether negatives are valid, such as refunds, or errors.", c, f"`{c}` ({r['Negative %']:.2f}%)")
            t = self._sort_profile(pd.DataFrame(rows).round(4), ["Missing %", "Outlier %"])
            self.profile_tables.append(t)
            sec.table("Numeric columns", t,
                      "Sorted by Missing %, then Outlier %, lowest first. "
                      "P1 to P99 are percentiles, and P25, Median, and P75 are the quartiles from df.describe(). Outlier % "
                      "uses the 1.5 × IQR rule. An absolute skew above 2 means a strongly skewed distribution.")

        cats = self._categorical_cols()
        if cats:
            rows = []
            for c in cats:
                full = df[c]
                vc = full.value_counts(dropna=True)
                nun = len(vc)
                share = vc / max(full.notna().sum(), 1)
                top_ok = nun and vc.iloc[0] >= MIN_VALUE_COUNT
                rare = int((share < RARE_SHARE).sum())
                rows.append({"Column": c, "Dtype": str(self.raw[c].dtype), "Count": int(full.notna().sum()),
                             "Missing %": full.isna().mean() * 100, "Unique": nun, "Unique %": nun / n * 100,
                             "Most common value": (short(vc.index[0], 30) if top_ok else "(hidden)") if nun else "",
                             "Most common count": int(vc.iloc[0]) if nun else 0,
                             "Most common share %": share.iloc[0] * 100 if nun else np.nan,
                             "Rare categories (<1%)": rare,
                             "Rows in rare categories %": share[share < RARE_SHARE].sum() * 100})
                if c == self.target:
                    self._flag(c, "target")
                    continue
                if nun > HIGH_CARDINALITY:
                    self._flag(c, "high cardinality")
                    self._issue("Medium", "Categorical columns", f"High cardinality (more than {HIGH_CARDINALITY} unique values)",
                                "Use target encoding, frequency encoding, or hashing instead of one hot encoding.",
                                c, f"`{c}` ({nun:,})")
                if rare > 10:
                    self._flag(c, f"{rare:,} rare categories")
                    self._issue("Low", "Categorical columns", "Many rare categories (more than 10 under 1% of rows)",
                                "Group rare categories into an Other bucket.", c, f"`{c}` ({rare:,})")
            self._spelling(cats)
            t = self._sort_profile(pd.DataFrame(rows).round(2), ["Missing %", "Unique %"])
            self.profile_tables.append(t)
            sec.table("Categorical columns", t,
                      "Sorted by Missing %, then Unique %, lowest first. "
                      f"The most common value is hidden when it appears fewer than {MIN_VALUE_COUNT} times. Rare "
                      "categories each cover under 1% of non missing rows.")
            if getattr(self, "_incons", None):
                sec.table("Values that differ only by case, spaces, or punctuation",
                          pd.DataFrame(self._incons[:30], columns=["Column", "Variants", "Number of variants", "Rows affected"]))

        if self.parsed_dates:
            self._dates_table(sec)
        else:
            self.passed.append("no date columns found")

        texts = self.cols("text")
        others = self.cols("id", "pii", "other")
        if texts:
            trows = []
            for c in texts:
                v = df[c].dropna().astype(str)
                L, W = v.str.len(), v.str.split().str.len()
                trows.append({"Column": c, "Count": len(v), "Missing %": df[c].isna().mean() * 100,
                              "Unique": nuniq(df[c]), "Min length": L.min(), "Median length": L.median(),
                              "P95 length": L.quantile(0.95), "Max length": L.max(), "Mean words": W.mean(),
                              "Empty or blank %": (v.str.strip() == "").mean() * 100})
            t = pd.DataFrame(trows).round(2)
            self.profile_tables.append(t)
            sec.table("Free text columns (length statistics only)", t)
        if others:
            names = {"id": "identifier", "pii": "possible personal data", "other": "lists, dicts, or objects"}
            t = pd.DataFrame([{"Column": c, "Role": names[self.roles[c]], "Dtype": str(self.raw[c].dtype),
                               "Count": int(df[c].notna().sum()), "Missing %": round(df[c].isna().mean() * 100, 2),
                               "Unique": nuniq(df[c])} for c in others])
            self.profile_tables.append(t)
            sec.table("Identifier, personal data, and other columns (values not shown)", t)

    def _sort_profile(self, t, keys):
        """Sort a profile table ascending by the given columns, so the cleanest columns come first.
        The target row, if present, stays at the top. Missing values in a sort column go last."""
        keys = [k for k in keys if k in t.columns]
        if not keys or t.empty:
            return t
        t = t.sort_values(keys, ascending=True, na_position="last", kind="mergesort")
        if self.target in set(t["Column"]):
            t = pd.concat([t[t["Column"] == self.target], t[t["Column"] != self.target]])
        return t.reset_index(drop=True)

    def _spelling(self, cats):
        df, incons = self.df, []
        for c in cats:
            s = df[c].dropna()
            if not is_text_like(s) or s.nunique() > 5000:
                continue
            vc = s.astype(str).value_counts()
            norm = vc.index.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
            groups = pd.Series(vc.index, index=norm).groupby(level=0).agg(list)
            for key, variants in groups.items():
                if len(variants) > 1 and key:
                    shown = [short(v, 25) for v in variants if vc[v] >= MIN_VALUE_COUNT]
                    hidden_n = len(variants) - len(shown)
                    desc = ", ".join(f"'{v}'" for v in shown[:5])
                    if hidden_n:
                        desc += f" and {hidden_n} rare variant(s)"
                    incons.append((c, desc, len(variants), int(vc[variants].sum())))
        incons.sort(key=lambda t: -t[3])
        self._incons = incons
        if not incons:
            self.passed.append("no values that differ only by case, spaces, or punctuation")
        for c in dict.fromkeys(t[0] for t in incons):
            k = sum(1 for t in incons if t[0] == c)
            self._flag(c, "inconsistent spelling")
            self._issue("Medium", "Categorical columns", "Values written in more than one way",
                        "Normalise case, spaces, and punctuation, or map variants to one value.", c, f"`{c}` ({k})")

    def _dates_table(self, sec):
        now = pd.Timestamp.now()
        rows = []
        for c, d in self.parsed_dates.items():
            nn = d.dropna()
            if nn.empty:
                continue
            future = (nn > now).mean() * 100
            old = (nn < pd.Timestamp("1900-01-02")).mean() * 100
            placeholder = nn.isin(PLACEHOLDER_DATES).mean() * 100
            rows.append({"Column": c, "Stored as": "text" if self.roles.get(c) == "datetime_text" else str(self.raw[c].dtype),
                         "Count": len(nn), "Missing %": d.isna().mean() * 100, "Unique": nn.nunique(),
                         "Earliest": nn.min(), "Latest": nn.max(), "Span (days)": (nn.max() - nn.min()).days,
                         "Future dates %": future, "Before 1900 %": old, "Placeholder dates %": placeholder})
            if future > 0:
                self._flag(c, "future dates")
                self._issue("Medium", "Dates", "Dates in the future",
                            "Check for data entry errors or placeholder dates, and make sure future information cannot leak into features.",
                            c, f"`{c}` ({future:.2f}%)")
            if placeholder > 0.5 or old > 0:
                self._flag(c, "placeholder dates")
                self._issue("Medium", "Dates", "Placeholder or impossible dates", "Treat these values as missing.",
                            c, f"`{c}` ({placeholder + old:.2f}%)")
        if rows:
            t = rnd(pd.DataFrame(rows), 2)
            self.profile_tables.append(t)
            sec.table("Date columns", t, "Placeholder dates are 1900-01-01, 1970-01-01, and 9999-12-31.")

    # ------------------------------------------------------------ 3. missing

    def _structural_missing(self, sec, df, partial):
        """Group columns that go missing together, look for a rule that explains each group, and give a verdict."""
        cols = [c for c in partial if c != self.target]
        cols = sorted(cols, key=lambda c: -df[c].isna().mean())[:MAX_MISSING_COLS]
        if not cols:
            return
        ind = df[cols].isna()
        corr = ind.astype(float).corr()
        parent = {c: c for c in cols}

        def find(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c

        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and r >= GROUP_CORR:
                    parent[find(a)] = find(b)
        members = {}
        for c in cols:
            members.setdefault(find(c), []).append(c)
        groups = sorted(members.values(), key=lambda g: (-len(g), -ind[g[0]].mean(), g[0]))

        # target rates when the group is missing or present
        y, ylabel = None, None
        if self.target and self.target in df.columns:
            yt = df[self.target]
            vals = sorted(yt.dropna().unique().tolist(), key=lambda v: str(v))
            if self.target_kind() == "continuous":
                y, ylabel = pd.to_numeric(yt, errors="coerce"), f"Mean `{self.target}`"
            elif len(vals) == 2:
                pos = vals[-1]
                y, ylabel = (yt == pos).astype(float).where(yt.notna()), f"Share `{self.target}` = {fmt(pos)}"

        # rule search on a sample: bucket every other column, find buckets that are almost all missing or present
        samp = df.sample(min(len(df), RULE_SAMPLE), random_state=0) if len(df) > RULE_SAMPLE else df
        buckets = {}
        for c in df.columns:
            if c == self.target or self.roles.get(c) in ("id", "text", "pii", "datetime"):
                continue
            v = samp[c]
            try:
                if pd.api.types.is_numeric_dtype(v) and nuniq(v) > 20:
                    codes_s = pd.qcut(v.rank(method="first"), 10, labels=False, duplicates="drop")
                    edges = v.groupby(codes_s).agg(["min", "max"])
                    labels = [f"{fmt(r['min'])} to {fmt(r['max'])}" for _, r in edges.iterrows()]
                    codes = codes_s.fillna(-1).astype(int).to_numpy()
                else:
                    codes, uniq = pd.factorize(v.astype(str).where(v.notna()))
                    labels = [str(u) for u in uniq]
            except (TypeError, ValueError):
                continue
            buckets[c] = (codes, labels)

        rows, rules, spec_groups = [], [], []
        for gi, g in enumerate(groups, 1):
            gm = ind[g]
            any_m, all_m = gm.any(axis=1), gm.all(axis=1)
            overall = float(ind[g[0]].mean())
            sm = samp[g[0]].isna().to_numpy().astype(float)
            best = None
            for c, (codes, labels) in buckets.items():
                if c in g:
                    continue
                ok = codes >= 0
                n = np.bincount(codes[ok], minlength=len(labels))
                m = np.bincount(codes[ok], weights=sm[ok], minlength=len(labels))
                tot_m = float(sm.sum())
                for k in np.where(n >= max(30, RULE_MIN_ROWS * len(samp)))[0]:
                    rate = m[k] / n[k]
                    purity = max(rate, 1 - rate)
                    if purity >= RULE_PURITY and abs(rate - overall) >= RULE_LIFT:
                        # share of the rows the rule speaks about (the missing rows, or the present rows) that it accounts for
                        share = (m[k] / tot_m if rate >= 0.5 else (n[k] - m[k]) / (len(samp) - tot_m)) if 0 < tot_m < len(samp) else 0.0
                        cand = (purity, n[k], c, labels[k], rate, share)
                        key = (cand[5], cand[0])
                        if best is None or key > (best[5], best[0]):
                            best = cand
            if best and best[0] >= RULE_STRUCTURAL and best[5] >= RULE_SHARE:
                verdict = "Likely structural (explained by a rule)"
            elif len(g) >= 2:
                verdict = "Probably structural (shared pattern)"
            elif best:
                verdict = "Partly explained by a rule"
            else:
                verdict = "No pattern found"
            row = {"Group": f"G{gi}",
                   "Columns": ", ".join(g[:6]) + (f" and {len(g) - 6} more" if len(g) > 6 else ""),
                   "Number of columns": len(g),
                   "Missing in all %": all_m.mean() * 100, "Missing in any %": any_m.mean() * 100}
            tr = {}
            if y is not None:
                a, b = y[any_m].mean(), y[~any_m].mean()
                row[f"{ylabel} when missing"], row[f"{ylabel} when present"] = a, b
                tr = {"target_when_missing": round(float(a), 4), "target_when_present": round(float(b), 4)}
            row["Verdict"] = verdict
            rows.append(row)
            if best:
                purity, n_k, c, lab, rate, share = best
                rules.append({"Group": f"G{gi}", "Rule": f"`{short(c, 30)}` = {short(lab, 30)}",
                              "Rows covered %": n_k / len(samp) * 100,
                              "Missing inside the rule %": rate * 100, "Missing overall %": overall * 100,
                              "Missing (or present) rows explained %": share * 100})
            spec_groups.append({"columns": [str(c) for c in g], "rows_missing_all_pct": round(float(all_m.mean() * 100), 3),
                                "rows_missing_any_pct": round(float(any_m.mean() * 100), 3), **tr, "verdict": verdict,
                                "rule": (f"{best[2]} = {best[3]}" if best else None),
                                "rule_share_explained_pct": (round(float(best[5] * 100), 1) if best else None)})
            if len(g) >= 2:
                for c in g:
                    self._issue("Medium", "Missing values", "Structural missing values (columns missing as a group)",
                                "Treat each group as one source: add one missing indicator per group, and impute or drop the group together.",
                                c, f"`{c}` (group G{gi})")
        self.missing_groups = spec_groups

        shown = rows[:MAX_MISSING_GROUPS]
        tbl = pd.DataFrame(shown).round(3)
        more = f" Showing {len(shown):,} of {len(rows):,} groups." if len(rows) > len(shown) else ""
        sec.text("**Structural missing values.** Missing values are structural when a value is absent by design, for "
                 "example when a field is not collected or does not apply for some rows. Columns that are always "
                 "missing in the same rows usually share a source, so they are grouped below.")
        sec.table("Missingness groups", tbl,
                  f"Columns whose missing indicators correlate at {GROUP_CORR:g} or more (see the note above on how this is "
                  "calculated) are put in one group, and a column with no partner is a group of one. Missing in all is the "
                  "share of rows where every column in the group is missing, and Missing in any is the share where at least "
                  "one is. When a target is set, the last columns compare the target when the group is missing and when it "
                  "is present, which shows whether a missing indicator would carry signal. The verdict is evidence, not "
                  "proof: a group that is missing together is probably structural, but only the data owner can confirm "
                  f"why.{more}")
        if rules:
            sec.table("Rules that explain the missingness", pd.DataFrame(rules).round(2),
                      f"For each group, every other column was split into its values (numeric columns into 10 equal sized "
                      f"ranges), and the table shows the largest value that covers at least {RULE_MIN_ROWS:.0%} of rows and "
                      f"is almost always missing or almost always present ({RULE_PURITY:.0%} or more) while the overall "
                      f"missing rate is at least {RULE_LIFT:.0%} different. Of the rules that qualify, the one that accounts for the most rows is shown. Explained is the share of the "
                      f"group's missing rows (or, for a rule where the column is almost always present, its present rows) that fall "
                      f"inside the rule. A purity of {RULE_STRUCTURAL:.0%} or more that explains at least {RULE_SHARE:.0%} of those "
                      f"rows counts as a rule that explains the group, and a smaller share is called partly explained. Rules are searched on a sample of up to {RULE_SAMPLE:,} rows.")
        else:
            sec.text("No single value of another column explains why a group is missing.")

    def _sec_missing(self, sec):
        df = self.raw
        miss = (df.isna().mean() * 100).sort_values(ascending=False)
        with_missing = miss[miss > 0]
        if with_missing.empty:
            sec.text("No column has missing values according to df.isna().")
            self.passed.append("no missing values")
        else:
            high, mid = (with_missing >= 50).sum(), ((with_missing >= 5) & (with_missing < 50)).sum()
            sec.text(f"**{len(with_missing):,} of {df.shape[1]:,} columns** have missing values: {high:,} are 50% or "
                     f"more missing, {mid:,} are between 5% and 50%, and {len(with_missing) - high - mid:,} are under 5%. "
                     "The exact percentage for every column is in the Missing % column of the profile tables.")
            show = with_missing.head(40)
            fig, axes = new_fig(9, max(2.4, 0.26 * len(show) + 1.0))
            ax = axes[0, 0]
            hbar(ax, [short(c, 30) for c in show.index], show.values, [PRIMARY] * len(show), fmt=pct_label)
            for x, lab in ((5, "5%"), (50, "50%")):
                ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
            ax.set_xlim(0, 108)
            ax.set_xlabel("Missing (%)")
            ax.set_ylabel("")
            ax.set_title("Missing values by column" + (" (top 40)" if len(with_missing) > 40 else ""), loc="left")
            self._fig(sec, "Missing values by column", fig, "missing_by_column")
            for c, v in with_missing.items():
                if v >= 50:
                    self._flag(c, "mostly missing")
                    self._issue("High", "Missing values", "Mostly missing (50% or more)",
                                "Decide whether to drop it, or keep a missing indicator if the missingness itself is informative.",
                                c, f"`{c}` ({v:.1f}%)")
                elif v >= 5:
                    self._issue("Medium", "Missing values", "Partly missing (5% to 50%)",
                                "Choose an imputation strategy and consider adding a missing indicator.", c, f"`{c}` ({v:.1f}%)")

        rows = []
        for c in df.columns:
            s = df[c]
            if is_text_like(s):
                low = s.dropna().astype(str).str.strip().str.lower()
                hits = low[low.isin(DISGUISED_TOKENS)]
                if len(hits):
                    toks = hits.value_counts().head(5)
                    rows.append((c, ", ".join(f"'{t}'" for t in toks.index), len(hits), len(hits) / len(df) * 100))
            elif pd.api.types.is_numeric_dtype(s) and self.roles[c] in ("numeric", "target") and s.notna().any():
                vc = s.value_counts()
                for v in NUMERIC_SENTINELS:
                    if v in vc.index and vc[v] / len(df) >= 0.01 and (v == s.min() or v == s.max()):
                        rows.append((c, fmt(v), int(vc[v]), vc[v] / len(df) * 100))
        if rows:
            sec.table("Possible disguised missing values",
                      pd.DataFrame(rows, columns=["Column", "Values found", "Rows", "Share of rows %"]).round(2),
                      "These values are not caught by df.isna(), but often mean the value is missing. Numeric values are "
                      "listed only when they are the column's minimum or maximum and cover at least 1% of rows.")
            for c, vals, cnt, share in rows:
                self._flag(c, "disguised missing values")
                self._issue("Medium" if share >= 1 else "Low", "Missing values", "Disguised missing values",
                            "Replace them with NaN before computing statistics or imputing.", c, f"`{c}` ({share:.1f}%)")
        else:
            self.passed.append("no disguised missing values")

        partial = [c for c in df.columns if 0 < df[c].isna().mean() < 1]
        if len(partial) >= 3:
            cols = sorted(partial, key=lambda c: -df[c].isna().mean())[:25]
            corr = df[cols].isna().astype(float).corr()
            k = len(cols)
            fig, axes = new_fig(min(11, 3.0 + 0.42 * k), min(9.5, 2.4 + 0.36 * k))
            ax = axes[0, 0]
            heatmap(ax, corr, diverging=True, annot=k <= 12)
            ax.set_title("Correlation of missing value indicators", loc="left")
            sec.text("Values near 1 mean two columns are usually missing in the same rows, which often means they come "
                     "from the same source table or join.")
            self._fig(sec, "Correlation of missing value indicators", fig, "missing_correlation")
            all_cols = sorted(partial, key=lambda c: -df[c].isna().mean())[:MAX_MISSING_COLS]
            ind = df[all_cols].isna()
            corr_all = ind.astype(float).corr()
            pairs = []
            for i, a in enumerate(all_cols):
                for b in all_cols[i + 1:]:
                    r = corr_all.loc[a, b]
                    if pd.notna(r) and r >= MISSING_CORR:
                        pairs.append((a, b, r, ind[a].mean() * 100, ind[b].mean() * 100, (ind[a] & ind[b]).mean() * 100))
            if pairs:
                pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
                shown = pairs[:MAX_MISSING_PAIRS]
                tbl = pd.DataFrame(shown, columns=["Column A", "Column B", "Missing correlation",
                                                   "Missing A %", "Missing B %", "Missing in both %"]).round(3)
                more = f" Showing the {len(shown):,} strongest of {len(pairs):,} pairs." if len(pairs) > len(shown) else ""
                sec.table(f"Columns that are missing together (correlation {MISSING_CORR:g} or more)", tbl,
                          "How this is calculated: for each column, every row is turned into 1 if the value is missing "
                          "and 0 if it is present. The value itself is not used, so numeric and categorical columns are "
                          "compared in the same way. The correlation is the Pearson correlation of these 0 or 1 "
                          "indicators over all rows, which is also called the phi coefficient. A value of 1 means two "
                          "columns are missing in exactly the same rows, and 0 means their missingness is unrelated. "
                          "Columns with no missing values are left out, because their indicator never changes. "
                          f"Up to {MAX_MISSING_COLS} columns with the most missing values are compared.{more}")
            else:
                self.passed.append(f"no column pairs with a missingness correlation of {MISSING_CORR:g} or more")

        self._structural_missing(sec, df, partial)

        dcol = self._primary_date()
        if dcol and not with_missing.empty:
            dates = self.parsed_dates[dcol]
            freq = self._freq(dates)
            top = [c for c in with_missing.index if c != dcol][:6]
            if top:
                grp = df[top].isna().groupby(dates.dt.to_period(freq)).mean() * 100
                if len(grp) >= 2:
                    fig, axes = new_fig(9, 3.4)
                    ax = axes[0, 0]
                    long = grp.reset_index(names="period").melt(id_vars="period", var_name="column", value_name="missing")
                    long["period"] = long["period"].dt.to_timestamp()
                    long["column"] = long["column"].map(lambda c: short(c, 25))
                    sns.lineplot(data=long, x="period", y="missing", hue="column", ax=ax, linewidth=1.6,
                                 palette=CATEGORICAL[:len(top)])
                    ax.set_ylabel("Missing (%)")
                    ax.set_xlabel("")
                    ax.set_title(f"Missing values over time by {self._freq_name(freq)} of {short(dcol, 25)}", loc="left")
                    date_axis(ax)
                    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
                    self._fig(sec, "Missing values over time", fig, "missing_over_time")

    # ------------------------------------------------------------ 4. distributions

    def _sec_distributions(self, sec, top_n=None, log_scale=None):
        log_scale = self.log_scale if log_scale is None else bool(log_scale)
        n_plots = self.max_plots if top_n is None else top_n
        rank = {c: i for i, c in enumerate(self.target_rank)} if top_n is not None else {}
        did = False
        num = [c for c in self._numeric_cols() if self.s[c].dropna().nunique() > 1]
        if num:
            did = True
            chosen = num
            if len(num) > n_plots or (top_n is not None and rank):
                def score(c):
                    v = self.s[c].dropna().astype(float)
                    return abs(v.skew()) if len(v) > 2 else 0
                if rank:
                    order = sorted(num, key=lambda c: (c != self.target, rank.get(c, len(rank)), -score(c)))
                    basis = "chosen by their association with the target"
                else:
                    order = sorted(num, key=lambda c: (c != self.target, -score(c)))
                    basis = "chosen by skew"
                keep = set(order[:n_plots])
                chosen = [c for c in order if c in keep] if rank else [c for c in num if c in keep]
                if len(chosen) < len(num):
                    sec.text(f"Histograms are shown for {len(chosen)} of {len(num)} numeric columns, {basis}. "
                             "Every column is in the profile table.")
            self._numeric_grid(sec, chosen, log_scale=log_scale)
        cats = [c for c in self._categorical_cols() if 1 < self.s[c].nunique() <= 5000 and c != self.target]
        if cats:
            did = True
            if rank:
                cats = sorted(cats, key=lambda c: rank.get(c, len(rank)))
            chosen = cats[:n_plots]
            if len(cats) > len(chosen):
                how = "the top" if rank else "the first"
                basis = ", chosen by their association with the target" if rank else ""
                sec.text(f"Top values are shown for {how} {len(chosen)} of {len(cats)} categorical columns{basis}.")
            self._categorical_grid(sec, chosen)
        if self.parsed_dates:
            did = True
            self._rows_over_time(sec)
        if not did:
            sec.skipped = "There are no numeric, categorical, or date columns to plot."

    def _numeric_grid(self, sec, cols, log_scale=False):
        ncol = min(3, len(cols))
        nrow = math.ceil(len(cols) / ncol)
        fig = Figure(figsize=(4.4 * ncol, 2.9 * nrow + 0.5), dpi=DPI, layout="constrained")
        subs = np.atleast_1d(fig.subfigures(nrow, ncol, wspace=0.03, hspace=0.04)).ravel()
        logs = 0
        for sf, c in zip(subs, cols):
            v = self.s[c].dropna().astype(float)
            sk = v.skew()
            log = log_scale and v.min() >= 0 and sk > 2
            slog = log_scale and not log and abs(sk) > 2 and v.min() < 0 < v.max()
            if log:
                v, logs = np.log1p(v), logs + 1
            elif slog:
                v, logs = np.sign(v) * np.log1p(v.abs()), logs + 1
            hv = v
            axb, axh = sf.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 4]))
            sns.boxplot(x=hv, ax=axb, color=PRIMARY_LIGHT, linewidth=0.9, fliersize=1.2, width=0.6,
                        flierprops=dict(markerfacecolor=MUTED, markeredgecolor=MUTED, alpha=0.4),
                        medianprops=dict(color=ACCENT, linewidth=1.6))
            axb.axis("off")
            bins = min(50, max(10, int(np.sqrt(hv.nunique()) * 2)))
            sns.histplot(x=hv, ax=axh, bins=bins, color=PRIMARY, edgecolor="white", linewidth=0.4, alpha=0.9)
            axh.set_ylabel("Rows")
            axh.set_xlabel("")
            axh.yaxis.set_major_formatter(THOUSANDS)
            axh.xaxis.set_major_formatter(THOUSANDS)
            axb.set_title(short(c, 34) + (" (log1p)" if log else " (signed log)" if slog else ""), loc="left", fontsize=10, fontweight="bold")
        for sf in subs[len(cols):]:
            sf.set_facecolor("none")
        note = (" Strongly skewed columns are plotted on a log scale: log1p means log(1 + x), and signed log means "
                "sign(x) × log(1 + |x|) for columns that also have negative values." if logs else "")
        if not logs and any(abs(self.s[c].dropna().astype(float).skew()) > 2 for c in cols):
            note = (" Every column is plotted on its original scale with all of its values. Pass log_scale=True to plot "
                    "strongly skewed columns on a log axis instead.")
        sec.text("Each panel shows a box plot above a histogram, and the orange line is the median. The profile table has "
                 "the exact statistics." + note)
        self._fig(sec, "Numeric distributions", fig, "numeric_distributions")

    def _categorical_grid(self, sec, cols):
        ncol = min(3, len(cols))
        nrow = math.ceil(len(cols) / ncol)
        fig, axes = new_fig(4.4 * ncol, 3.1 * nrow + 0.3, ncols=ncol, nrows=nrow, grid_axis="x")
        for ax, c in zip(axes.flat, cols):
            vc = self.s[c].value_counts(dropna=False)
            vc.index = ["(missing)" if pd.isna(i) else short(i, 22) for i in vc.index]
            vc = vc.groupby(level=0, sort=False).sum().sort_values(ascending=False)
            keep = vc[vc >= MIN_VALUE_COUNT].drop(["(missing)"], errors="ignore").head(10)
            extra = vc.drop(keep.index)
            miss = extra.get("(missing)", 0)
            other = extra.sum() - miss
            if other > 0:
                keep["Other"] = other
            if miss > 0:
                keep["(missing)"] = miss
            share = keep / vc.sum() * 100
            colors = [MUTED if i in ("Other", "(missing)") else PRIMARY for i in share.index]
            hbar(ax, [str(i) for i in share.index], share.values, colors, fontsize=7)
            ax.set_xlim(0, min(100, share.max() * 1.25 + 2))
            ax.set_xlabel("Share of rows (%)", fontsize=8)
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{short(c, 30)} ({self.s[c].nunique():,} values)", loc="left", fontsize=10, fontweight="bold")
        for ax in list(axes.flat)[len(cols):]:
            ax.axis("off")
        sec.text("Each panel shows the 10 most common values as a share of rows. Grey bars group the remaining values "
                 f"(Other) and missing values. Values seen fewer than {MIN_VALUE_COUNT} times are always in Other.")
        self._fig(sec, "Categorical top values", fig, "categorical_top_values")

    def _rows_over_time(self, sec):
        dcol = self._primary_date()
        d = self.parsed_dates[dcol].dropna()
        if d.empty:
            return
        freq = self._freq(d)
        counts = d.dt.to_period(freq).value_counts().sort_index()
        full = pd.period_range(counts.index.min(), counts.index.max(), freq=freq)
        counts = counts.reindex(full, fill_value=0)
        name = self._freq_name(freq)
        gaps = counts[counts == 0]
        med = counts.rolling(5, center=True, min_periods=3).median()
        drops = counts[(counts > 0) & (counts < 0.5 * med)]
        fig, axes = new_fig(10, 3.2)
        ax = axes[0, 0]
        x = counts.index.to_timestamp()
        sns.lineplot(x=x, y=counts.values, ax=ax, color=PRIMARY, linewidth=1.8)
        ax.fill_between(x, counts.values, color=PRIMARY, alpha=0.12)
        if len(gaps):
            ax.scatter(gaps.index.to_timestamp(), gaps.values, color=ALERT, s=16, zorder=3, label="No rows")
            ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel("Rows")
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set_title(f"Rows per {name} by {short(dcol, 30)}", loc="left")
        date_axis(ax)
        self._fig(sec, f"Rows per {name}", fig, "rows_over_time")
        notes = [f"Row counts are grouped by {name} of `{dcol}`, over {len(counts):,} periods."]
        if len(gaps):
            notes.append(f"{len(gaps):,} {name}(s) have no rows at all, for example "
                         + ", ".join(str(p) for p in gaps.index[:5]) + ".")
            self._issue("Medium", "Dates", f"Empty {name}s inside the date range",
                        "Check whether data is missing for those periods before training on a continuous time range.",
                        dcol, f"`{dcol}` ({len(gaps):,})")
        if len(drops):
            notes.append(f"{len(drops):,} {name}(s) have less than half the usual number of rows, for example "
                         + ", ".join(str(p) for p in drops.index[:5]) + ".")
            self._issue("Low", "Dates", f"{name.capitalize()}s with unusually low row counts",
                        "Check whether those periods are incomplete loads.", dcol, f"`{dcol}` ({len(drops):,})")
        if len(counts) > 3 and counts.iloc[-1] < 0.5 * counts.iloc[-4:-1].median():
            notes.append(f"The last {name} has far fewer rows than the ones before it, so it may be incomplete.")
        sec.bullets(notes)

    # ------------------------------------------------------------ dates helpers

    def _primary_date(self):
        if self.date_col and self.date_col in self.parsed_dates:
            return self.date_col
        return next(iter(self.parsed_dates), None)

    @staticmethod
    def _freq(dates):
        span = (dates.max() - dates.min()).days if dates.notna().any() else 0
        return "M" if span > 3 * 365 else ("W" if span > 90 else "D")

    @staticmethod
    def _freq_name(freq):
        return {"M": "month", "W": "week", "D": "day"}[freq]

    # ------------------------------------------------------------ 5. relationships

    def _sec_relationships(self, sec):
        """Compute the correlation matrices once, record issues, then lay out tables and the 1x2 heatmap figure."""
        s = self.s
        self.spearman, self.cramers, self._date_order = None, None, None
        self._rel_note = None
        num = [c for c in self.cols("numeric") if s[c].nunique() > 1]
        if len(num) > MAX_CORR_COLS:
            num = sorted(num, key=lambda c: s[c].isna().mean())[:MAX_CORR_COLS]
            self._rel_note = f"The Spearman matrix is limited to the {MAX_CORR_COLS} most complete numeric columns."
        if len(num) >= 2:
            self.spearman = s[num].corr(method="spearman")
        cats = [c for c in self.cols("categorical", "boolean") if 2 <= s[c].nunique() <= 50][:MAX_CRAMERS_COLS]
        if len(cats) >= 2:
            vals = pd.DataFrame(np.nan, index=cats, columns=cats)
            for i, a in enumerate(cats):
                vals.loc[a, a] = 1.0
                for b in cats[i + 1:]:
                    v = cramers_v(s[a].astype(str), s[b].astype(str))
                    vals.loc[a, b] = vals.loc[b, a] = v
            self.cramers = vals.astype(float)

        ct, vt = self.corr_threshold, self.cramers_threshold
        pairs = self.correlated_pairs(ct)
        if pairs is not None and len(pairs):
            partners = {}
            for a, b in zip(pairs["Column A"], pairs["Column B"]):
                partners.setdefault(a, []).append(b)
                partners.setdefault(b, []).append(a)
            for c, others in partners.items():
                self._flag(c, f"highly correlated with {others[0]}" if len(others) == 1
                           else f"highly correlated with {len(others)} columns")
            for a, b, v in pairs.head(15).itertuples(index=False):
                self._issue("Low", "Relationships", f"Highly correlated numeric pairs (|Spearman| ≥ {ct})",
                            "Consider keeping only one of each pair for linear models, and check whether one is derived from the other.",
                            detail=f"`{a}` and `{b}` ({v:.2f})")
        elif pairs is not None:
            self.passed.append(f"no numeric pairs with |Spearman| of {ct} or more")
        cpairs = self.associated_pairs(vt)
        if cpairs is not None and len(cpairs):
            for a, b, v in cpairs.head(10).itertuples(index=False):
                self._issue("Low", "Relationships", f"Strongly associated categorical pairs (Cramér's V ≥ {vt})",
                            "One of each pair may be redundant, or one may be a grouping of the other.",
                            detail=f"`{a}` and `{b}` ({v:.2f})")
        elif cpairs is not None:
            self.passed.append(f"no categorical pairs with Cramér's V of {vt} or more")

        dates = list(self.parsed_dates)
        if len(dates) >= 2:
            rows = []
            for i, a in enumerate(dates):
                for b in dates[i + 1:]:
                    both = self.parsed_dates[a].notna() & self.parsed_dates[b].notna()
                    if both.sum() == 0:
                        continue
                    da, db = self.parsed_dates[a][both], self.parsed_dates[b][both]
                    before, after = (da < db).mean() * 100, (da > db).mean() * 100
                    rows.append((a, b, before, (da == db).mean() * 100, after))
                    minority = min(before, after)
                    if 0 < minority < 5:
                        first, second = (a, b) if before > after else (b, a)
                        self._issue("Medium", "Relationships", "Dates in an unusual order",
                                    "Check those rows for data entry errors or time zone problems.",
                                    detail=f"`{second}` before `{first}` ({minority:.2f}%)")
            if rows:
                self._date_order = pd.DataFrame(rows, columns=["Column A", "Column B", "A before B %", "Same %",
                                                               "A after B %"]).round(2)
        if self.spearman is None and self.cramers is None and self._date_order is None:
            sec.skipped = "There are not enough numeric, categorical, or date columns to compare."
            return
        self._relationship_blocks(sec, ct, vt)
        self.figure_count += sum(1 for b in sec.blocks if b[0] == "figure")

    def correlated_pairs(self, threshold=None):
        """Numeric column pairs with |Spearman| at or above the threshold, strongest first. None if not computed."""
        if self.spearman is None:
            return None
        t = self.corr_threshold if threshold is None else threshold
        m, cols = self.spearman, list(self.spearman.columns)
        rows = [(a, b, m.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]
                if pd.notna(m.loc[a, b]) and abs(m.loc[a, b]) >= t]
        rows.sort(key=lambda r: -abs(r[2]))
        return pd.DataFrame(rows, columns=["Column A", "Column B", "Spearman"]).round(3)

    def associated_pairs(self, threshold=None):
        """Categorical column pairs with Cramér's V at or above the threshold, strongest first. None if not computed."""
        if self.cramers is None:
            return None
        t = self.cramers_threshold if threshold is None else threshold
        m, cols = self.cramers, list(self.cramers.columns)
        rows = [(a, b, m.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:]
                if pd.notna(m.loc[a, b]) and m.loc[a, b] >= t]
        rows.sort(key=lambda r: -r[2])
        return pd.DataFrame(rows, columns=["Column A", "Column B", "Cramér's V"]).round(3)

    def _relationship_blocks(self, sec, corr_threshold, cramers_threshold):
        """Tables of pairs above the thresholds first, then one figure with both heatmaps side by side."""
        if self._rel_note:
            sec.text(self._rel_note)
        for kind, pairs, t, label, measure in (
                ("num", self.correlated_pairs(corr_threshold), corr_threshold,
                 f"Highly correlated numeric pairs (|Spearman| ≥ {corr_threshold})", "numeric pairs have an absolute Spearman correlation"),
                ("cat", self.associated_pairs(cramers_threshold), cramers_threshold,
                 f"Strongly associated categorical pairs (Cramér's V ≥ {cramers_threshold})", "categorical pairs have a Cramér's V")):
            if pairs is None:
                continue
            if len(pairs):
                note = (f"{len(pairs):,} pairs found." + (" The 50 strongest are shown." if len(pairs) > 50 else ""))
                sec.table(label, pairs.head(50), note)
            else:
                sec.text(f"No {measure} of {t} or more.")
        panels = [(m, d, title) for m, d, title in (
            (self.spearman, True, "Spearman correlation"),
            (self.cramers, False, "Cramér's V"),
        ) if m is not None]
        if panels:
            sizes = [len(m) for m, _, _ in panels]
            widths = [min(9.0, 2.8 + 0.42 * k) for k in sizes]
            h = min(9.5, 2.6 + 0.4 * max(sizes))
            fig = Figure(figsize=(sum(widths) + 0.4, h), dpi=DPI, layout="constrained")
            axes = fig.subplots(1, len(panels), squeeze=False, gridspec_kw=dict(width_ratios=widths))[0]
            for ax, (m, diverging, title), k in zip(axes, panels, sizes):
                heatmap(ax, m, diverging=diverging, annot=k <= 13)
                ax.set_title(title, loc="left")
            what = {"Spearman correlation": "Spearman correlation between numeric columns (from minus 1 to 1)",
                    "Cramér's V": "Cramér's V between categorical columns with 2 to 50 values (from 0 to 1)"}
            names = [what[t] for _, _, t in panels]
            sec.text(("Left: " + names[0] + ". Right: " + names[1] + ".") if len(names) == 2 else names[0] + ".")
            stem = "correlation_heatmaps"
            sec.figure("Spearman correlation and Cramér's V", fig, stem)
        if self._date_order is not None:
            sec.table("Ordering of date columns", self._date_order,
                      "A small share of rows in the unusual order, such as an end date before a start date, often means bad data.")

    # ------------------------------------------------------------ 6. target

    def _sec_target(self, sec):
        if not self.target:
            sec.skipped = ("Target analysis was skipped because no target was chosen. To add it, rerun with "
                           "--target <column> --target-type continuous|categorical, or pass target= and target_type= "
                           "to EDAReport or eda_report.")
            return
        t, df, s = self.target, self.df, self.s
        y = df[t]
        kind = self.target_kind()
        if kind == "continuous" and not self._target_numeric_dtype():
            sec.skipped = (f"The target `{t}` was set as continuous, but it holds {y.dtype} values that are not numbers. "
                           "Convert it to a number, or rerun with the target set as categorical.")
            return
        is_num = kind == "continuous"
        task = "regression" if is_num else "classification"
        how = ("as you specified" if self.target_type else
               "based on its type and number of unique values; pass target_type to override this")
        lines = [f"`{t}` is treated as a **{kind}** target ({task}), {how}. It has {y.nunique():,} unique values, and "
                 f"{y.isna().mean():.2%} of target values are missing."]
        if y.isna().any():
            self._issue("High", "Target", "Target has missing values", "Find out why, and drop or separately handle those rows before training.",
                        message=f"The target `{t}` is missing for {y.isna().sum():,} rows ({y.isna().mean():.2%}).")
        if task == "regression":
            yy = y.dropna().astype(float)
            lines.append("Its distribution is in the numeric profile table and the distribution plots.")
            if y.nunique() <= 20:
                lines.append(f"Note: `{t}` has only {y.nunique()} unique values, so check that it really is continuous.")
            if abs(yy.skew()) > 2:
                self._issue("Medium", "Target", "Target is highly skewed",
                            "Consider a log transformed target, a Tweedie or Gamma loss, or a metric that suits skewed targets.",
                            message=f"The target `{t}` is highly skewed (skew {yy.skew():.1f}).")
        else:
            vc = y.value_counts(dropna=False)
            if len(vc) <= 10:
                lines.append("**Class balance:** " + "; ".join(
                    f"`{'(missing)' if pd.isna(k) else short(k, 20)}` {v / vc.sum():.1%} ({v:,} rows)" for k, v in vc.items()) + ".")
            else:
                sec.table("Class balance", pd.DataFrame({
                    "Class": [("(missing)" if pd.isna(i) else short(i, 30)) for i in vc.index],
                    "Rows": vc.values, "Share %": (vc.values / vc.sum() * 100).round(2)}).head(30))
            ratio = vc.iloc[0] / max(vc.iloc[-1], 1)
            if ratio >= 10:
                self._issue("Medium", "Target", "Imbalanced target",
                            "Use stratified splits, class weights, and metrics such as PR AUC instead of accuracy.",
                            message=f"The target `{t}` is imbalanced (largest class is {ratio:.0f} times the smallest).")
        sec.text(" ".join(lines[:1]) + ("\n\n" + "\n\n".join(lines[1:]) if len(lines) > 1 else ""))
        info = {"column": t, "type": kind, "task": task, "missing_pct": round(float(y.isna().mean() * 100), 3),
                "unique": int(y.nunique())}
        if task == "regression":
            info["skew"] = round(float(y.dropna().astype(float).skew()), 3)
        else:
            vc_ = y.value_counts()
            if len(vc_) <= 10:
                info["class_balance"] = {str(k): round(float(v / vc_.sum()), 4) for k, v in vc_.items()}
            if len(vc_):
                info["imbalance_ratio"] = round(float(vc_.iloc[0] / max(vc_.iloc[-1], 1)), 2)
        self.target_info = info

        ys = s[t]
        ok = ys.notna()
        y_code = ys.astype(float) if is_num else pd.Series(pd.factorize(ys)[0], index=ys.index).where(ok)
        binary = task == "classification" and ys.nunique() == 2
        feats_num = [c for c in self.cols("numeric") if s[c].nunique() > 1]
        feats_cat = [c for c in self.cols("categorical", "boolean") if 1 < s[c].nunique() <= 1000]
        rows = []
        for c in feats_num:
            x = s[c].astype(float)
            m = ok & x.notna()
            if m.sum() < 10:
                continue
            if task == "regression" or binary:
                v = x[m].rank().corr(y_code[m].rank())  # Spearman without needing scipy
                rows.append({"Feature": c, "Type": "numeric", "Measure": "Spearman", "Association": v,
                             "Strength": abs(v) if pd.notna(v) else np.nan})
            else:
                v = correlation_ratio(ys[m].astype(str), x[m])
                rows.append({"Feature": c, "Type": "numeric", "Measure": "Eta", "Association": v, "Strength": v})
        for c in feats_cat:
            m = ok & s[c].notna()
            if m.sum() < 10:
                continue
            if task == "regression":
                v = correlation_ratio(s[c][m].astype(str), y_code[m])
                rows.append({"Feature": c, "Type": "categorical", "Measure": "Eta", "Association": v, "Strength": v})
            else:
                v = cramers_v(s[c][m].astype(str), ys[m].astype(str))
                rows.append({"Feature": c, "Type": "categorical", "Measure": "Cramér's V", "Association": v, "Strength": v})
        assoc = pd.DataFrame(rows)
        mi = self._mutual_info(feats_num, feats_cat, task)
        ranked = []
        if not assoc.empty:
            if mi is not None:
                assoc["Mutual information"] = assoc["Feature"].map(mi)
            assoc = assoc.sort_values("Strength", ascending=False)
            ranked = list(assoc["Feature"])
            self.target_rank = [c for c in ranked]
            self.target_assoc_df = assoc
            self.target_ctx = (task, binary, feats_num, mi is not None)
            self.target_assoc = [
                {"feature": r["Feature"], "measure": r["Measure"], "association": round(float(r["Association"]), 4),
                 **({"mutual_information": round(float(r["Mutual information"]), 4)}
                    if "Mutual information" in r and pd.notna(r["Mutual information"]) else {})}
                for _, r in assoc.head(10).iterrows() if pd.notna(r["Association"])]
            self._target_table(sec, assoc, mi is not None)

        leaks = []
        for r in rows:
            if pd.notna(r["Strength"]) and r["Strength"] >= LEAKAGE_ASSOC:
                leaks.append((r["Feature"], f"association with the target is {r['Strength']:.3f}"))
        for c in df.columns:
            if c == t:
                continue
            if LEAK_HINTS.search(str(c)) or (len(str(t)) >= 3 and str(t).lower() in str(c).lower()):
                leaks.append((c, "the name suggests it may be recorded after the outcome or derived from the target"))
            miss = s[c].isna()
            if 0.01 < miss.mean() < 0.99 and y_code.notna().sum() > 10:
                v = miss.astype(float)[ok].corr(y_code[ok])
                if pd.notna(v) and abs(v) >= 0.5:
                    leaks.append((c, f"whether it is missing is strongly linked to the target (correlation {v:.2f})"))
        if leaks:
            merged = {}
            for c, why in leaks:
                merged.setdefault(c, []).append(why)
            self.leakage = {c: list(dict.fromkeys(w)) for c, w in merged.items()}
            sec.table("Possible leakage", pd.DataFrame([(c, "; ".join(w).capitalize() + ".") for c, w in merged.items()],
                                                       columns=["Feature", "Reasons"]),
                      "These are warnings, not proof. Confirm that each feature is known at prediction time.")
            for c, whys in merged.items():
                strong = any(w.startswith("association") for w in whys)
                self._flag(c, "possible leakage")
                self._issue("High" if strong else "Medium", "Target", "Possible leakage",
                            "Confirm the feature is available at prediction time, and drop it if it is not.", c, f"`{c}`")
        else:
            self.passed.append("no obvious leakage signals")

        self._target_grid(sec, ranked, task, binary, feats_num, self._assoc_info(assoc))

        dcol = self._primary_date()
        if dcol and (binary or task == "regression"):
            dates = self.parsed_dates[dcol].reindex(s.index)
            freq = self._freq(dates.dropna())
            yv = (ys == ys.value_counts().index[-1]).astype(float) * 100 if binary else ys.astype(float)
            g = yv[ok & dates.notna()].groupby(dates[ok & dates.notna()].dt.to_period(freq)).mean()
            if len(g) >= 3:
                fig, axes = new_fig(10, 3.2)
                ax = axes[0, 0]
                sns.lineplot(x=g.index.to_timestamp(), y=g.values, ax=ax, color=PRIMARY, linewidth=1.8)
                ax.set_ylabel("Target rate (%)" if binary else f"Mean {short(t, 20)}")
                ax.set_title(f"Target over time by {self._freq_name(freq)} of {short(dcol, 25)}", loc="left")
                date_axis(ax)
                self._fig(sec, "Target over time", fig, "target_over_time")

    @staticmethod
    def _assoc_info(assoc):
        """Map each feature to (measure, association, mutual information or None) for plot titles."""
        info = {}
        for _, r in assoc.iterrows():
            mi = r["Mutual information"] if "Mutual information" in assoc.columns and pd.notna(r.get("Mutual information")) else None
            info[r["Feature"]] = (r["Measure"], r["Association"], mi)
        return info

    @staticmethod
    def _filter_assoc(assoc, top_n=None, min_association=None, min_mutual_info=None):
        out = assoc
        if min_association is not None:
            out = out[out["Strength"] >= min_association]
        if min_mutual_info is not None and "Mutual information" in out.columns:
            out = out[out["Mutual information"] >= min_mutual_info]
        return out if top_n is None else out.head(top_n)

    def _target_table(self, sec, assoc, has_mi, top_n=None, min_association=None, min_mutual_info=None):
        shown = self._filter_assoc(assoc, None, min_association, min_mutual_info)
        total = len(shown)
        shown = shown.head(40 if top_n is None else top_n)
        filt = []
        if min_association is not None:
            filt.append(f"an absolute association of at least {min_association:g}")
        if min_mutual_info is not None:
            filt.append(f"a mutual information of at least {min_mutual_info:g}" if has_mi else "")
        filt = [f for f in filt if f]
        note = ("Spearman is used for numeric features with a numeric or binary target, Eta (correlation ratio) for "
                "numeric and categorical pairs, and Cramér's V for two categorical columns. All run from 0 to 1 in "
                "strength, and Spearman can be negative. Mutual information captures non linear relationships"
                + ("." if has_mi else ", but needs scikit-learn, which is not installed."))
        if filt:
            note += f" Only features with {' and '.join(filt)} are listed ({total:,} of {len(assoc):,} qualify)."
        if min_mutual_info is not None and not has_mi:
            note += " min_mutual_info was ignored because mutual information is not available."
        if shown.empty:
            sec.text("No feature meets the association and mutual information limits." + (" " + note if note else ""))
            return
        sec.table("Feature relationships with the target (strongest first)", shown.drop(columns="Strength").round(4), note)

    def _filtered_target_section(self, sec, top_n, min_association, min_mutual_info):
        for name, v in (("top_n", top_n), ("min_association", min_association), ("min_mutual_info", min_mutual_info)):
            if v is None:
                continue
            if name == "top_n" and (not isinstance(v, (int, np.integer)) or v < 1):
                raise ValueError("top_n must be a whole number of at least 1, or None")
            if name != "top_n" and (not isinstance(v, (int, float, np.integer, np.floating)) or v < 0):
                raise ValueError(f"{name} must be a number of at least 0, or None")
        task, binary, feats_num, has_mi = self.target_ctx
        assoc = self.target_assoc_df
        tmp_t, tmp_g = Section(sec.key, sec.title, sec.intro), Section(sec.key, sec.title, sec.intro)
        with style():
            self._target_table(tmp_t, assoc, has_mi, top_n, min_association, min_mutual_info)
            keep = self._filter_assoc(assoc, None, min_association, min_mutual_info)
            self._target_grid(tmp_g, list(keep["Feature"]), task, binary, feats_num, self._assoc_info(assoc), limit=top_n)
        new, placed_grid = [], False
        for b in sec.blocks:
            if b[0] == "table" and b[1].startswith("Feature relationships with the target"):
                new.extend(tmp_t.blocks)
            elif b[0] == "text" and b[1].startswith("The ") and "features most related to the target" in b[1]:
                continue
            elif b[0] == "figure" and b[1] == "Strongest features against the target":
                new.extend(tmp_g.blocks)
                placed_grid = True
            else:
                new.append(b)
        if not placed_grid:
            new.extend(tmp_g.blocks)
        out = Section(sec.key, sec.title, sec.intro)
        out.blocks = new
        return out

    def _target_grid(self, sec, ranked, task, binary, feats_num, info=None, limit=None):
        """One grid of the strongest features against the target."""
        t, s = self.target, self.s
        ys = s[t]
        ok = ys.notna()
        multi_small = task == "classification" and not binary and ys.nunique() <= 10
        panels = []
        for c in ranked:
            is_num_feat = c in feats_num
            if is_num_feat and (task == "regression" or binary or multi_small):
                panels.append(c)
            elif not is_num_feat and (task == "regression" or binary):
                panels.append(c)
            if len(panels) >= (limit if limit is not None else min(9, self.max_plots)):
                break
        if not panels:
            if limit is not None or info:
                sec.text("No feature meets the limits, so there are no plots.")
            return
        info = info or {}

        def title(c, n):
            if c not in info:
                return short(c, n)
            measure, a, mi = info[c]
            extra = f"{measure} {a:+.2f}" if measure == "Spearman" else f"{measure} {a:.2f}"
            if mi is not None:
                extra += f", MI {mi:.3f}"
            return short(c, n) + "\n" + extra

        pos = ys.value_counts().index[-1] if binary else None
        classes = list(ys.dropna().value_counts().index) if task == "classification" else []
        ncol = min(3, len(panels))
        nrow = math.ceil(len(panels) / ncol)
        fig, axes = new_fig(4.4 * ncol, 3.4 * nrow + 0.3, ncols=ncol, nrows=nrow)
        for ax, c in zip(axes.flat, panels):
            if c in feats_num:
                m = ok & s[c].notna()
                xv = s[c][m].astype(float)
                if task == "regression":
                    yv = ys[m].astype(float)
                    xl, xh = xv.quantile([0.005, 0.995])
                    yl, yh = yv.quantile([0.005, 0.995])
                    k = (xv >= xl) & (xv <= xh) & (yv >= yl) & (yv <= yh)
                    sns.histplot(x=xv[k], y=yv[k], bins=40, cmap="Blues", ax=ax, cbar=False, norm=mpl.colors.LogNorm(), vmin=None, vmax=None)
                    ax.xaxis.set_major_formatter(THOUSANDS)
                    ax.set_xlabel(short(c, 25))
                    ax.set_ylabel(short(t, 20))
                else:
                    lab = ys[m].map(lambda v: short(v, 10))
                    order = [short(k_, 10) for k_ in classes]
                    sns.boxplot(x=lab, y=xv, order=order, ax=ax, showfliers=False, width=0.55,
                                color=PRIMARY_LIGHT, linewidth=0.9, medianprops=dict(color=ACCENT, linewidth=1.6))
                    ax.set_xlabel(short(t, 20))
                    ax.set_ylabel("")
                ax.set_title(title(c, 30), loc="left", fontsize=9, fontweight="bold")
            else:
                yv = (ys == pos).astype(float) * 100 if binary else ys.astype(float)
                g = pd.DataFrame({"c": s[c], "y": yv})[ok].groupby("c", observed=True)["y"].agg(["mean", "count"])
                g = g[g["count"] >= MIN_VALUE_COUNT].sort_values("count", ascending=False).head(10)
                hbar(ax, [short(i, 20) for i in g.index], g["mean"].values, [PRIMARY] * len(g),
                     fmt=pct_label if binary else num_label, fontsize=7)
                if binary:
                    overall = yv[ok].mean()
                    ax.axvline(overall, color=ACCENT, linewidth=1.2, linestyle="--")
                ax.set_xlabel(f"Share of '{short(pos, 12)}' (%)" if binary else f"Mean {short(t, 20)}", fontsize=8)
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelsize=8)
                ax.set_title(title(c, 26) + " (top 10 values)" if c not in info else title(c, 26) + ", top 10 values",
                             loc="left", fontsize=9, fontweight="bold")
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        if task == "regression":
            how = ("Numeric features are 2D histograms against the target (darker means more rows, 0.5% tails trimmed). "
                   "Categorical features show the mean target for their 10 most common values.")
        elif binary:
            how = (f"Numeric features show box plots by class, without outliers. Categorical features show the share of "
                   f"'{short(pos, 15)}' for their 10 most common values, and the dashed line is the overall share.")
        else:
            how = "Each panel shows box plots of a numeric feature by class, without outliers."
        sec.text(f"The {len(panels)} features most related to the target, in order of strength. Each title shows the "
                 f"feature's association with the target (the measure is named) and its mutual information (MI). {how}")
        self._fig(sec, "Strongest features against the target", fig, "target_features")

    def _mutual_info(self, feats_num, feats_cat, task):
        if mutual_info_regression is None or not (feats_num or feats_cat):
            return None
        s = self.s if len(self.s) <= 50_000 else self.s.sample(50_000, random_state=self.random_state)
        y = s[self.target]
        ok = y.notna()
        s, y = s[ok], y[ok]
        X = pd.DataFrame(index=s.index)
        for c in feats_num:
            X[c] = s[c].astype(float).fillna(s[c].median())
        for c in feats_cat:
            top = s[c].astype(str).value_counts().index[:50]
            X[c] = pd.factorize(s[c].astype(str).where(s[c].astype(str).isin(top), "__other__"))[0]
        disc = [c in feats_cat for c in X.columns]
        try:
            if task == "regression":
                v = mutual_info_regression(X, y.astype(float), discrete_features=disc, random_state=self.random_state)
            else:
                v = mutual_info_classif(X, pd.factorize(y)[0], discrete_features=disc, random_state=self.random_state)
        except Exception:
            return None
        return dict(zip(X.columns, v))

    # ------------------------------------------------------------ 7. summary

    def _group_issues(self):
        groups = {}
        for d in self.issues:
            key = (d["severity"], d["section"], d["kind"])
            g = groups.setdefault(key, dict(severity=d["severity"], area=d["section"], kind=d["kind"],
                                            details=[], messages=[], action=d["action"]))
            if d["detail"]:
                g["details"].append(d["detail"])
            else:
                g["messages"].append(d["message"])
        out = []
        for g in groups.values():
            det = list(dict.fromkeys(g["details"]))
            cols = ", ".join(det[:MAX_LISTED]) + (f", and {len(det) - MAX_LISTED} more" if len(det) > MAX_LISTED else "")
            finding = g["kind"] if det else " ".join(dict.fromkeys(g["messages"]))
            out.append(dict(severity=g["severity"], area=g["area"], finding=finding, count=len(det),
                            columns=cols, action=g["action"] or "", kind=g["kind"]))
        out.sort(key=lambda g: (SEVERITY_RANK[g["severity"]], -g["count"]))
        self.issue_groups = out
        return out

    def _sec_summary(self, sec):
        if self.mode == "targeted":
            self._check_framing(sec)
        else:
            self._render_hints(sec)
        groups = self._group_issues()
        counts = pd.Series([g["severity"] for g in groups]).value_counts()
        if not groups:
            sec.text("No data quality issues were flagged.")
        else:
            sec.text(f"The checks found **{len(groups)} kinds of issue**: {counts.get('High', 0)} High, "
                     f"{counts.get('Medium', 0)} Medium, and {counts.get('Low', 0)} Low. High issues can change model "
                     "results or conclusions and should be resolved first. The Flags column in the profile tables shows "
                     "the same findings for each column.")
            sec.table("Issues by type", pd.DataFrame([
                {"Severity": g["severity"], "Area": g["area"], "Finding": g["finding"],
                 "Columns": f"{g['count']:,}: {g['columns']}" if g["count"] else "", "Suggested action": g["action"]}
                for g in groups]))
        if self.passed:
            p = list(dict.fromkeys(self.passed))
            sec.text("**Checks that found nothing:** " + "; ".join(p) + ".")
        kinds = {g["kind"] for g in groups}
        has = lambda *words: any(any(w in k.lower() for w in words) for k in kinds)
        steps = []
        if has("constant"):
            steps.append("Drop the constant and near constant columns listed above.")
        if has("stored as text", "mixed python"):
            steps.append("Fix the column types listed above.")
        if has("disguised"):
            steps.append("Replace disguised missing values with NaN, then rerun this report.")
        if has("mostly missing"):
            steps.append("Decide for each mostly missing column whether to drop it or keep a missing indicator.")
        if any(g["area"] == "Keys and duplicates" and g["severity"] != "Low" for g in groups):
            steps.append("Resolve duplicate rows or keys before splitting or aggregating the data.")
        if has("leakage"):
            steps.append("Confirm that the features flagged for possible leakage are known at prediction time.")
        if has("high cardinality"):
            steps.append("Choose an encoding for the high cardinality columns, such as target or frequency encoding.")
        if has("skewed", "outliers"):
            steps.append("Decide on transforms or capping for skewed numeric columns if you plan to use linear models.")
        if self.parsed_dates and self.target:
            steps.append("Use a time based split for validation, since the data has a date column and a target.")
        if any(c["status"] == "fail" for c in self.framing_checks):
            steps.insert(0, "Resolve the failed checks against the ML framing before splitting the data.")
        steps.append("This report does not change the data. Apply fixes in your own pipeline and rerun the report to confirm them.")
        sec.text("**Suggested next steps**")
        sec.bullets(steps)

    # ------------------------------------------------------------ shared spec

    def _check_framing(self, sec):
        """Compare the data with the framing in the shared spec. Adds a table to the summary section."""
        self.framing_checks, self.framing_questions = [], []
        fr = (self.spec or {}).get("framing") or {}
        data = (self.spec or {}).get("data") or {}
        checks, qs = self.framing_checks, self.framing_questions
        add = lambda check, status, detail: checks.append({"check": check, "status": status, "detail": detail})
        if not self.spec:
            return
        if fr.get("status") in (None, "not_started"):
            add("Problem framing", "skipped", "The spec has no framing yet, so the data could not be checked against it.")
            from ml_common import FRAMING_TODO
            qs.append((FRAMING_TODO, "framing"))
            self._render_framing(sec)
            return
        df = self.df
        tgt = fr.get("target") or {}
        tname, task_type = tgt.get("name"), fr.get("task_type")

        # Target
        if tname and tname in df.columns:
            add("Target column exists", "pass", f"`{tname}` is in the data.")
            if self.target and self.target != tname:
                add("Target matches the framing", "warn",
                    f"The framing names `{tname}`, but this report analysed `{self.target}`.")
        elif tname:
            if tgt.get("exists_in_data") is False:
                add("Target column exists", "warn", f"`{tname}` is not in the data yet. The framing says it must be "
                    f"built: {tgt.get('construction') or 'no construction rule given'}.")
                if not tgt.get("construction"):
                    qs.append((f"How should the target `{tname}` be built from the data?", "framing"))
            else:
                close = match_column(df, tname)[1]
                add("Target column exists", "fail", f"The framing names `{tname}`, but it is not in the data"
                    + (f". Closest names: {', '.join(f'`{c}`' for c in close)}." if close else "."))
                qs.append((f"The framing names the target `{tname}`, but the data has no such column. "
                          "Which column holds it, or how should it be built?", "framing"))
        else:
            add("Target column exists", "warn", "The framing does not name a target column yet.")

        # Task type against the analysed target
        kind = self.target_kind() if self.target else None
        if kind and task_type:
            want = {"binary_classification": "categorical", "multiclass_classification": "categorical",
                    "regression": "continuous", "forecasting": "continuous"}.get(task_type)
            nun = int(df[self.target].nunique())
            if want and want != kind:
                add("Target type matches the task type", "fail",
                    f"The framing says {task_type.replace('_', ' ')}, but `{self.target}` was analysed as {kind} "
                    f"({nun:,} unique values).")
                qs.append((f"The framing says {task_type.replace('_', ' ')}, but `{self.target}` looks {kind}. "
                          "Which is right?", "framing"))
            elif task_type == "binary_classification" and nun != 2:
                add("Target type matches the task type", "fail",
                    f"Binary classification needs two classes, but `{self.target}` has {nun:,}.")
                qs.append((f"`{self.target}` has {nun:,} classes, but the framing says binary. How should it be mapped "
                          "to two classes?", "framing"))
            else:
                add("Target type matches the task type", "pass", f"`{self.target}` is {kind}, as {task_type.replace('_', ' ')} needs.")
            share = (self.target_info.get("class_balance") or {})
            if task_type == "binary_classification" and len(share) == 2:
                minority = min(share.values())
                metric = str((fr.get("success") or {}).get("primary_metric") or "").lower()
                if minority < 0.05 and "accuracy" in metric:
                    add("Metric suits the class balance", "fail",
                        f"The smaller class is {minority:.1%} of rows, so accuracy is misleading.")
                    qs.append(("The positive class is rare, but the primary metric is accuracy. Switch to PR AUC or "
                              "recall at a fixed volume?", "framing"))
                elif minority < 0.05:
                    add("Metric suits the class balance", "pass" if metric else "warn",
                        f"The smaller class is {minority:.1%} of rows. "
                        + (f"The primary metric is {metric}." if metric else "No primary metric is set yet."))

        # Entity IDs and grain
        ents = [c for c in (fr.get("entity_id_cols") or []) if c]
        missing = [c for c in ents if c not in df.columns]
        if missing:
            add("Entity ID columns exist", "fail", "Not in the data: " + ", ".join(f"`{c}`" for c in missing) + ".")
            qs.append(("These entity ID columns from the framing are not in the data: " + ", ".join(missing)
                      + ". Which columns identify the entity?", "framing"))
        for c in [c for c in ents if c in df.columns]:
            nun, n = int(df[c].nunique()), len(df)
            if nun < n:
                add(f"Rows per `{c}`", "warn",
                    f"{nun:,} distinct values over {n:,} rows, about {n / max(nun, 1):.1f} rows each. Split by "
                    f"`{c}` or by time so the same entity is not in both train and test.")
            else:
                add(f"Rows per `{c}`", "pass", f"One row per `{c}`.")
        unit = fr.get("unit_of_prediction")
        pcol_ = (fr.get("prediction_time") or {}).get("column") or data.get("date_col")
        key = [c for c in ents if c in df.columns] + ([pcol_] if pcol_ in df.columns and pcol_ not in ents else [])
        if unit and key:
            dupk = int(df.duplicated(subset=key, keep=False).sum())
            ktxt = " + ".join(f"`{c}`" for c in key)
            if dupk == 0:
                add("Grain matches the unit of prediction", "pass",
                    f"Framing: {unit}. Every row has a unique {ktxt}, which matches.")
            else:
                add("Grain matches the unit of prediction", "warn",
                    f"Framing: {unit}. But {dupk:,} rows share their {ktxt} with another row. Decide whether to "
                    "aggregate them or whether the unit of prediction is finer than the framing says.")
                qs.append((f"{dupk:,} rows share the same {' + '.join(key)}. Should they be aggregated, or is the unit "
                           "of prediction finer than the framing says?", "framing"))
        elif unit:
            g = self.grain
            gtxt = ("one row per " + " + ".join(f"`{c}`" for c in g["columns"])) if g["columns"] else "unclear"
            add("Grain matches the unit of prediction", "warn",
                f"Framing: {unit}. Detected grain: {gtxt}. Set entity ID columns in the framing so this can be checked.")

        # Prediction time column
        pcol = (fr.get("prediction_time") or {}).get("column") or data.get("date_col")
        if pcol:
            if pcol not in df.columns:
                add("Prediction time column exists", "fail", f"`{pcol}` is not in the data.")
                qs.append((f"The prediction time column `{pcol}` is not in the data. Which column marks when the "
                          "prediction is made?", "framing"))
            elif pcol in self.parsed_dates:
                d = self.parsed_dates[pcol].dropna()
                add("Prediction time column exists", "pass",
                    f"`{pcol}` runs from {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}." if len(d) else f"`{pcol}` has no dates.")
            else:
                add("Prediction time column exists", "warn", f"`{pcol}` is in the data but could not be read as dates.")
        elif task_type == "forecasting":
            add("Prediction time column exists", "fail", "Forecasting needs a date column, and none is set in the spec.")

        # Excluded columns and leakage
        excl = [e.get("column") for e in (data.get("excluded_cols") or []) if isinstance(e, dict) and e.get("column")]
        if excl:
            absent = [c for c in excl if c not in df.columns]
            add("Excluded columns", "pass" if not absent else "warn",
                f"{len(excl) - len(absent)} of {len(excl)} excluded columns are in the data and will be kept out of features."
                + (" Not found: " + ", ".join(f"`{c}`" for c in absent) + "." if absent else ""))
        open_leaks = [c for c in self.leakage if c not in excl and c != self.target]
        if self.target:
            add("Leakage suspects handled", "pass" if not open_leaks else "warn",
                "No leakage suspects outside the excluded columns." if not open_leaks else
                "Not yet excluded: " + ", ".join(f"`{c}`" for c in open_leaks) + ".")
            for c in open_leaks:
                qs.append((f"Is `{c}` known at prediction time? EDA flagged it as possible leakage: "
                           + "; ".join(self.leakage[c]) + ".", "features"))

        for ch in checks:
            if ch["status"] == "fail":
                self._issue("High", "Framing", "Data does not match the ML framing (see the framing checks above)",
                            "Fix the data or update the framing with the ml-problem-framing skill.",
                            detail=ch["check"])
        if checks and not any(c["status"] == "fail" for c in checks):
            self.passed.append("the data matches the ML framing on every hard check")
        self._render_framing(sec)

    def framing_hints(self):
        """Facts from the data that help the ML framing. Column names and counts only."""
        if getattr(self, "_hints", None) is not None:
            return self._hints
        df, n = self.df, max(len(self.df), 1)
        cands = []
        for r in suggest_targets(df, top=5):
            if r["score"] <= 0:
                continue
            item = {"column": str(r["column"]), "suggested_type": r["suggested_type"], "unique": int(r["unique"]),
                    "missing_pct": round(float(r["missing_pct"]), 2), "reasons": list(r["reasons"])}
            y = df[r["column"]].dropna()
            if y.nunique() == 2:
                item["smaller_class_share"] = round(float(y.value_counts(normalize=True).min()), 4)
            cands.append(item)
        ents = []
        for c in df.columns:
            if self.roles.get(c) == "pii" or not self._hashable(c):
                continue
            if self.roles.get(c) == "id" or ID_NAME.search(str(c)) or c in self.id_cols:
                nun = int(df[c].nunique())
                if nun > 1:
                    ents.append({"column": str(c), "distinct": nun, "rows_per_value": round(n / nun, 2)})
        ents.sort(key=lambda e: -e["rows_per_value"])
        dates = []
        for c, d in self.parsed_dates.items():
            dd = d.dropna()
            if len(dd):
                dates.append({"column": str(c), "min": f"{dd.min():%Y-%m-%d}", "max": f"{dd.max():%Y-%m-%d}",
                              "span_days": int((dd.max() - dd.min()).days),
                              "missing_pct": round(float(d.isna().mean() * 100), 2)})
        main = self._primary_date()
        history = next((x["span_days"] for x in dates if x["column"] == str(main)), None)
        post = [str(c) for c in df.columns if LEAK_HINTS.search(str(c)) and c != self.target]
        self._hints = {"candidate_targets": cands, "grain": self.grain, "entity_id_candidates": ents[:8],
                       "date_columns": dates, "main_date": str(main) if main else None, "history_days": history,
                       "post_outcome_suspects": post}
        return self._hints

    def _render_hints(self, sec):
        """Discovery mode: show what the data says about the open framing questions."""
        h = self.framing_hints()
        sec.text("**Hints for the ML framing.** This is a run without a target. These facts help "
                 "choose the target, the unit of prediction, and the prediction time. Run the report again with the "
                 "confirmed target for target analysis and leakage checks.")
        items = []
        if h["candidate_targets"]:
            items.append("**Candidate targets:** " + "; ".join(
                f"`{c['column']}` ({c['suggested_type']}, {c['unique']:,} unique"
                + (f", smaller class {c['smaller_class_share']:.1%}" if "smaller_class_share" in c else "") + ")"
                for c in h["candidate_targets"]) + ". If none of these is the target, it has to be built from other columns.")
        else:
            items.append("**Candidate targets:** no column looks like a target, so it will probably have to be built "
                         "from other columns, for example from events in a time window.")
        g = h["grain"]
        items.append("**Grain:** " + ("one row per " + " + ".join(f"`{c}`" for c in g["columns"]) if g["columns"]
                                      else "unclear, no column or pair of columns is unique") + ".")
        if h["entity_id_candidates"]:
            items.append("**Possible entity IDs:** " + "; ".join(
                f"`{e['column']}` ({e['distinct']:,} values, {e['rows_per_value']:g} rows each)"
                for e in h["entity_id_candidates"]) + ". An ID with several rows each usually needs a grouped or time based split.")
        if h["date_columns"]:
            items.append("**Dates:** " + "; ".join(
                f"`{d['column']}` from {d['min']} to {d['max']} ({d['span_days']:,} days)" for d in h["date_columns"])
                + ". The outcome window and the test period both have to fit inside this history.")
        else:
            items.append("**Dates:** no date column was found, so a time based split and a prediction time column "
                         "are not available yet.")
        if h["post_outcome_suspects"]:
            items.append("**Possibly recorded after the outcome:** " + ", ".join(f"`{c}`" for c in h["post_outcome_suspects"])
                         + ". Confirm when each is filled in before using it as a feature.")
        sec.bullets(items)

    def _render_framing(self, sec):
        if not self.framing_checks:
            return
        label = {"pass": "Pass", "warn": "Check", "fail": "Fail", "skipped": "Skipped"}
        sec.table("Checks against the ML framing in ml_spec.yaml", pd.DataFrame(
            [{"Check": c["check"], "Result": label[c["status"]], "Detail": c["detail"]} for c in self.framing_checks]),
            "Fail means the data contradicts the framing. Check means a person should confirm something.")

    def spec_summary(self, source=None, report=None):
        """The eda section of the shared spec. Column names and summary numbers only, never raw values,
        except the target's class labels."""
        if not self._ran:
            self.run()
        role_map = {"numeric": "numeric", "categorical": "categorical", "boolean": "boolean", "datetime": "datetime",
                    "datetime_text": "datetime_as_text", "id": "identifier", "text": "free_text", "pii": "personal_data",
                    "other": "other"}
        by_role = {}
        for c, r in self.roles.items():
            if r != "target":
                by_role.setdefault(role_map.get(r, r), []).append(str(c))
        pdate = self._primary_date()
        prim = None
        if pdate:
            d = self.parsed_dates[pdate].dropna()
            prim = {"column": pdate, "min": f"{d.min():%Y-%m-%d}" if len(d) else None,
                    "max": f"{d.max():%Y-%m-%d}" if len(d) else None}
        issue_cols = {}
        for d in self.issues:
            if d["column"] is not None:
                issue_cols.setdefault((d["severity"], d["section"], d["kind"]), []).append(str(d["column"]))
        issues = []
        for g in self.issue_groups:
            cols = list(dict.fromkeys(issue_cols.get((g["severity"], g["area"], g["kind"]), [])))
            item = {"severity": g["severity"], "area": g["area"],
                    "finding": g["kind"] if cols else g["finding"], "action": g["action"]}
            if cols:
                item["columns"] = cols
            elif g["count"]:
                item["details"] = g["columns"]
            issues.append(item)
        excl = []
        why = {"id": "identifier", "text": "free text", "pii": "possible personal data"}
        for c, r in self.roles.items():
            if r in why:
                excl.append({"column": str(c), "reason": why[r]})
        for c, ws in self.leakage.items():
            if any(w.startswith("association") for w in ws) and c not in {e["column"] for e in excl}:
                excl.append({"column": str(c), "reason": "possible leakage: " + "; ".join(ws)})
        out = {
            "mode": self.mode,
            "run_at": pd.Timestamp.now().replace(microsecond=0).isoformat(),
            "source": source,
            "report": report,
            "rows": int(len(self.df)),
            "columns": int(self.df.shape[1]),
            "sampled": bool(self.sampled),
            "grain": self.grain,
            "duplicate_rows": int(self.duplicate_rows),
            "primary_date": prim,
            "columns_by_role": by_role,
            "target": self.target_info or None,
            "top_target_associations": self.target_assoc,
            "leakage_suspects": [{"column": str(c), "reasons": w} for c, w in self.leakage.items()],
            "suggested_exclusions": excl,
            "issues": issues,
            "missing_groups": self.missing_groups,
            "framing_checks": self.framing_checks,
            "framing_hints": self.framing_hints(),
        }
        return out

    def write_spec(self, path, source=None, report=None):
        """Write the eda section to the shared spec at path, creating the spec if needed.

        Also records the data file under data.sources and adds open questions raised by the framing checks.
        Returns (path, created).
        """
        from ml_common import load_or_create, update_section

        def rel(x):  # paths relative to the spec's folder, so runs from a notebook and a script match
            if not x:
                return x
            r = os.path.relpath(os.path.abspath(x), os.path.dirname(os.path.abspath(path)))
            return x if r.startswith("..") else r

        source, report = rel(source), rel(report)
        summary = self.spec_summary(source=source, report=report)
        spec, _ = load_or_create(path, name=self.name)
        sources = list((spec.get("data") or {}).get("sources") or [])
        if source:
            entry = {"path": source, "rows": int(len(self.df)), "columns": int(self.df.shape[1])}
            for i, s in enumerate(sources):
                if isinstance(s, dict) and s.get("path") and os.path.normpath(str(s["path"])) == os.path.normpath(source):
                    sources[i] = {**s, **entry}
                    break
            else:
                sources.append({**entry, "description": None})
        data_extra = {"sources": sources}
        dd = spec.get("data") or {}
        if self.date_col and not dd.get("date_col"):
            data_extra["date_col"] = self.date_col
        if self.id_cols and not dd.get("id_cols"):
            data_extra["id_cols"] = [c for c in self.id_cols
                                     if c not in ((spec.get("framing") or {}).get("entity_id_cols") or [])]
        if not self.framing_checks and (spec.get("framing") or {}).get("status") in (None, "not_started"):
            from ml_common import FRAMING_TODO
            self.framing_questions = [(FRAMING_TODO, "framing")]
        return update_section(path, "eda", summary, skill="ml-eda-report", replace=True,
                              open_questions=self.framing_questions, question_stage="framing",
                              extra={"data": data_extra}, name=self.name,
                              action=f"wrote eda for {source or self.name}")


TARGET_STRONG = re.compile(r"^(target|label|labels|y|class|outcome|response|ground_truth|gt)$", re.I)
TARGET_MEDIUM = re.compile(r"(^is_|^has_|^did_|^will_|churn|fraud|convert|conversion|click|ctr|default|purchase|"
                           r"_flag$|^flag_|_label$|_target$|^target_|retain|cancel|attrit|win|lapse|bad_)", re.I)
TARGET_WEAK = re.compile(r"(revenue|ltv|sales|amount|price|spend|score|rating|demand|qty|quantity|return)", re.I)


def suggest_targets(df, top=5):
    """Rank columns by how likely they are to be the target, using names and data properties.

    Returns a list of dicts sorted by score, with the reasons for each score. Only names, types, and
    counts are used in the output; no data values are returned.
    """
    n = max(len(df), 1)
    probe = df.sample(min(len(df), 50_000), random_state=0) if len(df) > 50_000 else df
    rows = []
    for i, c in enumerate(df.columns):
        y = df[c]
        name = str(c)
        try:
            nun = int(y.nunique())
        except TypeError:
            rows.append(dict(column=c, score=-9, reasons=["holds lists or dicts"], dtype=str(y.dtype), unique=None,
                             missing_pct=float(y.isna().mean() * 100), suggested_type=None))
            continue
        miss = float(y.isna().mean() * 100)
        numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
        boolean = pd.api.types.is_bool_dtype(y)
        kind = "continuous" if numeric and nun > 20 else "categorical"
        score, why = 0, []
        if TARGET_STRONG.search(name):
            score += 6; why.append("named like a target")
        elif TARGET_MEDIUM.search(name):
            score += 3; why.append("name suggests an outcome")
        elif TARGET_WEAK.search(name):
            score += 1; why.append("name suggests a measure that is often predicted")
        if nun == 2 or boolean:
            score += 2; why.append("binary")
        elif 3 <= nun <= 10:
            score += 1; why.append(f"{nun} classes")
        elif numeric and nun > 20:
            score += 0.5; why.append("continuous numeric")
        if i == 0 or i == df.shape[1] - 1:
            score += 1; why.append("first column" if i == 0 else "last column")
        if nun <= 1:
            score -= 8; why.append("constant")
        if nun / max(y.notna().sum(), 1) > 0.9 and not (numeric and y.dtype.kind == "f"):
            score -= 6; why.append("almost every value is unique, like an ID")
        if ID_NAME.search(name) or PII_NAME.search(name):
            score -= 6; why.append("named like an ID or personal data")
        if pd.api.types.is_datetime64_any_dtype(y):
            score -= 6; why.append("date column")
        if is_text_like(y):
            v = probe[c].dropna().astype(str).head(2000)
            if len(v) and v.str.contains(DATE_HINT_RE).mean() > 0.9 and pd.to_numeric(v, errors="coerce").notna().mean() < 0.5:
                score -= 6; why.append("looks like dates")
            elif len(v) and v.str.len().mean() > 30:
                score -= 6; why.append("free text")
            elif nun > 50:
                score -= 3; why.append("high cardinality text")
        if miss > 50:
            score -= 2; why.append(f"{miss:.0f}% missing")
        if LEAK_HINTS.search(name) and not TARGET_STRONG.search(name) and not TARGET_MEDIUM.search(name):
            score -= 0.5; why.append("name suggests data recorded after the outcome")
        rows.append(dict(column=c, score=score, reasons=why, dtype=str(y.dtype), unique=nun, missing_pct=miss,
                         suggested_type=kind))
    rows.sort(key=lambda r: -r["score"])
    return rows[:top] if top else rows


def match_column(df, name):
    """Return (exact_match, close_matches) for a column name the user typed."""
    import difflib
    cols = [str(c) for c in df.columns]
    if name in cols:
        return name, []
    lower = {c.lower().replace(" ", "_"): c for c in cols}
    key = str(name).lower().replace(" ", "_")
    if key in lower:
        return None, [lower[key]]
    return None, difflib.get_close_matches(str(name), cols, n=3, cutoff=0.6)


def describe_all(df, id_cols=None):
    """Privacy safe df.describe(include="all") for use in notebooks."""
    return EDAReport(df, id_cols=id_cols).describe_all()


def eda_report(df, target=None, date_col=None, id_cols=None, name="dataset",
               sample_rows=200_000, max_plots=24, show=True, out_dir=None, target_type=None,
               corr_threshold=HIGH_CORR, cramers_threshold=HIGH_CRAMERS_V, spec_path=None, mode=None,
               log_scale=False):
    """Run the full EDA. In a notebook, display every section inline.

    Pass spec_path (for example "ml_spec.yaml") to check the data against the framing in the shared spec and
    write the results to its eda section. The spec is created if it does not exist.

    Pass out_dir (for example "docs") to also write a markdown report with PNG plots.
    Returns the EDAReport object, so single sections can be shown again with eda.show("missing").
    """
    spec = None
    if spec_path and os.path.exists(spec_path):
        from ml_common import load
        spec = load(spec_path)
    eda = EDAReport(df, target=target, date_col=date_col, id_cols=id_cols, name=name,
                    sample_rows=sample_rows, max_plots=max_plots, target_type=target_type,
                    corr_threshold=corr_threshold, cramers_threshold=cramers_threshold, spec=spec, mode=mode,
                    log_scale=log_scale).run()
    if show:
        eda.show()
    report = None
    if out_dir:
        report = eda.to_markdown(out_dir)
    if spec_path:
        eda.write_spec(spec_path, source=None, report=report)
    return eda
