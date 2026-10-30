"""Statistics for the A/B testing skills: sample ratio checks, lift with confidence intervals, power and sample size,
variance reduction, multiple testing corrections, balance checks, and heterogeneity tests.

Uses numpy, scipy, and pandas only, so it runs in any project set up by ab-project-setup or ml-project-setup. Every
function returns plain dicts of numbers, so notebooks can show them as tables and write them to the spec.

The canonical copy lives in claude-skills/_ab_shared/ab_core.py, copied into the skills that compute statistics by
tools/sync_ab_shared.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

SRM_ALPHA = 0.001


# ---------------------------------------------------------------- helpers

def _z(alpha, sides):
    return stats.norm.ppf(1 - alpha / (2 if sides == "two" else 1))


def _p_from_z(z, sides, direction):
    """p value for a z (or large sample t) statistic. One sided tests look in the expected direction."""
    if sides == "two":
        return float(2 * stats.norm.sf(abs(z)))
    return float(stats.norm.sf(z) if direction != "decrease" else stats.norm.cdf(z))


def _p_from_t(t, df, sides, direction):
    if sides == "two":
        return float(2 * stats.t.sf(abs(t), df))
    return float(stats.t.sf(t, df) if direction != "decrease" else stats.t.cdf(t, df))


def fmt_p(p):
    """A p value for reports: three significant figures, or 'below 0.0001' for very small values."""
    if p is None:
        return "not computed"
    return "below 0.0001" if p < 1e-4 else f"{p:.3g}"


# ---------------------------------------------------------------- sample ratio mismatch

def srm_test(counts, expected_shares, alpha=SRM_ALPHA):
    """Chi-square goodness of fit of the units per variant against the planned shares.

    counts: {variant: units}. expected_shares: {variant: planned share}, which need not add up to exactly 1.
    A p value below alpha (0.001 by default) is a sample ratio mismatch: the assignment or the logging is broken,
    and the effect estimates cannot be trusted until the cause is found.
    """
    names = [k for k in expected_shares if k in counts]
    obs = np.array([counts[k] for k in names], dtype=float)
    shares = np.array([expected_shares[k] for k in names], dtype=float)
    shares = shares / shares.sum()
    exp = shares * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    return {
        "observed": {k: int(c) for k, c in zip(names, obs)},
        "expected": {k: float(e) for k, e in zip(names, exp)},
        "observed_shares": {k: float(c / obs.sum()) for k, c in zip(names, obs)},
        "expected_shares": {k: float(s) for k, s in zip(names, shares)},
        "chi2": float(chi2), "df": len(names) - 1, "p_value": float(p), "alpha": alpha,
        "passed": bool(p >= alpha),
    }


# ---------------------------------------------------------------- lift

def two_proportions(x_c, n_c, x_t, n_t, alpha=0.05, sides="two", direction="increase"):
    """Compare two conversion rates.

    The p value uses the pooled z test (the same as statsmodels proportions_ztest). The confidence interval of the
    absolute difference uses the unpooled standard error, and the interval of the relative lift uses the delta
    method on the ratio of the two rates.
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    pooled = (x_c + x_t) / (n_c + n_t)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = diff / se0 if se0 > 0 else 0.0
    se1 = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = _z(alpha, "two")  # intervals are always two sided, so they read the same way in every report
    out = {"method": "two proportion z test", "control_mean": p_c, "treatment_mean": p_t,
           "n_control": int(n_c), "n_treatment": int(n_t), "diff": diff,
           "diff_ci": [diff - zc * se1, diff + zc * se1], "stat": z, "p_value": _p_from_z(z, sides, direction)}
    out.update(_relative(p_c, p_t, p_c * (1 - p_c) / n_c, p_t * (1 - p_t) / n_t, zc))
    return out


def welch(mean_c, var_c, n_c, mean_t, var_t, n_t, alpha=0.05, sides="two", direction="increase"):
    """Compare two means with Welch's t test (unequal variances)."""
    diff = mean_t - mean_c
    vc, vt = var_c / n_c, var_t / n_t
    se = math.sqrt(vc + vt)
    t = diff / se if se > 0 else 0.0
    df = (vc + vt) ** 2 / ((vc ** 2) / (n_c - 1) + (vt ** 2) / (n_t - 1)) if se > 0 else float(n_c + n_t - 2)
    tc = stats.t.ppf(1 - alpha / 2, df)
    out = {"method": "Welch t test", "control_mean": mean_c, "treatment_mean": mean_t,
           "n_control": int(n_c), "n_treatment": int(n_t), "diff": diff,
           "diff_ci": [diff - tc * se, diff + tc * se], "stat": t, "df": df,
           "p_value": _p_from_t(t, df, sides, direction)}
    out.update(_relative(mean_c, mean_t, vc, vt, stats.norm.ppf(1 - alpha / 2)))
    return out


def _relative(m_c, m_t, var_mean_c, var_mean_t, zc):
    """Relative lift m_t / m_c - 1, with a delta method interval. var_mean_* are variances of the means."""
    if not m_c:
        return {"rel_lift": None, "rel_ci": [None, None]}
    rel = m_t / m_c - 1
    se = math.sqrt(var_mean_t / m_c ** 2 + (m_t ** 2) * var_mean_c / m_c ** 4)
    return {"rel_lift": rel, "rel_ci": [rel - zc * se, rel + zc * se]}


def compare(df, group_col, metric_col, control, treatment, metric_type="binary", alpha=0.05, sides="two",
            direction="increase"):
    """Compare one metric between the control and one treatment in a unit level DataFrame."""
    c = df.loc[df[group_col] == control, metric_col].dropna().astype(float)
    t = df.loc[df[group_col] == treatment, metric_col].dropna().astype(float)
    if metric_type == "binary":
        return two_proportions(c.sum(), len(c), t.sum(), len(t), alpha, sides, direction)
    return welch(c.mean(), c.var(ddof=1), len(c), t.mean(), t.var(ddof=1), len(t), alpha, sides, direction)


def bootstrap(c_values, t_values, alpha=0.05, n_boot=2000, seed=0):
    """Percentile bootstrap intervals for the difference in means and the relative lift. A check on the normal
    approximation, most useful for skewed metrics such as revenue."""
    rng = np.random.default_rng(seed)
    c, t = np.asarray(c_values, dtype=float), np.asarray(t_values, dtype=float)
    mc = np.array([rng.choice(c, len(c)).mean() for _ in range(n_boot)])
    mt = np.array([rng.choice(t, len(t)).mean() for _ in range(n_boot)])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rel = mt / np.where(mc == 0, np.nan, mc) - 1
    return {"method": f"percentile bootstrap, {n_boot:,} resamples",
            "diff_ci": [float(np.percentile(mt - mc, lo)), float(np.percentile(mt - mc, hi))],
            "rel_ci": [float(np.nanpercentile(rel, lo)), float(np.nanpercentile(rel, hi))]}


def label_result(res, mde_abs=None, direction="increase", alpha=0.05):
    """A plain label for a comparison: whether it is significant, in which direction, and whether the interval
    clears the smallest effect worth acting on."""
    sig = res["p_value"] < alpha
    good = (res["diff"] > 0) == (direction != "decrease")
    lo, hi = res["diff_ci"]
    if not sig:
        if mde_abs is not None and -abs(mde_abs) < lo and hi < abs(mde_abs):
            return "no meaningful effect: the interval rules out the minimum effect in both directions"
        return "inconclusive: not significant, and the interval still includes effects worth acting on"
    if mde_abs is None:
        return "significant improvement" if good else "significant harm"
    clears = (lo >= abs(mde_abs)) if direction != "decrease" else (hi <= -abs(mde_abs))
    if good:
        return "significant and practically meaningful improvement" if clears else \
            "significant improvement, but the interval does not clear the minimum effect"
    return "significant harm"


# ---------------------------------------------------------------- variance reduction

def cuped(y, x):
    """CUPED: y minus theta times (x minus its mean), with theta = cov(x, y) / var(x), estimated on all units.
    x must be recorded before assignment. Returns (adjusted y, theta, share of variance removed)."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    vx = x.var(ddof=1)
    theta = float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    adj = y - theta * (x - x.mean())
    vy = y.var(ddof=1)
    return adj, theta, float(1 - adj.var(ddof=1) / vy) if vy > 0 else 0.0


# ---------------------------------------------------------------- power and sample size

def sample_size_proportions(p_c, mde_abs, alpha=0.05, power=0.8, sides="two", ratio=1.0):
    """Units needed in the control (and ratio times as many in the treatment) to detect an absolute change of
    mde_abs from a baseline rate p_c."""
    p_t = p_c + mde_abs
    if not 0 < p_t < 1:
        raise ValueError(f"The baseline {p_c} plus the effect {mde_abs} is outside 0 to 1.")
    za, zb = _z(alpha, sides), stats.norm.ppf(power)
    pbar = (p_c + ratio * p_t) / (1 + ratio)
    num = za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / ratio)) + zb * math.sqrt(p_c * (1 - p_c) + p_t * (1 - p_t) / ratio)
    n_c = math.ceil(num ** 2 / mde_abs ** 2)
    return {"n_control": n_c, "n_treatment": math.ceil(n_c * ratio), "total": n_c + math.ceil(n_c * ratio)}


def sample_size_means(sd, mde_abs, alpha=0.05, power=0.8, sides="two", ratio=1.0):
    za, zb = _z(alpha, sides), stats.norm.ppf(power)
    n_c = math.ceil((za + zb) ** 2 * sd ** 2 * (1 + 1 / ratio) / mde_abs ** 2)
    return {"n_control": n_c, "n_treatment": math.ceil(n_c * ratio), "total": n_c + math.ceil(n_c * ratio)}


def power_proportions(p_c, mde_abs, n_c, n_t, alpha=0.05, sides="two"):
    """The chance of a significant result when the true absolute change is mde_abs."""
    p_t = p_c + mde_abs
    pbar = (n_c * p_c + n_t * p_t) / (n_c + n_t)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n_c + 1 / n_t))
    se1 = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    return float(stats.norm.cdf((abs(mde_abs) - _z(alpha, sides) * se0) / se1))


def power_means(sd, mde_abs, n_c, n_t, alpha=0.05, sides="two"):
    se = sd * math.sqrt(1 / n_c + 1 / n_t)
    return float(stats.norm.cdf(abs(mde_abs) / se - _z(alpha, sides)))


def mde_proportions(p_c, n_c, n_t, alpha=0.05, power=0.8, sides="two"):
    """The smallest absolute increase from p_c that this sample detects with the given power."""
    from scipy.optimize import brentq
    hi = (1 - p_c) * 0.999
    if power_proportions(p_c, hi, n_c, n_t, alpha, sides) < power:
        return None
    return float(brentq(lambda d: power_proportions(p_c, d, n_c, n_t, alpha, sides) - power, 1e-9, hi))


def mde_means(sd, n_c, n_t, alpha=0.05, power=0.8, sides="two"):
    return float((_z(alpha, sides) + stats.norm.ppf(power)) * sd * math.sqrt(1 / n_c + 1 / n_t))


def duration_days(total_units, units_per_day, min_days=7):
    """Days to collect the sample, rounded up to whole weeks so every weekday is covered equally."""
    if not units_per_day:
        return None
    days = math.ceil(total_units / units_per_day)
    return max(min_days, 7 * math.ceil(days / 7))


# ---------------------------------------------------------------- multiple testing

def holm(pvalues):
    """Holm adjusted p values (controls the chance of any false positive). Same order as the input."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p values (controls the share of false discoveries). Same order as the input."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, p[i] * m / (m - rank))
        adj[i] = running
    return adj.tolist()


# ---------------------------------------------------------------- balance and segments

def balance(df, group_col, cols, control, treatment):
    """Covariate balance between two variants on columns recorded before assignment. Numeric columns get the
    standardised mean difference (above 0.1 is usually a concern); other columns get a chi-square test."""
    rows = []
    c, t = df[df[group_col] == control], df[df[group_col] == treatment]
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype.kind in "biuf" and s.nunique() > 2:
            mc, mt = c[col].mean(), t[col].mean()
            sd = math.sqrt((c[col].var(ddof=1) + t[col].var(ddof=1)) / 2) or float("nan")
            smd = (mt - mc) / sd if sd == sd and sd > 0 else 0.0
            rows.append({"column": col, "kind": "numeric", "control": float(mc), "treatment": float(mt),
                         "smd": float(smd), "p_value": None, "flag": bool(abs(smd) > 0.1)})
        else:
            tab = np.array([[(c[col] == v).sum(), (t[col] == v).sum()] for v in s.dropna().unique()])
            tab = tab[tab.sum(axis=1) > 0]
            p = float(stats.chi2_contingency(tab)[1]) if tab.shape[0] > 1 else 1.0
            rows.append({"column": col, "kind": "categorical", "control": None, "treatment": None, "smd": None,
                         "p_value": p, "flag": bool(p < 0.01)})
    return rows


def heterogeneity(df, group_col, metric_col, segment_col, control, treatment, metric_type="binary", alpha=0.05,
                  min_units=100):
    """The effect in each segment of one column, and Cochran's Q test of whether the effects differ between
    segments. Segments with fewer than min_units in either variant are left out of the test and marked."""
    rows, diffs, weights = [], [], []
    for seg, part in df.groupby(segment_col, dropna=False, observed=True):
        nc, nt = (part[group_col] == control).sum(), (part[group_col] == treatment).sum()
        if min(nc, nt) < min_units:
            rows.append({"segment": str(seg), "n_control": int(nc), "n_treatment": int(nt), "too_small": True})
            continue
        r = compare(part, group_col, metric_col, control, treatment, metric_type, alpha)
        se = (r["diff_ci"][1] - r["diff_ci"][0]) / (2 * stats.norm.ppf(1 - alpha / 2))
        rows.append({"segment": str(seg), "n_control": int(nc), "n_treatment": int(nt), "too_small": False,
                     "control_mean": r["control_mean"], "treatment_mean": r["treatment_mean"], "diff": r["diff"],
                     "diff_ci": r["diff_ci"], "rel_lift": r["rel_lift"], "p_value": r["p_value"]})
        if se > 0:
            diffs.append(r["diff"])
            weights.append(1 / se ** 2)
    q = p = None
    if len(diffs) > 1:
        w, d = np.array(weights), np.array(diffs)
        pooled = (w * d).sum() / w.sum()
        q = float((w * (d - pooled) ** 2).sum())
        p = float(stats.chi2.sf(q, len(diffs) - 1))
    return {"column": segment_col, "segments": rows, "q": q, "df": len(diffs) - 1 if diffs else 0,
            "p_value": p}
