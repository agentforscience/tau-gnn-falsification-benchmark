"""Evaluation metrics and statistics for the tau-propagation benchmark.

Design constraint (see planning.md, "Non-negotiable evaluation constraint"): absolute-tau
spatial correlation is nearly uninformative because tau maps autocorrelate with themselves.
Every comparison here therefore reports BOTH the absolute-scale metric (for comparability with
the published literature) AND the Delta metric (change from baseline), plus the persistence null.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Pseudo-count for the log transform. The mouse pathology measures are SEMI-QUANTITATIVE
# ORDINAL grading scales (Kaufman: steps of 0.25 on [0, 2.75], 51% exact zeros; eNDM: similar).
# A tiny offset such as 1e-6 would place log10(0) six decades below the smallest observable
# grade, manufacturing an enormous artificial gap that dominates the correlation. We therefore
# use half the coarsest quantisation step. See REPORT.md, "Deviations from the plan".
LOG_EPS = 0.125


def logp(x: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    """log10 pathology with a quantisation-matched offset (cf. Cornblath 2021, Henderson 2019)."""
    return np.log10(np.clip(x, 0, None) + eps)


def transform(x: np.ndarray, mode: str = "raw") -> np.ndarray:
    """Apply the evaluation-scale transform. 'raw' is the Nexis convention for ordinal
    pathology grades; 'log' is the Cornblath/Henderson convention for continuous measures."""
    if mode == "raw":
        return np.asarray(x, float)
    if mode == "log":
        return logp(x)
    raise ValueError(mode)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson R, NaN-safe, returns nan if either vector is constant."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(stats.spearmanr(a[m], b[m]).statistic)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination against the mean of `y` (can be negative)."""
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fisher_z(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def fisher_z_test(r1: float, r2: float, n: int) -> tuple[float, float]:
    """Independent-sample Fisher r-to-z comparison of two correlations on n observations.

    Returns (z, two-sided p). Conservative here because the two correlations share the same
    target vector; we use it only as a descriptive effect-size scale and rely on the paired
    Wilcoxon across held-out experiments for inference.
    """
    if not np.isfinite(r1) or not np.isfinite(r2) or n < 4:
        return np.nan, np.nan
    se = np.sqrt(2.0 / (n - 3))
    z = (fisher_z(r1) - fisher_z(r2)) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def paired_test(a: list[float], b: list[float]) -> dict:
    """Paired comparison of two model scores across held-out folds.

    Reports Wilcoxon signed-rank (primary; no normality assumption, appropriate for n<=14),
    a paired t-test for reference, Cohen's dz, and a bootstrap CI on the mean difference.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    out = {"n": int(len(d)), "mean_a": float(a.mean()) if len(a) else np.nan,
           "mean_b": float(b.mean()) if len(b) else np.nan,
           "mean_diff": float(d.mean()) if len(d) else np.nan}
    if len(d) < 3 or np.allclose(d, 0):
        out.update(wilcoxon_p=np.nan, t_p=np.nan, cohens_dz=np.nan,
                   ci_low=np.nan, ci_high=np.nan, n_wins=int((d > 0).sum()))
        return out
    out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    out["cohens_dz"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
    lo, hi = bootstrap_ci(d)
    out["ci_low"], out["ci_high"] = lo, hi
    out["n_wins"] = int((d > 0).sum())
    return out


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of x."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def empirical_p(observed: float, null: np.ndarray, tail: str = "greater") -> float:
    """(1 + #{null at least as extreme}) / (1 + N) -- the standard permutation p-value."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return np.nan
    k = (null >= observed).sum() if tail == "greater" else (null <= observed).sum()
    return float((1 + k) / (1 + len(null)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray | None = None,
             log: bool = True) -> dict:
    """Full metric set for one prediction.

    Args:
        y_true: observed pathology at the target timepoint.
        y_pred: model prediction at that timepoint.
        y_base: pathology at the *baseline* timepoint (or the seed). If given, Delta metrics
                are computed, which is the informative comparison.
        log:    evaluate on log10 pathology (mouse convention) vs raw (human SUVR convention).
    """
    f = logp if log else (lambda z: np.asarray(z, float))  # noqa: E731
    yt, yp = f(y_true), f(y_pred)
    out = {"R": pearson(yt, yp), "rho": spearman(yt, yp), "R2": r2_score(yt, yp)}
    if y_base is not None:
        yb = f(y_base)
        dt, dp = yt - yb, yp - yb
        out.update({"dR": pearson(dt, dp), "drho": spearman(dt, dp), "dR2": r2_score(dt, dp)})
        # persistence null: predict "no change"
        out["R_persist"] = pearson(yt, yb)
        out["dR2_persist"] = r2_score(dt, np.zeros_like(dt))
    return out
