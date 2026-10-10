"""Metrics with honest uncertainty.

Confidence intervals use a *cluster* bootstrap: we resample patients, not ECGs,
because several ECGs (or 10-second windows) from one person are not
independent evidence.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def regression_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n = len(y)
    if n < 3:
        return {"n": n}
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    rho = float(stats.spearmanr(y, p).statistic) if np.std(p) > 0 else float("nan")
    return {
        "n": int(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": r,
        "spearman_rho": rho,
        "bias": float(np.mean(err)),
        "loa_low": float(np.mean(err) - 1.96 * np.std(err)),
        "loa_high": float(np.mean(err) + 1.96 * np.std(err)),
    }


def binary_metrics(y_true, score, threshold: float | None = None) -> dict:
    y, s = np.asarray(y_true, float), np.asarray(score, float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok].astype(int), s[ok]
    n_pos = int(y.sum())
    out = {"n": int(len(y)), "n_pos": n_pos, "prevalence": float(y.mean()) if len(y) else float("nan")}
    if n_pos == 0 or n_pos == len(y):
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    if threshold is None:  # Youden's J
        order = np.unique(s)
        best, threshold = -1.0, float(order[0])
        for t in order:
            pred = s >= t
            sens = (pred & (y == 1)).sum() / n_pos
            spec = (~pred & (y == 0)).sum() / (len(y) - n_pos)
            if sens + spec - 1 > best:
                best, threshold = sens + spec - 1, float(t)
    pred = s >= threshold
    out["threshold"] = float(threshold)
    out["sensitivity"] = float((pred & (y == 1)).sum() / n_pos)
    out["specificity"] = float((~pred & (y == 0)).sum() / (len(y) - n_pos))
    return out


def ef_classification_metrics(ef_true, ef_pred, cutoffs=(40.0, 50.0)) -> dict:
    """Detecting low EF from a *regression* output: score = -predicted EF."""
    out = {}
    for c in cutoffs:
        y = (np.asarray(ef_true, float) <= c).astype(float)
        y[~np.isfinite(np.asarray(ef_true, float))] = np.nan
        m = binary_metrics(y, -np.asarray(ef_pred, float))
        out[f"ef_le_{int(c)}"] = {k: m[k] for k in ("n", "n_pos", "auroc", "auprc", "sensitivity", "specificity")
                                  if k in m}
    return out


def cluster_bootstrap(
    df: pd.DataFrame, fn: Callable[[pd.DataFrame], float], cluster_col: str = "patient_id",
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and (1-alpha) percentile CI of ``fn(df)``, resampling clusters."""
    rng = np.random.default_rng(seed)
    point = fn(df)
    groups = df.groupby(cluster_col).indices
    keys = list(groups)
    idx_lists = [groups[k] for k in keys]
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_lists[i] for i in pick])
        try:
            v = fn(df.iloc[idx])
        except ValueError:
            continue
        if v is not None and np.isfinite(v):
            values.append(v)
    if not values:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_difference(
    df: pd.DataFrame, fn_a: Callable[[pd.DataFrame], float], fn_b: Callable[[pd.DataFrame], float],
    cluster_col: str = "patient_id", n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Bootstrap the difference fn_b - fn_a on the same resampled patients.

    Returns the point difference, 95% CI and a two-sided bootstrap p-value for
    "no difference".
    """
    rng = np.random.default_rng(seed)
    diff = fn_b(df) - fn_a(df)
    groups = df.groupby(cluster_col).indices
    idx_lists = list(groups.values())
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_lists), len(idx_lists))
        sub = df.iloc[np.concatenate([idx_lists[i] for i in pick])]
        try:
            d = fn_b(sub) - fn_a(sub)
        except ValueError:
            continue
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {"diff": float(diff), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(diff), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, p))}


def mae_fn(true_col: str, pred_col: str) -> Callable[[pd.DataFrame], float]:
    def f(d: pd.DataFrame) -> float:
        ok = d[true_col].notna() & d[pred_col].notna()
        return float(np.mean(np.abs(d.loc[ok, pred_col] - d.loc[ok, true_col])))
    return f


def r_fn(true_col: str, pred_col: str) -> Callable[[pd.DataFrame], float]:
    def f(d: pd.DataFrame) -> float:
        ok = d[true_col].notna() & d[pred_col].notna()
        return float(np.corrcoef(d.loc[ok, true_col], d.loc[ok, pred_col])[0, 1])
    return f


def r2_fn(true_col: str, pred_col: str) -> Callable[[pd.DataFrame], float]:
    def f(d: pd.DataFrame) -> float:
        ok = d[true_col].notna() & d[pred_col].notna()
        y, p = d.loc[ok, true_col].to_numpy(float), d.loc[ok, pred_col].to_numpy(float)
        return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
    return f


def auroc_fn(label_fn: Callable[[pd.Series], pd.Series], true_col: str, score_col: str, sign: float = 1.0):
    def f(d: pd.DataFrame) -> float:
        ok = d[true_col].notna() & d[score_col].notna()
        y = label_fn(d.loc[ok, true_col]).astype(int)
        if y.nunique() < 2:
            raise ValueError("single class")
        return float(roc_auc_score(y, sign * d.loc[ok, score_col]))
    return f


def interval_coverage(y_true, mean, sd, z: float = 1.645) -> float:
    """Share of truths inside mean +/- z*sd (z=1.645 -> nominal 90%)."""
    y, m, s = (np.asarray(a, float) for a in (y_true, mean, sd))
    ok = np.isfinite(y) & np.isfinite(m) & np.isfinite(s)
    return float(np.mean(np.abs(y[ok] - m[ok]) <= z * s[ok])) if ok.any() else float("nan")
