"""Static report figures (PNG) with one consistent, colour-blind-checked style.

Palette (validated with the dataviz skill's checker, light surface #fcfcfb):
blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100 - always in this order.
Every multi-series chart has a legend and the numbers also appear in report tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EMPHASIS_OFF = "#c3c2b7"


def _style(ax, grid_axis: str = "y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _fig(w=6.4, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def scatter_pred_vs_true(y, p, path, title, unit="%", lims=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    fig, ax = _fig(5.0, 4.8)
    _style(ax, grid_axis="both")
    lo, hi = lims or (min(y.min(), p.min()) - 2, max(y.max(), p.max()) + 2)
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    ax.scatter(y, p, s=22, color=SERIES[0], alpha=0.75, edgecolor=SURFACE, linewidth=1, zorder=2)
    r = np.corrcoef(y, p)[0, 1]
    mae = np.mean(np.abs(p - y))
    ax.text(0.03, 0.97, f"n = {len(y)}\nr = {r:.2f}\nMAE = {mae:.1f} {unit}", transform=ax.transAxes,
            va="top", fontsize=9, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Echo (truth), {unit}")
    ax.set_ylabel(f"CardioTwin from ECG, {unit}")
    ax.set_title(title, loc="left", fontsize=11)
    _save(fig, path)


def bland_altman(y, p, path, title, unit="%"):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    mean, diff = (y + p) / 2, p - y
    bias, sd = diff.mean(), diff.std()
    fig, ax = _fig(5.6, 4.0)
    _style(ax, grid_axis="y")
    ax.scatter(mean, diff, s=20, color=SERIES[0], alpha=0.7, edgecolor=SURFACE, linewidth=1)
    for v, lab in ((bias, f"bias {bias:+.1f}"), (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                   (bias - 1.96 * sd, f"-1.96 SD {bias - 1.96 * sd:+.1f}")):
        ax.axhline(v, color=INK2 if lab.startswith("bias") else MUTED, lw=1)
        ax.text(ax.get_xlim()[1], v, f" {lab}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel(f"Mean of echo and CardioTwin, {unit}")
    ax.set_ylabel(f"CardioTwin - echo, {unit}")
    ax.set_title(title, loc="left", fontsize=11)
    _save(fig, path)


def roc_curves(curves: list[tuple[str, np.ndarray, np.ndarray, float]], path, title):
    """curves: (label, y_true, score, auc)."""
    fig, ax = _fig(5.0, 4.8)
    _style(ax, grid_axis="both")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    for i, (label, y, s, auc) in enumerate(curves[:4]):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=SERIES[i], lw=2, solid_capstyle="round", label=f"{label}  (AUC {auc:.2f})")
    ax.set_xlabel("False-positive rate (1 - specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    ax.set_title(title, loc="left", fontsize=11)
    _save(fig, path)


def bar_with_ci(labels, values, lows, highs, path, title, xlabel, highlight: set[int] | None = None, fmt="{:.2f}"):
    """Horizontal bars with 95% CI whiskers; highlighted bars in blue, the rest gray."""
    n = len(labels)
    fig, ax = _fig(6.4, 0.55 * n + 1.3)
    _style(ax, grid_axis="x")
    y = np.arange(n)[::-1]
    colors = [SERIES[0] if (highlight is None or i in highlight) else EMPHASIS_OFF for i in range(n)]
    ax.barh(y, values, height=0.5, color=colors, zorder=2)
    err = [np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)]
    ax.errorbar(values, y, xerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for yi, v, hi in zip(y, values, highs):
        ax.text(max(hi, v), yi, "  " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(highs) * 1.18 if len(highs) else 1)
    ax.set_title(title, loc="left", fontsize=11)
    _save(fig, path)


def km_curves(groups: list[tuple[str, np.ndarray, np.ndarray]], path, title, horizon_years: float | None = None):
    """groups: (label, times_in_years, survival)."""
    fig, ax = _fig(6.0, 4.2)
    _style(ax, grid_axis="y")
    for i, (label, t, s) in enumerate(groups[:4]):
        ax.step(t, s, where="post", color=SERIES[i], lw=2, label=label)
    if horizon_years:
        ax.set_xlim(0, horizon_years)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Years after the ECG")
    ax.set_ylabel("Free of heart-failure event")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    ax.set_title(title, loc="left", fontsize=11)
    _save(fig, path)


def multi_target_bars(names, r_values, lows, highs, path, title):
    bar_with_ci(names, r_values, lows, highs, path, title, "Pearson r (CardioTwin vs echo), 95% CI")


def ecg_with_saliency(x: np.ndarray, sal: np.ndarray, fs: float, leads: list[str], path, title):
    n_leads = x.shape[0]
    t = np.arange(x.shape[1]) / fs
    fig, axes = plt.subplots(n_leads, 1, figsize=(9.5, 1.5 * n_leads + 0.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        _style(ax, grid_axis="")
        ax.fill_between(t, x[i].min() - 0.1, x[i].max() + 0.1, where=sal[i] > 0.35, color=SERIES[1],
                        alpha=0.18, lw=0, step="mid")
        ax.plot(t, x[i], color=INK, lw=0.8)
        ax.set_ylabel(leads[i], rotation=0, labelpad=14, fontsize=10, color=INK)
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(title, loc="left", fontsize=11)
    _save(fig, path)


def leakage_bars(labels, maes, path, title):
    fig, ax = _fig(7.2, 2.9)
    _style(ax, grid_axis="x")
    y = np.arange(len(labels))[::-1]
    ax.barh(y, maes, height=0.45, color=[SERIES[1], SERIES[0]][: len(labels)])
    for yi, v in zip(y, maes):
        ax.text(v, yi, f"  {v:.1f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK2)
    ax.set_xlabel("Test EF error (MAE, %-points)")
    ax.set_xlim(0, max(maes) * 1.25)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    _save(fig, path)


def segment_bars(segments: dict[str, float], path, title):
    names = list(segments)
    fig, ax = _fig(5.4, 2.8)
    _style(ax, grid_axis="x")
    y = np.arange(len(names))[::-1]
    vals = [segments[n] * 100 for n in names]
    ax.barh(y, vals, height=0.45, color=SERIES[0])
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK2)
    ax.set_xlabel("Share of EF attribution (integrated gradients)")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title(title, loc="left", fontsize=11)
    _save(fig, path)
