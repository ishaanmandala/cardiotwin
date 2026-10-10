"""Level 5: put an ECG on the screen, drawn on classic ECG paper.

Standard ECG paper runs at 25 mm/s and 10 mm/mV: one small square is 0.04 s by
0.1 mV, one big square is 0.2 s by 0.5 mV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .io import ECG  # noqa: E402


def plot_ecg(
    ecg: ECG, out_path: str | Path | None = None, seconds: float | None = 10.0,
    title: str | None = None, saliency: np.ndarray | None = None, row_mv: float = 3.0,
):
    """Draw every lead in its own row on ECG paper. Returns the matplotlib Figure.

    Args:
        ecg: the recording (any number of leads).
        out_path: optional PNG/SVG/PDF path to save to.
        seconds: how much of the recording to show (None = everything).
        title: figure title.
        saliency: optional ``(leads, samples)`` array in [0, 1] drawn as a red
            highlight behind the trace (see :mod:`cardiotwin.explain`).
        row_mv: vertical space per lead in millivolts.
    """
    n = ecg.n_samples if seconds is None else min(ecg.n_samples, int(seconds * ecg.fs))
    t = np.arange(n) / ecg.fs
    dur = n / ecg.fs
    n_leads = len(ecg.leads)
    # 25 mm/s and 10 mm/mV -> inches (1 inch = 25.4 mm)
    width_in = max(6.0, dur * 25 / 25.4)
    height_in = max(2.0, n_leads * row_mv * 10 / 25.4)
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fff5f5")
    y_top = 0.0
    y_bottom = -n_leads * row_mv
    for x in np.arange(0, dur + 1e-9, 0.04):
        major = abs(x / 0.2 - round(x / 0.2)) < 1e-6
        ax.axvline(x, color="#f4a6a6" if major else "#fbd5d5", lw=0.8 if major else 0.4, zorder=0)
    for y in np.arange(y_bottom, y_top + 1e-9, 0.1):
        major = abs(y / 0.5 - round(y / 0.5)) < 1e-6
        ax.axhline(y, color="#f4a6a6" if major else "#fbd5d5", lw=0.8 if major else 0.4, zorder=0)
    for i, name in enumerate(ecg.leads):
        offset = -(i + 0.5) * row_mv
        sig = ecg.signal[i, :n]
        if saliency is not None:
            sal = np.asarray(saliency)[i, :n]
            ax.fill_between(t, offset - row_mv / 2, offset + row_mv / 2, where=sal > 0.5,
                            color="#ff3b30", alpha=0.12, step="mid", zorder=1, lw=0)
        ax.plot(t, sig - np.median(sig) + offset, color="#111111", lw=0.8, zorder=2)
        ax.text(0.05, offset + row_mv * 0.28, name, fontsize=10, fontweight="bold", zorder=3)
    ax.set_xlim(0, dur)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xticks(np.arange(0, dur + 1e-9, 1.0))
    ax.set_yticks([])
    ax.set_xlabel("time (s)   |   25 mm/s, 10 mm/mV")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100)
    return fig
