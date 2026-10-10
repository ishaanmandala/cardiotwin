"""Level 15: which parts of the ECG does CardioTwin look at?

* :func:`integrated_gradients` - per-sample attribution for one target.
* :func:`segment_attribution` - how much of that attribution falls on the
  P wave, QRS complex, T wave or elsewhere (aligned to detected R peaks).
* :func:`lead_occlusion` - how much the prediction moves when a lead is blanked.
"""

from __future__ import annotations

import numpy as np
import torch

from .ecg.features import detect_r_peaks


def integrated_gradients(
    models: list[torch.nn.Module], x: np.ndarray, tab: np.ndarray | None, target_idx: int, steps: int = 24,
) -> np.ndarray:
    """Integrated gradients (zero baseline) of the predicted mean, averaged over an ensemble.

    Returns |attribution| with the same shape as ``x`` (leads, samples).
    """
    xt = torch.as_tensor(np.asarray(x, np.float32))[None]
    tt = torch.as_tensor(np.asarray(tab, np.float32))[None] if tab is not None and len(tab) else None
    alphas = torch.linspace(1.0 / steps, 1.0, steps)[:, None, None]
    total = torch.zeros_like(xt[0])
    for model in models:
        model.eval()
        scaled = (alphas * xt).requires_grad_(True)
        tab_in = tt.expand(steps, -1) if tt is not None else None
        out = model(scaled, tab_in)["mean"][:, target_idx].sum()
        (grad,) = torch.autograd.grad(out, scaled)
        total += (xt[0] * grad.mean(dim=0)).detach()
    return np.abs(total.numpy() / len(models))


def smooth_saliency(sal: np.ndarray, fs: float, win_s: float = 0.04) -> np.ndarray:
    """Moving-average smoothing and scaling to [0, 1] (99th percentile = 1)."""
    k = max(1, int(win_s * fs))
    kernel = np.ones(k) / k
    sm = np.stack([np.convolve(s, kernel, mode="same") for s in sal])
    hi = np.percentile(sm, 99) or 1.0
    return np.clip(sm / hi, 0, 1).astype(np.float32)


def segment_attribution(x: np.ndarray, sal: np.ndarray, fs: float) -> dict[str, float]:
    """Share of total attribution in P, QRS, T and other segments (beat-aligned)."""
    r = detect_r_peaks(x, fs)
    n = x.shape[-1]
    labels = np.full(n, "other", dtype=object)
    for p in r:
        # Later assignments win, so the QRS window overrides overlapping P/T windows.
        for a, b, name in ((-0.25, -0.07, "P"), (0.07, 0.45, "T"), (-0.07, 0.07, "QRS")):
            lo, hi = max(0, p + int(a * fs)), min(n, p + int(b * fs))
            if hi > lo:
                labels[lo:hi] = name
    per_sample = np.asarray(sal).sum(axis=0)
    total = per_sample.sum() or 1.0
    return {seg: float(per_sample[labels == seg].sum() / total) for seg in ("P", "QRS", "T", "other")}


@torch.no_grad()
def lead_occlusion(models: list[torch.nn.Module], x: np.ndarray, tab: np.ndarray | None, target_idx: int,
                   leads: list[str], scale: float = 1.0) -> dict[str, float]:
    """|change in prediction| (in target units, via ``scale``) when each lead is set to zero."""
    xt = torch.as_tensor(np.asarray(x, np.float32))[None]
    batch = xt.repeat(len(leads) + 1, 1, 1)
    for i in range(len(leads)):
        batch[i + 1, i] = 0.0
    tt = torch.as_tensor(np.asarray(tab, np.float32))[None].repeat(len(leads) + 1, 1) \
        if tab is not None and len(tab) else None
    preds = np.mean([m(batch, tt)["mean"][:, target_idx].numpy() for m in models], axis=0)
    return {lead: float(abs(preds[i + 1] - preds[0]) * scale) for i, lead in enumerate(leads)}
