"""Classic, hand-made ECG measurements (the kind a cardiologist reads off paper).

These feed a "classic" baseline (does a deep network beat simple ECG numbers?)
and are shown in the dashboard. They are intentionally simple and robust rather
than clinical-grade.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

FEATURE_NAMES = [
    "heart_rate", "rr_cv", "qrs_ms", "qrs_ptp_mean", "qrs_ptp_max", "t_amp_mean",
    "t_abs_mean", "p_amp_mean", "rt_peak_ms", "rtc_ms",
]


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """R-peak indices from a multi-lead signal (Pan-Tompkins style on all leads)."""
    if x.ndim == 1:
        x = x[None]
    nyq = fs / 2
    sos = sps.butter(2, [5 / nyq, min(20, 0.9 * nyq) / nyq], btype="bandpass", output="sos")
    filt = sps.sosfiltfilt(sos, x, axis=-1)
    energy = np.sum(np.diff(filt, axis=-1, prepend=filt[:, :1]) ** 2, axis=0)
    win = max(1, int(0.12 * fs))
    integ = np.convolve(energy, np.ones(win) / win, mode="same")
    height = 0.25 * np.percentile(integ, 99)
    peaks, _ = sps.find_peaks(integ, height=height, distance=int(0.3 * fs))
    # Refine each peak to the max absolute deflection of the band-passed sum nearby.
    combo = np.sum(np.abs(filt), axis=0)
    half = int(0.06 * fs)
    refined = []
    for p in peaks:
        a, b = max(0, p - half), min(len(combo), p + half)
        refined.append(a + int(np.argmax(combo[a:b])))
    return np.unique(np.asarray(refined, dtype=int))


def median_beat(x: np.ndarray, fs: float, r_peaks: np.ndarray, pre_s: float = 0.3, post_s: float = 0.55):
    pre, post = int(pre_s * fs), int(post_s * fs)
    beats = [x[:, r - pre : r + post] for r in r_peaks if r - pre >= 0 and r + post <= x.shape[1]]
    if not beats:
        return None, pre
    return np.median(np.stack(beats), axis=0), pre


def ecg_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Compute :data:`FEATURE_NAMES` from a preprocessed ``(leads, samples)`` ECG in mV."""
    out = dict.fromkeys(FEATURE_NAMES, float("nan"))
    x = np.asarray(x, dtype=np.float64)
    r = detect_r_peaks(x, fs)
    if len(r) < 3:
        return out
    rr = np.diff(r) / fs
    out["heart_rate"] = float(60.0 / np.median(rr))
    out["rr_cv"] = float(np.std(rr) / np.mean(rr))
    beat, pre = median_beat(x, fs, r)
    if beat is None:
        return out
    beat = beat - np.median(beat[:, : int(0.05 * fs)], axis=1, keepdims=True)
    vel = np.sqrt(np.sum(np.diff(beat, axis=1) ** 2, axis=0))
    search = slice(max(0, pre - int(0.15 * fs)), min(len(vel), pre + int(0.15 * fs)))
    seg = vel[search]
    thr = 0.15 * seg.max()
    above = np.where(seg > thr)[0]
    if len(above):
        on = search.start + above[0]
        off = search.start + above[-1]
        # Keep only the contiguous block around the R peak.
        qrs_ms = (off - on) / fs * 1000.0
        out["qrs_ms"] = float(np.clip(qrs_ms, 40, 260))
    else:
        on, off = pre - int(0.05 * fs), pre + int(0.05 * fs)
    ptp = np.ptp(beat[:, on : off + 1], axis=1)
    out["qrs_ptp_mean"] = float(np.mean(ptp))
    out["qrs_ptp_max"] = float(np.max(ptp))
    t_lo, t_hi = off + int(0.06 * fs), min(beat.shape[1], off + int(0.4 * fs))
    if t_hi > t_lo:
        tw = beat[:, t_lo:t_hi]
        idx = np.argmax(np.abs(tw), axis=1)
        t_amp = tw[np.arange(len(tw)), idx]
        out["t_amp_mean"] = float(np.mean(t_amp))
        out["t_abs_mean"] = float(np.mean(np.abs(t_amp)))
        mag = np.sqrt(np.sum(tw**2, axis=0))
        rt = (t_lo + int(np.argmax(mag)) - pre) / fs * 1000.0
        out["rt_peak_ms"] = float(rt)
        out["rtc_ms"] = float(rt / np.sqrt(np.median(rr)))
    p_lo, p_hi = max(0, pre - int(0.28 * fs)), max(1, on - int(0.03 * fs))
    if p_hi > p_lo:
        pw = beat[:, p_lo:p_hi]
        out["p_amp_mean"] = float(np.mean(np.max(np.abs(pw), axis=1)))
    return out


def features_for_rows(signals: np.ndarray, store_rows, fs: float, window: int | None = None) -> np.ndarray:
    """Feature matrix (len(store_rows), n_features), using the central window of each ECG."""
    feats = []
    for r in store_rows:
        x = np.asarray(signals[int(r)], dtype=np.float32)
        if window is not None and x.shape[-1] > window:
            s = (x.shape[-1] - window) // 2
            x = x[:, s : s + window]
        f = ecg_features(x, fs)
        feats.append([f[k] for k in FEATURE_NAMES])
    return np.asarray(feats, dtype=float)
