"""Signal cleaning: the same steps are applied in training and at inference time.

Pipeline (``preprocess_ecg``):
  1. Replace NaNs (some MIMIC-IV-ECG records contain gaps) by linear interpolation.
  2. Resample to the model sampling rate (250 Hz by default).
  3. Zero-phase Butterworth band-pass (0.5-45 Hz by default) to remove baseline
     wander and high-frequency noise, plus 50/60 Hz notch filters for mains hum.
  4. Remove each lead's median and clip to +/- ``clip_mv``.

Amplitudes stay in millivolts on purpose: QRS voltage carries information about
wall thickness (high voltage) and ventricular function (low voltage), so we do
not z-score each recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from .io import ECG


@dataclass(frozen=True)
class PreprocessConfig:
    fs: float = 250.0
    highpass_hz: float = 0.5
    lowpass_hz: float = 45.0
    notch_hz: tuple[float, ...] = (50.0, 60.0)
    clip_mv: float = 6.0
    window_s: float = 10.0

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.fs))

    def to_dict(self) -> dict:
        return {
            "fs": self.fs, "highpass_hz": self.highpass_hz, "lowpass_hz": self.lowpass_hz,
            "notch_hz": list(self.notch_hz), "clip_mv": self.clip_mv, "window_s": self.window_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessConfig:
        d = dict(d)
        if "notch_hz" in d:
            d["notch_hz"] = tuple(d["notch_hz"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN/inf samples per lead; all-NaN leads become zeros."""
    x = np.array(x, dtype=np.float32, copy=True)
    for i in range(x.shape[0]):
        bad = ~np.isfinite(x[i])
        if not bad.any():
            continue
        if bad.all():
            x[i] = 0.0
            continue
        idx = np.arange(x.shape[1])
        x[i, bad] = np.interp(idx[bad], idx[~bad], x[i, ~bad])
    return x


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling along the last axis (anti-aliased)."""
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    y = sps.resample_poly(x, frac.numerator, frac.denominator, axis=-1)
    return y.astype(np.float32)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    if low <= 0 and high >= 0.95 * nyq:
        return x
    if low <= 0:
        sos = sps.butter(order, high / nyq, btype="lowpass", output="sos")
    else:
        sos = sps.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    padlen = min(x.shape[-1] - 1, 3 * (2 * len(sos) + 1) * 10)
    return sps.sosfiltfilt(sos, x, axis=-1, padlen=padlen).astype(np.float32)


def notch(x: np.ndarray, fs: float, freq: float, q: float = 30.0) -> np.ndarray:
    if freq >= fs / 2.0 * 0.98:
        return x
    b, a = sps.iirnotch(freq, q, fs)
    return sps.filtfilt(b, a, x, axis=-1).astype(np.float32)


def preprocess_signal(x: np.ndarray, fs_in: float, cfg: PreprocessConfig) -> np.ndarray:
    x = fill_nans(x)
    x = resample(x, fs_in, cfg.fs)
    if x.shape[-1] > 32:
        x = bandpass(x, cfg.fs, cfg.highpass_hz, cfg.lowpass_hz)
        for f in cfg.notch_hz:
            if f < cfg.lowpass_hz * 1.6:  # only useful when the low-pass leaves mains energy
                x = notch(x, cfg.fs, f)
    x = x - np.median(x, axis=-1, keepdims=True)
    return np.clip(x, -cfg.clip_mv, cfg.clip_mv).astype(np.float32)


def preprocess_ecg(ecg: ECG, cfg: PreprocessConfig, leads: list[str] | None = None,
                   vcg_z_anterior: bool = False) -> ECG:
    """Select/derive ``leads`` (if given) and run the cleaning pipeline."""
    if leads is not None:
        ecg = ecg.select_leads(leads, vcg_z_anterior=vcg_z_anterior)
    out = preprocess_signal(ecg.signal, ecg.fs, cfg)
    meta = dict(ecg.meta)
    meta["preprocessed"] = cfg.to_dict()
    return ECG(out, cfg.fs, list(ecg.leads), meta)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Center-crop or zero-pad (symmetrically) the last axis to exactly ``n`` samples."""
    m = x.shape[-1]
    if m == n:
        return x
    if m > n:
        start = (m - n) // 2
        return x[..., start : start + n]
    pad = n - m
    width = [(0, 0)] * (x.ndim - 1) + [(pad // 2, pad - pad // 2)]
    return np.pad(x, width)


def sliding_windows(x: np.ndarray, n: int, stride: int) -> np.ndarray:
    """Split ``(leads, samples)`` into ``(k, leads, n)`` windows; at least one window."""
    if x.shape[-1] <= n:
        return fit_length(x, n)[None]
    starts = list(range(0, x.shape[-1] - n + 1, stride))
    return np.stack([x[:, s : s + n] for s in starts])


# --------------------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------------------


def quality_report(ecg: ECG, min_duration_s: float = 8.0) -> dict:
    """Cheap signal-quality checks. Returns {'ok': bool, 'warnings': [...], 'per_lead': {...}}."""
    warnings: list[str] = []
    per_lead: dict[str, dict] = {}
    x = ecg.signal
    nan_frac = float(np.mean(~np.isfinite(x))) if x.size else 1.0
    if nan_frac > 0.05:
        warnings.append(f"{nan_frac:.0%} of samples are missing (NaN)")
    if ecg.duration_s < min_duration_s:
        warnings.append(f"Recording is only {ecg.duration_s:.1f} s long; CardioTwin expects 10 s")
    xf = fill_nans(x)
    for i, name in enumerate(ecg.leads):
        lead = xf[i]
        std = float(np.std(lead))
        ptp = float(np.ptp(lead)) if lead.size else 0.0
        flags = []
        if std < 0.005:
            flags.append("flat")
        if ptp > 15.0:
            flags.append("very large amplitude (check units: expected millivolts)")
        sat = float(np.mean(np.abs(lead) >= 0.999 * np.max(np.abs(lead)))) if ptp > 0 else 0.0
        if sat > 0.02 and ptp > 0:
            flags.append("possible clipping")
        per_lead[name] = {"std_mv": round(std, 4), "ptp_mv": round(ptp, 3), "flags": flags}
        for f in flags:
            warnings.append(f"Lead {name}: {f}")
    n_flat = sum("flat" in v["flags"] for v in per_lead.values())
    ok = nan_frac <= 0.2 and n_flat < max(1, len(ecg.leads) // 2) and ecg.duration_s >= min_duration_s / 2
    if not ok:
        warnings.insert(0, "Signal quality is too poor for a reliable estimate")
    return {"ok": bool(ok), "warnings": warnings, "per_lead": per_lead}
