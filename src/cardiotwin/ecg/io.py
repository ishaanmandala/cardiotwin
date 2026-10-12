"""Reading ECGs from disk or uploads into one common in-memory format.

Every reader returns an :class:`ECG`: a ``(n_leads, n_samples)`` float32 array in
millivolts, its sampling rate, and the lead names. Downstream code never has to
care whether the ECG came from MIMIC-IV-ECG (WFDB, 12 leads, 500 Hz), the MUSIC
study (WFDB, X/Y/Z leads, 1000 Hz) or a CSV someone exported from a cart.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import STANDARD_12_LEADS, VCG_LEADS, canonical_lead_name

# Kors regression matrix (Kors et al., Eur Heart J 1990;11:1083-92): reconstructs
# Frank X/Y/Z from the 8 independent leads of a standard 12-lead ECG.
KORS_INPUT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
KORS_MATRIX = np.array(
    [
        [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],
        [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],
        [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],
    ],
    dtype=np.float32,
)

# Dower inverse transform: Frank X/Y/Z -> 8 independent leads (I, II, V1..V6).
DOWER_MATRIX = np.array(
    [
        [0.632, -0.235, 0.059],
        [0.235, 1.066, -0.132],
        [-0.515, 0.157, -0.917],
        [0.044, 0.164, -1.387],
        [0.882, 0.098, -1.277],
        [1.213, 0.127, -0.601],
        [1.125, 0.127, -0.086],
        [0.831, 0.076, 0.230],
    ],
    dtype=np.float32,
)


class ECGFormatError(ValueError):
    """Raised when an uploaded/loaded file cannot be interpreted as an ECG."""


@dataclass
class ECG:
    """A multi-lead ECG recording.

    Attributes:
        signal: float32 array of shape ``(n_leads, n_samples)`` in millivolts.
        fs: sampling frequency in Hz.
        leads: lead names in the same order as ``signal`` rows.
        meta: free-form metadata (record name, source, derived leads, ...).
    """

    signal: np.ndarray
    fs: float
    leads: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=np.float32)
        if self.signal.ndim == 1:
            self.signal = self.signal[None, :]
        if self.signal.ndim != 2:
            raise ECGFormatError(f"ECG signal must be 2-D (leads, samples); got shape {self.signal.shape}")
        self.leads = [canonical_lead_name(lead) for lead in self.leads]
        if len(self.leads) != self.signal.shape[0]:
            raise ECGFormatError(
                f"{len(self.leads)} lead names given for a signal with {self.signal.shape[0]} leads"
            )
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ECGFormatError(f"Invalid sampling frequency: {self.fs}")

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.fs)

    def lead(self, name: str) -> np.ndarray:
        return self.signal[self.leads.index(canonical_lead_name(name))]

    def has_leads(self, names: list[str]) -> bool:
        return all(canonical_lead_name(n) in self.leads for n in names)

    def select_leads(self, required: list[str], vcg_z_anterior: bool = False) -> ECG:
        """Return a new ECG with exactly ``required`` leads in that order.

        Missing limb leads are derived from I and II (Einthoven/Goldberger), and
        Frank X/Y/Z leads can be reconstructed from a 12-lead ECG with the Kors
        matrix (and vice versa with the Dower matrix). Derived leads are recorded
        in ``meta['derived_leads']``.

        ``vcg_z_anterior``: the Kors/Dower matrices use Z pointing *backwards*.
        Some datasets (e.g. MUSIC) record Z pointing *forwards*; set this flag to
        flip Z when converting.
        """
        z_sign = -1.0 if vcg_z_anterior else 1.0
        required = [canonical_lead_name(r) for r in required]
        sig = {name: self.signal[i] for i, name in enumerate(self.leads)}
        derived: list[str] = []

        if "I" in sig and "II" in sig:
            limb = {
                "III": lambda s: s["II"] - s["I"],
                "aVR": lambda s: -(s["I"] + s["II"]) / 2.0,
                "aVL": lambda s: s["I"] - s["II"] / 2.0,
                "aVF": lambda s: s["II"] - s["I"] / 2.0,
            }
            for name, fn in limb.items():
                if name in required and name not in sig:
                    sig[name] = fn(sig)
                    derived.append(name)

        need_vcg = [r for r in required if r in VCG_LEADS and r not in sig]
        if need_vcg and all(lead in sig for lead in KORS_INPUT_LEADS):
            stack = np.stack([sig[lead] for lead in KORS_INPUT_LEADS])
            xyz = KORS_MATRIX @ stack
            xyz[2] *= z_sign
            for i, name in enumerate(VCG_LEADS):
                if name in need_vcg:
                    sig[name] = xyz[i]
                    derived.append(name)

        need_12 = [r for r in required if r in STANDARD_12_LEADS and r not in sig]
        if need_12 and all(lead in sig for lead in VCG_LEADS) and not all(
            lead in sig for lead in KORS_INPUT_LEADS
        ):
            # Synthesize I, II, V1-V6 with the Dower matrix, then derive the remaining
            # limb leads on the recursive call.
            xyz = np.stack([sig[lead] for lead in VCG_LEADS])
            xyz[2] = xyz[2] * z_sign
            eight = DOWER_MATRIX @ xyz
            for i, name in enumerate(KORS_INPUT_LEADS):
                if name not in sig:
                    sig[name] = eight[i]
                    derived.append(name)
            meta = dict(self.meta)
            meta["derived_leads"] = sorted(set(meta.get("derived_leads", [])) | set(derived))
            expanded = ECG(np.stack(list(sig.values())), self.fs, list(sig), meta)
            return expanded.select_leads(required, vcg_z_anterior)

        missing = [r for r in required if r not in sig]
        if missing:
            raise ECGFormatError(
                f"ECG is missing required leads {missing}. Available leads: {self.leads}"
            )
        meta = dict(self.meta)
        if derived:
            meta["derived_leads"] = sorted(set(meta.get("derived_leads", [])) | set(derived))
        return ECG(np.stack([sig[r] for r in required]), self.fs, list(required), meta)

    def crop(self, start_s: float, duration_s: float) -> ECG:
        a = int(round(start_s * self.fs))
        b = a + int(round(duration_s * self.fs))
        return ECG(self.signal[:, a:b], self.fs, list(self.leads), dict(self.meta))


# --------------------------------------------------------------------------------------
# WFDB
# --------------------------------------------------------------------------------------


def read_wfdb(record_path: str | os.PathLike, sampfrom: int = 0, sampto: int | None = None) -> ECG:
    """Read a WFDB record (``.hea`` + ``.dat``) given its path without extension."""
    import wfdb

    record_path = str(record_path)
    if record_path.endswith(".hea") or record_path.endswith(".dat"):
        record_path = record_path[:-4]
    try:
        rec = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=sampto)
    except FileNotFoundError as exc:
        raise ECGFormatError(f"WFDB record not found: {record_path}") from exc
    if rec.p_signal is None:
        raise ECGFormatError(f"WFDB record has no physical signal: {record_path}")
    signal = rec.p_signal.T.astype(np.float32)
    units = [u.lower() if u else "mv" for u in (rec.units or ["mV"] * rec.n_sig)]
    for i, unit in enumerate(units):
        if unit in ("uv", "µv"):
            signal[i] /= 1000.0
        elif unit == "v":
            signal[i] *= 1000.0
    return ECG(
        signal=signal,
        fs=float(rec.fs),
        leads=list(rec.sig_name),
        meta={"record": Path(record_path).name, "source_format": "wfdb"},
    )


# --------------------------------------------------------------------------------------
# CSV / NumPy / JSON
# --------------------------------------------------------------------------------------

_TIME_COLUMNS = {"time", "t", "time_s", "seconds", "sec", "timestamp", "time (s)", "ms", "time_ms"}


def read_csv_ecg(source, fs: float | None = None, units: str = "mV") -> ECG:
    """Read an ECG from CSV: one column per lead, optional time column.

    If ``fs`` is not given, it is inferred from a time column (seconds, or
    milliseconds if the column is called ``ms``/``time_ms``).
    """
    df = pd.read_csv(source)
    if df.shape[1] == 0 or len(df) < 10:
        raise ECGFormatError("CSV has no data columns or too few rows to be an ECG")
    time_col = next((c for c in df.columns if str(c).strip().lower() in _TIME_COLUMNS), None)
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if fs is None:
            dt = np.nanmedian(np.diff(t))
            if not np.isfinite(dt) or dt <= 0:
                raise ECGFormatError("Could not infer sampling rate from the time column")
            if str(time_col).strip().lower() in ("ms", "time_ms"):
                dt /= 1000.0
            fs = 1.0 / dt
        df = df.drop(columns=[time_col])
    if fs is None:
        raise ECGFormatError("Sampling rate unknown: add a time column or pass fs")
    lead_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    data = df[lead_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).T
    if units.lower() in ("uv", "µv"):
        data /= 1000.0
    return ECG(signal=data, fs=round(float(fs), 3), leads=[str(c) for c in lead_cols],
               meta={"source_format": "csv"})


def read_array_ecg(array: np.ndarray, fs: float, leads: list[str] | None = None) -> ECG:
    """Wrap a raw array. Accepts (leads, samples) or (samples, leads)."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2:
        raise ECGFormatError(f"Expected a 2-D array, got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    if leads is None:
        leads = _default_lead_names(arr.shape[0])
    return ECG(arr, fs, list(leads), {"source_format": "array"})


def _default_lead_names(n: int) -> list[str]:
    if n == 12:
        return list(STANDARD_12_LEADS)
    if n == 3:
        return list(VCG_LEADS)
    return [f"lead{i + 1}" for i in range(n)]


def read_npz_ecg(source) -> ECG:
    """Read the CardioTwin ``.npz`` format (keys: signal, fs, leads[, meta_json])."""
    with np.load(source, allow_pickle=False) as z:
        if "signal" not in z or "fs" not in z:
            raise ECGFormatError("NPZ ECG files need 'signal' and 'fs' arrays")
        leads = [str(x) for x in z["leads"]] if "leads" in z else None
        ecg = read_array_ecg(z["signal"], float(z["fs"]), leads)
        if "meta_json" in z:
            ecg.meta.update(json.loads(str(z["meta_json"])))
    ecg.meta["source_format"] = "npz"
    return ecg


def save_npz_ecg(ecg: ECG, path: str | os.PathLike, meta: dict | None = None) -> None:
    payload = dict(ecg.meta)
    payload.update(meta or {})
    np.savez_compressed(
        path,
        signal=ecg.signal.astype(np.float32),
        fs=np.float32(ecg.fs),
        leads=np.array(ecg.leads),
        meta_json=np.array(json.dumps(payload, default=str)),
    )


def read_json_ecg(data: dict) -> ECG:
    """JSON ECG: {"fs": 500, "leads": ["I", ...], "signal": [[...], ...]}."""
    if "signal" not in data or "fs" not in data:
        raise ECGFormatError("JSON ECG needs 'signal' and 'fs' keys")
    return read_array_ecg(np.asarray(data["signal"], dtype=np.float32), float(data["fs"]), data.get("leads"))


# --------------------------------------------------------------------------------------
# Uploads (used by the web app): a mapping of filename -> bytes.
# --------------------------------------------------------------------------------------


def _check_wfdb_header(header: bytes, uploaded: set[str]) -> None:
    """Refuse headers that point at files other than the ones uploaded.

    A WFDB header names its signal file(s). On a public server, a crafted header
    could name a path such as ``../../etc/passwd``; only plain file names that
    were part of the same upload are allowed.
    """
    try:
        text = header.decode("latin-1")
    except Exception as exc:  # pragma: no cover - latin-1 decodes any bytes
        raise ECGFormatError("Unreadable .hea file") from exc
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ECGFormatError("Empty .hea file")
    first = lines[0].split()
    if len(first) < 2 or "/" in first[0] or "\\" in first[0]:
        raise ECGFormatError("Invalid WFDB header (multi-segment records are not supported)")
    try:
        n_sig = int(first[1])
    except ValueError as exc:
        raise ECGFormatError("Invalid WFDB header: number of signals is not a number") from exc
    for line in lines[1 : 1 + n_sig]:
        fname = line.split()[0]
        if fname != Path(fname).name or fname in (".", "..") or fname.startswith("~") or "\\" in fname:
            raise ECGFormatError(f"Invalid signal file name in header: {fname!r}")
        if fname not in uploaded:
            raise ECGFormatError(f"The header refers to '{fname}', which was not uploaded. "
                                 "Upload the .hea file together with its .dat file.")


def load_ecg_from_files(files: dict[str, bytes], fs: float | None = None) -> ECG:
    """Interpret one uploaded ECG, which may span several files (``.hea`` + ``.dat``)."""
    if not files:
        raise ECGFormatError("No files uploaded")
    by_ext: dict[str, list[str]] = {}
    for name in files:
        by_ext.setdefault(Path(name).suffix.lower(), []).append(name)

    if ".hea" in by_ext:
        hea_name = by_ext[".hea"][0]
        _check_wfdb_header(files[hea_name], {Path(n).name for n in files})
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in files.items():
                safe = Path(name).name  # never trust client-provided directories
                (Path(tmp) / safe).write_bytes(content)
            record = Path(tmp) / Path(hea_name).stem
            try:
                ecg = read_wfdb(record)
            except ECGFormatError:
                raise
            except Exception as exc:  # wfdb raises many exception types for bad files
                raise ECGFormatError(
                    f"Could not read WFDB record '{Path(hea_name).stem}': {exc}. "
                    "Upload the .hea file together with its .dat file."
                ) from exc
        return ecg
    if ".dat" in by_ext:
        raise ECGFormatError("A WFDB .dat file needs its .hea header file too")

    name, content = next(iter(files.items()))
    ext = Path(name).suffix.lower()
    try:
        if ext == ".csv" or ext == ".txt":
            return read_csv_ecg(io.BytesIO(content), fs=fs)
        if ext == ".npz":
            return read_npz_ecg(io.BytesIO(content))
        if ext == ".npy":
            if fs is None:
                raise ECGFormatError("A .npy upload needs the sampling rate (fs)")
            return read_array_ecg(np.load(io.BytesIO(content), allow_pickle=False), fs)
        if ext == ".json":
            return read_json_ecg(json.loads(content.decode("utf-8")))
    except ECGFormatError:
        raise
    except Exception as exc:
        raise ECGFormatError(f"Could not read '{name}': {exc}") from exc
    raise ECGFormatError(
        f"Unsupported file type '{ext}'. Use WFDB (.hea + .dat), .csv, .npz, .npy or .json"
    )
