"""Preprocess every ECG in a cohort once and keep them in one memory-mapped array.

Reading thousands of WFDB files every epoch is slow, so ``build_signal_store``
cleans each ECG (see :mod:`cardiotwin.ecg.preprocess`) and writes:

    out_dir/signals.npy      float16 array (n_ecgs, n_leads, n_samples)
    out_dir/cohort.csv       the cohort table + ``store_row`` and ``store_ok``
    out_dir/store_meta.json  sampling rate, leads, preprocessing settings

Long recordings (e.g. 5-minute MUSIC excerpts) are kept long so training can
draw a different random 10-second window each epoch.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ecg.io import read_wfdb
from ..ecg.preprocess import PreprocessConfig, fit_length, preprocess_ecg

log = logging.getLogger(__name__)


def _load_one(args) -> tuple[int, np.ndarray | None, str | None]:
    i, path, leads, cfg_dict, n_samples, trim_s, z_anterior = args
    cfg = PreprocessConfig.from_dict(cfg_dict)
    try:
        ecg = read_wfdb(path)
        ecg = preprocess_ecg(ecg, cfg, leads, vcg_z_anterior=z_anterior)
        x = ecg.signal
        trim = int(trim_s * cfg.fs)
        if trim and x.shape[1] > n_samples + 2 * trim:
            x = x[:, trim:-trim]  # drop filter edge transients when we have room
        x = fit_length(x, n_samples)
        return i, x.astype(np.float16), None
    except Exception as exc:  # report and continue: one bad file should not stop the build
        return i, None, f"{type(exc).__name__}: {exc}"


def build_signal_store(
    cohort: pd.DataFrame, ecg_root: str | Path, out_dir: str | Path, leads: list[str],
    cfg: PreprocessConfig | None = None, store_seconds: float = 10.0, workers: int = 4,
    edge_trim_s: float = 1.0, vcg_z_anterior: bool = False,
) -> SignalStore:
    cfg = cfg or PreprocessConfig()
    ecg_root, out_dir = Path(ecg_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = cohort.reset_index(drop=True).copy()
    n_samples = int(round(store_seconds * cfg.fs))
    arr = np.lib.format.open_memmap(out_dir / "signals.npy", mode="w+", dtype=np.float16,
                                    shape=(len(cohort), len(leads), n_samples))
    ok = np.zeros(len(cohort), dtype=bool)
    errors: dict[int, str] = {}
    jobs = [(i, str(ecg_root / p), leads, cfg.to_dict(), n_samples, edge_trim_s, vcg_z_anterior)
            for i, p in enumerate(cohort["ecg_path"])]
    executor = cf.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(_load_one, jobs, chunksize=16) if executor else map(_load_one, jobs)
        for n_done, (i, x, err) in enumerate(results, 1):
            if x is not None and np.isfinite(x).all():
                arr[i] = x
                ok[i] = True
            else:
                errors[i] = err or "non-finite values"
            if n_done % 1000 == 0:
                log.info("  preprocessed %d/%d ECGs", n_done, len(jobs))
    finally:
        if executor:
            executor.shutdown()
    arr.flush()
    del arr
    cohort["store_row"] = np.arange(len(cohort))
    cohort["store_ok"] = ok
    cohort["store_error"] = pd.Series(errors, dtype=object).reindex(range(len(cohort)))
    cohort.to_csv(out_dir / "cohort.csv", index=False)
    meta = {"fs": cfg.fs, "leads": list(leads), "n_samples": n_samples, "store_seconds": store_seconds,
            "vcg_z_anterior": vcg_z_anterior,
            "preprocess": cfg.to_dict(), "n_ok": int(ok.sum()), "n_failed": int((~ok).sum())}
    (out_dir / "store_meta.json").write_text(json.dumps(meta, indent=2))
    if errors:
        log.warning("%d ECGs failed to load; first error: %s", len(errors), next(iter(errors.values())))
    return SignalStore.open(out_dir)


@dataclass
class SignalStore:
    signals: np.ndarray  # memmap (n, leads, samples), float16
    cohort: pd.DataFrame
    meta: dict
    path: Path

    @classmethod
    def open(cls, path: str | Path) -> SignalStore:
        path = Path(path)
        meta = json.loads((path / "store_meta.json").read_text())
        signals = np.load(path / "signals.npy", mmap_mode="r")
        cohort = pd.read_csv(path / "cohort.csv", low_memory=False)
        cohort["patient_id"] = cohort["patient_id"].astype(str)
        cohort["ecg_id"] = cohort["ecg_id"].astype(str)
        return cls(signals=signals, cohort=cohort, meta=meta, path=path)

    @property
    def fs(self) -> float:
        return float(self.meta["fs"])

    @property
    def leads(self) -> list[str]:
        return list(self.meta["leads"])

    def usable(self) -> pd.DataFrame:
        return self.cohort.loc[self.cohort["store_ok"].astype(bool)].reset_index(drop=True)
