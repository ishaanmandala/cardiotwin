"""PTB-XL: 21,799 open 12-lead ECGs used to *pretrain* the CardioTwin ECG encoder.

PTB-XL has no echocardiograms, but its cardiologist labels (left ventricular
hypertrophy, bundle-branch blocks, old infarcts, atrial fibrillation, ...) plus
age and sex teach a network what ECG morphology looks like. Starting from that
encoder, the small echo-labelled datasets (MUSIC, a MIMIC subset) need to learn
much less from scratch - the classic transfer-learning recipe.

Citation: Wagner P, Strodthoff N, Bousseljot R, Samek W, Schaeffter T. PTB-XL, a
large publicly available electrocardiography dataset (v1.0.3). PhysioNet (2022).
doi:10.13026/kfzx-aw45. Scientific Data 2020;7:154. License: CC BY 4.0.
"""

from __future__ import annotations

import ast
import concurrent.futures as cf
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .music import _fetch

log = logging.getLogger(__name__)

PTBXL_BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
PTBXL_METADATA = ["ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"]


def _download_record(base_url: str, rel: str, out_dir: Path) -> bool:
    ok = True
    for ext in (".hea", ".dat"):
        dst = out_dir / f"{rel}{ext}"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = _fetch(f"{base_url}{rel}{ext}")
        except Exception as exc:  # keep going, report at the end
            log.error("PTB-XL download failed for %s%s: %s", rel, ext, exc)
            ok = False
            continue
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(dst)
    return ok


def download_ptbxl(out_dir: str | Path, workers: int = 16, limit: int | None = None,
                   base_url: str = PTBXL_BASE_URL) -> pd.DataFrame:
    """Download PTB-XL metadata and the 100 Hz records (~0.5 GB)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in PTBXL_METADATA:
        if not (out_dir / name).exists():
            (out_dir / name).write_bytes(_fetch(base_url + name))
    db = pd.read_csv(out_dir / "ptbxl_database.csv")
    rels = db["filename_lr"].tolist()[: limit or None]
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, ok in enumerate(pool.map(lambda r: _download_record(base_url, r, out_dir), rels), 1):
            failed += not ok
            if i % 2000 == 0:
                log.info("  %d/%d PTB-XL records", i, len(rels))
    log.info("PTB-XL: %d records downloaded (%d failed)", len(rels) - failed, failed)
    return db


def ptbxl_cohort(raw_dir: str | Path, min_count: int = 200, require_ecg: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Cohort table with one binary ``dx_<code>`` column per SCP statement seen >= ``min_count`` times.

    Also returns the list of label columns. Age >89 is coded as 300 in PTB-XL and capped at 90 here.
    """
    raw_dir = Path(raw_dir)
    db = pd.read_csv(raw_dir / "ptbxl_database.csv")
    codes = db["scp_codes"].map(ast.literal_eval)
    counts = Counter(c for d in codes for c in d)
    keep = sorted(c for c, n in counts.items() if n >= min_count)
    labels = []
    for c in keep:
        col = "dx_" + c.lower().replace("/", "_").replace("_", "_").rstrip("_")
        db[col] = codes.map(lambda d, c=c: int(c in d))
        labels.append(col)
    db["patient_id"] = "ptbxl_" + db["patient_id"].astype(int).astype(str)
    db["ecg_id"] = "ptbxl_" + db["ecg_id"].astype(str)
    db["ecg_path"] = db["filename_lr"]
    db["age"] = db["age"].clip(upper=90)
    db["sex"] = db["sex"].map({0: "M", 1: "F"})
    db["source"] = "PTB-XL"
    if require_ecg:
        exists = db["ecg_path"].map(lambda p: (raw_dir / f"{p}.dat").exists())
        db = db.loc[exists].copy()
    cols = ["patient_id", "ecg_id", "ecg_path", "age", "sex", "strat_fold", "source"] + labels
    return db[cols].reset_index(drop=True), labels


def label_prevalence(df: pd.DataFrame, labels: list[str]) -> dict[str, float]:
    return {lab: float(np.mean(df[lab])) for lab in labels}
