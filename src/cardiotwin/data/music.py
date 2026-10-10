"""The MUSIC study: an open dataset with real ECGs *and* real echo measurements.

MUSIC ("MUerte Subita en Insuficiencia Cardiaca", Sudden Cardiac Death in Chronic
Heart Failure) followed 992 heart-failure patients from 8 Spanish hospitals. For
most patients it provides a 20-minute high-resolution ECG (orthogonal X/Y/Z
leads at 1000 Hz), an echocardiogram (LVEF, LV diameters, wall thickness, left
atrial size, RV size) and ~4 years of follow-up (cardiac deaths, transplants).

It is open access (ODbL) on PhysioNet and mirrored on AWS Open Data, which makes
it a real-data testbed for the CardioTwin pipeline while MIMIC-IV-Echo access
(credentialed) is pending.

Please cite when using it:
  * Martin-Yebra A, Martinez JP, Laguna P. MUSIC (Sudden Cardiac Death in Chronic
    Heart Failure). PhysioNet. https://doi.org/10.13026/z3m7-rf58 (v1.0.1; the AWS
    mirror used here serves v1.0.0).
  * Martin-Yebra A, et al. The MUSIC Database: Sudden Cardiac Death in Heart
    Failure Patients. Computing in Cardiology 2024;51. doi:10.22489/CinC.2024.355
  * Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation
    2000;101(23):e215-e220.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import TARGETS, lv_mass_ase

log = logging.getLogger(__name__)

MUSIC_BASE_URL = "https://physionet-open.s3.amazonaws.com/music-sudden-cardiac-death/1.0.0/"
MUSIC_METADATA_FILES = ["subject-info.csv", "subject-info_codes.csv", "subject-info_definitions.csv", "LICENSE.txt"]
HIGH_RES_DIR = "High-resolution_ECG"

# Raw MUSIC column -> CardioTwin canonical name.
MUSIC_COLUMNS = {
    "Patient ID": "patient_id",
    "Age": "age",
    "Gender (male=1)": "sex_male",
    "LVEF (%)": "lvef",
    "Left ventricle end-diastolic diameter (mm)": "lvedd",
    "Left ventricle end-systolic diameter (mm)": "lvesd",
    "Septal thickness (mm)": "ivsd",
    "Left ventricular posterior wall thickness (mm)": "lvpwd",
    "Left atrial size (mm)": "la_diam",
    "Right ventricle end-diastolic diameter (mm)": "rv_diam",
    "NYHA class": "nyha",
    "HF etiology - Diagnosis": "hf_etiology",
    "Prior Myocardial Infarction (yes=1)": "prior_mi",
    "History of hypertension (yes=1)": "hypertension",
    "Diabetes (yes=1)": "diabetes",
    "Pro-BNP (ng/L)": "nt_probnp",
    "Mitral valve insufficiency ": "mitral_regurg",
    "Hig-resolution ECG available": "has_high_res_ecg",
    "ECG rhythm ": "ecg_rhythm",
    "QRS duration (ms)": "clinical_qrs_ms",
    "Left ventricular hypertrophy (yes=1)": "clinical_lvh",
    "Intraventricular conduction disorder": "conduction_disorder",
    "Prior implantable device": "prior_device",
    "Follow-up period from enrollment (days)": "followup_days",
    "Exit of the study": "exit_code",
    "Cause of death": "death_cause",
    "Body Mass Index (Kg/m2)": "bmi",
    "Betablockers (yes=1)": "beta_blocker",
}

RHYTHM_NAMES = {0: "sinus", 1: "atrial fibrillation", 2: "atrial flutter", 3: "paced"}

MUSIC_TARGETS = ["lvef", "lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "rv_diam", "lv_mass"]


def _fetch(url: str, byte_range: tuple[int, int] | None = None, retries: int = 4, timeout: float = 60) -> bytes:
    """GET a URL (optionally a byte range) with exponential-backoff retries."""
    headers = {"User-Agent": "cardiotwin/0.1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404):
                raise
            if attempt == retries:
                raise
            log.warning("Download failed (%s), retrying in %.0fs: %s", exc, delay, url)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def parse_header(text: str) -> dict:
    """Parse the parts of a WFDB header we need for byte-range excerpts."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    first = lines[0].split()
    n_sig, fs, sig_len = int(first[1]), float(first[2].split("/")[0]), int(first[3])
    fmts = {ln.split()[1].split("+")[0].split("x")[0].split(":")[0] for ln in lines[1 : 1 + n_sig]}
    if fmts != {"16"}:
        raise ValueError(f"Only WFDB format 16 supported for excerpts, got {fmts}")
    return {"record": first[0], "n_sig": n_sig, "fs": fs, "sig_len": sig_len, "lines": lines}


def excerpt_header(header: dict, n_frames: int) -> str:
    first = header["lines"][0].split()
    first[3] = str(n_frames)
    out = [" ".join(first)] + header["lines"][1:]
    return "\n".join(out) + "\n"


def download_patient_excerpt(
    patient_id: str, out_dir: Path, start_s: float = 60.0, duration_s: float = 300.0,
    base_url: str = MUSIC_BASE_URL,
) -> Path | None:
    """Download ``duration_s`` seconds of a patient's high-resolution ECG.

    Uses an HTTP byte-range request so only the excerpt is transferred (~1.8 MB
    for 5 minutes instead of 7 MB for the full 20-minute recording). The excerpt
    is written as a valid WFDB record (same header, shortened length).
    """
    rec = f"{patient_id}_H"
    dst_dir = out_dir / HIGH_RES_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    hea_path, dat_path = dst_dir / f"{rec}.hea", dst_dir / f"{rec}.dat"
    if hea_path.exists() and dat_path.exists():
        return dst_dir / rec
    try:
        header = parse_header(_fetch(f"{base_url}{HIGH_RES_DIR}/{rec}.hea").decode("latin-1"))
    except urllib.error.HTTPError as exc:
        log.warning("No high-resolution ECG header for %s (%s)", patient_id, exc)
        return None
    bytes_per_frame = 2 * header["n_sig"]
    fs, total = header["fs"], header["sig_len"]
    n_frames = min(int(duration_s * fs), total)
    start = int(start_s * fs)
    if start + n_frames > total:
        start = max(0, total - n_frames)
    payload = _fetch(
        f"{base_url}{HIGH_RES_DIR}/{rec}.dat",
        byte_range=(start * bytes_per_frame, (start + n_frames) * bytes_per_frame - 1),
    )
    if len(payload) != n_frames * bytes_per_frame:
        raise OSError(f"{rec}: expected {n_frames * bytes_per_frame} bytes, got {len(payload)}")
    tmp = dat_path.with_suffix(".dat.part")
    tmp.write_bytes(payload)
    tmp.rename(dat_path)
    hea_path.write_text(excerpt_header(header, n_frames))
    return dst_dir / rec


def download_music(
    out_dir: str | Path, start_s: float = 60.0, duration_s: float = 300.0,
    workers: int = 8, limit: int | None = None, base_url: str = MUSIC_BASE_URL,
) -> pd.DataFrame:
    """Download MUSIC metadata plus high-resolution ECG excerpts for every patient."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in MUSIC_METADATA_FILES:
        path = out_dir / name
        if not path.exists():
            path.write_bytes(_fetch(base_url + name))
    info = read_music_subject_info(out_dir / "subject-info.csv")
    ids = info.loc[info["has_high_res_ecg"] == 1, "patient_id"].tolist()
    if limit:
        ids = ids[:limit]
    log.info("Downloading %d high-resolution ECG excerpts to %s", len(ids), out_dir)
    done, failed = 0, []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_patient_excerpt, pid, out_dir, start_s, duration_s, base_url): pid
            for pid in ids
        }
        for fut in cf.as_completed(futures):
            pid = futures[fut]
            try:
                if fut.result() is not None:
                    done += 1
            except Exception as exc:  # keep going; report at the end
                failed.append(pid)
                log.error("Failed to download %s: %s", pid, exc)
            if (done + len(failed)) % 50 == 0:
                log.info("  %d/%d done", done + len(failed), len(ids))
    log.info("Downloaded %d excerpts (%d failed)", done, len(failed))
    return info


def read_music_subject_info(path: str | Path) -> pd.DataFrame:
    """Load ``subject-info.csv`` (semicolon separated, decimal commas) with canonical names."""
    raw = pd.read_csv(path, sep=";", decimal=",", encoding="latin-1", low_memory=False)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    missing = [c for c in MUSIC_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"MUSIC subject-info.csv is missing expected columns: {missing}")
    df = raw[list(MUSIC_COLUMNS)].rename(columns=MUSIC_COLUMNS).copy()
    for col in df.columns:
        if col != "patient_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def music_cohort(raw_dir: str | Path, require_ecg: bool = True) -> pd.DataFrame:
    """Build the CardioTwin cohort table for MUSIC (one row per patient/ECG).

    Adds cleaned echo targets, derived LV mass, rhythm labels and follow-up
    outcomes:
      * ``event_hf``: pump-failure death or heart transplant during follow-up
      * ``event_cardiac_death``: sudden cardiac death or pump-failure death
      * ``time_days``: follow-up time
    """
    raw_dir = Path(raw_dir)
    df = read_music_subject_info(raw_dir / "subject-info.csv")
    df["sex"] = np.where(df["sex_male"] == 1, "M", np.where(df["sex_male"] == 0, "F", None))
    for name in ["lvef", "lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "rv_diam"]:
        lo, hi = TARGETS[name].plausible_range
        df.loc[(df[name] < lo) | (df[name] > hi), name] = np.nan
    # A systolic diameter larger than the diastolic one is a data-entry error.
    df.loc[df["lvesd"] > df["lvedd"], "lvesd"] = np.nan
    df["lv_mass"] = lv_mass_ase(df["lvedd"], df["ivsd"], df["lvpwd"])
    df["rhythm"] = df["ecg_rhythm"].map(RHYTHM_NAMES)
    cause = df["death_cause"].fillna(0)
    exit_code = df["exit_code"].fillna(0)
    df["event_hf"] = (cause.isin([6, 7]) | (exit_code == 2)).astype(int)
    df["event_cardiac_death"] = cause.isin([3, 6, 7]).astype(int)
    df["event_scd"] = (cause == 3).astype(int)
    df["time_days"] = df["followup_days"]
    df["ecg_id"] = df["patient_id"] + "_H"
    df["ecg_path"] = HIGH_RES_DIR + "/" + df["ecg_id"]
    df["source"] = "MUSIC"
    if require_ecg:
        exists = df["ecg_path"].map(lambda p: (raw_dir / f"{p}.hea").exists())
        df = df.loc[(df["has_high_res_ecg"] == 1) & exists].copy()
    return df.reset_index(drop=True)
