"""LEVEL 6: tables with pandas - find EF and match ECGs to echoes.

Run from the repository root (after `cardiotwin download-music` or tutorial 01):
    python tutorials/02_echo_table.py
"""

from pathlib import Path

import pandas as pd

from cardiotwin.data.mimic import PairingConfig, pair_ecg_echo
from cardiotwin.data.music import _fetch, read_music_subject_info

# ---------------------------------------------------------------------------
# Part 1: a real echo table (MUSIC). One row per patient.
# ---------------------------------------------------------------------------
path = Path("data/raw/music/subject-info.csv")
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_fetch("https://physionet-open.s3.amazonaws.com/music-sudden-cardiac-death/1.0.0/subject-info.csv"))

info = read_music_subject_info(path)
print(info[["patient_id", "age", "sex_male", "lvef", "lvedd", "la_diam"]].head())
print()
print("How many patients have an EF?", info["lvef"].notna().sum())
print("Average EF:", round(info["lvef"].mean(), 1), "%")
print("Patients with EF <= 40%:", (info["lvef"] <= 40).sum())

# ---------------------------------------------------------------------------
# Part 2: the MIMIC problem. ECGs and echoes are separate tables with times.
# We want pairs: same patient, ECG within 24 hours of the echo.
# ---------------------------------------------------------------------------
ecgs = pd.DataFrame({
    "subject_id": [1, 1, 2, 3],
    "study_id": ["ECG001", "ECG002", "ECG003", "ECG004"],
    "ecg_time": pd.to_datetime(["2150-01-02 09:00", "2150-06-01 10:00", "2150-01-05 08:00", "2150-01-12 23:00"]),
    "path": ["...", "...", "...", "..."],
})
echoes = pd.DataFrame({
    "subject_id": [1, 2, 3],
    "echo_id": ["ECHO-A", "ECHO-B", "ECHO-C"],
    "echo_time": pd.to_datetime(["2150-01-02 14:00", "2150-01-05 11:00", "2150-01-12 09:00"]),
    "lvef": [61, 38, 29],
})
pairs = pair_ecg_echo(ecgs, echoes, PairingConfig(max_hours=24))
print()
print("Training table (one row = one ECG with the right answer):")
print(pairs[["subject_id", "study_id", "echo_id", "hours_apart", "lvef"]])
# ECG002 is 5 months away from any echo, so it is (correctly) left out.
