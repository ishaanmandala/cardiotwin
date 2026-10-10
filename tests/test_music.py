"""MUSIC loader: header parsing, byte-range excerpts and the cohort table (no network needed)."""

import numpy as np
import pandas as pd
import pytest
import wfdb

from cardiotwin.data import music
from cardiotwin.data.music import MUSIC_COLUMNS, excerpt_header, music_cohort, parse_header
from cardiotwin.ecg.io import read_wfdb

HEADER = "P0001_H 3 1000 1196000 \nP0001_H.dat 16+0 400 16 0  X \nP0001_H.dat 16+0 400 16 0  Y \nP0001_H.dat 16+0 400 16 0  Z \n"


def test_parse_and_rewrite_header():
    h = parse_header(HEADER)
    assert (h["n_sig"], h["fs"], h["sig_len"]) == (3, 1000.0, 1196000)
    new = excerpt_header(h, 5000)
    assert new.splitlines()[0].split()[3] == "5000"
    assert new.splitlines()[1:] == h["lines"][1:]


def test_excerpt_download_uses_correct_byte_range(tmp_path, monkeypatch):
    # A fake "remote" record: 3 leads, 1000 Hz, 20 s, ramp signal so offsets are checkable.
    remote = tmp_path / "remote"
    remote.mkdir()
    n = 20000
    ramp = (np.arange(n) % 30000).astype(np.int16)
    raw = np.stack([ramp, -ramp, ramp // 2], axis=1)
    (remote / "P0001_H.dat").write_bytes(raw.astype("<i2").tobytes())
    (remote / "P0001_H.hea").write_text(HEADER.replace("1196000", str(n)))
    calls = []

    def fake_fetch(url, byte_range=None, **_):
        calls.append(byte_range)
        data = (remote / url.rsplit("/", 1)[1]).read_bytes()
        return data if byte_range is None else data[byte_range[0] : byte_range[1] + 1]

    monkeypatch.setattr(music, "_fetch", fake_fetch)
    rec = music.download_patient_excerpt("P0001", tmp_path / "out", start_s=5.0, duration_s=3.0, base_url="x/")
    e = read_wfdb(rec)
    assert e.signal.shape == (3, 3000) and e.fs == 1000
    assert calls[1] == (5000 * 6, 8000 * 6 - 1)
    np.testing.assert_allclose(e.signal[0] * 400, ramp[5000:8000], atol=1e-3)
    # Second call is a no-op (already downloaded).
    music.download_patient_excerpt("P0001", tmp_path / "out", start_s=5.0, duration_s=3.0, base_url="x/")
    assert len(calls) == 2


def _fake_subject_info(path, ecg_dir=None):
    rows = []
    for i, (ef, cause, exit_code) in enumerate([(30, 6, 3), (55, 0, None), (40, 3, 3), (25, 0, 2)]):
        r = dict.fromkeys(MUSIC_COLUMNS, "")
        r.update({"Patient ID": f"P{i + 1:04d}", "Age": 60 + i, "Gender (male=1)": i % 2, "LVEF (%)": ef,
                  "Left ventricle end-diastolic diameter (mm)": 60, "Left ventricle end-systolic diameter (mm)": 70 if i == 1 else 45,
                  "Septal thickness (mm)": 10, "Left ventricular posterior wall thickness (mm)": "9,5",
                  "Left atrial size (mm)": 45, "Right ventricle end-diastolic diameter (mm)": 25,
                  "Hig-resolution ECG available": 1, "ECG rhythm ": i % 2, "Cause of death": cause,
                  "Exit of the study": exit_code if exit_code is not None else "",
                  "Follow-up period from enrollment (days)": 1000 + i})
        rows.append(r)
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)


def test_music_cohort_targets_and_outcomes(tmp_path):
    _fake_subject_info(tmp_path / "subject-info.csv")
    c = music_cohort(tmp_path, require_ecg=False)
    assert c["lvpwd"].iloc[0] == pytest.approx(9.5)  # decimal comma parsed
    assert np.isnan(c["lvesd"].iloc[1])  # LVESD > LVEDD is rejected
    assert c["event_hf"].tolist() == [1, 0, 0, 1]  # pump failure (6) and transplant (exit 2)
    assert c["event_cardiac_death"].tolist() == [1, 0, 1, 0]
    assert c["sex"].tolist() == ["F", "M", "F", "M"]
    assert c["lv_mass"].notna().all()
    assert c["ecg_path"].iloc[0] == "High-resolution_ECG/P0001_H"
    assert len(music_cohort(tmp_path, require_ecg=True)) == 0  # no ECG files present


def test_real_music_header_format_is_wfdb_readable(tmp_path):
    sig = np.zeros((100, 3))
    wfdb.wrsamp("P0009_H", fs=1000, units=["mV"] * 3, sig_name=["X", "Y", "Z"], p_signal=sig, fmt=["16"] * 3,
                adc_gain=[400.0] * 3, baseline=[0] * 3, write_dir=str(tmp_path))
    h = parse_header((tmp_path / "P0009_H.hea").read_text())
    assert h["n_sig"] == 3 and h["sig_len"] == 100
