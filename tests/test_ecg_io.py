import io
import json

import numpy as np
import pytest
import wfdb

from cardiotwin.constants import STANDARD_12_LEADS, VCG_LEADS, canonical_lead_name
from cardiotwin.ecg.io import (
    ECG,
    ECGFormatError,
    load_ecg_from_files,
    read_csv_ecg,
    read_npz_ecg,
    read_wfdb,
    save_npz_ecg,
)


def _ecg(n_leads=12, n=5000, fs=500.0, seed=0):
    rng = np.random.default_rng(seed)
    leads = STANDARD_12_LEADS if n_leads == 12 else VCG_LEADS
    return ECG(rng.normal(0, 0.3, size=(n_leads, n)).astype(np.float32), fs, list(leads))


def test_lead_name_aliases():
    assert canonical_lead_name("AVR") == "aVR"
    assert canonical_lead_name("lead ii") == "II"
    assert canonical_lead_name("v5") == "V5"
    assert canonical_lead_name("X") == "X"


def test_ecg_validates_shapes():
    with pytest.raises(ECGFormatError):
        ECG(np.zeros((2, 10)), 500, ["I"])
    with pytest.raises(ECGFormatError):
        ECG(np.zeros((1, 10)), 0, ["I"])
    e = ECG(np.zeros(100), 100, ["I"])
    assert e.signal.shape == (1, 100) and e.duration_s == 1.0


def test_limb_leads_are_derived_from_i_and_ii():
    rng = np.random.default_rng(1)
    i, ii = rng.normal(size=(2, 500)).astype(np.float32)
    e = ECG(np.stack([i, ii]), 500, ["I", "II"]).select_leads(["I", "II", "III", "aVR", "aVL", "aVF"])
    np.testing.assert_allclose(e.lead("III"), ii - i, atol=1e-6)
    np.testing.assert_allclose(e.lead("aVR"), -(i + ii) / 2, atol=1e-6)
    np.testing.assert_allclose(e.lead("aVL"), i - ii / 2, atol=1e-6)
    np.testing.assert_allclose(e.lead("aVF"), ii - i / 2, atol=1e-6)
    assert set(e.meta["derived_leads"]) == {"III", "aVR", "aVL", "aVF"}


def test_missing_leads_raise_helpful_error():
    e = ECG(np.zeros((1, 100)), 100, ["V1"])
    with pytest.raises(ECGFormatError, match="missing required leads"):
        e.select_leads(["I", "II"])


def test_vcg_round_trip_through_dower_and_kors():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 500)
    xyz = np.stack([np.sin(2 * np.pi * f * t + p) for f, p in ((3, 0), (5, 1), (7, 2))]).astype(np.float32)
    xyz += rng.normal(0, 0.01, size=xyz.shape).astype(np.float32)
    twelve = ECG(xyz, 500, VCG_LEADS).select_leads(STANDARD_12_LEADS)
    back = ECG(twelve.signal, 500, STANDARD_12_LEADS).select_leads(VCG_LEADS)
    for k in range(3):
        assert np.corrcoef(xyz[k], back.signal[k])[0, 1] > 0.9


def test_z_axis_convention_flag_flips_z_only():
    e = _ecg(12)
    post = e.select_leads(VCG_LEADS)
    ant = e.select_leads(VCG_LEADS, vcg_z_anterior=True)
    np.testing.assert_allclose(post.lead("X"), ant.lead("X"))
    np.testing.assert_allclose(post.lead("Z"), -ant.lead("Z"), atol=1e-6)


def test_csv_with_time_column_infers_fs():
    t = np.arange(1000) / 250.0
    csv = "time,I,II\n" + "\n".join(f"{a:.4f},{np.sin(a):.4f},{np.cos(a):.4f}" for a in t)
    e = read_csv_ecg(io.StringIO(csv))
    assert e.fs == pytest.approx(250.0, rel=1e-3)
    assert e.leads == ["I", "II"] and e.n_samples == 1000


def test_csv_without_time_needs_fs():
    csv = "I,II\n" + "\n".join("0.1,0.2" for _ in range(50))
    with pytest.raises(ECGFormatError):
        read_csv_ecg(io.StringIO(csv))
    assert read_csv_ecg(io.StringIO(csv), fs=500).fs == 500


def test_wfdb_round_trip_and_upload(tmp_path):
    e = _ecg(12)
    wfdb.wrsamp("rec1", fs=500, units=["mV"] * 12, sig_name=STANDARD_12_LEADS, p_signal=e.signal.T.astype(float),
                fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(tmp_path))
    r = read_wfdb(tmp_path / "rec1")
    assert r.leads == STANDARD_12_LEADS and r.fs == 500
    np.testing.assert_allclose(r.signal, e.signal, atol=2e-3)
    files = {"rec1.hea": (tmp_path / "rec1.hea").read_bytes(), "rec1.dat": (tmp_path / "rec1.dat").read_bytes()}
    up = load_ecg_from_files(files)
    np.testing.assert_allclose(up.signal, r.signal)
    with pytest.raises(ECGFormatError, match=".hea"):
        load_ecg_from_files({"rec1.dat": files["rec1.dat"]})


def test_npz_and_json_uploads(tmp_path):
    e = _ecg(3, n=2500, fs=250)
    save_npz_ecg(e, tmp_path / "a.npz", {"age": 61})
    back = read_npz_ecg(tmp_path / "a.npz")
    assert back.leads == VCG_LEADS and back.meta["age"] == 61
    up = load_ecg_from_files({"a.npz": (tmp_path / "a.npz").read_bytes()})
    assert up.signal.shape == (3, 2500)
    payload = json.dumps({"fs": 250, "leads": VCG_LEADS, "signal": e.signal.tolist()}).encode()
    assert load_ecg_from_files({"x.json": payload}).fs == 250


def test_unsupported_upload():
    with pytest.raises(ECGFormatError, match="Unsupported"):
        load_ecg_from_files({"x.pdf": b"%PDF"})
    with pytest.raises(ECGFormatError):
        load_ecg_from_files({})
