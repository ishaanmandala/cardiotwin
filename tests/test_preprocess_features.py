import numpy as np
import pytest

from cardiotwin.data.synthetic import sample_heart, sample_traits, simulate_ecg
from cardiotwin.ecg.features import detect_r_peaks, ecg_features
from cardiotwin.ecg.io import ECG
from cardiotwin.ecg.preprocess import (
    PreprocessConfig,
    fill_nans,
    fit_length,
    preprocess_ecg,
    quality_report,
    resample,
    sliding_windows,
)


def test_resample_length():
    x = np.random.default_rng(0).normal(size=(3, 5000)).astype(np.float32)
    assert resample(x, 500, 250).shape == (3, 2500)
    assert resample(x, 1000, 250).shape == (3, 1250)


def test_fill_nans_interpolates():
    x = np.array([[0.0, np.nan, 2.0, np.nan], [np.nan] * 4], dtype=np.float32)
    y = fill_nans(x)
    np.testing.assert_allclose(y[0], [0, 1, 2, 2])
    np.testing.assert_allclose(y[1], 0)


def test_bandpass_removes_wander_and_mains():
    fs = 500.0
    t = np.arange(int(10 * fs)) / fs
    qrs_like = np.sin(2 * np.pi * 10 * t)
    x = (qrs_like + 1.0 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 60 * t))[None]
    out = preprocess_ecg(ECG(x.astype(np.float32), fs, ["II"]), PreprocessConfig(fs=250)).signal[0]
    ref = qrs_like[::2]
    mid = slice(250, -250)
    assert np.corrcoef(out[mid], ref[mid])[0, 1] > 0.98
    assert np.abs(out[mid] - ref[mid]).max() < 0.25


def test_fit_length_and_windows():
    x = np.arange(10, dtype=np.float32)[None]
    assert fit_length(x, 6).tolist() == [[2, 3, 4, 5, 6, 7]]
    assert fit_length(x, 14).shape == (1, 14)
    w = sliding_windows(np.zeros((2, 100)), 40, 20)
    assert w.shape == (4, 2, 40)
    assert sliding_windows(np.zeros((2, 10)), 40, 20).shape == (1, 2, 40)


def test_quality_flags_flat_and_short():
    x = np.zeros((3, 1000), dtype=np.float32)
    x[0] = np.random.default_rng(0).normal(0, 0.2, 1000)
    rep = quality_report(ECG(x, 500, ["X", "Y", "Z"]))
    assert not rep["ok"]
    assert any("flat" in w for w in rep["warnings"])
    assert any("only 2.0 s" in w for w in rep["warnings"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r_peaks_and_heart_rate_match_simulation(seed):
    rng = np.random.default_rng(seed)
    tr = sample_traits(rng)
    tr.af_u = 0.99  # force sinus rhythm
    h = sample_heart(rng, tr, lvef=60)
    ecg, info = simulate_ecg(h, tr, rng, fs=500)
    pre = preprocess_ecg(ecg, PreprocessConfig())
    f = ecg_features(pre.signal, pre.fs)
    assert f["heart_rate"] == pytest.approx(info["heart_rate"], rel=0.08)
    assert len(detect_r_peaks(pre.signal, pre.fs)) >= int(info["heart_rate"] / 6) - 1


def test_low_ef_widens_qrs_in_simulator():
    widths = {}
    for ef in (65, 20):
        vals = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            tr = sample_traits(rng)
            tr.lbbb_u, tr.mi_u, tr.af_u = 0.99, 0.99, 0.99
            ecg, _ = simulate_ecg(sample_heart(rng, tr, lvef=ef), tr, rng)
            pre = preprocess_ecg(ecg, PreprocessConfig())
            vals.append(ecg_features(pre.signal, pre.fs)["qrs_ms"])
        widths[ef] = np.nanmean(vals)
    assert widths[20] > widths[65] + 10
