"""Checkpoint round-trip, the predictor and the web API."""

import io

import numpy as np
import pytest
import wfdb
from fastapi.testclient import TestClient

from cardiotwin.app.server import create_app
from cardiotwin.constants import STANDARD_12_LEADS
from cardiotwin.data.synthetic import sample_heart, sample_traits, simulate_ecg
from cardiotwin.ecg.io import ECG, ECGFormatError
from cardiotwin.inference import CardioTwinPredictor, ModelRegistry


@pytest.fixture(scope="module")
def client(tiny_model_dir):
    return TestClient(create_app(tiny_model_dir.parent))


def _sim(seed=0, fs=500, duration=10.0, leads="12"):
    rng = np.random.default_rng(seed)
    tr = sample_traits(rng)
    ecg, _ = simulate_ecg(sample_heart(rng, tr), tr, rng, fs=fs, duration_s=duration, leads=leads)
    return ecg


def test_predictor_round_trip(tiny_model_dir):
    p = CardioTwinPredictor.load(tiny_model_dir)
    assert len(p.members) == 2 and p.targets == ["lvef", "lvedd", "ivsd", "la_diam"]
    res = p.predict(_sim(), age=60, sex="F")
    assert set(res["predictions"]) == set(p.targets)
    for v in res["predictions"].values():
        assert np.isfinite(v["mean"]) and v["sd"] > 0
    assert np.asarray(res["ecg"]["signal"]).shape == (12, 2500)
    assert np.asarray(res["explain"]["saliency"]).shape == (12, 2500)
    assert abs(sum(res["explain"]["segments"].values()) - 1) < 1e-6
    assert res["profile"]["structural_risk"]["level"] in {"LOW", "MODERATE", "HIGH"}
    assert res["profile"]["sex"] == "F"


def test_predictor_handles_other_rates_lengths_and_lead_sets(tiny_model_dir):
    p = CardioTwinPredictor.load(tiny_model_dir)
    long_ecg = _sim(fs=1000, duration=30)
    res = p.predict(long_ecg, explain=False)
    assert res["input"]["windows_used"] > 1
    short = _sim(duration=6)
    assert any("only 6.0 s" in w for w in p.predict(short, explain=False)["warnings"])
    # Only the 8 independent leads: the other 4 are derived.
    full = _sim()
    eight = full.select_leads(["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"])
    res = p.predict(eight, explain=False)
    assert set(res["input"]["derived_leads"]) == {"III", "aVR", "aVL", "aVF"}
    with pytest.raises(ECGFormatError):
        p.predict(ECG(np.zeros((1, 5000)), 500, ["V1"]))


def test_registry_prefers_matching_model(tiny_model_dir):
    reg = ModelRegistry(tiny_model_dir.parent)
    assert len(reg) == 1
    assert reg.get().name == "syn-ecg"
    with pytest.raises(KeyError):
        reg.get("nope")


def test_api_health_models_demos(client):
    assert client.get("/api/health").json()["models"] == 1
    models = client.get("/api/models").json()["models"]
    assert models[0]["name"] == "syn-ecg" and models[0]["leads"] == STANDARD_12_LEADS
    demos = client.get("/api/demos").json()["demos"]
    assert demos[0]["id"] == "demo01" and demos[0]["truth"]["lvef"] == 30.0
    page = client.get("/")
    assert page.status_code == 200 and "CardioTwin" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_api_predict_demo(client):
    r = client.post("/api/predict/demo", json={"demo_id": "demo01"})
    assert r.status_code == 200
    body = r.json()
    lvef = next(i for i in body["profile"]["current"] if i["target"] == "lvef")
    assert lvef["truth"] == 30.0
    assert client.post("/api/predict/demo", json={"demo_id": "../etc/passwd"}).status_code == 404


def test_api_predict_uploads(client, tmp_path):
    ecg = _sim()
    # CSV with a time column
    buf = io.StringIO()
    buf.write("time," + ",".join(ecg.leads) + "\n")
    for k in range(ecg.n_samples):
        buf.write(f"{k / ecg.fs:.4f}," + ",".join(f"{v:.4f}" for v in ecg.signal[:, k]) + "\n")
    r = client.post("/api/predict", files=[("files", ("ecg.csv", buf.getvalue().encode(), "text/csv"))],
                    data={"age": "55", "sex": "M"})
    assert r.status_code == 200, r.text
    # WFDB pair
    wfdb.wrsamp("up", fs=500, units=["mV"] * 12, sig_name=ecg.leads, p_signal=ecg.signal.T.astype(float),
                fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(tmp_path))
    files = [("files", ("up.hea", (tmp_path / "up.hea").read_bytes(), "text/plain")),
             ("files", ("up.dat", (tmp_path / "up.dat").read_bytes(), "application/octet-stream"))]
    r = client.post("/api/predict", files=files)
    assert r.status_code == 200, r.text
    # NumPy array needs fs
    npy = io.BytesIO()
    np.save(npy, ecg.signal)
    r = client.post("/api/predict", files=[("files", ("x.npy", npy.getvalue(), "application/octet-stream"))])
    assert r.status_code == 422 and "sampling rate" in r.json()["detail"]
    r = client.post("/api/predict", files=[("files", ("x.npy", npy.getvalue(), "application/octet-stream"))],
                    data={"fs": "500"})
    assert r.status_code == 200


def test_api_rejects_bad_input(client):
    r = client.post("/api/predict", files=[("files", ("x.pdf", b"%PDF-1.4", "application/pdf"))])
    assert r.status_code == 422
    r = client.post("/api/predict", files=[("files", ("x.csv", b"a,b\n1,2\n", "text/csv"))], data={"age": "abc"})
    assert r.status_code == 422
    r = client.post("/api/predict", files=[("files", ("x.csv", b"V1\n" + b"0.1\n" * 3000, "text/csv"))],
                    data={"fs": "500"})
    assert r.status_code == 422 and "missing required leads" in r.json()["detail"]


def test_app_without_models(tmp_path):
    c = TestClient(create_app(tmp_path / "empty"))
    assert c.get("/api/models").json()["models"] == []
    assert c.post("/api/predict/demo", json={"demo_id": "demo01"}).status_code == 503
