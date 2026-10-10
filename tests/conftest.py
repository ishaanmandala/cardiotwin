"""Shared fixtures: a small synthetic cohort, its signal store and a tiny trained model."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from cardiotwin.constants import STANDARD_12_LEADS
from cardiotwin.data.splits import patient_split
from cardiotwin.data.store import build_signal_store
from cardiotwin.data.synthetic import make_synthetic_cohort
from cardiotwin.ecg.io import ECG, save_npz_ecg
from cardiotwin.ecg.preprocess import PreprocessConfig
from cardiotwin.inference import save_checkpoint
from cardiotwin.training.trainer import TrainConfig, train_model

logging.getLogger("cardiotwin").setLevel(logging.WARNING)
torch.set_num_threads(2)

TARGETS = ["lvef", "lvedd", "ivsd", "la_diam"]


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    df = make_synthetic_cohort(out, n_patients=90, ecgs_per_patient=(1, 2), seed=1)
    return out, df


@pytest.fixture(scope="session")
def store(synth_dir):
    out, df = synth_dir
    return build_signal_store(df, out, out / "store", STANDARD_12_LEADS, PreprocessConfig(), store_seconds=10.0,
                              workers=1)


@pytest.fixture(scope="session")
def tiny_model_dir(store, tmp_path_factory):
    """A 2-member ensemble trained for 2 epochs: just enough to exercise inference and the app."""
    rows = store.usable()
    rows["split"] = patient_split(rows, seed=0)
    tr, va = rows[rows.split == "train"], rows[rows.split == "val"]
    members = [
        train_model(store.signals, tr, va, TrainConfig(targets=TARGETS, tabular=tab, epochs=2, width=8, seed=s,
                                                       crops_per_ecg=1), window=2500, verbose=False)
        for s, tab in ((0, []), (1, []))
    ]
    out = tmp_path_factory.mktemp("models") / "syn-ecg"
    meta = {"name": "syn-ecg", "title": "Synthetic test model", "leads": store.leads,
            "preprocess": store.meta["preprocess"],
            "test_metrics": {"lvef": {"n": 10, "mae": 9.0, "pearson_r": 0.5}},
            "future_model": None}
    save_checkpoint(out, members, meta)
    demo_dir = out / "demo"
    demo_dir.mkdir()
    rng = np.random.default_rng(0)
    from cardiotwin.data.synthetic import sample_heart, sample_traits, simulate_ecg

    tr_ = sample_traits(rng)
    ecg, _ = simulate_ecg(sample_heart(rng, tr_, lvef=30), tr_, rng)
    save_npz_ecg(ECG(ecg.signal, ecg.fs, ecg.leads), demo_dir / "demo01.npz",
                 {"label": "Demo patient 1", "age": 70.0, "sex": "M", "truth": {"lvef": 30.0}})
    return out
