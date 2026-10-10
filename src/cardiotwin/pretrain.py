"""Pretrain the CardioTwin ECG encoder on PTB-XL (transfer learning).

    cardiotwin download-ptbxl
    cardiotwin pretrain --leads xyz --vcg-z-anterior --out models/encoders/ptbxl-xyz.pt   # for MUSIC
    cardiotwin pretrain --leads 12  --out models/encoders/ptbxl-12lead.pt                 # for MIMIC-IV

Targets: ~40 cardiologist statements (``dx_*``), age and sex. PTB-XL's
recommended folds are used: 1-8 train, 9 validation, 10 test.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import STANDARD_12_LEADS, VCG_LEADS
from .data.ptbxl import ptbxl_cohort
from .data.store import SignalStore, build_signal_store
from .ecg.preprocess import PreprocessConfig
from .evaluation.metrics import binary_metrics, regression_metrics
from .training.trainer import TrainConfig, predict_windows, train_model

log = logging.getLogger(__name__)


def build_ptbxl_store(raw_dir: str | Path, out_dir: str | Path, leads: str = "xyz", vcg_z_anterior: bool = False,
                      fs: float = 250.0, workers: int = 4) -> tuple[SignalStore, list[str]]:
    cohort, labels = ptbxl_cohort(raw_dir)
    cohort["is_male"] = (cohort["sex"] == "M").astype(int)
    out_dir = Path(out_dir)
    if (out_dir / "store_meta.json").exists():
        store = SignalStore.open(out_dir)
        if len(store.cohort) == len(cohort) and store.meta.get("vcg_z_anterior") == vcg_z_anterior:
            log.info("Reusing PTB-XL store at %s", out_dir)
            return store, labels
    lead_list = VCG_LEADS if leads == "xyz" else STANDARD_12_LEADS
    store = build_signal_store(cohort, raw_dir, out_dir, lead_list, PreprocessConfig(fs=fs), store_seconds=10.0,
                               workers=workers, edge_trim_s=0.0, vcg_z_anterior=vcg_z_anterior)
    return store, labels


def pretrain_encoder(store: SignalStore, labels: list[str], out_path: str | Path, epochs: int = 12,
                     width: int = 16, kernel: int = 9, seed: int = 0, batch_size: int = 128) -> dict:
    rows = store.usable()
    targets = ["age", "is_male"] + labels
    tr = rows.loc[rows["strat_fold"] <= 8]
    va = rows.loc[rows["strat_fold"] == 9]
    te = rows.loc[rows["strat_fold"] == 10]
    cfg = TrainConfig(targets=targets, tabular=[], epochs=epochs, patience=4, batch_size=batch_size, lr=2e-3,
                      crops_per_ecg=1, width=width, kernel=kernel, dropout=0.1, seed=seed, eval_windows=1)
    t0 = time.time()
    window = int(round(cfg.window_s * store.fs))
    m = train_model(store.signals, tr, va, cfg, window, verbose=True)
    preds = predict_windows([m], store.signals, te, window, 1)
    aucs = {}
    for lab in ["is_male"] + labels:
        res = binary_metrics(preds[f"{lab}_true"], preds[f"{lab}_pred"])
        if "auroc" in res:
            aucs[lab] = res["auroc"]
    age = regression_metrics(preds["age_true"], preds["age_pred"])
    summary = {
        "dataset": "PTB-XL v1.0.3 (CC BY 4.0), doi:10.13026/kfzx-aw45",
        "leads": store.leads, "vcg_z_anterior": bool(store.meta.get("vcg_z_anterior", False)),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "labels": labels, "test_macro_auroc": float(np.mean([v for k, v in aucs.items() if k != "is_male"])),
        "test_auroc": aucs, "test_age_mae": age.get("mae"), "best_epoch": m.best_epoch,
        "minutes": (time.time() - t0) / 60,
    }
    state = {k: v.half() if v.is_floating_point() else v for k, v in m.model.state_dict().items()
             if k.startswith(("stem.", "body."))}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": m.model_config.to_dict(), "encoder_state": state, "summary": summary}, out_path)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    log.info("Pretrained encoder saved to %s (test macro AUROC %.3f, age MAE %.1f y)", out_path,
             summary["test_macro_auroc"], summary["test_age_mae"])
    return summary


def describe_labels(store: SignalStore, labels: list[str]) -> pd.DataFrame:
    rows = store.usable()
    return pd.DataFrame({"label": labels, "prevalence": [rows[lab].mean() for lab in labels]})
