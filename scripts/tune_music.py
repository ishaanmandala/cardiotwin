"""Small hyperparameter check on MUSIC, using development fold 0 ONLY.

The held-out test split is computed exactly as in the experiment (same seed) and
removed before anything else happens, so the test patients never influence the
choice. Run:  python scripts/tune_music.py
"""

from __future__ import annotations

import json
import logging
import sys
import time

import pandas as pd

from cardiotwin.data.splits import patient_kfold, patient_split
from cardiotwin.data.store import SignalStore
from cardiotwin.evaluation.metrics import regression_metrics
from cardiotwin.training.trainer import TrainConfig, aggregate_per_ecg, predict_windows, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
SEED = 7
TARGETS = ["lvef", "lvedd", "lvesd", "ivsd", "lvpwd", "la_diam", "rv_diam", "lv_mass"]

store = SignalStore.open("data/processed/music")
rows = store.usable()
rows = rows.loc[rows["lvef"].notna()].reset_index(drop=True)
rows["split"] = patient_split(rows, (0.8, 0.0, 0.2), seed=SEED).replace({"val": "train"})
dev = rows.loc[rows.split == "train"].reset_index(drop=True)
dev["fold"] = patient_kfold(dev, 5, seed=SEED)
tr, va = dev[dev.fold != 0], dev[dev.fold == 0]
W = 2500

grid = {
    "base_w16": {},
    "w8": {"width": 8},
    "w24_drop03": {"width": 24, "dropout": 0.3},
    "w16_lr1e-3_wd5e-2": {"lr": 1e-3, "weight_decay": 0.05},
    "w16_k15": {"kernel": 15},
}
only = sys.argv[1:] or list(grid)
results = {}
for name in only:
    over = grid[name]
    cfg = TrainConfig.from_dict({"targets": TARGETS, "epochs": 40, "patience": 8, "crops_per_ecg": 8,
                                 "seed": SEED, "eval_windows": 3, **over})
    t0 = time.time()
    m = train_model(store.signals, tr, va, cfg, W, verbose=False)
    p = aggregate_per_ecg(predict_windows([m], store.signals, va, W, 9), TARGETS)
    res = {t: round(regression_metrics(p[f"{t}_true"], p[f"{t}_pred"])["pearson_r"], 3) for t in TARGETS}
    res["lvef_mae"] = round(regression_metrics(p["lvef_true"], p["lvef_pred"])["mae"], 2)
    res["best_epoch"] = m.best_epoch
    res["minutes"] = round((time.time() - t0) / 60, 1)
    results[name] = res
    logging.info("%s: %s", name, res)
print(json.dumps(results, indent=2))
pd.DataFrame(results).T.to_csv("runs/music_tuning.csv")
