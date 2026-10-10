"""Level 7: train CardioTwin ("Dumb AI") and get predictions out of it."""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..constants import target_spec
from ..data.dataset import ECGWindowDataset, TabularEncoder, TargetNormalizer
from ..data.splits import assert_no_patient_leakage
from ..models.resnet1d import CardioTwinNet, ModelConfig
from .losses import multitask_loss

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    targets: list[str] = field(default_factory=lambda: ["lvef"])
    tabular: list[str] = field(default_factory=list)  # e.g. ["age", "sex"]
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    patience: int = 8
    crops_per_ecg: int = 4
    eval_windows: int = 3
    width: int = 16
    kernel: int = 9
    dropout: float = 0.2
    nll_weight: float = 0.5
    grad_clip: float = 1.0
    augment: bool = True
    seed: int = 0
    num_threads: int = 4
    window_s: float = 10.0
    init_from: str | None = None  # pretrained encoder (see cardiotwin.pretrain)
    encoder_lr_mult: float = 1.0  # learning-rate multiplier for pretrained layers

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainedModel:
    model: CardioTwinNet
    model_config: ModelConfig
    normalizer: TargetNormalizer
    tab_encoder: TabularEncoder | None
    train_config: TrainConfig
    history: list[dict]
    best_epoch: int


def load_encoder(model: CardioTwinNet, path: str) -> None:
    """Copy pretrained stem/body weights into ``model`` (shapes must match)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    enc_cfg = payload["model_config"]
    for key in ("n_leads", "width", "kernel"):
        if enc_cfg[key] != getattr(model.cfg, key):
            raise ValueError(f"Pretrained encoder has {key}={enc_cfg[key]}, model has {getattr(model.cfg, key)}")
    state = {k: v.float() if v.is_floating_point() else v for k, v in payload["encoder_state"].items()}
    missing = model.load_state_dict(state, strict=False)
    not_loaded = [k for k in missing.missing_keys if k.startswith(("stem.", "body."))]
    if not_loaded:
        raise ValueError(f"Encoder weights missing for {not_loaded[:5]}")
    log.info("Initialized encoder from %s", path)


def set_seed(seed: int) -> None:
    np.random.seed(seed)  # noqa: NPY002 - third-party code may still use the global RNG
    torch.manual_seed(seed)


def _batches_to_device(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch["x"].float(), batch["tab"].float(), batch["y"].float(), batch["mask"].float()


def evaluate_loss(model, loader, is_binary, nll_weight) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, tab, y, mask = _batches_to_device(batch)
            out = model(x, tab if tab.shape[-1] else None)
            _, stats = multitask_loss(out, y, mask, is_binary, nll_weight)
            total += stats["point_loss"] * len(x)
            n += len(x)
    return total / max(n, 1)


def train_model(
    signals: np.ndarray, train_rows: pd.DataFrame, val_rows: pd.DataFrame, cfg: TrainConfig,
    window: int, verbose: bool = True, check_leakage: bool = True,
) -> TrainedModel:
    """Fit one model with early stopping on the validation loss.

    ``train_rows``/``val_rows`` are cohort rows (with ``store_row``) from a
    patient-level split; the function refuses to run if a patient is in both
    (``check_leakage=False`` exists only for the Level 8 demonstration).
    """
    if check_leakage:
        check = pd.concat([train_rows.assign(_s="train"), val_rows.assign(_s="val")])
        assert_no_patient_leakage(check, "_s")
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)

    normalizer = TargetNormalizer.fit(train_rows, cfg.targets)
    tab_encoder = TabularEncoder.fit(train_rows, cfg.tabular) if cfg.tabular else None
    train_ds = ECGWindowDataset(signals, train_rows, cfg.targets, normalizer, tab_encoder, window,
                                train=True, crops_per_ecg=cfg.crops_per_ecg, augment=cfg.augment, seed=cfg.seed)
    val_ds = ECGWindowDataset(signals, val_rows, cfg.targets, normalizer, tab_encoder, window,
                              train=False, eval_windows=cfg.eval_windows)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=len(train_ds) > cfg.batch_size,
                              generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    mcfg = ModelConfig(
        n_leads=int(signals.shape[1]), targets=list(cfg.targets),
        target_kinds=[target_spec(t).kind for t in cfg.targets],
        tab_dim=tab_encoder.dim if tab_encoder else 0, width=cfg.width, kernel=cfg.kernel, dropout=cfg.dropout,
    )
    model = CardioTwinNet(mcfg)
    is_binary = torch.tensor([k == "binary" for k in mcfg.target_kinds])
    if cfg.init_from:
        load_encoder(model, cfg.init_from)
        enc = [p for n, p in model.named_parameters() if n.startswith(("stem.", "body."))]
        rest = [p for n, p in model.named_parameters() if not n.startswith(("stem.", "body."))]
        groups = [{"params": enc, "lr": cfg.lr * cfg.encoder_lr_mult}, {"params": rest, "lr": cfg.lr}]
    else:
        groups = [{"params": list(model.parameters()), "lr": cfg.lr}]
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup = steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_loss, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            x, tab, y, mask = _batches_to_device(batch)
            out = model(x, tab if tab.shape[-1] else None)
            loss, stats = multitask_loss(out, y, mask, is_binary, cfg.nll_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            run += stats["point_loss"] * len(x)
            n += len(x)
        val_loss = evaluate_loss(model, val_loader, is_binary, cfg.nll_weight)
        history.append({"epoch": epoch, "train_loss": run / max(n, 1), "val_loss": val_loss,
                        "seconds": time.time() - t0})
        if verbose:
            log.info("epoch %2d  train %.4f  val %.4f  (%.1fs)", epoch, run / max(n, 1), val_loss, time.time() - t0)
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainedModel(model, mcfg, normalizer, tab_encoder, cfg, history, best_epoch)


@torch.no_grad()
def predict_windows(
    models: list[TrainedModel], signals: np.ndarray, rows: pd.DataFrame, window: int, eval_windows: int = 3,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Predict every eval window of every row with an ensemble.

    Returns one row per (ECG, window) with ``<t>_pred`` (mean), ``<t>_sd`` (total
    uncertainty = average aleatoric variance + spread across ensemble members),
    ``<t>_true`` and identifying columns. Binary targets give a probability.
    """
    first = models[0]
    targets = first.model_config.targets
    kinds = first.model_config.target_kinds
    means, variances = [], []
    for tm in models:
        ds = ECGWindowDataset(signals, rows, targets, tm.normalizer, tm.tab_encoder, window,
                              train=False, eval_windows=eval_windows)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        mu_all, var_all = [], []
        tm.model.eval()
        for batch in loader:
            tab = batch["tab"].float()
            out = tm.model(batch["x"].float(), tab if tab.shape[-1] else None)
            mu = out["mean"].numpy()
            var = np.exp(out["log_var"].numpy())
            mu_all.append(mu)
            var_all.append(var)
        mu = np.concatenate(mu_all)
        var = np.concatenate(var_all)
        mu_real = tm.normalizer.inverse_mean(mu)
        var_real = var * tm.normalizer.std**2
        for j, k in enumerate(kinds):
            if k == "binary":
                mu_real[:, j] = 1 / (1 + np.exp(-mu[:, j]))
                var_real[:, j] = 0.0
        means.append(mu_real)
        variances.append(var_real)
        index = ds.index
    means_a, vars_a = np.stack(means), np.stack(variances)
    mean = means_a.mean(axis=0)
    total_var = vars_a.mean(axis=0) + means_a.var(axis=0)
    rows = rows.reset_index(drop=True)
    row_idx = np.array([i for i, _ in index])
    cols = {
        "patient_id": rows["patient_id"].to_numpy()[row_idx],
        "ecg_id": rows["ecg_id"].to_numpy()[row_idx],
        "window_start": [s for _, s in index],
    }
    for j, t in enumerate(targets):
        cols[f"{t}_pred"] = mean[:, j]
        cols[f"{t}_sd"] = np.sqrt(total_var[:, j])
        cols[f"{t}_true"] = pd.to_numeric(rows[t], errors="coerce").to_numpy()[row_idx]
    return pd.DataFrame(cols)


def aggregate_per_ecg(window_preds: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Average window predictions per ECG (the 'multiple 10-s windows' estimate)."""
    agg = {"patient_id": "first"}
    for t in targets:
        agg[f"{t}_pred"] = "mean"
        agg[f"{t}_sd"] = "mean"
        agg[f"{t}_true"] = "first"
    return window_preds.groupby("ecg_id", sort=False).agg(agg).reset_index()
