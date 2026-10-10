"""PyTorch dataset: 10-second ECG windows + age/sex + echo targets (with missing values)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import target_spec


@dataclass
class TargetNormalizer:
    """Standardizes regression targets with training-set mean/std (binary targets pass through)."""

    names: list[str]
    mean: np.ndarray
    std: np.ndarray
    kinds: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, names: list[str]) -> TargetNormalizer:
        kinds = [target_spec(n).kind for n in names]
        mean = np.zeros(len(names), dtype=np.float32)
        std = np.ones(len(names), dtype=np.float32)
        for i, (n, k) in enumerate(zip(names, kinds)):
            if k == "regression":
                v = pd.to_numeric(df[n], errors="coerce").to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if len(v) >= 2:
                    mean[i], std[i] = float(v.mean()), float(max(v.std(), 1e-6))
        return cls(list(names), mean, std, kinds)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_mean(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def inverse_std(self, z_std: np.ndarray) -> np.ndarray:
        return z_std * self.std

    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(), "kinds": self.kinds}

    @classmethod
    def from_dict(cls, d: dict) -> TargetNormalizer:
        return cls(list(d["names"]), np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32),
                   list(d["kinds"]))


@dataclass
class TabularEncoder:
    """Encodes age and sex (and optional extra numeric columns) as model inputs."""

    columns: list[str]
    mean: dict
    std: dict

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> TabularEncoder:
        mean, std = {}, {}
        for c in columns:
            if c == "sex":
                continue
            v = pd.to_numeric(df[c], errors="coerce")
            mean[c] = float(v.mean()) if v.notna().any() else 0.0
            std[c] = float(v.std()) if v.notna().sum() > 1 and v.std() > 0 else 1.0
        return cls(list(columns), mean, std)

    @property
    def dim(self) -> int:
        return len(self.columns)

    def encode_row(self, row: dict) -> np.ndarray:
        out = []
        for c in self.columns:
            v = row.get(c)
            if c == "sex":
                out.append(1.0 if v == "M" else (0.0 if v == "F" else 0.5))
            else:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = np.nan
                out.append(0.0 if not np.isfinite(f) else (f - self.mean[c]) / self.std[c])
        return np.asarray(out, dtype=np.float32)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.stack([self.encode_row(r) for r in df[self.columns].to_dict("records")])

    def to_dict(self) -> dict:
        return {"columns": self.columns, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> TabularEncoder:
        return cls(list(d["columns"]), dict(d["mean"]), dict(d["std"]))


def eval_window_starts(n_total: int, n_window: int, n_windows: int) -> list[int]:
    """Evenly spaced, deterministic window starts (used for validation/testing)."""
    if n_total <= n_window or n_windows <= 1:
        return [max(0, (n_total - n_window) // 2)]
    return [int(round(s)) for s in np.linspace(0, n_total - n_window, n_windows)]


class ECGWindowDataset(Dataset):
    """Yields dicts with ``x`` (leads, window), ``tab``, ``y`` (standardized), ``mask``, ``row``.

    Training mode draws a random window per item (``crops_per_ecg`` items per ECG
    per epoch) and applies light augmentation. Evaluation mode uses
    ``eval_windows`` fixed windows per ECG.
    """

    def __init__(
        self, signals: np.ndarray, rows: pd.DataFrame, targets: list[str], normalizer: TargetNormalizer,
        tab_encoder: TabularEncoder | None, window: int, train: bool, crops_per_ecg: int = 1,
        eval_windows: int = 1, augment: bool = True, seed: int = 0,
    ):
        self.signals = signals
        self.rows = rows.reset_index(drop=True)
        self.targets = targets
        self.normalizer = normalizer
        self.window = window
        self.train = train
        self.augment = augment and train
        self.rng = np.random.default_rng(seed)
        self.store_rows = self.rows["store_row"].to_numpy(dtype=int)
        y = self.rows[targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) \
            if targets else np.zeros((len(self.rows), 0), np.float32)
        self.mask = np.isfinite(y).astype(np.float32)
        self.y = np.nan_to_num(normalizer.transform(y) if targets else y).astype(np.float32)
        self.tab = tab_encoder.encode(self.rows) if tab_encoder else np.zeros((len(self.rows), 0), np.float32)
        n_total = signals.shape[-1]
        if "fixed_start" in self.rows.columns:
            # Pre-cut windows (used by the Level 8 leakage demonstration).
            fixed = self.rows["fixed_start"].to_numpy(dtype=int)
            self.index = [(i, int(fixed[i])) for i in range(len(self.rows))]
        elif train:
            self.index = [(i, -1) for i in range(len(self.rows)) for _ in range(crops_per_ecg)]
        else:
            starts = eval_window_starts(n_total, window, eval_windows)
            self.index = [(i, s) for i in range(len(self.rows)) for s in starts]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        i, start = self.index[k]
        sig = self.signals[self.store_rows[i]]
        n_total = sig.shape[-1]
        if start < 0:
            start = int(self.rng.integers(0, max(1, n_total - self.window + 1)))
        x = np.asarray(sig[:, start : start + self.window], dtype=np.float32)
        if x.shape[-1] < self.window:
            x = np.pad(x, ((0, 0), (0, self.window - x.shape[-1])))
        if self.augment:
            x = x * self.rng.uniform(0.85, 1.15)
            x = x + self.rng.normal(0, self.rng.uniform(0.0, 0.03), size=x.shape).astype(np.float32)
            t = np.linspace(0, 1, x.shape[-1], dtype=np.float32)
            x = x + (self.rng.uniform(-0.1, 0.1) * np.sin(2 * np.pi * self.rng.uniform(0.1, 0.5) * 10 * t
                                                          + self.rng.uniform(0, 6.28))).astype(np.float32)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "tab": torch.from_numpy(self.tab[i]),
            "y": torch.from_numpy(self.y[i]),
            "mask": torch.from_numpy(self.mask[i]),
            "row": i,
        }
