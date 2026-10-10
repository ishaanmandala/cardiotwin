"""The CardioTwin network: a 1-D ResNet that reads a 10-second ECG.

Input:  ``x`` of shape (batch, leads, samples), in millivolts.
        ``tab`` of shape (batch, tab_dim): optional age/sex features.
Output: for every target a predicted mean and a log-variance (how unsure the
        model is about that particular ECG), plus a 128-d embedding.

For binary targets (e.g. "EF will decline") the "mean" output is a logit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn


@dataclass
class ModelConfig:
    n_leads: int
    targets: list[str]
    target_kinds: list[str]
    tab_dim: int = 0
    width: int = 16
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)
    kernel: int = 9
    dropout: float = 0.2
    embed_dim: int = 128
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blocks_per_stage"] = list(self.blocks_per_stage)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = dict(d)
        d["blocks_per_stage"] = tuple(d.get("blocks_per_stage", (2, 2, 2, 2)))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ResBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.AvgPool1d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv1d(c_in, c_out, 1, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        skip = self.skip(x)
        if skip.shape[-1] != out.shape[-1]:
            skip = skip[..., : out.shape[-1]]
        return self.act(out + skip)


class CardioTwinNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.n_leads, w, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers, c_in = [], w
        for stage, n_blocks in enumerate(cfg.blocks_per_stage):
            c_out = w * (2**stage)
            for b in range(n_blocks):
                stride = 2 if (b == 0 and stage > 0) else 1
                layers.append(ResBlock1d(c_in, c_out, cfg.kernel, stride, cfg.dropout))
                c_in = c_out
        self.body = nn.Sequential(*layers)
        self.ecg_dim = 2 * c_in
        self.tab = nn.Sequential(nn.Linear(cfg.tab_dim, 32), nn.ReLU(inplace=True)) if cfg.tab_dim else None
        fused = self.ecg_dim + (32 if cfg.tab_dim else 0)
        self.trunk = nn.Sequential(
            nn.Linear(fused, cfg.embed_dim), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)
        )
        n_t = len(cfg.targets)
        self.mean_head = nn.Linear(cfg.embed_dim, n_t)
        self.logvar_head = nn.Linear(cfg.embed_dim, n_t)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(x))
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)

    def forward(self, x: torch.Tensor, tab: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.features(x)
        if self.tab is not None:
            if tab is None or tab.shape[-1] != self.cfg.tab_dim:
                raise ValueError(f"Model expects {self.cfg.tab_dim} tabular features (age/sex)")
            h = torch.cat([h, self.tab(tab)], dim=1)
        emb = self.trunk(h)
        return {
            "mean": self.mean_head(emb),
            "log_var": self.logvar_head(emb).clamp(-7.0, 5.0),
            "embedding": emb,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
