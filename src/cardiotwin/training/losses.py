"""Multi-task loss that copes with missing echo measurements.

Not every echo reports every measurement, so each target has a mask. Each
target's loss is averaged over the examples that have it, then the targets are
averaged, so a rarely-measured target still counts as much as EF.

Regression targets: MSE on standardized values (drives the prediction) plus a
Gaussian negative log-likelihood on the log-variance head with the mean
*detached*, so learning "how unsure am I" never pulls the prediction itself.
Binary targets: binary cross-entropy on logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multitask_loss(
    out: dict[str, torch.Tensor], y: torch.Tensor, mask: torch.Tensor, is_binary: torch.Tensor,
    nll_weight: float = 0.5, target_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    mean, log_var = out["mean"], out["log_var"]
    reg = (~is_binary).float()[None, :]
    binm = is_binary.float()[None, :]
    sq = (mean - y) ** 2
    nll = 0.5 * (log_var + (y - mean.detach()) ** 2 * torch.exp(-log_var))
    bce = F.binary_cross_entropy_with_logits(mean, y.clamp(0, 1), reduction="none")
    per_elem = reg * (sq + nll_weight * nll) + binm * bce
    counts = mask.sum(dim=0)
    per_target = (per_elem * mask).sum(dim=0) / counts.clamp(min=1.0)
    present = (counts > 0).float()
    if target_weights is not None:
        present = present * target_weights
    loss = (per_target * present).sum() / present.sum().clamp(min=1e-6)
    with torch.no_grad():
        mse_only = ((sq * reg + bce * binm) * mask).sum(dim=0) / counts.clamp(min=1.0)
        stats = {"loss": float(loss), "point_loss": float((mse_only * present).sum() / present.sum().clamp(min=1e-6))}
    return loss, stats
