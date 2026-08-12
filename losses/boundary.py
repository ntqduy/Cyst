from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dice import foreground_logits, soft_dice_loss


def _pad_after_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    spatial_dims = value.ndim - 2
    spatial_index = dim - 2
    pad = [0, 0] * spatial_dims
    pad[2 * (spatial_dims - 1 - spatial_index) + 1] = 1
    return F.pad(value, tuple(pad))


def _gradient_boundary_map(value: torch.Tensor) -> torch.Tensor:
    """Create a simple differentiable 2D/3D boundary map from finite differences."""
    gradients = []
    for dim in range(2, value.ndim):
        if int(value.shape[dim]) <= 1:
            continue
        gradients.append(_pad_after_dim(torch.diff(value, dim=dim).abs(), dim=dim))
    if not gradients:
        return torch.zeros_like(value)
    return torch.stack(gradients, dim=0).sum(dim=0).clamp(0.0, 1.0)


class BoundaryLoss(nn.Module):
    """Boundary Dice loss using tensor-only finite-difference edges for 2D/3D."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_logits, binary = foreground_logits(logits, target)
        probability = torch.sigmoid(fg_logits.float())
        pred_boundary = _gradient_boundary_map(probability)
        target_boundary = _gradient_boundary_map(binary.float())
        return soft_dice_loss(pred_boundary, target_boundary, smooth=self.smooth)
