from __future__ import annotations

import torch
from torch import nn

from .dice import foreground_logits


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss with beta > alpha to penalize false negatives."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_logits, binary = foreground_logits(logits, target)
        probability = torch.sigmoid(fg_logits.float())
        binary = binary.float()
        reduce_dims = tuple(range(2, probability.ndim))
        true_positive = torch.sum(probability * binary, dim=reduce_dims)
        false_positive = torch.sum(probability * (1.0 - binary), dim=reduce_dims)
        false_negative = torch.sum((1.0 - probability) * binary, dim=reduce_dims)
        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        return torch.pow(1.0 - tversky.clamp(0.0, 1.0), self.gamma).mean()
