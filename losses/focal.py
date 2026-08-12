from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dice import foreground_logits


class BinaryBCEWithLogitsLoss(nn.Module):
    """BCEWithLogits for binary segmentation with one- or two-channel logits."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_logits, binary = foreground_logits(logits, target)
        return F.binary_cross_entropy_with_logits(fg_logits.float(), binary.float())


BCEWithLogitsLoss = BinaryBCEWithLogitsLoss


class FocalLoss(nn.Module):
    """Binary focal loss for lesion segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fg_logits, binary = foreground_logits(logits, target)
        bce = F.binary_cross_entropy_with_logits(fg_logits.float(), binary.float(), reduction="none")
        probability_t = torch.exp(-bce)
        alpha_t = self.alpha * binary + (1.0 - self.alpha) * (1.0 - binary)
        loss = alpha_t * torch.pow(1.0 - probability_t, self.gamma) * bce
        return loss.mean()
