from __future__ import annotations

"""Compatibility wrapper for the modular losses package."""

import torch

from losses import (
    BCEWithLogitsLoss,
    BoundaryLoss,
    CompositeSegmentationLoss,
    DiceLoss,
    EncoderAttentionLoss,
    FocalLoss,
    FocalTverskyLoss,
    SegmentationCriterion,
    build_loss,
)
from losses.composite import UNet3PlusHybridLoss, binary_iou_loss, ms_ssim_loss
from losses.dice import foreground_logits, resize_to_shape, soft_dice_loss, target_spatial_shape


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return resize_to_shape(logits, target_spatial_shape(target))


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    include_background: bool = False,
    smooth: float = 1e-6,
) -> torch.Tensor:
    del num_classes, include_background
    return DiceLoss(smooth=smooth, from_logits=True)(logits, target)


def unet3plus_hybrid_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    bce_weight: float = 1.0,
    iou_weight: float = 1.0,
    msssim_weight: float = 1.0,
) -> torch.Tensor:
    del num_classes
    return UNet3PlusHybridLoss(
        bce_weight=bce_weight,
        iou_weight=iou_weight,
        msssim_weight=msssim_weight,
    )(logits, target)


__all__ = [
    "BCEWithLogitsLoss",
    "BoundaryLoss",
    "CompositeSegmentationLoss",
    "DiceLoss",
    "EncoderAttentionLoss",
    "FocalLoss",
    "FocalTverskyLoss",
    "SegmentationCriterion",
    "build_loss",
    "binary_iou_loss",
    "dice_loss",
    "foreground_logits",
    "ms_ssim_loss",
    "resize_logits_to_target",
    "soft_dice_loss",
    "unet3plus_hybrid_loss",
]
