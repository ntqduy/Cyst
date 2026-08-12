from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .boundary import BoundaryLoss
from .dice import DiceLoss, foreground_logits, soft_dice_loss
from .encoder_attention import EncoderAttentionLoss
from .focal import BinaryBCEWithLogitsLoss, FocalLoss
from .focal_tversky import FocalTverskyLoss


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalise_loss_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "training" in config and isinstance(config.get("training"), Mapping):
        train_cfg = _as_mapping(config.get("training"))
        raw_loss = config.get("loss", train_cfg.get("loss", "ce_dice"))
    else:
        train_cfg = _as_mapping(config)
        raw_loss = train_cfg.get("loss", "ce_dice")

    if isinstance(raw_loss, Mapping):
        loss_cfg = dict(raw_loss)
    else:
        loss_cfg = {"name": str(raw_loss)}
    merged = dict(train_cfg)
    merged.update(loss_cfg)
    merged["name"] = str(merged.get("name", "ce_dice")).lower()
    return merged


def _tensor_zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().sum() * 0.0


def binary_iou_loss(probability: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probability = probability.float()
    target = target.float()
    reduce_dims = tuple(range(2, probability.ndim))
    intersection = torch.sum(probability * target, dim=reduce_dims)
    union = torch.sum(probability + target, dim=reduce_dims) - intersection
    iou = (intersection + float(smooth)) / (union + float(smooth))
    return 1.0 - iou.mean()


def ms_ssim_loss(probability: torch.Tensor, target: torch.Tensor, levels: int = 5) -> torch.Tensor:
    probability = probability.float()
    target = target.float()
    if probability.ndim != 4:
        return probability.new_tensor(0.0)
    weights = probability.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = int(min(levels, weights.numel()))
    min_spatial = int(min(probability.shape[-2:]))
    while levels > 1 and min_spatial < 2 ** (levels - 1):
        levels -= 1
    weights = weights[:levels]
    weights = weights / weights.sum()

    scores = []
    x = probability
    y = target
    for _ in range(levels):
        mean_x = x.mean(dim=(-2, -1))
        mean_y = y.mean(dim=(-2, -1))
        var_x = x.var(dim=(-2, -1), unbiased=False)
        var_y = y.var(dim=(-2, -1), unbiased=False)
        cov_xy = ((x - mean_x[..., None, None]) * (y - mean_y[..., None, None])).mean(dim=(-2, -1))
        c1 = 0.01**2
        c2 = 0.03**2
        ssim = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / (
            (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
        )
        scores.append(ssim.clamp(min=1e-6, max=1.0).mean())
        if min(x.shape[-2:]) < 2:
            break
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)

    stacked = torch.stack(scores)
    weights = weights[: stacked.numel()]
    weights = weights / weights.sum()
    return 1.0 - torch.prod(stacked**weights)


class UNet3PlusHybridLoss(nn.Module):
    """Legacy UNet 3+ hybrid foreground loss: BCE + IoU + MS-SSIM."""

    requires_encoder_features = False

    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 1.0, msssim_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.iou_weight = float(iou_weight)
        self.msssim_weight = float(msssim_weight)
        self.last_components: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, encoder_features: Any | None = None) -> torch.Tensor:
        fg_logits, binary = foreground_logits(logits, target)
        probability = torch.sigmoid(fg_logits)
        bce = F.binary_cross_entropy_with_logits(fg_logits.float(), binary.float())
        iou = binary_iou_loss(probability, binary)
        msssim = ms_ssim_loss(probability, binary)
        total_weight = max(self.bce_weight + self.iou_weight + self.msssim_weight, 1e-12)
        total = (self.bce_weight * bce + self.iou_weight * iou + self.msssim_weight * msssim) / total_weight
        self.last_components = {
            "loss_total": total.detach(),
            "loss_bce": bce.detach(),
            "loss_iou": iou.detach(),
            "loss_msssim": msssim.detach(),
        }
        return total


class CompositeSegmentationLoss(nn.Module):
    """Composable binary segmentation loss used for L1-L6 ablations."""

    def __init__(self, config: Mapping[str, Any], num_classes: int = 2) -> None:
        super().__init__()
        cfg = _normalise_loss_config(config)
        self.loss_name = str(cfg.get("name", "dice_bce")).lower()
        original_loss_name = self.loss_name
        self.num_classes = int(num_classes)
        self.auxiliary_weight = float(cfg.get("auxiliary_weight", cfg.get("deep_supervision_weight", 1.0)))
        self.last_components: dict[str, torch.Tensor] = {}

        self.dice = DiceLoss(smooth=float(cfg.get("smooth", 1e-6)), from_logits=True)
        self.bce = BinaryBCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=float(cfg.get("focal_alpha", cfg.get("alpha_focal", 0.25))), gamma=float(cfg.get("focal_gamma", 2.0)))
        self.focal_tversky = FocalTverskyLoss(
            alpha=float(cfg.get("alpha", 0.3)),
            beta=float(cfg.get("beta", 0.7)),
            gamma=float(cfg.get("gamma", 1.33)),
            smooth=float(cfg.get("smooth", 1e-6)),
        )
        self.boundary = BoundaryLoss(smooth=float(cfg.get("smooth", 1e-6)))
        self.attention = EncoderAttentionLoss(
            attention_weights=cfg.get("attention_weights", None),
            smooth=float(cfg.get("smooth", 1e-6)),
            eps=float(cfg.get("eps", 1e-6)),
        )

        legacy_ce_dice = original_loss_name in {"ce_dice", "dice_ce", "combined"}
        default_dice_weight = 0.5 if legacy_ce_dice else 1.0
        default_bce_weight = 0.5 if legacy_ce_dice else 1.0
        self.lambda_dice = float(cfg.get("lambda_dice", cfg.get("dice_weight", default_dice_weight)))
        self.lambda_bce = float(cfg.get("lambda_bce", cfg.get("bce_weight", cfg.get("ce_weight", default_bce_weight))))
        self.lambda_focal = float(cfg.get("lambda_focal", 1.0))
        self.lambda_focal_tversky = float(cfg.get("lambda_focal_tversky", 1.0))
        self.lambda_boundary = float(cfg.get("lambda_boundary", 0.2))
        self.lambda_attention = float(cfg.get("lambda_attention", 0.1))

        aliases = {
            "ce_dice": "dice_bce",
            "dice_ce": "dice_bce",
            "combined": "dice_bce",
            "bce_dice": "dice_bce",
            "dice_bce": "dice_bce",
            "dice_focal": "dice_focal",
            "dice_focal_tversky": "dice_focal_tversky",
            "dice_ft": "dice_focal_tversky",
            "dice_focal_tversky_boundary": "dice_focal_tversky_boundary",
            "dice_ft_boundary": "dice_focal_tversky_boundary",
            "dice_focal_tversky_attention": "dice_focal_tversky_attention",
            "dice_ft_attention": "dice_focal_tversky_attention",
            "proposed": "proposed",
            "l1": "dice_bce",
            "l2": "dice_focal",
            "l3": "dice_focal_tversky",
            "l4": "dice_focal_tversky_boundary",
            "l5": "dice_focal_tversky_attention",
            "l6": "proposed",
        }
        if self.loss_name in {"ce", "cross_entropy"}:
            self.loss_name = "bce"
        else:
            self.loss_name = aliases.get(self.loss_name, self.loss_name)
        if self.loss_name not in {
            "dice",
            "bce",
            "dice_bce",
            "dice_focal",
            "dice_focal_tversky",
            "dice_focal_tversky_boundary",
            "dice_focal_tversky_attention",
            "proposed",
        }:
            raise ValueError(
                "Unsupported loss '{}'. Available: dice_bce, dice_focal, dice_focal_tversky, "
                "dice_focal_tversky_boundary, dice_focal_tversky_attention, proposed, unet3plus_hybrid.".format(
                    cfg.get("name")
                )
            )
        self.requires_encoder_features = self.loss_name in {"dice_focal_tversky_attention", "proposed"}

    def _component(self, key: str, value: torch.Tensor, components: dict[str, torch.Tensor]) -> torch.Tensor:
        components[key] = value
        return value

    def forward(self, logits: torch.Tensor, target: torch.Tensor, encoder_features: Any | None = None) -> torch.Tensor:
        components: dict[str, torch.Tensor] = {}
        total = _tensor_zero_like(logits)

        if self.loss_name == "dice":
            dice = self._component("loss_dice", self.dice(logits, target), components)
            total = total + self.lambda_dice * dice
        elif self.loss_name == "bce":
            bce = self._component("loss_bce", self.bce(logits, target), components)
            total = total + self.lambda_bce * bce
        else:
            dice = self._component("loss_dice", self.dice(logits, target), components)
            total = total + self.lambda_dice * dice

        if self.loss_name == "dice_bce":
            bce = self._component("loss_bce", self.bce(logits, target), components)
            total = total + self.lambda_bce * bce
        elif self.loss_name == "dice_focal":
            focal = self._component("loss_focal", self.focal(logits, target), components)
            total = total + self.lambda_focal * focal
        elif self.loss_name in {
            "dice_focal_tversky",
            "dice_focal_tversky_boundary",
            "dice_focal_tversky_attention",
            "proposed",
        }:
            focal_tversky = self._component("loss_focal_tversky", self.focal_tversky(logits, target), components)
            total = total + self.lambda_focal_tversky * focal_tversky

        if self.loss_name in {"dice_focal_tversky_boundary", "proposed"}:
            boundary = self._component("loss_boundary", self.boundary(logits, target), components)
            total = total + self.lambda_boundary * boundary

        if self.loss_name in {"dice_focal_tversky_attention", "proposed"} and encoder_features is not None:
            attention = self._component("loss_attention", self.attention(encoder_features, target), components)
            total = total + self.lambda_attention * attention

        components["loss_total"] = total
        self.last_components = {key: value.detach() for key, value in components.items()}
        return total


class SegmentationCriterion(nn.Module):
    """Backward-compatible criterion wrapper for the existing training code."""

    def __init__(self, config: Mapping[str, Any], num_classes: int) -> None:
        super().__init__()
        cfg = _normalise_loss_config(config)
        loss_name = str(cfg.get("name", "ce_dice")).lower()
        if loss_name in {"unet3plus_hybrid", "unet_3_plus_hybrid", "hybrid", "bce_iou_msssim"}:
            self.loss = UNet3PlusHybridLoss(
                bce_weight=float(cfg.get("hybrid_bce_weight", 1.0)),
                iou_weight=float(cfg.get("hybrid_iou_weight", 1.0)),
                msssim_weight=float(cfg.get("hybrid_msssim_weight", 1.0)),
            )
        else:
            self.loss = CompositeSegmentationLoss(config, num_classes=num_classes)
        self.auxiliary_weight = float(cfg.get("auxiliary_weight", cfg.get("deep_supervision_weight", 1.0)))
        self.requires_encoder_features = bool(getattr(self.loss, "requires_encoder_features", False))

    @property
    def last_components(self) -> dict[str, torch.Tensor]:
        return dict(getattr(self.loss, "last_components", {}))

    @property
    def loss_name(self) -> str:
        return str(getattr(self.loss, "loss_name", self.loss.__class__.__name__))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, encoder_features: Any | None = None) -> torch.Tensor:
        return self.loss(logits, target, encoder_features=encoder_features)


def build_loss(config: Mapping[str, Any], num_classes: int = 2) -> SegmentationCriterion:
    """Factory used by tests and the runner."""
    return SegmentationCriterion(config, num_classes=num_classes)
