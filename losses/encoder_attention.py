from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .dice import binary_target, resize_to_shape, soft_dice_loss


_DEFAULT_FOUR_STAGE_WEIGHTS = (0.05, 0.10, 0.20, 0.40)


def extract_encoder_feature_list(features: Any) -> list[torch.Tensor]:
    """Extract encoder tensors from common feature containers without layer-name assumptions."""
    if features is None:
        return []
    if isinstance(features, torch.Tensor):
        return [features]
    if hasattr(features, "features"):
        return extract_encoder_feature_list(features.features)
    if isinstance(features, Mapping):
        for key in ("encoder", "encoder_features", "skips"):
            if key in features:
                return extract_encoder_feature_list(features[key])
        if "bottleneck" in features and isinstance(features["bottleneck"], torch.Tensor):
            return [features["bottleneck"]]
        return [item for item in features.values() if isinstance(item, torch.Tensor)]
    if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        output: list[torch.Tensor] = []
        for item in features:
            output.extend(extract_encoder_feature_list(item))
        return output
    return []


def _auto_weights(num_stages: int, reference: torch.Tensor) -> torch.Tensor:
    weights = torch.arange(1, int(num_stages) + 1, device=reference.device, dtype=reference.dtype)
    return weights / weights.sum().clamp_min(1e-12)


def _resolve_weights(weights: str | Sequence[float] | None, num_stages: int, reference: torch.Tensor) -> torch.Tensor:
    if num_stages <= 0:
        raise ValueError("EncoderAttentionLoss requires at least one encoder feature tensor.")
    if weights is None:
        if num_stages == 4:
            return reference.new_tensor(_DEFAULT_FOUR_STAGE_WEIGHTS)
        return _auto_weights(num_stages, reference)
    if isinstance(weights, str):
        if weights.lower() != "auto":
            raise ValueError("attention_weights must be 'auto' or a list of floats.")
        return _auto_weights(num_stages, reference)
    values = [float(item) for item in weights]
    if len(values) != num_stages:
        raise ValueError(
            f"attention_weights length mismatch: got {len(values)} weights for {num_stages} encoder features."
        )
    return reference.new_tensor(values)


def _attention_map(feature: torch.Tensor, eps: float) -> torch.Tensor:
    if feature.ndim not in {4, 5}:
        raise ValueError(f"Encoder feature must be 2D or 3D feature map, got {tuple(feature.shape)}.")
    attention = feature.float().abs().mean(dim=1, keepdim=True)
    reduce_dims = tuple(range(2, attention.ndim))
    min_value = attention.amin(dim=reduce_dims, keepdim=True)
    max_value = attention.amax(dim=reduce_dims, keepdim=True)
    return (attention - min_value) / (max_value - min_value + float(eps))


class EncoderAttentionLoss(nn.Module):
    """Supervise encoder attention maps against downsampled target masks."""

    def __init__(self, attention_weights: str | Sequence[float] | None = None, smooth: float = 1e-6, eps: float = 1e-6) -> None:
        super().__init__()
        self.attention_weights = attention_weights
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, encoder_features: Any, target: torch.Tensor) -> torch.Tensor:
        feature_list = extract_encoder_feature_list(encoder_features)
        if not feature_list:
            return target.new_tensor(0.0, dtype=torch.float32)
        weights = _resolve_weights(self.attention_weights, len(feature_list), feature_list[0].float())
        losses = []
        for weight, feature in zip(weights, feature_list):
            attention = _attention_map(feature, eps=self.eps)
            target_map = binary_target(target, spatial_shape=attention.shape[2:])
            if tuple(target_map.shape[2:]) != tuple(attention.shape[2:]):
                target_map = resize_to_shape(target_map, attention.shape[2:], mode="nearest")
            losses.append(weight * soft_dice_loss(attention, target_map, smooth=self.smooth))
        return torch.stack(losses).sum()
