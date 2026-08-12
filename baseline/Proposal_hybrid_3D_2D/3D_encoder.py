from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _clean_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _normalization(channels: int, name: str) -> nn.Module:
    normalized = str(name or "batchnorm").lower()
    if normalized in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm3d(channels)
    if normalized in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if normalized in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D normalization: {name}")


class ConvNormAct3D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        normalization: str = "batchnorm",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _normalization(int(out_channels), normalization),
            nn.ReLU(inplace=True),
        )


class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels, normalization=normalization),
            ConvNormAct3D(out_channels, out_channels, normalization=normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3DEncoder(nn.Module):
    """Five-stage 3D encoder returning UNet skip features [x1, x2, x3, x4, x5]."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        normalization: str = "batchnorm",
        pool_kernel_size: int = 2,
        **_: Any,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [int(base_channels) * (2**index) for index in range(5)]
        if len(channels) != 5:
            raise ValueError(f"UNet3DEncoder expects 5 channel stages, got {channels}")

        self.in_channels = int(in_channels)
        self.channels = [int(item) for item in channels]
        self.normalization = str(normalization)
        stage_in_channels = [self.in_channels] + self.channels[:-1]
        self.stages = nn.ModuleList(
            [EncoderBlock3D(in_ch, out_ch, normalization=self.normalization) for in_ch, out_ch in zip(stage_in_channels, self.channels)]
        )
        self.pools = nn.ModuleList(
            [nn.MaxPool3d(kernel_size=int(pool_kernel_size), stride=int(pool_kernel_size), ceil_mode=True) for _ in range(4)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"UNet3DEncoder expects [B,C,D,H,W], got {tuple(x.shape)}")

        features: list[torch.Tensor] = []
        current = x
        for index, stage in enumerate(self.stages):
            current = stage(current)
            features.append(current)
            if index < len(self.pools):
                current = self.pools[index](current)
        return features

    def load_partial_state_dict(self, checkpoint: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        state_dict = _checkpoint_state_dict(checkpoint)
        target_state = self.state_dict()
        updates = {}
        skipped = []
        prefix = str(prefix or "")

        for raw_key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = _clean_key(str(raw_key))
            if prefix:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix) :]
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
                updates[key] = value
            elif key in target_state:
                skipped.append({"key": key, "reason": f"shape {tuple(value.shape)} != {tuple(target_state[key].shape)}"})

        merged = dict(target_state)
        merged.update(updates)
        self.load_state_dict(merged, strict=False)
        missing = [key for key in target_state if key not in updates]
        return {
            "loaded_keys": sorted(updates),
            "skipped_keys": skipped,
            "missing_keys": missing,
            "num_loaded_keys": len(updates),
            "num_skipped_keys": len(skipped),
            "num_missing_keys": len(missing),
        }
