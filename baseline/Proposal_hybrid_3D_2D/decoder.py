from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _clean_key(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _normalization_3d(channels: int, name: str) -> nn.Module:
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


def _normalization_2d(channels: int, name: str) -> nn.Module:
    normalized = str(name or "batchnorm").lower()
    if normalized in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    if normalized in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 2D normalization: {name}")


class ConvNormAct3D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        normalization: str = "batchnorm",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=False),
            _normalization_3d(int(out_channels), normalization),
            nn.ReLU(inplace=True),
        )


class ConvNormAct2D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        normalization: str = "batchnorm",
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, bias=False),
            _normalization_2d(int(out_channels), normalization),
            nn.ReLU(inplace=True),
        )


class FusionBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3D(in_channels, out_channels, normalization=normalization),
            ConvNormAct3D(out_channels, out_channels, normalization=normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FusionBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct2D(in_channels, out_channels, normalization=normalization),
            ConvNormAct2D(out_channels, out_channels, normalization=normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def resize3d(feature: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    if tuple(feature.shape[-3:]) == tuple(size):
        return feature
    return F.interpolate(feature, size=size, mode="trilinear", align_corners=False)


def resize2d(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(size):
        return feature
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


class PartialLoadMixin:
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


class UNet3Plus3DDecoder(PartialLoadMixin, nn.Module):
    """UNet 3+ full-scale 3D decoder with the original five-branch skip formula."""

    decoder_name = "full_scale"

    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        num_classes: int = 2,
        fusion_channels: int | None = None,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        if len(self.channels) != 5:
            raise ValueError(f"UNet3Plus3DDecoder expects 5 channel stages, got {self.channels}")
        self.num_classes = int(num_classes)
        self.fusion_channels = int(fusion_channels or self.channels[0])
        self.up_channels = self.fusion_channels * 5
        self.deep_supervision = bool(deep_supervision)
        self.projections = nn.ModuleDict()
        self.fusion_blocks = nn.ModuleDict()
        self.deep_heads = nn.ModuleDict()

        for target_index in range(4):
            key = str(target_index)
            branch_specs = self._branch_specs(target_index)
            self.projections[key] = nn.ModuleList(
                [
                    ConvNormAct3D(self._branch_channels(source_kind, source_index), self.fusion_channels, normalization=normalization)
                    for source_kind, source_index in branch_specs
                ]
            )
            self.fusion_blocks[key] = ConvNormAct3D(
                self.fusion_channels * len(branch_specs),
                self.up_channels,
                normalization=normalization,
            )
            if self.deep_supervision:
                self.deep_heads[key] = nn.Conv3d(self.up_channels, self.num_classes, kernel_size=1)
        self.segmentation_head = nn.Conv3d(self.up_channels, self.num_classes, kernel_size=1)

    @staticmethod
    def history_indices(target_index: int) -> list[int]:
        return list(range(3, int(target_index), -1))

    def _branch_channels(self, source_kind: str, source_index: int) -> int:
        if source_kind == "encoder":
            return self.channels[int(source_index)]
        return self.up_channels

    @staticmethod
    def _branch_specs(target_index: int) -> list[tuple[str, int]]:
        target_index = int(target_index)
        if target_index == 3:
            return [("encoder", 0), ("encoder", 1), ("encoder", 2), ("encoder", 3), ("encoder", 4)]
        if target_index == 2:
            return [("encoder", 0), ("encoder", 1), ("encoder", 2), ("decoder", 3), ("encoder", 4)]
        if target_index == 1:
            return [("encoder", 0), ("encoder", 1), ("decoder", 2), ("decoder", 3), ("encoder", 4)]
        if target_index == 0:
            return [("encoder", 0), ("decoder", 1), ("decoder", 2), ("decoder", 3), ("encoder", 4)]
        raise ValueError(f"Unsupported UNet3Plus decoder target index: {target_index}")

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(skips) != 5:
            raise ValueError(f"UNet3Plus3DDecoder expects 5 skip tensors, got {len(skips)}")

        decoded: dict[int, torch.Tensor] = {}
        for target_index in (3, 2, 1, 0):
            target_size = tuple(int(item) for item in skips[target_index].shape[-3:])
            key = str(target_index)
            projected = []
            for projection, (source_kind, source_index) in zip(self.projections[key], self._branch_specs(target_index)):
                source = skips[source_index] if source_kind == "encoder" else decoded[source_index]
                projected.append(projection(resize3d(source, target_size)))
            decoded[target_index] = self.fusion_blocks[key](torch.cat(projected, dim=1))

        logits = self.segmentation_head(decoded[0])
        if not self.deep_supervision:
            return logits
        full_size = tuple(int(item) for item in skips[0].shape[-3:])
        return tuple([logits] + [resize3d(self.deep_heads[str(index)](decoded[index]), full_size) for index in (1, 2, 3)])


class UNet3Plus2DDecoder(nn.Module):
    """UNet 3+ full-scale 2D decoder, matching Proposal_Model_Experiment."""

    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        num_classes: int = 2,
        fusion_channels: int | None = None,
        normalization: str = "batchnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        if len(self.channels) != 5:
            raise ValueError(f"UNet3Plus2DDecoder expects 5 channel stages, got {self.channels}")
        self.fusion_channels = int(fusion_channels or self.channels[0])
        self.projections = nn.ModuleDict()
        self.decoder_projections = nn.ModuleDict()
        self.fusion_blocks = nn.ModuleDict()

        for target_index in range(4):
            key = str(target_index)
            history = UNet3Plus3DDecoder.history_indices(target_index)
            self.projections[key] = nn.ModuleList(
                [
                    ConvNormAct2D(source, self.fusion_channels, kernel_size=1, padding=0, normalization=normalization)
                    for source in self.channels
                ]
            )
            self.decoder_projections[key] = nn.ModuleList(
                [
                    ConvNormAct2D(self.channels[source], self.fusion_channels, kernel_size=1, padding=0, normalization=normalization)
                    for source in history
                ]
            )
            self.fusion_blocks[key] = FusionBlock2D(
                self.fusion_channels * (len(self.channels) + len(history)),
                self.channels[target_index],
                normalization=normalization,
            )
        self.segmentation_head = nn.Conv2d(self.channels[0], int(num_classes), kernel_size=1)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(skips) != 5:
            raise ValueError(f"UNet3Plus2DDecoder expects 5 skip tensors, got {len(skips)}")

        decoded: dict[int, torch.Tensor] = {}
        for target_index in (3, 2, 1, 0):
            key = str(target_index)
            target_size = tuple(int(item) for item in skips[target_index].shape[-2:])
            projected = [projection(resize2d(source, target_size)) for projection, source in zip(self.projections[key], skips)]
            for projection, source_index in zip(self.decoder_projections[key], UNet3Plus3DDecoder.history_indices(target_index)):
                projected.append(projection(resize2d(decoded[source_index], target_size)))
            decoded[target_index] = self.fusion_blocks[key](torch.cat(projected, dim=1))
        return self.segmentation_head(decoded[0])


__all__ = ["UNet3Plus2DDecoder", "UNet3Plus3DDecoder"]
