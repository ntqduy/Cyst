from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from utils.model_output import BaseSegmentationModel


def _channels_from_base(base_channels: int, stages: int = 5, cap: int = 320) -> tuple[int, ...]:
    base = int(base_channels)
    return tuple(min(base * (2**index), int(cap)) for index in range(int(stages)))


def _norm3d(channels: int, normalization: str) -> nn.Module:
    normalization = str(normalization).lower()
    if normalization == "instancenorm":
        return nn.InstanceNorm3d(channels, affine=True)
    if normalization == "batchnorm":
        return nn.BatchNorm3d(channels)
    if normalization == "groupnorm":
        groups = min(16, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalization == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {normalization}")


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(out_channels, normalization),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm3d(out_channels, normalization),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(float(dropout)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownStage3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ConvBlock3D(out_channels, out_channels, normalization=normalization, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpStage3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, normalization: str, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock3D(out_channels + skip_channels, out_channels, normalization=normalization, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([skip, x], dim=1))


class NNUNet3D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feature_channels: Sequence[int] | None = None,
        base_channels: int = 32,
        normalization: str = "instancenorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in (feature_channels or _channels_from_base(base_channels)))
        if len(channels) < 2:
            raise ValueError("NNUNet3D expects at least 2 feature stages.")
        self.model_name = "nnunet"
        self.backbone_name = "nnunet3d_encoder"
        self.set_architecture_config(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            feature_channels=list(channels),
            base_channels=int(base_channels),
            normalization=normalization,
            dropout=float(dropout),
        )

        self.stem = ConvBlock3D(int(in_channels), channels[0], normalization=normalization, dropout=float(dropout))
        self.encoder = nn.ModuleList(
            DownStage3D(channels[index - 1], channels[index], normalization=normalization, dropout=float(dropout))
            for index in range(1, len(channels))
        )
        self.decoder = nn.ModuleList(
            UpStage3D(channels[index], channels[index - 1], channels[index - 1], normalization=normalization, dropout=float(dropout))
            for index in range(len(channels) - 1, 0, -1)
        )
        self.head = nn.Conv3d(channels[0], int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        skips = [self.stem(x)]
        for stage in self.encoder:
            skips.append(stage(skips[-1]))

        x = skips[-1]
        for stage, skip in zip(self.decoder, reversed(skips[:-1])):
            x = stage(x, skip)
        logits = self.head(x)
        output = self.build_output(logits, features={"encoder": skips, "decoder": x})
        if return_features:
            return output.logits, output.features
        return output.logits


nnUNet3D = NNUNet3D
NNUnet3D = NNUNet3D
nnUnet3D = NNUNet3D
