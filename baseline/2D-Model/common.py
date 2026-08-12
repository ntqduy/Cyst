from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn.functional as F
from torch import nn


def make_norm(num_channels: int, normalization: str = "batchnorm") -> nn.Module:
    normalization = normalization.lower()
    if normalization == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    if normalization == "instancenorm":
        return nn.InstanceNorm2d(num_channels)
    if normalization == "groupnorm":
        num_groups = min(16, num_channels)
        while num_groups > 1 and num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if normalization == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization '{normalization}'.")


def resize_like(source: torch.Tensor, target: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return F.interpolate(source, size=target.shape[-2:], mode=mode, align_corners=False if mode != "nearest" else None)


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        normalization: str = "batchnorm",
        activation: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            make_norm(out_channels, normalization),
            activation(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        normalization: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            ConvNormAct2d(in_channels, mid_channels, normalization=normalization, dropout=dropout),
            ConvNormAct2d(mid_channels, out_channels, normalization=normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct2d(in_channels, out_channels, normalization=normalization, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels, normalization),
        )
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels, normalization),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.activation(x + residual)
