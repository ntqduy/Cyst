from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from networks.common import make_norm
from utils.model_output import BaseSegmentationModel


class ConvBlock(nn.Module):
    def __init__(self, stages: int, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        layers = []
        for stage in range(stages):
            current_in = in_channels if stage == 0 else out_channels
            layers.extend(
                [
                    nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False),
                    make_norm(out_channels, normalization),
                    nn.ReLU(inplace=True),
                ]
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, stages: int, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        layers = []
        for stage in range(stages):
            current_in = in_channels if stage == 0 else out_channels
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(make_norm(out_channels, normalization))
            if stage != stages - 1:
                layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels, normalization),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.skip(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            make_norm(out_channels, normalization),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            make_norm(out_channels, normalization),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        normalization: str = "batchnorm",
        has_dropout: bool = False,
        has_residual: bool = False,
    ) -> None:
        super().__init__()
        block_cls = ResidualConvBlock if has_residual else ConvBlock

        self.block_one = block_cls(1, in_channels, base_channels, normalization=normalization)
        self.block_one_dw = DownsampleBlock(base_channels, base_channels * 2, normalization=normalization)

        self.block_two = block_cls(2, base_channels * 2, base_channels * 2, normalization=normalization)
        self.block_two_dw = DownsampleBlock(base_channels * 2, base_channels * 4, normalization=normalization)

        self.block_three = block_cls(3, base_channels * 4, base_channels * 4, normalization=normalization)
        self.block_three_dw = DownsampleBlock(base_channels * 4, base_channels * 8, normalization=normalization)

        self.block_four = block_cls(3, base_channels * 8, base_channels * 8, normalization=normalization)
        self.block_four_dw = DownsampleBlock(base_channels * 8, base_channels * 16, normalization=normalization)

        self.block_five = block_cls(3, base_channels * 16, base_channels * 16, normalization=normalization)
        self.dropout = nn.Dropout2d(p=0.5)
        self.has_dropout = has_dropout

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.block_one(x)
        x2 = self.block_two(self.block_one_dw(x1))
        x3 = self.block_three(self.block_two_dw(x2))
        x4 = self.block_four(self.block_three_dw(x3))
        x5 = self.block_five(self.block_four_dw(x4))
        if self.has_dropout:
            x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        normalization: str = "batchnorm",
        has_dropout: bool = False,
        has_residual: bool = False,
    ) -> None:
        super().__init__()
        block_cls = ResidualConvBlock if has_residual else ConvBlock

        self.block_five_up = UpsampleBlock(base_channels * 16, base_channels * 8, normalization=normalization)
        self.block_six = block_cls(3, base_channels * 8, base_channels * 8, normalization=normalization)
        self.block_six_up = UpsampleBlock(base_channels * 8, base_channels * 4, normalization=normalization)

        self.block_seven = block_cls(3, base_channels * 4, base_channels * 4, normalization=normalization)
        self.block_seven_up = UpsampleBlock(base_channels * 4, base_channels * 2, normalization=normalization)

        self.block_eight = block_cls(2, base_channels * 2, base_channels * 2, normalization=normalization)
        self.block_eight_up = UpsampleBlock(base_channels * 2, base_channels, normalization=normalization)

        self.block_nine = block_cls(1, base_channels, base_channels, normalization=normalization)
        self.dropout = nn.Dropout2d(p=0.5)
        self.has_dropout = has_dropout
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2, x3, x4, x5 = features

        x = self.block_five_up(x5)
        if x.shape[-2:] != x4.shape[-2:]:
            x = F.interpolate(x, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        x = x + x4

        x = self.block_six(x)
        x = self.block_six_up(x)
        if x.shape[-2:] != x3.shape[-2:]:
            x = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = x + x3

        x = self.block_seven(x)
        x = self.block_seven_up(x)
        if x.shape[-2:] != x2.shape[-2:]:
            x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = x + x2

        x = self.block_eight(x)
        x = self.block_eight_up(x)
        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = x + x1

        decoder_features = self.block_nine(x)
        if self.has_dropout:
            decoder_features = self.dropout(decoder_features)
        logits = self.head(decoder_features)
        return logits, features[-1]


class VNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 16,
        normalization: str = "batchnorm",
        has_dropout: bool = False,
        has_residual: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = "vnet"
        self.backbone_name = "vnet_encoder"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
        )
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
        )
        self.decoder = Decoder(
            num_classes=num_classes,
            base_channels=base_channels,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        encoded_features = self.encoder(x)
        logits, bottleneck = self.decoder(encoded_features)
        output = self.build_output(logits, features={"encoder": encoded_features, "bottleneck": bottleneck})
        if return_features:
            return output.logits, output.features
        return output.logits


VNet = VNet2D
