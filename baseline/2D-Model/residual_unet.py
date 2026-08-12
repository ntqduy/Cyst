from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from networks.common import ResidualBlock2d
from utils.model_output import BaseSegmentationModel


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock2d(out_channels + skip_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ResidualUNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        normalization: str = "batchnorm",
        decoder_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in feature_channels)
        if len(channels) != 5:
            raise ValueError("ResidualUNet2D expects exactly 5 encoder stages.")

        self.model_name = "resunet"
        self.backbone_name = "residual_encoder"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_channels=list(channels),
            normalization=normalization,
            decoder_dropout=decoder_dropout,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ResidualBlock2d(in_channels, channels[0], normalization=normalization)
        self.enc2 = ResidualBlock2d(channels[0], channels[1], normalization=normalization)
        self.enc3 = ResidualBlock2d(channels[1], channels[2], normalization=normalization)
        self.enc4 = ResidualBlock2d(channels[2], channels[3], normalization=normalization)
        self.bottleneck = ResidualBlock2d(channels[3], channels[4], normalization=normalization)

        self.dec4 = DecoderBlock(channels[4], channels[3], channels[3], normalization=normalization)
        self.dec3 = DecoderBlock(channels[3], channels[2], channels[2], normalization=normalization)
        self.dec2 = DecoderBlock(channels[2], channels[1], channels[1], normalization=normalization)
        self.dec1 = DecoderBlock(channels[1], channels[0], channels[0], normalization=normalization)

        self.dropout = nn.Dropout2d(decoder_dropout)
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))

        up1 = self.dec4(x5, x4)
        up2 = self.dropout(self.dec3(up1, x3))
        up3 = self.dropout(self.dec2(up2, x2))
        decoder_features = self.dec1(up3, x1)
        logits = self.head(decoder_features)
        features = {
            "bottleneck": x5,
            "encoder": {"enc1": x1, "enc2": x2, "enc3": x3, "enc4": x4, "bottleneck": x5},
            "decoder": {"up1": up1, "up2": up2, "up3": up3, "up4": decoder_features, "final": decoder_features},
        }
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits, features = self.forward_features(x)
        output = self.build_output(logits, features=features)
        if return_features:
            return output.logits, output.features
        return output.logits


ResidualUNet = ResidualUNet2D
