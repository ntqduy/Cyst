from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.model_output import BaseSegmentationModel


def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != reference.shape[-2:]:
        x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class RecurrentBlock(nn.Module):
    def __init__(self, channels: int, t: int = 2) -> None:
        super().__init__()
        self.t = int(t)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        for _ in range(1, self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.rrcnn = nn.Sequential(
            RecurrentBlock(out_channels, t=t),
            RecurrentBlock(out_channels, t=t),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x + self.rrcnn(x)


class AttentionBlock(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int) -> None:
        super().__init__()
        self.gate_transform = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip_transform = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.relu(self.gate_transform(gate) + self.skip_transform(skip))
        attention = self.psi(attention)
        return skip * attention


class AttentionUNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (64, 128, 256, 512, 1024),
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in feature_channels)
        if len(channels) != 5:
            raise ValueError("AttentionUNet2D expects exactly 5 feature stages.")
        self.model_name = "att_unet"
        self.backbone_name = "attention_unet_encoder"
        self.set_architecture_config(in_channels=in_channels, num_classes=num_classes, feature_channels=list(channels))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.conv4 = ConvBlock(channels[2], channels[3])
        self.conv5 = ConvBlock(channels[3], channels[4])

        self.up5 = UpConv(channels[4], channels[3])
        self.att5 = AttentionBlock(channels[3], channels[3], channels[3] // 2)
        self.up_conv5 = ConvBlock(channels[3] * 2, channels[3])
        self.up4 = UpConv(channels[3], channels[2])
        self.att4 = AttentionBlock(channels[2], channels[2], channels[2] // 2)
        self.up_conv4 = ConvBlock(channels[2] * 2, channels[2])
        self.up3 = UpConv(channels[2], channels[1])
        self.att3 = AttentionBlock(channels[1], channels[1], channels[1] // 2)
        self.up_conv3 = ConvBlock(channels[1] * 2, channels[1])
        self.up2 = UpConv(channels[1], channels[0])
        self.att2 = AttentionBlock(channels[0], channels[0], max(1, channels[0] // 2))
        self.up_conv2 = ConvBlock(channels[0] * 2, channels[0])

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        d5 = self.up5(x5)
        d5 = _match_spatial(d5, x4)
        a4 = self.att5(d5, x4)
        d5 = self.up_conv5(torch.cat((a4, d5), dim=1))
        d4 = self.up4(d5)
        d4 = _match_spatial(d4, x3)
        a3 = self.att4(d4, x3)
        d4 = self.up_conv4(torch.cat((a3, d4), dim=1))
        d3 = self.up3(d4)
        d3 = _match_spatial(d3, x2)
        a2 = self.att3(d3, x2)
        d3 = self.up_conv3(torch.cat((a2, d3), dim=1))
        d2 = self.up2(d3)
        d2 = _match_spatial(d2, x1)
        a1 = self.att2(d2, x1)
        decoder_features = self.up_conv2(torch.cat((a1, d2), dim=1))
        logits = self.head(decoder_features)
        features = {
            "bottleneck": x5,
            "encoder": {"stem": x1, "down1": x2, "down2": x3, "down3": x4, "down4": x5},
            "decoder": {"up1": d5, "up2": d4, "up3": d3, "up4": decoder_features, "final": decoder_features},
        }
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits, features = self.forward_features(x)
        output = self.build_output(logits, features=features)
        if return_features:
            return output.logits, output.features
        return output.logits


class R2UNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (64, 128, 256, 512, 1024),
        recurrent_steps: int = 2,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in feature_channels)
        if len(channels) != 5:
            raise ValueError("R2UNet2D expects exactly 5 feature stages.")
        self.model_name = "r2unet"
        self.backbone_name = "recurrent_residual_unet_encoder"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_channels=list(channels),
            recurrent_steps=int(recurrent_steps),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rrcnn1 = RRCNNBlock(in_channels, channels[0], t=recurrent_steps)
        self.rrcnn2 = RRCNNBlock(channels[0], channels[1], t=recurrent_steps)
        self.rrcnn3 = RRCNNBlock(channels[1], channels[2], t=recurrent_steps)
        self.rrcnn4 = RRCNNBlock(channels[2], channels[3], t=recurrent_steps)
        self.rrcnn5 = RRCNNBlock(channels[3], channels[4], t=recurrent_steps)

        self.up5 = UpConv(channels[4], channels[3])
        self.up_rrcnn5 = RRCNNBlock(channels[3] * 2, channels[3], t=recurrent_steps)
        self.up4 = UpConv(channels[3], channels[2])
        self.up_rrcnn4 = RRCNNBlock(channels[2] * 2, channels[2], t=recurrent_steps)
        self.up3 = UpConv(channels[2], channels[1])
        self.up_rrcnn3 = RRCNNBlock(channels[1] * 2, channels[1], t=recurrent_steps)
        self.up2 = UpConv(channels[1], channels[0])
        self.up_rrcnn2 = RRCNNBlock(channels[0] * 2, channels[0], t=recurrent_steps)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x1 = self.rrcnn1(x)
        x2 = self.rrcnn2(self.pool(x1))
        x3 = self.rrcnn3(self.pool(x2))
        x4 = self.rrcnn4(self.pool(x3))
        x5 = self.rrcnn5(self.pool(x4))

        d5 = self.up5(x5)
        d5 = _match_spatial(d5, x4)
        d5 = self.up_rrcnn5(torch.cat((x4, d5), dim=1))
        d4 = self.up4(d5)
        d4 = _match_spatial(d4, x3)
        d4 = self.up_rrcnn4(torch.cat((x3, d4), dim=1))
        d3 = self.up3(d4)
        d3 = _match_spatial(d3, x2)
        d3 = self.up_rrcnn3(torch.cat((x2, d3), dim=1))
        d2 = self.up2(d3)
        d2 = _match_spatial(d2, x1)
        decoder_features = self.up_rrcnn2(torch.cat((x1, d2), dim=1))
        logits = self.head(decoder_features)
        features = {
            "bottleneck": x5,
            "encoder": {"stem": x1, "down1": x2, "down2": x3, "down3": x4, "down4": x5},
            "decoder": {"up1": d5, "up2": d4, "up3": d3, "up4": decoder_features, "final": decoder_features},
        }
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits, features = self.forward_features(x)
        output = self.build_output(logits, features=features)
        if return_features:
            return output.logits, output.features
        return output.logits


AttUNet = AttentionUNet2D
R2UNet = R2UNet2D
