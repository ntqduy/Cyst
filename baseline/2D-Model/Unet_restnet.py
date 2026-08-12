from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from networks.common import DoubleConv2d
from utils.model_output import BaseSegmentationModel
from utils.pretrained_cache import ensure_pretrain_cache, load_cached_state_dict


def _build_resnet152(encoder_pretrained: bool):
    if not encoder_pretrained:
        return models.resnet152(weights=None)
    ensure_pretrain_cache()
    try:
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=None)
        state_dict = load_cached_state_dict(weights.url, min_bytes=100_000_000)
        model.load_state_dict(state_dict)
        return model
    except (AttributeError, TypeError):
        try:
            return models.resnet152(pretrained=True)
        except Exception as error:
            warnings.warn(
                f"Pretrained ResNet152 could not be loaded/downloaded ({error}). Falling back to random initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                return models.resnet152(weights=None)
            except TypeError:
                return models.resnet152(pretrained=False)
    except Exception as error:
        warnings.warn(
            f"Pretrained ResNet152 could not be loaded/downloaded ({error}). Falling back to random initialization.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            return models.resnet152(weights=None)
        except TypeError:
            return models.resnet152(pretrained=False)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2d(out_channels + skip_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FinalUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv2d(in_channels, out_channels, normalization=normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetResNet152(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        normalization: str = "batchnorm",
        encoder_pretrained: bool = True,
    ) -> None:
        super().__init__()
        backbone = _build_resnet152(encoder_pretrained=encoder_pretrained)
        self.model_name = "unet_resnet152"
        self.backbone_name = "resnet152"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            normalization=normalization,
            encoder_pretrained=bool(encoder_pretrained),
            encoder_name="resnet152",
        )

        if in_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            if encoder_pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        backbone.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                    else:
                        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.center = DoubleConv2d(2048, 2048, normalization=normalization)
        self.dec4 = DecoderBlock(2048, 1024, 512, normalization=normalization)
        self.dec3 = DecoderBlock(512, 512, 256, normalization=normalization)
        self.dec2 = DecoderBlock(256, 256, 128, normalization=normalization)
        self.dec1 = DecoderBlock(128, 64, 64, normalization=normalization)
        self.final_up = FinalUpBlock(64, 32, normalization=normalization)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        center = self.center(x4)
        up1 = self.dec4(center, x3)
        up2 = self.dec3(up1, x2)
        up3 = self.dec2(up2, x1)
        up4 = self.dec1(up3, x0)
        final = self.final_up(up4)
        logits = self.head(final)
        features = {
            "bottleneck": center,
            "encoder": {"stem": x0, "down1": x1, "down2": x2, "down3": x3, "down4": center},
            "decoder": {"up1": up1, "up2": up2, "up3": up3, "up4": final, "final": final},
        }
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits, features = self.forward_features(x)
        output = self.build_output(logits, features=features)
        if return_features:
            return output.logits, output.features
        return output.logits


UNetRestNet152 = UNetResNet152
