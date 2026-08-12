from __future__ import annotations

import warnings
from typing import Any, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from networks.common import DoubleConv2d
from utils.model_output import BaseSegmentationModel
from utils.pretrained_cache import ensure_cached_checkpoint, ensure_pretrain_cache


_CUSTOM_ENCODERS = {"", "default", "unet", "unet_encoder"}
_SMP_ENCODER_ALIASES = {
    "mobi": "mobilenet_v2",
    "mobile": "mobilenet_v2",
    "mobilenet": "mobilenet_v2",
    "mobilenetv2": "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
}
_IMAGENET_ENCODER_CHECKPOINTS = {
    "resnet34": ("https://download.pytorch.org/models/resnet34-333f7ec4.pth", 70_000_000),
    "resnet50": ("https://download.pytorch.org/models/resnet50-19c8e357.pth", 90_000_000),
    "resnet101": ("https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", 150_000_000),
    "resnet152": ("https://download.pytorch.org/models/resnet152-b121ed2d.pth", 200_000_000),
    "mobilenet_v2": ("https://download.pytorch.org/models/mobilenet_v2-b0353104.pth", 10_000_000),
}


def _normalise_encoder_name(value: str | None) -> str:
    name = str(value or "unet_encoder").strip().lower()
    if name in _CUSTOM_ENCODERS:
        return "unet_encoder"
    return _SMP_ENCODER_ALIASES.get(name, name)


def _prepare_encoder_weights(encoder_name: str, encoder_weights: str | None) -> str | None:
    if not encoder_weights:
        return None
    ensure_pretrain_cache()
    if str(encoder_weights).lower() != "imagenet":
        return encoder_weights
    checkpoint = _IMAGENET_ENCODER_CHECKPOINTS.get(str(encoder_name).lower())
    if checkpoint is not None:
        url, min_bytes = checkpoint
        try:
            ensure_cached_checkpoint(url, min_bytes=min_bytes)
        except Exception as error:
            warnings.warn(
                f"Pretrained UNet encoder '{encoder_name}' is enabled but the checkpoint could not be loaded/downloaded "
                f"({error}). Falling back to random initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
    return encoder_weights


def _smp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    smp_kwargs = dict(kwargs)
    for key in ("feature_channels", "normalization", "dropout", "bilinear", "backbone", "encoder_name"):
        smp_kwargs.pop(key, None)
    return smp_kwargs


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm", dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv2d(in_channels, out_channels, normalization=normalization, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2d(out_channels + skip_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        backbone: str = "unet_encoder",
        encoder_name: str | None = None,
        encoder_weights: str | None = "imagenet",
        encoder_pretrained: bool | None = None,
        feature_channels: Sequence[int] = (32, 64, 128, 256, 512),
        normalization: str = "batchnorm",
        dropout: float = 0.0,
        bilinear: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        encoder_name = _normalise_encoder_name(encoder_name or backbone)
        if encoder_pretrained is not None:
            encoder_weights = "imagenet" if bool(encoder_pretrained) else None

        self.model_name = "unet"
        self.backbone_name = encoder_name

        if encoder_name != "unet_encoder":
            encoder_weights = _prepare_encoder_weights(encoder_name=encoder_name, encoder_weights=encoder_weights)
            try:
                import segmentation_models_pytorch as smp
            except ImportError as error:
                raise ImportError(
                    "UNet2D with CNN backbones requires segmentation-models-pytorch. "
                    "Install it with `pip install -r requirements.txt`."
                ) from error
            smp_kwargs = _smp_kwargs(kwargs)
            try:
                self.model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=num_classes,
                    **smp_kwargs,
                )
            except Exception as error:
                if not encoder_weights:
                    raise
                warnings.warn(
                    f"Unable to initialize UNet with pretrained '{encoder_name}' encoder ({error}). "
                    "Retrying from scratch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                encoder_weights = None
                self.model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=num_classes,
                    **smp_kwargs,
                )
            self.set_architecture_config(
                in_channels=in_channels,
                num_classes=num_classes,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                **smp_kwargs,
            )
            return

        channels = tuple(int(channel) for channel in feature_channels)
        if len(channels) != 5:
            raise ValueError("UNet2D expects exactly 5 encoder stages.")

        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_name=encoder_name,
            feature_channels=list(channels),
            normalization=normalization,
            dropout=dropout,
            bilinear=bilinear,
        )
        self.stem = DoubleConv2d(in_channels, channels[0], normalization=normalization, dropout=dropout)
        self.down1 = DownBlock(channels[0], channels[1], normalization=normalization, dropout=dropout)
        self.down2 = DownBlock(channels[1], channels[2], normalization=normalization, dropout=dropout)
        self.down3 = DownBlock(channels[2], channels[3], normalization=normalization, dropout=dropout)
        self.down4 = DownBlock(channels[3], channels[4], normalization=normalization, dropout=dropout)

        self.up1 = UpBlock(channels[4], channels[3], channels[3], normalization=normalization, bilinear=bilinear)
        self.up2 = UpBlock(channels[3], channels[2], channels[2], normalization=normalization, bilinear=bilinear)
        self.up3 = UpBlock(channels[2], channels[1], channels[1], normalization=normalization, bilinear=bilinear)
        self.up4 = UpBlock(channels[1], channels[0], channels[0], normalization=normalization, bilinear=bilinear)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        up1 = self.up1(x4, x3)
        up2 = self.up2(up1, x2)
        up3 = self.up3(up2, x1)
        decoder_features = self.up4(up3, x0)
        logits = self.head(decoder_features)
        features = {
            "bottleneck": x4,
            "encoder": {"stem": x0, "down1": x1, "down2": x2, "down3": x3, "down4": x4},
            "decoder": {"up1": up1, "up2": up2, "up3": up3, "up4": decoder_features, "final": decoder_features},
        }
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if hasattr(self, "model"):
            if return_features and all(hasattr(self.model, name) for name in ("encoder", "decoder", "segmentation_head")):
                encoder_features = list(self.model.encoder(x))
                decoder_features = self.model.decoder(*encoder_features)
                logits = self.model.segmentation_head(decoder_features)
                return logits, {
                    "encoder": encoder_features,
                    "decoder": {"final": decoder_features},
                }
            logits = self.model(x)
            output = self.build_output(logits, features={"smp_model": "Unet", "encoder_name": self.backbone_name})
            if return_features:
                return output.logits, {}
            return output.logits

        logits, features = self.forward_features(x)
        output = self.build_output(
            logits,
            features=features,
            aux={"feature_channels": list(self.head.weight.shape[1:2])},
        )
        if return_features:
            return output.logits, output.features
        return output.logits


UNet = UNet2D
