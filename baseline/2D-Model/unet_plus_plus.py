from __future__ import annotations

import warnings
from typing import Any, Sequence

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
                f"Pretrained UNet++ encoder '{encoder_name}' is enabled but the checkpoint could not be loaded/downloaded "
                f"({error}). Falling back to random initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
    return encoder_weights


def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _upsample_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return _match_spatial(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), reference)


def _smp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    smp_kwargs = dict(kwargs)
    for key in ("feature_channels", "normalization", "dropout", "backbone", "encoder_name"):
        smp_kwargs.pop(key, None)
    return smp_kwargs


class UNetPlusPlus2D(BaseSegmentationModel):
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        encoder_name = _normalise_encoder_name(encoder_name or backbone)
        if encoder_pretrained is not None:
            encoder_weights = "imagenet" if bool(encoder_pretrained) else None
        self.model_name = "unet_plus_plus"
        self.backbone_name = encoder_name

        if encoder_name == "unet_encoder":
            channels = tuple(int(channel) for channel in feature_channels)
            if len(channels) != 5:
                raise ValueError("UNetPlusPlus2D expects exactly 5 encoder stages for unet_encoder.")
            self.set_architecture_config(
                in_channels=in_channels,
                num_classes=num_classes,
                encoder_name=encoder_name,
                feature_channels=list(channels),
                normalization=normalization,
                dropout=float(dropout),
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv0_0 = DoubleConv2d(in_channels, channels[0], normalization=normalization, dropout=float(dropout))
            self.conv1_0 = DoubleConv2d(channels[0], channels[1], normalization=normalization, dropout=float(dropout))
            self.conv2_0 = DoubleConv2d(channels[1], channels[2], normalization=normalization, dropout=float(dropout))
            self.conv3_0 = DoubleConv2d(channels[2], channels[3], normalization=normalization, dropout=float(dropout))
            self.conv4_0 = DoubleConv2d(channels[3], channels[4], normalization=normalization, dropout=float(dropout))

            self.conv0_1 = DoubleConv2d(channels[0] + channels[1], channels[0], normalization=normalization)
            self.conv1_1 = DoubleConv2d(channels[1] + channels[2], channels[1], normalization=normalization)
            self.conv2_1 = DoubleConv2d(channels[2] + channels[3], channels[2], normalization=normalization)
            self.conv3_1 = DoubleConv2d(channels[3] + channels[4], channels[3], normalization=normalization)

            self.conv0_2 = DoubleConv2d(channels[0] * 2 + channels[1], channels[0], normalization=normalization)
            self.conv1_2 = DoubleConv2d(channels[1] * 2 + channels[2], channels[1], normalization=normalization)
            self.conv2_2 = DoubleConv2d(channels[2] * 2 + channels[3], channels[2], normalization=normalization)

            self.conv0_3 = DoubleConv2d(channels[0] * 3 + channels[1], channels[0], normalization=normalization)
            self.conv1_3 = DoubleConv2d(channels[1] * 3 + channels[2], channels[1], normalization=normalization)

            self.conv0_4 = DoubleConv2d(channels[0] * 4 + channels[1], channels[0], normalization=normalization)
            self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
            return

        encoder_weights = _prepare_encoder_weights(encoder_name=encoder_name, encoder_weights=encoder_weights)
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "UNetPlusPlus2D requires segmentation-models-pytorch. "
                "Install it with `pip install segmentation-models-pytorch` or `pip install -r requirements.txt`."
            ) from error
        smp_kwargs = _smp_kwargs(kwargs)
        try:
            self.model = smp.UnetPlusPlus(
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
                f"Unable to initialize UNet++ with pretrained '{encoder_name}' encoder ({error}). "
                "Retrying from scratch.",
                RuntimeWarning,
                stacklevel=2,
            )
            encoder_weights = None
            self.model = smp.UnetPlusPlus(
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

    def forward_features(self, x: torch.Tensor):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, _upsample_like(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, _upsample_like(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, _upsample_like(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, _upsample_like(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, _upsample_like(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, _upsample_like(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, _upsample_like(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, _upsample_like(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, _upsample_like(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, _upsample_like(x1_3, x0_0)], dim=1))
        logits = self.head(x0_4)
        features = {
            "encoder": {"x0_0": x0_0, "x1_0": x1_0, "x2_0": x2_0, "x3_0": x3_0, "x4_0": x4_0},
            "decoder": {"x0_1": x0_1, "x0_2": x0_2, "x0_3": x0_3, "x0_4": x0_4},
        }
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if not hasattr(self, "model"):
            logits, features = self.forward_features(x)
            output = self.build_output(logits, features=features)
            if return_features:
                return output.logits, output.features
            return output.logits

        if return_features and all(hasattr(self.model, name) for name in ("encoder", "decoder", "segmentation_head")):
            encoder_features = list(self.model.encoder(x))
            decoder_features = self.model.decoder(*encoder_features)
            logits = self.model.segmentation_head(decoder_features)
            return logits, {
                "encoder": encoder_features,
                "decoder": {"final": decoder_features},
            }
        logits = self.model(x)
        output = self.build_output(
            logits,
            features={"smp_model": "UnetPlusPlus", "encoder_name": self.backbone_name},
        )
        if return_features:
            return output.logits, {}
        return output.logits


UNetPlusPlus = UNetPlusPlus2D
