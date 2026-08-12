from __future__ import annotations

import inspect
from typing import Any, Sequence

import torch
from torch import nn

from utils.model_output import BaseSegmentationModel


def _normalise_size(value: Sequence[int] | int | None) -> tuple[int, int]:
    if value is None:
        return (256, 256)
    if isinstance(value, int):
        return (int(value), int(value))
    values = [int(item) for item in value]
    if len(values) < 2:
        raise ValueError("SwinUNet2D image_size must contain [height, width].")
    return (values[0], values[1])


def _normalise_norm_name(value: str) -> str:
    key = str(value or "instance").lower()
    if key in {"instancenorm", "instance_norm", "instance"}:
        return "instance"
    if key in {"batchnorm", "batch_norm", "batch"}:
        return "batch"
    return key


def _swinunetr_kwargs(cls, requested: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(cls).parameters
    return {key: value for key, value in requested.items() if key in parameters}


class SwinUNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: Sequence[int] | int | None = None,
        img_size: Sequence[int] | int | None = None,
        feature_size: int = 24,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: str = "instance",
        normalization: str | None = None,
        drop_rate: float = 0.0,
        dropout: float | None = None,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        downsample: str = "merging",
        use_v2: bool = False,
        backbone: str = "swin_transformer",
        **_: Any,
    ) -> None:
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as error:
            raise ImportError("SwinUNet2D requires MONAI. Install requirements.txt first.") from error

        resolved_size = _normalise_size(img_size or image_size)
        if normalization is not None:
            norm_name = _normalise_norm_name(normalization)
        if dropout is not None:
            drop_rate = float(dropout)

        requested = {
            "img_size": resolved_size,
            "in_channels": int(in_channels),
            "out_channels": int(num_classes),
            "feature_size": int(feature_size),
            "depths": tuple(int(item) for item in depths),
            "num_heads": tuple(int(item) for item in num_heads),
            "norm_name": norm_name,
            "drop_rate": float(drop_rate),
            "attn_drop_rate": float(attn_drop_rate),
            "dropout_path_rate": float(dropout_path_rate),
            "normalize": bool(normalize),
            "use_checkpoint": bool(use_checkpoint),
            "spatial_dims": 2,
            "downsample": downsample,
            "use_v2": bool(use_v2),
        }
        self.model = SwinUNETR(**_swinunetr_kwargs(SwinUNETR, requested))
        self.model_name = "swin_unet"
        self.backbone_name = str(backbone or "swin_transformer")
        self.set_architecture_config(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            image_size=list(resolved_size),
            feature_size=int(feature_size),
            depths=list(depths),
            num_heads=list(num_heads),
            norm_name=norm_name,
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            dropout_path_rate=float(dropout_path_rate),
            normalize=bool(normalize),
            use_checkpoint=bool(use_checkpoint),
            downsample=downsample,
            use_v2=bool(use_v2),
            backbone=self.backbone_name,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = self.model(x)
        output = self.build_output(logits, features={"backend": "monai.SwinUNETR", "spatial_dims": 2})
        if return_features:
            return output.logits, {}
        return output.logits


SwinUnet2D = SwinUNet2D
SwinUNet = SwinUNet2D
