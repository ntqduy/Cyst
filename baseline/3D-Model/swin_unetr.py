from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


logger = logging.getLogger(__name__)


class SwinUNETR3D(nn.Module):
    """Project-compatible wrapper around MONAI's 3D SwinUNETR."""

    model_name = "swin_unetr"
    backbone_name = "swin_transformer_3d"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: Sequence[int] = (256, 256, 64),
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str = "merging",
        use_v2: bool = False,
        pretrained_path: str | None = None,
        load_pretrained: bool = False,
    ) -> None:
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as error:
            raise ImportError(
                "Swin-UNETR requires MONAI. Install the project requirements with "
                "`pip install -r requirements.txt`."
            ) from error

        spatial_size = tuple(int(value) for value in image_size[:3])
        if len(spatial_size) != 3:
            raise ValueError(f"SwinUNETR3D image_size must be [H, W, D], got {spatial_size}.")
        invalid = [value for value in spatial_size if value % 32 != 0]
        if invalid:
            raise ValueError(
                "Every SwinUNETR3D image_size dimension must be divisible by 32; "
                f"got {spatial_size}."
            )

        requested: dict[str, Any] = {
            # MONAI < 1.5 requires img_size; newer releases infer it at runtime.
            "img_size": spatial_size,
            "in_channels": int(in_channels),
            "out_channels": int(num_classes),
            "patch_size": 2,
            "depths": tuple(int(value) for value in depths),
            "num_heads": tuple(int(value) for value in num_heads),
            "feature_size": int(feature_size),
            "norm_name": norm_name,
            "drop_rate": float(drop_rate),
            "attn_drop_rate": float(attn_drop_rate),
            "dropout_path_rate": float(dropout_path_rate),
            "normalize": bool(normalize),
            "use_checkpoint": bool(use_checkpoint),
            "spatial_dims": int(spatial_dims),
            "downsample": downsample,
            "use_v2": bool(use_v2),
        }
        parameters = inspect.signature(SwinUNETR).parameters
        self.model = SwinUNETR(**{key: value for key, value in requested.items() if key in parameters})

        if load_pretrained:
            if not pretrained_path:
                raise ValueError("model.args_3d.pretrained_path is required when load_pretrained=true.")
            checkpoint_path = Path(pretrained_path).expanduser()
            if not checkpoint_path.is_absolute():
                pretrain_root = Path(os.environ.get("CYST_PRETRAIN_DIR", Path(__file__).resolve().parents[2] / "pretrain"))
                checkpoint_path = pretrain_root / checkpoint_path
            checkpoint_path = checkpoint_path.resolve()
            if not checkpoint_path.is_file():
                logger.warning(
                    "Swin-UNETR pretrained checkpoint not found: %s. Training from scratch.",
                    checkpoint_path,
                )
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                    raise ValueError(
                        f"Expected an SSL checkpoint containing 'state_dict': {checkpoint_path}"
                    )
                if not hasattr(self.model, "load_from"):
                    raise RuntimeError("Installed MONAI SwinUNETR does not support SSL load_from().")
                try:
                    self.model.load_from(checkpoint)
                except (KeyError, RuntimeError) as error:
                    raise RuntimeError(
                        f"Swin-UNETR pretrained weights are incompatible with feature_size={feature_size}: "
                        f"{checkpoint_path}"
                    ) from error
                logger.info("Loaded Swin-UNETR pretrained checkpoint: %s", checkpoint_path)

    def forward(self, image):
        return self.model(image)
