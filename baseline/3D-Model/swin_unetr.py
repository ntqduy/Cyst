from __future__ import annotations

import inspect
from typing import Any, Sequence

from torch import nn


class SwinUNETR3D(nn.Module):
    """Project-compatible wrapper around MONAI's 3D SwinUNETR."""

    model_name = "swin_unetr"
    backbone_name = "swin_transformer_3d"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: Sequence[int] = (256, 256, 64),
        feature_size: int = 24,
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

    def forward(self, image):
        return self.model(image)
