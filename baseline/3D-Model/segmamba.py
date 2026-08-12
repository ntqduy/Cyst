from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from torch import nn


class SegMamba3D(nn.Module):
    """Project adapter for the original 3D SegMamba implementation."""

    model_name = "segmamba"
    backbone_name = "mamba_3d_encoder"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: Sequence[int] = (128, 128, 128),
        depths: Sequence[int] = (2, 2, 2, 2),
        feature_channels: Sequence[int] = (48, 96, 192, 384),
        hidden_size: int = 768,
        drop_path_rate: float = 0.0,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
    ) -> None:
        super().__init__()
        spatial_size = tuple(int(value) for value in image_size[:3])
        if len(spatial_size) != 3:
            raise ValueError(f"SegMamba image_size must be [H, W, D], got {spatial_size}.")
        if spatial_size != (128, 128, 128):
            raise ValueError(
                "The original SegMamba encoder hard-codes nslices=[64,32,16,8] and "
                f"expects image_size [128,128,128], got {list(spatial_size)}."
            )

        segmamba_root = Path(__file__).resolve().parents[1] / "SegMamba"
        bundled_mamba = segmamba_root / "mamba"
        if str(bundled_mamba) not in sys.path:
            sys.path.insert(0, str(bundled_mamba))

        try:
            from SegMamba.model_segmamba.segmamba import SegMamba
        except (ImportError, ModuleNotFoundError) as error:
            raise ImportError(
                "SegMamba requires its bundled CUDA extensions. Install causal-conv1d and "
                "mamba_ssm from baseline/SegMamba before running this experiment."
            ) from error

        self.model = SegMamba(
            in_chans=int(in_channels),
            out_chans=int(num_classes),
            depths=[int(value) for value in depths],
            feat_size=[int(value) for value in feature_channels],
            drop_path_rate=float(drop_path_rate),
            hidden_size=int(hidden_size),
            norm_name=norm_name,
            conv_block=bool(conv_block),
            res_block=bool(res_block),
            spatial_dims=3,
        )

    def forward(self, image):
        return self.model(image)
