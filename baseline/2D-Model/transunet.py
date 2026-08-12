from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from TransUNet.networks.vit_seg_modeling import CONFIGS, VisionTransformer


class TransUNet2D(nn.Module):
    """Adapter for the official TransUNet implementation used by this project."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: Sequence[int] = (256, 256),
        variant: str = "R50-ViT-B_16",
        n_skip: int = 3,
        pretrained_path: str | None = None,
        load_pretrained: bool = True,
        vis: bool = False,
    ) -> None:
        super().__init__()
        if int(in_channels) not in {1, 3}:
            raise ValueError(
                "TransUNet expects 1 or 3 input channels. Set slice_2d.num_slices to 1 or 3."
            )
        if variant not in CONFIGS:
            raise ValueError(f"Unknown TransUNet variant '{variant}'. Available: {', '.join(CONFIGS)}")

        height, width = (int(image_size[0]), int(image_size[1]))
        if height != width:
            raise ValueError(f"TransUNet requires square inputs, got {height}x{width}.")
        if height % 16 != 0:
            raise ValueError(f"TransUNet input size must be divisible by 16, got {height}.")

        config = copy.deepcopy(CONFIGS[variant])
        config.n_classes = int(num_classes)
        config.n_skip = int(n_skip)
        if config.patches.get("grid") is not None:
            config.patches.grid = (height // 16, width // 16)

        self.model = VisionTransformer(
            config=config,
            img_size=height,
            num_classes=int(num_classes),
            vis=bool(vis),
        )
        self.model_name = "transunet"

        if load_pretrained:
            if not pretrained_path:
                raise ValueError("model.args.pretrained_path is required when load_pretrained=true.")
            checkpoint = Path(pretrained_path).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parents[2] / checkpoint
            checkpoint = checkpoint.resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"TransUNet pretrained checkpoint not found: {checkpoint}. "
                    "Expected an ImageNet .npz checkpoint."
                )
            with np.load(checkpoint) as weights:
                self.model.load_from(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
