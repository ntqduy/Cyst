from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from utils.model_output import BaseSegmentationModel


def _load_original_module():
    models_dir = Path(__file__).resolve().parent / "Unet_3_plus" / "UNet-Version" / "models"
    module_path = models_dir / "UNet_3Plus.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing UNet 3+ source file: {module_path}")

    models_dir_text = str(models_dir)
    if models_dir_text not in sys.path:
        sys.path.insert(0, models_dir_text)

    module_name = "_cyst_unet_3_plus_original"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import UNet 3+ from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.np = np
    return module


def _probabilities_to_logits(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.logit(value.clamp(float(eps), 1.0 - float(eps)))


def _binary_probabilities_to_two_class_logits(value: torch.Tensor) -> torch.Tensor:
    if value.shape[1] >= 2:
        return _probabilities_to_logits(value[:, :2])
    foreground_probability = value[:, :1]
    foreground_logits = _probabilities_to_logits(foreground_probability)
    background_logits = torch.zeros_like(foreground_logits)
    return torch.cat((background_logits, foreground_logits), dim=1)


def _unet3plus_output_to_logits(value: torch.Tensor, num_classes: int) -> torch.Tensor:
    if int(num_classes) == 2:
        return _binary_probabilities_to_two_class_logits(value)
    return _probabilities_to_logits(value)


class UNet3Plus2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feature_scale: int = 4,
        is_deconv: bool = True,
        is_batchnorm: bool = True,
        deep_supervision: bool = False,
        cgm: bool = False,
        backbone: str = "unet3plus_encoder",
        internal_num_classes: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        original = _load_original_module()
        if cgm:
            cls = getattr(original, "UNet_3Plus_DeepSup_CGM")
            deep_supervision = True
        elif deep_supervision:
            cls = getattr(original, "UNet_3Plus_DeepSup")
        else:
            cls = getattr(original, "UNet_3Plus")

        self.num_classes = int(num_classes)
        self.internal_num_classes = int(internal_num_classes) if internal_num_classes is not None else self.num_classes
        self.model = cls(
            in_channels=int(in_channels),
            n_classes=self.internal_num_classes,
            feature_scale=int(feature_scale),
            is_deconv=bool(is_deconv),
            is_batchnorm=bool(is_batchnorm),
        )
        self.model_name = "unet3plus_hybrid_cgm" if cgm else ("unet_3_plus_deepsup" if deep_supervision else "unet_3_plus")
        self.backbone_name = str(backbone or "unet3plus_encoder")
        self.deep_supervision = bool(deep_supervision)
        self.cgm = bool(cgm)
        self.set_architecture_config(
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            internal_num_classes=self.internal_num_classes,
            feature_scale=int(feature_scale),
            is_deconv=bool(is_deconv),
            is_batchnorm=bool(is_batchnorm),
            deep_supervision=self.deep_supervision,
            cgm=self.cgm,
            backbone=self.backbone_name,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        output = self.model(x)
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        logits = [_unet3plus_output_to_logits(item, self.num_classes) for item in outputs]
        main_logits = logits[0]
        result = self.build_output(
            main_logits,
            features={
                "deep_outputs": logits[1:],
                "deep_supervision": self.deep_supervision,
                "cgm": self.cgm,
            },
        )
        if return_features:
            return result.logits, {"deep_outputs": logits[1:]}
        if self.deep_supervision and len(logits) > 1:
            return tuple(logits)
        return result.logits


UNet_3Plus2D = UNet3Plus2D
UNet3plus2D = UNet3Plus2D
Unet3Plus2D = UNet3Plus2D
