from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from utils.model_output import BaseSegmentationModel

try:
    from .adapt_slice import SliceSelector
except ImportError:  # pragma: no cover - supports direct file execution/import.
    _selector_path = Path(__file__).resolve().with_name("adapt_slice.py")
    _selector_spec = importlib.util.spec_from_file_location("_cyst_unet3plus_slice_selector", _selector_path)
    if _selector_spec is None or _selector_spec.loader is None:
        raise
    _selector_module = importlib.util.module_from_spec(_selector_spec)
    _selector_spec.loader.exec_module(_selector_module)
    SliceSelector = getattr(_selector_module, "SliceSelector")


def _load_original_module():
    models_dir = Path(__file__).resolve().parents[1] / "2D-Model" / "Unet_3_plus" / "UNet-Version" / "models"
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


def _main_logits(output: Any, num_classes: int) -> torch.Tensor:
    outputs = output if isinstance(output, (tuple, list)) else (output,)
    for item in outputs:
        if isinstance(item, torch.Tensor):
            if item.ndim >= 4 and item.shape[1] == int(num_classes):
                return item
    if outputs and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    raise TypeError(f"Cannot extract logits from UNet3Plus output of type {type(output)!r}")


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


class UNet3PlusSlice(BaseSegmentationModel):
    """Run UNet 3+ on proposal-selected 2D slices and lift logits to a volume."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feature_scale: int = 4,
        is_deconv: bool = True,
        is_batchnorm: bool = True,
        deep_supervision: bool = False,
        cgm: bool = False,
        internal_num_classes: int | None = None,
        slice_selection: dict[str, Any] | None = None,
        volume_fusion: str = "linear_interpolate",
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.slice_selection_cfg = dict(slice_selection or {})
        self.slice_selector = SliceSelector.from_config(self.slice_selection_cfg)
        self.volume_fusion = str(volume_fusion or "linear_interpolate").lower()
        if self.volume_fusion not in {"linear_interpolate", "nearest"}:
            raise ValueError("volume_fusion must be 'linear_interpolate' or 'nearest'.")

        self.slice_model = UNet3Plus2D(
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            feature_scale=int(feature_scale),
            is_deconv=bool(is_deconv),
            is_batchnorm=bool(is_batchnorm),
            deep_supervision=bool(deep_supervision),
            cgm=bool(cgm),
            internal_num_classes=internal_num_classes,
        )
        self.model_name = "unet_3_plus_slice"
        self.backbone_name = "unet3plus_slice_encoder"
        self.deep_supervision = bool(deep_supervision)
        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            feature_scale=int(feature_scale),
            is_deconv=bool(is_deconv),
            is_batchnorm=bool(is_batchnorm),
            deep_supervision=self.deep_supervision,
            cgm=bool(cgm),
            slice_selection=self.slice_selection_cfg,
            volume_fusion=self.volume_fusion,
            backbone=self.backbone_name,
        )

    def forward(self, volume: torch.Tensor, return_features: bool = False):
        if volume.ndim == 4:
            output = self.slice_model(volume, return_features=return_features)
            return output
        if volume.ndim != 5:
            raise ValueError(f"UNet3PlusSlice expects [B,C,D,H,W] or [B,C,H,W], got {tuple(volume.shape)}")

        batch_size, _, depth, height, width = volume.shape
        selected_slices, slice_indices = self.slice_selector(volume)
        num_selected = int(selected_slices.shape[1])
        flat_slices = selected_slices.reshape(batch_size * num_selected, *selected_slices.shape[2:])

        slice_output = self.slice_model(flat_slices)
        flat_logits = _main_logits(slice_output, self.num_classes)
        slice_logits = flat_logits.reshape(batch_size, num_selected, flat_logits.shape[1], flat_logits.shape[-2], flat_logits.shape[-1])
        volume_logits = self._lift_slice_logits(slice_logits, slice_indices, depth)
        if tuple(volume_logits.shape[-2:]) != (height, width):
            volume_logits = torch.nn.functional.interpolate(volume_logits, size=(depth, height, width), mode="trilinear", align_corners=False)

        if return_features:
            return self.build_output(
                volume_logits,
                features={
                    "slice_indices": slice_indices,
                    "selected_slices": selected_slices,
                    "slice_logits": slice_logits,
                },
            )
        return volume_logits

    def _lift_slice_logits(self, slice_logits: torch.Tensor, slice_indices: torch.Tensor, depth: int) -> torch.Tensor:
        if self.volume_fusion == "nearest":
            return self._nearest_lift(slice_logits, slice_indices, depth)
        return self._linear_lift(slice_logits, slice_indices, depth)

    @staticmethod
    def _unique_sorted(indices: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_indices, order = torch.sort(indices.to(dtype=torch.long))
        sorted_values = values.index_select(0, order)
        if sorted_indices.numel() <= 1:
            return sorted_indices, sorted_values
        keep = torch.ones_like(sorted_indices, dtype=torch.bool)
        keep[1:] = sorted_indices[1:] != sorted_indices[:-1]
        return sorted_indices[keep], sorted_values[keep]

    def _linear_lift(self, slice_logits: torch.Tensor, slice_indices: torch.Tensor, depth: int) -> torch.Tensor:
        batch_outputs = []
        positions = torch.arange(int(depth), device=slice_logits.device, dtype=torch.long)
        for batch_index in range(int(slice_logits.shape[0])):
            indices, values = self._unique_sorted(slice_indices[batch_index], slice_logits[batch_index])
            if int(indices.numel()) == 1:
                batch_outputs.append(values[0].unsqueeze(1).expand(-1, int(depth), -1, -1).contiguous())
                continue

            raw_right = torch.searchsorted(indices, positions, right=False)
            right = raw_right.clamp(0, int(indices.numel()) - 1)
            left = (right - 1).clamp(0, int(indices.numel()) - 1)
            left = torch.where(raw_right == 0, right, left)
            left = torch.where(raw_right >= int(indices.numel()), right, left)

            left_positions = indices.index_select(0, left).float()
            right_positions = indices.index_select(0, right).float()
            denom = (right_positions - left_positions).clamp_min(1.0)
            weight = ((positions.float() - left_positions) / denom).clamp(0.0, 1.0)

            left_values = values.index_select(0, left)
            right_values = values.index_select(0, right)
            lifted = left_values * (1.0 - weight[:, None, None, None]) + right_values * weight[:, None, None, None]
            batch_outputs.append(lifted.permute(1, 0, 2, 3).contiguous())
        return torch.stack(batch_outputs, dim=0)

    def _nearest_lift(self, slice_logits: torch.Tensor, slice_indices: torch.Tensor, depth: int) -> torch.Tensor:
        batch_outputs = []
        positions = torch.arange(int(depth), device=slice_logits.device, dtype=torch.long)
        for batch_index in range(int(slice_logits.shape[0])):
            indices, values = self._unique_sorted(slice_indices[batch_index], slice_logits[batch_index])
            distances = (positions[:, None] - indices[None, :]).abs()
            nearest = torch.argmin(distances, dim=1)
            lifted = values.index_select(0, nearest)
            batch_outputs.append(lifted.permute(1, 0, 2, 3).contiguous())
        return torch.stack(batch_outputs, dim=0)


UNet_3Plus2D = UNet3Plus2D
UNet3plus2D = UNet3Plus2D
Unet3Plus2D = UNet3Plus2D
UNet3PlusSliceVolume = UNet3PlusSlice
UNet_3PlusSlice = UNet3PlusSlice
Unet3PlusSlice = UNet3PlusSlice
