from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from utils.model_output import BaseSegmentationModel
from utils.pretrained_cache import ensure_cached_checkpoint, ensure_pretrain_cache


_BACKBONE_CHECKPOINTS = {
    "resnet50": ("https://download.pytorch.org/models/resnet50-19c8e357.pth", 90_000_000),
    "resnet101": ("https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", 150_000_000),
    "resnet152": ("https://download.pytorch.org/models/resnet152-b121ed2d.pth", 200_000_000),
    "mobilenetv2": ("https://download.pytorch.org/models/mobilenet_v2-b0353104.pth", 10_000_000),
}
_DEFAULT_BACKBONE_LABELS = {"", "default", "unet", "unet_encoder"}
_BACKBONE_ALIASES = {
    "mobi": "mobilenetv2",
    "mobile": "mobilenetv2",
    "mobilenet": "mobilenetv2",
    "mobilenet_v2": "mobilenetv2",
    "mobilenetv2": "mobilenetv2",
}


def _deeplab_root() -> Path:
    return Path(__file__).resolve().parent / "Deeplab" / "DeepLabV3Plus-Pytorch"


def _import_modeling():
    root = str(_deeplab_root())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for module_name in list(sys.modules):
        if module_name == "network" or module_name.startswith("network."):
            del sys.modules[module_name]
    return importlib.import_module("network.modeling")


def _smp_encoder_name(backbone: str) -> str:
    if backbone == "mobilenetv2":
        return "mobilenet_v2"
    return backbone


def _build_smp_deeplab(
    arch_type: str,
    backbone: str,
    in_channels: int,
    num_classes: int,
    output_stride: int,
    pretrained_backbone: bool,
    **kwargs: Any,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise ImportError(
            "DeepLab requires either baseline/2D-Model/Deeplab/DeepLabV3Plus-Pytorch "
            "or segmentation-models-pytorch. Install requirements.txt or copy the Deeplab repo folder."
        ) from error

    encoder_name = _smp_encoder_name(backbone)
    encoder_weights = "imagenet" if bool(pretrained_backbone) else None
    model_cls = smp.DeepLabV3Plus if arch_type == "deeplabv3plus" else smp.DeepLabV3
    filtered = dict(kwargs)
    for key in (
        "backbone",
        "pretrained_backbone",
        "encoder_pretrained",
        "encoder_weights",
        "feature_channels",
        "normalization",
        "dropout",
        "bilinear",
        "separable_conv",
    ):
        filtered.pop(key, None)
    filtered.setdefault("encoder_output_stride", int(output_stride))
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=int(in_channels),
        classes=int(num_classes),
        **filtered,
    )


def _prepare_pretrained(backbone: str, pretrained_backbone: bool) -> bool:
    ensure_pretrain_cache()
    if not pretrained_backbone:
        return False
    checkpoint = _BACKBONE_CHECKPOINTS.get(backbone)
    if checkpoint is not None:
        url, min_bytes = checkpoint
        try:
            ensure_cached_checkpoint(url, min_bytes=min_bytes)
        except Exception as error:
            warnings.warn(
                f"Pretrained DeepLab backbone '{backbone}' is enabled but the checkpoint could not be loaded/downloaded "
                f"({error}). Falling back to random initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
    return True


def _normalise_backbone(value: str | None) -> str:
    backbone = str(value or "resnet50").strip().lower()
    if backbone in _DEFAULT_BACKBONE_LABELS:
        return "resnet50"
    return _BACKBONE_ALIASES.get(backbone, backbone)


def _replace_first_conv(module: nn.Module, in_channels: int, pretrained: bool) -> bool:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3 and child.groups == 1:
            replacement = nn.Conv2d(
                in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            with torch.no_grad():
                if pretrained:
                    weight = child.weight.detach()
                    replacement.weight.copy_(weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
                    if child.bias is not None and replacement.bias is not None:
                        replacement.bias.copy_(child.bias.detach())
                else:
                    nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
                    if replacement.bias is not None:
                        nn.init.zeros_(replacement.bias)
            setattr(module, name, replacement)
            return True
        if _replace_first_conv(child, in_channels, pretrained):
            return True
    return False


def _replace_aspp_pooling_batchnorm(module: nn.Module) -> None:
    for child in module.modules():
        if child.__class__.__name__ == "ASPPPooling" and len(child) >= 3 and isinstance(child[2], nn.BatchNorm2d):
            channels = int(child[2].num_features)
            groups = min(32, channels)
            while groups > 1 and channels % groups != 0:
                groups -= 1
            child[2] = nn.GroupNorm(num_groups=groups, num_channels=channels)


class _DeepLabBase2D(BaseSegmentationModel):
    arch_type = "deeplabv3"
    model_name = "deeplab"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        backbone: str = "resnet50",
        output_stride: int = 16,
        pretrained_backbone: bool = True,
        encoder_pretrained: bool | None = None,
        separable_conv: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        backbone = _normalise_backbone(backbone)
        if backbone not in {"resnet50", "resnet101", "resnet152", "mobilenetv2"}:
            raise ValueError("DeepLab supports backbone: unet_encoder(default resnet50), resnet50, resnet101, resnet152, mobi/mobilenetv2.")
        if encoder_pretrained is not None:
            pretrained_backbone = bool(encoder_pretrained)

        pretrained_backbone = _prepare_pretrained(backbone, bool(pretrained_backbone))
        backend = "local_deeplab"
        try:
            modeling = _import_modeling()
        except ModuleNotFoundError as error:
            if error.name not in {"network", "network.modeling"}:
                raise
            backend = "segmentation_models_pytorch"
            try:
                self.model = _build_smp_deeplab(
                    arch_type=self.arch_type,
                    backbone=backbone,
                    in_channels=int(in_channels),
                    num_classes=int(num_classes),
                    output_stride=int(output_stride),
                    pretrained_backbone=bool(pretrained_backbone),
                    **kwargs,
                )
            except Exception as smp_error:
                if not pretrained_backbone:
                    raise
                warnings.warn(
                    f"Unable to initialize DeepLab with pretrained '{backbone}' SMP backbone ({smp_error}). "
                    "Retrying from scratch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pretrained_backbone = False
                self.model = _build_smp_deeplab(
                    arch_type=self.arch_type,
                    backbone=backbone,
                    in_channels=int(in_channels),
                    num_classes=int(num_classes),
                    output_stride=int(output_stride),
                    pretrained_backbone=False,
                    **kwargs,
                )
        else:
            builder_backbone = "mobilenet" if backbone == "mobilenetv2" else backbone
            builder_name = f"{self.arch_type}_{builder_backbone}"
            try:
                if hasattr(modeling, builder_name):
                    builder = getattr(modeling, builder_name)
                    self.model = builder(
                        num_classes=num_classes,
                        output_stride=int(output_stride),
                        pretrained_backbone=bool(pretrained_backbone),
                    )
                else:
                    self.model = modeling._load_model(
                        self.arch_type,
                        backbone,
                        num_classes,
                        output_stride=int(output_stride),
                        pretrained_backbone=bool(pretrained_backbone),
                    )
            except Exception as error:
                if not pretrained_backbone:
                    raise
                warnings.warn(
                    f"Unable to initialize DeepLab with pretrained '{backbone}' backbone ({error}). "
                    "Retrying from scratch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pretrained_backbone = False
                if hasattr(modeling, builder_name):
                    builder = getattr(modeling, builder_name)
                    self.model = builder(
                        num_classes=num_classes,
                        output_stride=int(output_stride),
                        pretrained_backbone=False,
                    )
                else:
                    self.model = modeling._load_model(
                        self.arch_type,
                        backbone,
                        num_classes,
                        output_stride=int(output_stride),
                        pretrained_backbone=False,
                    )
            if int(in_channels) != 3:
                if not _replace_first_conv(self.model, int(in_channels), bool(pretrained_backbone)):
                    raise RuntimeError("Unable to adapt DeepLab first convolution to requested in_channels.")
            _replace_aspp_pooling_batchnorm(self.model)

        if bool(separable_conv):
            try:
                converter = importlib.import_module("network._deeplab").convert_to_separable_conv
            except ModuleNotFoundError:
                converter = None
            if converter is not None:
                self.model = converter(self.model)

        self.backbone_name = backbone
        self.set_architecture_config(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            backbone=backbone,
            output_stride=int(output_stride),
            pretrained_backbone=bool(pretrained_backbone),
            separable_conv=bool(separable_conv),
            backend=backend,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if return_features and all(hasattr(self.model, name) for name in ("encoder", "decoder", "segmentation_head")):
            encoder_features = list(self.model.encoder(x))
            decoder_features = self.model.decoder(*encoder_features)
            logits = self.model.segmentation_head(decoder_features)
            return logits, {
                "encoder": encoder_features,
                "decoder": {"final": decoder_features},
            }
        logits = self.model(x)
        output = self.build_output(logits, features={"backbone": self.backbone_name, "arch": self.arch_type})
        if return_features:
            return output.logits, {}
        return output.logits


class DeepLab2D(_DeepLabBase2D):
    arch_type = "deeplabv3"
    model_name = "deeplab"


class DeepLabPlusPlus2D(_DeepLabBase2D):
    arch_type = "deeplabv3plus"
    model_name = "deeplab_plus_plus"


DeepLab = DeepLab2D
DeepLabPlusPlus = DeepLabPlusPlus2D
