from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from utils.model_output import BaseSegmentationModel
except Exception:  # pragma: no cover - used only when importing this file standalone.

    class BaseSegmentationModel(nn.Module):
        def set_architecture_config(self, **kwargs) -> None:
            self.architecture_config = dict(kwargs)

        def build_output(self, logits, features=None, aux=None):
            return {"logits": logits, "features": features, "aux": aux or {}}


def _import_local_module(module_name: str):
    if __package__:
        return importlib.import_module(f"{__package__}.{module_name}")

    module_path = Path(__file__).resolve().with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(f"_proposal_hybrid_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_encoder_2d_module = _import_local_module("2D_encoder")
_encoder_3d_module = _import_local_module("3D_encoder")
_decoder_module = _import_local_module("decoder")
_selector_module = _import_local_module("selection_slice")

UNet3Plus2DEncoder = getattr(_encoder_2d_module, "UNet3Plus2DEncoder")
UNet3DEncoder = getattr(_encoder_3d_module, "UNet3DEncoder")
UNet3Plus3DDecoder = getattr(_decoder_module, "UNet3Plus3DDecoder")
SliceSelector = getattr(_selector_module, "SliceSelector")


def _channels(channels: list[int] | tuple[int, ...] | None, base_channels: int) -> list[int]:
    if channels is None:
        channels = [int(base_channels) * (2**index) for index in range(5)]
    result = [int(item) for item in channels]
    if len(result) != 5:
        raise ValueError(f"Expected 5 channel stages, got {result}")
    return result


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)!r}.")
    for key in ("model_state", "state_dict", "model_state_dict", "network_weights"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu")
    return _checkpoint_state_dict(checkpoint)


def _normalization_2d(channels: int, name: str) -> nn.Module:
    normalized = str(name or "batchnorm").lower()
    if normalized in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if normalized in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm2d(channels, affine=True)
    if normalized in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 2D normalization: {name}")


def _normalization_3d(channels: int, name: str) -> nn.Module:
    normalized = str(name or "batchnorm").lower()
    if normalized in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm3d(channels)
    if normalized in {"instance", "instancenorm", "in"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if normalized in {"group", "groupnorm", "gn"}:
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D normalization: {name}")


class ConvNormAct2D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, normalization: str = "batchnorm") -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            _normalization_2d(out_channels, normalization),
            nn.ReLU(inplace=True),
        )


class ConvNormAct3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, normalization: str = "batchnorm") -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            _normalization_3d(out_channels, normalization),
            nn.ReLU(inplace=True),
        )


class FusionBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct2D(in_channels, out_channels, normalization=normalization),
            ConvNormAct2D(out_channels, out_channels, normalization=normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3Plus2DDecoder(nn.Module):
    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        num_classes: int,
        fusion_channels: int | None = None,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = [int(item) for item in channels]
        if len(self.channels) != 5:
            raise ValueError(f"UNet3Plus2DDecoder expects 5 channel stages, got {self.channels}")
        self.fusion_channels = int(fusion_channels or self.channels[0])
        self.projections = nn.ModuleDict()
        self.decoder_projections = nn.ModuleDict()
        self.fusion_blocks = nn.ModuleDict()

        for target_index in range(4):
            key = str(target_index)
            history = UNet3Plus3DDecoder.history_indices(target_index)
            self.projections[key] = nn.ModuleList(
                [
                    ConvNormAct2D(source, self.fusion_channels, kernel_size=1, normalization=normalization)
                    for source in self.channels
                ]
            )
            self.decoder_projections[key] = nn.ModuleList(
                [
                    ConvNormAct2D(self.channels[source], self.fusion_channels, kernel_size=1, normalization=normalization)
                    for source in history
                ]
            )
            self.fusion_blocks[key] = FusionBlock2D(
                self.fusion_channels * (len(self.channels) + len(history)),
                self.channels[target_index],
                normalization=normalization,
            )
        self.segmentation_head = nn.Conv2d(self.channels[0], int(num_classes), kernel_size=1)

    @staticmethod
    def _resize(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tuple(feature.shape[-2:]) == tuple(size):
            return feature
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, skips: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(skips) != 5:
            raise ValueError(f"UNet3Plus2DDecoder expects 5 skip tensors, got {len(skips)}")
        decoded: dict[int, torch.Tensor] = {}
        for target_index in (3, 2, 1, 0):
            target_size = tuple(int(item) for item in skips[target_index].shape[-2:])
            key = str(target_index)
            projected = [projection(self._resize(source, target_size)) for projection, source in zip(self.projections[key], skips)]
            for projection, source_index in zip(self.decoder_projections[key], UNet3Plus3DDecoder.history_indices(target_index)):
                projected.append(projection(self._resize(decoded[source_index], target_size)))
            decoded[target_index] = self.fusion_blocks[key](torch.cat(projected, dim=1))
        return self.segmentation_head(decoded[0])


class SlicePositionEncoder(nn.Module):
    def __init__(
        self,
        channels: list[int] | tuple[int, ...],
        embedding_dim: int = 32,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        self.max_positions = int(max_positions)
        self.embedding = nn.Embedding(self.max_positions, int(embedding_dim))
        self.projections = nn.ModuleList([nn.Linear(int(embedding_dim), int(channel)) for channel in channels])

    def forward_2d(self, features: list[torch.Tensor], slice_indices: torch.Tensor | None) -> list[torch.Tensor]:
        if slice_indices is None:
            batch_size = features[0].shape[0]
            slice_indices = torch.zeros(batch_size, device=features[0].device, dtype=torch.long)
        if slice_indices.ndim == 2:
            slice_indices = slice_indices[:, 0]
        slice_indices = slice_indices.to(device=features[0].device, dtype=torch.long).clamp_(0, self.max_positions - 1)
        encoded = self.embedding(slice_indices)
        output = []
        for feature, projection in zip(features, self.projections):
            bias = projection(encoded).view(feature.shape[0], feature.shape[1], 1, 1)
            output.append(feature + bias)
        return output

    def forward_selected(self, features: list[torch.Tensor], slice_indices: torch.Tensor) -> list[torch.Tensor]:
        slice_indices = slice_indices.to(device=features[0].device, dtype=torch.long).clamp_(0, self.max_positions - 1)
        encoded = self.embedding(slice_indices)
        output = []
        for feature, projection in zip(features, self.projections):
            bias = projection(encoded).permute(0, 1, 2).unsqueeze(-1).unsqueeze(-1)
            output.append(feature + bias)
        return output


def _main_logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _normalise_encoder_fusion_mode(mode: str | None) -> str:
    value = str(mode or "concat").lower().replace("-", "_")
    if value in {"concat", "cat", "concat_conv", "concat_1x1", "concat_conv1x1"}:
        return "concat"
    if value in {"add", "sum", "add_conv", "add_1x1", "add_conv1x1"}:
        return "add"
    raise ValueError("encoder_fusion_mode must be concat or add.")


class FullUNet3Plus2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        use_position_encoder: bool = False,
        position_embedding_dim: int = 32,
        max_position_embeddings: int = 512,
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = _channels(encoder_channels or channels, base_channels)
        self.num_classes = int(num_classes)
        self.use_position_encoder = bool(use_position_encoder)
        self.expects_slice_indices = self.use_position_encoder
        self.encoder_2d = UNet3Plus2DEncoder(in_channels=int(in_channels), channels=self.channels, normalization=normalization)
        self.position_encoder = (
            SlicePositionEncoder(self.channels, embedding_dim=position_embedding_dim, max_positions=max_position_embeddings)
            if self.use_position_encoder
            else None
        )
        self.decoder_2d = UNet3Plus2DDecoder(self.channels, num_classes=self.num_classes, deep_supervision=deep_supervision, normalization=normalization)
        self.model_name = "FullUNet3Plus2D"
        self.backbone_name = "unet3plus2d_encoder"
        self.deep_supervision = bool(deep_supervision)
        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            encoder_channels=self.channels,
            deep_supervision=self.deep_supervision,
            use_position_encoder=self.use_position_encoder,
        )

    def forward(self, x: torch.Tensor, slice_indices: torch.Tensor | None = None, return_features: bool = False):
        features = self.encoder_2d(x)
        if self.position_encoder is not None:
            features = self.position_encoder.forward_2d(features, slice_indices)
        output = self.decoder_2d(features)
        logits = _main_logits(output)
        if return_features:
            return self.build_output(
                logits,
                features={"encoder": features, "deep_outputs": list(output[1:]) if isinstance(output, tuple) else []},
            )
        return output if isinstance(output, tuple) and self.deep_supervision else logits


class FullUnet3D_3_plus(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        **_: Any,
    ) -> None:
        super().__init__()
        self.channels = _channels(encoder_channels or channels, base_channels)
        self.num_classes = int(num_classes)
        self.encoder_3d = UNet3DEncoder(in_channels=int(in_channels), channels=self.channels, normalization=normalization)
        self.decoder_3d = UNet3Plus3DDecoder(
            channels=self.channels,
            num_classes=self.num_classes,
            deep_supervision=deep_supervision,
            normalization=normalization,
        )
        self.model_name = "FullUnet3D_3_plus"
        self.backbone_name = "unet3d_encoder"
        self.deep_supervision = bool(deep_supervision)
        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            encoder_channels=self.channels,
            deep_supervision=self.deep_supervision,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.encoder_3d(x)
        output = self.decoder_3d(features)
        logits = _main_logits(output)
        if return_features:
            return self.build_output(
                logits,
                features={"encoder": features, "deep_outputs": list(output[1:]) if isinstance(output, tuple) else []},
            )
        return output if isinstance(output, tuple) and self.deep_supervision else logits


FullUNet3D = FullUnet3D_3_plus


class Hybrid3D2DUNet3Plus(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        encoder_2d_channels: list[int] | tuple[int, ...] | None = None,
        encoder_3d_channels: list[int] | tuple[int, ...] | None = None,
        channels: list[int] | tuple[int, ...] | None = None,
        base_channels: int = 32,
        slice_selection: Mapping[str, Any] | None = None,
        deep_supervision: bool = False,
        normalization: str = "batchnorm",
        use_position_encoder: bool = False,
        position_embedding_dim: int = 32,
        max_position_embeddings: int = 512,
        encoder_fusion_mode: str = "concat",
        freeze_2d_encoder: bool = False,
        freeze_3d_encoder: bool = False,
        freeze_3d_decoder: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder_2d_channels = _channels(encoder_2d_channels or channels, base_channels)
        self.encoder_3d_channels = _channels(encoder_3d_channels or channels, base_channels)
        if len(self.encoder_2d_channels) != len(self.encoder_3d_channels):
            raise ValueError("2D and 3D encoders must expose the same number of stages.")

        self.num_classes = int(num_classes)
        self.use_position_encoder = bool(use_position_encoder)
        self.encoder_fusion_mode = _normalise_encoder_fusion_mode(encoder_fusion_mode)
        self.slice_selection_cfg = dict(slice_selection or {})
        self.slice_selector = SliceSelector.from_config(self.slice_selection_cfg)
        self.encoder_2d = UNet3Plus2DEncoder(in_channels=int(in_channels), channels=self.encoder_2d_channels, normalization=normalization)
        self.encoder_3d = UNet3DEncoder(in_channels=int(in_channels), channels=self.encoder_3d_channels, normalization=normalization)
        self.position_encoder = (
            SlicePositionEncoder(self.encoder_2d_channels, embedding_dim=position_embedding_dim, max_positions=max_position_embeddings)
            if self.use_position_encoder
            else None
        )
        self.fusion_projections = nn.ModuleList(
            [
                nn.Conv3d(ch2d, ch3d, kernel_size=1, bias=False)
                if self.encoder_fusion_mode == "add" and ch2d != ch3d
                else nn.Identity()
                for ch2d, ch3d in zip(self.encoder_2d_channels, self.encoder_3d_channels)
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(ch3d + ch2d if self.encoder_fusion_mode == "concat" else ch3d, ch3d, kernel_size=1, bias=False),
                    _normalization_3d(ch3d, normalization),
                    nn.ReLU(inplace=True),
                )
                for ch2d, ch3d in zip(self.encoder_2d_channels, self.encoder_3d_channels)
            ]
        )
        self.decoder_3d = UNet3Plus3DDecoder(
            channels=self.encoder_3d_channels,
            num_classes=self.num_classes,
            deep_supervision=deep_supervision,
            normalization=normalization,
        )
        self.model_name = "Hybrid3D2DUNet3PlusSliceInject"
        self.backbone_name = "hybrid_unet3plus_2d3d_slice_inject_encoder"
        self.deep_supervision = bool(deep_supervision)
        self.pretrain_loading_report: dict[str, Any] | None = None

        if freeze_2d_encoder:
            self._freeze(self.encoder_2d)
        if freeze_3d_encoder:
            self._freeze(self.encoder_3d)
        if freeze_3d_decoder:
            self._freeze(self.decoder_3d)

        self.set_architecture_config(
            model_name=self.model_name,
            in_channels=int(in_channels),
            num_classes=self.num_classes,
            encoder_2d_channels=self.encoder_2d_channels,
            encoder_3d_channels=self.encoder_3d_channels,
            slice_selection=self.slice_selection_cfg,
            deep_supervision=self.deep_supervision,
            use_position_encoder=self.use_position_encoder,
            encoder_fusion_mode=self.encoder_fusion_mode,
            freeze_2d_encoder=bool(freeze_2d_encoder),
            freeze_3d_encoder=bool(freeze_3d_encoder),
            freeze_3d_decoder=bool(freeze_3d_decoder),
        )

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _encode_2d(self, volume: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        selected_slices, slice_indices = self.slice_selector(volume)
        features_2d = self.encoder_2d(selected_slices)
        if self.position_encoder is not None:
            features_2d = self.position_encoder.forward_selected(features_2d, slice_indices)
        return features_2d, slice_indices

    def _scatter_2d_features_to_3d(
        self,
        feature_2d: torch.Tensor,
        feature_3d: torch.Tensor,
        slice_indices: torch.Tensor,
        stage_index: int,
    ) -> torch.Tensor:
        if feature_2d.ndim != 5:
            raise ValueError(f"Hybrid 2D features must be [B,K,C,H,W], got {tuple(feature_2d.shape)}")

        batch_size, num_slices, channels, height, width = feature_2d.shape
        target_hw = tuple(int(item) for item in feature_3d.shape[-2:])
        if (height, width) != target_hw:
            feature_2d = F.interpolate(
                feature_2d.reshape(batch_size * num_slices, channels, height, width),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_slices, channels, *target_hw)

        depth = int(feature_3d.shape[-3])
        divisor = 2 ** int(stage_index)
        mapped_indices = (slice_indices.to(device=feature_3d.device, dtype=torch.long) // divisor).clamp_(0, depth - 1)
        source = feature_2d.to(device=feature_3d.device, dtype=feature_3d.dtype).permute(0, 2, 1, 3, 4).contiguous()
        scatter_index = mapped_indices[:, None, :, None, None].expand(batch_size, channels, num_slices, *target_hw)

        lifted = feature_3d.new_zeros((batch_size, channels, depth, *target_hw))
        lifted.scatter_add_(2, scatter_index, source)

        counts = feature_3d.new_zeros((batch_size, 1, depth, 1, 1))
        count_index = mapped_indices[:, None, :, None, None]
        counts.scatter_add_(2, count_index, torch.ones_like(count_index, dtype=feature_3d.dtype))
        return lifted / counts.clamp_min(1.0)

    def _fuse_stage_features(self, feature_3d: torch.Tensor, lifted_2d: torch.Tensor, stage_index: int) -> torch.Tensor:
        projection = self.fusion_projections[stage_index]
        block = self.fusion_blocks[stage_index]
        if self.encoder_fusion_mode == "concat":
            return block(torch.cat([feature_3d, lifted_2d], dim=1))
        projected_2d = projection(lifted_2d)
        return block(feature_3d + projected_2d)

    def _encode_3d_with_slice_fusion(
        self,
        features_2d: list[torch.Tensor],
        slice_indices: torch.Tensor,
        volume: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        raw_3d_features: list[torch.Tensor] = []
        lifted_2d_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []

        current = volume
        for stage_index, stage in enumerate(self.encoder_3d.stages):
            feature_3d = stage(current)
            lifted_2d = self._scatter_2d_features_to_3d(features_2d[stage_index], feature_3d, slice_indices, stage_index)
            fused = self._fuse_stage_features(feature_3d, lifted_2d, stage_index)

            raw_3d_features.append(feature_3d)
            lifted_2d_features.append(lifted_2d)
            fused_features.append(fused)

            if stage_index < len(self.encoder_3d.pools):
                current = self.encoder_3d.pools[stage_index](fused)

        return fused_features, raw_3d_features, lifted_2d_features

    def forward(self, volume: torch.Tensor, return_features: bool = False):
        features_2d, slice_indices = self._encode_2d(volume)
        fused, features_3d, lifted_2d = self._encode_3d_with_slice_fusion(features_2d, slice_indices, volume)
        output = self.decoder_3d(fused)
        logits = _main_logits(output)
        if return_features:
            return self.build_output(
                logits,
                features={
                    "encoder": fused,
                    "encoder_2d": features_2d,
                    "encoder_3d": features_3d,
                    "lifted_2d": lifted_2d,
                    "slice_indices": slice_indices,
                    "deep_outputs": list(output[1:]) if isinstance(output, tuple) else [],
                },
            )
        return output if isinstance(output, tuple) and self.deep_supervision else logits

    def load_pretrained_components(
        self,
        ckpt_2d: str | Path,
        ckpt_3d: str | Path,
        load_2d_encoder: bool = True,
        load_3d_encoder: bool = True,
        load_3d_decoder: bool = True,
        strict: bool = False,
        freeze_2d_encoder: bool = False,
        freeze_3d_encoder: bool = False,
        freeze_3d_decoder: bool = False,
        log_path: str | Path | None = None,
        ckpt_3d_decoder: str | Path | None = None,
    ) -> dict[str, Any]:
        ckpt_2d = Path(ckpt_2d)
        ckpt_3d = Path(ckpt_3d)
        ckpt_3d_decoder = Path(ckpt_3d_decoder) if ckpt_3d_decoder is not None else ckpt_3d
        if load_2d_encoder and not ckpt_2d.exists():
            raise FileNotFoundError(f"Missing Stage 1 2D checkpoint. Please run experiment.stage=train_2d first: {ckpt_2d}")
        if load_3d_encoder and not ckpt_3d.exists():
            raise FileNotFoundError(f"Missing Stage 2 3D encoder checkpoint. Please run experiment.stage=train_3d first: {ckpt_3d}")
        if load_3d_decoder and not ckpt_3d_decoder.exists():
            raise FileNotFoundError(f"Missing Stage 2 3D decoder checkpoint: {ckpt_3d_decoder}")

        state_2d = _load_checkpoint(ckpt_2d) if load_2d_encoder else {}
        state_3d_encoder = _load_checkpoint(ckpt_3d) if load_3d_encoder else {}
        if load_3d_decoder:
            state_3d_decoder = (
                state_3d_encoder
                if load_3d_encoder and ckpt_3d_decoder == ckpt_3d
                else _load_checkpoint(ckpt_3d_decoder)
            )
        else:
            state_3d_decoder = {}

        report: dict[str, Any] = {
            "stage1_2d_ckpt": str(ckpt_2d),
            "stage2_3d_ckpt": str(ckpt_3d),
            "stage2_3d_encoder_ckpt": str(ckpt_3d),
            "stage2_3d_decoder_ckpt": str(ckpt_3d_decoder),
            "loaded_2d_encoder": False,
            "loaded_3d_encoder": False,
            "loaded_3d_decoder": False,
            "components": {},
        }
        if load_2d_encoder:
            component = self._load_component(self.encoder_2d, state_2d, source_prefix="encoder_2d.", label="2D encoder", strict=strict)
            report["components"]["2d_encoder"] = component
            report["loaded_2d_encoder"] = component["num_loaded_keys"] > 0
            if self.position_encoder is not None:
                position_component = self._load_component(
                    self.position_encoder,
                    state_2d,
                    source_prefix="position_encoder.",
                    label="2D position encoder",
                    strict=False,
                )
                report["components"]["2d_position_encoder"] = position_component
                report["loaded_2d_position_encoder"] = position_component["num_loaded_keys"] > 0
            if freeze_2d_encoder:
                self._freeze(self.encoder_2d)
                if self.position_encoder is not None:
                    self._freeze(self.position_encoder)
        if load_3d_encoder:
            component = self._load_component(self.encoder_3d, state_3d_encoder, source_prefix="encoder_3d.", label="3D encoder", strict=strict)
            report["components"]["3d_encoder"] = component
            report["loaded_3d_encoder"] = component["num_loaded_keys"] > 0
            if freeze_3d_encoder:
                self._freeze(self.encoder_3d)
        if load_3d_decoder:
            component = self._load_component(self.decoder_3d, state_3d_decoder, source_prefix="decoder_3d.", label="3D decoder", strict=strict)
            report["components"]["3d_decoder"] = component
            report["loaded_3d_decoder"] = component["num_loaded_keys"] > 0
            if freeze_3d_decoder:
                self._freeze(self.decoder_3d)

        report["num_loaded_2d_keys"] = report.get("components", {}).get("2d_encoder", {}).get("num_loaded_keys", 0)
        report["num_loaded_2d_position_keys"] = report.get("components", {}).get("2d_position_encoder", {}).get("num_loaded_keys", 0)
        report["num_loaded_3d_encoder_keys"] = report.get("components", {}).get("3d_encoder", {}).get("num_loaded_keys", 0)
        report["num_loaded_3d_decoder_keys"] = report.get("components", {}).get("3d_decoder", {}).get("num_loaded_keys", 0)
        report["num_skipped_keys"] = sum(item.get("num_skipped_keys", 0) for item in report["components"].values())
        self.pretrain_loading_report = report

        if log_path is not None:
            self._write_pretrain_log(Path(log_path), report)
        return report

    @staticmethod
    def _load_component(
        module: nn.Module,
        state_dict: Mapping[str, Any],
        source_prefix: str,
        label: str,
        strict: bool = False,
    ) -> dict[str, Any]:
        target_state = module.state_dict()
        updates = {}
        skipped = []
        seen_prefixed_key = False

        for raw_key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = _strip_module_prefix(str(raw_key))
            if clean_key.startswith(source_prefix):
                seen_prefixed_key = True
                target_key = clean_key[len(source_prefix) :]
            elif not seen_prefixed_key and clean_key in target_state:
                target_key = clean_key
            else:
                continue

            if target_key in target_state and tuple(target_state[target_key].shape) == tuple(value.shape):
                updates[target_key] = value
            elif target_key in target_state:
                skipped.append(
                    {
                        "key": clean_key,
                        "target_key": target_key,
                        "reason": f"shape {tuple(value.shape)} != {tuple(target_state[target_key].shape)}",
                    }
                )
            else:
                skipped.append({"key": clean_key, "target_key": target_key, "reason": "target key not found"})

        merged = dict(target_state)
        merged.update(updates)
        module.load_state_dict(merged, strict=False)
        missing = [key for key in target_state if key not in updates]
        if strict and missing:
            raise RuntimeError(f"{label} missing keys after partial load: {missing[:20]}")
        return {
            "label": label,
            "source_prefix": source_prefix,
            "loaded_keys": sorted(updates),
            "skipped_keys": skipped,
            "missing_keys": missing,
            "num_loaded_keys": len(updates),
            "num_skipped_keys": len(skipped),
            "num_missing_keys": len(missing),
        }

    @staticmethod
    def _write_pretrain_log(path: Path, report: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("Stage 3 pretrained loading report\n")
            handle.write(f"stage1_2d_ckpt: {report.get('stage1_2d_ckpt')}\n")
            handle.write(f"stage2_3d_ckpt: {report.get('stage2_3d_ckpt')}\n")
            for name, component in dict(report.get("components", {})).items():
                handle.write("\n")
                handle.write(f"[{name}]\n")
                handle.write(f"loaded: {component.get('num_loaded_keys', 0)}\n")
                handle.write(f"skipped: {component.get('num_skipped_keys', 0)}\n")
                handle.write(f"missing: {component.get('num_missing_keys', 0)}\n")
                handle.write("loaded_keys:\n")
                for key in component.get("loaded_keys", [])[:200]:
                    handle.write(f"  - {key}\n")
                handle.write("skipped_keys:\n")
                for item in component.get("skipped_keys", [])[:200]:
                    handle.write(f"  - {json.dumps(item, sort_keys=True)}\n")
                handle.write("missing_keys:\n")
                for key in component.get("missing_keys", [])[:200]:
                    handle.write(f"  - {key}\n")


Hybrid3D2DUNet3PlusSliceInject = Hybrid3D2DUNet3Plus


__all__ = [
    "FullUNet3Plus2D",
    "FullUnet3D_3_plus",
    "FullUNet3D",
    "Hybrid3D2DUNet3Plus",
    "Hybrid3D2DUNet3PlusSliceInject",
]
