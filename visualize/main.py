from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError as error:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required. Install it with `pip install pyyaml`.") from error

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from training.data import (
    CystRecord,
    build_2d_volume_eval_datasets,
    build_datasets_from_records,
    build_kfold_record_splits,
)
from training.metrics import confusion_counts, metrics_from_counts, surface_metrics
from training.model_factory import build_model
from training.runner import (
    _cfg_with_checkpoint_shape_hints,
    _checkpoint_state_dict,
    _prepare_proposal_stage_config,
    _strip_module_prefix,
)
from training.torch_utils import ensure_model_on_device, extract_logits, load_model_state, predict_from_logits, resize_logits
from training.utils import get_nested, set_seed


METRIC_FIELDS = [
    "fold",
    "sample_id",
    "source",
    "model_name",
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95",
    "gt_voxels",
    "pred_voxels",
    "tp",
    "tn",
    "fp",
    "fn",
    "checkpoint_path",
]

CI_FIELDS = [
    "model_name",
    "n",
    "mean_dice",
    "std_dice",
    "standard_error",
    "normal_ci95_low",
    "normal_ci95_high",
    "bootstrap_mean",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "bootstrap_n",
]

EVALUATION_SPLIT = "test"

SUMMARY_FIELDS = [
    "model_name",
    "fold",
    "split",
    "n",
    "global_dice",
    "global_iou",
    "global_precision",
    "global_recall",
    "mean_sample_dice",
    "std_sample_dice",
    "mean_sample_iou",
    "mean_hd95",
    "gt_voxels",
    "pred_voxels",
    "tp",
    "tn",
    "fp",
    "fn",
]

SOURCE_SUMMARY_FIELDS = [
    "model_name",
    "fold",
    "split",
    "source",
    "n",
    "global_dice",
    "global_iou",
    "global_precision",
    "global_recall",
    "mean_sample_dice",
    "std_sample_dice",
    "mean_sample_iou",
    "mean_hd95",
    "gt_voxels",
    "pred_voxels",
    "tp",
    "tn",
    "fp",
    "fn",
]

KFOLD_SUMMARY_FIELDS = [
    "model_name",
    "split",
    "folds",
    "fold_ids",
    "runner_style_dice_mean",
    "runner_style_dice_std",
    "runner_style_dice_mean_pm_std",
    "runner_style_iou_mean",
    "runner_style_iou_std",
    "runner_style_precision_mean",
    "runner_style_recall_mean",
    "fold_sample_dice_mean",
    "fold_sample_dice_std",
]

SOURCE_KFOLD_SUMMARY_FIELDS = [
    "model_name",
    "split",
    "source",
    "folds",
    "fold_ids",
    "runner_style_dice_mean",
    "runner_style_dice_std",
    "runner_style_dice_mean_pm_std",
    "runner_style_iou_mean",
    "runner_style_iou_std",
    "runner_style_precision_mean",
    "runner_style_recall_mean",
    "fold_sample_dice_mean",
    "fold_sample_dice_std",
]

VALIDATION_FIELDS = [
    "model_name",
    "component",
    "fold",
    "status",
    "checkpoint_dir",
    "checkpoint_path",
    "stage",
    "model_type",
    "encoder_2d",
    "encoder_3d",
    "decoder_model",
    "decoder_style",
    "slice_mode",
    "proposal_order",
    "build_name",
    "backbone",
    "arch_encoder_2d",
    "arch_encoder_3d",
    "arch_decoder_model",
    "arch_decoder_style",
    "arch_slice_mode",
    "arch_fusion",
    "params",
    "checkpoint_epoch",
    "checkpoint_best_metric",
    "error",
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    stage: str
    checkpoint_dir: Path
    encoder_2d: str | None = None
    encoder_3d: str | None = None
    decoder_model: str | None = None
    decoder_style: str | None = None
    slice_mode: str | None = None
    proposal_order: str | None = None
    is_fusion: bool = False
    fusion_2d_dir: Path | None = None
    fusion_3d_dir: Path | None = None


@dataclass
class BuiltModel:
    model: torch.nn.Module
    cfg: Mapping[str, Any]
    num_classes: int
    model_type: str
    checkpoint_path: Path
    build_name: str
    backbone: str
    checkpoint_epoch: Any = ""
    checkpoint_best_metric: Any = ""


@dataclass
class InferenceResult:
    sample_id: str
    source: str
    image: np.ndarray
    gt: np.ndarray
    logits: torch.Tensor
    pred: np.ndarray


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and bool(value.get("__replace__", False)):
            merged[key] = {nested_key: nested_value for nested_key, nested_value in value.items() if nested_key != "__replace__"}
            continue
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Mapping[str, Any]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    base_config = payload.get("extends", payload.get("base_config"))
    if not base_config:
        return payload
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    override = dict(payload)
    override.pop("extends", None)
    override.pop("base_config", None)
    return _deep_merge(_load_yaml(base_path), override)


def _base_cfg() -> dict[str, Any]:
    return dict(_load_yaml(PROJECT_ROOT / "config" / "Proposal_Experiment" / "model_experiment_base.yaml"))


def _set_nested(mapping: dict[str, Any], path: str, value: Any) -> None:
    cursor = mapping
    keys = path.split(".")
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def _model_cfg_from_request(model_config: ModelConfig) -> dict[str, Any]:
    cfg = _base_cfg()
    _set_nested(cfg, "project.name", "cyst_proposal_model_experiment_visualize")
    _set_nested(cfg, "experiment.name", "Proposal_Model_Experiment")
    _set_nested(cfg, "experiment.stage", model_config.stage)
    _set_nested(cfg, "model.position_encoder.enabled", False)
    _set_nested(cfg, "k_fold.enabled", True)
    _set_nested(cfg, "k_fold.num_folds", 3)
    _set_nested(cfg, "evaluation.evaluate_2d_as_volume", True)
    _set_nested(cfg, "evaluation.batch_size", 1)
    _set_nested(cfg, "evaluation.batch_size_3d", 1)

    if model_config.encoder_2d is not None:
        _set_nested(cfg, "model.encoder_2d.type", model_config.encoder_2d)
    if model_config.encoder_3d is not None:
        _set_nested(cfg, "model.encoder_3d.type", model_config.encoder_3d)
    if model_config.decoder_model is not None:
        block = "model.decoder" if model_config.stage == "hybrid" else "model.decoder_3d"
        _set_nested(cfg, f"{block}.model", model_config.decoder_model)
    if model_config.decoder_style is not None:
        block = "model.decoder" if model_config.stage == "hybrid" else "model.decoder_3d"
        _set_nested(cfg, f"{block}.style", model_config.decoder_style)
    if model_config.slice_mode is not None:
        _set_nested(cfg, "model.slice_selection.mode", model_config.slice_mode)
    if model_config.proposal_order is not None:
        _set_nested(cfg, "model.slice_selection.proposal.selection_order", model_config.proposal_order)

    return dict(_prepare_proposal_stage_config(cfg))


def _apply_proposal_order_override(cfg: Mapping[str, Any], model_config: ModelConfig) -> dict[str, Any]:
    patched = dict(cfg)
    if model_config.proposal_order is not None:
        _set_nested(patched, "model.slice_selection.proposal.selection_order", model_config.proposal_order)
        _set_nested(patched, "slice_2d.proposal.selection_order", model_config.proposal_order)
    return patched


def get_model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            name="Unet2D",
            stage="train_2d",
            encoder_2d="unet",
            slice_mode="proposal",
            proposal_order="different",
            checkpoint_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/1_2D_pretrain/unet/no_pos/proposal/seed_42",
        ),
        ModelConfig(
            name="Unet++",
            stage="train_2d",
            encoder_2d="unetpp",
            slice_mode="proposal",
            proposal_order="different",
            checkpoint_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/1_2D_pretrain/unetpp/no_pos/proposal/seed_42",
        ),
        ModelConfig(
            name="Unet3+",
            stage="train_2d",
            encoder_2d="unet3plus",
            slice_mode="proposal",
            proposal_order="different",
            checkpoint_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/1_2D_pretrain/unet3plus/no_pos/proposal/seed_42",
        ),
        ModelConfig(
            name="Unet3D",
            stage="train_3d",
            encoder_3d="unet3d",
            decoder_model="unet3d",
            decoder_style="same_scale",
            checkpoint_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/2_3D_pretrain/unet3d_unet3d_same_scale/seed_42",
        ),
        ModelConfig(
            name="nnUNet",
            stage="train_3d",
            encoder_3d="nnunet3d",
            decoder_model="nnunet3d",
            decoder_style="same_scale",
            checkpoint_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/2_3D_pretrain/nnunet3d_nnunet3d_same_scale/seed_42",
        ),
        ModelConfig(
            name="Ours",
            stage="hybrid",
            encoder_2d="unet3plus",
            encoder_3d="unet3plus3d",
            decoder_model="unet3plus3d",
            decoder_style="full_scale",
            slice_mode="uniform",
            checkpoint_dir=PROJECT_ROOT
            / "outputs/Proposal_Model_Experiment/3_hybrid/seed_42/unet3plus_unet3plus3d_unet3plus3d_full_scale/no_pos/uniform",
        ),
        ModelConfig(
            name="Fusion_Late",
            stage="fusion_late",
            checkpoint_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/fusion_late",
            slice_mode="uniform",
            is_fusion=True,
            fusion_2d_dir=PROJECT_ROOT / "outputs/Proposal_Model_Experiment/1_2D_pretrain/unet3plus/no_pos/uniform/seed_42",
            fusion_3d_dir=PROJECT_ROOT
            / "outputs/Proposal_Model_Experiment/2_3D_pretrain/unet3plus3d_unet3plus3d_full_scale/seed_42",
        ),
    ]


def sanitize_model_name(name: str) -> str:
    # Keep paper display names in CSV while producing portable folder names if needed.
    return str(name).replace("/", "_").replace("\\", "_").strip() or "model"


def _checkpoint_priority_names() -> list[str]:
    return [
        "best_model.pth",
        "model_best.pth",
        "best.pth",
        "checkpoint_best.pth",
        "checkpoint.pth",
        "last_model.pth",
        "last.pth",
    ]


def _fold_tokens(fold: int) -> list[str]:
    return [f"fold_{fold}", f"fold{fold}", f"fold-{fold}", f"fold_{fold + 1:02d}"]


def _has_exact_fold_part(path: Path, tokens: Sequence[str]) -> bool:
    token_set = {str(token).lower() for token in tokens}
    return any(part.lower() in token_set for part in path.parts)


def _has_any_fold_part(path: Path) -> bool:
    return any(part.lower().startswith(("fold_", "fold-", "fold")) for part in path.parts)


def _candidate_files_in_dir(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in _checkpoint_priority_names():
        candidates.extend([directory / name, directory / "checkpoints" / name, directory / "checkpoint" / name])
    candidates.extend(sorted(directory.glob("*.ckpt")))
    candidates.extend(sorted((directory / "checkpoints").glob("*.ckpt")) if (directory / "checkpoints").exists() else [])
    return candidates


def find_checkpoint(model_dir: str | Path, fold: int, require_fold_specific: bool = True) -> Path:
    root = Path(model_dir)
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint folder does not exist for fold {fold}: {root}")

    tokens = _fold_tokens(int(fold))
    token_lower = [token.lower() for token in tokens]

    for token in tokens:
        for folder in (root / token, root / token / "checkpoints", root / token / "checkpoint"):
            for candidate in _candidate_files_in_dir(folder):
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()

    recursive_files = [path for path in root.rglob("*") if path.is_file() and (path.suffix in {".pth", ".ckpt"} or path.name in _checkpoint_priority_names())]
    fold_specific = [path for path in recursive_files if _has_exact_fold_part(path.relative_to(root), token_lower)]
    for name in _checkpoint_priority_names():
        for candidate in fold_specific:
            if candidate.name == name:
                return candidate.resolve()
    if fold_specific:
        return sorted(fold_specific)[0].resolve()

    has_other_fold_files = any(_has_any_fold_part(path.relative_to(root)) for path in recursive_files)
    if require_fold_specific and has_other_fold_files:
        raise FileNotFoundError(
            f"Could not find a checkpoint for fold {fold} under {root}. "
            f"Found fold-like checkpoint paths, but none matched exact fold tokens {tokens}."
        )

    if require_fold_specific:
        raise FileNotFoundError(
            f"Could not find a fold-specific checkpoint for fold {fold} under {root}. "
            f"Tried exact fold tokens {tokens}."
        )

    for candidate in _candidate_files_in_dir(root):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    for name in _checkpoint_priority_names():
        matches = [path for path in recursive_files if path.name == name]
        if matches:
            return sorted(matches)[0].resolve()

    ckpt_matches = sorted(root.rglob("*.ckpt"))
    if ckpt_matches:
        return ckpt_matches[0].resolve()

    raise FileNotFoundError(
        "Could not find checkpoint for "
        f"fold {fold} under {root}. Tried fold tokens {tokens} and names {_checkpoint_priority_names()}."
    )


def _load_checkpoint(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)!r}: {path}")
    state_dict = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
    return checkpoint, state_dict


def _config_from_checkpoint_or_default(model_config: ModelConfig, checkpoint: Mapping[str, Any], state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_cfg = checkpoint.get("config")
    if isinstance(raw_cfg, Mapping):
        cfg = dict(raw_cfg)
        cfg = _apply_proposal_order_override(cfg, model_config)
        cfg = dict(_prepare_proposal_stage_config(cfg))
        cfg = _apply_proposal_order_override(cfg, model_config)
    else:
        cfg = _model_cfg_from_request(model_config)
        cfg = _apply_proposal_order_override(cfg, model_config)
    return _cfg_with_checkpoint_shape_hints(cfg, state_dict)


def _dataset_in_channels(cfg: Mapping[str, Any]) -> int:
    model_type = str(get_nested(cfg, "model.type", "2D")).upper()
    if model_type == "2D":
        return max(1, int(get_nested(cfg, "slice_2d.num_slices", get_nested(cfg, "model.in_channels", 1))))
    return 1


def _normal_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace("+", "plus")


def _architecture_value(built: BuiltModel | None, *keys: str) -> str:
    if built is None:
        return ""
    architecture_config = getattr(built.model, "architecture_config", None)
    if not isinstance(architecture_config, Mapping):
        return ""
    for key in keys:
        value = architecture_config.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _parameter_count(model: torch.nn.Module | None) -> int | str:
    if model is None:
        return ""
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _expected_model_type(model_config: ModelConfig) -> str:
    return "2D" if model_config.stage == "train_2d" else "3D"


def _require_config_match(errors: list[str], cfg: Mapping[str, Any], path: str, expected: Any, label: str | None = None) -> None:
    actual = get_nested(cfg, path, None)
    if _normal_token(actual) != _normal_token(expected):
        errors.append(f"{label or path}: expected {expected!r}, got {actual!r}")


def validate_checkpoint_config(model_config: ModelConfig, cfg: Mapping[str, Any], checkpoint_path: Path) -> None:
    if model_config.is_fusion:
        return
    errors: list[str] = []
    actual_stage = str(get_nested(cfg, "experiment.stage", "")).lower()
    expected_stage = str(model_config.stage).lower()
    # Standalone baseline checkpoints (for example SegMamba/Swin-UNETR) do
    # not use Proposal's experiment.stage field. An empty requested stage
    # means "do not constrain stage", while model.type is still validated.
    if expected_stage and actual_stage != expected_stage:
        errors.append(f"experiment.stage: expected {expected_stage!r}, got {actual_stage!r}")

    actual_model_type = str(get_nested(cfg, "model.type", "")).upper()
    expected_type = _expected_model_type(model_config)
    if actual_model_type != expected_type:
        errors.append(f"model.type: expected {expected_type!r}, got {actual_model_type!r}")

    if model_config.encoder_2d is not None:
        _require_config_match(errors, cfg, "model.encoder_2d.type", model_config.encoder_2d, "encoder_2d")
    if model_config.encoder_3d is not None:
        _require_config_match(errors, cfg, "model.encoder_3d.type", model_config.encoder_3d, "encoder_3d")
    if model_config.decoder_model is not None:
        decoder_path = "model.decoder.model" if expected_stage == "hybrid" else "model.decoder_3d.model"
        _require_config_match(errors, cfg, decoder_path, model_config.decoder_model, "decoder_model")
    if model_config.decoder_style is not None:
        decoder_path = "model.decoder.style" if expected_stage == "hybrid" else "model.decoder_3d.style"
        _require_config_match(errors, cfg, decoder_path, model_config.decoder_style, "decoder_style")
    if model_config.slice_mode is not None:
        _require_config_match(errors, cfg, "model.slice_selection.mode", model_config.slice_mode, "slice_mode")
    if model_config.proposal_order is not None:
        _require_config_match(errors, cfg, "model.slice_selection.proposal.selection_order", model_config.proposal_order, "proposal_order")

    if errors:
        joined = "; ".join(errors)
        raise RuntimeError(f"Checkpoint config does not match {model_config.name} for {checkpoint_path}: {joined}")


def build_model_by_name(model_name: str, checkpoint_path: str | Path, device: torch.device, fallback_config: ModelConfig | None = None) -> BuiltModel:
    path = Path(checkpoint_path)
    checkpoint, state_dict = _load_checkpoint(path)
    model_config = fallback_config or next((item for item in get_model_configs() if item.name == model_name), None)
    if model_config is None:
        raise ValueError(f"Unknown model name: {model_name}")
    cfg = _config_from_checkpoint_or_default(model_config, checkpoint, state_dict)
    validate_checkpoint_config(model_config, cfg, path)
    result = build_model(cfg, dataset_in_channels=_dataset_in_channels(cfg))
    try:
        load_model_state(result.model, state_dict)
    except RuntimeError as error:
        raise RuntimeError(f"Failed to load weights for {model_config.name} from {path}") from error
    model = ensure_model_on_device(result.model, device)
    model.eval()
    return BuiltModel(
        model=model,
        cfg=cfg,
        num_classes=result.num_classes,
        model_type=str(get_nested(cfg, "model.type", "2D")).upper(),
        checkpoint_path=path.resolve(),
        build_name=result.name,
        backbone=result.backbone,
        checkpoint_epoch=checkpoint.get("epoch", "") if isinstance(checkpoint, Mapping) else "",
        checkpoint_best_metric=checkpoint.get("best_metric", "") if isinstance(checkpoint, Mapping) else "",
    )


def _records_for_fold(cfg: Mapping[str, Any], fold: int) -> Mapping[str, Sequence[CystRecord]]:
    splits, _metadata = build_kfold_record_splits(cfg)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Fold must be between 0 and {len(splits) - 1}, got {fold}.")
    return splits[fold]


def build_dataloader_for_fold(cfg: Mapping[str, Any], fold: int, split: str, model_type: str | None = None, num_workers: int = 0):
    records = _records_for_fold(cfg, fold)
    if split not in records:
        raise KeyError(f"Split {split!r} is not available for fold {fold}. Available splits: {sorted(records)}")
    selected = {split: records[split]}
    resolved_type = (model_type or str(get_nested(cfg, "model.type", "2D"))).upper()
    if resolved_type == "2D":
        dataset = build_2d_volume_eval_datasets(cfg, selected)[split]
    else:
        datasets, _ = build_datasets_from_records(cfg, selected, augment_train=False)
        dataset = datasets[split]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=max(0, int(num_workers)), pin_memory=False)
    return loader, dataset


def _model_uses_slice_indices(model: torch.nn.Module) -> bool:
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    return bool(getattr(base, "expects_slice_indices", False))


def _call_model(model: torch.nn.Module, image: torch.Tensor, slice_indices: torch.Tensor | None = None):
    kwargs: dict[str, Any] = {}
    if slice_indices is not None and _model_uses_slice_indices(model):
        kwargs["slice_indices"] = slice_indices
    return model(image, **kwargs)


def _as_single_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (1,):
            output[key] = value[0]
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            output[key] = value[0]
        else:
            output[key] = value
    return output


def _text_value(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return _text_value(value[0])
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return str(value.item())
        return str(value.detach().cpu().tolist())
    return str(value)


def _source_prefix(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text).name
    for suffix in (".nii.gz", ".nii", ".gz"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    match = re.match(r"([A-Za-z]+)", name)
    return match.group(1).upper() if match else ""


def _infer_source(sample_id: str, image_path: str | None = None) -> str:
    prefix = _source_prefix(image_path) or _source_prefix(sample_id)
    if prefix:
        return prefix
    print(f"WARNING: source could not be inferred for sample {sample_id}; using unknown.")
    return "UNKNOWN"


def _stack_slice_logits_as_dhw(logit_slices: torch.Tensor, axis: int) -> torch.Tensor:
    # Input [D, C, H, W]. Insert slice depth back into the source spatial axis,
    # then move that depth axis to the canonical [C, D, H, W] layout.
    volume = torch.stack([item for item in logit_slices], dim=int(axis) + 1)
    if int(axis) + 1 != 1:
        volume = torch.movedim(volume, int(axis) + 1, 1)
    return volume.unsqueeze(0).contiguous()


def _target_to_dhw(target: np.ndarray | torch.Tensor, axis: int) -> np.ndarray:
    array = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)
    return np.moveaxis(array, int(axis), 0).astype(np.int64, copy=False)


def _image_slices_to_dhw(image_slices: torch.Tensor) -> np.ndarray:
    # Use the central input channel for display.
    channels = int(image_slices.shape[1])
    return image_slices[:, channels // 2].detach().cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def run_2d_inference_slice_by_slice(model_2d: torch.nn.Module, sample: Mapping[str, Any], device: torch.device, num_classes: int, slice_batch_size: int = 8) -> InferenceResult:
    sample = _as_single_sample(sample)
    image_slices = sample["image_slices"].to(device, non_blocking=True)
    axis = int(_text_value(sample.get("slice_axis", 2)))
    logits_chunks: list[torch.Tensor] = []
    slice_batch_size = max(1, int(slice_batch_size))
    for start in range(0, int(image_slices.shape[0]), slice_batch_size):
        image_batch = image_slices[start : start + slice_batch_size]
        slice_indices = torch.arange(start, start + int(image_batch.shape[0]), device=device, dtype=torch.long)
        logits = extract_logits(_call_model(model_2d, image_batch, slice_indices=slice_indices), num_classes=num_classes)
        logits_chunks.append(logits.detach().cpu())
    logits_slices = torch.cat(logits_chunks, dim=0)
    logits_volume = _stack_slice_logits_as_dhw(logits_slices, axis=axis)
    gt = _target_to_dhw(sample["label"], axis=axis)
    pred = predict_from_logits(logits_volume, target_shape=tuple(gt.shape)).squeeze(0).cpu().numpy().astype(np.uint8)
    sample_id = _text_value(sample.get("case_id", "unknown"))
    image_path = _text_value(sample.get("image_path", ""))
    return InferenceResult(
        sample_id=sample_id,
        source=_infer_source(sample_id, image_path),
        image=_image_slices_to_dhw(image_slices.detach().cpu()),
        gt=gt,
        logits=resize_logits(logits_volume, tuple(gt.shape)).detach().cpu(),
        pred=pred,
    )


@torch.no_grad()
def run_3d_inference(model_3d: torch.nn.Module, sample: Mapping[str, Any], device: torch.device, num_classes: int) -> InferenceResult:
    sample = _as_single_sample(sample)
    image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
    label = sample["label"].detach().cpu().numpy().astype(np.int64, copy=False)
    logits = extract_logits(_call_model(model_3d, image), num_classes=num_classes)
    logits = resize_logits(logits, tuple(label.shape)).detach().cpu()
    pred = predict_from_logits(logits, target_shape=tuple(label.shape)).squeeze(0).cpu().numpy().astype(np.uint8)
    sample_id = _text_value(sample.get("case_id", "unknown"))
    image_path = _text_value(sample.get("image_path", ""))
    return InferenceResult(
        sample_id=sample_id,
        source=_infer_source(sample_id, image_path),
        image=sample["image"][0].detach().cpu().numpy().astype(np.float32, copy=False),
        gt=label,
        logits=logits,
        pred=pred,
    )


def run_inference(model: BuiltModel, sample: Mapping[str, Any], model_name: str, device: torch.device) -> InferenceResult:
    if model.model_type == "2D":
        return run_2d_inference_slice_by_slice(
            model.model,
            sample,
            device,
            model.num_classes,
            slice_batch_size=int(get_nested(model.cfg, "evaluation.slice_batch_size", 8)),
        )
    return run_3d_inference(model.model, sample, device, model.num_classes)


def align_logits_to_gt_shape(logits: torch.Tensor, gt_shape: Sequence[int]) -> torch.Tensor:
    return resize_logits(logits, tuple(int(item) for item in gt_shape))


def postprocess_logits(logits: torch.Tensor) -> torch.Tensor:
    return predict_from_logits(logits)


@torch.no_grad()
def run_late_fusion_inference(model_2d: BuiltModel, model_3d: BuiltModel, sample_2d: Mapping[str, Any], sample_3d: Mapping[str, Any], device: torch.device, slice_batch_size: int = 8) -> InferenceResult:
    result_2d = run_2d_inference_slice_by_slice(model_2d.model, sample_2d, device, model_2d.num_classes, slice_batch_size=slice_batch_size)
    result_3d = run_3d_inference(model_3d.model, sample_3d, device, model_3d.num_classes)
    target_shape = tuple(int(item) for item in result_2d.gt.shape)
    logits_2d = align_logits_to_gt_shape(result_2d.logits.to(device), target_shape)
    logits_3d = align_logits_to_gt_shape(result_3d.logits.to(device), target_shape)
    if logits_2d.shape[1] != logits_3d.shape[1]:
        raise RuntimeError(f"Fusion_Late class mismatch: 2D logits {tuple(logits_2d.shape)}, 3D logits {tuple(logits_3d.shape)}")
    fused_logits = ((logits_2d + logits_3d) / 2.0).detach().cpu()
    pred = predict_from_logits(fused_logits, target_shape=target_shape).squeeze(0).cpu().numpy().astype(np.uint8)
    return InferenceResult(
        sample_id=result_2d.sample_id,
        source=result_2d.source,
        image=result_2d.image,
        gt=result_2d.gt,
        logits=fused_logits,
        pred=pred,
    )


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    counts = confusion_counts(pred, gt)
    summary = metrics_from_counts(**counts)
    surface = surface_metrics(pred, gt)
    return {
        "dice": float(summary["Dice"]),
        "iou": float(summary["IoU"]),
        "precision": float(summary["Precision"]),
        "recall": float(summary["Recall"]),
        "hd95": float(surface["HD95"]),
        "gt_voxels": int(np.asarray(gt > 0).sum()),
        "pred_voxels": int(np.asarray(pred > 0).sum()),
        "tp": int(counts["tp"]),
        "tn": int(counts["tn"]),
        "fp": int(counts["fp"]),
        "fn": int(counts["fn"]),
    }


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(tmp_path, path)


def _iter_dataset(loader: DataLoader) -> Iterable[Mapping[str, Any]]:
    for sample in loader:
        yield sample


def _sample_matches(sample: Mapping[str, Any], sample_id: str) -> bool:
    return _text_value(_as_single_sample(sample).get("case_id", "")) == str(sample_id)


def _find_sample(loader: DataLoader, sample_id: str) -> Mapping[str, Any]:
    for sample in _iter_dataset(loader):
        if _sample_matches(sample, sample_id):
            return sample
    raise KeyError(f"Sample {sample_id!r} was not found in this fold.")


def _device_from_gpu(gpu_id: int | None) -> torch.device:
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _metrics_output_path(results_dir: Path, model_name: str, fold: int) -> Path:
    return results_dir / "metrics" / sanitize_model_name(model_name) / f"fold_{fold}_per_sample_metrics.csv"


def _summary_output_path(results_dir: Path, model_name: str, fold: int | str) -> Path:
    return results_dir / "metrics" / sanitize_model_name(model_name) / f"fold_{fold}_test_summary.csv"


def _source_summary_output_path(results_dir: Path, model_name: str, fold: int | str) -> Path:
    return results_dir / "metrics" / sanitize_model_name(model_name) / f"fold_{fold}_source_summary.csv"


def _numeric_value(row: Mapping[str, Any], field: str, default: float = math.nan) -> float:
    value = row.get(field, default)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def _finite_std(values: Sequence[float]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def summarize_metric_rows(rows: Sequence[Mapping[str, Any]], model_name: str, fold: int | str, split: str = EVALUATION_SPLIT) -> dict[str, Any]:
    n = len(rows)
    count_fields = ("tp", "tn", "fp", "fn")
    counts = {field: int(sum(_numeric_value(row, field, 0.0) for row in rows)) for field in count_fields}
    global_metrics = metrics_from_counts(**counts) if n else {}
    return {
        "model_name": model_name,
        "fold": fold,
        "split": split,
        "n": n,
        "global_dice": float(global_metrics.get("Dice", math.nan)),
        "global_iou": float(global_metrics.get("IoU", math.nan)),
        "global_precision": float(global_metrics.get("Precision", math.nan)),
        "global_recall": float(global_metrics.get("Recall", math.nan)),
        "mean_sample_dice": _finite_mean([_numeric_value(row, "dice") for row in rows]),
        "std_sample_dice": _finite_std([_numeric_value(row, "dice") for row in rows]),
        "mean_sample_iou": _finite_mean([_numeric_value(row, "iou") for row in rows]),
        "mean_hd95": _finite_mean([_numeric_value(row, "hd95") for row in rows]),
        "gt_voxels": int(sum(_numeric_value(row, "gt_voxels", 0.0) for row in rows)),
        "pred_voxels": int(sum(_numeric_value(row, "pred_voxels", 0.0) for row in rows)),
        **counts,
    }


def summarize_source_metric_rows(rows: Sequence[Mapping[str, Any]], model_name: str, fold: int | str, split: str = EVALUATION_SPLIT) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        source = str(row.get("source", "UNKNOWN") or "UNKNOWN").upper()
        grouped.setdefault(source, []).append(row)
    summaries: list[dict[str, Any]] = []
    for source in sorted(grouped):
        summary = summarize_metric_rows(grouped[source], model_name, fold, split=split)
        summary["source"] = source
        summaries.append(summary)
    return summaries


def _format_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean) and math.isnan(std):
        return "nan"
    return f"{mean:.6g} +/- {std:.6g}"


def summarize_kfold_style(fold_summaries: Sequence[Mapping[str, Any]], model_name: str, split: str = EVALUATION_SPLIT) -> dict[str, Any]:
    fold_ids = [str(item.get("fold", "")) for item in fold_summaries]
    dice_values = [_numeric_value(item, "global_dice") for item in fold_summaries]
    dice_mean = _finite_mean(dice_values)
    dice_std = _finite_std(dice_values)
    return {
        "model_name": model_name,
        "split": split,
        "folds": len(fold_summaries),
        "fold_ids": ";".join(fold_ids),
        "runner_style_dice_mean": dice_mean,
        "runner_style_dice_std": dice_std,
        "runner_style_dice_mean_pm_std": _format_mean_std(dice_mean, dice_std),
        "runner_style_iou_mean": _finite_mean([_numeric_value(item, "global_iou") for item in fold_summaries]),
        "runner_style_iou_std": _finite_std([_numeric_value(item, "global_iou") for item in fold_summaries]),
        "runner_style_precision_mean": _finite_mean([_numeric_value(item, "global_precision") for item in fold_summaries]),
        "runner_style_recall_mean": _finite_mean([_numeric_value(item, "global_recall") for item in fold_summaries]),
        "fold_sample_dice_mean": _finite_mean([_numeric_value(item, "mean_sample_dice") for item in fold_summaries]),
        "fold_sample_dice_std": _finite_std([_numeric_value(item, "mean_sample_dice") for item in fold_summaries]),
    }


def summarize_source_kfold_style(source_fold_summaries: Sequence[Mapping[str, Any]], model_name: str, split: str = EVALUATION_SPLIT) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_fold_summaries:
        source = str(row.get("source", "UNKNOWN") or "UNKNOWN").upper()
        grouped.setdefault(source, []).append(row)
    rows: list[dict[str, Any]] = []
    for source in sorted(grouped):
        summary = summarize_kfold_style(grouped[source], model_name, split=split)
        summary["source"] = source
        rows.append(summary)
    return rows


def evaluate_model_fold(model_config: ModelConfig, fold: int, device: torch.device, results_dir: Path, num_workers: int = 0, save_predictions: bool = False) -> Path:
    checkpoint_path = find_checkpoint(model_config.checkpoint_dir, fold)
    print(f"[metrics] fold={fold} model={model_config.name} device={device} checkpoint={checkpoint_path}")
    built = build_model_by_name(model_config.name, checkpoint_path, device, fallback_config=model_config)
    loader, dataset = build_dataloader_for_fold(built.cfg, fold, EVALUATION_SPLIT, built.model_type, num_workers=num_workers)
    print(f"[metrics] fold={fold} split={EVALUATION_SPLIT} full_volume=true model={model_config.name} samples={len(dataset)}")

    rows: list[dict[str, Any]] = []
    slice_batch_size = int(get_nested(built.cfg, "evaluation.slice_batch_size", 8))
    for sample in _iter_dataset(loader):
        if built.model_type == "2D":
            inference = run_2d_inference_slice_by_slice(built.model, sample, device, built.num_classes, slice_batch_size=slice_batch_size)
        else:
            inference = run_3d_inference(built.model, sample, device, built.num_classes)
        metrics = compute_sample_metrics(inference.pred, inference.gt)
        rows.append(
            {
                "fold": fold,
                "sample_id": inference.sample_id,
                "source": inference.source,
                "model_name": model_config.name,
                **metrics,
                "checkpoint_path": str(checkpoint_path),
            }
        )
        if save_predictions:
            _save_prediction(results_dir / "cache" / sanitize_model_name(model_config.name) / f"fold_{fold}", inference)

    output_path = _metrics_output_path(results_dir, model_config.name, fold)
    _atomic_write_csv(output_path, rows, METRIC_FIELDS)
    summary = summarize_metric_rows(rows, model_config.name, fold)
    summary_path = _summary_output_path(results_dir, model_config.name, fold)
    _atomic_write_csv(summary_path, [summary], SUMMARY_FIELDS)
    source_summary_path = _source_summary_output_path(results_dir, model_config.name, fold)
    _atomic_write_csv(source_summary_path, summarize_source_metric_rows(rows, model_config.name, fold), SOURCE_SUMMARY_FIELDS)
    print(
        f"[metrics] saved={output_path} summary={summary_path} "
        f"sample_mean_dice={summary['mean_sample_dice']:.5f} global_dice={summary['global_dice']:.5f}"
    )
    return output_path


def _save_prediction(folder: Path, result: InferenceResult) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{result.sample_id}.npz", pred=result.pred.astype(np.uint8), gt=result.gt.astype(np.uint8))


def _fusion_late_config() -> ModelConfig:
    for model_config in get_model_configs():
        if model_config.name == "Fusion_Late":
            return model_config
    raise KeyError("Fusion_Late config is missing from get_model_configs().")


def _fusion_model_configs(fusion_config: ModelConfig | None = None) -> tuple[ModelConfig, ModelConfig]:
    fusion_config = fusion_config or _fusion_late_config()
    if fusion_config.fusion_2d_dir is None or fusion_config.fusion_3d_dir is None:
        raise ValueError("Fusion_Late must define fusion_2d_dir and fusion_3d_dir.")
    return (
        ModelConfig(
            name="Fusion_Late_2D_Unet3+",
            stage="train_2d",
            encoder_2d="unet3plus",
            slice_mode=fusion_config.slice_mode or "uniform",
            checkpoint_dir=Path(fusion_config.fusion_2d_dir),
        ),
        ModelConfig(
            name="Fusion_Late_3D_Unet3+",
            stage="train_3d",
            encoder_3d="unet3plus3d",
            decoder_model="unet3plus3d",
            decoder_style="full_scale",
            checkpoint_dir=Path(fusion_config.fusion_3d_dir),
        ),
    )


def _validate_fusion_components(model_2d: BuiltModel, model_3d: BuiltModel) -> None:
    if model_2d.model_type != "2D":
        raise RuntimeError(f"Fusion_Late expected a 2D component, got {model_2d.model_type} from {model_2d.checkpoint_path}.")
    if model_3d.model_type != "3D":
        raise RuntimeError(f"Fusion_Late expected a 3D component, got {model_3d.model_type} from {model_3d.checkpoint_path}.")


def _validation_row(
    model_config: ModelConfig,
    fold: int,
    component: str,
    checkpoint_dir: Path,
    checkpoint_path: Path | None = None,
    built: BuiltModel | None = None,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    cfg = built.cfg if built is not None else {}
    return {
        "model_name": model_config.name,
        "component": component,
        "fold": int(fold),
        "status": status,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "stage": get_nested(cfg, "experiment.stage", ""),
        "model_type": "" if built is None else built.model_type,
        "encoder_2d": get_nested(cfg, "model.encoder_2d.type", ""),
        "encoder_3d": get_nested(cfg, "model.encoder_3d.type", ""),
        "decoder_model": get_nested(cfg, "model.decoder.model", get_nested(cfg, "model.decoder_3d.model", "")),
        "decoder_style": get_nested(cfg, "model.decoder.style", get_nested(cfg, "model.decoder_3d.style", "")),
        "slice_mode": get_nested(cfg, "model.slice_selection.mode", ""),
        "proposal_order": get_nested(cfg, "model.slice_selection.proposal.selection_order", "closest(default)"),
        "build_name": "" if built is None else built.build_name,
        "backbone": "" if built is None else built.backbone,
        "arch_encoder_2d": _architecture_value(built, "encoder_2d_type", "encoder_2d"),
        "arch_encoder_3d": _architecture_value(built, "encoder_3d_type", "encoder_3d", "encoder_type"),
        "arch_decoder_model": _architecture_value(built, "decoder_model"),
        "arch_decoder_style": _architecture_value(built, "decoder_style"),
        "arch_slice_mode": get_nested(getattr(built.model, "architecture_config", {}) if built is not None else {}, "slice_selection.mode", ""),
        "arch_fusion": _architecture_value(built, "encoder_fusion_mode"),
        "params": _parameter_count(None if built is None else built.model),
        "checkpoint_epoch": "" if built is None else built.checkpoint_epoch,
        "checkpoint_best_metric": "" if built is None else built.checkpoint_best_metric,
        "error": error,
    }


def _validate_one_model_checkpoint(model_config: ModelConfig, fold: int, component: str, device: torch.device) -> dict[str, Any]:
    checkpoint_path: Path | None = None
    try:
        checkpoint_path = find_checkpoint(model_config.checkpoint_dir, fold, require_fold_specific=True)
        built = build_model_by_name(model_config.name, checkpoint_path, device, fallback_config=model_config)
        return _validation_row(
            model_config=model_config,
            fold=fold,
            component=component,
            checkpoint_dir=model_config.checkpoint_dir,
            checkpoint_path=checkpoint_path,
            built=built,
        )
    except Exception as error:
        return _validation_row(
            model_config=model_config,
            fold=fold,
            component=component,
            checkpoint_dir=model_config.checkpoint_dir,
            checkpoint_path=checkpoint_path,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )


def run_validate(folds: Sequence[int], results_dir: Path) -> Path:
    device = torch.device("cpu")
    rows: list[dict[str, Any]] = []
    for model_config in get_model_configs():
        if model_config.is_fusion:
            cfg_2d, cfg_3d = _fusion_model_configs(model_config)
            for fold in folds:
                row_2d = _validate_one_model_checkpoint(cfg_2d, int(fold), "fusion_2d", device)
                row_2d["model_name"] = "Fusion_Late"
                rows.append(row_2d)
                row_3d = _validate_one_model_checkpoint(cfg_3d, int(fold), "fusion_3d", device)
                row_3d["model_name"] = "Fusion_Late"
                rows.append(row_3d)
            continue
        for fold in folds:
            rows.append(_validate_one_model_checkpoint(model_config, int(fold), "main", device))

    output_path = results_dir / "metrics" / "checkpoint_validation.csv"
    _atomic_write_csv(output_path, rows, VALIDATION_FIELDS)
    errors = [row for row in rows if row.get("status") != "ok"]
    for row in rows:
        print(
            f"[validate] status={row['status']} model={row['model_name']} component={row['component']} "
            f"fold={row['fold']} checkpoint={row['checkpoint_path'] or row['checkpoint_dir']}"
        )
        if row.get("error"):
            print(f"[validate] error={row['error']}")
    print(f"[validate] saved={output_path} rows={len(rows)} errors={len(errors)}")
    return output_path


def evaluate_fusion_late_fold(fold: int, device: torch.device, results_dir: Path, num_workers: int = 0, save_predictions: bool = False) -> Path:
    cfg_2d, cfg_3d = _fusion_model_configs()
    ckpt_2d = find_checkpoint(cfg_2d.checkpoint_dir, fold)
    ckpt_3d = find_checkpoint(cfg_3d.checkpoint_dir, fold)
    print(f"[fusion_late] fold={fold} device={device} checkpoint_2d={ckpt_2d}")
    print(f"[fusion_late] fold={fold} device={device} checkpoint_3d={ckpt_3d}")
    model_2d = build_model_by_name(cfg_2d.name, ckpt_2d, device, fallback_config=cfg_2d)
    model_3d = build_model_by_name(cfg_3d.name, ckpt_3d, device, fallback_config=cfg_3d)
    _validate_fusion_components(model_2d, model_3d)

    loader_2d, dataset_2d = build_dataloader_for_fold(model_2d.cfg, fold, EVALUATION_SPLIT, "2D", num_workers=num_workers)
    loader_3d, _dataset_3d = build_dataloader_for_fold(model_3d.cfg, fold, EVALUATION_SPLIT, "3D", num_workers=num_workers)
    samples_3d = {_text_value(_as_single_sample(sample).get("case_id", "")): sample for sample in _iter_dataset(loader_3d)}
    print(f"[fusion_late] fold={fold} split={EVALUATION_SPLIT} full_volume=true samples={len(dataset_2d)}")

    rows: list[dict[str, Any]] = []
    slice_batch_size = int(get_nested(model_2d.cfg, "evaluation.slice_batch_size", 8))
    for sample_2d in _iter_dataset(loader_2d):
        single = _as_single_sample(sample_2d)
        sample_id = _text_value(single.get("case_id", "unknown"))
        sample_3d = samples_3d.get(sample_id)
        if sample_3d is None:
            raise KeyError(f"Fusion_Late could not find matching 3D sample for {sample_id} in fold {fold}.")
        inference = run_late_fusion_inference(model_2d, model_3d, sample_2d, sample_3d, device, slice_batch_size=slice_batch_size)
        metrics = compute_sample_metrics(inference.pred, inference.gt)
        rows.append(
            {
                "fold": fold,
                "sample_id": inference.sample_id,
                "source": inference.source,
                "model_name": "Fusion_Late",
                **metrics,
                "checkpoint_path": f"2D:{ckpt_2d};3D:{ckpt_3d}",
            }
        )
        if save_predictions:
            _save_prediction(results_dir / "fusion_late" / "predictions" / f"fold_{fold}", inference)

    output_path = _metrics_output_path(results_dir, "Fusion_Late", fold)
    _atomic_write_csv(output_path, rows, METRIC_FIELDS)
    fusion_path = results_dir / "fusion_late" / f"fold_{fold}_per_sample_metrics.csv"
    _atomic_write_csv(fusion_path, rows, METRIC_FIELDS)
    summary = summarize_metric_rows(rows, "Fusion_Late", fold)
    summary_path = _summary_output_path(results_dir, "Fusion_Late", fold)
    _atomic_write_csv(summary_path, [summary], SUMMARY_FIELDS)
    fusion_summary_path = results_dir / "fusion_late" / f"fold_{fold}_test_summary.csv"
    _atomic_write_csv(fusion_summary_path, [summary], SUMMARY_FIELDS)
    source_summaries = summarize_source_metric_rows(rows, "Fusion_Late", fold)
    source_summary_path = _source_summary_output_path(results_dir, "Fusion_Late", fold)
    _atomic_write_csv(source_summary_path, source_summaries, SOURCE_SUMMARY_FIELDS)
    fusion_source_summary_path = results_dir / "fusion_late" / f"fold_{fold}_source_summary.csv"
    _atomic_write_csv(fusion_source_summary_path, source_summaries, SOURCE_SUMMARY_FIELDS)
    print(
        f"[fusion_late] saved={output_path} mirror={fusion_path} "
        f"sample_mean_dice={summary['mean_sample_dice']:.5f} global_dice={summary['global_dice']:.5f}"
    )
    return output_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_model_metrics(results_dir: Path, model_name: str, folds: Sequence[int]) -> Path:
    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    source_fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        path = _metrics_output_path(results_dir, model_name, int(fold))
        if not path.exists():
            raise FileNotFoundError(f"Missing per-fold metrics CSV: {path}")
        fold_rows = _read_csv(path)
        rows.extend(fold_rows)
        fold_summary = summarize_metric_rows(fold_rows, model_name, int(fold))
        fold_summaries.append(fold_summary)
        _atomic_write_csv(_summary_output_path(results_dir, model_name, int(fold)), [fold_summary], SUMMARY_FIELDS)
        source_summaries = summarize_source_metric_rows(fold_rows, model_name, int(fold))
        source_fold_summaries.extend(source_summaries)
        _atomic_write_csv(_source_summary_output_path(results_dir, model_name, int(fold)), source_summaries, SOURCE_SUMMARY_FIELDS)
    output_path = results_dir / "metrics" / sanitize_model_name(model_name) / "all_folds_per_sample_metrics.csv"
    _atomic_write_csv(output_path, rows, METRIC_FIELDS)
    summary = summarize_metric_rows(rows, model_name, "all")
    summary_path = results_dir / "metrics" / sanitize_model_name(model_name) / "all_folds_test_summary.csv"
    _atomic_write_csv(summary_path, [summary], SUMMARY_FIELDS)
    kfold_summary = summarize_kfold_style(fold_summaries, model_name)
    kfold_summary_path = results_dir / "metrics" / sanitize_model_name(model_name) / "kfold_test_summary.csv"
    _atomic_write_csv(kfold_summary_path, [kfold_summary], KFOLD_SUMMARY_FIELDS)
    source_summary = summarize_source_metric_rows(rows, model_name, "all")
    source_summary_path = results_dir / "metrics" / sanitize_model_name(model_name) / "all_folds_source_summary.csv"
    _atomic_write_csv(source_summary_path, source_summary, SOURCE_SUMMARY_FIELDS)
    source_kfold_summary = summarize_source_kfold_style(source_fold_summaries, model_name)
    source_kfold_summary_path = results_dir / "metrics" / sanitize_model_name(model_name) / "source_kfold_test_summary.csv"
    _atomic_write_csv(source_kfold_summary_path, source_kfold_summary, SOURCE_KFOLD_SUMMARY_FIELDS)
    if model_name == "Fusion_Late":
        fusion_path = results_dir / "fusion_late" / "all_folds_per_sample_metrics.csv"
        _atomic_write_csv(fusion_path, rows, METRIC_FIELDS)
        fusion_summary_path = results_dir / "fusion_late" / "all_folds_test_summary.csv"
        _atomic_write_csv(fusion_summary_path, [summary], SUMMARY_FIELDS)
        fusion_kfold_summary_path = results_dir / "fusion_late" / "kfold_test_summary.csv"
        _atomic_write_csv(fusion_kfold_summary_path, [kfold_summary], KFOLD_SUMMARY_FIELDS)
        fusion_source_summary_path = results_dir / "fusion_late" / "all_folds_source_summary.csv"
        _atomic_write_csv(fusion_source_summary_path, source_summary, SOURCE_SUMMARY_FIELDS)
        fusion_source_kfold_summary_path = results_dir / "fusion_late" / "source_kfold_test_summary.csv"
        _atomic_write_csv(fusion_source_kfold_summary_path, source_kfold_summary, SOURCE_KFOLD_SUMMARY_FIELDS)
    return output_path


def aggregate_all_metrics(results_dir: Path, folds: Sequence[int] | None = None, model_names: Sequence[str] | None = None) -> Path:
    if model_names is None:
        model_names = [item.name for item in get_model_configs()]
    else:
        model_names = [str(item) for item in model_names]
    if folds is None:
        folds = [0, 1, 2]
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    kfold_summary_rows: list[dict[str, Any]] = []
    source_summary_rows: list[dict[str, Any]] = []
    source_kfold_summary_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        aggregate_model_metrics(results_dir, model_name, folds)
        model_rows = _read_csv(results_dir / "metrics" / sanitize_model_name(model_name) / "all_folds_per_sample_metrics.csv")
        rows.extend(model_rows)
        summary_rows.append(summarize_metric_rows(model_rows, model_name, "all"))
        kfold_summary_rows.extend(_read_csv(results_dir / "metrics" / sanitize_model_name(model_name) / "kfold_test_summary.csv"))
        source_summary_rows.extend(_read_csv(results_dir / "metrics" / sanitize_model_name(model_name) / "all_folds_source_summary.csv"))
        source_kfold_summary_rows.extend(_read_csv(results_dir / "metrics" / sanitize_model_name(model_name) / "source_kfold_test_summary.csv"))
    output_path = results_dir / "metrics" / "all_folds_per_sample_metrics.csv"
    _atomic_write_csv(output_path, rows, METRIC_FIELDS)
    summary_path = results_dir / "metrics" / "test_summary_by_model.csv"
    _atomic_write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    kfold_summary_path = results_dir / "metrics" / "kfold_test_summary_by_model.csv"
    _atomic_write_csv(kfold_summary_path, kfold_summary_rows, KFOLD_SUMMARY_FIELDS)
    source_summary_path = results_dir / "metrics" / "source_summary_by_model.csv"
    _atomic_write_csv(source_summary_path, source_summary_rows, SOURCE_SUMMARY_FIELDS)
    source_kfold_summary_path = results_dir / "metrics" / "source_kfold_summary_by_model.csv"
    _atomic_write_csv(source_kfold_summary_path, source_kfold_summary_rows, SOURCE_KFOLD_SUMMARY_FIELDS)
    print(f"[aggregate] saved={output_path} rows={len(rows)}")
    print(f"[aggregate] saved={summary_path} rows={len(summary_rows)}")
    print(f"[aggregate] saved={kfold_summary_path} rows={len(kfold_summary_rows)}")
    print(f"[aggregate] saved={source_summary_path} rows={len(source_summary_rows)}")
    print(f"[aggregate] saved={source_kfold_summary_path} rows={len(source_kfold_summary_rows)}")
    return output_path


def available_metric_model_names(results_dir: Path, folds: Sequence[int] | None = None) -> list[str]:
    folds = [0, 1, 2] if folds is None else [int(item) for item in folds]
    names: list[str] = []
    for model_config in get_model_configs():
        if any(_metrics_output_path(results_dir, model_config.name, fold).exists() for fold in folds):
            names.append(model_config.name)
    return names


def compute_normal_ci(dice_values: Sequence[float]) -> dict[str, float]:
    values = np.asarray([float(item) for item in dice_values], dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "se": math.nan, "low": math.nan, "high": math.nan}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else math.nan
    return {"n": n, "mean": mean, "std": std, "se": se, "low": mean - 1.96 * se, "high": mean + 1.96 * se}


def compute_bootstrap_ci(dice_values: Sequence[float], n_bootstrap: int = 1000, seed: int = 42) -> tuple[dict[str, float], list[float]]:
    values = np.asarray([float(item) for item in dice_values], dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"mean": math.nan, "low": math.nan, "high": math.nan, "n_bootstrap": int(n_bootstrap)}, []
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(sample.mean()))
    array = np.asarray(means, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "low": float(np.percentile(array, 2.5)),
        "high": float(np.percentile(array, 97.5)),
        "n_bootstrap": int(n_bootstrap),
    }, means


def save_bootstrap_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_csv(path, rows, ["model_name", "bootstrap_id", "n", "mean_dice"])


def compute_and_save_ci_tables(results_dir: Path, n_bootstrap: int = 1000, seed: int = 42) -> None:
    global_path = results_dir / "metrics" / "all_folds_per_sample_metrics.csv"
    kfold_summary_path = results_dir / "metrics" / "kfold_test_summary_by_model.csv"
    if not global_path.exists():
        aggregate_all_metrics(results_dir, model_names=available_metric_model_names(results_dir) or None)
    elif not kfold_summary_path.exists():
        existing_rows = _read_csv(global_path)
        existing_model_names = sorted({row["model_name"] for row in existing_rows if row.get("model_name")})
        existing_folds = sorted({int(float(row["fold"])) for row in existing_rows if row.get("fold") not in {None, ""}})
        aggregate_all_metrics(results_dir, folds=existing_folds or None, model_names=existing_model_names or None)
    rows = _read_csv(global_path)
    by_model: dict[str, list[float]] = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(float(row["dice"]))

    ci_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    for model_index, model_name in enumerate(sorted(by_model)):
        values = by_model[model_name]
        normal = compute_normal_ci(values)
        bootstrap, means = compute_bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=int(seed) + model_index)
        ci_rows.append(
            {
                "model_name": model_name,
                "n": normal["n"],
                "mean_dice": normal["mean"],
                "std_dice": normal["std"],
                "standard_error": normal["se"],
                "normal_ci95_low": normal["low"],
                "normal_ci95_high": normal["high"],
                "bootstrap_mean": bootstrap["mean"],
                "bootstrap_ci95_low": bootstrap["low"],
                "bootstrap_ci95_high": bootstrap["high"],
                "bootstrap_n": bootstrap["n_bootstrap"],
            }
        )
        boot_rows.extend({"model_name": model_name, "bootstrap_id": index, "n": normal["n"], "mean_dice": mean} for index, mean in enumerate(means))

    ci_path = results_dir / "metrics" / "dice_95ci_by_model.csv"
    boots_path = results_dir / "metrics" / f"boots_{int(n_bootstrap)}.csv"
    _atomic_write_csv(ci_path, ci_rows, CI_FIELDS)
    save_bootstrap_csv(boots_path, boot_rows)
    print(f"[ci] saved={ci_path}")
    print(f"[ci] saved={boots_path}")

    fusion_values = by_model.get("Fusion_Late", [])
    normal = compute_normal_ci(fusion_values)
    bootstrap, means = compute_bootstrap_ci(fusion_values, n_bootstrap=n_bootstrap, seed=seed)
    fusion_ci = [
        {
            "model_name": "Fusion_Late",
            "n": normal["n"],
            "mean_dice": normal["mean"],
            "std_dice": normal["std"],
            "standard_error": normal["se"],
            "normal_ci95_low": normal["low"],
            "normal_ci95_high": normal["high"],
            "bootstrap_mean": bootstrap["mean"],
            "bootstrap_ci95_low": bootstrap["low"],
            "bootstrap_ci95_high": bootstrap["high"],
            "bootstrap_n": bootstrap["n_bootstrap"],
        }
    ]
    fusion_dir = results_dir / "fusion_late"
    _atomic_write_csv(fusion_dir / "fusion_late_95ci.csv", fusion_ci, CI_FIELDS)
    save_bootstrap_csv(
        fusion_dir / f"boots_{int(n_bootstrap)}.csv",
        [{"model_name": "Fusion_Late", "bootstrap_id": index, "n": normal["n"], "mean_dice": mean} for index, mean in enumerate(means)],
    )
    print(f"[ci] saved={fusion_dir / 'fusion_late_95ci.csv'}")


def compute_and_save_fusion_late_ci(results_dir: Path, n_bootstrap: int = 1000, seed: int = 42) -> None:
    fusion_path = results_dir / "fusion_late" / "all_folds_per_sample_metrics.csv"
    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing Fusion_Late all-fold metrics CSV: {fusion_path}")
    rows = _read_csv(fusion_path)
    values = [float(row["dice"]) for row in rows if row.get("model_name") == "Fusion_Late"]
    normal = compute_normal_ci(values)
    bootstrap, means = compute_bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)
    fusion_dir = results_dir / "fusion_late"
    _atomic_write_csv(
        fusion_dir / "fusion_late_95ci.csv",
        [
            {
                "model_name": "Fusion_Late",
                "n": normal["n"],
                "mean_dice": normal["mean"],
                "std_dice": normal["std"],
                "standard_error": normal["se"],
                "normal_ci95_low": normal["low"],
                "normal_ci95_high": normal["high"],
                "bootstrap_mean": bootstrap["mean"],
                "bootstrap_ci95_low": bootstrap["low"],
                "bootstrap_ci95_high": bootstrap["high"],
                "bootstrap_n": bootstrap["n_bootstrap"],
            }
        ],
        CI_FIELDS,
    )
    save_bootstrap_csv(
        fusion_dir / f"boots_{int(n_bootstrap)}.csv",
        [{"model_name": "Fusion_Late", "bootstrap_id": index, "n": normal["n"], "mean_dice": mean} for index, mean in enumerate(means)],
    )
    print(f"[ci] saved={fusion_dir / 'fusion_late_95ci.csv'}")
    print(f"[ci] saved={fusion_dir / f'boots_{int(n_bootstrap)}.csv'}")


def _row_fold(row: Mapping[str, Any]) -> int:
    try:
        return int(float(row.get("fold", -1)))
    except (TypeError, ValueError):
        return -1


def _sample_score(row: Mapping[str, Any]) -> float:
    dice = _numeric_value(row, "dice")
    iou = _numeric_value(row, "iou")
    if not (math.isfinite(dice) and math.isfinite(iou)):
        return math.nan
    return float(dice + iou)


def _valid_visual_candidate(row: Mapping[str, Any]) -> bool:
    score = _sample_score(row)
    gt_voxels = _numeric_value(row, "gt_voxels", 0.0)
    return math.isfinite(score) and score > 0.0 and gt_voxels > 0.0


def select_top_ours_samples(metrics_df: Sequence[Mapping[str, Any]], fold: int, k: int = 10, require_different_source: bool = False) -> list[dict[str, Any]]:
    ours = [
        dict(row)
        for row in metrics_df
        if _row_fold(row) == int(fold) and str(row.get("model_name")) == "Ours" and _valid_visual_candidate(row)
    ]
    ours.sort(key=_sample_score, reverse=True)
    if len(ours) <= k:
        return ours
    if not require_different_source:
        return ours[:k]

    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in ours:
        source = str(row.get("source", "unknown"))
        if source not in seen_sources:
            selected.append(row)
            seen_sources.add(source)
        if len(selected) >= k:
            return selected
    print(f"WARNING: fold {fold}: could not choose {k} top Ours samples from different sources; using top-{k}.")
    return ours[:k]


def load_sample_by_id(sample_id: str, fold: int, cfg: Mapping[str, Any], model_type: str, num_workers: int = 0) -> Mapping[str, Any]:
    loader, _dataset = build_dataloader_for_fold(cfg, fold, EVALUATION_SPLIT, model_type, num_workers=num_workers)
    return _find_sample(loader, sample_id)


def _resize_mask_to_shape(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    mask = np.asarray(mask)
    target = tuple(int(item) for item in shape)
    if tuple(mask.shape) == target:
        return mask
    factors = [target_dim / max(1, current_dim) for target_dim, current_dim in zip(target, mask.shape)]
    return zoom(mask.astype(np.float32), factors, order=0) > 0.5


def _resize_image_to_shape(image: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    target = tuple(int(item) for item in shape)
    if tuple(image.shape) == target:
        return image
    factors = [target_dim / max(1, current_dim) for target_dim, current_dim in zip(target, image.shape)]
    return zoom(image, factors, order=1).astype(np.float32)


def _normalise_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _overlay_slice(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    base = np.stack([image, image, image], axis=-1)
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    mask_bool = mask.astype(bool)
    output = base.copy()
    output[mask_bool] = (1.0 - alpha) * output[mask_bool] + alpha * red[mask_bool]
    return output


def _projection_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32).max(axis=0)


def _projection_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).max(axis=0).astype(np.float32)


def _display_plane(
    image_volume: np.ndarray,
    mask_volume: np.ndarray,
    z_index: int,
    sample_id: str,
    sample_index: int | None,
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    if str(view) == "mip":
        image_plane = _projection_image(image_volume)
        mask_plane = _projection_mask(mask_volume)
    else:
        image_plane = np.asarray(image_volume)[int(z_index)]
        mask_plane = np.asarray(mask_volume)[int(z_index)]
    return (
        _orient_display_slice(image_plane, sample_id=sample_id, sample_index=sample_index),
        _orient_display_slice(mask_plane, sample_id=sample_id, sample_index=sample_index),
    )


def _view_text(view: str, z_index: int) -> str:
    return "MIP" if str(view) == "mip" else f"slice={int(z_index)}"


DISPLAY_ROTATION_K = -1
DISPLAY_EXTRA_ROTATION_K_BY_SAMPLE_INDEX = {
    0: 2,
}
DISPLAY_EXTRA_ROTATION_K_BY_SAMPLE_ID: dict[str, int] = {}


def _orient_display_slice(array: np.ndarray, sample_id: str, sample_index: int | None = None) -> np.ndarray:
    rotation_k = DISPLAY_ROTATION_K
    if sample_index is not None:
        rotation_k += DISPLAY_EXTRA_ROTATION_K_BY_SAMPLE_INDEX.get(int(sample_index), 0)
    rotation_k += DISPLAY_EXTRA_ROTATION_K_BY_SAMPLE_ID.get(str(sample_id), 0)
    return np.ascontiguousarray(np.rot90(np.asarray(array), k=rotation_k, axes=(0, 1)))


def _best_slice_index(gt: np.ndarray, ours_pred: np.ndarray | None = None) -> int:
    gt = np.asarray(gt) > 0
    if gt.any():
        return int(np.argmax(gt.reshape(gt.shape[0], -1).sum(axis=1)))
    if ours_pred is not None and np.asarray(ours_pred).any():
        pred = np.asarray(ours_pred) > 0
        return int(np.argmax(pred.reshape(pred.shape[0], -1).sum(axis=1)))
    return int(gt.shape[0] // 2)


def _metric_lookup(metrics_rows: Sequence[Mapping[str, Any]], fold: int, sample_id: str) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in metrics_rows:
        if _row_fold(row) == int(fold) and str(row.get("sample_id")) == str(sample_id):
            lookup[str(row.get("model_name"))] = {
                "dice": _numeric_value(row, "dice"),
                "iou": _numeric_value(row, "iou"),
                "score": _sample_score(row),
            }
    return lookup


def _metric_text(metrics: Mapping[str, float] | None, percent: bool = False) -> str:
    if not metrics:
        return ""
    dice = float(metrics.get("dice", math.nan))
    iou = float(metrics.get("iou", math.nan))
    score = float(metrics.get("score", math.nan))
    if percent:
        parts = []
        if math.isfinite(dice):
            parts.append(f"D {dice * 100:.2f}%")
        if math.isfinite(iou):
            parts.append(f"I {iou * 100:.2f}%")
        return "\n".join(parts)
    parts = []
    if math.isfinite(dice):
        parts.append(f"Dice={dice:.4f}")
    if math.isfinite(iou):
        parts.append(f"IoU={iou:.4f}")
    if math.isfinite(score):
        parts.append(f"S={score:.4f}")
    return "  ".join(parts)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "sample"


def _predict_for_visuals(model_config: ModelConfig, fold: int, sample_ids: Sequence[str], device: torch.device, num_workers: int) -> dict[str, InferenceResult]:
    if model_config.is_fusion:
        cfg_2d, cfg_3d = _fusion_model_configs(model_config)
        model_2d = build_model_by_name(cfg_2d.name, find_checkpoint(cfg_2d.checkpoint_dir, fold), device, fallback_config=cfg_2d)
        model_3d = build_model_by_name(cfg_3d.name, find_checkpoint(cfg_3d.checkpoint_dir, fold), device, fallback_config=cfg_3d)
        _validate_fusion_components(model_2d, model_3d)
        loader_2d, _ = build_dataloader_for_fold(model_2d.cfg, fold, EVALUATION_SPLIT, "2D", num_workers=num_workers)
        loader_3d, _ = build_dataloader_for_fold(model_3d.cfg, fold, EVALUATION_SPLIT, "3D", num_workers=num_workers)
        samples_2d = {_text_value(_as_single_sample(sample).get("case_id", "")): sample for sample in _iter_dataset(loader_2d)}
        samples_3d = {_text_value(_as_single_sample(sample).get("case_id", "")): sample for sample in _iter_dataset(loader_3d)}
        return {
            sample_id: run_late_fusion_inference(
                model_2d,
                model_3d,
                samples_2d[sample_id],
                samples_3d[sample_id],
                device,
                slice_batch_size=int(get_nested(model_2d.cfg, "evaluation.slice_batch_size", 8)),
            )
            for sample_id in sample_ids
        }

    checkpoint = find_checkpoint(model_config.checkpoint_dir, fold)
    built = build_model_by_name(model_config.name, checkpoint, device, fallback_config=model_config)
    loader, _ = build_dataloader_for_fold(built.cfg, fold, EVALUATION_SPLIT, built.model_type, num_workers=num_workers)
    samples = {_text_value(_as_single_sample(sample).get("case_id", "")): sample for sample in _iter_dataset(loader)}
    predictions: dict[str, InferenceResult] = {}
    for sample_id in sample_ids:
        sample = samples[sample_id]
        if built.model_type == "2D":
            predictions[sample_id] = run_2d_inference_slice_by_slice(
                built.model,
                sample,
                device,
                built.num_classes,
                slice_batch_size=int(get_nested(built.cfg, "evaluation.slice_batch_size", 8)),
            )
        else:
            predictions[sample_id] = run_3d_inference(built.model, sample, device, built.num_classes)
    return predictions


def plot_sample_visualization(
    axes: np.ndarray,
    sample_id: str,
    columns: Sequence[str],
    predictions: Mapping[str, Mapping[str, InferenceResult]],
    metric_values: Mapping[str, Mapping[str, float]],
    show_titles: bool,
    sample_index: int | None = None,
    view: str = "slice",
) -> None:
    gt_ref = predictions["Ours"][sample_id].gt
    image_ref = _normalise_display(_resize_image_to_shape(predictions["Ours"][sample_id].image, gt_ref.shape))
    ours_pred = _resize_mask_to_shape(predictions["Ours"][sample_id].pred, gt_ref.shape)
    z_index = _best_slice_index(gt_ref, ours_pred=ours_pred)
    ours_text = _metric_text(metric_values.get("Ours"))

    for column_index, name in enumerate(columns):
        top_ax = axes[0, column_index]
        bottom_ax = axes[1, column_index]
        if name == "GT":
            mask = gt_ref > 0
            title = "GT"
        else:
            mask = _resize_mask_to_shape(predictions[name][sample_id].pred, gt_ref.shape)
            title = name

        image_plane, mask_slice = _display_plane(image_ref, mask, z_index, sample_id, sample_index, view)
        overlay = _overlay_slice(image_plane, mask_slice, alpha=0.5)
        top_ax.imshow(np.zeros_like(mask_slice, dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        top_ax.imshow(np.ma.masked_where(~mask_slice.astype(bool), mask_slice), cmap="Reds", vmin=0, vmax=1)
        bottom_ax.imshow(overlay)

        if show_titles:
            top_ax.set_title(title, fontsize=11, pad=4)
        if column_index == 0:
            sample_text = f"{sample_id} | {_view_text(view, z_index)}"
            if ours_text:
                sample_text = f"{sample_text} | Ours {ours_text}"
            top_ax.text(
                0.0,
                1.12,
                sample_text,
                transform=top_ax.transAxes,
                ha="left",
                va="bottom",
                color="black",
                fontsize=8,
                fontweight="bold",
                clip_on=False,
            )
        if name != "GT":
            metric_text = _metric_text(metric_values.get(name), percent=True)
            if metric_text:
                top_ax.text(
                    0.5,
                    0.03,
                    metric_text,
                    transform=top_ax.transAxes,
                    ha="center",
                    va="bottom",
                    color="yellow",
                    fontsize=10,
                    fontweight="bold",
                )
        for ax in (top_ax, bottom_ax):
            ax.set_axis_off()


VISUAL_ASSET_FIELDS = [
    "fold",
    "view",
    "rank",
    "sample_id",
    "source",
    "slice_index",
    "model_name",
    "dice",
    "iou",
    "score",
    "img_path",
    "gt_path",
    "predict_path",
    "panel_path",
]


def _save_gray_png(plt, path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.asarray(image), cmap="gray", vmin=0.0, vmax=1.0)


def _save_panel_png(plt, path: Path, image_slice: np.ndarray, gt_slice: np.ndarray, pred_slice: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0), dpi=180)
    axes[0].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("img", fontsize=8)
    axes[1].imshow(_overlay_slice(image_slice, gt_slice, alpha=0.55))
    axes[1].set_title("gt", fontsize=8)
    axes[2].imshow(_overlay_slice(image_slice, pred_slice, alpha=0.55))
    axes[2].set_title("predict", fontsize=8)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def export_visual_assets(
    fold: int,
    sample_ids: Sequence[str],
    columns: Sequence[str],
    predictions: Mapping[str, Mapping[str, InferenceResult]],
    metrics_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    plt,
    view: str = "slice",
) -> Path:
    fold_root = output_root / f"fold_{int(fold)}"
    manifest_rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        gt_ref = predictions["Ours"][sample_id].gt
        image_ref = _normalise_display(_resize_image_to_shape(predictions["Ours"][sample_id].image, gt_ref.shape))
        ours_pred = _resize_mask_to_shape(predictions["Ours"][sample_id].pred, gt_ref.shape)
        z_index = _best_slice_index(gt_ref, ours_pred=ours_pred)
        metrics = _metric_lookup(metrics_rows, fold, sample_id)
        source = predictions["Ours"][sample_id].source
        file_stem = f"{sample_index + 1:02d}_{_safe_filename(sample_id)}_{str(view)}_z{z_index:03d}"
        image_slice, gt_slice = _display_plane(image_ref, gt_ref > 0, z_index, sample_id, sample_index, view)

        for model_name in [name for name in columns if name != "GT"]:
            model_dir = fold_root / sanitize_model_name(model_name)
            pred = _resize_mask_to_shape(predictions[model_name][sample_id].pred, gt_ref.shape)
            _, pred_slice = _display_plane(image_ref, pred.astype(np.float32), z_index, sample_id, sample_index, view)
            img_path = model_dir / "img" / f"{file_stem}.png"
            gt_path = model_dir / "gt" / f"{file_stem}.png"
            pred_path = model_dir / "predict" / f"{file_stem}.png"
            panel_path = model_dir / "panel" / f"{file_stem}.png"
            _save_gray_png(plt, img_path, image_slice)
            _save_gray_png(plt, gt_path, gt_slice)
            _save_gray_png(plt, pred_path, pred_slice)
            model_metrics = metrics.get(model_name, {})
            title = f"{model_name} | {sample_id} | {_view_text(view, z_index)}"
            metric_text = _metric_text(model_metrics)
            if metric_text:
                title = f"{title} | {metric_text}"
            _save_panel_png(plt, panel_path, image_slice, gt_slice, pred_slice, title)
            manifest_rows.append(
                {
                    "fold": int(fold),
                    "view": str(view),
                    "rank": sample_index + 1,
                    "sample_id": sample_id,
                    "source": source,
                    "slice_index": z_index,
                    "model_name": model_name,
                    "dice": model_metrics.get("dice", ""),
                    "iou": model_metrics.get("iou", ""),
                    "score": model_metrics.get("score", ""),
                    "img_path": str(img_path),
                    "gt_path": str(gt_path),
                    "predict_path": str(pred_path),
                    "panel_path": str(panel_path),
                }
            )

    manifest_path = fold_root / f"selected_visual_assets_{str(view)}.csv"
    _atomic_write_csv(manifest_path, manifest_rows, VISUAL_ASSET_FIELDS)
    print(f"[visualize] assets={fold_root} manifest={manifest_path}")
    return manifest_path


def create_visualization_pdf(
    fold: int,
    include_fusion_late: bool = True,
    results_dir: Path = Path("results"),
    device: torch.device | None = None,
    num_workers: int = 0,
    top_k: int = 10,
    export_assets: bool = True,
    visual_assets_dir: Path | None = None,
    view: str = "slice",
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as error:  # pragma: no cover - environment guard
        raise SystemExit("matplotlib is required for visualization PDFs.") from error

    metrics_path = results_dir / "metrics" / "all_folds_per_sample_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Please run --run metrics first or provide existing metrics CSV.")
    metrics_rows = _read_csv(metrics_path)
    selected = select_top_ours_samples(metrics_rows, fold, k=int(top_k), require_different_source=False)
    if len(selected) < int(top_k):
        print(f"WARNING: fold {fold}: only found {len(selected)} valid Ours samples for top-{int(top_k)} visualization.")
    sample_ids = [row["sample_id"] for row in selected]
    if not sample_ids:
        raise RuntimeError(f"No Ours samples found for fold {fold}.")

    device = device or _device_from_gpu(None)
    base_columns = ["GT", "Unet2D", "Unet++", "Unet3+", "Unet3D", "nnUNet"]
    columns = base_columns + (["Fusion_Late"] if include_fusion_late else []) + ["Ours"]
    model_configs = {item.name: item for item in get_model_configs()}
    predictions: dict[str, dict[str, InferenceResult]] = {}
    for model_name in [name for name in columns if name != "GT"]:
        print(f"[visualize] fold={fold} model={model_name} samples={sample_ids}")
        predictions[model_name] = _predict_for_visuals(model_configs[model_name], fold, sample_ids, device, num_workers)

    output_path = (
        results_dir
        / "visualizations"
        / f"fold_{fold}_top_{len(sample_ids)}_ours_dice_iou_{str(view)}_{'with' if include_fusion_late else 'without'}_fusion_late.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 2 * len(sample_ids)
    figsize = (max(8.0, 1.45 * len(columns)), max(4.0, 2.05 * len(sample_ids)))
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(rows, len(columns), figsize=figsize, dpi=300)
        if rows == 2:
            axes = np.asarray(axes).reshape(2, len(columns))
        for sample_index, sample_id in enumerate(sample_ids):
            row_axes = axes[sample_index * 2 : sample_index * 2 + 2, :]
            metric_values = _metric_lookup(metrics_rows, fold, sample_id)
            plot_sample_visualization(
                row_axes,
                sample_id=sample_id,
                columns=columns,
                predictions=predictions,
                metric_values=metric_values,
                show_titles=sample_index == 0,
                sample_index=sample_index,
                view=str(view),
            )
        fig.subplots_adjust(wspace=0.03, hspace=0.18)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    if export_assets:
        export_visual_assets(
            fold=int(fold),
            sample_ids=sample_ids,
            columns=columns,
            predictions=predictions,
            metrics_rows=metrics_rows,
            output_root=Path(visual_assets_dir) if visual_assets_dir is not None else PROJECT_ROOT / "visualize",
            plt=plt,
            view=str(view),
        )
    print(f"[visualize] saved={output_path}")
    return output_path


def _worker_evaluate_model(payload: tuple[ModelConfig, int, str, int | None, int, bool]) -> str:
    model_config, fold, results_dir, gpu_id, num_workers, save_predictions = payload
    set_seed(42 + int(fold), deterministic=True)
    device = _device_from_gpu(gpu_id)
    return str(evaluate_model_fold(model_config, fold, device, Path(results_dir), num_workers=num_workers, save_predictions=save_predictions))


def _worker_evaluate_fusion(payload: tuple[int, str, int | None, int, bool]) -> str:
    fold, results_dir, gpu_id, num_workers, save_predictions = payload
    set_seed(42 + int(fold), deterministic=True)
    device = _device_from_gpu(gpu_id)
    return str(evaluate_fusion_late_fold(fold, device, Path(results_dir), num_workers=num_workers, save_predictions=save_predictions))


def _visual_views(view: str) -> list[str]:
    key = str(view or "slice").lower()
    if key == "both":
        return ["slice", "mip"]
    if key in {"slice", "mip"}:
        return [key]
    raise ValueError(f"Unsupported visualization view: {view!r}")


def _worker_visualize_fold(payload: tuple[int, str, int | None, int, int, bool, str | None, str]) -> str:
    fold, results_dir, gpu_id, num_workers, top_k, export_assets, visual_assets_dir, view = payload
    set_seed(42 + int(fold), deterministic=True)
    device = _device_from_gpu(gpu_id)
    asset_root = None if visual_assets_dir is None else Path(visual_assets_dir)
    outputs: list[Path] = []
    for view_name in _visual_views(view):
        output_with = create_visualization_pdf(
            int(fold),
            include_fusion_late=True,
            results_dir=Path(results_dir),
            device=device,
            num_workers=num_workers,
            top_k=int(top_k),
            export_assets=bool(export_assets),
            visual_assets_dir=asset_root,
            view=view_name,
        )
        output_without = create_visualization_pdf(
            int(fold),
            include_fusion_late=False,
            results_dir=Path(results_dir),
            device=device,
            num_workers=num_workers,
            top_k=int(top_k),
            export_assets=False,
            visual_assets_dir=asset_root,
            view=view_name,
        )
        outputs.extend([output_with, output_without])
    return ";".join(str(path) for path in outputs)


def _gpu_for_job(index: int, gpus: Sequence[int] | None) -> int | None:
    if not gpus:
        return None
    return int(gpus[index % len(gpus)])


def _run_jobs(jobs: Sequence[tuple[Any, ...]], worker_fn, parallel: bool, max_workers: int | None = None) -> None:
    if not parallel:
        for job in jobs:
            worker_fn(job)
        return
    workers = max(1, min(len(jobs), int(max_workers or (os.cpu_count() or 1)))) if jobs else 1
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, job) for job in jobs]
        for future in as_completed(futures):
            print(f"[parallel] completed={future.result()}")


def run_metrics(folds: Sequence[int], results_dir: Path, gpus: Sequence[int] | None, parallel: bool, num_workers: int, include_fusion_late: bool, save_predictions: bool) -> None:
    base_models = [item for item in get_model_configs() if not item.is_fusion]
    jobs: list[tuple[Any, ...]] = []
    index = 0
    for model_config in base_models:
        for fold in folds:
            jobs.append((model_config, int(fold), str(results_dir), _gpu_for_job(index, gpus), int(num_workers), bool(save_predictions)))
            index += 1
    max_workers = len(gpus) if gpus else None
    _run_jobs(jobs, _worker_evaluate_model, parallel=parallel, max_workers=max_workers)
    if include_fusion_late:
        run_fusion_late(folds, results_dir, gpus, parallel, num_workers, save_predictions)
        model_names = [item.name for item in base_models] + ["Fusion_Late"]
    else:
        model_names = [item.name for item in base_models]
    aggregate_all_metrics(results_dir, folds=folds, model_names=model_names)


def run_fusion_late(folds: Sequence[int], results_dir: Path, gpus: Sequence[int] | None, parallel: bool, num_workers: int, save_predictions: bool) -> None:
    jobs = [
        (int(fold), str(results_dir), _gpu_for_job(index, gpus), int(num_workers), bool(save_predictions))
        for index, fold in enumerate(folds)
    ]
    max_workers = len(gpus) if gpus else None
    _run_jobs(jobs, _worker_evaluate_fusion, parallel=parallel, max_workers=max_workers)
    aggregate_model_metrics(results_dir, "Fusion_Late", folds)


def run_visualize(
    folds: Sequence[int],
    results_dir: Path,
    gpus: Sequence[int] | None,
    num_workers: int,
    parallel: bool,
    top_k: int,
    export_assets: bool,
    visual_assets_dir: Path | None,
    view: str,
) -> None:
    jobs = [
        (
            int(fold),
            str(results_dir),
            _gpu_for_job(index, gpus),
            int(num_workers),
            int(top_k),
            bool(export_assets),
            None if visual_assets_dir is None else str(visual_assets_dir),
            str(view),
        )
        for index, fold in enumerate(folds)
    ]
    max_workers = len(gpus) if gpus else None
    _run_jobs(jobs, _worker_visualize_fold, parallel=parallel, max_workers=max_workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Proposal_Model_Experiment checkpoints and create paper visualizations.")
    parser.add_argument("--run", choices=["all", "metrics", "visualize", "fusion_late", "ci", "validate"], default="all")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", "--output-root", dest="results_dir", type=str, default="results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--view", choices=["slice", "mip", "both"], default="slice")
    parser.add_argument("--visual-assets-dir", type=str, default=None)
    parser.add_argument("--no-export-assets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed), deterministic=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item) for item in args.folds]
    gpus = [int(item) for item in args.gpus] if args.gpus else None
    visual_assets_dir = Path(args.visual_assets_dir) if args.visual_assets_dir else PROJECT_ROOT / "visualize"
    export_assets = not bool(args.no_export_assets)
    print(f"[main] run={args.run} folds={folds} gpus={gpus} parallel={args.parallel} results_dir={results_dir}")

    if args.run == "metrics":
        run_metrics(folds, results_dir, gpus, args.parallel, args.num_workers, include_fusion_late=True, save_predictions=args.save_predictions)
    elif args.run == "validate":
        run_validate(folds, results_dir)
    elif args.run == "fusion_late":
        run_fusion_late(folds, results_dir, gpus, args.parallel, args.num_workers, save_predictions=args.save_predictions)
        compute_and_save_fusion_late_ci(results_dir, n_bootstrap=args.bootstrap, seed=args.seed)
    elif args.run == "ci":
        aggregate_all_metrics(results_dir, folds=folds, model_names=available_metric_model_names(results_dir, folds) or None)
        compute_and_save_ci_tables(results_dir, n_bootstrap=args.bootstrap, seed=args.seed)
    elif args.run == "visualize":
        run_visualize(folds, results_dir, gpus, args.num_workers, args.parallel, args.top_k, export_assets, visual_assets_dir, args.view)
    elif args.run == "all":
        run_metrics(folds, results_dir, gpus, args.parallel, args.num_workers, include_fusion_late=False, save_predictions=args.save_predictions)
        run_fusion_late(folds, results_dir, gpus, args.parallel, args.num_workers, save_predictions=args.save_predictions)
        aggregate_all_metrics(results_dir, folds=folds)
        compute_and_save_ci_tables(results_dir, n_bootstrap=args.bootstrap, seed=args.seed)
        run_visualize(folds, results_dir, gpus, args.num_workers, args.parallel, args.top_k, export_assets, visual_assets_dir, args.view)
    else:  # pragma: no cover - argparse protects this.
        raise ValueError(f"Unsupported --run value: {args.run}")


if __name__ == "__main__":
    main()
