from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import nibabel as nib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualize import main as viz  # noqa: E402


DEFAULT_SEGMAMBA_CHECKPOINT = Path(
    "outputs/segmamba/default/fold_01/checkpoint/best.pth"
)
DEFAULT_SWIN_UNETR_CHECKPOINT = Path(
    "outputs/swin_unetr/default/fold_01/checkpoint/last.pth"
)


@dataclass(frozen=True)
class BonusModel:
    display_name: str
    folder_name: str
    checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add SegMamba and Swin-UNETR predictions to an existing visualization fold."
    )
    parser.add_argument("--fold-dir", type=Path, default=Path(__file__).resolve().parent / "fold_1")
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Zero-based dataset fold. fold_01 checkpoints correspond to fold-index 0.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--segmamba-checkpoint", type=Path, default=DEFAULT_SEGMAMBA_CHECKPOINT)
    parser.add_argument("--swin-unetr-checkpoint", type=Path, default=DEFAULT_SWIN_UNETR_CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-missing-checkpoint", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _atomic_upsert_csv(
    path: Path,
    new_rows: list[Mapping[str, Any]],
    replaced_names: set[str],
) -> None:
    old_rows, old_fields = _read_csv(path)
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    def is_replaced(row: Mapping[str, Any]) -> bool:
        return str(row.get("model_name", "")).strip().lower().replace("-", "_") in replaced_names

    kept_rows = [row for row in old_rows if not is_replaced(row)]
    fields = list(old_fields)
    for row in new_rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in [*kept_rows, *new_rows]:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def _reference_rows(path: Path) -> list[dict[str, str]]:
    rows, _ = _read_csv(path)
    if not rows:
        return []
    ours = [row for row in rows if str(row.get("model_name", "")).strip().lower() == "ours"]
    candidates = ours or rows
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in candidates:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        view = str(row.get("view", path.stem.removeprefix("selected_visual_assets_")) or "slice")
        unique.setdefault((view, sample_id), row)
    return sorted(unique.values(), key=lambda row: (int(float(row.get("rank", 0) or 0)), row.get("sample_id", "")))


def _manifest_filename(reference: Mapping[str, Any], view: str, sample_id: str, rank: int, slice_index: int) -> str:
    for field in ("predict_path", "gt_path", "img_path", "panel_path"):
        value = str(reference.get(field, "")).strip()
        if value:
            return Path(value).name
    return f"{rank:02d}_{viz._safe_filename(sample_id)}_{view}_z{slice_index:03d}.png"


def _load_selected_samples(built: viz.BuiltModel, fold_index: int, split: str, sample_ids: set[str], num_workers: int):
    loader, _ = viz.build_dataloader_for_fold(
        built.cfg,
        fold_index,
        split,
        built.model_type,
        num_workers=num_workers,
    )
    selected: dict[str, Mapping[str, Any]] = {}
    for batch in loader:
        sample = viz._as_single_sample(batch)
        sample_id = viz._text_value(sample.get("case_id", ""))
        if sample_id in sample_ids:
            selected[sample_id] = batch
            if len(selected) == len(sample_ids):
                break
    missing = sorted(sample_ids - set(selected))
    if missing:
        raise KeyError(f"Selected samples were not found in fold {fold_index}, split={split}: {missing}")
    return selected


def _build(checkpoint: Path, display_name: str, device: torch.device) -> viz.BuiltModel:
    # SegMamba/Swin-UNETR are standalone baseline experiments. Their saved
    # configs correctly declare model.type=3D but do not use Proposal's
    # experiment.stage field, so stage validation must remain unset here.
    fallback = viz.ModelConfig(name=display_name, stage="", checkpoint_dir=checkpoint.parent)
    return viz.build_model_by_name(display_name, checkpoint, device, fallback_config=fallback)


def _resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _result_as_dhw(result: viz.InferenceResult, cfg: Mapping[str, Any]) -> viz.InferenceResult:
    """Match visualize/main.py's canonical display convention [D,H,W]."""
    if str(viz.get_nested(cfg, "training.volume_layout", "HWD")).upper() == "DHW":
        return result
    depth_axis = int(viz.get_nested(cfg, "training.depth_axis", viz.get_nested(cfg, "slice_2d.axis", 2)))
    return viz.InferenceResult(
        sample_id=result.sample_id,
        source=result.source,
        image=np.moveaxis(result.image, depth_axis, 0),
        gt=np.moveaxis(result.gt, depth_axis, 0),
        logits=result.logits,
        pred=np.moveaxis(result.pred, depth_axis, 0),
    )


def _source_depth(sample: Mapping[str, Any], cfg: Mapping[str, Any]) -> int | None:
    single = viz._as_single_sample(sample)
    image_path = Path(viz._text_value(single.get("image_path", "")))
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if not image_path.is_file():
        return None
    shape = nib.load(str(image_path)).shape
    depth_axis = int(viz.get_nested(cfg, "training.depth_axis", viz.get_nested(cfg, "slice_2d.axis", 2)))
    if depth_axis < 0 or depth_axis >= len(shape):
        return None
    return int(shape[depth_axis])


def _scale_depth_index(reference_index: int, reference_depth: int | None, model_depth: int) -> int:
    if model_depth <= 1:
        return 0
    if reference_depth is None or reference_depth <= 1:
        return int(np.clip(reference_index, 0, model_depth - 1))
    ratio = float(np.clip(reference_index, 0, reference_depth - 1)) / float(reference_depth - 1)
    return int(round(ratio * float(model_depth - 1)))


def _save_model_assets(
    model_spec: BonusModel,
    built: viz.BuiltModel,
    samples: Mapping[str, Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    fold_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_rows: list[dict[str, Any]] = []
    model_dir = fold_dir / model_spec.folder_name
    for reference in references:
        sample_id = str(reference["sample_id"])
        result = viz.run_3d_inference(built.model, samples[sample_id], device, built.num_classes)
        result = _result_as_dhw(result, built.cfg)
        view = str(reference.get("view", "slice") or "slice")
        rank = int(float(reference.get("rank", 0) or 0))
        reference_z = int(float(reference.get("slice_index", -1) or -1))
        raw_depth = _source_depth(samples[sample_id], built.cfg)
        if reference_z < 0:
            model_z = viz._best_slice_index(result.gt, ours_pred=result.pred)
            reference_z = model_z
        else:
            model_z = _scale_depth_index(reference_z, raw_depth, int(result.gt.shape[0]))

        image_volume = viz._normalise_display(viz._resize_image_to_shape(result.image, result.gt.shape))
        prediction = viz._resize_mask_to_shape(result.pred, result.gt.shape)
        filename = _manifest_filename(reference, view, sample_id, rank, reference_z)
        image_slice, gt_slice = viz._display_plane(image_volume, result.gt > 0, model_z, sample_id, rank - 1, view)
        _, pred_slice = viz._display_plane(image_volume, prediction > 0, model_z, sample_id, rank - 1, view)

        img_path = model_dir / "img" / filename
        gt_path = model_dir / "gt" / filename
        pred_path = model_dir / "predict" / filename
        panel_path = model_dir / "panel" / filename
        viz._save_gray_png(plt, img_path, image_slice)
        viz._save_gray_png(plt, gt_path, gt_slice)
        viz._save_gray_png(plt, pred_path, pred_slice)
        metrics = viz.compute_sample_metrics(prediction, result.gt)
        title = (
            f"{model_spec.display_name} | {sample_id} | ref_z={reference_z} model_z={model_z} | "
            f"Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f}"
        )
        viz._save_panel_png(plt, panel_path, image_slice, gt_slice, pred_slice, title)
        score = float(metrics["dice"]) + float(metrics["iou"])
        output_rows.append(
            {
                "fold": reference.get("fold", 1),
                "view": view,
                "rank": rank,
                "sample_id": sample_id,
                "source": result.source,
                "slice_index": reference_z,
                "model_name": model_spec.display_name,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "score": score if math.isfinite(score) else "",
                "img_path": str(img_path),
                "gt_path": str(gt_path),
                "predict_path": str(pred_path),
                "panel_path": str(panel_path),
            }
        )
        print(
            f"[{model_spec.display_name}] sample={sample_id} ref_z={reference_z}/{raw_depth} "
            f"model_z={model_z}/{result.gt.shape[0]} pred_volume={int(prediction.sum())} "
            f"pred_slice={int(np.asarray(pred_slice).sum())} saved={pred_path}"
        )
    return output_rows


def main() -> None:
    args = parse_args()
    # Checkpoint configs store dataset.root and split files as project-relative
    # paths (for example data/all_train.txt). Make them independent of the
    # directory from which this script is launched.
    os.chdir(PROJECT_ROOT)
    fold_dir = args.fold_dir.resolve()
    manifests = sorted(fold_dir.glob("selected_visual_assets_*.csv"))
    if not manifests:
        raise FileNotFoundError(f"No selected_visual_assets_*.csv found in {fold_dir}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_specs = [
        BonusModel("SegMamba", "SegMamba", args.segmamba_checkpoint),
        BonusModel("Swin_UNETR", "Swin_UNETR", args.swin_unetr_checkpoint),
    ]

    for manifest in manifests:
        references = _reference_rows(manifest)
        if not references:
            print(f"WARNING: no selected samples in {manifest}; skipped")
            continue
        sample_ids = {str(row["sample_id"]) for row in references}
        rows_to_add: list[dict[str, Any]] = []
        for spec in model_specs:
            checkpoint = _resolve_project_path(spec.checkpoint)
            if not checkpoint.is_file():
                message = f"Checkpoint not found for {spec.display_name}: {checkpoint}"
                if args.skip_missing_checkpoint:
                    print(f"WARNING: {message}; skipped")
                    continue
                raise FileNotFoundError(message)
            print(f"[{spec.display_name}] loading checkpoint={checkpoint} device={device}")
            built = _build(checkpoint, spec.display_name, device)
            samples = _load_selected_samples(
                built,
                int(args.fold_index),
                str(args.split),
                sample_ids,
                int(args.num_workers),
            )
            rows_to_add.extend(_save_model_assets(spec, built, samples, references, fold_dir, device))
            del built
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        replacement_names = {
            str(row.get("model_name", "")).strip().lower().replace("-", "_")
            for row in rows_to_add
        }
        _atomic_upsert_csv(manifest, rows_to_add, replacement_names)
        print(f"[CSV] upserted={manifest} new_rows={len(rows_to_add)} backup={manifest}.bak")


if __name__ == "__main__":
    main()
