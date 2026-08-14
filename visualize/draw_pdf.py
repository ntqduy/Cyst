from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "fold_1"
DEFAULT_OUTPUT_DIR = ROOT / "file"

SAMPLE_FILES = [
    #"01_NU28_slice_z013.png",
    "03_NYU0064_slice_z012.png",
    "06_MCF27_slice_z020.png",
    "07_MCF05_slice_z020.png",
]

# SAMPLE_FILES = [
#     #"01_NU28_slice_z013.png",
#     "01_NU28_mip_z013.png",
#     "03_NYU0064_mip_z012.png",
#     "07_MCF05_mip_z020.png",
#     "14_CAD291_mip_z015.png",
#     "15_IU66_mip_z015.png",
# ]
# Rotation per input file, in multiples of 90 degrees counter-clockwise.
# Examples:
#   0  = keep as-is
#   1  = rotate 90 degrees counter-clockwise
#   2  = rotate 180 degrees
#   3  = rotate 270 degrees counter-clockwise / 90 degrees clockwise
#  -1  = rotate 90 degrees clockwise
# ROTATE_K_BY_FILE = {
#     "01_NU28_mip_z013.png" : 2,
#     "03_NYU0064_mip_z012.png": 0,
#     "07_MCF05_mip_z020.png": 0,
#     "14_CAD291_mip_z015.png": 2,
#     "15_IU66_mip_z015.png": 0,
# }

ROTATE_K_BY_FILE = {
    "01_NU28_slice_z013.png": 0,
    "03_NYU0064_slice_z012.png": 0,
    "06_MCF27_slice_z020.png": 0,
    "07_MCF05_slice_z020.png": 0,
}

LAYOUTS = {
    "file_1_without_fusion_late.pdf": [
        "Ground Truth",
        "Unet2D",
        "Unet++",
        "Unet3+",
        "Unet3D",
        "nnUnet",
        "SegMamba",
        "Swin UNETR",
        "Ours",
    ],
    "file_2_with_fusion_late.pdf": [
        "Ground Truth",
        "Unet2D",
        "Unet++",
        "Unet3+",
        "Unet3D",
        "nnUnet",
        "SegMamba",
        "Swin UNETR",
        "Fusion_Late",
        "Ours",
    ],
}

MODEL_ALIASES = {
    "Unet2D": ["Unet2D", "UNet2D", "Unet", "UNet"],
    "Unet++": ["Unet++", "UNet++", "UnetPP", "UNetPP", "UnetPlusPlus", "UNetPlusPlus"],
    "Unet3+": ["Unet3+", "UNet3+", "Unet3Plus", "UNet3Plus", "Unet_3_plus", "UNet_3_plus"],
    "Unet3D": ["Unet3D", "UNet3D", "Unet_3D", "UNet_3D"],
    "nnUnet": ["nnUnet", "nnUNet", "nnunet", "nn_unet"],
    "SegMamba": ["SegMamba", "segmamba", "Seg_Mamba", "seg_mamba"],
    "Swin UNETR": ["Swin_UNETR", "SwinUNETR", "swin_unetr", "swin-unetr", "Swin UNETR"],
    "Fusion_Late": ["Fusion_Late", "FusionLate", "Fusion Late"],
    "Ours": ["Ours", "ours"],
}

IMAGE_DIRS = ["img", "image", "images", "input"]
GT_DIRS = ["gt", "mask", "masks", "label", "labels"]
PRED_DIRS = ["predict", "prediction", "pred", "preds", "mask", "masks"]
PANEL_DIRS = ["panel", "panels", ""]
GT_COLUMN = "Ground Truth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw fixed visualization PDFs from exported fold assets.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder like visualize/fold_1.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for generated PDFs.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.55, help="Mask overlay alpha.")
    parser.add_argument("--strict", action="store_true", help="Fail when an expected image or mask is missing.")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 2:
        return normalise_gray(array)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return np.asarray(array, dtype=np.float32) / 255.0


def read_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"))
    return array > 0


def normalise_gray(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return values
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def as_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 3:
        return np.clip(image, 0.0, 1.0)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def overlay_mask(image: np.ndarray, mask: np.ndarray | None, alpha: float = 0.55) -> np.ndarray:
    base = as_rgb(image).copy()
    if mask is None:
        return base
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape != base.shape[:2]:
        mask_bool = resize_mask(mask_bool, base.shape[:2])
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * red[mask_bool]
    return base


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape[:2]) == tuple(shape):
        return mask.astype(bool)
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil.resize((int(shape[1]), int(shape[0])), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def rotate_for_file(array: np.ndarray | None, filename: str) -> np.ndarray | None:
    if array is None:
        return None
    rotate_k = int(ROTATE_K_BY_FILE.get(filename, 0))
    if rotate_k == 0:
        return array
    return np.ascontiguousarray(np.rot90(np.asarray(array), k=rotate_k, axes=(0, 1)))


def candidate_model_dirs(input_dir: Path, model_name: str) -> list[Path]:
    aliases = MODEL_ALIASES.get(model_name, [model_name])
    candidates = []
    for alias in aliases:
        path = input_dir / alias
        if path.exists():
            candidates.append(path)
    return candidates


def find_named_file(folder: Path, subdirs: Iterable[str], filename: str) -> Path | None:
    for subdir in subdirs:
        path = folder / filename if not subdir else folder / subdir / filename
        if path.exists():
            return path
    for subdir in subdirs:
        search_root = folder if not subdir else folder / subdir
        if search_root.exists():
            matches = sorted(search_root.rglob(filename))
            if matches:
                return matches[0]
    return None


def find_model_asset(input_dir: Path, model_name: str, kind: str, filename: str) -> Path | None:
    if kind == "image":
        subdirs = IMAGE_DIRS
    elif kind == "gt":
        subdirs = GT_DIRS
    elif kind == "pred":
        subdirs = PRED_DIRS
    elif kind == "panel":
        subdirs = PANEL_DIRS
    else:
        raise ValueError(f"Unsupported asset kind: {kind}")

    for model_dir in candidate_model_dirs(input_dir, model_name):
        found = find_named_file(model_dir, subdirs, filename)
        if found is not None:
            return found
    return None


def find_shared_asset(input_dir: Path, kind: str, filename: str) -> Path | None:
    for model_name in ["Ours", "Unet3+", "Unet2D", "Unet++", "Unet3D", "nnUnet", "SegMamba", "Swin UNETR", "Fusion_Late"]:
        found = find_model_asset(input_dir, model_name, kind, filename)
        if found is not None:
            return found
    return None


def find_shared_image_and_gt(input_dir: Path, filename: str) -> tuple[Path | None, Path | None]:
    """Find an image/GT pair from the same architecture directory."""
    for model_name in ["Ours", "Unet3+", "Unet2D", "Unet++", "Unet3D", "nnUnet", "SegMamba", "Swin  UNETR", "Fusion_Late"]:
        image_path = find_model_asset(input_dir, model_name, "image", filename)
        gt_path = find_model_asset(input_dir, model_name, "gt", filename)
        if image_path is not None and gt_path is not None:
            return image_path, gt_path
    return find_shared_asset(input_dir, "image", filename), find_shared_asset(input_dir, "gt", filename)


def draw_missing(ax, label: str) -> None:
    ax.set_facecolor("#f4f4f4")
    ax.text(0.5, 0.5, f"MISSING\n{label}", ha="center", va="center", fontsize=8, color="crimson")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_cell(
    ax,
    image: np.ndarray | None,
    mask: np.ndarray | None,
    label: str,
    alpha: float,
    strict: bool,
    require_mask: bool = False,
) -> None:
    if image is None:
        if strict:
            raise FileNotFoundError(label)
        draw_missing(ax, label)
        return
    if require_mask and (mask is None or not np.asarray(mask).astype(bool).any()):
        if strict:
            raise ValueError(f"Missing or empty mask for {label}")
        draw_missing(ax, f"missing/empty mask: {label}")
        return
    ax.imshow(overlay_mask(image, mask, alpha=alpha))
    ax.set_xticks([])
    ax.set_yticks([])


def draw_layout(input_dir: Path, output_path: Path, columns: list[str], alpha: float, dpi: int, strict: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = len(SAMPLE_FILES)
    fig_width = max(8.0, 1.55 * len(columns))
    fig_height = max(5.0, 1.7 * rows)
    fig, axes = plt.subplots(rows, len(columns), figsize=(fig_width, fig_height), dpi=dpi)
    axes = np.asarray(axes).reshape(rows, len(columns))

    for col_idx, column in enumerate(columns):
        axes[0, col_idx].set_title(column, fontsize=11, pad=6)

    for row_idx, filename in enumerate(SAMPLE_FILES):
        image_path, gt_path = find_shared_image_and_gt(input_dir, filename)
        shared_image = rotate_for_file(read_image(image_path), filename) if image_path is not None else None
        gt_mask = rotate_for_file(read_mask(gt_path), filename) if gt_path is not None else None
        gt_pixels = int(np.count_nonzero(gt_mask)) if gt_mask is not None else 0
        print(f"[GT] file={filename} image={image_path} mask={gt_path} positive_pixels={gt_pixels}")

        for col_idx, column in enumerate(columns):
            ax = axes[row_idx, col_idx]
            if column == GT_COLUMN:
                draw_cell(
                    ax,
                    shared_image,
                    gt_mask,
                    label=f"GT: {filename}",
                    alpha=alpha,
                    strict=strict,
                    require_mask=True,
                )
                continue

            pred_path = find_model_asset(input_dir, column, "pred", filename)
            model_image_path = find_model_asset(input_dir, column, "image", filename)
            panel_path = find_model_asset(input_dir, column, "panel", filename)
            image = rotate_for_file(read_image(model_image_path), filename) if model_image_path is not None else shared_image
            pred_mask = rotate_for_file(read_mask(pred_path), filename) if pred_path is not None else None

            if image is None and panel_path is not None:
                image = rotate_for_file(read_image(panel_path), filename)
                pred_mask = None
            draw_cell(
                ax,
                image,
                pred_mask,
                label=f"{column}: {filename}",
                alpha=alpha,
                strict=strict,
            )

    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    args = parse_args()
    for output_name, columns in LAYOUTS.items():
        draw_layout(
            input_dir=args.input_dir,
            output_path=args.output_dir / output_name,
            columns=columns,
            alpha=float(args.alpha),
            dpi=int(args.dpi),
            strict=bool(args.strict),
        )


if __name__ == "__main__":
    main()
