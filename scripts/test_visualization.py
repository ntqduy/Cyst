from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.visualization import _prepare_visual_sample, _slice_index_for_volume_sample


def test_2d_volume_visualization_uses_slice_axis() -> None:
    cfg = {
        "slice_2d": {"axis": 2},
        "visualization": {"slice_axis": 0, "slice_position": "label_foreground"},
    }
    sample = {
        "image_slices": torch.zeros(35, 1, 256, 256),
        "label_slices": torch.zeros(35, 256, 256, dtype=torch.long),
        "label": torch.zeros(256, 256, 35, dtype=torch.long),
        "slice_index": -1,
    }
    sample["label"][168, 120, 20] = 1
    sample["label_slices"][20, 168, 120] = 1

    slice_index = _slice_index_for_volume_sample(sample, cfg)
    if slice_index != 20:
        raise AssertionError(f"Expected foreground slice 20, got {slice_index}.")

    image, label, image_for_display, target_shape, selected_slice_index = _prepare_visual_sample(
        sample,
        cfg,
        model_type="2D",
        device=torch.device("cpu"),
    )
    if selected_slice_index != 20:
        raise AssertionError(f"Expected selected slice 20, got {selected_slice_index}.")
    if tuple(image.shape) != (1, 1, 256, 256):
        raise AssertionError(f"Unexpected prepared image shape: {tuple(image.shape)}.")
    if tuple(label.shape) != (256, 256) or tuple(image_for_display.shape) != (1, 256, 256):
        raise AssertionError("Prepared visualization arrays have unexpected shapes.")


def main() -> None:
    test_2d_volume_visualization_uses_slice_axis()
    print("visualization slice-axis regression: ok")


if __name__ == "__main__":
    main()
