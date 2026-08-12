from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split all_train.txt into train_new.txt and val_new.txt.")
    parser.add_argument("--input", type=Path, default=Path("data/all_train.txt"), help="Input CSV split file.")
    parser.add_argument("--train-output", type=Path, default=Path("data/train_new.txt"), help="Output train split file.")
    parser.add_argument("--val-output", type=Path, default=Path("data/val_new.txt"), help="Output validation split file.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio. Default: 0.10.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--skip-analysis", action="store_true", help="Only split files, do not run dataset analysis.")
    parser.add_argument("--analysis-output-root", type=Path, default=Path("outpus_analysis"), help="Analysis output root.")
    parser.add_argument("--analysis-run-name", type=str, default=None, help="Optional analysis output subfolder name.")
    parser.add_argument(
        "--analysis-splits",
        nargs="+",
        default=["train_new", "val_new", "test"],
        help="Split files to analyze after splitting.",
    )
    parser.add_argument("--analysis-num-visuals", type=int, default=10, help="Number of visualization samples per split.")
    parser.add_argument(
        "--analysis-visual-selection",
        choices=["per_source", "fixed"],
        default="per_source",
        help="Visualization selection mode for analysis.",
    )
    parser.add_argument("--analysis-visuals-per-source", type=int, default=1, help="Number of visualization samples per source.")
    parser.add_argument("--analysis-slice-axis", type=int, default=2, help="Axis used for 2D slice visualization.")
    parser.add_argument("--analysis-label-threshold", type=float, default=0.0, help="Mask threshold for label presence.")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows: list[dict[str, str]], val_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    val_count = max(1, min(len(shuffled) - 1, val_count))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    return train_rows, val_rows


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_rows(args.input)
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    write_rows(args.train_output, fieldnames, train_rows)
    write_rows(args.val_output, fieldnames, val_rows)
    print(f"Input samples: {len(rows)}")
    print(f"Train samples: {len(train_rows)} -> {args.train_output}")
    print(f"Val samples: {len(val_rows)} -> {args.val_output}")

    if not args.skip_analysis:
        from analysis_data.analyze_dataset import run_analysis

        analysis_args = argparse.Namespace(
            data_root=args.input.parent,
            output_root=args.analysis_output_root,
            run_name=args.analysis_run_name,
            splits=args.analysis_splits,
            num_visuals=args.analysis_num_visuals,
            visual_selection=args.analysis_visual_selection,
            visuals_per_source=args.analysis_visuals_per_source,
            seed=args.seed,
            slice_axis=args.analysis_slice_axis,
            label_threshold=args.analysis_label_threshold,
            skip_visualization=False,
        )
        output_dir = run_analysis(analysis_args)
        print(f"Analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
