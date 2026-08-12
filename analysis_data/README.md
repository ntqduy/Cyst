# Cyst Data Analysis

Run from the project root:

```bash
python analysis_data/analyze_dataset.py
```

By default this analyzes `data/train_new.txt`, `data/val_new.txt`, and
`data/test.txt`, matching the non-k-fold training config. To inspect the older
`train.txt`/`val.txt` split, pass `--splits train val test`.

Default output:

```text
outpus_analysis/cyst_data_analysis_YYYYMMDD_HHMMSS/
```

Main files:

- `split_summary.csv`: total samples and labeled samples per train/val/test split.
- `sample_stats.csv`: per-sample source, shape, spacing, label presence, voxel counts.
- `source_summary.csv`: sample counts and ratios grouped by split and source prefix such as `EMC`, `IU`, `NYU`.
- `source_split_matrix.csv`: compact source-by-split count table.
- `shape_summary.csv`: shape frequency summary.
- `source_size_distribution.csv`: H/W/D, image voxel, lesion voxel, and positive-ratio distribution grouped by split and source.
- `source_shape_distribution.csv`: exact image-shape frequency grouped by split and source.
- `size_distribution/{split}/source_size_distribution.csv`: the same size distribution separated into train/val/test folders.
- `size_distribution/{split}/source_shape_distribution.csv`: exact shape counts separated into train/val/test folders.
- `size_distribution/{split}/source_size_boxplot.pdf`: quick boxplot of Dim0, Dim1, Dim2/depth, and lesion voxels for each source.
- `visualizations/{split}/`: by default, one 2D preview per source in each split with Image, GT, and overlay.
- `summary.json` and `summary.txt`: machine-readable and quick text summaries.

Useful options:

```bash
python analysis_data/analyze_dataset.py --slice-axis 2 --seed 42
python analysis_data/analyze_dataset.py --visuals-per-source 2
python analysis_data/analyze_dataset.py --visual-selection fixed --num-visuals 10
python analysis_data/analyze_dataset.py --skip-visualization
python analysis_data/analyze_dataset.py --output-root outputs_analysis
```
