# Camera Trap Data Pipeline

This repo processes camera-trap datasets by (1) cropping images once per camera and (2) generating day-interval JSONs and analysis results. Below is the up-to-date flow and how to run it.

## Overview

- Crop images once per camera, then reuse for any day interval.
- Outputs are written to `/fs/scratch/PAS2099/camera-trap-benchmark` by default (set in code).

## Prerequisites

- Python packages: `pip install -r requirements.txt`
- Link raw images (symlinks) once using `datasets/image_setup.sh` if needed.

```bash
bash datasets/image_setup.sh
```

## Outputs

### Cropped images + JSONs (default root: `/fs/scratch/PAS2099/camera-trap-benchmark`)
```
<root>/<dataset>/<camera>/
├── images/                 # Cropped images (created once)
├── .images_cropped         # Marker file (prevents re-cropping)
├── 15/                     # JSONs for 15-day interval
│   ├── train.json
│   ├── train-all.json      # merged ckps under key `ckp_-1`
│   └── test.json
├── 30/
│   ├── train.json
│   ├── train-all.json
│   └── test.json
└── 60/
    ├── train.json
    ├── train-all.json
    └── test.json
```

### Analysis results (in `analysis_results/`)
- Aggregated CSVs per interval (e.g., `metrics_results_15days.csv`, `metrics_results_30days.csv`, `metrics_results_60days.csv`).
- Subfolders per dataset/camera with plots, if enabled by config.

## How to run

Use the wrapper `run.py`.

- Crop once (per camera), then stop:
```bash
python run.py --config run_config.yaml --days 30 --crop_images
```

- Prepare JSONs for multiple day intervals (no image writes):
```bash
python run.py --config run_config.yaml --days 15 30 60 --prepare_data
```

- Run analysis only (expects cropped images and JSONs to exist):
```bash
python run.py --config run_config.yaml --days 15 30 60 --run_analysis
```

- First-time full pipeline (crop + JSONs, analysis optional):
```bash
python run.py --config run_config.yaml --days 15 30 60 --crop_images --prepare_data
```

Common flags:
- `--dataset <names...>` override datasets from the YAML.
- `--wandb` enable logging (if installed).

## Configuration

Edit `run_config.yaml` (or point `--config` to your file). It controls:
- which datasets/cameras to use
- filtering toggles and thresholds
- analysis outputs under `analysis_results/`

Note: The output root for crops/JSONs is set in code as `/fs/scratch/PAS2099/camera-trap-benchmark` (see `tools/pipeline.py` and `tools/data_prep/image_prep.py`). Change in code if you need a different location.

### Config file quick reference (run_config.yaml)

Core pieces you’ll set:
- Global filters (top-level booleans): toggle filters in `datasets/dataset_filter.py`.
    - no_multi_classes, exclude_classes, no_bbox, no_low_bbox_conf, only_animal_class_bbox, no_datetime, min_image_count, no_missing_images
- Analysis
    - analysis_path: output root (CSVs/plots)
    - metrics_analysis: enable ts_l1_accumulated, ts_l1_full, gini_index, l1_test
    - plot_analysis: enable ckp_piechart, count_histogram
- Dataset selection
    - all_datasets or datasets: which dataset modules to load; CLI `--dataset` can override
    - Per-dataset: all_camera/cameras and filtering_config thresholds

Tiny example
```yaml
analysis_path: analysis_results
metrics_analysis: { ts_l1_accumulated: true, ts_l1_full: true, gini_index: true, l1_test: true }
plot_analysis: { ckp_piechart: true, count_histogram: true }

no_multi_classes: true
exclude_classes: true
no_bbox: true
no_low_bbox_conf: true
min_image_count: true

datasets: ["safari"]
safari:
    filtering_config: { classes_to_exclude: [0,7,35], bbox_min_conf: 0.8, min_image_count_threshold: 1000 }
    APN: { all_camera: true, cameras: [] }
```

Notes
- Thresholds live under each dataset’s `filtering_config` and are used only when the corresponding global filter is true.
- `safari` is an aggregator with many sub-datasets; others like `caltech`/`na` use `all_camera`/`cameras` directly.
- Datetime formats can come from `setting.yaml` if present; otherwise defaults are used.

 
## Repo layout
```
datasets/
    dataset.py            # dataset aggregator/loader
    dataset_filter.py     # filtering utilities
    image_setup.sh        # optional symlink setup for raw images
tools/
    pipeline.py           # orchestration (prep ckps, crop, jsons, analysis)
    analysis/             # analysis modules
    data_prep/            # checkpoint + image prep logic
run.py                    # thin wrapper that calls tools.pipeline.run
```

## Notes
- JSONs include a merged `train-all.json` with all ckps under key `ckp_-1` for convenience.
