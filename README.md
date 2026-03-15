# MS-COCO Multi-Label Classification

PyTorch project for 80-class multi-label image classification on an MS-COCO-style dataset.

The project keeps `training.py` and `testing.py` at the repository root. Support modules and utility scripts are organized under the `utils/` package.

## Repository Structure

- `training.py`: training/validation pipeline, checkpointing, TensorBoard logging, and post-training artifact generation.
- `testing.py`: prediction generation for all run folders by default.
- `utils/config.py`: global paths, classes, and model defaults.
- `utils/models_factory.py`: backbone registry and classifier-head replacement.
- `utils/dataset_readers.py`: train/test dataset loaders.
- `utils/training_utils.py`: training/validation loops and helpers.
- `utils/generate_confusion_matrix.py`: confusion-matrix generation utilities (single run and batch mode).
- `utils/model_performance_table.py`: aggregate run metadata into CSV.
- `utils/tensorboard_logging.py`, `utils/references.py`, `utils/metadata_utils.py`: logging/metadata/constants helpers.
- `report_images/`: report figures.
- `trained_models/`: run artifacts, predictions, TensorBoard event files, and summary table.
- `REPORT.md`: project report.

## Included Runs and Artifacts

Some trained runs are already included in the repository (the same runs discussed in `REPORT.md`).

Large checkpoint files (`*.pt`) are git-ignored and may not be present in a fresh clone.

For included runs, the repository still provides useful artifacts in `trained_models/`, including:

- `run_config.json` (configuration + results + runtime metadata)
- `confusion_matrix.png`
- `predictions.json`
- TensorBoard event files under `trained_models/tensorboard_runs/`
- `trained_models/model_performance_table.csv`

## Supported Backbones

From `utils/models_factory.py` (`MODEL_SPECS`):

- `resnet18`
- `resnet50`
- `densenet121`
- `mobilenet_v2`
- `mobilenet_v3_large`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `efficientnet_v2_s`
- `efficientnet_b4`
- `convnext_tiny`
- `convnext_small`
- `convnext_base`
- `convnext_large`
- `regnet_y_800mf`
- `swin_t`
- `swin_v2_t`
- `swin_v2_s`
- `swin_v2_b`
- `vit_b_16`

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Paths and Dataset Layout

Paths are now project-local (no user home directory dependency):

- `DATASET_FOLDER = <repo>/ms-coco`
- `PRETRAINED_MODELS_FOLDER = <repo>/pre-trained_models`
- `TRAINED_MODELS_FOLDER = <repo>/trained_models`

Expected dataset structure:

```text
<repo>/
|-- ms-coco/
|   |-- images/
|   |   |-- train-resized/   # train images (.jpg)
|   |   `-- test-resized/    # test images (.jpg)
|   `-- labels/
|       `-- train/           # one .cls file per train image
|-- pre-trained_models/
`-- trained_models/
```

Each `.cls` file contains one class index per line (`0..79`) matching `utils/config.py::CLASSES`.

## Current Runtime Defaults

### `utils/config.py`

- `MODEL_NAME = "convnext_tiny"` (active assignment)
- `FREEZE_BACKBONE = True`
- `BEST_MODEL_PATH = trained_models/best_model.pt`

### `training.py`

- `NUM_EPOCHS = 14`
- `TRAIN_BATCH_SIZE_FROZEN = 32`
- `TRAIN_BATCH_SIZE_UNFROZEN = 16`
- `VAL_BATCH_SIZE = 32`
- `GRAD_ACCUM_STEPS_FROZEN = 1`
- `GRAD_ACCUM_STEPS_UNFROZEN = 1`
- `UNFREEZE_BACKBONE_EPOCH = 1`
- `UNFREEZE_LAST_N_BACKBONE_LAYERS = None`
- differential LR:
    - `BACKBONE_BASE_LR = 1e-5`
    - `HEAD_BASE_LR = 1e-4`
    - `LR_MILESTONES = (5, 10)`
    - `LR_DECAY_FACTOR = 1e-2`
- `VAL_SPLIT = 0.05`
- `SEED = 42`
- `NUM_WORKERS = 10`
- early stopping enabled (`patience = 4`, `min_delta = 0.0`)
- TensorBoard enabled (`USE_TENSORBOARD = True`)

### `testing.py`

- `GENERATE_FOR_ALL_RUNS = True`
- `BATCH_SIZE = 32`
- `NUM_WORKERS = 0`
- `TH_MULTI_LABEL = 0.5`
- `OVERWRITE_EXISTING = False`

### `utils/generate_confusion_matrix.py`

- `GENERATE_FOR_ALL_RUNS = True`
- `SPLIT = "val"`
- `VAL_SPLIT = 0.05`
- `BATCH_SIZE = 64`
- `NUM_WORKERS = 0`
- `NORMALIZE = "rows"`
- `TOP_K_CLASSES = 40`
- `OVERWRITE_EXISTING = False`

## Training

```bash
python training.py
```

Training creates/updates:

- `trained_models/<model_name>_<timestamp>/best_model.pt`
- `trained_models/<model_name>_<timestamp>/run_config.json`
- `trained_models/<model_name>_<timestamp>/confusion_matrix.png` (generated if missing)
- `trained_models/<model_name>_<timestamp>/predictions.json` (generated if missing)
- `trained_models/best_model.pt` (active checkpoint)
- TensorBoard logs in `trained_models/tensorboard_runs/<model_name>/<run_name>/`

Launch TensorBoard:

```bash
tensorboard --logdir trained_models/tensorboard_runs
```

## Prediction Generation

```bash
python testing.py
```

Default behavior scans `trained_models/**/best_model.pt` and writes missing `predictions.json` files per run.

## Confusion Matrix Generation

```bash
python -m utils.generate_confusion_matrix
```

Default behavior scans `trained_models/**/best_model.pt` and writes missing `confusion_matrix.png` files per run.

## Model Performance Table

```bash
python -m utils.model_performance_table
```

Writes `trained_models/model_performance_table.csv` from run configs (preferred) or checkpoints (fallback).

## Prediction JSON Format

```json
{
    "000000000139": [0, 56, 57, 60, 62],
    "000000000285": [21]
}
```

Keys are image filenames without `.jpg`; values are predicted class indices.
