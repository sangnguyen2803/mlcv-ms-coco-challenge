from pathlib import Path
import json
import time
from datetime import datetime, timezone
from typing import Iterable

import torch
from torch.utils.data import DataLoader, random_split

from utils.config import (
    BEST_MODEL_PATH,
    FREEZE_BACKBONE,
    MODEL_NAME,
    NUM_CLASSES,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
)
from utils.dataset_readers import COCOTrainImageDataset
from utils.generate_confusion_matrix import generate_missing_confusion_matrices_for_runs
from utils.metadata_utils import checkpoint_inference_threshold
from utils.models_factory import AVAILABLE_MODELS, create_model, freeze_all, unfreeze_last_n_backbone_layers
from testing import generate_missing_predictions_for_runs
from utils.references import (
    CKPT_BEST_EPOCH,
    CKPT_BEST_THRESHOLD,
    CKPT_BEST_VAL_ACCURACY,
    CKPT_BEST_VAL_F1,
    CKPT_BEST_VAL_LOSS,
    CKPT_BEST_VAL_PRECISION,
    CKPT_BEST_VAL_RECALL,
    CKPT_BATCH_SIZE,
    CKPT_LEARNING_RATE,
    CKPT_LEARNING_RATES,
    CKPT_MODEL_NAME,
    CKPT_RUN_DURATION_SECONDS,
    CKPT_STATE_DICT,
    CKPT_THRESHOLD,
    CKPT_TOTAL_EPOCHS,
    METRIC_ACCURACY,
    METRIC_F1,
    METRIC_LOSS,
    METRIC_PRECISION,
    METRIC_RECALL,
)
from utils.training_utils import (
    print_section,
    tokenize_float,
    train_loop,
    tune_threshold_on_validation,
    validation_loop,
)

TENSORBOARD_AVAILABLE: bool

try:
    from torch.utils.tensorboard import SummaryWriter

    from utils.tensorboard_logging import (
        configure_custom_scalar_layout,
        log_hparams_summary,
        log_run_configuration,
        update_graphs,
    )

    TENSORBOARD_AVAILABLE = True
except ModuleNotFoundError:
    TENSORBOARD_AVAILABLE = False

# Batch sizing.
TRAIN_BATCH_SIZE_FROZEN: int = 32
TRAIN_BATCH_SIZE_UNFROZEN: int = 16
VAL_BATCH_SIZE: int = 32

# Keep effective batch size high even when unfrozen batch must be small.
GRAD_ACCUM_STEPS_FROZEN: int = 1
GRAD_ACCUM_STEPS_UNFROZEN: int = 1

USE_AMP: bool = True
AMP_DTYPE: torch.dtype = torch.float16

NUM_EPOCHS: int = 1

TRAIN_METRICS_EVERY_N_EPOCHS: int = 1
VAL_EVERY_N_EPOCHS: int = 1

# Freeze/unfreeze schedule (independent from LR schedule).
FREEZE_BACKBONE_AT_START: bool = FREEZE_BACKBONE
UNFREEZE_BACKBONE_EPOCH: int = 2  # 1-based epoch index; ignored when not freezing at start.
# None => full backbone unfreeze. Set an integer >= 1 to unfreeze only the last n backbone layers.
UNFREEZE_LAST_N_BACKBONE_LAYERS: int | None = None

# Base LR used only when `USE_DIFFERENTIAL_LR` is disabled.
LEARNING_RATE: float = 1e-2

# LR schedule (independent from freeze/unfreeze schedule).
USE_DIFFERENTIAL_LR: bool = True
BACKBONE_BASE_LR: float = 1e-5
HEAD_BASE_LR: float = 1e-4
LR_MILESTONES: tuple[int, ...] = (5, 10)
LR_DECAY_FACTOR: float = 1e-2

VAL_SPLIT: float = 0.05
SEED: int = 42
NUM_WORKERS: int = 10

# Default threshold for multi-label prediction probabilities.
TH_MULTI_LABEL: float = 0.5
THRESHOLD_CANDIDATES: tuple[float, ...] = tuple(i / 100 for i in range(5, 96, 5))
MBATCH_LOSS_GROUP: int = -1

EARLY_STOPPING_ENABLED: bool = True
EARLY_STOPPING_PATIENCE: int = 4
EARLY_STOPPING_MIN_DELTA: float = 0.0

USE_TENSORBOARD: bool = True
TRAINED_MODELS_ROOT: Path = BEST_MODEL_PATH.parent
ACTIVE_CHECKPOINT_FILENAME: str = BEST_MODEL_PATH.name
MODEL_PATH: Path = TRAINED_MODELS_ROOT / ACTIVE_CHECKPOINT_FILENAME
RUN_CONFIG_FILENAME: str = "run_config.json"
RUN_CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.png"
RUN_PREDICTIONS_FILENAME: str = "predictions.json"
TENSORBOARD_RUNS_DIRNAME: str = "tensorboard_runs"


def should_run_eval(epoch: int, every_n_epochs: int, force_last: bool, total_epochs: int) -> bool:
    if every_n_epochs > 0 and (epoch + 1) % every_n_epochs == 0:
        return True
    return force_last and epoch == total_epochs - 1


def build_training_plan_token() -> str:
    milestones_token = "-".join(str(milestone) for milestone in LR_MILESTONES) if LR_MILESTONES else "none"
    unfreeze_token = (
        str(UNFREEZE_BACKBONE_EPOCH)
        if FREEZE_BACKBONE_AT_START and 1 <= UNFREEZE_BACKBONE_EPOCH <= NUM_EPOCHS
        else "none"
    )
    if USE_DIFFERENTIAL_LR:
        lr_token = (
            f"blr{tokenize_float(BACKBONE_BASE_LR, precision=6)}" f"-hlr{tokenize_float(HEAD_BASE_LR, precision=6)}"
        )
    else:
        lr_token = f"lr{tokenize_float(LEARNING_RATE, precision=6)}"
    partial_unfreeze_token = "all" if UNFREEZE_LAST_N_BACKBONE_LAYERS is None else str(UNFREEZE_LAST_N_BACKBONE_LAYERS)
    return (
        f"uf{unfreeze_token}-frz{int(FREEZE_BACKBONE_AT_START)}-{lr_token}"
        f"-ul{partial_unfreeze_token}"
        f"-bs{TRAIN_BATCH_SIZE_FROZEN}to{TRAIN_BATCH_SIZE_UNFROZEN}"
        f"-ga{GRAD_ACCUM_STEPS_FROZEN}to{GRAD_ACCUM_STEPS_UNFROZEN}"
        f"-amp{int(USE_AMP)}-ms{milestones_token}"
    )


def build_batch_size_token() -> str:
    has_unfreeze = FREEZE_BACKBONE_AT_START and 1 <= UNFREEZE_BACKBONE_EPOCH <= NUM_EPOCHS
    if has_unfreeze:
        return f"{TRAIN_BATCH_SIZE_FROZEN}to{TRAIN_BATCH_SIZE_UNFROZEN}"
    if FREEZE_BACKBONE_AT_START:
        return str(TRAIN_BATCH_SIZE_FROZEN)
    return str(TRAIN_BATCH_SIZE_UNFROZEN)


def build_run_output_dir(root: Path, model_name: str, started_at: datetime) -> tuple[Path, str]:
    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    base_name = f"{model_name}_{timestamp}"
    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_dir.name


def build_tensorboard_run_name(run_id: str) -> str:
    lr_token = (
        f"d{tokenize_float(BACKBONE_BASE_LR, precision=6)}-{tokenize_float(HEAD_BASE_LR, precision=6)}"
        if USE_DIFFERENTIAL_LR
        else f"s{tokenize_float(LEARNING_RATE, precision=6)}"
    )
    unfreeze_epoch_token = str(UNFREEZE_BACKBONE_EPOCH) if FREEZE_BACKBONE_AT_START else "na"
    unfreeze_layers_token = "all" if UNFREEZE_LAST_N_BACKBONE_LAYERS is None else str(UNFREEZE_LAST_N_BACKBONE_LAYERS)
    run_suffix = run_id.split("_")[-1]
    return (
        f"e{NUM_EPOCHS}"
        f"_bs{TRAIN_BATCH_SIZE_FROZEN}-{TRAIN_BATCH_SIZE_UNFROZEN}"
        f"_ga{GRAD_ACCUM_STEPS_FROZEN}-{GRAD_ACCUM_STEPS_UNFROZEN}"
        f"_frz{int(FREEZE_BACKBONE_AT_START)}"
        f"_ufep{unfreeze_epoch_token}"
        f"_ufn{unfreeze_layers_token}"
        f"_lr{lr_token}"
        f"_vs{tokenize_float(VAL_SPLIT, precision=3)}"
        f"_sd{SEED}"
        f"_amp{int(USE_AMP)}"
        f"_{run_suffix}"
    )


def json_default(value: object):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=json_default)


def configure_trainable_state(
    net: torch.nn.Module,
    head_params: list[torch.nn.Parameter],
    freeze_backbone_now: bool,
) -> tuple[str, list[str]]:
    freeze_all(net)
    for param in head_params:
        param.requires_grad = True

    if freeze_backbone_now:
        return "frozen backbone (head-only fine-tuning)", []

    if UNFREEZE_LAST_N_BACKBONE_LAYERS is None:
        for param in net.parameters():
            param.requires_grad = True
        return "unfrozen backbone (full-model fine-tuning)", ["<all backbone layers>"]

    unfrozen_layer_names = unfreeze_last_n_backbone_layers(
        net,
        UNFREEZE_LAST_N_BACKBONE_LAYERS,
        head_params=head_params,
    )
    mode_text = (
        f"partially unfrozen backbone (last {len(unfrozen_layer_names)} layers + head)"
        if unfrozen_layer_names
        else "frozen backbone (head-only fine-tuning)"
    )
    return mode_text, unfrozen_layer_names


def build_optimizer(
    net: torch.nn.Module,
    head_params: list[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    if USE_DIFFERENTIAL_LR:
        head_param_ids = {id(param) for param in head_params}
        all_params = list(net.parameters())
        backbone_params = [param for param in all_params if id(param) not in head_param_ids]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": BACKBONE_BASE_LR})
        param_groups.append({"params": head_params, "lr": HEAD_BASE_LR})
        return torch.optim.Adam(param_groups)
    return torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.MultiStepLR | None:
    if not LR_MILESTONES:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=(*LR_MILESTONES,),
        gamma=LR_DECAY_FACTOR,
    )


def build_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def iter_subset_label_paths(train_subset) -> Iterable[Path]:
    dataset = train_subset.dataset
    annotations_dir = Path(dataset.annotations_dir)
    for idx in train_subset.indices:
        yield annotations_dir / dataset.img_labels[idx]


def compute_pos_weight(train_subset, num_classes: int) -> torch.Tensor:
    class_positives = torch.zeros(num_classes, dtype=torch.float64)
    num_samples = len(train_subset)
    for label_path in iter_subset_label_paths(train_subset):
        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                class_index = int(stripped)
                if 0 <= class_index < num_classes:
                    class_positives[class_index] += 1.0
    class_negatives = float(num_samples) - class_positives
    pos_weight = torch.where(class_positives > 0, class_negatives / class_positives, torch.ones_like(class_positives))
    return pos_weight.to(dtype=torch.float32)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started_at = datetime.now(timezone.utc)
    training_plan_token = build_training_plan_token()
    batch_size_token = build_batch_size_token()
    run_output_dir, run_id = build_run_output_dir(TRAINED_MODELS_ROOT, MODEL_NAME, run_started_at)
    run_model_path = run_output_dir / ACTIVE_CHECKPOINT_FILENAME
    run_config_path = run_output_dir / RUN_CONFIG_FILENAME
    run_confusion_matrix_path = run_output_dir / RUN_CONFUSION_MATRIX_FILENAME
    run_predictions_path = run_output_dir / RUN_PREDICTIONS_FILENAME
    tensorboard_runs_root = TRAINED_MODELS_ROOT / TENSORBOARD_RUNS_DIRNAME / MODEL_NAME
    tensorboard_run_name = build_tensorboard_run_name(run_id)
    tensorboard_log_dir = tensorboard_runs_root / tensorboard_run_name

    training_config = {
        "model_name": MODEL_NAME,
        "device": device.type,
        "epochs": NUM_EPOCHS,
        "train_batch_size_frozen": TRAIN_BATCH_SIZE_FROZEN,
        "train_batch_size_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN,
        "val_batch_size": VAL_BATCH_SIZE,
        "grad_accum_steps_frozen": GRAD_ACCUM_STEPS_FROZEN,
        "grad_accum_steps_unfrozen": GRAD_ACCUM_STEPS_UNFROZEN,
        "effective_batch_frozen": TRAIN_BATCH_SIZE_FROZEN * GRAD_ACCUM_STEPS_FROZEN,
        "effective_batch_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN * GRAD_ACCUM_STEPS_UNFROZEN,
        "use_amp": USE_AMP,
        "batch_size_token": batch_size_token,
        "freeze_backbone_at_start": FREEZE_BACKBONE_AT_START,
        "unfreeze_backbone_epoch": (UNFREEZE_BACKBONE_EPOCH if FREEZE_BACKBONE_AT_START else "n/a"),
        "unfreeze_last_n_backbone_layers": (
            UNFREEZE_LAST_N_BACKBONE_LAYERS if UNFREEZE_LAST_N_BACKBONE_LAYERS is not None else "all"
        ),
        "use_differential_lr": USE_DIFFERENTIAL_LR,
        "learning_rate": LEARNING_RATE if not USE_DIFFERENTIAL_LR else "n/a",
        "backbone_base_lr": BACKBONE_BASE_LR if USE_DIFFERENTIAL_LR else "n/a",
        "head_base_lr": HEAD_BASE_LR if USE_DIFFERENTIAL_LR else "n/a",
        "lr_milestones": LR_MILESTONES,
        "lr_decay_factor": LR_DECAY_FACTOR,
        "training_plan_token": training_plan_token,
        "th_multi_label": TH_MULTI_LABEL,
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "val_split": VAL_SPLIT,
        "num_workers": NUM_WORKERS,
        "train_metrics_every_n_epochs": TRAIN_METRICS_EVERY_N_EPOCHS,
        "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
        "early_stopping_enabled": EARLY_STOPPING_ENABLED,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "trained_models_root": TRAINED_MODELS_ROOT,
        "run_id": run_id,
        "run_started_at_utc": run_started_at.isoformat(),
        "run_output_dir": run_output_dir,
        "run_model_path": run_model_path,
        "run_config_path": run_config_path,
        "run_confusion_matrix_path": run_confusion_matrix_path,
        "run_predictions_path": run_predictions_path,
        "tensorboard_log_dir": tensorboard_log_dir,
        "active_checkpoint_filename": ACTIVE_CHECKPOINT_FILENAME,
        "active_checkpoint_path": MODEL_PATH,
    }
    print_section("TRAINING START CONFIG", training_config)

    net, transform, head_params = create_model(MODEL_NAME, NUM_CLASSES, pretrained=True)
    if transform is None:
        raise RuntimeError("No transform available. Use a pretrained model or provide a custom transform.")

    full_dataset = COCOTrainImageDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)

    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_batch_size_now = TRAIN_BATCH_SIZE_FROZEN if FREEZE_BACKBONE_AT_START else TRAIN_BATCH_SIZE_UNFROZEN
    grad_accum_steps_now = GRAD_ACCUM_STEPS_FROZEN if FREEZE_BACKBONE_AT_START else GRAD_ACCUM_STEPS_UNFROZEN
    train_loader = build_loader(train_set, batch_size=train_batch_size_now, shuffle=True)
    val_loader = build_loader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=False)

    net = net.to(device)
    head_params_list = list(head_params)
    current_mode_text, active_backbone_layers = configure_trainable_state(
        net, head_params_list, FREEZE_BACKBONE_AT_START
    )
    print(f"Training mode at start: {current_mode_text}")
    if active_backbone_layers and active_backbone_layers != ["<all backbone layers>"]:
        print(
            f"Trainable backbone layers at start ({len(active_backbone_layers)}): {', '.join(active_backbone_layers)}"
        )

    should_unfreeze_later = FREEZE_BACKBONE_AT_START and 1 <= UNFREEZE_BACKBONE_EPOCH <= NUM_EPOCHS
    if should_unfreeze_later:
        target_layers = UNFREEZE_LAST_N_BACKBONE_LAYERS if UNFREEZE_LAST_N_BACKBONE_LAYERS is not None else "all"
        print(f"Backbone unfreeze scheduled at epoch {UNFREEZE_BACKBONE_EPOCH} (layers={target_layers}).")

    optimizer = build_optimizer(net, head_params_list)
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP and device.type == "cuda" else None

    tensorboard_hparams = {
        "model_name": MODEL_NAME,
        "epochs": NUM_EPOCHS,
        "train_batch_size_frozen": TRAIN_BATCH_SIZE_FROZEN,
        "train_batch_size_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN,
        "val_batch_size": VAL_BATCH_SIZE,
        "grad_accum_steps_frozen": GRAD_ACCUM_STEPS_FROZEN,
        "grad_accum_steps_unfrozen": GRAD_ACCUM_STEPS_UNFROZEN,
        "use_amp": USE_AMP,
        "freeze_backbone_at_start": FREEZE_BACKBONE_AT_START,
        "unfreeze_backbone_epoch": UNFREEZE_BACKBONE_EPOCH if FREEZE_BACKBONE_AT_START else -1,
        "unfreeze_last_n_backbone_layers": (
            "all" if UNFREEZE_LAST_N_BACKBONE_LAYERS is None else UNFREEZE_LAST_N_BACKBONE_LAYERS
        ),
        "use_differential_lr": USE_DIFFERENTIAL_LR,
        "learning_rate": LEARNING_RATE if not USE_DIFFERENTIAL_LR else -1.0,
        "backbone_base_lr": BACKBONE_BASE_LR if USE_DIFFERENTIAL_LR else -1.0,
        "head_base_lr": HEAD_BASE_LR if USE_DIFFERENTIAL_LR else -1.0,
        "lr_milestones": ",".join(str(value) for value in LR_MILESTONES) if LR_MILESTONES else "none",
        "lr_decay_factor": LR_DECAY_FACTOR,
        "val_split": VAL_SPLIT,
        "threshold_init": TH_MULTI_LABEL,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
    }

    pos_weight = compute_pos_weight(train_set, NUM_CLASSES).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    run_best_f1 = -1.0
    run_best_epoch = -1
    run_best_threshold = TH_MULTI_LABEL
    run_best_checkpoint = None
    last_train_results = None
    last_val_results = None
    last_tuned_threshold = TH_MULTI_LABEL
    ran_train_eval_last_epoch = False
    no_improve_eval_count = 0
    completed_epochs = 0
    early_stopped = False
    early_stop_reason = "none"
    summary_writer = None
    if USE_TENSORBOARD and TENSORBOARD_AVAILABLE:
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
        configure_custom_scalar_layout(summary_writer, lr_group_count=len(optimizer.param_groups))
        log_run_configuration(
            summary_writer,
            run_name=f"{MODEL_NAME}/{tensorboard_run_name}",
            run_config=tensorboard_hparams,
        )

    backbone_is_unfrozen = not FREEZE_BACKBONE_AT_START
    for epoch in range(NUM_EPOCHS):
        epoch_index = epoch + 1
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch_index}:")

        if should_unfreeze_later and not backbone_is_unfrozen and epoch_index >= UNFREEZE_BACKBONE_EPOCH:
            current_mode_text, active_backbone_layers = configure_trainable_state(net, head_params_list, False)
            backbone_is_unfrozen = True
            train_batch_size_now = TRAIN_BATCH_SIZE_UNFROZEN
            grad_accum_steps_now = GRAD_ACCUM_STEPS_UNFROZEN
            train_loader = build_loader(train_set, batch_size=train_batch_size_now, shuffle=True)
            print(
                f"Backbone unfrozen at epoch {epoch_index}. Mode: {current_mode_text} | "
                f"train_batch_size={train_batch_size_now}, grad_accum_steps={grad_accum_steps_now}"
            )
            if active_backbone_layers and active_backbone_layers != ["<all backbone layers>"]:
                print(
                    "Unfrozen backbone layers " f"({len(active_backbone_layers)}): {', '.join(active_backbone_layers)}"
                )

        lr_values = [f"{group['lr']:.6g}" for group in optimizer.param_groups]
        current_lr_text = lr_values[0] if len(lr_values) == 1 else f"[{', '.join(lr_values)}]"

        run_train_metrics = should_run_eval(
            epoch,
            TRAIN_METRICS_EVERY_N_EPOCHS,
            force_last=False,
            total_epochs=NUM_EPOCHS,
        )
        run_val_metrics = should_run_eval(
            epoch,
            VAL_EVERY_N_EPOCHS,
            force_last=True,
            total_epochs=NUM_EPOCHS,
        )
        if summary_writer and run_val_metrics:
            run_train_metrics = True

        mbatch_losses = train_loop(
            train_loader,
            net,
            criterion,
            optimizer,
            device,
            mbatch_loss_group=MBATCH_LOSS_GROUP,
            progress_label="    Training",
            grad_accum_steps=grad_accum_steps_now,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            scaler=scaler,
        )

        train_results = None
        val_results = None
        tuned_threshold = TH_MULTI_LABEL
        if run_val_metrics:
            tuned_threshold, val_results = tune_threshold_on_validation(
                val_loader,
                net,
                criterion,
                NUM_CLASSES,
                device,
                one_hot=True,
                threshold_candidates=THRESHOLD_CANDIDATES,
                progress_label="    Validating",
                apply_sigmoid=True,
            )
            last_val_results = val_results
            last_tuned_threshold = tuned_threshold

        if run_train_metrics:
            train_eval_threshold = tuned_threshold if run_val_metrics else run_best_threshold
            train_results = validation_loop(
                train_loader,
                net,
                criterion,
                NUM_CLASSES,
                device,
                multi_label=True,
                th_multi_label=train_eval_threshold,
                one_hot=True,
                progress_label=None,
                apply_sigmoid=True,
            )
            last_train_results = train_results
        ran_train_eval_last_epoch = train_results is not None

        if summary_writer and train_results is not None and val_results is not None:
            update_graphs(
                summary_writer,
                epoch,
                train_results,
                val_results,
                mbatch_group=MBATCH_LOSS_GROUP,
                mbatch_count=len(train_loader),
                mbatch_losses=mbatch_losses or [],
            )
            for lr_group_index, param_group in enumerate(optimizer.param_groups):
                summary_writer.add_scalar(f"LR/group_{lr_group_index}", float(param_group["lr"]), epoch_index)

        if val_results is not None:
            current_metric = float(val_results[METRIC_F1])
            if current_metric > run_best_f1 + EARLY_STOPPING_MIN_DELTA:
                run_best_f1 = current_metric
                run_best_epoch = epoch_index
                run_best_threshold = tuned_threshold
                no_improve_eval_count = 0
                run_best_checkpoint = {
                    CKPT_MODEL_NAME: MODEL_NAME,
                    CKPT_STATE_DICT: net.state_dict(),
                    CKPT_BEST_VAL_F1: run_best_f1,
                    CKPT_BEST_VAL_LOSS: float(val_results[METRIC_LOSS]),
                    CKPT_BEST_VAL_ACCURACY: float(val_results[METRIC_ACCURACY]),
                    CKPT_BEST_VAL_PRECISION: float(val_results[METRIC_PRECISION]),
                    CKPT_BEST_VAL_RECALL: float(val_results[METRIC_RECALL]),
                    CKPT_BEST_EPOCH: run_best_epoch,
                    CKPT_TOTAL_EPOCHS: NUM_EPOCHS,
                    CKPT_BEST_THRESHOLD: run_best_threshold,
                    CKPT_BATCH_SIZE: train_batch_size_now,
                    "train_batch_size_frozen": TRAIN_BATCH_SIZE_FROZEN,
                    "train_batch_size_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN,
                    "val_batch_size": VAL_BATCH_SIZE,
                    "grad_accum_steps_frozen": GRAD_ACCUM_STEPS_FROZEN,
                    "grad_accum_steps_unfrozen": GRAD_ACCUM_STEPS_UNFROZEN,
                    "effective_batch_frozen": TRAIN_BATCH_SIZE_FROZEN * GRAD_ACCUM_STEPS_FROZEN,
                    "effective_batch_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN * GRAD_ACCUM_STEPS_UNFROZEN,
                    "use_amp": USE_AMP,
                    CKPT_LEARNING_RATE: float(optimizer.param_groups[0]["lr"]),
                    CKPT_LEARNING_RATES: [float(group["lr"]) for group in optimizer.param_groups],
                    "use_differential_lr": USE_DIFFERENTIAL_LR,
                    "base_learning_rate": LEARNING_RATE if not USE_DIFFERENTIAL_LR else None,
                    "backbone_base_lr": BACKBONE_BASE_LR if USE_DIFFERENTIAL_LR else None,
                    "head_base_lr": HEAD_BASE_LR if USE_DIFFERENTIAL_LR else None,
                    "lr_milestones": LR_MILESTONES,
                    "lr_decay_factor": LR_DECAY_FACTOR,
                    "freeze_backbone_at_start": FREEZE_BACKBONE_AT_START,
                    "unfreeze_backbone_epoch": UNFREEZE_BACKBONE_EPOCH if should_unfreeze_later else None,
                    "unfreeze_last_n_backbone_layers": UNFREEZE_LAST_N_BACKBONE_LAYERS,
                    "training_plan_token": training_plan_token,
                    CKPT_THRESHOLD: TH_MULTI_LABEL,
                    "val_split": VAL_SPLIT,
                    "train_metrics_every_n_epochs": TRAIN_METRICS_EVERY_N_EPOCHS,
                    "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
                    "seed": SEED,
                    "num_workers": NUM_WORKERS,
                    "num_classes": NUM_CLASSES,
                    "early_stopping_enabled": EARLY_STOPPING_ENABLED,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                }
                print(
                    f"New run-best F1 at epoch {run_best_epoch}: "
                    f"val_f1={run_best_f1:.4f}, th={run_best_threshold:.2f}"
                )
            else:
                no_improve_eval_count += 1

        train_f1_text = f"{float(train_results[METRIC_F1]):.4f}" if train_results is not None else "skipped"
        val_f1_text = f"{float(val_results[METRIC_F1]):.4f}" if val_results is not None else "skipped"
        epoch_seconds = int(time.perf_counter() - epoch_start)
        print(
            f"Done: Epoch {epoch_index}/{NUM_EPOCHS} "
            f"mode={current_mode_text} "
            f"bs={train_batch_size_now} "
            f"accum={grad_accum_steps_now} "
            f"lr={current_lr_text} "
            f"train_f1={train_f1_text} "
            f"val_f1={val_f1_text} "
            f"time taken = {epoch_seconds} seconds"
        )
        if scheduler is not None:
            scheduler.step()
        completed_epochs = epoch_index

        if EARLY_STOPPING_ENABLED and val_results is not None and no_improve_eval_count >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            early_stop_reason = (
                f"validation F1 did not improve by > {EARLY_STOPPING_MIN_DELTA} for "
                f"{EARLY_STOPPING_PATIENCE} validation checks"
            )
            print(f"Early stopping at epoch {epoch_index}/{NUM_EPOCHS}: " f"{early_stop_reason}")
            break

    if completed_epochs > 0 and not ran_train_eval_last_epoch:
        final_train_eval_threshold = last_tuned_threshold if last_val_results is not None else run_best_threshold
        print("Running final train evaluation at training end " f"(threshold={final_train_eval_threshold:.2f})...")
        last_train_results = validation_loop(
            train_loader,
            net,
            criterion,
            NUM_CLASSES,
            device,
            multi_label=True,
            th_multi_label=final_train_eval_threshold,
            one_hot=True,
            progress_label="    Final Train Eval",
            apply_sigmoid=True,
        )

    if run_best_checkpoint is None:
        if summary_writer:
            summary_writer.close()
        print("No validation results were produced; no checkpoint saved.")
        return

    run_best_checkpoint["run_id"] = run_id
    run_best_checkpoint["run_started_at_utc"] = run_started_at.isoformat()
    run_best_checkpoint["run_output_dir"] = str(run_output_dir)

    run_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(run_best_checkpoint, run_model_path)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(run_best_checkpoint, MODEL_PATH)

    net.load_state_dict(run_best_checkpoint[CKPT_STATE_DICT])
    net.eval()
    selected_threshold = checkpoint_inference_threshold(run_best_checkpoint, TH_MULTI_LABEL)
    selected_train_results = validation_loop(
        train_loader,
        net,
        criterion,
        NUM_CLASSES,
        device,
        multi_label=True,
        th_multi_label=selected_threshold,
        one_hot=True,
        progress_label="    Train Eval",
        apply_sigmoid=True,
    )
    selected_train_f1 = float(selected_train_results[METRIC_F1])
    selected_val_f1 = float(run_best_checkpoint[CKPT_BEST_VAL_F1])

    run_confusion_matrix_existed_before = run_confusion_matrix_path.exists()
    confusion_matrix_error = None
    confusion_matrix_summary = None
    confusion_matrix_batch_summary = None
    try:
        confusion_matrix_batch_summary = generate_missing_confusion_matrices_for_runs(
            runs_root=TRAINED_MODELS_ROOT,
            checkpoint_filename=ACTIVE_CHECKPOINT_FILENAME,
            confusion_matrix_filename=RUN_CONFUSION_MATRIX_FILENAME,
            overwrite=False,
            split="val",
            val_split=VAL_SPLIT,
            seed=SEED,
            batch_size=VAL_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            th_multi_label=None,
            normalize="rows",
            top_k_classes=40,
        )
        run_entry = None
        for item in confusion_matrix_batch_summary.get("runs", []):
            if item.get("checkpoint_path") == str(run_model_path):
                run_entry = item
                break

        if run_entry is not None:
            confusion_matrix_summary = run_entry
        else:
            confusion_matrix_summary = {
                "checkpoint_path": str(run_model_path),
                "output_path": str(run_confusion_matrix_path),
                "model_name": MODEL_NAME,
                "generated": run_confusion_matrix_path.exists() and not run_confusion_matrix_existed_before,
                "skipped_existing": run_confusion_matrix_existed_before,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001
        confusion_matrix_error = str(exc)

    run_predictions_existed_before = run_predictions_path.exists()
    predictions_error = None
    predictions_summary = None
    predictions_batch_summary = None
    try:
        predictions_batch_summary = generate_missing_predictions_for_runs(
            runs_root=TRAINED_MODELS_ROOT,
            checkpoint_filename=ACTIVE_CHECKPOINT_FILENAME,
            predictions_filename=RUN_PREDICTIONS_FILENAME,
            overwrite=False,
            batch_size=VAL_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            th_multi_label=TH_MULTI_LABEL,
        )
        run_entry = None
        for item in predictions_batch_summary.get("runs", []):
            if item.get("checkpoint_path") == str(run_model_path):
                run_entry = item
                break

        if run_entry is not None:
            predictions_summary = run_entry
        else:
            predictions_summary = {
                "checkpoint_path": str(run_model_path),
                "output_path": str(run_predictions_path),
                "model_name": MODEL_NAME,
                "generated": run_predictions_path.exists() and not run_predictions_existed_before,
                "skipped_existing": run_predictions_existed_before,
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001
        predictions_error = str(exc)

    run_finished_at = datetime.now(timezone.utc)
    run_duration_seconds = round((run_finished_at - run_started_at).total_seconds(), 3)
    run_best_checkpoint["run_finished_at_utc"] = run_finished_at.isoformat()
    run_best_checkpoint[CKPT_RUN_DURATION_SECONDS] = run_duration_seconds
    torch.save(run_best_checkpoint, run_model_path)
    torch.save(run_best_checkpoint, MODEL_PATH)

    run_metadata = {
        "run": {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "started_at_utc": run_started_at.isoformat(),
            "finished_at_utc": run_finished_at.isoformat(),
            "duration_seconds": run_duration_seconds,
        },
        "paths": {
            "run_output_dir": run_output_dir,
            "run_checkpoint_path": run_model_path,
            "run_confusion_matrix_path": run_confusion_matrix_path,
            "run_predictions_path": run_predictions_path,
            "active_checkpoint_path": MODEL_PATH,
        },
        "configuration": {
            "num_classes": NUM_CLASSES,
            "epochs": NUM_EPOCHS,
            "train_batch_size_frozen": TRAIN_BATCH_SIZE_FROZEN,
            "train_batch_size_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN,
            "val_batch_size": VAL_BATCH_SIZE,
            "grad_accum_steps_frozen": GRAD_ACCUM_STEPS_FROZEN,
            "grad_accum_steps_unfrozen": GRAD_ACCUM_STEPS_UNFROZEN,
            "effective_batch_frozen": TRAIN_BATCH_SIZE_FROZEN * GRAD_ACCUM_STEPS_FROZEN,
            "effective_batch_unfrozen": TRAIN_BATCH_SIZE_UNFROZEN * GRAD_ACCUM_STEPS_UNFROZEN,
            "use_amp": USE_AMP,
            "amp_dtype": AMP_DTYPE,
            "freeze_backbone_at_start": FREEZE_BACKBONE_AT_START,
            "unfreeze_backbone_epoch": UNFREEZE_BACKBONE_EPOCH if should_unfreeze_later else None,
            "unfreeze_last_n_backbone_layers": UNFREEZE_LAST_N_BACKBONE_LAYERS,
            "use_differential_lr": USE_DIFFERENTIAL_LR,
            "learning_rate": LEARNING_RATE if not USE_DIFFERENTIAL_LR else None,
            "backbone_base_lr": BACKBONE_BASE_LR if USE_DIFFERENTIAL_LR else None,
            "head_base_lr": HEAD_BASE_LR if USE_DIFFERENTIAL_LR else None,
            "lr_milestones": LR_MILESTONES,
            "lr_decay_factor": LR_DECAY_FACTOR,
            "val_split": VAL_SPLIT,
            "seed": SEED,
            "num_workers": NUM_WORKERS,
            "th_multi_label_initial": TH_MULTI_LABEL,
            "threshold_candidates": THRESHOLD_CANDIDATES,
            "train_metrics_every_n_epochs": TRAIN_METRICS_EVERY_N_EPOCHS,
            "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
            "early_stopping_enabled": EARLY_STOPPING_ENABLED,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "use_tensorboard": USE_TENSORBOARD and TENSORBOARD_AVAILABLE,
            "training_plan_token": training_plan_token,
            "batch_size_token": batch_size_token,
        },
        "dataset": {
            "full_dataset_size": len(full_dataset),
            "train_size": len(train_set),
            "val_size": len(val_set),
            "train_images_dir": TRAIN_IMAGES_DIR,
            "train_labels_dir": TRAIN_LABELS_DIR,
        },
        "results": {
            "best_epoch": int(run_best_checkpoint[CKPT_BEST_EPOCH]),
            "total_epochs": int(run_best_checkpoint[CKPT_TOTAL_EPOCHS]),
            "best_val_f1": selected_val_f1,
            "best_val_loss": float(run_best_checkpoint.get(CKPT_BEST_VAL_LOSS, 0.0)),
            "best_val_accuracy": float(run_best_checkpoint.get(CKPT_BEST_VAL_ACCURACY, 0.0)),
            "best_val_precision": float(run_best_checkpoint.get(CKPT_BEST_VAL_PRECISION, 0.0)),
            "best_val_recall": float(run_best_checkpoint.get(CKPT_BEST_VAL_RECALL, 0.0)),
            "best_threshold": selected_threshold,
            "selected_train_f1_eval": selected_train_f1,
            "last_train_f1": float(last_train_results[METRIC_F1]) if last_train_results else None,
            "last_val_f1": float(last_val_results[METRIC_F1]) if last_val_results else None,
            "completed_epochs": completed_epochs,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
        },
        "artifacts": {
            "checkpoint_file": run_model_path.name,
            "config_file": run_config_path.name,
            "confusion_matrix_file": run_confusion_matrix_path.name,
            "confusion_matrix_generated": (
                bool(confusion_matrix_summary.get("generated")) if confusion_matrix_summary else False
            ),
            "confusion_matrix_exists": run_confusion_matrix_path.exists(),
            "confusion_matrix_batch_summary": confusion_matrix_batch_summary,
            "confusion_matrix_summary": confusion_matrix_summary,
            "confusion_matrix_error": confusion_matrix_error,
            "predictions_file": run_predictions_path.name,
            "predictions_generated": bool(predictions_summary.get("generated")) if predictions_summary else False,
            "predictions_exists": run_predictions_path.exists(),
            "predictions_batch_summary": predictions_batch_summary,
            "predictions_summary": predictions_summary,
            "predictions_error": predictions_error,
            "tensorboard_log_dir": tensorboard_log_dir,
        },
    }
    write_json(run_config_path, run_metadata)

    summary_items = {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "run_dir": run_output_dir,
        "best_epoch": f"{run_best_checkpoint[CKPT_BEST_EPOCH]}of{run_best_checkpoint[CKPT_TOTAL_EPOCHS]}",
        "best_val_f1": f"{selected_val_f1:.4f}",
        "best_threshold": f"{selected_threshold:.2f}",
        "selected_train_f1_eval": f"{selected_train_f1:.4f}",
        "completed_epochs": completed_epochs,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason,
        "run_checkpoint_path": run_model_path,
        "run_config_path": run_config_path,
        "run_confusion_matrix_path": (
            run_confusion_matrix_path if confusion_matrix_error is None else f"FAILED ({confusion_matrix_error})"
        ),
        "run_predictions_path": run_predictions_path if predictions_error is None else f"FAILED ({predictions_error})",
        "active_checkpoint_path": MODEL_PATH,
    }
    print_section("TRAINING SUMMARY", summary_items)

    if summary_writer:
        log_hparams_summary(
            summary_writer,
            hparams=tensorboard_hparams,
            metrics={
                "hparams/best_val_f1": float(selected_val_f1),
                "hparams/best_val_loss": float(run_best_checkpoint.get(CKPT_BEST_VAL_LOSS, 0.0)),
                "hparams/best_val_accuracy": float(run_best_checkpoint.get(CKPT_BEST_VAL_ACCURACY, 0.0)),
                "hparams/best_val_precision": float(run_best_checkpoint.get(CKPT_BEST_VAL_PRECISION, 0.0)),
                "hparams/best_val_recall": float(run_best_checkpoint.get(CKPT_BEST_VAL_RECALL, 0.0)),
                "hparams/selected_train_f1_eval": float(selected_train_f1),
                "hparams/completed_epochs": float(completed_epochs),
            },
        )
        summary_writer.close()


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")
    if UNFREEZE_LAST_N_BACKBONE_LAYERS is not None and UNFREEZE_LAST_N_BACKBONE_LAYERS < 1:
        raise ValueError("UNFREEZE_LAST_N_BACKBONE_LAYERS must be None or an integer >= 1.")

    t = time.perf_counter()
    main()
    print(f"Time taken: {time.perf_counter() - t: .2f}")
