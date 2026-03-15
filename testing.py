import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.config import BEST_MODEL_PATH, MODEL_NAME, NUM_CLASSES, TEST_IMAGES_DIR
from utils.dataset_readers import COCOTestImageDataset
from utils.metadata_utils import (
    checkpoint_epoch_token,
    checkpoint_inference_threshold,
    checkpoint_model_name,
    checkpoint_total_epochs,
    epoch_token,
)
from utils.models_factory import AVAILABLE_MODELS, create_model
from utils.references import (
    CKPT_BATCH_SIZE,
    CKPT_BEST_EPOCH,
    CKPT_BEST_THRESHOLD,
    CKPT_BEST_VAL_F1,
    CKPT_LEARNING_RATE,
    CKPT_LEARNING_RATES,
    CKPT_STATE_DICT,
    CKPT_THRESHOLD,
)
from utils.training_utils import ProgressBar, print_section, tokenize_float


# Testing configuration.
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0
TH_MULTI_LABEL: float = 0.5

MODEL_PATH: Path = BEST_MODEL_PATH
RUNS_ROOT: Path = BEST_MODEL_PATH.parent

OUTPUT_PATH: Path = Path("predictions.json")
RUN_CHECKPOINT_FILENAME: str = "best_model.pt"
RUN_PREDICTIONS_FILENAME: str = "predictions.json"

GENERATE_FOR_ALL_RUNS: bool = True
OVERWRITE_EXISTING: bool = False


def build_predictions_path(
    base_path: Path,
    model_name: str,
    estimated_f1: float | None,
    best_epoch: int | None,
    total_epochs: int | None,
    train_batch_size: int | None,
    train_learning_rate: float | None,
    train_th_multi_label: float | None,
    train_best_threshold: float | None,
    test_batch_size: int,
    test_th_multi_label: float,
) -> Path:
    suffix = base_path.suffix or ".json"
    stem = base_path.stem
    f1_token = tokenize_float(estimated_f1) if estimated_f1 is not None else "na"
    epoch_value = epoch_token(best_epoch, total_epochs)
    train_bs_token = str(train_batch_size) if train_batch_size is not None else "na"
    train_lr_token = tokenize_float(train_learning_rate, precision=6) if train_learning_rate is not None else "na"
    train_th_token = tokenize_float(train_th_multi_label, precision=3) if train_th_multi_label is not None else "na"
    best_th_token = tokenize_float(train_best_threshold, precision=3) if train_best_threshold is not None else "na"
    file_name = (
        f"{stem}_{model_name}"
        f"_f1-{f1_token}"
        f"_ep-{epoch_value}"
        f"_bs-{train_bs_token}"
        f"_lr-{train_lr_token}"
        f"_th-{train_th_token}"
        f"_bth-{best_th_token}"
        f"_testbs-{test_batch_size}"
        f"_testth-{tokenize_float(test_th_multi_label, precision=3)}"
        f"{suffix}"
    )
    return base_path.with_name(file_name)


def run_testing_for_checkpoint(
    *,
    model_path: Path,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    th_multi_label: float = 0.5,
    auto_name_output: bool = True,
    print_config: bool = True,
    print_summary: bool = True,
) -> dict[str, object]:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be >= 1.")
    if num_workers < 0:
        raise ValueError("NUM_WORKERS must be >= 0.")
    if th_multi_label < 0 or th_multi_label > 1:
        raise ValueError("TH_MULTI_LABEL must be between 0 and 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint_model_name(checkpoint, MODEL_NAME)
    estimated_f1 = checkpoint.get(CKPT_BEST_VAL_F1)
    best_epoch = checkpoint.get(CKPT_BEST_EPOCH)
    total_epochs = checkpoint_total_epochs(checkpoint)
    train_batch_size = checkpoint.get(CKPT_BATCH_SIZE)
    train_learning_rate = checkpoint.get(CKPT_LEARNING_RATE)
    train_learning_rates = checkpoint.get(CKPT_LEARNING_RATES)
    train_th_multi_label = checkpoint.get(CKPT_THRESHOLD)
    train_best_threshold = checkpoint.get(CKPT_BEST_THRESHOLD)
    inference_threshold = checkpoint_inference_threshold(checkpoint, th_multi_label)

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {', '.join(AVAILABLE_MODELS)}")

    resolved_output_path = (
        build_predictions_path(
            output_path,
            model_name=model_name,
            estimated_f1=float(estimated_f1) if estimated_f1 is not None else None,
            best_epoch=int(best_epoch) if best_epoch is not None else None,
            total_epochs=int(total_epochs) if total_epochs is not None else None,
            train_batch_size=int(train_batch_size) if train_batch_size is not None else None,
            train_learning_rate=float(train_learning_rate) if train_learning_rate is not None else None,
            train_th_multi_label=float(train_th_multi_label) if train_th_multi_label is not None else None,
            train_best_threshold=float(train_best_threshold) if train_best_threshold is not None else None,
            test_batch_size=batch_size,
            test_th_multi_label=inference_threshold,
        )
        if auto_name_output
        else output_path
    )

    if print_config:
        test_config = {
            "model_name": model_name,
            "device": device.type,
            "checkpoint_path": model_path,
            "estimated_best_val_f1": f"{float(estimated_f1):.4f}" if estimated_f1 is not None else "n/a",
            "estimated_best_epoch": checkpoint_epoch_token(checkpoint),
            "total_epochs(from_ckpt)": total_epochs if total_epochs is not None else "n/a",
            "train_batch_size(from_ckpt)": train_batch_size if train_batch_size is not None else "n/a",
            "train_learning_rate(from_ckpt)": train_learning_rate if train_learning_rate is not None else "n/a",
            "train_learning_rates(from_ckpt)": train_learning_rates if train_learning_rates is not None else "n/a",
            "train_th_multi_label(from_ckpt)": train_th_multi_label if train_th_multi_label is not None else "n/a",
            "train_best_threshold(from_ckpt)": train_best_threshold if train_best_threshold is not None else "n/a",
            "test_batch_size": batch_size,
            "test_th_multi_label": inference_threshold,
            "test_num_workers": num_workers,
            "output_path": resolved_output_path,
        }
        print_section("TESTING START CONFIG", test_config)

    net, transform, _ = create_model(model_name, NUM_CLASSES, pretrained=True)
    if transform is None:
        raise RuntimeError("No transform available. Use a pretrained model or provide a custom transform.")

    test_dataset = COCOTestImageDataset(TEST_IMAGES_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    net.load_state_dict(checkpoint[CKPT_STATE_DICT])
    net = net.to(device)
    net.eval()

    output: dict[str, list[int]] = {}
    progress_bar = None
    total_batches = len(test_loader)
    if total_batches > 0:
        progress_bar = ProgressBar(total=total_batches, start_at=0, label="    Testing")
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = net(images)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > inference_threshold
            for i, name in enumerate(names):
                indices = predictions[i].nonzero(as_tuple=False).squeeze(1).tolist()
                output[name] = indices
            if progress_bar:
                progress_bar.increment()
    if progress_bar:
        progress_bar.finish()

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    summary = {
        "model_name": model_name,
        "checkpoint_path": str(model_path),
        "estimated_best_val_f1": f"{float(estimated_f1):.4f}" if estimated_f1 is not None else "n/a",
        "estimated_best_epoch": checkpoint_epoch_token(checkpoint),
        "total_epochs": total_epochs if total_epochs is not None else "n/a",
        "test_threshold": float(inference_threshold),
        "num_test_images": len(test_dataset),
        "output_path": str(resolved_output_path),
    }
    if print_summary:
        print_section("TESTING SUMMARY", summary)
    return summary


def ensure_predictions_for_checkpoint(
    *,
    model_path: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    th_multi_label: float = 0.5,
    auto_name_output: bool = False,
    print_config: bool = True,
    print_summary: bool = True,
) -> dict[str, object]:
    resolved_output_path = output_path if output_path is not None else model_path.parent / RUN_PREDICTIONS_FILENAME
    if resolved_output_path.exists() and not overwrite:
        summary = {
            "model_name": "unknown",
            "checkpoint_path": str(model_path),
            "output_path": str(resolved_output_path),
            "generated": False,
            "skipped_existing": True,
            "error": None,
        }
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            summary["model_name"] = checkpoint_model_name(checkpoint, MODEL_NAME)
        except Exception:
            pass
        if print_summary:
            print_section("TESTING SUMMARY", summary)
        return summary

    summary = run_testing_for_checkpoint(
        model_path=model_path,
        output_path=resolved_output_path,
        batch_size=batch_size,
        num_workers=num_workers,
        th_multi_label=th_multi_label,
        auto_name_output=auto_name_output,
        print_config=print_config,
        print_summary=print_summary,
    )
    summary["generated"] = True
    summary["skipped_existing"] = False
    summary["error"] = None
    return summary


def generate_missing_predictions_for_runs(
    *,
    runs_root: Path,
    checkpoint_filename: str = "best_model.pt",
    predictions_filename: str = "predictions.json",
    overwrite: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    th_multi_label: float = 0.5,
) -> dict[str, object]:
    checkpoint_paths = []
    for path in sorted(runs_root.glob(f"**/{checkpoint_filename}")):
        if path.parent == runs_root:
            continue
        checkpoint_paths.append(path)

    total = len(checkpoint_paths)
    generated = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    runs: list[dict[str, object]] = []

    for checkpoint_path in checkpoint_paths:
        output_path = checkpoint_path.parent / predictions_filename
        try:
            summary = ensure_predictions_for_checkpoint(
                model_path=checkpoint_path,
                output_path=output_path,
                overwrite=overwrite,
                batch_size=batch_size,
                num_workers=num_workers,
                th_multi_label=th_multi_label,
                auto_name_output=False,
                print_config=False,
                print_summary=False,
            )
            if summary.get("generated"):
                generated += 1
            else:
                skipped += 1
            runs.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "output_path": str(output_path),
                    "model_name": summary.get("model_name"),
                    "generated": bool(summary.get("generated")),
                    "skipped_existing": bool(summary.get("skipped_existing")),
                    "error": summary.get("error"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            run_error = str(exc)
            runs.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "output_path": str(output_path),
                    "model_name": None,
                    "generated": False,
                    "skipped_existing": False,
                    "error": run_error,
                }
            )
            failures.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "error": run_error,
                }
            )

    summary = {
        "runs_root": str(runs_root),
        "total_runs": total,
        "generated": generated,
        "skipped_existing": skipped,
        "failed": failed,
        "overwrite_existing": overwrite,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "th_multi_label": th_multi_label,
        "failures": failures,
        "runs": runs,
    }
    print_section("TESTING BATCH SUMMARY", summary)
    return summary


def main() -> None:
    if GENERATE_FOR_ALL_RUNS:
        summary = generate_missing_predictions_for_runs(
            runs_root=RUNS_ROOT,
            checkpoint_filename=RUN_CHECKPOINT_FILENAME,
            predictions_filename=RUN_PREDICTIONS_FILENAME,
            overwrite=OVERWRITE_EXISTING,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            th_multi_label=TH_MULTI_LABEL,
        )
        if summary["total_runs"] > 0:
            return

    ensure_predictions_for_checkpoint(
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        overwrite=OVERWRITE_EXISTING,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        th_multi_label=TH_MULTI_LABEL,
        auto_name_output=True,
    )


if __name__ == "__main__":
    if MODEL_NAME not in AVAILABLE_MODELS:
        raise ValueError(f"MODEL_NAME must be one of: {', '.join(AVAILABLE_MODELS)}")
    main()
