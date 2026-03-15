import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from utils.config import BEST_MODEL_PATH, CLASSES, MODEL_NAME, NUM_CLASSES, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR
from utils.dataset_readers import COCOTrainImageDataset
from utils.metadata_utils import checkpoint_inference_threshold, checkpoint_model_name
from utils.models_factory import AVAILABLE_MODELS, MODEL_SPECS, create_model
from utils.references import CKPT_STATE_DICT
from utils.training_utils import ProgressBar, print_section


MODEL_PATH = BEST_MODEL_PATH
RUNS_ROOT = BEST_MODEL_PATH.parent

# Dataset split options: "val", "train", "all"
SPLIT = "val"
VAL_SPLIT = 0.05

SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 0

# None => checkpoint best_threshold/th_multi_label/0.5 fallback.
TH_MULTI_LABEL = None

# Plot options: "none", "rows", "all"
NORMALIZE = "rows"
TOP_K_CLASSES = 40  # 0 => render all classes
OUTPUT_PATH = Path("trained_models") / "confusion_matrix.png"

# Batch generation over run folders:
GENERATE_FOR_ALL_RUNS = True
RUN_CHECKPOINT_FILENAME = "best_model.pt"
RUN_CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
OVERWRITE_EXISTING = False


def _resolve_transform(model_name: str):
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(AVAILABLE_MODELS)}")
    return spec.weights.transforms()


def _validate_args(
    *,
    model_path: Path,
    split: str,
    val_split: float,
    batch_size: int,
    top_k_classes: int,
    normalize: str,
    th_multi_label: float | None,
) -> None:
    if split not in {"val", "train", "all"}:
        raise ValueError("SPLIT must be one of: 'val', 'train', 'all'.")
    if val_split <= 0 or val_split >= 1:
        raise ValueError("VAL_SPLIT must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("BATCH_SIZE must be >= 1.")
    if top_k_classes < 0:
        raise ValueError("TOP_K_CLASSES must be >= 0.")
    if normalize not in {"none", "rows", "all"}:
        raise ValueError("NORMALIZE must be one of: 'none', 'rows', 'all'.")
    if th_multi_label is not None and (th_multi_label < 0 or th_multi_label > 1):
        raise ValueError("TH_MULTI_LABEL must be between 0 and 1 when set.")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")


def _select_dataset(
    full_dataset: torch.utils.data.Dataset,
    *,
    split: str,
    val_split: float,
    seed: int,
) -> torch.utils.data.Dataset:
    if split == "all":
        return full_dataset

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return val_set if split == "val" else train_set


def _update_pairwise_confusion_matrix(
    matrix: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> None:
    predictions = torch.where(scores > threshold, 1.0, 0.0)
    top1_predictions = torch.argmax(scores, dim=1)

    for sample_idx in range(labels.shape[0]):
        true_indices = labels[sample_idx].nonzero(as_tuple=False).squeeze(1)
        if true_indices.numel() == 0:
            continue

        pred_indices = predictions[sample_idx].nonzero(as_tuple=False).squeeze(1)
        if pred_indices.numel() == 0:
            pred_indices = torch.tensor([int(top1_predictions[sample_idx])], dtype=torch.long)

        pred_list = pred_indices.tolist()
        for true_idx in true_indices.tolist():
            for pred_idx in pred_list:
                matrix[true_idx, pred_idx] += 1


def _build_plot_data(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...],
    normalize: str,
    top_k_classes: int,
) -> tuple[torch.Tensor, list[str]]:
    support = confusion_matrix.sum(dim=1)
    selected_indices = torch.arange(confusion_matrix.size(0))
    if top_k_classes > 0 and top_k_classes < confusion_matrix.size(0):
        selected_indices = torch.topk(support, top_k_classes).indices
        selected_indices, _ = torch.sort(selected_indices)

    matrix = confusion_matrix.index_select(0, selected_indices).index_select(1, selected_indices).float()
    if normalize == "rows":
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = torch.where(row_sums > 0, matrix / row_sums, torch.zeros_like(matrix))
    elif normalize == "all":
        total = matrix.sum()
        matrix = matrix / total if total > 0 else torch.zeros_like(matrix)

    labels = [class_names[int(index)] for index in selected_indices]
    return matrix, labels


def _plot_confusion_matrix(
    matrix: torch.Tensor,
    labels: list[str],
    *,
    title: str,
    normalize: str,
    output_path: Path,
) -> None:
    classes_count = len(labels)
    fig_size = max(8.0, min(30.0, classes_count * 0.35))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(classes_count))
    ax.set_yticks(range(classes_count))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    annotate = classes_count <= 25
    if annotate:
        for i in range(classes_count):
            for j in range(classes_count):
                value = matrix[i, j].item()
                text = f"{value:.2f}" if normalize != "none" else str(int(round(value)))
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_confusion_matrix_for_checkpoint(
    *,
    model_path: Path,
    output_path: Path,
    split: str = "val",
    val_split: float = 0.05,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    th_multi_label: float | None = None,
    normalize: str = "rows",
    top_k_classes: int = 40,
    progress_label: str = "    Evaluating",
    print_config: bool = True,
    print_summary: bool = True,
) -> dict[str, object]:
    _validate_args(
        model_path=model_path,
        split=split,
        val_split=val_split,
        batch_size=batch_size,
        top_k_classes=top_k_classes,
        normalize=normalize,
        th_multi_label=th_multi_label,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu")
    model_name = checkpoint_model_name(checkpoint, MODEL_NAME)
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available: {', '.join(AVAILABLE_MODELS)}")

    threshold = th_multi_label if th_multi_label is not None else checkpoint_inference_threshold(checkpoint, 0.5)

    if print_config:
        config_items = {
            "model_path": model_path,
            "model_name": model_name,
            "split": split,
            "val_split": val_split,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "threshold": threshold,
            "normalize": normalize,
            "top_k_classes": top_k_classes if top_k_classes > 0 else "all",
            "device": device.type,
            "output_path": output_path,
        }
        print_section("CONFUSION MATRIX CONFIG", config_items)

    transform = _resolve_transform(model_name)
    dataset = COCOTrainImageDataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, transform=transform)
    selected_dataset = _select_dataset(
        dataset,
        split=split,
        val_split=val_split,
        seed=seed,
    )
    if len(selected_dataset) == 0:
        raise RuntimeError("Selected dataset split is empty; cannot generate a confusion matrix.")
    dataloader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    net, _, _ = create_model(model_name, NUM_CLASSES, pretrained=False)
    net.load_state_dict(checkpoint[CKPT_STATE_DICT])
    net = net.to(device)
    net.eval()

    confusion_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    progress_bar = ProgressBar(total=len(dataloader), start_at=0, label=progress_label)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = net(images)
            scores = torch.sigmoid(logits)
            _update_pairwise_confusion_matrix(confusion_matrix, scores.cpu(), labels.cpu(), float(threshold))
            progress_bar.increment()
    progress_bar.finish()

    plot_matrix, plot_labels = _build_plot_data(
        confusion_matrix,
        CLASSES,
        normalize=normalize,
        top_k_classes=top_k_classes,
    )
    plot_title = f"Pairwise confusion matrix ({model_name}, split={split}, th={threshold:.2f})"
    _plot_confusion_matrix(
        plot_matrix,
        plot_labels,
        title=plot_title,
        normalize=normalize,
        output_path=output_path,
    )

    summary = {
        "model_name": model_name,
        "threshold": float(threshold),
        "split": split,
        "num_samples": len(selected_dataset),
        "classes_rendered": len(plot_labels),
        "output_path": str(output_path),
    }
    if print_summary:
        print_section("CONFUSION MATRIX SUMMARY", summary)
    return summary


def ensure_confusion_matrix_for_checkpoint(
    *,
    model_path: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
    split: str = "val",
    val_split: float = 0.05,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    th_multi_label: float | None = None,
    normalize: str = "rows",
    top_k_classes: int = 40,
    progress_label: str = "    Evaluating",
    print_config: bool = True,
    print_summary: bool = True,
) -> dict[str, object]:
    resolved_output_path = output_path if output_path is not None else model_path.parent / RUN_CONFUSION_MATRIX_FILENAME
    if resolved_output_path.exists() and not overwrite:
        summary = {
            "model_name": "unknown",
            "threshold": None,
            "split": split,
            "num_samples": 0,
            "classes_rendered": 0,
            "output_path": str(resolved_output_path),
            "generated": False,
            "skipped_existing": True,
        }
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            summary["model_name"] = checkpoint_model_name(checkpoint, MODEL_NAME)
            summary["threshold"] = checkpoint_inference_threshold(checkpoint, 0.5)
        except Exception:
            pass
        if print_summary:
            print_section("CONFUSION MATRIX SUMMARY", summary)
        return summary

    summary = generate_confusion_matrix_for_checkpoint(
        model_path=model_path,
        output_path=resolved_output_path,
        split=split,
        val_split=val_split,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        th_multi_label=th_multi_label,
        normalize=normalize,
        top_k_classes=top_k_classes,
        progress_label=progress_label,
        print_config=print_config,
        print_summary=print_summary,
    )
    summary["generated"] = True
    summary["skipped_existing"] = False
    return summary


def generate_missing_confusion_matrices_for_runs(
    *,
    runs_root: Path,
    checkpoint_filename: str = "best_model.pt",
    confusion_matrix_filename: str = "confusion_matrix.png",
    overwrite: bool = False,
    split: str = "val",
    val_split: float = 0.05,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    th_multi_label: float | None = None,
    normalize: str = "rows",
    top_k_classes: int = 40,
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

    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        output_path = checkpoint_path.parent / confusion_matrix_filename
        run_label = f"    Confusion [{index}/{total}]"
        try:
            summary = ensure_confusion_matrix_for_checkpoint(
                model_path=checkpoint_path,
                output_path=output_path,
                overwrite=overwrite,
                split=split,
                val_split=val_split,
                seed=seed,
                batch_size=batch_size,
                num_workers=num_workers,
                th_multi_label=th_multi_label,
                normalize=normalize,
                top_k_classes=top_k_classes,
                progress_label=run_label,
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
                    "error": None,
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
        "split": split,
        "val_split": val_split,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "normalize": normalize,
        "top_k_classes": top_k_classes,
        "failures": failures,
        "runs": runs,
    }
    print_section("CONFUSION MATRIX BATCH SUMMARY", summary)
    return summary


def main() -> None:
    if GENERATE_FOR_ALL_RUNS:
        summary = generate_missing_confusion_matrices_for_runs(
            runs_root=RUNS_ROOT,
            checkpoint_filename=RUN_CHECKPOINT_FILENAME,
            confusion_matrix_filename=RUN_CONFUSION_MATRIX_FILENAME,
            overwrite=OVERWRITE_EXISTING,
            split=SPLIT,
            val_split=VAL_SPLIT,
            seed=SEED,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            th_multi_label=TH_MULTI_LABEL,
            normalize=NORMALIZE,
            top_k_classes=TOP_K_CLASSES,
        )
        if summary["total_runs"] > 0:
            return

    ensure_confusion_matrix_for_checkpoint(
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        overwrite=OVERWRITE_EXISTING,
        split=SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        th_multi_label=TH_MULTI_LABEL,
        normalize=NORMALIZE,
        top_k_classes=TOP_K_CLASSES,
    )


if __name__ == "__main__":
    main()
