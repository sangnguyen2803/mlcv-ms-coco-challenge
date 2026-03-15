from collections.abc import Mapping
from typing import Any

from utils.references import (
    CKPT_BEST_EPOCH,
    CKPT_BEST_THRESHOLD,
    CKPT_MODEL_NAME,
    CKPT_NUM_EPOCHS_LEGACY,
    CKPT_THRESHOLD,
    CKPT_TOTAL_EPOCHS,
)


def checkpoint_model_name(checkpoint: Mapping[str, Any], default_model_name: str) -> str:
    return str(checkpoint.get(CKPT_MODEL_NAME, default_model_name))


def checkpoint_total_epochs(checkpoint: Mapping[str, Any]) -> int | None:
    value = checkpoint.get(CKPT_TOTAL_EPOCHS, checkpoint.get(CKPT_NUM_EPOCHS_LEGACY))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def checkpoint_inference_threshold(checkpoint: Mapping[str, Any], default_threshold: float) -> float:
    train_threshold = checkpoint.get(CKPT_THRESHOLD)
    fallback = train_threshold if train_threshold is not None else default_threshold
    value = checkpoint.get(CKPT_BEST_THRESHOLD, fallback)
    return float(value)


def checkpoint_epoch_token(checkpoint: Mapping[str, Any]) -> str:
    best_epoch = checkpoint.get(CKPT_BEST_EPOCH)
    total_epochs = checkpoint_total_epochs(checkpoint)
    if best_epoch is None:
        return "n/a"
    if total_epochs is not None:
        return f"{best_epoch}of{total_epochs}"
    return str(best_epoch)


def epoch_token(best_epoch: int | None, total_epochs: int | None) -> str:
    if best_epoch is None:
        return "na"
    if total_epochs is not None:
        return f"{best_epoch}of{total_epochs}"
    return str(best_epoch)
