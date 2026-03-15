import json
from pathlib import Path

import torch

from utils.config import TRAINED_MODELS_FOLDER
from utils.references import CKPT_STATE_DICT

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas is required for this script. Install with: pip install pandas") from exc


# Programmatic controls
CHECKPOINTS_ROOT = TRAINED_MODELS_FOLDER
RUN_CONFIG_GLOB = "**/run_config.json"
CHECKPOINT_GLOB = "**/*.pt"
SKIP_DEFAULT_BEST_MODEL = True  # Skip root-level active checkpoint duplicates.
SORT_BY = "results.best_val_f1"
SORT_ASCENDING = False
GROUP_BY_COLUMNS: list[str] = []  # Example: ["run.model_name"] or ["configuration.freeze_backbone_at_start"]
GROUP_MODE = "best"  # Options: "best", "mean"
OUTPUT_CSV_NAME = "model_performance_table.csv"
OUTPUT_COLUMNS: list[str] = [
    "model_name",
    "model_path_relative",
    "results.best_val_f1",
    "results.best_val_accuracy",
    "results.best_val_precision",
    "results.best_val_recall",
    "run.duration_seconds",
    "results.completed_epochs",
    "configuration.learning_rate",
    "configuration.lr_decay_factor",
    "configuration.lr_milestones",
]


def _flatten_dict(data: dict[str, object], prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in data.items():
        key_name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=f"{key_name}."))
        else:
            flat[key_name] = value
    return flat


def _load_checkpoint_row(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint = {k: v for k, v in checkpoint.items() if k != CKPT_STATE_DICT}
    row = {
        "row_source": "checkpoint",
        "checkpoint_filename": path.name,
        "checkpoint_path": str(path),
    }
    row.update(_flatten_dict(checkpoint))
    return row


def _load_run_config_row(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)
    row = {
        "row_source": "run_config",
        "run_config_filename": path.name,
        "run_config_path": str(path),
        "run_dir": str(path.parent),
    }
    row.update(_flatten_dict(run_config))
    return row


def _collect_run_config_rows(root: Path, pattern: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob(pattern)):
        try:
            rows.append(_load_run_config_row(path))
        except Exception as exc:
            print(f"Skipping run config {path}: {exc}")
    return rows


def _collect_checkpoint_rows(root: Path, pattern: str, skip_default_best_model: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob(pattern)):
        if skip_default_best_model and path.name == "best_model.pt":
            continue
        try:
            rows.append(_load_checkpoint_row(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return rows


def _group_dataframe(df: pd.DataFrame, *, sort_by: str | None = None) -> pd.DataFrame:
    if not GROUP_BY_COLUMNS:
        return df
    missing = [col for col in GROUP_BY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot group by missing columns: {missing}")

    if GROUP_MODE == "best":
        if sort_by and sort_by in df.columns:
            ordered = df.sort_values(sort_by, ascending=SORT_ASCENDING)
            return ordered.groupby(GROUP_BY_COLUMNS, dropna=False, as_index=False).head(1).reset_index(drop=True)
        return df.groupby(GROUP_BY_COLUMNS, dropna=False, as_index=False).head(1).reset_index(drop=True)

    if GROUP_MODE == "mean":
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        keep_numeric = [col for col in numeric if col not in GROUP_BY_COLUMNS]
        grouped = df.groupby(GROUP_BY_COLUMNS, dropna=False)[keep_numeric].mean().reset_index()
        grouped["group_count"] = df.groupby(GROUP_BY_COLUMNS, dropna=False).size().to_numpy()
        return grouped

    raise ValueError(f"Unsupported GROUP_MODE='{GROUP_MODE}'. Use 'best' or 'mean'.")


def _normalize_epoch_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Backward compatibility: older checkpoints may only have num_epochs.
    if "results.total_epochs" not in df.columns and "results.num_epochs" in df.columns:
        df["results.total_epochs"] = df["results.num_epochs"]
    if "results.num_epochs" in df.columns:
        df = df.drop(columns=["results.num_epochs"])

    if "total_epochs" not in df.columns and "num_epochs" in df.columns:
        df["total_epochs"] = df["num_epochs"]
    if "num_epochs" in df.columns:
        df = df.drop(columns=["num_epochs"])
    return df


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "run.duration_seconds": ("run.duration_seconds", "run_duration_seconds"),
        "results.best_val_loss": ("results.best_val_loss", "best_val_loss"),
        "results.best_val_accuracy": ("results.best_val_accuracy", "best_val_accuracy"),
        "results.best_val_precision": ("results.best_val_precision", "best_val_precision"),
        "results.best_val_recall": ("results.best_val_recall", "best_val_recall"),
        "results.best_val_f1": ("results.best_val_f1", "best_val_f1", "configuration.best_val_f1"),
    }
    for target, candidates in mappings.items():
        if target not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[target] = df[candidate]
                    break
        if target not in df.columns:
            df[target] = pd.NA
    return df


def _normalize_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "results.completed_epochs": ("results.completed_epochs", "completed_epochs"),
        "configuration.learning_rate": ("configuration.learning_rate", "learning_rate"),
        "configuration.lr_decay_factor": ("configuration.lr_decay_factor", "lr_decay_factor"),
        "configuration.lr_milestones": ("configuration.lr_milestones", "lr_milestones"),
    }
    for target, candidates in mappings.items():
        if target not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[target] = df[candidate]
                    break
        if target not in df.columns:
            df[target] = pd.NA

    if "configuration.learning_rate" in df.columns:
        missing_lr = df["configuration.learning_rate"].isna()
        has_backbone = "configuration.backbone_base_lr" in df.columns
        has_head = "configuration.head_base_lr" in df.columns
        if has_backbone and has_head:
            both_present = df["configuration.backbone_base_lr"].notna() & df["configuration.head_base_lr"].notna()
            fill_mask = missing_lr & both_present
            df.loc[fill_mask, "configuration.learning_rate"] = (
                df.loc[fill_mask, "configuration.backbone_base_lr"].astype(str)
                + "/"
                + df.loc[fill_mask, "configuration.head_base_lr"].astype(str)
            )
    return df


def _to_relative_model_path(path_value: object, root: Path) -> object:
    if path_value is None or (isinstance(path_value, float) and pd.isna(path_value)):
        return pd.NA
    text = str(path_value)
    if not text:
        return pd.NA
    path = Path(text)
    if not path.is_absolute():
        return str(path).replace("\\", "/")

    try:
        relative = path.resolve().relative_to(root.resolve())
        return str(relative).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _normalize_identity_columns(df: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    model_name_candidates = ("run.model_name", "model_name", "configuration.model_name")
    if "model_name" not in df.columns:
        for candidate in model_name_candidates:
            if candidate in df.columns:
                df["model_name"] = df[candidate]
                break
    if "model_name" not in df.columns:
        df["model_name"] = pd.NA

    checkpoint_path_candidates = ("paths.run_checkpoint_path", "checkpoint_path")
    checkpoint_path_source = None
    for candidate in checkpoint_path_candidates:
        if candidate in df.columns:
            checkpoint_path_source = candidate
            break
    if checkpoint_path_source is None:
        df["model_path_relative"] = pd.NA
    else:
        df["model_path_relative"] = df[checkpoint_path_source].apply(lambda value: _to_relative_model_path(value, root))
    return df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS]


def _resolve_sort_column(df: pd.DataFrame) -> str | None:
    candidates = [SORT_BY]
    if SORT_BY != "results.best_val_f1":
        candidates.append("results.best_val_f1")
    if SORT_BY != "best_val_f1":
        candidates.append("best_val_f1")
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def main() -> None:
    root = Path(CHECKPOINTS_ROOT)
    output_csv_path = root / OUTPUT_CSV_NAME
    if not root.exists():
        raise SystemExit(f"Checkpoint root does not exist: {root}")

    print(f"Checkpoint root: {root.resolve()}")
    print(f"Run config glob: {RUN_CONFIG_GLOB}")
    print(f"Checkpoint glob: {CHECKPOINT_GLOB}")
    print(f"Skip default best_model.pt: {SKIP_DEFAULT_BEST_MODEL}")
    print()

    rows = _collect_run_config_rows(root, RUN_CONFIG_GLOB)
    if rows:
        print(f"Loaded {len(rows)} rows from run_config.json files.")
    else:
        print("No run_config.json files found; falling back to checkpoint metadata.")
        rows = _collect_checkpoint_rows(root, CHECKPOINT_GLOB, skip_default_best_model=SKIP_DEFAULT_BEST_MODEL)
    if not rows and SKIP_DEFAULT_BEST_MODEL:
        print("No rows after skipping default checkpoints; retrying with best_model.pt included.")
        rows = _collect_checkpoint_rows(root, CHECKPOINT_GLOB, skip_default_best_model=False)
    if not rows:
        raise SystemExit(
            f"No run configs found matching '{RUN_CONFIG_GLOB}' and no checkpoints found matching '{CHECKPOINT_GLOB}'."
        )

    df = pd.DataFrame(rows)
    df = _normalize_epoch_columns(df)
    df = _normalize_metric_columns(df)
    df = _normalize_config_columns(df)
    df = _normalize_identity_columns(df, root=root)
    sort_column = _resolve_sort_column(df)
    if sort_column is None:
        print(f"Sort column '{SORT_BY}' not found. Grouping/ordering will be unsorted.")
    df = _group_dataframe(df, sort_by=sort_column)

    if sort_column is not None:
        df = df.sort_values(sort_column, ascending=SORT_ASCENDING)
    df = _select_output_columns(df)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(df)} rows to: {output_csv_path.resolve()}")


if __name__ == "__main__":
    main()
