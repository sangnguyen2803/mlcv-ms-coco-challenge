import gc
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from utils.models_factory import MODEL_SPECS, _configure_pretrained_weights_cache


@dataclass(slots=True)
class FailedDownload:
    model_name: str
    error: str


def _expected_weight_filenames() -> set[str]:
    expected: set[str] = set()
    for spec in MODEL_SPECS.values():
        weights = spec.weights
        if weights is None:
            continue
        url = getattr(weights, "url", "")
        if not url:
            continue
        file_name = url.rsplit("/", 1)[-1].split("?", 1)[0]
        if file_name:
            expected.add(file_name)
    return expected


def _weights_cache_dir() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints"


def _download_all_model_weights() -> tuple[list[str], list[FailedDownload]]:
    total = len(MODEL_SPECS)
    downloaded: list[str] = []
    failed: list[FailedDownload] = []

    print(f"Found {total} models in MODEL_SPECS.")
    for index, (model_name, spec) in enumerate(MODEL_SPECS.items(), start=1):
        weights = spec.weights
        if weights is None:
            print(f"[{index}/{total}] {model_name}: skipped (no pretrained weights configured)")
            continue

        started = time.perf_counter()
        try:
            print(f"[{index}/{total}] {model_name}: downloading/loading weights...")
            model = spec.builder(weights=weights)
            model.eval()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.perf_counter() - started
            print(f"[{index}/{total}] {model_name}: OK ({elapsed:.2f}s)")
            downloaded.append(model_name)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            print(f"[{index}/{total}] {model_name}: FAILED after {elapsed:.2f}s ({exc})")
            failed.append(FailedDownload(model_name=model_name, error=str(exc)))

    return downloaded, failed


def _prune_stale_weight_files(expected_files: set[str]) -> list[str]:
    checkpoints_dir = _weights_cache_dir()
    if not checkpoints_dir.exists():
        return []

    removed: list[str] = []
    for path in checkpoints_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".pth", ".pt"}:
            continue
        if path.name in expected_files:
            continue
        path.unlink()
        removed.append(path.name)
    return removed


def main() -> None:
    _configure_pretrained_weights_cache()
    expected_files = _expected_weight_filenames()
    checkpoints_dir = _weights_cache_dir()

    started = time.perf_counter()
    downloaded, failed = _download_all_model_weights()
    removed = _prune_stale_weight_files(expected_files)
    elapsed = time.perf_counter() - started

    print("\nSync summary")
    print(f"- Cache directory: {checkpoints_dir}")
    print(f"- Expected weight files from MODEL_SPECS: {len(expected_files)}")
    print(f"- Successful downloads/loads: {len(downloaded)}")
    print(f"- Failed downloads/loads: {len(failed)}")
    print(f"- Removed stale local weight files: {len(removed)}")
    print(f"- Total time: {elapsed:.2f}s")

    if removed:
        print("\nRemoved files")
        for name in removed:
            print(f"- {name}")

    if failed:
        print("\nFailures")
        for failure in failed:
            print(f"- {failure.model_name}: {failure.error}")


if __name__ == "__main__":
    main()
