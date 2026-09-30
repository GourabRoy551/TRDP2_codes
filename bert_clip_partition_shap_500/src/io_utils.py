"""Paths, configuration, CSV/JSON I/O, hashing and deterministic seeding."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"
OUTPUTS = ROOT / "outputs"
VALUES_DIR = OUTPUTS / "values"
METRICS_DIR = OUTPUTS / "metrics"
PLOTS_DIR = OUTPUTS / "plots"
CASE_STUDY_DIR = PLOTS_DIR / "case_studies"
CHECKPOINT_DIR = OUTPUTS / "checkpoints"
REPORTS_DIR = OUTPUTS / "reports"
MODEL_KEYS = ("bert", "clip")


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to this experiment folder."""
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = load_json(path or CONFIG_PATH)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")
    return config


def config_sha256(path: Path | None = None) -> str:
    return sha256_file(path or CONFIG_PATH)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _cell(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "" if math.isnan(value) else str(value)
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows atomically; floats keep full round-trip precision."""
    materialized = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _cell(value) for key, value in row.items()})
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_determinism(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA is initialised, so this is
    called before any model is created.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Full float32 matmuls: TF32 would add ~1e-3 relative error to every score.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> Any:
    """Resolve ``auto`` to CUDA when available, otherwise fall back to CPU."""
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_offline_environment(local_files_only: bool) -> None:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Prevents Transformers from starting an online safetensors conversion check.
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


def environment_versions() -> dict[str, str]:
    import platform

    import numpy
    import scipy
    import shap
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "shap": shap.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_runtime"] = str(torch.version.cuda)
    return info
