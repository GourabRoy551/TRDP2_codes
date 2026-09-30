"""Shared fixtures. Tests read the saved outputs; one test also loads the frozen models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

# pandas/pyarrow (via shap) must be imported before CUDA initialises in the rfem env.
import shap  # noqa: F401,E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from source_link import OUTPUT_DIR, SOURCE_ROOT, load_own_config  # noqa: E402
from io_utils import load_config, read_csv  # noqa: E402


@pytest.fixture(scope="session")
def own_config():
    return load_own_config()


@pytest.fixture(scope="session")
def source_config():
    return load_config()


@pytest.fixture(scope="session")
def checks():
    from io_utils import load_json

    return load_json(OUTPUT_DIR / "checks.json")


@pytest.fixture(scope="session")
def per_sentence():
    return read_csv(OUTPUT_DIR / "per_sentence_metrics.csv")


@pytest.fixture(scope="session")
def audit():
    return read_csv(OUTPUT_DIR / "perturbation_audit.csv")


@pytest.fixture(scope="session")
def dataset_rows():
    return read_csv(SOURCE_ROOT / "data" / "evaluation_500.csv")


__all__ = ["OUTPUT_DIR", "SOURCE_ROOT"]
