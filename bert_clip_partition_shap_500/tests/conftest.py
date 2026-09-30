"""Shared fixtures. Tests read the saved outputs; a few also load the frozen models."""

from __future__ import annotations

import sys
from pathlib import Path

# pandas/pyarrow (via shap) must be imported before CUDA initialises in the rfem env.
import shap  # noqa: F401
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_utils import CHECKPOINT_DIR, METRICS_DIR, VALUES_DIR, load_config, read_csv  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def rows(config):
    from data_validation import load_evaluation_rows

    return load_evaluation_rows(config)


@pytest.fixture(scope="session")
def sentence_results():
    return read_csv(VALUES_DIR / "sentence_results.csv")


@pytest.fixture(scope="session")
def word_values():
    return read_csv(VALUES_DIR / "word_shap_values.csv")


@pytest.fixture(scope="session")
def model_scores():
    return read_csv(VALUES_DIR / "model_scores.csv")


@pytest.fixture(scope="session")
def tokenizers(config):
    from transformers import BertTokenizer, CLIPTokenizerFast

    local = bool(config["local_files_only"])
    try:
        return {
            "bert": BertTokenizer.from_pretrained(config["models"]["bert"]["name_or_path"], local_files_only=local),
            "clip": CLIPTokenizerFast.from_pretrained(config["models"]["clip"]["name_or_path"], local_files_only=local),
        }
    except OSError as error:  # pragma: no cover - only when the local model cache is missing
        pytest.skip(f"local tokenizers unavailable: {error}")


__all__ = ["ROOT", "CHECKPOINT_DIR", "METRICS_DIR", "VALUES_DIR"]
