"""Extract, validate, and aggregate two-output SHAP explanations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


BERT_SHAP_DIR = Path(__file__).resolve().parents[2]
EXISTING_SRC = BERT_SHAP_DIR / "src"
if str(EXISTING_SRC) not in sys.path:
    sys.path.insert(0, str(EXISTING_SRC))

from token_aggregation import (  # noqa: E402
    bert_feature_tokens,
    content_feature_indices,
    normalized_token,
    wordpiece_groups,
)


def unpack_dual_explanation(
    explanation: Any, output_count: int = 2
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return tokens, a feature-by-output SHAP matrix, and output base values."""
    data = np.asarray(explanation.data, dtype=object)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)

    if data.ndim > 1:
        data = data[0]
    tokens = [str(value) for value in data.reshape(-1).tolist()]

    if values.ndim == 3:
        if values.shape[0] != 1:
            raise RuntimeError("Expected exactly one sentence explanation.")
        values = values[0]
    elif values.ndim == 1 and output_count == 1:
        values = values[:, None]
    values = np.asarray(values, dtype=float)
    if values.shape != (len(tokens), output_count):
        raise RuntimeError(
            "Unexpected SHAP value shape: "
            f"{values.shape}; expected {(len(tokens), output_count)}."
        )

    bases = base_values.reshape(-1)
    if bases.size != output_count:
        raise RuntimeError(
            f"Expected {output_count} base values, received {bases.size}."
        )
    return tokens, values, bases.astype(float)


def aggregate_wordpieces_dual(
    tokens: Sequence[str], values: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Sum each output's SHAP values over WordPieces belonging to one word."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(tokens):
        raise ValueError("values must have shape (number of tokens, outputs).")
    groups = wordpiece_groups(tokens)
    words = [word for word, _ in groups]
    aggregated = np.asarray(
        [matrix[indices].sum(axis=0) for _, indices in groups], dtype=float
    )
    if not groups:
        aggregated = np.empty((0, matrix.shape[1]), dtype=float)
    return words, aggregated


def calculate_additivity_dual(
    model_outputs: Sequence[float],
    base_values: Sequence[float],
    shap_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct each class logit and return output-minus-reconstruction."""
    outputs = np.asarray(model_outputs, dtype=float).reshape(-1)
    bases = np.asarray(base_values, dtype=float).reshape(-1)
    matrix = np.asarray(shap_values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != outputs.size:
        raise ValueError("SHAP matrix output dimension does not match model outputs.")
    if bases.shape != outputs.shape:
        raise ValueError("Base values and model outputs must have the same shape.")
    reconstructed = bases + matrix.sum(axis=0)
    residuals = outputs - reconstructed
    return reconstructed, residuals


def margin_components(
    shap_values: np.ndarray, base_values: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Return POS-minus-NEG SHAP values and the corresponding base value."""
    matrix = np.asarray(shap_values, dtype=float)
    bases = np.asarray(base_values, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or bases.size != 2:
        raise ValueError("Margin conversion requires exactly NEG and POS outputs.")
    return matrix[:, 1] - matrix[:, 0], float(bases[1] - bases[0])


__all__ = [
    "aggregate_wordpieces_dual",
    "bert_feature_tokens",
    "calculate_additivity_dual",
    "content_feature_indices",
    "margin_components",
    "normalized_token",
    "unpack_dual_explanation",
    "wordpiece_groups",
]
