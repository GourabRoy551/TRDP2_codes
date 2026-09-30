"""Utilities for aligning SHAP features with BERT WordPieces.

SHAP's text masker explains tokenizer-level features.  BERT can split one word
into several WordPieces (for example, ``res``, ``##ili``, ``##ence``).  The
helpers here keep the exact token positions for masking, while also producing
readable word-level values for tables and plots.
"""

from __future__ import annotations

import string
from typing import Any, Sequence

import numpy as np
from transformers import BertTokenizer


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}


def unpack_explanation(explanation: Any) -> tuple[list[str], np.ndarray, float]:
    """Extract one sentence's feature text, SHAP values, and expected value."""
    data = np.asarray(explanation.data, dtype=object)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)

    if data.ndim > 1:
        data = data[0]
    tokens = [str(value) for value in data.reshape(-1).tolist()]

    if values.ndim > 1:
        values = values[0]
    values = np.asarray(values, dtype=float).reshape(len(tokens), -1)[:, 0]
    base_value = float(base_values.reshape(-1)[0])

    if len(tokens) != len(values):
        raise RuntimeError(
            f"SHAP returned {len(tokens)} features but {len(values)} values."
        )
    return tokens, values, base_value


def bert_feature_tokens(
    tokenizer: BertTokenizer,
    text: str,
    max_length: int,
    expected_count: int,
) -> list[str]:
    """Return BERT tokens aligned one-to-one with SHAP's leaf features."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    if len(tokens) != expected_count:
        raise RuntimeError(
            "BERT token/SHAP feature alignment failed: "
            f"{len(tokens)} model tokens versus {expected_count} SHAP features."
        )
    return tokens


def normalized_token(token: str) -> str:
    """Remove tokenizer spacing markers without removing ``##`` boundaries."""
    return token.strip().replace("\u0120", "").replace("\u2581", "")


def is_punctuation_token(token: str) -> bool:
    """Return true when every character in a non-empty token is punctuation."""
    value = normalized_token(token)
    return bool(value) and all(character in string.punctuation for character in value)


def content_feature_indices(tokens: Sequence[str]) -> list[int]:
    """Return positions that represent content rather than special/punctuation tokens."""
    indices: list[int] = []
    for index, raw_token in enumerate(tokens):
        token = normalized_token(str(raw_token))
        if not token or token.upper() in SPECIAL_TOKENS:
            continue
        if is_punctuation_token(token):
            continue
        indices.append(index)
    return indices


def wordpiece_groups(tokens: Sequence[str]) -> list[tuple[str, list[int]]]:
    """Group readable words with all BERT feature positions that form each word."""
    groups: list[tuple[str, list[int]]] = []
    for index, raw_token in enumerate(tokens):
        token = normalized_token(str(raw_token))
        if not token or token.upper() in SPECIAL_TOKENS or is_punctuation_token(token):
            continue

        if token.startswith("##") and groups:
            previous_word, previous_indices = groups[-1]
            groups[-1] = (previous_word + token[2:], [*previous_indices, index])
        else:
            groups.append((token, [index]))
    return groups


def aggregate_wordpieces(
    tokens: Sequence[str], values: Sequence[float]
) -> tuple[list[str], list[float]]:
    """Sum signed SHAP values across the WordPieces belonging to each word."""
    if len(tokens) != len(values):
        raise ValueError("tokens and values must have the same length.")

    groups = wordpiece_groups(tokens)
    words = [word for word, _ in groups]
    word_values = [
        float(sum(float(values[index]) for index in indices))
        for _, indices in groups
    ]
    return words, word_values


def calculate_additivity(
    model_margin: float, base_value: float, shap_values: Sequence[float]
) -> tuple[float, float]:
    """Reconstruct the margin and return ``model margin - reconstruction``."""
    reconstructed_margin = float(base_value + np.asarray(shap_values).sum())
    residual = float(model_margin - reconstructed_margin)
    return reconstructed_margin, residual
