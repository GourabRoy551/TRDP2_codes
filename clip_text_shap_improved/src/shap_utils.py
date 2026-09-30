"""Partition-SHAP extraction, additivity checks, and CLIP BPE aggregation."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np


SPECIAL_TOKENS = {"<|startoftext|>", "<|endoftext|>"}
CONTRACTION = re.compile(r"^['’](?:s|t|re|ve|ll|d|m)$", re.IGNORECASE)
HYPHENS = {"-", "‐", "‑", "–"}


def create_text_masker(tokenizer: Any) -> Any:
    import shap

    return shap.maskers.Text(tokenizer, mask_token="", collapse_mask_token=True)


def model_tokens(tokenizer: Any, text: str, max_length: int) -> list[str]:
    encoded = tokenizer(
        text, add_special_tokens=True, truncation=True, max_length=max_length
    )
    return [str(value) for value in tokenizer.convert_ids_to_tokens(encoded["input_ids"])]


def unpack_explanation(
    explanation: Any, expected_outputs: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != expected_outputs:
        raise ValueError(
            f"Expected (features, {expected_outputs}) SHAP values, got {values.shape}."
        )
    data = explanation.data
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = data[0]
    features = [str(value) for value in np.asarray(data, dtype=object).reshape(-1)]
    bases = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if len(features) != values.shape[0] or bases.size != expected_outputs:
        raise ValueError("SHAP features, values and base values do not align.")
    return features, values, bases


def additivity(
    scores: Sequence[float], bases: Sequence[float], values: np.ndarray
) -> dict[str, np.ndarray]:
    score_array = np.asarray(scores, dtype=float)
    base_array = np.asarray(bases, dtype=float)
    shap_sum = np.asarray(values, dtype=float).sum(axis=0)
    reconstructed = base_array + shap_sum
    return {
        "shap_sum": shap_sum,
        "reconstructed": reconstructed,
        "residual": score_array - reconstructed,
    }


def _word_row(
    indices: list[int], text: str, values: np.ndarray, special: bool
) -> dict[str, Any]:
    clean = text.strip() or text
    return {
        "feature_indices": list(indices),
        "word": clean,
        "values": values[indices].sum(axis=0),
        "is_special": special,
        "is_content": bool(not special and re.search(r"[A-Za-z0-9]", clean)),
    }


def aggregate_bpe(
    feature_texts: Sequence[str],
    tokens: Sequence[str],
    values: np.ndarray,
) -> list[dict[str, Any]]:
    """Sum CLIP BPE pieces into readable words without changing total SHAP mass."""
    if len(feature_texts) != len(tokens) or len(tokens) != len(values):
        raise ValueError("Feature texts, CLIP tokens and SHAP rows must align.")
    groups: list[dict[str, Any]] = []
    pending_indices: list[int] = []
    pending_text = ""

    def flush() -> None:
        nonlocal pending_indices, pending_text
        if pending_indices:
            groups.append(_word_row(pending_indices, pending_text, values, False))
            pending_indices, pending_text = [], ""

    for index, (feature_text, token) in enumerate(
        zip(feature_texts, tokens, strict=True)
    ):
        if token in SPECIAL_TOKENS:
            flush()
            groups.append(_word_row([index], token, values, True))
            continue
        pending_indices.append(index)
        pending_text += feature_text
        if token.endswith("</w>"):
            flush()
    flush()

    contracted: list[dict[str, Any]] = []
    for group in groups:
        if (
            contracted
            and not group["is_special"]
            and CONTRACTION.match(str(group["word"]))
            and not contracted[-1]["is_special"]
        ):
            previous = contracted[-1]
            previous["feature_indices"].extend(group["feature_indices"])
            previous["word"] += group["word"]
            previous["values"] = previous["values"] + group["values"]
            previous["is_content"] = True
        else:
            contracted.append(group)

    merged = list(contracted)
    index = 0
    while index + 2 < len(merged):
        left, middle, right = merged[index : index + 3]
        if (
            middle["word"] in HYPHENS
            and left["is_content"]
            and right["is_content"]
        ):
            merged[index : index + 3] = [
                {
                    "feature_indices": left["feature_indices"]
                    + middle["feature_indices"]
                    + right["feature_indices"],
                    "word": left["word"] + middle["word"] + right["word"],
                    "values": left["values"] + middle["values"] + right["values"],
                    "is_special": False,
                    "is_content": True,
                }
            ]
        else:
            index += 1
    for word_index, group in enumerate(merged):
        group["word_index"] = word_index
    return merged


def merge_word_values(groups: Sequence[dict[str, Any]], output_index: int) -> dict[str, float]:
    """Combine repeated display words for rank-comparison calculations."""
    result: dict[str, float] = {}
    for group in groups:
        if group["is_special"] or not group["is_content"]:
            continue
        word = str(group["word"]).casefold()
        result[word] = result.get(word, 0.0) + float(group["values"][output_index])
    return result
