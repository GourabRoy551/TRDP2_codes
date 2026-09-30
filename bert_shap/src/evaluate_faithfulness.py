"""Masking-based faithfulness evaluation for word-level SHAP rankings."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from model_wrapper import BertSentimentMargin
from token_aggregation import wordpiece_groups


def masked_text_from_feature_mask(masker: Any, mask: np.ndarray, text: str) -> str:
    """Apply SHAP's own masker so evaluation uses the same missing-feature rule."""
    masked_arguments = masker(np.asarray(mask, dtype=bool), text)
    if not isinstance(masked_arguments, tuple):
        masked_arguments = (masked_arguments,)
    text_argument = np.asarray(masked_arguments[0], dtype=object).reshape(-1)
    if text_argument.size < 1:
        raise RuntimeError("The SHAP text masker returned no masked text.")
    return str(text_argument[0])


def compute_faithfulness(
    sentence_id: str,
    gold_label: str,
    prediction: str,
    text: str,
    original_margin: float,
    tokens: Sequence[str],
    values: np.ndarray,
    fractions: Sequence[float],
    masker: Any,
    model_function: BertSentimentMargin,
) -> list[dict[str, Any]]:
    """Compute comprehensiveness and sufficiency for ranked word groups.

    Ranking is prediction-direction aware.  For a NEG prediction, strongly
    negative values are supporting features; for POS, strongly positive values
    are supporting features.  Every WordPiece of a selected word is masked or
    retained together.
    """
    groups = wordpiece_groups(tokens)
    if not groups:
        return []

    direction = 1.0 if prediction == "POS" else -1.0
    ranked_groups = sorted(
        groups,
        key=lambda group: direction
        * float(sum(float(values[index]) for index in group[1])),
        reverse=True,
    )
    content_indices = [index for _, indices in groups for index in indices]
    original_decision_score = direction * original_margin
    rows: list[dict[str, Any]] = []

    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Faithfulness fractions must be in the interval (0, 1].")

        selected_count = min(
            len(ranked_groups),
            max(1, int(math.ceil(fraction * len(ranked_groups)))),
        )
        selected_groups = ranked_groups[:selected_count]
        selected_indices = [
            index for _, indices in selected_groups for index in indices
        ]

        # Comprehensiveness: remove the strongest supporting words.
        remove_mask = np.ones(len(tokens), dtype=bool)
        remove_mask[selected_indices] = False
        comprehensiveness_text = masked_text_from_feature_mask(
            masker, remove_mask, text
        )
        comprehensiveness_margin = float(model_function([comprehensiveness_text])[0])
        comprehensiveness_score = direction * comprehensiveness_margin

        # Sufficiency: retain only the strongest supporting content words.
        keep_mask = np.ones(len(tokens), dtype=bool)
        keep_mask[content_indices] = False
        keep_mask[selected_indices] = True
        sufficiency_text = masked_text_from_feature_mask(masker, keep_mask, text)
        sufficiency_margin = float(model_function([sufficiency_text])[0])
        sufficiency_score = direction * sufficiency_margin

        rows.append(
            {
                "sentence_id": sentence_id,
                "gold_label": gold_label,
                "prediction": prediction,
                "fraction": float(fraction),
                "selected_count": selected_count,
                "content_word_count": len(groups),
                "selected_token_count": len(selected_indices),
                "selected_feature_indices": " ".join(
                    str(index) for index in selected_indices
                ),
                "selected_features": " | ".join(
                    word for word, _ in selected_groups
                ),
                "original_margin": original_margin,
                "original_decision_score": original_decision_score,
                "comprehensiveness_margin": comprehensiveness_margin,
                "comprehensiveness_drop": (
                    original_decision_score - comprehensiveness_score
                ),
                "sufficiency_margin": sufficiency_margin,
                "sufficiency_decision_score": sufficiency_score,
                "sufficiency_gap": original_decision_score - sufficiency_score,
                "comprehensiveness_text": comprehensiveness_text,
                "sufficiency_text": sufficiency_text,
            }
        )
    return rows
