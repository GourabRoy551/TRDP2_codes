"""Choose five case-study sentences from labels, predictions, length and IDs only.

This runs after model scoring and before any SHAP value exists, so the choice
cannot depend on explanations. Every rule is deterministic and ends with the
sentence ID as the final tie-breaker; a sentence is never chosen twice.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from io_utils import utc_now


def _contrast_pattern(markers: Sequence[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(marker) for marker in markers) + r")\b", re.IGNORECASE)


def select_representatives(candidates: Sequence[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """``candidates`` rows need: sentence_id, text, gold_label, bert_prediction,
    clip_prediction, content_words, bert_multi_piece_words, clip_multi_piece_words."""
    low, high = int(settings["readable_min_content_words"]), int(settings["readable_max_content_words"])
    contrast = _contrast_pattern(settings["contrast_markers"])
    both_correct = [
        row for row in candidates
        if row["bert_prediction"] == row["gold_label"] and row["clip_prediction"] == row["gold_label"]
    ]
    readable = lambda row: low <= int(row["content_words"]) <= high  # noqa: E731
    roles: list[tuple[str, str, Callable[[], list[dict[str, Any]]]]] = [
        (
            "correct_positive",
            f"gold POS, predicted POS by both BERT and CLIP, {low}-{high} content words; lowest sentence ID",
            lambda: sorted((r for r in both_correct if r["gold_label"] == "POS" and readable(r)), key=lambda r: r["sentence_id"]),
        ),
        (
            "correct_negative",
            f"gold NEG, predicted NEG by both BERT and CLIP, {low}-{high} content words; lowest sentence ID",
            lambda: sorted((r for r in both_correct if r["gold_label"] == "NEG" and readable(r)), key=lambda r: r["sentence_id"]),
        ),
        (
            "contrastive",
            "contains a contrast marker ("
            + ", ".join(settings["contrast_markers"])
            + f"), correct for both models, <= {settings['contrastive_max_content_words']} content words; "
            "most content words, then lowest sentence ID",
            lambda: sorted(
                (
                    r for r in both_correct
                    if contrast.search(r["text"]) and int(r["content_words"]) <= int(settings["contrastive_max_content_words"])
                ),
                key=lambda r: (-int(r["content_words"]), r["sentence_id"]),
            ),
        ),
        (
            "long_multi_subword",
            "correct for both models; most whole words split into >=2 internal pieces "
            "(BERT WordPiece + CLIP BPE counts), then most content words, then lowest sentence ID",
            lambda: sorted(
                both_correct,
                key=lambda r: (
                    -(int(r["bert_multi_piece_words"]) + int(r["clip_multi_piece_words"])),
                    -int(r["content_words"]),
                    r["sentence_id"],
                ),
            ),
        ),
        (
            "bert_clip_disagreement",
            f"BERT and CLIP predictions disagree, {low}-{high} content words; lowest sentence ID "
            "(fallback: any misclassified sentence by lowest ID)",
            lambda: sorted(
                (r for r in candidates if r["bert_prediction"] != r["clip_prediction"] and readable(r)),
                key=lambda r: r["sentence_id"],
            )
            or sorted(
                (r for r in candidates if r["bert_prediction"] != r["gold_label"] or r["clip_prediction"] != r["gold_label"]),
                key=lambda r: r["sentence_id"],
            ),
        ),
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    timestamp = utc_now()
    for order, (role, rule, pool) in enumerate(roles, start=1):
        choice = next((row for row in pool() if row["sentence_id"] not in used), None)
        if choice is None:
            raise RuntimeError(f"No candidate satisfies the {role} rule.")
        used.add(choice["sentence_id"])
        selected.append(
            {
                "selection_order": order,
                "role": role,
                "selection_rule": rule,
                "sentence_id": choice["sentence_id"],
                "text": choice["text"],
                "gold_label": choice["gold_label"],
                "bert_prediction": choice["bert_prediction"],
                "clip_prediction": choice["clip_prediction"],
                "content_words": choice["content_words"],
                "bert_multi_piece_words": choice["bert_multi_piece_words"],
                "clip_multi_piece_words": choice["clip_multi_piece_words"],
                "selection_uses_shap_values": False,
                "selected_at_utc": timestamp,
            }
        )
    return selected
