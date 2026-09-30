"""Shared final-evaluation, faithfulness, bootstrap and comparison helpers."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

from classifier_head import classification_metrics
from word_masking import WordUnit, split_word_units


CONTRAST_PATTERN = re.compile(r"\b(?:but|although|however|yet|while|though)\b", re.I)


def score_in_batches(
    scorer: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        batches.append(np.asarray(scorer(texts[start : start + batch_size]), dtype=float))
    return np.concatenate(batches, axis=0)


def labels_from_rows(rows: Sequence[dict[str, str]]) -> np.ndarray:
    return np.asarray([0 if row["gold_label"] == "NEG" else 1 for row in rows], dtype=np.int64)


def length_group(word_count: int, short_max: int, medium_max: int) -> str:
    if word_count <= short_max:
        return "short"
    if word_count <= medium_max:
        return "medium"
    return "long"


def masked_text(masker: Any, text: str, units: Sequence[WordUnit], kept: set[int]) -> str:
    """Render one whole-word coalition while preserving invariant boundaries."""
    mask = np.ones(len(units) + 2, dtype=bool)
    content_indices = {unit.index for unit in units if unit.is_content}
    for index in content_indices - kept:
        mask[index + 1] = False
    return str(np.asarray(masker(mask, text)[0], dtype=object).reshape(-1)[0])


def explanation_concentration(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=float))
    total = float(absolute.sum())
    if total <= 1e-15:
        return {
            "margin_absolute_mass": total,
            "top_1_concentration": 0.0,
            "top_5_concentration": 0.0,
            "normalized_entropy": 0.0,
            "one_minus_normalized_entropy": 1.0,
        }
    probabilities = absolute / total
    ordered = np.sort(probabilities)[::-1]
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    normalized = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    return {
        "margin_absolute_mass": total,
        "top_1_concentration": float(ordered[:1].sum()),
        "top_5_concentration": float(ordered[:5].sum()),
        "normalized_entropy": normalized,
        "one_minus_normalized_entropy": 1.0 - normalized,
    }


def compute_faithfulness(
    scorer: Callable[[Sequence[str]], np.ndarray],
    masker: Any,
    text: str,
    margin_values: np.ndarray,
    original_margin: float,
    fractions: Sequence[float],
    random_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Measure top-ranked deletion/retention against deterministic random baselines."""
    units = split_word_units(text)
    content = [unit.index for unit in units if unit.is_content]
    if not content:
        raise ValueError("Faithfulness requires at least one lexical word.")
    direction = 1.0 if original_margin >= 0 else -1.0
    original_decision_score = direction * float(original_margin)
    signed = direction * np.asarray(margin_values, dtype=float)
    ranked = sorted(content, key=lambda index: (-signed[index], index))
    rng = np.random.default_rng(seed)

    requests: list[tuple[str, float, int, str]] = []
    texts: list[str] = []
    all_content = set(content)
    for fraction in fractions:
        k = max(1, min(len(content), int(math.ceil(float(fraction) * len(content)))))
        top = set(ranked[:k])
        texts.append(masked_text(masker, text, units, all_content - top))
        requests.append(("top_delete", float(fraction), k, ""))
        texts.append(masked_text(masker, text, units, top))
        requests.append(("top_retain", float(fraction), k, ""))
        for repeat in range(random_repeats):
            random_removed = set(rng.choice(content, size=k, replace=False).tolist())
            texts.append(masked_text(masker, text, units, all_content - random_removed))
            requests.append(("random_delete", float(fraction), k, str(repeat)))

    scores = np.asarray(scorer(texts), dtype=float)
    details: dict[float, dict[str, Any]] = defaultdict(
        lambda: {"random_margins": [], "random_flips": []}
    )
    for request, score in zip(requests, scores, strict=True):
        kind, fraction, k, _ = request
        margin = float(score[2])
        prediction = "POS" if margin > 0 else "NEG"
        if kind == "top_delete":
            details[fraction].update(
                {
                    "k": k,
                    "top_delete_margin": margin,
                    "top_delete_prediction_flip": int(np.sign(margin) != np.sign(original_margin)),
                }
            )
        elif kind == "top_retain":
            details[fraction].update(
                {
                    "top_retain_margin": margin,
                    "top_retain_prediction_flip": int(np.sign(margin) != np.sign(original_margin)),
                }
            )
        else:
            details[fraction]["random_margins"].append(margin)
            details[fraction]["random_flips"].append(
                int(np.sign(margin) != np.sign(original_margin))
            )

    rows: list[dict[str, Any]] = []
    epsilon = 1e-6
    for fraction in fractions:
        item = details[float(fraction)]
        top_delete_decision = direction * float(item["top_delete_margin"])
        top_retain_decision = direction * float(item["top_retain_margin"])
        random_decisions = direction * np.asarray(item["random_margins"], dtype=float)
        top_drop = original_decision_score - top_delete_decision
        sufficiency_gap = original_decision_score - top_retain_decision
        random_drops = original_decision_score - random_decisions
        rows.append(
            {
                "fraction": float(fraction),
                "removed_or_retained_words": int(item["k"]),
                "original_margin": float(original_margin),
                "original_decision_score": original_decision_score,
                "top_delete_margin": float(item["top_delete_margin"]),
                "comprehensiveness_drop": float(top_drop),
                "normalized_comprehensiveness_drop": float(
                    top_drop / (abs(original_decision_score) + epsilon)
                ),
                "top_delete_prediction_flip": int(item["top_delete_prediction_flip"]),
                "top_retain_margin": float(item["top_retain_margin"]),
                "sufficiency_gap": float(sufficiency_gap),
                "normalized_sufficiency_gap": float(
                    sufficiency_gap / (abs(original_decision_score) + epsilon)
                ),
                "top_retain_prediction_flip": int(item["top_retain_prediction_flip"]),
                "random_delete_mean_margin": float(np.mean(item["random_margins"])),
                "random_deletion_drop": float(np.mean(random_drops)),
                "normalized_random_deletion_drop": float(
                    np.mean(random_drops) / (abs(original_decision_score) + epsilon)
                ),
                "random_deletion_flip_rate": float(np.mean(item["random_flips"])),
                "random_repeats": random_repeats,
            }
        )
    aggregate = {
        "top_deletion_aopc": float(np.mean([row["comprehensiveness_drop"] for row in rows])),
        "random_deletion_aopc": float(np.mean([row["random_deletion_drop"] for row in rows])),
        "normalized_top_deletion_aopc": float(
            np.mean([row["normalized_comprehensiveness_drop"] for row in rows])
        ),
        "normalized_random_deletion_aopc": float(
            np.mean([row["normalized_random_deletion_drop"] for row in rows])
        ),
        "mean_sufficiency_gap": float(np.mean([row["sufficiency_gap"] for row in rows])),
        "mean_normalized_sufficiency_gap": float(
            np.mean([row["normalized_sufficiency_gap"] for row in rows])
        ),
        "mean_top_deletion_flip_rate": float(
            np.mean([row["top_delete_prediction_flip"] for row in rows])
        ),
        "mean_random_deletion_flip_rate": float(
            np.mean([row["random_deletion_flip_rate"] for row in rows])
        ),
    }
    return rows, aggregate


def select_representative_sentences(
    rows: Sequence[dict[str, str]],
    zero_scores: np.ndarray,
    trained_scores: np.ndarray,
) -> list[dict[str, Any]]:
    """Select five presentation examples without consulting SHAP values."""
    enriched = []
    for index, row in enumerate(rows):
        gold = row["gold_label"]
        zero_prediction = "POS" if zero_scores[index, 2] > 0 else "NEG"
        trained_prediction = "POS" if trained_scores[index, 2] > 0 else "NEG"
        enriched.append(
            {
                **row,
                "index": index,
                "zero_prediction": zero_prediction,
                "trained_prediction": trained_prediction,
                "both_correct": zero_prediction == gold and trained_prediction == gold,
                "word_count": len(split_word_units(row["text"])),
                "trained_abs_margin": abs(float(trained_scores[index, 2])),
            }
        )

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(role: str, candidates: Sequence[dict[str, Any]], reason: str) -> None:
        choice = next((item for item in candidates if item["sentence_id"] not in used), None)
        if choice is None:
            choice = next(item for item in enriched if item["sentence_id"] not in used)
        used.add(choice["sentence_id"])
        selected.append(
            {
                "selection_order": len(selected) + 1,
                "role": role,
                "sentence_id": choice["sentence_id"],
                "text": choice["text"],
                "gold_label": choice["gold_label"],
                "zero_shot_prediction": choice["zero_prediction"],
                "trained_head_prediction": choice["trained_prediction"],
                "selection_reason": reason,
                "selection_uses_shap_values": False,
            }
        )

    positive = sorted(
        [item for item in enriched if item["gold_label"] == "POS" and item["both_correct"]],
        key=lambda item: item["sentence_id"],
    )
    negative = sorted(
        [item for item in enriched if item["gold_label"] == "NEG" and item["both_correct"]],
        key=lambda item: item["sentence_id"],
    )
    contrastive = sorted(
        [item for item in enriched if CONTRAST_PATTERN.search(item["text"]) and item["both_correct"]],
        key=lambda item: (-item["word_count"], item["sentence_id"]),
    )
    longest = sorted(
        [item for item in enriched if item["both_correct"]],
        key=lambda item: (-int(item["clip_bpe_token_count"]), item["sentence_id"]),
    )
    errors = sorted(
        [item for item in enriched if item["trained_prediction"] != item["gold_label"]],
        key=lambda item: (item["trained_abs_margin"], item["sentence_id"]),
    )
    if not errors:
        errors = sorted(enriched, key=lambda item: (item["trained_abs_margin"], item["sentence_id"]))

    take("correct_positive", positive, "First sentence ID correctly classified as POS by both frozen CLIP conditions.")
    take("correct_negative", negative, "First sentence ID correctly classified as NEG by both frozen CLIP conditions.")
    take("contrastive", contrastive, "Longest jointly correct sentence containing a predeclared contrast marker.")
    take("long_multi_token", longest, "Highest CLIP-BPE token count among jointly correct sentences.")
    take("trained_head_error", errors, "Lowest-absolute-margin trained-head error; confidence-based fallback if required.")
    return selected


def bootstrap_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    sentence_rows: Sequence[dict[str, Any]],
    metric_fields: Sequence[str],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Percentile bootstrap intervals for prediction and explanation means."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    class_indices = [np.flatnonzero(labels == class_index) for class_index in (0, 1)]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        # Stratification keeps both sentiment classes represented and preserves
        # the balanced final-set design in every bootstrap replicate.
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices if len(group)]
        )
        rng.shuffle(indices)
        metrics = classification_metrics(labels[indices], logits[indices, :2])
        for name in ("accuracy", "macro_f1", "balanced_accuracy"):
            samples[name].append(float(metrics[name]))
        for field in metric_fields:
            samples[field].append(
                float(np.mean([float(sentence_rows[index][field]) for index in indices]))
            )
    point_metrics = classification_metrics(labels, logits[:, :2])
    result = []
    for name, values in samples.items():
        point = (
            float(point_metrics[name])
            if name in point_metrics
            else float(np.mean([float(row[name]) for row in sentence_rows]))
        )
        result.append(
            {
                "metric": name,
                "point_estimate": point,
                "ci_lower_95": float(np.percentile(values, 2.5)),
                "ci_upper_95": float(np.percentile(values, 97.5)),
                "bootstrap_iterations": iterations,
            }
        )
    return result


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size <= 1 or np.allclose(left, right, atol=1e-15, rtol=0.0):
        return 1.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def top_indices(values: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-np.abs(values), kind="stable")[: min(k, len(values))].tolist())


def compare_word_explanations(
    left: np.ndarray, right: np.ndarray, content_mask: np.ndarray, top_k: int
) -> dict[str, float]:
    left_values = np.asarray(left, dtype=float)[content_mask]
    right_values = np.asarray(right, dtype=float)[content_mask]
    left_top = top_indices(left_values, top_k)
    right_top = top_indices(right_values, top_k)
    union = left_top | right_top
    count = max(1, min(top_k, len(left_values)))
    signs_left = np.sign(np.where(np.abs(left_values) <= 1e-12, 0.0, left_values))
    signs_right = np.sign(np.where(np.abs(right_values) <= 1e-12, 0.0, right_values))
    return {
        "abs_rank_spearman": safe_spearman(np.abs(left_values), np.abs(right_values)),
        "top_k_overlap_rate": len(left_top & right_top) / count,
        "top_k_jaccard": len(left_top & right_top) / len(union) if union else 1.0,
        "sign_agreement": float(np.mean(signs_left == signs_right)) if len(left_values) else 1.0,
    }


def subgroup_rows(
    condition: str, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    definitions = {
        "gold_label": ["NEG", "POS"],
        "correctness": ["correct", "incorrect"],
        "length_group": ["short", "medium", "long"],
    }
    for group_type, groups in definitions.items():
        for group in groups:
            if group_type == "correctness":
                selected = [row for row in rows if ("correct" if int(row["correct"]) else "incorrect") == group]
            else:
                selected = [row for row in rows if str(row[group_type]) == group]
            if not selected:
                continue
            result.append(
                {
                    "condition": condition,
                    "group_type": group_type,
                    "group": group,
                    "sentences": len(selected),
                    "accuracy": float(np.mean([int(row["correct"]) for row in selected])),
                    "mean_top_deletion_aopc": float(np.mean([float(row["top_deletion_aopc"]) for row in selected])),
                    "mean_random_deletion_aopc": float(np.mean([float(row["random_deletion_aopc"]) for row in selected])),
                    "mean_top_5_concentration": float(np.mean([float(row["top_5_concentration"]) for row in selected])),
                    "mean_shap_runtime_seconds": float(np.mean([float(row["shap_runtime_seconds"]) for row in selected])),
                }
            )
    return result
