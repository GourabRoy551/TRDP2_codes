"""Normalized deletion AOPC with a deterministic random-deletion baseline.

For a sentence with original margin m:
  direction d = +1 if m > 0 (POS prediction) else -1 (NEG prediction)
  decision score s(x) = d * margin(x)
For each fraction f, delete the top-k content words, k = max(1, ceil(f * n_content)):
  drop_f            = s(x) - s(x without top-k)
  normalized_drop_f = drop_f / (|s(x)| + epsilon)
  AOPC              = mean_f normalized_drop_f
Rankings:
  shap_abs     (primary)       content words by |phi_margin| descending, ties by position
  shap_signed  (supplementary) content words by d * phi_margin descending (support for the prediction)
  random       (baseline)      five deterministic random orders per sentence, identical for both models
Deletion uses the same whole-word masker as SHAP, so punctuation is never deleted.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from model_outputs import MARGIN, decision_direction, predict_label
from whole_word_masker import WordUnit, render_coalition


PRIMARY_RANKING = "shap_abs"
RANKINGS = ("shap_abs", "shap_signed", "random")


def deletion_count(fraction: float, content_count: int) -> int:
    # round() guards against float artefacts such as 0.3 * 10 = 3.0000000000000004.
    return min(content_count, max(1, math.ceil(round(float(fraction) * content_count, 9))))


def random_orders(content: Sequence[int], repeats: int, seed: int, row_index: int) -> list[list[int]]:
    """Deterministic per-sentence permutations; independent of the model being evaluated."""
    rng = np.random.default_rng([int(seed), int(row_index)])
    return [[int(value) for value in rng.permutation(np.asarray(content))] for _ in range(repeats)]


def compute_faithfulness(
    scorer: Any,
    masker: Any,
    text: str,
    units: Sequence[WordUnit],
    margin_values: np.ndarray,
    original_margin: float,
    fractions: Sequence[float],
    repeats: int,
    seed: int,
    row_index: int,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content = [unit.index for unit in units if unit.is_content]
    if not content:
        raise ValueError("Deletion faithfulness needs at least one content word.")
    direction = decision_direction(original_margin)
    original_decision = direction * float(original_margin)
    original_prediction = predict_label(original_margin)
    values = np.asarray(margin_values, dtype=np.float64)
    orders: list[tuple[str, int, list[int]]] = [
        ("shap_abs", 0, sorted(content, key=lambda index: (-abs(values[index]), index))),
        ("shap_signed", 0, sorted(content, key=lambda index: (-direction * values[index], index))),
    ]
    orders += [("random", repeat, order) for repeat, order in enumerate(random_orders(content, repeats, seed, row_index))]

    requests, texts = [], []
    for fraction in fractions:
        k = deletion_count(fraction, len(content))
        for ranking, repeat, order in orders:
            deleted = set(order[:k])
            keep = [unit.index not in deleted for unit in units]
            texts.append(render_coalition(masker, text, keep))
            requests.append((float(fraction), k, ranking, repeat, sorted(deleted)))
    before = int(scorer.text_evaluations)
    started = time.perf_counter()
    scores = np.asarray(scorer(texts), dtype=np.float64)
    runtime = time.perf_counter() - started
    evaluations = int(scorer.text_evaluations) - before

    rows = []
    for (fraction, k, ranking, repeat, deleted), perturbed, score in zip(requests, texts, scores, strict=True):
        margin_after = float(score[MARGIN])
        decision_after = direction * margin_after
        drop = original_decision - decision_after
        prediction_after = predict_label(margin_after)
        rows.append(
            {
                "fraction": fraction,
                "ranking": ranking,
                "random_repeat": repeat if ranking == "random" else "",
                "content_words": len(content),
                "deleted_count": k,
                "deleted_word_indices": " ".join(str(index) for index in deleted),
                "deleted_words": " | ".join(units[index].text for index in deleted),
                "perturbed_text": perturbed,
                "original_margin": float(original_margin),
                "direction": direction,
                "original_decision_score": original_decision,
                "margin_after": margin_after,
                "decision_score_after": decision_after,
                "score_drop": drop,
                "normalized_drop": drop / (abs(original_decision) + epsilon),
                "prediction_after": prediction_after,
                "prediction_flip": int(prediction_after != original_prediction),
            }
        )
    return rows, summarize_faithfulness(rows, runtime, evaluations)


def summarize_faithfulness(rows: Sequence[dict[str, Any]], runtime: float = 0.0, evaluations: int = 0) -> dict[str, Any]:
    def mean_of(ranking: str, field: str) -> float:
        # Random repeats are averaged within each fraction first, then across fractions.
        fractions = sorted({row["fraction"] for row in rows})
        per_fraction = [
            float(np.mean([row[field] for row in rows if row["ranking"] == ranking and row["fraction"] == fraction]))
            for fraction in fractions
        ]
        return float(np.mean(per_fraction))

    summary: dict[str, Any] = {}
    for ranking in RANKINGS:
        summary[f"aopc_{ranking}"] = mean_of(ranking, "normalized_drop")
        summary[f"raw_aopc_{ranking}"] = mean_of(ranking, "score_drop")
        summary[f"flip_rate_{ranking}"] = mean_of(ranking, "prediction_flip")
    summary["aopc_shap_minus_random"] = summary["aopc_shap_abs"] - summary["aopc_random"]
    summary["faithfulness_runtime_seconds"] = runtime
    summary["faithfulness_model_evaluations"] = evaluations
    return summary
