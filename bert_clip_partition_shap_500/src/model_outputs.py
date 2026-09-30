"""The three-output contract shared by both frozen scorers.

Every scorer returns an ``(n, 3)`` float64 array ordered as

    column 0: score_NEG(x)
    column 1: score_POS(x)
    column 2: margin(x) = score_POS(x) - score_NEG(x)

The class scores are the float32 model outputs (BERT logits or CLIP cosine
similarities) widened to float64; the margin is then formed in float64 so that it
is exactly the difference of the two reported class scores.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


OUTPUT_NAMES = ("NEG", "POS", "POS-NEG")
OUTPUT_KEYS = ("neg", "pos", "margin")
CLASS_NAMES = ("NEG", "POS")
NEG, POS, MARGIN = 0, 1, 2


def as_text_list(texts: Any) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(value) for value in np.asarray(texts, dtype=object).reshape(-1).tolist()]


def assemble_outputs(class_scores: Any) -> np.ndarray:
    """Build ``[NEG, POS, POS-NEG]`` from an ``(n, 2)`` tensor/array already in NEG, POS order."""
    if hasattr(class_scores, "detach"):
        class_scores = class_scores.detach().float().cpu().numpy()
    ordered = np.asarray(class_scores, dtype=np.float64)
    if ordered.ndim != 2 or ordered.shape[1] != 2:
        raise ValueError(f"Expected (n, 2) class scores, got {ordered.shape}.")
    margin = ordered[:, POS] - ordered[:, NEG]
    return np.column_stack([ordered[:, NEG], ordered[:, POS], margin])


def predict_label(margin: float) -> str:
    """POS when the margin is strictly positive; NEG when it is zero or negative."""
    return "POS" if float(margin) > 0.0 else "NEG"


def decision_direction(margin: float) -> float:
    """+1 for a POS prediction, -1 for a NEG prediction (consistent with ``predict_label``)."""
    return 1.0 if float(margin) > 0.0 else -1.0


def score_in_batches(
    scorer: Callable[[Sequence[str]], np.ndarray], texts: Sequence[str], batch_size: int
) -> np.ndarray:
    batches = [
        np.asarray(scorer(list(texts[start : start + batch_size])), dtype=np.float64)
        for start in range(0, len(texts), batch_size)
    ]
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 3))


def parity_check(
    scorer: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    batch_size: int,
    tolerance: float,
) -> dict[str, Any]:
    """Verify shape, column semantics, prediction rule and batched-versus-single parity."""
    batched = score_in_batches(scorer, texts, batch_size)
    single = np.concatenate([np.asarray(scorer([text]), dtype=np.float64) for text in texts])
    margin_error = float(np.max(np.abs(batched[:, MARGIN] - (batched[:, POS] - batched[:, NEG]))))
    predictions = [predict_label(value) for value in batched[:, MARGIN]]
    rule_ok = all(
        (prediction == "POS") == (margin > 0.0)
        for prediction, margin in zip(predictions, batched[:, MARGIN], strict=True)
    )
    batch_difference = float(np.max(np.abs(batched - single)))
    single_predictions = [predict_label(value) for value in single[:, MARGIN]]
    checks = {
        "shape_is_n_by_3": batched.shape == (len(texts), 3),
        "all_finite": bool(np.all(np.isfinite(batched))),
        "column_2_equals_pos_minus_neg": margin_error == 0.0,
        "prediction_rule_margin_gt_0_is_pos": rule_ok,
        "batched_matches_individual": batch_difference <= tolerance,
        "batched_and_individual_predictions_identical": predictions == single_predictions,
    }
    return {
        "sentences_checked": len(texts),
        "output_shape": list(batched.shape),
        "max_margin_identity_error": margin_error,
        "max_batched_vs_individual_abs_difference": batch_difference,
        "batch_parity_tolerance": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_frozen(model: Any) -> dict[str, Any]:
    """Confirm evaluation mode and that no parameter can receive gradients."""
    parameters = list(model.parameters())
    trainable = sum(int(parameter.requires_grad) for parameter in parameters)
    return {
        "eval_mode": not bool(model.training),
        "parameter_tensors": len(parameters),
        "parameters_total": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_tensors": trainable,
        "frozen": (not model.training) and trainable == 0,
    }


def freeze(model: Any) -> Any:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
