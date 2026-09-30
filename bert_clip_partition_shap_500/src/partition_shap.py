"""Whole-word Partition SHAP for the three outputs NEG, POS and POS-NEG.

Pipeline per sentence:
  sentence -> whole-word units -> shap.maskers.Text (hierarchical partition tree over
  units) -> PartitionExplainer coalitions -> perturbed sentence (deleted units) ->
  model tokenizer (WordPiece / BPE) -> frozen model -> [NEG, POS, POS-NEG] ->
  one SHAP value per unit and output.

Local additivity (per output c):
  reconstructed_c = base_c + sum_i phi_c(i)          (word units only)
  residual_c      = |score_c(x) - reconstructed_c|
Margin linearity:
  phi_margin(i) ~= phi_POS(i) - phi_NEG(i);  base_margin ~= base_POS - base_NEG
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from model_outputs import MARGIN, NEG, OUTPUT_NAMES, POS
from whole_word_masker import WordUnit


def create_explainer(scorer: Callable[[Sequence[str]], np.ndarray], masker: Any) -> Any:
    import shap

    return shap.Explainer(scorer, masker, algorithm="partition", output_names=list(OUTPUT_NAMES))


def unpack_explanation(explanation: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values, dtype=np.float64)
    if values.ndim == 3:
        values = values[0]
    bases = np.asarray(explanation.base_values, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(OUTPUT_NAMES) or bases.size != len(OUTPUT_NAMES):
        raise ValueError(f"Unexpected SHAP shapes: values {values.shape}, bases {bases.shape}.")
    return values, bases


def additivity_report(scores: np.ndarray, bases: np.ndarray, word_values: np.ndarray) -> dict[str, Any]:
    reconstructed = bases + word_values.sum(axis=0)
    residual = np.abs(np.asarray(scores, dtype=np.float64) - reconstructed)
    return {
        "reconstructed": reconstructed,
        "residual": residual,
        "max_residual": float(residual.max()),
    }


def margin_linearity(word_values: np.ndarray, bases: np.ndarray) -> dict[str, float]:
    word_error = np.abs(word_values[:, MARGIN] - (word_values[:, POS] - word_values[:, NEG]))
    return {
        "max_word_margin_linearity_error": float(word_error.max()) if word_error.size else 0.0,
        "base_margin_linearity_error": float(abs(bases[MARGIN] - (bases[POS] - bases[NEG]))),
    }


def explain_sentence(
    explainer: Any,
    scorer: Any,
    text: str,
    units: Sequence[WordUnit],
    scores: np.ndarray,
    max_evals: int,
    batch_size: int,
) -> dict[str, Any]:
    """Explain one sentence and return values, bases and all numerical checks."""
    before = int(scorer.text_evaluations)
    started = time.perf_counter()
    explanation = explainer([text], max_evals=int(max_evals), batch_size=int(batch_size), silent=True)
    runtime = time.perf_counter() - started
    evaluations = int(scorer.text_evaluations) - before
    word_values, bases = unpack_explanation(explanation)
    if word_values.shape[0] != len(units):
        raise ValueError(f"SHAP returned {word_values.shape[0]} features for {len(units)} whole-word units.")
    feature_names = [str(value).strip() for value in np.asarray(explanation.data, dtype=object).reshape(-1)]
    if feature_names != [unit.text for unit in units]:
        raise ValueError("SHAP feature names do not match the whole-word units.")
    additivity = additivity_report(scores, bases, word_values)
    return {
        "word_values": word_values,
        "base_values": bases,
        "reconstructed": additivity["reconstructed"],
        "additivity_residual": additivity["residual"],
        "max_additivity_residual": additivity["max_residual"],
        **margin_linearity(word_values, bases),
        "shap_runtime_seconds": runtime,
        "shap_model_evaluations": evaluations,
    }
