"""Comprehensiveness, sufficiency and deletion AOPC for whole-word POS-NEG margin SHAP.

For one sentence x and one model:
  m(x)   POS-NEG margin; d = +1 if m(x) > 0 (POS prediction) else -1
  s(x)   = d * m(x), the decision score of the predicted class
  order  content words by |phi_margin| descending, ties by word position
         (the source experiment's primary ``shap_abs`` deletion ranking)
  k_f    = max(1, ceil(f * n_content))           (source ``deletion_count``)
  x\\r_f  x with its top-k_f content words deleted  (punctuation never deleted)
  r_f    x keeping only its top-k_f content words and all punctuation

  comp_f = s(x) - s(x\\r_f)                          higher = better
  suff_f = s(x) - s(r_f)                            lower  = better
  raw comprehensiveness, raw sufficiency = mean over the fractions f
  deletion AOPC = mean_f comp_f / (|s(x)| + eps)    (source definition, unchanged)

Comprehensiveness and sufficiency are reported after division by one per-model
constant S = mean |s(x)| over the evaluation set, so BERT logits and CLIP cosine
margins share a unitless scale without dividing by near-zero single-sentence margins.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import source_link  # noqa: F401  (puts the source src/ on sys.path)
from faithfulness import deletion_count
from model_outputs import decision_direction


def fraction_label(fraction: float) -> str:
    return f"f{int(round(float(fraction) * 100)):02d}"


def rank_content_words(is_content: Sequence[int], margin_values: Sequence[float]) -> list[int]:
    """Content-word indices by |phi_margin| descending, ties by position (source ``shap_abs``)."""
    values = np.asarray(margin_values, dtype=np.float64)
    content = [index for index, flag in enumerate(is_content) if flag]
    return sorted(content, key=lambda index: (-abs(values[index]), index))


def top_k_by_fraction(order: Sequence[int], fractions: Sequence[float]) -> list[tuple[float, int, list[int]]]:
    """(fraction, k, sorted top-k word indices) for every deletion fraction."""
    cuts = []
    for fraction in fractions:
        k = deletion_count(fraction, len(order))
        cuts.append((float(fraction), k, sorted(order[:k])))
    return cuts


def comprehensiveness_keep(is_content: Sequence[int], top: Sequence[int]) -> list[bool]:
    """Keep every unit except the top-k content words."""
    removed = set(top)
    return [index not in removed for index in range(len(is_content))]


def sufficiency_keep(is_content: Sequence[int], top: Sequence[int]) -> list[bool]:
    """Keep the top-k content words and every non-content (punctuation) unit."""
    kept = set(top)
    return [(not flag) or index in kept for index, flag in enumerate(is_content)]


def sentence_metrics(
    original_margin: float, comp_margins: Sequence[float], suff_margins: Sequence[float], epsilon: float
) -> dict[str, Any]:
    """Raw per-fraction and fraction-averaged metrics for one sentence (native score units)."""
    direction = decision_direction(original_margin)
    decision = direction * float(original_margin)
    comp = decision - direction * np.asarray(comp_margins, dtype=np.float64)
    suff = decision - direction * np.asarray(suff_margins, dtype=np.float64)
    normalized = comp / (abs(decision) + epsilon)
    return {
        "direction": direction,
        "decision_score": decision,
        "comp_by_fraction": comp,
        "suff_by_fraction": suff,
        "normalized_drop_by_fraction": normalized,
        "raw_comprehensiveness": float(np.mean(comp)),
        "raw_sufficiency": float(np.mean(suff)),
        "deletion_aopc": float(np.mean(normalized)),
    }


def rationale_columns(columns: Sequence[str], keywords: Sequence[str]) -> list[str]:
    """Dataset columns whose name suggests a human rationale / evidence annotation."""
    return [column for column in columns if any(keyword in column.lower() for keyword in keywords)]
