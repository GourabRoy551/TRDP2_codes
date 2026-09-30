"""Masking-based faithfulness evaluation for patch-level SHAP rankings.

Mirrors ``bert_shap/src/evaluate_faithfulness.py``. Patches are ranked by their
signed SHAP value (largest first = regions that most preserve CLIP's image
representation). For each fraction of the 49 patches:

* comprehensiveness: blur the top patches, keep the rest;
  drop = original score - masked score (higher is better);
* sufficiency: keep only the top patches, blur the rest;
  gap = original score - masked score (lower is better);
* random baseline: blur the same number of randomly chosen patches.

Blurring uses exactly the same blurred image that SHAP uses as its
"missing feature" value, so evaluation and explanation share one rule.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


def patch_mask(selected: Sequence[int], grid_shape: tuple[int, int], patch_size: int) -> np.ndarray:
    """Boolean (H, W) mask that is True inside the selected patches."""
    rows, cols = grid_shape
    mask = np.zeros((rows * patch_size, cols * patch_size), dtype=bool)
    for index in selected:
        row, col = divmod(int(index), cols)
        mask[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = True
    return mask


def compose(image: np.ndarray, blurred: np.ndarray, blur_mask: np.ndarray) -> np.ndarray:
    """Return image with the True region of blur_mask replaced by blurred pixels."""
    out = image.copy()
    out[blur_mask] = blurred[blur_mask]
    return out


def compute_faithfulness(
    image_id: str,
    image: np.ndarray,
    blurred: np.ndarray,
    grid: np.ndarray,
    original_score: float,
    fractions: Sequence[float],
    patch_size: int,
    model_function: Callable[[np.ndarray], np.ndarray],
    random_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    flat = grid.reshape(-1)
    n_patches = flat.size
    ranked = [int(i) for i in np.argsort(-flat, kind="stable")]
    rng = np.random.default_rng(seed)

    batch: list[np.ndarray] = []
    plan: list[tuple[float, int, list[int], list[list[int]]]] = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Faithfulness fractions must be in the interval (0, 1].")
        k = min(n_patches, max(1, int(math.ceil(fraction * n_patches))))
        top = ranked[:k]
        top_mask = patch_mask(top, grid.shape, patch_size)
        batch.append(compose(image, blurred, top_mask))       # comprehensiveness
        batch.append(compose(image, blurred, ~top_mask))      # sufficiency
        random_sets = []
        for _ in range(random_repeats):
            chosen = sorted(int(i) for i in rng.choice(n_patches, size=k, replace=False))
            random_sets.append(chosen)
            batch.append(compose(image, blurred, patch_mask(chosen, grid.shape, patch_size)))
        plan.append((float(fraction), k, top, random_sets))

    scores = np.asarray(model_function(np.stack(batch)), dtype=np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    cursor = 0
    for fraction, k, top, random_sets in plan:
        comp = float(scores[cursor])
        suff = float(scores[cursor + 1])
        random_scores = scores[cursor + 2: cursor + 2 + len(random_sets)]
        cursor += 2 + len(random_sets)
        random_drop = float(np.mean(original_score - random_scores)) if len(random_sets) else float("nan")
        comp_drop = original_score - comp
        rows.append(
            {
                "image_id": image_id,
                "fraction": fraction,
                "selected_patch_count": k,
                "total_patch_count": n_patches,
                "selected_patches": " ".join(str(i) for i in top),
                "original_score": original_score,
                "comprehensiveness_score": comp,
                "comprehensiveness_drop": comp_drop,
                "sufficiency_score": suff,
                "sufficiency_gap": original_score - suff,
                "random_deletion_mean_score": float(np.mean(random_scores)) if len(random_sets) else float("nan"),
                "random_deletion_drop": random_drop,
                "shap_minus_random_drop": comp_drop - random_drop,
                "random_repeats": len(random_sets),
            }
        )
    return rows


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties (no SciPy needed)."""
    def ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="stable")
        result = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            end = start
            while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
                end += 1
            result[order[start:end + 1]] = (start + end) / 2.0
            start = end + 1
        return result

    a, b = ranks(first), ranks(second)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stability_row(image_id: str, reference: np.ndarray, rerun: np.ndarray, budgets: tuple[int, int], top_k: int = 5) -> dict[str, Any]:
    """Compare the 49 patch values from two SHAP budgets."""
    ref, new = reference.reshape(-1), rerun.reshape(-1)
    top_ref = set(np.argsort(-ref, kind="stable")[:top_k].tolist())
    top_new = set(np.argsort(-new, kind="stable")[:top_k].tolist())
    signs = np.sign(ref) == np.sign(new)
    return {
        "image_id": image_id,
        "reference_max_evals": budgets[0],
        "rerun_max_evals": budgets[1],
        "spearman": spearman(ref, new),
        f"top_{top_k}_overlap": len(top_ref & top_new) / top_k,
        "sign_agreement": float(np.mean(signs)),
        "max_abs_patch_difference": float(np.max(np.abs(ref - new))),
        "same_top_patch": int(int(np.argmax(ref)) == int(np.argmax(new))),
    }
