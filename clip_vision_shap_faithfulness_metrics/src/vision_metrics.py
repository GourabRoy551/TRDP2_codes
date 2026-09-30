"""Deletion / insertion curves, their AUCs and AOPC for 7x7 patch-level SHAP.

For one image x (the 224x224 input CLIP sees) with blurred copy b and score
f(.) = cosine(CLIP embedding of the input, CLIP embedding of x), so f(x) = 1:

  order      the 49 patches by |SHAP| descending, ties by patch index
  D_k        x with the top-k patches replaced by b          (k = 0..49)
  I_k        b with the top-k patches restored from x        (k = 0..49)
  deletion   f(D_0), ..., f(D_49): starts at f(x), ends at f(b)
  insertion  f(I_0), ..., f(I_49): starts at f(b), ends at f(x)
  DAUC, IAUC trapezoidal area under each curve over x = k / 49 in [0, 1]
  AOPC       mean over fractions p of [f(x) - f(D_k(p))], k(p) = max(1, ceil(p * 49))

Removal and restoration use the source experiment's ``patch_mask`` and ``compose`` with
the blurred image SHAP used as its masker background, so the metrics perturb exactly
the "missing" value the explanation was computed against.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

import source_link  # noqa: F401  (puts the source src/ on sys.path)
from evaluate_faithfulness import compose, patch_mask


def rank_patches(grid: np.ndarray) -> list[int]:
    """Patch indices by |SHAP| descending, ties broken by patch index."""
    flat = np.abs(np.asarray(grid, dtype=np.float64).reshape(-1))
    return [int(index) for index in np.argsort(-flat, kind="stable")]


def fraction_count(fraction: float, n_patches: int) -> int:
    """Same rule as the source faithfulness evaluation: max(1, ceil(fraction * n))."""
    return min(n_patches, max(1, int(math.ceil(fraction * n_patches))))


def deletion_images(image: np.ndarray, blurred: np.ndarray, order: Sequence[int], grid_shape: tuple[int, int], patch_size: int) -> np.ndarray:
    """D_0..D_n: the top-k patches replaced by the blurred image."""
    return np.stack([compose(image, blurred, patch_mask(order[:k], grid_shape, patch_size)) for k in range(len(order) + 1)])


def insertion_images(image: np.ndarray, blurred: np.ndarray, order: Sequence[int], grid_shape: tuple[int, int], patch_size: int) -> np.ndarray:
    """I_0..I_n: the blurred image with the top-k patches restored."""
    return np.stack([compose(image, blurred, ~patch_mask(order[:k], grid_shape, patch_size)) for k in range(len(order) + 1)])


def curve_auc(scores: Sequence[float]) -> float:
    """Trapezoidal area under a curve sampled at x = 0, 1/n, ..., 1."""
    values = np.asarray(scores, dtype=np.float64)
    step = 1.0 / (values.size - 1)
    return float(step * (values.sum() - 0.5 * (values[0] + values[-1])))


def clip_box(box: Sequence[float], crop: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Intersection of a box with the crop CLIP sees; None when they do not overlap."""
    x0, y0 = max(box[0], crop[0]), max(box[1], crop[1])
    x1, y1 = min(box[2], crop[2]), min(box[3], crop[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def pointing_hit(point: tuple[float, float], boxes: Sequence[Sequence[float]], tolerance: float) -> bool:
    """True when the point lies inside any box enlarged by ``tolerance`` pixels on every side."""
    x, y = point
    return any(x0 - tolerance <= x <= x1 + tolerance and y0 - tolerance <= y <= y1 + tolerance for x0, y0, x1, y1 in boxes)


def aopc(original: float, deletion_scores: Sequence[float], fractions: Sequence[float]) -> tuple[float, list[float]]:
    """Mean drop f(x) - f(D_k) over the fraction cut-offs; also returns each drop."""
    values = np.asarray(deletion_scores, dtype=np.float64)
    n_patches = values.size - 1
    drops = [float(original - values[fraction_count(fraction, n_patches)]) for fraction in fractions]
    return float(np.mean(drops)), drops
