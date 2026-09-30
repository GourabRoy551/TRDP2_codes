"""Pixel-level SHAP unpacking, 7x7 patch aggregation and additivity checks.

This plays the role that ``token_aggregation.py`` plays for BERT: SHAP returns
one value per pixel and colour channel; we sum the channels into a pixel map
and then sum each 32x32 block into one value per ViT-B/32 patch token. Summing
preserves the total, so the patch values still satisfy SHAP additivity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter


def box_blur(image: np.ndarray, kernel: int) -> np.ndarray:
    """Box blur equivalent to SHAP's ``blur(k,k)`` masker, without OpenCV.

    Uses a mirror border (OpenCV's default BORDER_REFLECT_101).
    """
    if kernel < 1:
        raise ValueError("Blur kernel must be at least 1 pixel.")
    blurred = uniform_filter(
        np.asarray(image, dtype=np.float32), size=(kernel, kernel, 1), mode="mirror"
    )
    return blurred.astype(np.float32)


def unpack_image_explanation(explanation: Any, image_shape: tuple[int, ...]) -> tuple[np.ndarray, float]:
    """Return (H, W, C) SHAP values and the scalar base value for one image."""
    values = np.asarray(explanation.values, dtype=np.float64)
    base = np.asarray(explanation.base_values, dtype=np.float64).reshape(-1)
    if base.size != 1:
        raise ValueError(f"Expected one explained output, got base values {base.shape}.")
    expected = int(np.prod(image_shape))
    if values.size != expected:
        raise ValueError(f"Expected {expected} SHAP values, got shape {values.shape}.")
    return values.reshape(image_shape), float(base[0])


def pixel_map(values_hwc: np.ndarray) -> np.ndarray:
    """Sum colour channels -> one SHAP value per pixel."""
    return np.asarray(values_hwc, dtype=np.float64).sum(axis=-1)


def patch_grid(pixel_values: np.ndarray, patch_size: int) -> np.ndarray:
    """Sum non-overlapping patch_size x patch_size blocks -> (rows, cols) grid."""
    height, width = pixel_values.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image size must be a multiple of the patch size.")
    rows, cols = height // patch_size, width // patch_size
    return pixel_values.reshape(rows, patch_size, cols, patch_size).sum(axis=(1, 3))


def calculate_additivity(score: float, base_value: float, values: np.ndarray) -> dict[str, float]:
    """SHAP efficiency check: score = base + sum(values)."""
    total = float(np.sum(values))
    reconstructed = base_value + total
    return {
        "sum_shap_values": total,
        "reconstructed_score": reconstructed,
        "additivity_residual": float(score - reconstructed),
    }


def patch_rows(image_id: str, grid: np.ndarray, patch_size: int) -> list[dict[str, Any]]:
    """Long-format table: one row per patch, with rank by signed value."""
    rows_count, cols_count = grid.shape
    flat = grid.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, flat.size + 1)
    abs_order = np.argsort(-np.abs(flat), kind="stable")
    abs_rank = np.empty_like(abs_order)
    abs_rank[abs_order] = np.arange(1, flat.size + 1)
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(flat):
        row, col = divmod(index, cols_count)
        rows.append(
            {
                "image_id": image_id,
                "patch_index": index,
                "patch_row": row,
                "patch_col": col,
                "x0": col * patch_size,
                "y0": row * patch_size,
                "x1": (col + 1) * patch_size,
                "y1": (row + 1) * patch_size,
                "shap_value": float(value),
                "abs_shap_value": float(abs(value)),
                "signed_rank": int(rank[index]),
                "abs_rank": int(abs_rank[index]),
            }
        )
    return rows


def patch_label(index: int, cols: int) -> str:
    row, col = divmod(int(index), cols)
    return f"({row},{col})"
