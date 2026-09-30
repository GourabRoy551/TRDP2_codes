"""SHAP reconstruction, patch aggregation and masking tests (no CLIP needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_faithfulness import compose, compute_faithfulness, patch_mask, spearman, stability_row  # noqa: E402
from patch_aggregation import (  # noqa: E402
    box_blur,
    calculate_additivity,
    patch_grid,
    patch_rows,
    pixel_map,
    unpack_image_explanation,
)


class PatchAggregationTest(unittest.TestCase):
    def test_patch_grid_conserves_total(self) -> None:
        rng = np.random.default_rng(0)
        pixels = rng.normal(size=(224, 224))
        grid = patch_grid(pixels, 32)
        self.assertEqual(grid.shape, (7, 7))
        self.assertAlmostEqual(grid.sum(), pixels.sum(), places=9)
        self.assertAlmostEqual(grid[2, 5], pixels[64:96, 160:192].sum(), places=9)

    def test_pixel_map_sums_channels(self) -> None:
        values = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
        np.testing.assert_allclose(pixel_map(values), values.sum(axis=-1))

    def test_additivity_residual_sign_and_size(self) -> None:
        result = calculate_additivity(1.0, 0.6, np.array([0.1, 0.25]))
        self.assertAlmostEqual(result["reconstructed_score"], 0.95)
        self.assertAlmostEqual(result["additivity_residual"], 0.05)

    def test_patch_rows_ranks(self) -> None:
        grid = np.array([[0.5, -0.2], [0.1, 0.0]])
        rows = patch_rows("X", grid, 32)
        by_index = {r["patch_index"]: r for r in rows}
        self.assertEqual(by_index[0]["signed_rank"], 1)
        self.assertEqual(by_index[1]["signed_rank"], 4)
        self.assertEqual(by_index[1]["abs_rank"], 2)
        self.assertEqual((by_index[3]["x0"], by_index[3]["y0"]), (32, 32))


class MaskingTest(unittest.TestCase):
    def test_box_blur_keeps_constant_image(self) -> None:
        image = np.full((64, 64, 3), 0.4, dtype=np.float32)
        np.testing.assert_allclose(box_blur(image, 16), image, atol=1e-6)

    def test_box_blur_smooths_and_keeps_shape(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.random((64, 64, 3)).astype(np.float32)
        blurred = box_blur(image, 32)
        self.assertEqual(blurred.shape, image.shape)
        self.assertLess(blurred.std(), image.std())

    def test_patch_mask_and_compose(self) -> None:
        mask = patch_mask([0, 3], (2, 2), 4)
        self.assertEqual(mask.sum(), 32)
        self.assertTrue(mask[0:4, 0:4].all() and mask[4:8, 4:8].all())
        image = np.zeros((8, 8, 3))
        out = compose(image, np.ones((8, 8, 3)), mask)
        self.assertEqual(out[mask].min(), 1.0)
        self.assertEqual(out[~mask].max(), 0.0)


class PartitionShapToyTest(unittest.TestCase):
    """Run real Partition SHAP on a tiny image with a known linear model."""

    def test_partition_shap_additivity_on_linear_model(self) -> None:
        try:
            import shap
        except ImportError:
            self.skipTest("shap is not installed")
        rng = np.random.default_rng(2)
        image = rng.random((64, 64, 3)).astype(np.float32)
        blurred = box_blur(image, 32)
        weights = rng.normal(size=image.shape)

        def model(batch: np.ndarray) -> np.ndarray:
            batch = np.asarray(batch, dtype=np.float64).reshape(-1, *image.shape)
            return (batch * weights).sum(axis=(1, 2, 3)).reshape(-1, 1)

        masker = shap.maskers.Image(blurred, shape=image.shape)
        explainer = shap.Explainer(model, masker, algorithm="partition", output_names=["score"])
        explanation = explainer(image[None], max_evals=300, batch_size=16, silent=True)
        values, base = unpack_image_explanation(explanation, image.shape)
        score = float(model(image[None])[0, 0])
        self.assertAlmostEqual(base, float(model(blurred[None])[0, 0]), places=6)
        result = calculate_additivity(score, base, values)
        self.assertLess(abs(result["additivity_residual"]), 1e-6)
        grid = patch_grid(pixel_map(values), 32)
        self.assertAlmostEqual(grid.sum(), values.sum(), places=6)


class FaithfulnessTest(unittest.TestCase):
    def test_faithfulness_on_linear_patch_model(self) -> None:
        image = np.ones((64, 64, 3), dtype=np.float32)
        blurred = np.zeros_like(image)
        patch_weights = np.array([[4.0, 1.0], [2.0, 0.5]])
        total = patch_weights.sum()

        def model(batch: np.ndarray) -> np.ndarray:
            batch = np.asarray(batch).reshape(-1, 64, 64, 3)
            kept = batch.mean(axis=3).reshape(-1, 2, 32, 2, 32).mean(axis=(2, 4))
            return ((kept * patch_weights).sum(axis=(1, 2)) / total).reshape(-1, 1)

        rows = compute_faithfulness("T", image, blurred, patch_weights, 1.0, [0.25, 0.5], 32, model, 5, 0)
        self.assertAlmostEqual(rows[0]["comprehensiveness_drop"], 4.0 / total)
        self.assertAlmostEqual(rows[0]["sufficiency_gap"], 1 - 4.0 / total)
        self.assertAlmostEqual(rows[1]["comprehensiveness_drop"], 6.0 / total)
        self.assertGreaterEqual(rows[1]["shap_minus_random_drop"], 0)

    def test_spearman_and_stability(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(spearman(a, a * 10), 1.0)
        self.assertAlmostEqual(spearman(a, -a), -1.0)
        row = stability_row("X", a, a + 0.01, (500, 1000))
        self.assertEqual(row["top_5_overlap"], 1.0)
        self.assertEqual(row["same_top_patch"], 1)


if __name__ == "__main__":
    unittest.main()
