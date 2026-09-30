"""Unit tests for dual-output SHAP arithmetic and WordPiece aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from token_aggregation_dual import (  # noqa: E402
    aggregate_wordpieces_dual,
    calculate_additivity_dual,
    margin_components,
)


class DualMathTest(unittest.TestCase):
    def test_class_additivity(self) -> None:
        bases = np.asarray([0.5, -0.2])
        values = np.asarray([[0.2, -0.1], [0.3, 1.0], [-0.1, 0.4]])
        outputs = bases + values.sum(axis=0)
        reconstructed, residuals = calculate_additivity_dual(
            outputs, bases, values
        )
        np.testing.assert_allclose(reconstructed, outputs, rtol=0, atol=1e-12)
        np.testing.assert_allclose(residuals, [0.0, 0.0], rtol=0, atol=1e-12)

    def test_wordpiece_aggregation_preserves_each_class_sum(self) -> None:
        tokens = ["[CLS]", "res", "##ili", "##ence", ".", "[SEP]"]
        values = np.asarray(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [0.2, -0.2],
                [-0.05, 0.4],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        words, aggregated = aggregate_wordpieces_dual(tokens, values)
        self.assertEqual(words, ["resilience"])
        np.testing.assert_allclose(aggregated, [[0.25, 0.1]], atol=1e-12)

    def test_margin_is_pos_minus_neg(self) -> None:
        values = np.asarray([[0.2, 0.7], [-0.3, 0.1]])
        margin_values, margin_base = margin_components(values, [0.4, -0.2])
        np.testing.assert_allclose(margin_values, [0.5, 0.4], atol=1e-12)
        self.assertAlmostEqual(margin_base, -0.6)


if __name__ == "__main__":
    unittest.main()
