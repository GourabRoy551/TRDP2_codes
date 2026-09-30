"""Unit checks for SHAP additivity and BERT WordPiece aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from token_aggregation import (  # noqa: E402
    aggregate_wordpieces,
    calculate_additivity,
    content_feature_indices,
    wordpiece_groups,
)


class AdditivityTest(unittest.TestCase):
    def test_base_plus_shap_values_reconstructs_margin(self) -> None:
        reconstructed, residual = calculate_additivity(
            model_margin=2.5,
            base_value=0.25,
            shap_values=[0.5, 1.0, 0.75],
        )
        self.assertAlmostEqual(reconstructed, 2.5)
        self.assertAlmostEqual(residual, 0.0)

    def test_nonzero_residual_is_reported_with_correct_sign(self) -> None:
        reconstructed, residual = calculate_additivity(2.6, 0.25, [0.5, 1.0, 0.75])
        self.assertAlmostEqual(reconstructed, 2.5)
        self.assertAlmostEqual(residual, 0.1)

    def test_wordpiece_aggregation_preserves_signed_sum(self) -> None:
        tokens = ["[CLS]", "res", "##ili", "##ence", ".", "[SEP]"]
        values = [0.0, 0.10, 0.20, -0.05, 0.0, 0.0]
        words, aggregated = aggregate_wordpieces(tokens, values)
        self.assertEqual(words, ["resilience"])
        self.assertAlmostEqual(aggregated[0], 0.25)
        self.assertEqual(wordpiece_groups(tokens), [("resilience", [1, 2, 3])])

    def test_content_indices_exclude_special_tokens_and_punctuation(self) -> None:
        tokens = ["[CLS]", "painfully", ",", "boring", "[SEP]"]
        self.assertEqual(content_feature_indices(tokens), [1, 3])


if __name__ == "__main__":
    unittest.main()
