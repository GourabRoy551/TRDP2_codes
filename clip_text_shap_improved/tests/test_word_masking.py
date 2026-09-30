"""Structural and SHAP checks for genuine whole-word masking."""

from __future__ import annotations

import unittest

import numpy as np
import shap

from shap_utils import additivity, unpack_explanation
from word_masking import create_word_masker, split_word_units


class WordMaskingTests(unittest.TestCase):
    def test_contractions_and_hyphen_compounds_are_indivisible(self) -> None:
        units = [unit.text for unit in split_word_units("It's paint-by-numbers, but well-acted.")]
        self.assertEqual(
            units,
            ["It's", "paint-by-numbers", ",", "but", "well-acted", "."],
        )

    def test_masker_has_one_feature_per_unit_plus_boundaries(self) -> None:
        text = "A deeply moving film."
        masker = create_word_masker()
        self.assertEqual(masker.shape(text)[1], len(split_word_units(text)) + 2)
        mask = np.ones(masker.shape(text)[1], dtype=bool)
        mask[2] = False
        masked = str(np.asarray(masker(mask, text)[0], dtype=object).reshape(-1)[0])
        self.assertNotIn("deeply", masked)
        self.assertIn("moving", masked)

    def test_partition_shap_reconstructs_three_outputs(self) -> None:
        def model(texts: np.ndarray) -> np.ndarray:
            rows = []
            for text in np.asarray(texts, dtype=object).reshape(-1):
                value = str(text)
                negative = -0.01 * len(value)
                positive = 0.1 * value.casefold().count("good")
                rows.append([negative, positive, positive - negative])
            return np.asarray(rows)

        text = "A good, well-acted film."
        masker = create_word_masker()
        explainer = shap.Explainer(
            model, masker, algorithm="partition", output_names=["NEG", "POS", "MARGIN"]
        )
        explanation = explainer([text], max_evals=100, batch_size=32, silent=True)
        _, values, bases = unpack_explanation(explanation, 3)
        result = additivity(model([text])[0], bases, values)
        np.testing.assert_allclose(result["residual"], np.zeros(3), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
