"""Fast deterministic checks for the frozen-embedding linear head."""

from __future__ import annotations

import unittest

import numpy as np

from classifier_head import linear_logits, train_linear_head


class ClassifierHeadTests(unittest.TestCase):
    def test_training_is_reproducible_on_fixed_embeddings(self) -> None:
        rng = np.random.default_rng(12)
        features = rng.normal(size=(240, 12)).astype(np.float32)
        labels = (features[:, 0] - 0.4 * features[:, 1] > 0).astype(np.int64)
        arguments = dict(
            train_embeddings=features[:200],
            train_labels=labels[:200],
            validation_embeddings=features[200:],
            validation_labels=labels[200:],
            weight_decay=0.001,
            learning_rate=0.02,
            batch_size=64,
            maximum_epochs=20,
            patience=5,
            seed=42,
            balanced_class_weights=True,
        )
        first, _ = train_linear_head(**arguments)
        second, _ = train_linear_head(**arguments)
        np.testing.assert_allclose(first["weights"], second["weights"], atol=0)
        np.testing.assert_allclose(first["bias"], second["bias"], atol=0)
        logits = linear_logits(features[200:], first["weights"], first["bias"])
        self.assertEqual(logits.shape, (40, 2))


if __name__ == "__main__":
    unittest.main()
