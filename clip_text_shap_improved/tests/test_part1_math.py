"""Fast unit tests for the three-output margin identities."""

from __future__ import annotations

import unittest

import numpy as np

from shap_utils import additivity


class Part1MathTests(unittest.TestCase):
    def test_direct_margin_equals_positive_minus_negative(self) -> None:
        class_values = np.asarray([[0.2, 0.4], [-0.1, 0.3], [0.05, -0.2]])
        direct_margin = class_values[:, 1] - class_values[:, 0]
        values = np.column_stack([class_values, direct_margin])
        bases = np.asarray([0.7, 0.6, -0.1])
        scores = bases + values.sum(axis=0)
        result = additivity(scores, bases, values)
        np.testing.assert_allclose(result["residual"], np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(values[:, 2], values[:, 1] - values[:, 0], atol=0)


if __name__ == "__main__":
    unittest.main()
