"""Completed Part 4 artifacts must satisfy the frozen-head gate."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, read_csv


ROOT = Path(__file__).resolve().parents[1]


class Part4GateTests(unittest.TestCase):
    def test_part4_gate_and_saved_head(self) -> None:
        output = ROOT / "outputs" / "part4_frozen_head"
        checks = load_json(output / "values" / "part4_checks.json")
        self.assertTrue(checks["gate_passed"])
        self.assertTrue(all(checks["tests"].values()))
        self.assertFalse(checks["final_evaluation_used"])
        self.assertEqual(checks["training_validation_normalized_overlap"], 0)
        self.assertGreater(checks["macro_f1_improvement"], 0)
        self.assertTrue((output / "values" / "clip_linear_head.npz").exists())
        self.assertEqual(len(read_csv(output / "values" / "validation_predictions.csv")), 6735)
        self.assertEqual(len(read_csv(output / "values" / "trained_head_shap_summary.csv")), 5)

    def test_part3_and_part4_plot_pairing(self) -> None:
        roots = [
            ROOT / "outputs" / "part3_word_partition" / "plots",
            ROOT / "outputs" / "part4_frozen_head" / "plots",
        ]
        for root in roots:
            png = {path.relative_to(root).with_suffix("") for path in root.rglob("*.png")}
            pdf = {path.relative_to(root).with_suffix("") for path in root.rglob("*.pdf")}
            self.assertEqual(png, pdf)
        self.assertEqual(len(list((roots[0] / "presentation_examples").glob("*.png"))), 5)
        self.assertEqual(len(list((roots[1] / "trained_head_shap").glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
