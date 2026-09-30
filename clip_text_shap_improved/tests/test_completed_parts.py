"""Post-run gates for the completed Part 1 and Part 2 artifacts."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, read_csv


ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT / "outputs" / "part1_margin_baseline"
PART2 = ROOT / "outputs" / "part2_prompt_validation"


class CompletedPartTests(unittest.TestCase):
    def test_part1_gate_and_row_counts(self) -> None:
        checks = load_json(PART1 / "values" / "part1_checks.json")
        self.assertTrue(checks["gate_passed"])
        self.assertTrue(all(checks["tests"].values()))
        self.assertEqual(len(read_csv(PART1 / "values" / "sentence_summary.csv")), 20)
        self.assertLessEqual(checks["maximum_absolute_additivity_residual"], 1e-5)
        self.assertLessEqual(checks["maximum_absolute_margin_linearity_error"], 1e-10)

    def test_part2_selection_and_boundaries(self) -> None:
        summary = load_json(PART2 / "values" / "part2_summary.json")
        manifest = load_json(ROOT / "configs" / "frozen_prompt_manifest.json")
        self.assertEqual(summary["selected_family"], "balanced_diverse_ensemble")
        self.assertEqual(summary["validation_rows"], 6735)
        self.assertEqual(summary["stability_rows"], 40)
        self.assertFalse(summary["final_evaluation_used"])
        self.assertFalse(manifest["selection_uses_final_evaluation"])
        self.assertEqual(len(read_csv(PART2 / "values" / "prompt_performance.csv")), 5)
        self.assertEqual(len(read_csv(PART2 / "values" / "validation_predictions.csv")), 6735 * 5)
        self.assertEqual(len(read_csv(PART2 / "values" / "stability_margin_shap_additivity.csv")), 40 * 5)

    def test_every_plot_has_png_and_pdf(self) -> None:
        for part in (PART1, PART2):
            png = {path.relative_to(part / "plots").with_suffix("") for path in (part / "plots").rglob("*.png")}
            pdf = {path.relative_to(part / "plots").with_suffix("") for path in (part / "plots").rglob("*.pdf")}
            self.assertEqual(png, pdf, part.name)
        examples = list((PART1 / "plots" / "presentation_examples").glob("*.png"))
        self.assertEqual(len(examples), 5)


if __name__ == "__main__":
    unittest.main()
