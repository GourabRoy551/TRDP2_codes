"""Post-run integrity checks for the final BERT--CLIP comparison."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, read_csv, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "part7_final_comparison"


class Part7GateTests(unittest.TestCase):
    def test_part7_gate_and_manifest(self) -> None:
        checks = load_json(OUTPUT / "values" / "part7_checks.json")
        frozen = load_json(ROOT / "configs" / "frozen_part7_results_manifest.json")
        self.assertTrue(checks["gate_passed"])
        self.assertTrue(all(checks["tests"].values()))
        self.assertTrue(frozen["gate_passed"])
        self.assertEqual(checks["dataset_sha256"], frozen["dataset_sha256"])
        self.assertEqual(checks["dataset_sha256"], sha256_file(ROOT / "data" / "final_evaluation_500.csv"))
        self.assertLessEqual(checks["maximum_bert_additivity_residual"], checks["additivity_tolerance"])
        self.assertLessEqual(checks["maximum_bert_margin_linearity_error"], checks["margin_linearity_tolerance"])

    def test_complete_three_model_comparison(self) -> None:
        values = OUTPUT / "values"
        self.assertEqual(len(read_csv(values / "bert_sentence_results.csv")), 500)
        self.assertEqual(len(read_csv(values / "pairwise_explanation_comparison.csv")), 1500)
        self.assertEqual(len(read_csv(values / "final_model_performance.csv")), 3)
        self.assertEqual(len(read_csv(values / "pairwise_explanation_summary.csv")), 3)
        self.assertGreater(len(read_csv(values / "bert_word_shap_values.csv")), 9000)

    def test_part7_plot_pairing(self) -> None:
        plot_root = OUTPUT / "plots"
        png = {path.relative_to(plot_root).with_suffix("") for path in plot_root.rglob("*.png")}
        pdf = {path.relative_to(plot_root).with_suffix("") for path in plot_root.rglob("*.pdf")}
        self.assertEqual(png, pdf)
        self.assertEqual(len(png), 10)
        self.assertEqual(len(list((plot_root / "case_studies").glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
