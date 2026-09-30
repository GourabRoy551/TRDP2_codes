"""Post-run integrity checks for the complete Part 6 evaluation."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, read_csv, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "part6_full_evaluation"


class Part6GateTests(unittest.TestCase):
    def test_part6_gate_and_frozen_dataset(self) -> None:
        checks = load_json(OUTPUT / "values" / "part6_checks.json")
        dataset = ROOT / "data" / "final_evaluation_500.csv"
        self.assertTrue(checks["gate_passed"])
        self.assertTrue(all(checks["tests"].values()))
        self.assertEqual(checks["evaluation_rows"], 500)
        self.assertEqual(checks["class_counts"], {"NEG": 250, "POS": 250})
        self.assertEqual(checks["dataset_sha256"], sha256_file(dataset))
        self.assertEqual(checks["frozen_shap_budget"], 500)

    def test_complete_numerical_results(self) -> None:
        values = OUTPUT / "values"
        self.assertEqual(len(read_csv(values / "sentence_results.csv")), 1000)
        self.assertEqual(len(read_csv(values / "faithfulness_by_fraction.csv")), 4000)
        self.assertGreater(len(read_csv(values / "word_shap_values.csv")), 20000)
        self.assertEqual(len(read_csv(values / "bootstrap_confidence_intervals.csv")), 16)

    def test_part6_plot_pairing(self) -> None:
        plot_root = OUTPUT / "plots"
        png = {path.relative_to(plot_root).with_suffix("") for path in plot_root.rglob("*.png")}
        pdf = {path.relative_to(plot_root).with_suffix("") for path in plot_root.rglob("*.pdf")}
        self.assertEqual(png, pdf)
        self.assertEqual(len(png), 10)
        self.assertEqual(len(list((plot_root / "presentation_examples").glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
