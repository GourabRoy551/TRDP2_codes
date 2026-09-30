"""Post-run integrity checks for the Part 5 SHAP-budget selection."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, read_csv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "part5_shap_budget"


class Part5GateTests(unittest.TestCase):
    def test_budget_gate_and_selection_manifest(self) -> None:
        checks = load_json(OUTPUT / "values" / "part5_checks.json")
        manifest = load_json(ROOT / "configs" / "frozen_shap_budget_manifest.json")
        self.assertTrue(checks["gate_passed"])
        self.assertTrue(all(checks["tests"].values()))
        self.assertFalse(checks["final_evaluation_used"])
        self.assertFalse(manifest["selection_uses_final_evaluation"])
        self.assertEqual(checks["stability_sentences"], 40)
        self.assertEqual(checks["run_rows"], 2 * 40 * 3)
        self.assertEqual(checks["comparison_rows"], 2 * 40 * 3)
        self.assertEqual(manifest["selected_budget"], checks["selected_budget"])
        self.assertEqual(manifest["selected_budget"], min(manifest["eligible_budgets"]))

    def test_each_condition_passes_the_selected_budget(self) -> None:
        checks = load_json(OUTPUT / "values" / "part5_checks.json")
        rows = read_csv(OUTPUT / "values" / "budget_stability_summary.csv")
        selected = [
            row for row in rows if int(row["budget"]) == int(checks["selected_budget"])
        ]
        self.assertEqual({row["condition"] for row in selected}, set(checks["conditions"]))
        self.assertTrue(all(row["all_thresholds_passed"] == "True" for row in selected))
        self.assertEqual(len(read_csv(OUTPUT / "values" / "determinism_checks.csv")), 10)
        self.assertGreater(len(read_csv(OUTPUT / "values" / "budget_word_shap_values.csv")), 0)

    def test_part5_plot_pairing(self) -> None:
        plot_root = OUTPUT / "plots"
        png = {path.relative_to(plot_root).with_suffix("") for path in plot_root.rglob("*.png")}
        pdf = {path.relative_to(plot_root).with_suffix("") for path in plot_root.rglob("*.pdf")}
        self.assertEqual(png, pdf)
        self.assertEqual(len(png), 4)


if __name__ == "__main__":
    unittest.main()
