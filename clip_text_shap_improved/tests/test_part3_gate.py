"""Completed Part 3 artifacts must satisfy the masking gate."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, read_csv


ROOT = Path(__file__).resolve().parents[1]


class Part3GateTests(unittest.TestCase):
    def test_part3_gate(self) -> None:
        output = ROOT / "outputs" / "part3_word_partition"
        checks = load_json(output / "values" / "part3_checks.json")
        self.assertTrue(checks["gate_passed"])
        self.assertTrue(all(checks["tests"].values()))
        self.assertEqual(checks["partial_word_mask_violations"], 0)
        self.assertEqual(checks["unassigned_non_special_bpe"], 0)
        self.assertEqual(len(read_csv(output / "values" / "sentence_comparison.csv")), 40)


if __name__ == "__main__":
    unittest.main()
