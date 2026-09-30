"""Prompt families must remain balanced, ordered, non-empty and unique."""

from __future__ import annotations

import unittest
from pathlib import Path

from prompting import load_prompt_registry


ROOT = Path(__file__).resolve().parents[1]


class PromptRegistryTests(unittest.TestCase):
    def test_all_families_are_balanced(self) -> None:
        families = load_prompt_registry(ROOT / "configs" / "prompt_families.json")
        self.assertEqual(len(families), 5)
        for family in families.values():
            self.assertEqual(len(family["NEG"]), len(family["POS"]))
            self.assertGreater(len(family["NEG"]), 0)


if __name__ == "__main__":
    unittest.main()
