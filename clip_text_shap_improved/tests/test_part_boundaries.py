"""Check that Parts 1 and 2 use only their approved datasets."""

from __future__ import annotations

import unittest
from pathlib import Path

from io_utils import load_json, project_path, read_csv


ROOT = Path(__file__).resolve().parents[1]


class PartBoundaryTests(unittest.TestCase):
    def test_part1_pilot_matches_legacy_records(self) -> None:
        config = load_json(ROOT / "configs" / "part1_margin.json")
        prepared = {
            row["sentence_id"]: (row["text"], row["gold_label"])
            for row in read_csv(project_path(config["pilot_path"]))
        }
        legacy = {
            row["sentence_id"]: (row["text"], row["gold_label"])
            for row in read_csv(project_path(config["legacy_pilot_path"]))
        }
        self.assertEqual(prepared, legacy)

    def test_part2_paths_do_not_reference_final_evaluation(self) -> None:
        config = load_json(ROOT / "configs" / "part2_prompt_validation.json")
        serialized = str(config).casefold()
        self.assertNotIn("final_evaluation_500", serialized)
        self.assertEqual(len(read_csv(project_path(config["validation_path"]))), 6735)
        self.assertEqual(len(read_csv(project_path(config["stability_path"]))), 40)


if __name__ == "__main__":
    unittest.main()
