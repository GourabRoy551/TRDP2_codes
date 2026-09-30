"""Checks for curated, balanced dataset construction."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from prepare_dataset import (  # noqa: E402
    project_path,
    read_curated,
    read_existing,
    read_sst2,
    validate_curated_sources,
)


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8")
        )
        cls.existing = read_existing(
            project_path(cls.config["existing_sentences_path"])
        )
        cls.curated = read_curated(
            project_path(cls.config["curated_additions_path"])
        )
        cls.candidates = read_sst2(project_path(cls.config["sst2_dev_path"]))

    def test_existing_set_is_unchanged_s1_to_s10(self) -> None:
        self.assertEqual(
            [row["sentence_id"] for row in self.existing],
            [f"S{index}" for index in range(1, 11)],
        )

    def test_curated_set_is_balanced_and_source_valid(self) -> None:
        validate_curated_sources(self.curated, self.candidates)
        combined = [*self.existing, *self.curated]
        self.assertEqual(len(combined), 20)
        self.assertEqual(
            Counter(row["gold_label"] for row in combined),
            Counter({"NEG": 10, "POS": 10}),
        )

    def test_curated_sentences_are_clean_and_complete(self) -> None:
        self.assertEqual(
            [row["sentence_id"] for row in self.curated],
            [f"D{index:02d}" for index in range(1, 11)],
        )
        forbidden = (" ,", " .", " ?", " !", " 's", " n't", "``", "...")
        for row in self.curated:
            text = row["text"]
            self.assertTrue(text[0].isupper(), text)
            self.assertIn(text[-1], ".!?", text)
            self.assertGreaterEqual(len(text.split()), 7, text)
            for fragment in forbidden:
                self.assertNotIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
