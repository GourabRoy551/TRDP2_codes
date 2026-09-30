"""Independent integrity checks for the prepared SST-2 datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class PreparedDatasetTests(unittest.TestCase):
    def test_expected_counts_and_balance(self) -> None:
        expected = {
            "train_development.csv": (60612, {"NEG": 26801, "POS": 33811}),
            "prompt_validation.csv": (6735, {"NEG": 2978, "POS": 3757}),
            "shap_stability_40.csv": (40, {"NEG": 20, "POS": 20}),
            "final_evaluation_500.csv": (500, {"NEG": 250, "POS": 250}),
            "pilot_20_curated.csv": (20, {"NEG": 10, "POS": 10}),
        }
        for name, (row_count, labels) in expected.items():
            rows = read_csv(name)
            self.assertEqual(len(rows), row_count, name)
            self.assertEqual(dict(Counter(row["gold_label"] for row in rows)), labels, name)

    def test_no_normalized_text_leakage(self) -> None:
        training = {row["normalized_text_sha256"] for row in read_csv("train_development.csv")}
        validation = {row["normalized_text_sha256"] for row in read_csv("prompt_validation.csv")}
        self.assertFalse(training & validation)

    def test_training_exclusions_are_explicit(self) -> None:
        rows = read_csv("excluded_train_rows.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_row_id"] for row in rows}, {"4246", "8304"})
        self.assertEqual({row["text"] for row in rows}, {"(", ")"})
        self.assertEqual({row["exclusion_reason"] for row in rows}, {"empty_normalized_text"})

    def test_final_set_is_unique_balanced_and_within_clip_limit(self) -> None:
        rows = read_csv("final_evaluation_500.csv")
        self.assertEqual(len({row["normalized_text_sha256"] for row in rows}), 500)
        self.assertFalse(any(int(row["clip_bpe_token_count"]) > 77 for row in rows))
        inspected = {"29", "59", "129", "181", "212", "327", "429", "504", "591", "603"}
        self.assertFalse({row["source_row_id"] for row in rows} & inspected)

    def test_manifest_hashes(self) -> None:
        for manifest_name in ["train_internal_split_manifest.json", "final_evaluation_manifest.json"]:
            manifest = json.loads((DATA / manifest_name).read_text(encoding="utf-8"))
            for output in manifest["outputs"].values():
                path = DATA / output["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, output["sha256"], path.name)


if __name__ == "__main__":
    unittest.main()

