"""The copied evaluation set is exactly the frozen, balanced 500-sentence file."""

from __future__ import annotations

from collections import Counter

from data_validation import normalize_text, order_fingerprint
from io_utils import load_json, project_path, sha256_file


def test_exactly_500_rows(rows):
    assert len(rows) == 500


def test_class_balance_250_250(rows):
    assert Counter(row["gold_label"] for row in rows) == {"NEG": 250, "POS": 250}


def test_only_neg_pos_labels_and_no_missing_text(rows):
    assert {row["gold_label"] for row in rows} == {"NEG", "POS"}
    assert all(row["text"].strip() for row in rows)


def test_no_duplicate_ids_or_normalized_text(rows):
    ids = [row["sentence_id"] for row in rows]
    assert len(set(ids)) == 500
    assert len({normalize_text(row["text"]) for row in rows}) == 500


def test_copied_hash_matches_source_and_manifests(config):
    copied = sha256_file(project_path(config["dataset"]["evaluation_path"]))
    source = sha256_file(project_path(config["dataset"]["source_path"]))
    source_manifest = load_json(project_path(config["dataset"]["source_manifest_path"]))
    manifest = load_json(project_path(config["dataset"]["manifest_path"]))
    assert copied == source == source_manifest["outputs"]["final_evaluation"]["sha256"]
    assert manifest["sha256_copied"] == copied
    assert manifest["all_checks_passed"] is True
    assert all(manifest["validation"]["checks"].values())


def test_both_models_use_the_dataset_sentence_order(rows, model_scores, sentence_results):
    expected = [row["sentence_id"] for row in rows]
    for model in ("bert", "clip"):
        scored = [row["sentence_id"] for row in model_scores if row["model"] == model]
        explained = [row["sentence_id"] for row in sentence_results if row["model"] == model]
        assert scored == expected
        assert explained == expected
    manifest = load_json(project_path("data/dataset_manifest.json"))
    assert manifest["validation"]["sentence_order_sha256"] == order_fingerprint(expected)
