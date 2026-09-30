"""Completeness and internal consistency of the saved metric outputs."""

from __future__ import annotations

import numpy as np
import pytest

from io_utils import VALUES_DIR, read_csv
from metrics_core import fraction_label
from source_link import OUTPUT_DIR

MODELS = ("bert", "clip")


def test_all_gates_passed(checks):
    assert checks["gate_passed"] is True
    assert checks["checks"] and all(checks["checks"].values())
    assert checks["checks"]["source_experiment_unchanged"] is True
    assert checks["rationale"]["dataset_has_human_rationale_annotations"] is False
    assert checks["rationale"]["lookup_sources"].get("not_found", 0) == 0


def test_per_sentence_rows_cover_both_models_in_dataset_order(per_sentence, dataset_rows):
    assert len(per_sentence) == 1000
    expected = [row["sentence_id"] for row in dataset_rows]
    for model in MODELS:
        assert [row["sentence_id"] for row in per_sentence if row["model"] == model] == expected


def test_headline_values_are_fraction_means(per_sentence, own_config):
    labels = [fraction_label(f) for f in own_config["fractions"]]
    for row in per_sentence:
        for metric in ("comprehensiveness", "sufficiency"):
            per_fraction = [float(row[f"{metric}_{label}"]) for label in labels]
            assert float(row[metric]) == pytest.approx(np.mean(per_fraction), abs=1e-12)
            scale = float(row["model_scale_mean_abs_decision_score"])
            assert float(row[metric]) == pytest.approx(float(row[f"{metric}_raw"]) / scale, rel=1e-12)
        drops = [float(row[f"deletion_normalized_drop_{label}"]) for label in labels]
        assert float(row["deletion_aopc"]) == pytest.approx(np.mean(drops), abs=1e-12)


def test_deletion_aopc_reproduces_source_sentence_results(per_sentence):
    source = {(r["model"], r["sentence_id"]): float(r["aopc_shap_abs"]) for r in read_csv(VALUES_DIR / "sentence_results.csv")}
    for row in per_sentence:
        assert float(row["deletion_aopc"]) == pytest.approx(source[(row["model"], row["sentence_id"])], abs=1e-12)


def test_scale_is_mean_absolute_margin(per_sentence):
    for model in MODELS:
        subset = [row for row in per_sentence if row["model"] == model]
        scale = np.mean([abs(float(row["original_margin"])) for row in subset])
        for row in subset:
            assert float(row["model_scale_mean_abs_decision_score"]) == pytest.approx(scale, rel=1e-12)


def test_comparison_table_matches_per_sentence_means(per_sentence):
    from run_metrics import TABLE_ROWS

    table = {row["Metric"]: row for row in read_csv(OUTPUT_DIR / "comparison_table.csv")}
    assert list(table) == list(TABLE_ROWS)
    for metric, column in TABLE_ROWS.items():
        for model, label in (("bert", "BERT"), ("clip", "CLIP")):
            subset = [row for row in per_sentence if row["model"] == model]
            if column.startswith("rationale_"):
                subset = [row for row in subset if row["rationale_evaluated"] == "1"]
            mean = np.mean([float(row[column]) for row in subset])
            assert float(table[metric][label]) == pytest.approx(mean, abs=1e-12)


def test_rationale_sentences_identical_for_both_models(per_sentence, checks):
    evaluated = {model: [row["sentence_id"] for row in per_sentence if row["model"] == model and row["rationale_evaluated"] == "1"]
                 for model in MODELS}
    assert evaluated["bert"] == evaluated["clip"]
    assert len(evaluated["bert"]) == checks["rationale"]["evaluated_sentences"]
    excluded = [row for row in per_sentence if row["rationale_evaluated"] == "0"]
    assert all(row["rationale_reference_word_count"] == "0" and row["rationale_f1"] == "NA" for row in excluded)
    assert sorted({row["sentence_id"] for row in excluded}) == sorted(checks["rationale"]["excluded_sentence_ids"])


def test_rationale_scores_recompute_from_audit_files(per_sentence, audit):
    from rationale_reference import is_sentiment_word

    reference: dict[str, set[int]] = {}
    for row in read_csv(OUTPUT_DIR / "rationale_reference_words.csv"):
        rating = float(row["sst_rating"]) if row["sst_rating"] else None
        assert int(row["is_reference"]) == int(is_sentiment_word(rating))
        if int(row["is_reference"]):
            reference.setdefault(row["sentence_id"], set()).add(int(row["word_index"]))
    rows = {(row["model"], row["sentence_id"]): row for row in per_sentence}
    for item in audit:
        gold = reference.get(item["sentence_id"], set())
        row = rows[(item["model"], item["sentence_id"])]
        label = fraction_label(float(item["fraction"]))
        if not gold:
            assert row[f"rationale_f1_{label}"] == "NA"
            continue
        top = {int(value) for value in item["top_word_indices"].split()}
        hits = len(top & gold)
        precision, recall = hits / len(top), hits / len(gold)
        f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
        assert float(row[f"rationale_precision_{label}"]) == pytest.approx(precision, abs=1e-12)
        assert float(row[f"rationale_recall_{label}"]) == pytest.approx(recall, abs=1e-12)
        assert float(row[f"rationale_f1_{label}"]) == pytest.approx(f1, abs=1e-12)


def test_audit_texts_are_exact_whole_word_complements(audit, dataset_rows, source_config):
    from metrics_core import comprehensiveness_keep, sufficiency_keep
    from whole_word_masker import create_word_masker, render_coalition, split_word_units

    assert len(audit) == 4000
    text_of = {row["sentence_id"]: row["text"] for row in dataset_rows}
    masker = create_word_masker()
    for row in audit:
        text = text_of[row["sentence_id"]]
        units = split_word_units(text)
        is_content = [int(unit.is_content) for unit in units]
        top = [int(value) for value in row["top_word_indices"].split()]
        assert len(top) == int(row["top_k"]) and all(is_content[index] for index in top)
        assert " | ".join(units[index].text for index in top) == row["top_words"]
        assert render_coalition(masker, text, sufficiency_keep(is_content, top)) == row["sufficiency_text"]
        assert render_coalition(masker, text, comprehensiveness_keep(is_content, top)) == row["comprehensiveness_text"]
        tolerance = float(source_config["tolerances"]["batch_parity"][row["model"]])
        assert abs(float(row["comprehensiveness_margin_rescored"]) - float(row["comprehensiveness_margin_stored"])) <= tolerance


def test_sufficiency_is_zero_when_every_content_word_is_kept(audit, source_config):
    full = [row for row in audit if row["top_k"] == row["content_word_count"]]
    assert full, "expected some short sentences where k equals the content-word count"
    for row in full:
        tolerance = float(source_config["tolerances"]["batch_parity"][row["model"]])
        assert abs(float(row["sufficiency_margin"]) - float(row["original_margin"])) <= tolerance
