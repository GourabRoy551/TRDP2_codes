"""All 1,000 explanation records, derived tables, bootstrap and pre-SHAP selection are complete."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from io_utils import CHECKPOINT_DIR, METRICS_DIR, REPORTS_DIR, VALUES_DIR, load_json, read_csv


@pytest.mark.parametrize("model", ["bert", "clip"])
def test_500_complete_checkpoint_records_per_model(model, rows):
    with (CHECKPOINT_DIR / f"{model}_shap_records.jsonl").open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    assert len(records) == 500
    assert [record["sentence_id"] for record in records] == [row["sentence_id"] for row in rows]
    for record in records:
        assert len(record["word_values"]) == len(record["words"]) and all(len(v) == 3 for v in record["word_values"])
        assert len(record["base_values"]) == 3


def test_1000_sentence_rows(sentence_results):
    assert len(sentence_results) == 1000
    assert Counter(row["model"] for row in sentence_results) == {"bert": 500, "clip": 500}
    required = {
        "model", "sentence_id", "text", "gold_label", "prediction", "correct", "neg_score", "pos_score", "margin",
        "neg_base_value", "pos_base_value", "margin_base_value", "neg_additivity_residual", "pos_additivity_residual",
        "margin_additivity_residual", "max_additivity_residual", "margin_linearity_error", "shap_runtime_seconds",
        "shap_model_evaluations", "word_count", "internal_token_count", "aopc_shap_abs", "aopc_random",
        "flip_rate_shap_abs", "flip_rate_random",
    }
    assert required <= set(sentence_results[0])


def test_word_rows_cover_every_word(sentence_results, word_values):
    for model in ("bert", "clip"):
        expected = sum(int(row["word_count"]) for row in sentence_results if row["model"] == model)
        assert sum(row["model"] == model for row in word_values) == expected


def test_faithfulness_rows_complete(config):
    rows = read_csv(VALUES_DIR / "faithfulness_by_fraction.csv")
    fractions = config["faithfulness"]["fractions"]
    repeats = int(config["faithfulness"]["random_repeats"])
    assert len(rows) == 1000 * len(fractions) * (2 + repeats)
    assert {float(row["fraction"]) for row in rows} == {float(value) for value in fractions}
    random_rows = [row for row in rows if row["ranking"] == "random"]
    assert {int(row["random_repeat"]) for row in random_rows} == set(range(repeats))


def test_500_pairwise_rows_and_primary_metrics():
    pairs = read_csv(METRICS_DIR / "pairwise_explanation_comparison.csv")
    assert len(pairs) == 500 and len({row["sentence_id"] for row in pairs}) == 500
    summary = {(row["metric"], row["subset"]) for row in read_csv(METRICS_DIR / "pairwise_explanation_summary.csv")}
    for metric in ("spearman_abs_margin", "top_k_overlap_rate", "top_k_jaccard", "sign_agreement"):
        for subset in ("all", "gold_NEG", "gold_POS", "both_correct", "not_both_correct"):
            assert (metric, subset) in summary
    classification = read_csv(METRICS_DIR / "classification_metrics.csv")
    assert {row["model"] for row in classification} == {"bert", "clip"}
    assert all(0.0 <= float(row["macro_f1"]) <= 1.0 for row in classification)


def test_bootstrap_uses_1000_iterations():
    rows = read_csv(METRICS_DIR / "bootstrap_confidence_intervals.csv")
    assert rows and all(int(row["iterations"]) == 1000 for row in rows)
    metrics = {(row["scope"], row["metric"]) for row in rows}
    for model in ("bert", "clip"):
        assert (model, "macro_f1") in metrics and (model, "mean_aopc_shap_abs") in metrics and (model, "mean_aopc_random") in metrics
    assert ("bert_vs_clip[all]", "spearman_abs_margin") in metrics
    assert ("bert_vs_clip[all]", "top_k_overlap_rate") in metrics


def test_all_metric_files_exist():
    for name in ("classification_metrics", "bootstrap_confidence_intervals", "additivity_summary", "pairwise_explanation_comparison",
                 "pairwise_explanation_summary", "subgroup_results", "runtime_summary"):
        assert (METRICS_DIR / f"{name}.csv").stat().st_size > 0
    assert load_json(METRICS_DIR / "acceptance_checks.json")["gate_passed"] is True


def test_representatives_chosen_before_any_shap_record():
    selection = read_csv(VALUES_DIR / "representative_selection.csv")
    assert len(selection) == 5
    assert [row["role"] for row in selection] == [
        "correct_positive", "correct_negative", "contrastive", "long_multi_subword", "bert_clip_disagreement"]
    assert all(row["selection_uses_shap_values"] == "False" for row in selection)
    first_shap = min(
        json.loads(open(CHECKPOINT_DIR / f"{model}_shap_records.jsonl", encoding="utf-8").readline())["completed_at_utc"]
        for model in ("bert", "clip")
    )
    assert all(row["selected_at_utc"] < first_shap for row in selection)


def test_reports_exist():
    assert (REPORTS_DIR / "FINAL_REPORT.md").stat().st_size > 5000
    assert (REPORTS_DIR / "RESULTS_SUMMARY.md").stat().st_size > 500
