"""Additivity and margin linearity, recomputed independently from the saved CSV values."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from io_utils import METRICS_DIR, read_csv


def word_matrix(word_values, model):
    grouped = defaultdict(list)
    for row in word_values:
        if row["model"] == model:
            grouped[row["sentence_id"]].append(row)
    return {
        sentence_id: np.asarray(
            [[float(r["neg_shap"]), float(r["pos_shap"]), float(r["margin_shap"])] for r in sorted(rows, key=lambda r: int(r["word_index"]))]
        )
        for sentence_id, rows in grouped.items()
    }


@pytest.mark.parametrize("model", ["bert", "clip"])
def test_every_explanation_has_three_complete_outputs(model, word_values, sentence_results):
    matrices = word_matrix(word_values, model)
    assert len(matrices) == 500
    for row in sentence_results:
        if row["model"] != model:
            continue
        matrix = matrices[row["sentence_id"]]
        assert matrix.shape == (int(row["word_count"]), 3)
        assert np.all(np.isfinite(matrix))
        for key in ("neg_base_value", "pos_base_value", "margin_base_value"):
            assert np.isfinite(float(row[key]))


@pytest.mark.parametrize("model", ["bert", "clip"])
def test_local_additivity_within_documented_tolerance(model, config, word_values, sentence_results):
    tolerance = float(config["tolerances"]["additivity"][model])
    matrices = word_matrix(word_values, model)
    worst = 0.0
    for row in sentence_results:
        if row["model"] != model:
            continue
        scores = np.asarray([float(row["neg_score"]), float(row["pos_score"]), float(row["margin"])])
        bases = np.asarray([float(row["neg_base_value"]), float(row["pos_base_value"]), float(row["margin_base_value"])])
        residual = np.abs(scores - (bases + matrices[row["sentence_id"]].sum(axis=0)))
        np.testing.assert_allclose(residual, [float(row[f"{k}_additivity_residual"]) for k in ("neg", "pos", "margin")], rtol=0, atol=1e-12)
        worst = max(worst, float(residual.max()))
    assert worst <= tolerance, f"{model}: max residual {worst:.3e} > {tolerance:.0e}"


@pytest.mark.parametrize("model", ["bert", "clip"])
def test_margin_shap_equals_pos_minus_neg_shap(model, config, word_values, sentence_results):
    tolerance = float(config["tolerances"]["margin_linearity"])
    for matrix in word_matrix(word_values, model).values():
        assert np.max(np.abs(matrix[:, 2] - (matrix[:, 1] - matrix[:, 0]))) <= tolerance
    for row in sentence_results:
        if row["model"] == model:
            base_error = abs(float(row["margin_base_value"]) - (float(row["pos_base_value"]) - float(row["neg_base_value"])))
            assert base_error <= float(config["tolerances"]["base_margin_linearity"])


def test_additivity_summary_reports_all_pass():
    rows = read_csv(METRICS_DIR / "additivity_summary.csv")
    assert {row["model"] for row in rows} == {"bert", "clip"}
    assert all(row["all_pass"] == "True" for row in rows)
    assert all(int(row["fail_count"]) == 0 for row in rows)
