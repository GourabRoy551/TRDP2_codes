"""Metric definitions on small hand-checked inputs."""

from __future__ import annotations

import numpy as np
import pytest

from metrics_core import (
    comprehensiveness_keep,
    rank_content_words,
    rationale_columns,
    sentence_metrics,
    sufficiency_keep,
    top_k_by_fraction,
)
from whole_word_masker import create_word_masker, render_coalition, split_word_units


def test_ranking_uses_absolute_margin_ties_by_position_and_skips_punctuation():
    is_content = [1, 1, 1, 0, 1]
    values = [0.5, -2.0, 0.5, 9.0, -0.1]
    assert rank_content_words(is_content, values) == [1, 0, 2, 4]


def test_top_k_counts_follow_source_deletion_count():
    order = list(range(15))
    cuts = top_k_by_fraction(order, [0.1, 0.2, 0.3, 0.5])
    assert [k for _, k, _ in cuts] == [2, 3, 5, 8]
    assert top_k_by_fraction([4], [0.1, 0.5]) == [(0.1, 1, [4]), (0.5, 1, [4])]


def test_keep_masks_split_content_words_and_always_keep_punctuation():
    is_content = [1, 1, 0, 1, 1, 0]
    top = [1, 4]
    comp = comprehensiveness_keep(is_content, top)
    suff = sufficiency_keep(is_content, top)
    assert comp == [True, False, True, True, False, True]
    assert suff == [False, True, True, False, True, True]
    for index, flag in enumerate(is_content):
        if flag:
            assert comp[index] != suff[index]
        else:
            assert comp[index] and suff[index]


def test_masks_render_whole_word_texts():
    text = "the film is n't bad , just dull ."
    units = split_word_units(text)
    is_content = [int(unit.is_content) for unit in units]
    masker = create_word_masker()
    top = [3, 6]  # "is n't" is one unit (index 2); index 3 = "bad", index 6 = "dull"
    assert [units[i].text for i in top] == ["bad", "dull"]
    assert render_coalition(masker, text, comprehensiveness_keep(is_content, top)) == "the film is n't , just ."
    assert render_coalition(masker, text, sufficiency_keep(is_content, top)) == "bad , dull ."


def test_sentence_metrics_for_a_negative_prediction():
    # margin -2 -> NEG prediction, d = -1, s(x) = 2
    result = sentence_metrics(-2.0, comp_margins=[-1.0, 0.5], suff_margins=[-2.0, -3.0], epsilon=1e-6)
    assert result["direction"] == -1.0
    assert result["decision_score"] == 2.0
    np.testing.assert_allclose(result["comp_by_fraction"], [1.0, 2.5])
    np.testing.assert_allclose(result["suff_by_fraction"], [0.0, -1.0])
    assert result["raw_comprehensiveness"] == pytest.approx(1.75)
    assert result["raw_sufficiency"] == pytest.approx(-0.5)
    assert result["deletion_aopc"] == pytest.approx(1.75 / (2.0 + 1e-6))


def test_sentence_metrics_for_a_positive_prediction():
    result = sentence_metrics(0.004, comp_margins=[0.001], suff_margins=[0.003], epsilon=1e-6)
    assert result["direction"] == 1.0
    assert result["raw_comprehensiveness"] == pytest.approx(0.003)
    assert result["raw_sufficiency"] == pytest.approx(0.001)


@pytest.fixture()
def tiny_lexicon(tmp_path):
    from rationale_reference import SstLexicon

    dictionary = tmp_path / "dictionary.txt"
    labels = tmp_path / "sentiment_labels.txt"
    dictionary.write_text("good|0\nGood|1\nHollywood|2\nholly|3\nthat|4\n's|5\nn't|6\ndull|7\nis n't|8\n", encoding="utf-8")
    labels.write_text("phrase ids|sentiment values\n0|0.9\n1|0.1\n2|0.7\n3|0.5\n4|0.5\n5|0.5\n6|0.3\n7|0.4\n8|0.35\n", encoding="utf-8")
    return SstLexicon(dictionary, labels)


def test_lexicon_lookup_order(tiny_lexicon):
    assert tiny_lexicon.rate("good") == (0.9, "exact")  # lowercase entry wins over "Good"
    assert tiny_lexicon.rate("hollywood") == (0.7, "case_variant_mean")
    assert tiny_lexicon.rate("is n't") == (0.35, "exact")  # a clitic unit present as a phrase
    assert tiny_lexicon.rate("that 's") == (0.5, "token_components")
    assert tiny_lexicon.rate("was n't") == (None, "not_found")  # "was" is missing
    assert tiny_lexicon.rate("unseen") == (None, "not_found")


def test_sst_neutral_interval_boundaries():
    from rationale_reference import is_sentiment_word

    assert is_sentiment_word(0.4) and is_sentiment_word(0.2) and is_sentiment_word(0.60001)
    assert not is_sentiment_word(0.40001) and not is_sentiment_word(0.6) and not is_sentiment_word(None)


def test_sentence_reference_uses_content_words_only(tiny_lexicon):
    from rationale_reference import sentence_reference

    rows = sentence_reference(tiny_lexicon, ["good", ",", "holly", "dull"], [1, 0, 1, 1])
    assert [row["word_index"] for row in rows] == [0, 2, 3]
    assert [row["is_reference"] for row in rows] == [1, 0, 1]


def test_precision_recall_f1():
    from rationale_reference import precision_recall_f1

    assert precision_recall_f1([1, 2], [2, 5, 7]) == pytest.approx((0.5, 1 / 3, 0.4))
    assert precision_recall_f1([1], [2]) == (0.0, 0.0, 0.0)
    assert precision_recall_f1([2, 5], [2, 5]) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        precision_recall_f1([1], [])


def test_rationale_column_search(dataset_rows, own_config):
    keywords = own_config["rationale"]["reference_columns_searched"]
    assert rationale_columns(list(dataset_rows[0].keys()), keywords) == []
    assert rationale_columns(["sentence_id", "Human_Rationale_Mask"], keywords) == ["Human_Rationale_Mask"]
