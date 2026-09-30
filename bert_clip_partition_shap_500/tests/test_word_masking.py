"""Whole-word units: complete deletion, shared boundaries and non-content punctuation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from io_utils import CHECKPOINT_DIR, load_json, read_csv
from whole_word_masker import audit_sentence_masking, create_word_masker, render_coalition, split_word_units


def texts(units):
    return [unit.text for unit in units]


def test_contractions_and_clitics_stay_complete():
    units = split_word_units("we have n't seen the son 's room , it 's the '70s .")
    assert texts(units) == ["we", "have n't", "seen", "the", "son 's", "room", ",", "it 's", "the", "'70s", "."]


def test_hyphenated_compounds_numbers_and_abbreviations_stay_complete():
    units = split_word_units("a heavy-handed , fly-on-the-wall k-19 sequel by mr. smith costs 1.5 million")
    assert "heavy-handed" in texts(units) and "fly-on-the-wall" in texts(units) and "k-19" in texts(units)
    assert "mr." in texts(units) and "1.5" in texts(units)


def test_punctuation_is_separate_and_non_content():
    units = split_word_units("dull , tedious -- and ... forgettable -lrb- sort of -rrb- !")
    punctuation = [unit for unit in units if not unit.is_content]
    assert [unit.text for unit in punctuation] == [",", "--", "...", "-lrb-", "-rrb-", "!"]
    assert all(unit.is_content for unit in units if unit.text.isalpha())


def test_deletion_removes_whole_units_only():
    masker = create_word_masker()
    text = "the son 's room is n't heavy-handed ."
    units = split_word_units(text)
    keep = [unit.text not in {"son 's", "is n't"} for unit in units]
    assert render_coalition(masker, text, keep) == "the room heavy-handed ."
    assert render_coalition(masker, text, [True] * len(units)) == text
    assert render_coalition(masker, text, [False] * len(units)) == ""


def test_masker_features_are_exactly_the_word_units():
    masker = create_word_masker()
    text = "a sometimes tedious film ."
    assert masker.shape(text) == (1, len(split_word_units(text)))
    assert [name for name in masker.feature_names(text)[0]] == texts(split_word_units(text))


def test_no_partial_word_masking_on_all_500_sentences(rows, tokenizers):
    """Random and leave-one-out coalitions re-tokenize to exactly the kept units' pieces."""
    masker = create_word_masker()
    rng = np.random.default_rng(7)
    totals = defaultdict(int)
    for row in rows:
        audit = audit_sentence_masking(masker, tokenizers, row["text"], rng, random_coalitions=5)
        totals["bert"] += audit["bert_partial_word_violations"]
        totals["clip"] += audit["clip_partial_word_violations"]
        assert audit["full_coalition_reproduces_text"] and audit["empty_coalition_is_empty_string"]
    assert dict(totals) == {"bert": 0, "clip": 0}


def test_saved_masking_audit_and_smoke_gate():
    audit = read_csv(CHECKPOINT_DIR / "masking_audit.csv")
    assert len(audit) == 500
    assert sum(int(row["bert_partial_word_violations"]) for row in audit) == 0
    assert sum(int(row["clip_partial_word_violations"]) for row in audit) == 0
    smoke = load_json(CHECKPOINT_DIR / "smoke_test.json")
    assert smoke["gate_passed"] is True and smoke["masking_audit"]["passed"] is True


def test_bert_and_clip_share_identical_word_boundaries(rows, word_values):
    per_model = defaultdict(lambda: defaultdict(list))
    for row in word_values:
        per_model[row["model"]][row["sentence_id"]].append((int(row["word_index"]), row["word"], row["is_content"]))
    for row in rows:
        expected = [(unit.index, unit.text, str(int(unit.is_content))) for unit in split_word_units(row["text"])]
        assert sorted(per_model["bert"][row["sentence_id"]]) == expected
        assert sorted(per_model["clip"][row["sentence_id"]]) == expected


@pytest.mark.parametrize("model", ["bert", "clip"])
def test_every_internal_token_belongs_to_one_word(model, model_scores):
    selected = [row for row in model_scores if row["model"] == model]
    assert sum(int(row["unassigned_tokens"]) for row in selected) == 0
    assert sum(int(row["tokens_spanning_multiple_units"]) for row in selected) == 0
