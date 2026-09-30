"""External whole-word units, the SHAP text masker built on them, and masking audits.

One segmentation is shared by BERT and CLIP. SHAP masks whole units; a masked unit
is deleted (mask token ``""``) and the remaining string is re-tokenized by the
model's own WordPiece or BPE tokenizer. Model sub-tokens therefore never become
explanation features, and no word can be removed only partially.

Unit rules (applied to the PTB-tokenized SST-2 text):
* words, including internal apostrophes and hyphenated compounds (``heavy-handed``,
  ``fly-on-the-wall``, ``k-19``);
* PTB-split clitics (``n't 's 're 've 'll 'd 'm``) are merged with the preceding
  word, so the contraction ``is n't`` or possessive ``son 's`` is one unit;
* other apostrophe-initial tokens (``'70s``, ``'em``) and decimal numbers are single units;
* common abbreviations keep their period (``mr.``, ``vs.``);
* runs of punctuation (``.``, ``--``, ``...``, ``-lrb-``) are separate NON-content units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


MASKER_VERSION = "whole_word_v3_clitic_merge_no_boundary_placeholders"
_HYPHEN = "-‐‑–"
TOKEN_PATTERN = re.compile(
    rf"""
    (?P<bracket>-(?:lrb|rrb|lsb|rsb|lcb|rcb)-)
  | (?P<abbreviation>\b(?:mr|mrs|ms|dr|vs|jr|sr)\.)
  | (?P<number>\d+(?:[.,:]\d+)+)
  | (?P<apostrophe_word>['’]\w+(?:[{_HYPHEN}]\w+)*)
  | (?P<word>\w+(?:['’]\w+)*(?:[{_HYPHEN}]\w+(?:['’]\w+)*)*)
  | (?P<punctuation>[^\w\s]+)
    """,
    flags=re.UNICODE | re.VERBOSE | re.IGNORECASE,
)
CLITICS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}


@dataclass(frozen=True)
class WordUnit:
    index: int
    text: str
    start: int
    end: int
    is_content: bool
    kind: str


def _raw_units(text: str) -> list[tuple[str, int, int, str]]:
    return [(match.group(0), match.start(), match.end(), match.lastgroup or "") for match in TOKEN_PATTERN.finditer(text)]


def split_word_units(text: str) -> list[WordUnit]:
    """Split a sentence into the shared whole-word explanation units."""
    merged: list[list[Any]] = []
    for value, start, end, kind in _raw_units(text):
        is_clitic = value.replace("’", "'").lower() in CLITICS
        if (
            is_clitic
            and merged
            and merged[-1][3] in {"word", "abbreviation", "number", "clitic_merged"}
            and text[merged[-1][2] : start].strip() == ""
        ):
            merged[-1][2] = end
            merged[-1][3] = "clitic_merged"
            continue
        merged.append([value, start, end, kind])
    units = []
    for index, (_, start, end, kind) in enumerate(merged):
        value = text[start:end]
        content = kind not in {"punctuation", "bracket"} and bool(re.search(r"[^\W_]", value))
        units.append(WordUnit(index, value, start, end, content, kind))
    return units


class WordCoalitionTokenizer:
    """Minimal tokenizer API consumed by ``shap.maskers.Text``: one id per whole-word unit.

    No boundary placeholders are emitted. Earlier TRDP maskers added two empty
    placeholders; under a finite Partition budget SHAP splits unresolved cluster
    credit evenly across leaves, so those invisible placeholders absorbed part of
    the attribution and word-level additivity failed.
    """

    mask_token = None
    sep_token = None
    sep_token_id = None

    def __call__(self, text: str, return_offsets_mapping: bool = False, **_: Any) -> dict[str, Any]:
        units = split_word_units(str(text))
        result: dict[str, Any] = {"input_ids": list(range(len(units)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(unit.start, unit.end) for unit in units]
        return result


def create_word_masker() -> Any:
    """SHAP text masker that deletes masked whole-word units (identical for both models)."""
    import shap

    class WholeWordDeletionMasker(shap.maskers.Text):
        def __init__(self) -> None:
            # Text() derives a mask-token id by tokenizing the mask string; "" has no
            # units, so a one-unit placeholder is used for that lookup only.
            super().__init__(WordCoalitionTokenizer(), mask_token="...", collapse_mask_token=True)
            self.mask_token = ""  # masked units are deleted, never replaced

    return WholeWordDeletionMasker()


def render_coalition(masker: Any, text: str, keep: Sequence[bool]) -> str:
    """Return the perturbed sentence for a unit-level coalition using the SHAP masker itself."""
    mask = np.asarray(keep, dtype=bool)
    return str(np.asarray(masker(mask, text)[0], dtype=object).reshape(-1)[0])


def model_tokens(tokenizer: Any, text: str) -> list[str]:
    """Model sub-tokens without special tokens and without truncation."""
    if not text:
        return []
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    return [str(token) for token in tokenizer.convert_ids_to_tokens(ids)]


def map_tokens_to_units(tokenizer: Any, text: str, units: Sequence[WordUnit]) -> dict[str, Any]:
    """Assign every internal model token to exactly one whole-word unit via character offsets."""
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    per_unit: list[list[str]] = [[] for _ in units]
    unassigned, spanning = 0, 0
    for token, (start, end) in zip(tokens, encoded["offset_mapping"], strict=True):
        overlaps = [unit.index for unit in units if max(start, unit.start) < min(end, unit.end)]
        if not overlaps:
            unassigned += 1
            continue
        spanning += int(len(overlaps) > 1)
        per_unit[overlaps[0]].append(str(token))
    return {
        "tokens_per_unit": per_unit,
        "token_count": len(tokens),
        "unassigned_tokens": unassigned,
        "tokens_spanning_multiple_units": spanning,
        "multi_piece_units": sum(len(pieces) > 1 for pieces in per_unit),
    }


def audit_sentence_masking(
    masker: Any, tokenizers: dict[str, Any], text: str, rng: np.random.Generator, random_coalitions: int
) -> dict[str, Any]:
    """Empirically verify that deletion acts on complete words only.

    For every tested coalition and tokenizer, the tokens of the perturbed sentence
    must equal the concatenated tokens of exactly the kept units. Any partial
    removal of a word (or a changed segmentation of a kept word) breaks equality.
    """
    units = split_word_units(text)
    n = len(units)
    coalitions = [np.ones(n, dtype=bool), np.zeros(n, dtype=bool)]
    for index in range(n):
        leave_one_out = np.ones(n, dtype=bool)
        leave_one_out[index] = False
        coalitions.append(leave_one_out)
    coalitions.extend(rng.integers(0, 2, size=n).astype(bool) for _ in range(random_coalitions))
    unit_tokens = {name: [model_tokens(tokenizer, unit.text) for unit in units] for name, tokenizer in tokenizers.items()}
    violations = {name: 0 for name in tokenizers}
    full_text_preserved = render_coalition(masker, text, coalitions[0]) == text
    empty_is_blank = render_coalition(masker, text, coalitions[1]) == ""
    for keep in coalitions:
        perturbed = render_coalition(masker, text, keep)
        for name, tokenizer in tokenizers.items():
            expected = [piece for index, pieces in enumerate(unit_tokens[name]) if keep[index] for piece in pieces]
            violations[name] += int(model_tokens(tokenizer, perturbed) != expected)
    return {
        "units": n,
        "content_units": sum(unit.is_content for unit in units),
        "coalitions_checked": len(coalitions),
        "full_coalition_reproduces_text": full_text_preserved,
        "empty_coalition_is_empty_string": empty_is_blank,
        **{f"{name}_partial_word_violations": count for name, count in violations.items()},
    }
