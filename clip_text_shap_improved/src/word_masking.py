"""Whole-word SHAP tokenization and CLIP-BPE-to-word mapping diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


# A lexical unit may contain apostrophe contractions and repeated hyphen compounds.
# Remaining non-whitespace characters (normally punctuation) become their own units.
WORD_OR_PUNCT = re.compile(
    r"[\w]+(?:['’][\w]+)*(?:[-‐‑–][\w]+(?:['’][\w]+)*)*|[^\w\s]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class WordUnit:
    index: int
    text: str
    start: int
    end: int
    is_content: bool


def split_word_units(text: str) -> list[WordUnit]:
    units = []
    for index, match in enumerate(WORD_OR_PUNCT.finditer(text)):
        value = match.group(0)
        units.append(
            WordUnit(
                index=index,
                text=value,
                start=match.start(),
                end=match.end(),
                is_content=bool(re.search(r"[\w]", value, flags=re.UNICODE)),
            )
        )
    return units


class WordCoalitionTokenizer:
    """Minimal tokenizer API used by `shap.maskers.Text`.

    Boundary placeholders are included so SHAP treats the first and final positions
    as invariant, matching the explicit special-token treatment used by CLIP.
    """

    mask_token = None
    sep_token = None
    sep_token_id = None

    def __call__(
        self, text: str, return_offsets_mapping: bool = False, **_: Any
    ) -> dict[str, Any]:
        units = split_word_units(str(text))
        offsets = [(0, 0), *[(unit.start, unit.end) for unit in units], (len(text), len(text))]
        result: dict[str, Any] = {"input_ids": [0, *range(2, len(units) + 2), 1]}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def create_word_masker() -> Any:
    import shap

    return shap.maskers.Text(
        WordCoalitionTokenizer(), mask_token="", collapse_mask_token=True
    )


def map_clip_bpe_to_words(tokenizer: Any, text: str) -> list[dict[str, Any]]:
    """Map every CLIP BPE offset to one complete word/punctuation unit."""
    units = split_word_units(text)
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=77,
        return_offsets_mapping=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    rows: list[dict[str, Any]] = []
    for bpe_index, (token, offset) in enumerate(
        zip(tokens, encoded["offset_mapping"], strict=True)
    ):
        start, end = int(offset[0]), int(offset[1])
        special = start == end
        overlaps = [
            unit
            for unit in units
            if max(start, unit.start) < min(end, unit.end)
        ]
        chosen = max(
            overlaps,
            key=lambda unit: min(end, unit.end) - max(start, unit.start),
            default=None,
        )
        rows.append(
            {
                "bpe_index": bpe_index,
                "bpe_token": str(token),
                "offset_start": start,
                "offset_end": end,
                "is_special": int(special),
                "word_index": "" if chosen is None else chosen.index,
                "word": "" if chosen is None else chosen.text,
                "word_start": "" if chosen is None else chosen.start,
                "word_end": "" if chosen is None else chosen.end,
                "overlapping_word_count": len(overlaps),
            }
        )
    return rows


def audit_mask_coalitions(
    masker: Any,
    tokenizer: Any,
    text: str,
    seed: int,
    random_masks: int = 20,
) -> dict[str, Any]:
    """Empirically verify that every BPE piece of a word shares one mask decision."""
    mapping = map_clip_bpe_to_words(tokenizer, text)
    feature_count = int(masker.shape(text)[1])
    # Word features are shifted by one because feature 0 is the invariant boundary.
    word_to_bpe: dict[int, list[int]] = {}
    for row in mapping:
        if row["word_index"] != "":
            word_to_bpe.setdefault(int(row["word_index"]), []).append(int(row["bpe_index"]))
    rng = np.random.default_rng(seed)
    violations = 0
    empty_outputs = 0
    for _ in range(random_masks):
        mask = rng.integers(0, 2, size=feature_count, dtype=np.int8).astype(bool)
        mask[0] = True
        mask[-1] = True
        masked = str(np.asarray(masker(mask, text)[0], dtype=object).reshape(-1)[0])
        empty_outputs += int(not masked.strip())
        for word_index, bpe_indices in word_to_bpe.items():
            decisions = {bool(mask[word_index + 1]) for _ in bpe_indices}
            violations += int(len(decisions) != 1)
    unassigned = sum(
        not int(row["is_special"]) and row["word_index"] == "" for row in mapping
    )
    multiword = sum(int(row["overlapping_word_count"]) > 1 for row in mapping)
    return {
        "word_feature_count": feature_count - 2,
        "non_special_bpe_count": sum(not int(row["is_special"]) for row in mapping),
        "unassigned_non_special_bpe": unassigned,
        "bpe_spanning_multiple_word_units": multiword,
        "partial_word_mask_violations": violations,
        "random_masks_checked": random_masks,
        "empty_masked_outputs": empty_outputs,
    }
