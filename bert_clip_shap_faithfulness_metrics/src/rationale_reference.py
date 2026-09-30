"""Word-level sentiment reference from the Stanford Sentiment Treebank (proxy for human rationales).

SST annotators rated every phrase of every parse tree, including every single word, on a 0-1
positivity scale, with each phrase shown out of context. The SST README cut-offs are
[0, 0.2] very negative, (0.2, 0.4] negative, (0.4, 0.6] neutral, (0.6, 0.8] positive,
(0.8, 1.0] very positive. A content word of an evaluation sentence is a *reference sentiment
word* when its rating lies outside the neutral interval, i.e. rating <= 0.4 or rating > 0.6.

These are context-free human word ratings, not rationales collected for each sentence's label.
Precision / recall / F1 against them therefore measure agreement with human lexical sentiment,
a proxy for token-level rationale F1.

Lookup of one whole-word unit (evaluation texts are lowercased PTB tokens):
  1. the phrase itself, when SST has it in lowercase                      -> "exact"
  2. otherwise its case variants (e.g. "Hollywood"), mean rating          -> "case_variant_mean"
  3. otherwise (clitic-merged units such as "that 's") each space-separated token by 1-2;
     rating = the token rating furthest from 0.5                          -> "token_components"
  Units still not found get no rating and are never reference words.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

NEUTRAL_LOW, NEUTRAL_HIGH = 0.4, 0.6


class SstLexicon:
    def __init__(self, dictionary_path: Path, labels_path: Path) -> None:
        ratings: dict[int, float] = {}
        for line in Path(labels_path).read_text(encoding="utf-8").splitlines()[1:]:
            phrase_id, value = line.split("|")
            ratings[int(phrase_id)] = float(value)
        self.exact: dict[str, float] = {}
        variants: dict[str, list[float]] = defaultdict(list)
        for line in Path(dictionary_path).read_text(encoding="utf-8").splitlines():
            phrase, phrase_id = line.rsplit("|", 1)
            if phrase == phrase.lower():
                self.exact[phrase] = ratings[int(phrase_id)]
            else:
                variants[phrase.lower()].append(ratings[int(phrase_id)])
        self.variant_mean = {key: sum(values) / len(values) for key, values in variants.items()}

    def _single(self, text: str) -> tuple[float | None, str]:
        key = text.lower()
        if key in self.exact:
            return self.exact[key], "exact"
        if key in self.variant_mean:
            return self.variant_mean[key], "case_variant_mean"
        return None, "not_found"

    def rate(self, unit_text: str) -> tuple[float | None, str]:
        rating, source = self._single(unit_text)
        tokens = unit_text.split()
        if rating is not None or len(tokens) < 2:
            return rating, source
        parts = [self._single(token)[0] for token in tokens]
        if any(part is None for part in parts):
            return None, "not_found"
        return max(parts, key=lambda value: abs(value - 0.5)), "token_components"


def is_sentiment_word(rating: float | None) -> bool:
    return rating is not None and (rating <= NEUTRAL_LOW or rating > NEUTRAL_HIGH)


def sentence_reference(lexicon: SstLexicon, words: Sequence[str], is_content: Sequence[int]) -> list[dict[str, Any]]:
    """One row per content word: SST rating, lookup source and reference membership."""
    rows = []
    for index, (word, flag) in enumerate(zip(words, is_content, strict=True)):
        if not flag:
            continue
        rating, source = lexicon.rate(word)
        rows.append({"word_index": index, "word": word, "sst_rating": rating, "lookup_source": source,
                     "is_reference": int(is_sentiment_word(rating))})
    return rows


def precision_recall_f1(selected: Sequence[int], reference: Sequence[int]) -> tuple[float, float, float]:
    """Token-level P / R / F1 of a selected word set against a non-empty reference set."""
    chosen, gold = set(selected), set(reference)
    if not chosen or not gold:
        raise ValueError("Precision/recall need non-empty selected and reference sets.")
    hits = len(chosen & gold)
    precision, recall = hits / len(chosen), hits / len(gold)
    f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
