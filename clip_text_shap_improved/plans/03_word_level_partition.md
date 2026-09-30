# Part 3 — Whole-Word Hierarchical Partition SHAP

## Purpose

Avoid perturbations that remove only part of a BPE-split word while preserving the raw BPE explanation for transparency.

## Planned work

1. Map every CLIP BPE token to its original text span and word group.
2. Construct word groups containing all associated BPE indices.
3. Build a hierarchy in which BPE pieces merge inside their word before words merge into larger sentence coalitions.
4. Mask unavailable word groups as complete units.
5. Run BPE-level and whole-word Partition SHAP on the same sentences.
6. Compare additivity, ranking, faithfulness, stability and runtime.

## Required tests

- every non-special BPE token belongs to exactly one group;
- no partial word appears in a masked perturbation;
- contractions and hyphenated compounds are handled consistently;
- word contributions conserve the associated BPE contribution sums;
- special-token handling remains explicit;
- both output modes reconstruct their target scores.

## Deliverables

- BPE-to-word mapping tables;
- hierarchy diagnostics;
- side-by-side BPE and whole-word plots;
- accuracy, stability and runtime comparison;
- short Part 3 conclusion.

## Stop condition

Do not train a classifier until the masking method and hierarchy have been reviewed.

