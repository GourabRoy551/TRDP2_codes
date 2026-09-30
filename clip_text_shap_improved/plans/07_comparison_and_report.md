# Part 7 — BERT Comparison and Final Report

**Status: completed.** Fine-tuned BERT was evaluated and explained on the identical
500 sentences with the same whole-word masker, Partition-SHAP method and frozen
budget. All pairwise comparison tables, five aggregate figures, five case studies,
the validation manifest and the LaTeX report were produced. The Part 7 integrity
gate passed.

## Purpose

Compare BERT, zero-shot CLIP and trained-head CLIP under clearly defined, fair conditions.

## Planned work

1. Restrict comparisons to identical evaluation sentences.
2. Confirm class order, text normalization and word-alignment rules.
3. Compare prediction performance.
4. Compare explanations using scale-independent rank, overlap, sign and faithfulness metrics.
5. Analyse agreement separately for correct and incorrect predictions.
6. Document examples where the models agree and disagree.
7. Prepare the final academic report and presentation figures.

## Required interpretation rules

- do not compare raw BERT-logit SHAP magnitude with raw CLIP-cosine SHAP magnitude;
- distinguish zero-shot CLIP from trained-head CLIP;
- do not treat descriptive softmax shares as calibrated probabilities;
- do not claim superiority from isolated examples;
- report negative and inconclusive findings;
- state dataset, prompt and masking limitations.

## Deliverables

- final comparison tables;
- annotated comparison plots in PNG and PDF;
- qualitative case studies;
- LaTeX academic report;
- presentation-ready summary;
- reproducibility and limitations sections.

## Completion condition

The project is complete only when every reported number is traceable to a saved configuration, dataset manifest and result file.

