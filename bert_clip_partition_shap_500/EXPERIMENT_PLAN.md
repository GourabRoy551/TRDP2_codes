# Experiment plan: BERT vs CLIP Text Encoder, whole-word Partition SHAP (500 sentences)

## Question

On one fixed, balanced set of 500 SST-2 sentences, how do a fine-tuned sentiment
classifier (BERT, SST-2) and a frozen zero-shot CLIP text encoder differ in
(1) prediction quality and (2) the whole-word evidence that Partition SHAP
attributes to their NEG, POS and POS−NEG outputs?

## Fixed design decisions

| Item | Decision |
|---|---|
| Dataset | `data/evaluation_500.csv`, byte-identical copy of `clip_text_shap_improved/data/final_evaluation_500.csv` (250 NEG / 250 POS, SHA-256 `61163db6…25ac`). Never resampled, edited, reordered or relabelled. |
| Model 1 | `textattack/bert-base-uncased-SST-2`, frozen, eval mode. Outputs: raw NEG logit, raw POS logit. |
| Model 2 | `openai/clip-vit-base-patch32` text encoder, frozen, zero-shot. Outputs: cosine similarity to the NEG and POS prototypes of the frozen `balanced_diverse_ensemble` prompt family (8 + 8 prompts, selected in `clip_text_shap_improved` Part 2 without the final set). No classification head. |
| Output order | `[NEG, POS, POS−NEG]`, margin = POS − NEG, prediction POS iff margin > 0. |
| Training | None. No new model training or fine-tuning is performed in this experiment. |
| Explanation features | Shared whole-word units (contractions/clitics, hyphenated compounds, numbers and abbreviations kept whole; punctuation runs separate and non-content). |
| Masking | Masked units are deleted; the remaining string is re-tokenized by the model's own tokenizer (WordPiece / BPE). Same deletion strategy for both models. |
| Algorithm | `shap.PartitionExplainer` over SHAP's hierarchical text partition of the word units; `max_evals = 500`; all three outputs explained jointly. |
| Seeds | 42 for Python, NumPy, PyTorch; deterministic cuDNN/cuBLAS; TF32 disabled. |
| Device | CUDA when available, otherwise CPU. |

## Stages and gates

| Part | Stage (`--stage`) | Gate that must pass before the next part |
|---|---|---|
| 1 | `prepare` | 500 rows, 250/250, unique IDs, no duplicate normalized text, no missing text, NEG/POS only, SHA-256 equals source file and source manifest |
| 2 | `score` | BERT label mapping verified on 10 independent sentences (and fails when swapped); CLIP prompts equal the frozen family; both models frozen; `[n,3]` outputs, column 2 = column 1 − column 0 exactly, prediction rule, batched = individual (≤ 2e-5 BERT, ≤ 1e-5 CLIP) on 20 sentences; same sentence order; case studies selected here, before any SHAP value exists |
| 3 | `smoke` | Masking audit on all 500 sentences × (full, empty, leave-one-out, 10 random coalitions) × both tokenizers: zero partial-word removals; 5-sentence SHAP per model within additivity and margin-linearity tolerances; repeat run reproduces values |
| 4 | `shap --model bert` | Checkpoint after every sentence (JSONL + fsync), progress/CSV snapshot every 25; resume validates each stored record |
| 5 | `shap --model clip` | as Part 4 |
| 6 | `metrics` | 1,000 sentence records, all word rows, 500 pairwise rows, additivity + linearity pass, 1,000 bootstrap iterations |
| 7 | `plots`, `report`, tests | 5 aggregate + 5 case-study PNG/PDF pairs; full pytest suite passes |

## Numerical tolerances

| Check | Tolerance | Reason |
|---|---|---|
| Margin linearity φ_margin − (φ_POS − φ_NEG) | 1e-6 | margin formed in float64 from float32 class scores, so only float64 round-off remains |
| Base linearity | 1e-6 | as above |
| Additivity, CLIP | 1e-5 | cosine similarities O(0.2); float32 batch-shape noise ~1e-7 |
| Additivity, BERT | 2e-5 | logits O(4); float32 batch-shape noise ~1e-6 |

Scores and SHAP values are never modified to meet a tolerance.

## Primary metrics

1. **Macro-F1** (plus accuracy, balanced accuracy, per-class P/R/F1, confusion counts).
2. **SHAP additivity residual** |score_c − (base_c + Σ_i φ_c(i))| per output (mean, median, p95, max, pass/fail).
3. **Normalized deletion AOPC**: rank content words by |φ_margin|; delete top 10/20/30/50 %;
   drop in decision score d·margin (d = +1 for a POS prediction, −1 for NEG),
   normalized by |original decision score| + 1e-6; mean over the four fractions. Baseline:
   five deterministic random orders per sentence (identical for both models). Prediction-flip rates for both.
4. **Spearman ρ** between BERT and CLIP |φ_margin| over identical content words (defined for ≥ 3 content words and non-constant rankings).
5. **Top-5 overlap and Jaccard** of the |φ_margin| top-5 content words, plus sign agreement (supporting).

Confidence intervals: 1,000-iteration percentile bootstrap, NEG and POS resampled
separately in each replicate (balanced design preserved), identical resamples for
all metrics (paired), seed 42.

## Interpretation rules

Raw BERT logits and CLIP similarities are never compared by magnitude; CLIP scores are
not treated as probabilities; SHAP is not treated as proof of causal importance; the
five case studies are illustrations only; no numerical claim is made against RFEM,
which was not rerun on this dataset.
