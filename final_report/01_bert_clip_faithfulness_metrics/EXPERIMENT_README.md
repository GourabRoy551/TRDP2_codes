# BERT vs CLIP-text — comprehensiveness, sufficiency, deletion AOPC, rationale F1

Extends `../bert_clip_partition_shap_500` with four explanation metrics, computed identically
for BERT (`textattack/bert-base-uncased-SST-2`) and the frozen zero-shot CLIP ViT-B/32 text
encoder on the same 500 SST-2 sentences (250 NEG / 250 POS).

Nothing in the source experiment is changed. Its SHAP values, models, dataset, masker and
results are read from there as-is; this folder only adds code and new output files. The source
folder is hashed before and after each run (`source_experiment_unchanged` in `outputs/checks.json`).
No SHAP value is recomputed and no model is trained.

## Result

| Metric | BERT | CLIP |
|---|---|---|
| Comprehensiveness (higher = better) | 0.7715 | 0.7049 |
| Sufficiency (lower = better) | 0.2231 | 0.3307 |
| Deletion AOPC (higher = better) | 0.8902 | -0.3681 |
| Rationale Precision, SST lexical proxy (higher = better) | 0.4658 | 0.3277 |
| Rationale Recall, SST lexical proxy (higher = better) | 0.6778 | 0.4886 |
| Rationale F1, SST lexical proxy (higher = better) | 0.4935 | 0.3497 |

The first three rows are means over the 500 sentences. The rationale rows are means over the
468 sentences that contain at least one reference sentiment word; these are the same sentences
for both models. No confidence intervals or statistical tests.

## Definitions

Explanation target: the POS−NEG margin. For each model and sentence x:

- d = +1 for a POS prediction (margin > 0), −1 for NEG; decision score s(x) = d · margin(x).
- Ranking: content words by |φ_margin| descending, ties by word position. This is the source
  experiment's primary `shap_abs` deletion ranking. Punctuation units are never ranked or removed.
- Fractions f ∈ {10%, 20%, 30%, 50%}; k_f = max(1, ⌈f · n_content⌉) (source `deletion_count`).
- Masking: the source whole-word deletion masker (masked words are deleted, the rest is
  re-tokenized by the model's own tokenizer).

| Metric | Per sentence | Scale |
|---|---|---|
| Comprehensiveness | mean_f [ s(x) − s(x without its top-k_f words) ] | ÷ S_model |
| Sufficiency | mean_f [ s(x) − s(only the top-k_f words + all punctuation) ] | ÷ S_model |
| Deletion AOPC | mean_f [ s(x) − s(x without its top-k_f words) ] / (\|s(x)\| + 1e-6) | per sentence (source definition) |
| Rationale P / R / F1 | mean_f of P = \|T_f ∩ R\| / k_f, R = \|T_f ∩ R\| / \|R\|, F1 = 2PR / (P + R) | none (unitless) |

T_f = the top-k_f SHAP-ranked content words (the same words removed for comprehensiveness and
kept for sufficiency). R = the sentence's reference sentiment words (below). F1 = 0 when T_f ∩ R is empty.

S_model = mean |s(x)| over the 500 sentences: **BERT 6.0591** (logit units), **CLIP 0.0046170**
(cosine-similarity units). Dividing by one constant per model puts the BERT logit margin and the
CLIP cosine margin on a common unitless scale. It also avoids dividing by near-zero margins
for single sentences: 73 CLIP sentences (0 BERT) have |margin| < 0.001.

Comprehensiveness and Deletion AOPC use the same deletion texts and scores. They differ only in
normalization: per model versus per sentence. Deletion AOPC is the source experiment's
normalized AOPC and reproduces its stored per-sentence values exactly. The source report quotes
the same means: 0.890 BERT, −0.368 CLIP.

### Rationale reference: SST human word-level sentiment (a lexical proxy)

`evaluation_500.csv` has only sentence-level labels, and SST-2 has no human rationale
annotations. The reference is therefore built from the Stanford Sentiment Treebank's own human
annotations. SST annotators rated every phrase of every parse tree, including every single word,
on a 0–1 positivity scale. Each phrase was rated **out of context**.

- **Files:** `dictionary.txt` and `sentiment_labels.txt` from the original SST release, whose
  paths are in `config.json`. Its `dev.tsv` matches the SHA-256 in the source dataset manifest,
  so it is the same SST release as the 500 sentences.
- **Reference sentiment word:** a content word whose rating lies outside SST's neutral class
  (0.4, 0.6] (README cut-offs), i.e. rating ≤ 0.4 or > 0.6. This gives 1,365 of the 8,404
  content words.
- **Lookup order for each whole-word unit:**
  - the lowercase phrase itself (8,099 words);
  - otherwise the mean over its case variants, e.g. *Hollywood* (268 words);
  - otherwise, for clitic-merged units such as *that 's*, the token rating furthest from 0.5
    (37 words).

  Every content word receives a rating.
- **Excluded sentences:** 32 sentences contain no reference word, so recall is undefined for them.
  They are excluded for both models; their IDs are in `checks.json`.
- **Model-independent:** the reference does not depend on the model. BERT and CLIP are scored
  against identical reference sets over identical sentences.

**Limitation.** These are human ratings of words in isolation, not rationales annotated for each
sentence's label. The metric measures agreement with human lexical sentiment and should be
reported as a proxy for token-level rationale F1.

- Context is lost: "tries too hard" is not marked, because each of its words is neutral alone.
- A few mildly rated words, such as *out*, *us* and *life*, fall just outside the neutral band
  and are included.

## Checks (all must pass before any metric file is written)

| Check | Result |
|---|---|
| 500 validated records per model (source record validation: text hash, settings signature, stored scores) | pass |
| Rebuilt ranking = stored `shap_abs` deleted words, all 4,000 model × sentence × fraction cases | pass |
| Rebuilt deletion texts = stored perturbed texts (4,000) | pass |
| Re-scored originals vs stored scores | max diff 0 (BERT), 0 (CLIP) |
| Re-scored deletion margins vs stored | 1.28e-5 BERT (tol 2e-5), 4.8e-7 CLIP (tol 1e-5) |
| Deletion AOPC vs stored records and `sentence_results.csv` | max diff 0 |
| BERT label mapping / CLIP frozen prompt family re-verified; both models frozen | pass |
| Word units identical for both models; SST `dev.tsv` matches the dataset manifest; every content word rated | pass |
| Source experiment folder unchanged (119 files hashed before and after) | pass |

Stored original and deletion scores are used for the metrics; re-scoring is a check only.
The 2,000 sufficiency texts per model are new model evaluations. A second run reproduces every
output byte for byte.

## Running

Conda environment `rfem` (Python 3.11, torch 2.14 + CUDA 12.6, transformers 5.4, shap 0.51).
Models load offline from the local Hugging Face cache. The run takes about 20 s on an RTX 4060 Laptop GPU.

```bat
run_metrics.bat
```

This runs `src\run_metrics.py`, then `pytest tests` (23 tests). The SST files are read from the
paths in `config.json`.

## Layout

| Path | Content |
|---|---|
| `config.json` | fractions, epsilon, ranking, scale, SST reference paths and rules; fractions and epsilon are checked against the source config |
| `src/source_link.py` | read-only import path to `../bert_clip_partition_shap_500/src` (bytecode writing disabled) |
| `src/metrics_core.py` | ranking, top-k, keep masks, per-sentence metric formulas |
| `src/rationale_reference.py` | SST word-sentiment lexicon, reference words, precision / recall / F1 |
| `src/run_metrics.py` | gates, scoring, outputs |
| `tests/` | metric formulas, SST lookup rules, output consistency (all rationale values recomputed from the audit files), sufficiency re-scoring |
| `outputs/comparison_table.{md,csv,tex}` | the Metric \| BERT \| CLIP table |
| `outputs/per_sentence_metrics.csv` | 1,000 rows (model × sentence): headline metrics and per-fraction values (scaled and raw; rationale P/R/F1 = NA for the 32 sentences without reference words) |
| `outputs/aggregate_by_fraction.csv` | means per model and fraction (supplementary) |
| `outputs/perturbation_audit.csv` | 4,000 rows: top-k words, comprehensiveness and sufficiency texts and margins |
| `outputs/rationale_reference_words.csv` | 8,404 rows: every content word with its SST rating, lookup source and reference flag |
| `outputs/checks.json` | every gate, tolerances, scale constants, input hashes, environment |
| `outputs/run.log` | run log |
