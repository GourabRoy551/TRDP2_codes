# BERT vs CLIP-text: explanation metrics for whole-word POS−NEG margin Partition SHAP

500 SST-2 sentences (250 NEG / 250 POS). Both models use the same sentences, the same stored
SHAP values (POS−NEG margin), the same word ranking, the same whole-word deletion masker and
the same fractions (10%, 20%, 30%, 50%). Values are means over the 500 sentences; rationale metrics are means
over the 468 sentences that contain at least one reference sentiment word (identical for both models).

| Metric | BERT | CLIP |
|---|---|---|
| Comprehensiveness (higher = better) | 0.7715 | 0.7049 |
| Sufficiency (lower = better) | 0.2231 | 0.3307 |
| Deletion AOPC (higher = better) | 0.8902 | -0.3681 |
| Rationale Precision, SST lexical proxy (higher = better) | 0.4658 | 0.3277 |
| Rationale Recall, SST lexical proxy (higher = better) | 0.6778 | 0.4886 |
| Rationale F1, SST lexical proxy (higher = better) | 0.4935 | 0.3497 |

**Definitions.** d = +1 for a POS prediction, −1 for NEG; decision score s(x) = d·(POS−NEG margin).
Content words are ranked by |φ_margin| (ties by position); k = max(1, ⌈f·n_content⌉); punctuation is never removed.

- **Comprehensiveness** = mean over f of [s(x) − s(x without its top-k words)] / S_model.
- **Sufficiency** = mean over f of [s(x) − s(only the top-k words, punctuation kept)] / S_model.
- S_model = mean |s(x)| over the 500 sentences: BERT 6.05907 (logit units), CLIP 0.00461703 (cosine units).
- **Deletion AOPC** = mean over f of [s(x) − s(x without its top-k words)] / (|s(x)| + 1e-6); the source experiment's
  definition, reproduced from its stored deletion scores. Per-sentence normalization divides by each sentence's own
  margin; 73 CLIP and 0 BERT sentences have |margin| < 0.001.
- **Rationale Precision / Recall / F1 (SST lexical proxy)**: the evaluation set has no human rationale annotations, so the
  reference is built from the Stanford Sentiment Treebank's human word-level sentiment ratings (0–1 positivity, each word
  rated out of context). Reference words = content words rated outside SST's neutral interval (0.4, 0.6]
  (1365 of 8404 content words). SHAP-selected words = the same top-k words as above.
  Per sentence: P = |top-k ∩ ref| / k, R = |top-k ∩ ref| / |ref|, F1 = 2PR / (P + R), averaged over the four fractions.
  32 sentences without any reference word are excluded (recall undefined).
  These ratings are context-free lexical sentiment, not rationales for each sentence's label, so this is a proxy for
  token-level rationale F1.
