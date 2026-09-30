# Part 1 — Baseline Parity and Discriminative Margin

## Purpose

Reproduce the completed 20-sentence zero-shot CLIP baseline in the new folder and add a sentiment-discriminative `POS−NEG` output without changing the old experiment.

## Planned work

1. Create the independent configuration and source structure.
2. Reuse the approved 20 sentence records through a verified copy and manifest.
3. Reproduce the same CLIP model, prompts, tokenizer, prototypes and `[NEG, POS]` scores.
4. Add the margin score:

   ```text
   s_margin(x) = s_POS(x) - s_NEG(x)
   ```

5. Explain `[NEG, POS, MARGIN]` with Partition SHAP.
6. Calculate a derived margin explanation:

   ```text
   derived margin SHAP for token i
       = POS SHAP for token i - NEG SHAP for token i

   phi[i, margin] = phi[i, POS] - phi[i, NEG]
   ```

7. Compare direct and derived margin values token by token and word by word.
8. Add three-column token, word and sentence-summary plots.

## Required tests

- baseline score parity with `clip_text_shap`;
- tokenizer and SHAP-feature alignment;
- NEG, POS and margin additivity;
- direct-versus-derived margin agreement;
- BPE-to-word conservation;
- PNG/PDF output pairing.

## Deliverables

- independent runnable code;
- NEG/POS/margin CSV files;
- complete numerical tables for all 20 sentences;
- five combined NEG/POS/margin overview plots selected by the documented
  presentation-example rule;
- compact aggregate plots for the full 20-sentence result;
- parity and additivity report;
- short Part 1 conclusion.

## Stop condition

Do not begin prompt modification until the Part 1 results have been reviewed and approved.

