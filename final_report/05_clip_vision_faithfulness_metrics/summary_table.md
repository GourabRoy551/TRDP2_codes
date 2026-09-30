# CLIP Vision: patch-level SHAP explanation metrics

CLIP ViT-B/32 image encoder, images I1–I10, stored 7×7 patch-level Partition SHAP values.
Values are means over the images.

| Metric | CLIP Vision |
|---|---|
| Deletion AUC ↓ | 0.7109 |
| Insertion AUC ↑ | 0.8492 |
| AOPC ↑ | 0.1995 |
| Pointing/Localization Accuracy ↑ | 0.9000 |

**Target score.** f = cosine similarity between the CLIP image embedding of the perturbed 224×224 input and
that of the original input (the explained output of the source experiment; the original scores 1).
**Ranking.** The 49 patches (32×32 px) by |SHAP| descending, ties by patch index.
**Masking.** A removed patch is replaced by the 128×128 box-blurred image, the background SHAP itself used.

- **Deletion AUC**: blur the top-ranked patches one at a time (k = 0…49) and take the trapezoidal area under
  f against the fraction of patches removed (0–1). Starts at f(original) = 1, ends at f(fully blurred).
- **Insertion AUC**: start from the fully blurred image and restore the top-ranked patches one at a time; area
  under f against the fraction restored. Starts at f(fully blurred), ends at 1.
- **AOPC**: mean of f(original) − f(top-k patches blurred) at k = 10/20/30/50 % of the patches
  (5, 10, 15, 25 patches), the same cut-offs as the text AOPC.
- **Pointing/Localization Accuracy**: 9 of 10 images are hits. A hit means the centre of the highest-|SHAP|
  patch falls inside a box of the image's target object (clipped to the CLIP crop), with the standard 15-pixel
  tolerance (without tolerance: 0.7000). The images have no dataset boxes or masks;
  the boxes were annotated by the authors without looking at the SHAP maps. The explained score is image-only,
  so this measures whether the most important patch lies on the main object, not class-specific localization.

Fully blurred scores range 0.296–0.564 (mean 0.452), so both curves stay inside [0.296, 1].
