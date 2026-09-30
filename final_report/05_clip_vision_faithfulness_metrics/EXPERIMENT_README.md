# CLIP Vision: deletion AUC, insertion AUC, AOPC, pointing game

Extends `../clip_vision_shap` with four explainability metrics for its stored patch-level
Partition SHAP values: CLIP ViT-B/32 image encoder, images I1–I10, run `clip_vision_i1_i10`.

Nothing in the source experiment is changed. The SHAP values, the CLIP model, the images, the
224×224 input preparation, the 7×7 patch construction, the blur baseline and the existing
results are all read as they are. This folder only adds code and new output files. The source
folder and the image folder are hashed before and after every run. No SHAP value is
recomputed, and CLIP is neither retrained nor modified (its parameter checksum is compared
before and after scoring).

## Result

| Metric | CLIP Vision |
|---|---|
| Deletion AUC ↓ | 0.7109 |
| Insertion AUC ↑ | 0.8492 |
| AOPC ↑ | 0.1995 |
| Pointing/Localization Accuracy ↑ | 0.9000 (9 of 10 images) |

Means over the 10 images. No confidence intervals or statistical tests. Pointing accuracy uses
author-annotated boxes and the standard 15-pixel tolerance; without tolerance it is 0.7000 (7 of 10).

| Image | DAUC ↓ | IAUC ↑ | AOPC ↑ | Pointing target | Hit (15 px) | Hit (0 px) |
|---|---:|---:|---:|---|:---:|:---:|
| I1 | 0.7418 | 0.8776 | 0.1820 | horse | 0 | 0 |
| I2 | 0.7341 | 0.8752 | 0.1341 | people | 1 | 0 |
| I3 | 0.7677 | 0.8902 | 0.1562 | person | 1 | 1 |
| I4 | 0.7679 | 0.9331 | 0.1790 | kayak with paddler | 1 | 1 |
| I5 | 0.6560 | 0.8820 | 0.3358 | dogs | 1 | 1 |
| I6 | 0.6366 | 0.7640 | 0.2075 | painted puppet-theatre cart | 1 | 1 |
| I7 | 0.6863 | 0.8063 | 0.1928 | people (vendors) | 1 | 1 |
| I8 | 0.6783 | 0.8018 | 0.2179 | person | 1 | 1 |
| I9 | 0.7645 | 0.8297 | 0.1559 | people (couple) | 1 | 0 |
| I10 | 0.6761 | 0.8318 | 0.2333 | child | 1 | 1 |

## Definitions

- **Target score.** f = cosine similarity between the CLIP image embedding of the perturbed
  224×224 input and that of the original input. This is exactly the output the source
  experiment explained. The original input scores 1. CLIP's vision encoder has no class
  output, and the source experiment used no text, prompts or labels, so this image-only
  score is the target.
- **Ranking.** The 49 patches (7×7 grid of 32×32 px, one per ViT-B/32 token) are ordered by
  |SHAP| descending, ties by patch index. All 490 stored patch values are ≥ 0, so this equals
  the signed ranking the source faithfulness evaluation used.
- **Masking and baseline.** These reuse the existing method. A removed patch is replaced by the
  128×128 box-blurred copy of the input, which is the background SHAP's own masker used
  (source `box_blur`, `patch_mask` and `compose`). The fully blurred image is therefore the
  shared baseline for deletion and insertion. It scores 0.296–0.564 (mean 0.452), not 0.

| Metric | Definition |
|---|---|
| Deletion AUC | Blur the top-ranked patches one at a time, k = 0…49. DAUC is the trapezoidal area under f against the fraction of patches removed (0–1). The curve runs from 1 to f(fully blurred). |
| Insertion AUC | Start from the fully blurred image and restore the top-ranked patches one at a time. IAUC is the area under f against the fraction restored. The curve runs from f(fully blurred) to 1. |
| AOPC | mean over p ∈ {10, 20, 30, 50 %} of f(original) − f(top-k(p) patches blurred), with k(p) = max(1, ⌈p·49⌉) = 5, 10, 15, 25 patches. These are the same cut-offs as the source faithfulness evaluation and the BERT / CLIP-text AOPC. |
| Pointing game | Hit when the centre of the highest-\|SHAP\| patch lies inside an approved box of the image's target object. The box is clipped to the CLIP crop and enlarged by 15 px. Accuracy = hits / images. |

AOPC equals the mean of the source experiment's stored comprehensiveness drops
(0.1374, 0.1697, 0.2126, 0.2782 in its `run_report.md`), because the same patches are removed
from the same inputs.

### Pointing game: boxes and rule

The image set has no dataset boxes or masks. `data/images.csv` holds only `image_id, image_path,
width, height, sha256`, and `clip_vision_images/` holds only the ten JPEGs. One target-object box
per image (two for the two horses in I1) was therefore annotated for this experiment, in
`annotations/pointing_game_boxes.csv`:

- The boxes were drafted from the images alone, without looking at the SHAP maps.
- They were reviewed and approved by the author on 2026-09-30, unchanged from the draft.
- No image was excluded.

The report must call them *author-annotated*, not benchmark ground truth.

**The rule:**
- **Point:** the centre of the highest-|SHAP| patch (the first patch of the ranking), mapped to
  original-image pixels.
- **Box:** clipped to the CLIP centre crop, because CLIP cannot see outside it.
- **Hit:** the point lies inside the box enlarged by 15 px on every side. This is the tolerance
  of the standard pointing game.

**Points to keep in mind when reading the result:**
- **The two tolerance-only hits are near misses.** I2 is 10 px below the people box and I9 is
  12 px right of the couple. That's why the strict (0 px) accuracy is 0.70.
- **The one clear miss is I1.** The top patch lies on the burning stump, not the horse.
- **Group boxes are easier targets.** I2 (people) and I7 (vendors) use one large box around a
  group, which is easier to hit than a single small object.
- **This is not class-specific localization.** The explained score is image-only (similarity
  to the image's own embedding). The metric shows whether the most important patch lies on the
  main object, not whether CLIP localizes a named class.
- **Ten images are a qualitative check, not a benchmark.**

## Checks (all must pass before any metric file is written)

| Check | Result |
|---|---|
| Stored patch maps: `shap_arrays.npz` = `patch_shap_values.csv` | exact |
| Rebuilt ranking = stored `abs_rank`, and = patches selected by the source faithfulness evaluation | pass (10/10) |
| Original and fully blurred scores vs stored `image_summary.csv` | max diff 0.0 |
| Deletion / insertion end points vs original and fully blurred scores | ≤ 3.6e-7 (tol 1e-5) |
| Deletion at 10/20/30/50 % vs stored comprehensiveness scores; insertion vs stored sufficiency scores | ≤ 3.6e-7 (tol 1e-5) |
| CLIP in eval mode; parameter checksum unchanged | pass |
| Pointing boxes: every row reviewed (approved/excluded), every image covered, one target per image | pass |
| Source experiment (129 files) and image folder (10 files) unchanged | pass |

A second run reproduces the per-image values, the curve points and the table byte for byte.
The run makes 1,030 model evaluations. Per image, that is 50 deletion + 50 insertion points,
plus the reference embedding, the original score and the fully blurred score.

## Running

Conda environment `rfem` (Python 3.11, torch 2.14 + CUDA 12.6, transformers 5.4; the same
versions as the source run). The model loads offline from the local Hugging Face cache.
The run takes about 10–15 s on an RTX 4060 Laptop GPU.

```bat
run_vision_metrics.bat
```

This runs `src\run_vision_metrics.py`, then `pytest tests` (13 tests).

## Layout

| Path | Content |
|---|---|
| `config.json` | source run, ranking, masking, curve steps, AOPC fractions, tolerances, pointing-game rule |
| `annotations/pointing_game_boxes.csv` | approved author-annotated target boxes (original-image pixels) |
| `annotations/review/` | review images of the boxes (no SHAP shown); `src/render_annotation_review.py` redraws them |
| `src/source_link.py` | read-only import path to `../clip_vision_shap/src` (bytecode writing disabled) |
| `src/vision_metrics.py` | ranking, deletion / insertion sequences, AUC, AOPC |
| `src/curve_plots.py` | deletion and insertion curve figures |
| `src/run_vision_metrics.py` | gates, scoring, outputs |
| `tests/` | metric definitions on synthetic inputs, output consistency (AUC and AOPC recomputed from the curve points) |
| `outputs/summary_table.{md,csv,tex}` | Metric \| CLIP Vision table |
| `outputs/per_image_metrics.csv` | per image: DAUC, IAUC, AOPC and its four drops, original / blurred score, highest-SHAP patch location, pointing-game target and hit (15 px and 0 px) |
| `outputs/plots/pointing_game.{png,pdf}` | each image with its box, the CLIP crop and the highest-SHAP patch (hit / miss) |
| `outputs/curve_points.csv` | 500 rows: every deletion / insertion curve point with the patch changed at that step |
| `outputs/plots/deletion_insertion_curves_mean.{png,pdf}` | mean deletion and insertion curves (individual images in grey) |
| `outputs/plots/deletion_insertion_curves_per_image.{png,pdf}` | small multiples, one panel per image |
| `outputs/checks.json` | every gate, errors, annotation search, versions |
| `outputs/run.log` | run log |
