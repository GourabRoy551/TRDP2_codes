# Report materials

Copies of the results and figures for the report, collected in one place. The originals remain
in their experiment folders. `MANIFEST.csv` lists every file with its source path and SHA-256.
Each copy was verified to be byte-identical to its source. To refresh the copies after
re-running an experiment:

```bat
%USERPROFILE%\miniconda3\envs\rfem\python.exe collect_report_materials.py
```

| Folder | Files | Source |
|---|---|---|
| `01_bert_clip_faithfulness_metrics/` | 8 | `bert_clip_shap_faithfulness_metrics/outputs/` (+ its README) |
| `02_dual_class_bert_S1_S10/` | 100 (S1–S10 × 5 figures × PNG/PDF) | `bert_shap/dual_class_bert/outputs/plots/bert_dual_20_curated/` |
| `03_clip_text_shap_S1_S10/` | 100 (S1–S10 × 5 figures × PNG/PDF) | `clip_text_shap/outputs/plots/clip_text_dual_20_curated/` |
| `04_clip_vision_shap_I1_I10_patch_overlay_values/` | 20 (I1–I10 × PNG/PDF) | `clip_vision_shap/outputs/plots/clip_vision_i1_i10/` |
| `05_clip_vision_faithfulness_metrics/` | 15 | `clip_vision_shap_faithfulness_metrics/outputs/` and `annotations/` (+ its README) |
| `report2/` | LaTeX report | `main.tex`, `TRDP2_SHAP_Report.pdf`, `build_report.bat`; uses the figures of 01–05 directly |

## 01 — BERT vs CLIP-text explanation metrics (500 SST-2 sentences)

| Metric | BERT | CLIP |
|---|---|---|
| Comprehensiveness (higher = better) | 0.7715 | 0.7049 |
| Sufficiency (lower = better) | 0.2231 | 0.3307 |
| Deletion AOPC (higher = better) | 0.8902 | -0.3681 |
| Rationale Precision, SST lexical proxy | 0.4658 | 0.3277 |
| Rationale Recall, SST lexical proxy | 0.6778 | 0.4886 |
| Rationale F1, SST lexical proxy | 0.4935 | 0.3497 |

- `comparison_table.md` / `.csv` / `.tex`: the table (the `.tex` file is a ready `tabular`).
- `per_sentence_metrics.csv`: 1,000 rows (model × sentence), with per-fraction values.
- `aggregate_by_fraction.csv`: means at 10 / 20 / 30 / 50 %.
- `perturbation_audit.csv`: the exact removed/kept words and texts behind every value.
- `rationale_reference_words.csv`: every content word with its SST rating and reference flag.
- `checks.json`: all verification gates.
- `EXPERIMENT_README.md`: definitions, normalization and checks in full.

Points to state in the report:

- **Scaling:** comprehensiveness and sufficiency are divided by each model's mean |decision
  margin| (BERT 6.059 logits, CLIP 0.00462 cosine units).
- **Deletion AOPC:** this is the per-sentence-normalized AOPC already reported in the
  500-sentence experiment. It uses the same deletions as comprehensiveness; only the
  normalization differs.
- **CLIP's negative AOPC mean:** 73 CLIP sentences have |margin| < 0.001, and dividing by
  those near-zero margins drives the mean negative.
- **Rationale P/R/F1 is a proxy.** SST-2 has no human rationale annotations. The reference is
  the SST human word-level sentiment ratings: words rated outside the neutral interval (0.4, 0.6].
  Those ratings were given out of context, so this is agreement with human lexical sentiment, not
  with sentence-specific rationales.
- **Rationale averaging:** P, R and F1 use the same top-k SHAP words at 10/20/30/50 %, averaged
  over the four fractions.
- **Rationale sample:** 468 of the 500 sentences. The other 32 contain no reference word, and
  they are the same sentences for both models.

## 02 — Dual-class BERT, S1–S10

Frozen `textattack/bert-base-uncased-SST-2`, Partition SHAP for the NEG and POS logits. Each
sentence folder `S<n>/` contains five figures, each as PNG and PDF:

| File | Content |
|---|---|
| `S<n>_dual_class_word_bar` | signed word-level SHAP bars, NEG and POS panels |
| `S<n>_token_class_matrix` | WordPiece token × class SHAP matrix |
| `S<n>_word_class_matrix` | word × class SHAP matrix |
| `S<n>_summary_matrix` | sentence summary: scores, base values, SHAP sums, residuals |
| `S<n>_presentation_overview` | combined word and sentence overview |

## 03 — CLIP text (zero-shot, prompt prototypes), S1–S10

The CLIP ViT-B/32 text encoder, with Partition SHAP for the cosine similarity to the NEG and POS
prompt prototypes. The file types are the same as in 02 (`S<n>_dual_class_word_bar`, ...), so the
S1–S10 figures can be paired one-to-one with the BERT figures.

### S1–S10 sentences (identical in 02 and 03)

| ID | Gold | Sentence |
|---|---|---|
| S1 | POS | The film is a beautiful and moving portrait of human resilience. |
| S2 | NEG | This movie is an absolute waste of time and money. |
| S3 | POS | It's a bit slow at times but the performances are outstanding. |
| S4 | NEG | A dull, tedious and completely forgettable experience. |
| S5 | POS | The direction is inspired and the acting is nothing short of brilliant. |
| S6 | POS | A masterpiece of storytelling with breathtaking visuals and emotion. |
| S7 | NEG | Painfully boring and utterly devoid of any originality or charm. |
| S8 | POS | The screenplay is weak but the lead actor delivers a captivating turn. |
| S9 | NEG | A hollow and disappointing sequel that betrays everything the original stood for. |
| S10 | POS | Funny, heartfelt and endlessly entertaining from beginning to end. |

## 04 — CLIP vision, I1–I10 patch overlay with values

The CLIP ViT-B/32 image encoder, with Partition SHAP over the 7×7 input patches.
`I<n>_SHAP_patch_overlay_values` shows each patch's SHAP value on the model input (the centre
crop). The explained score is the image's similarity to its own embedding (image-only), not a
class or a caption; the images are not connected to S1–S10. Source images are in
`clip_vision_images/images/`.

## 05 — CLIP vision explanation metrics (I1–I10)

| Metric | CLIP Vision |
|---|---|
| Deletion AUC ↓ | 0.7109 |
| Insertion AUC ↑ | 0.8492 |
| AOPC ↑ | 0.1995 |
| Pointing/Localization Accuracy ↑ | 0.9000 (9 of 10; 0.7000 without tolerance) |

- `summary_table.md` / `.csv` / `.tex`: the table (the `.tex` file is a ready `tabular`).
- `per_image_metrics.csv`: DAUC, IAUC, AOPC and the pointing-game hit for each of the 10 images.
- `pointing_game.png` / `.pdf`: each image with its target box and the highest-SHAP patch
  (hit or miss).
- `pointing_game_boxes.csv` / `pointing_game_boxes_review.png`: the author-annotated target
  boxes and their review sheet.
- `curve_points.csv`: all 50 points of every deletion and insertion curve.
- `deletion_insertion_curves_mean.png` / `.pdf`: presentation figure showing the mean curves
  with individual images in grey.
- `deletion_insertion_curves_per_image.png` / `.pdf`: one panel per image.
- `checks.json` and `EXPERIMENT_README.md`: the verification gates and full definitions.

Points to state in the report:

- **Target score:** the image's cosine similarity to its own original CLIP embedding. This is
  the same image-only output the SHAP values explain; there is no class.
- **Patch ranking:** by |SHAP|. All patch values are ≥ 0, so this equals the signed ranking.
- **Removal and baseline:** a removed patch is replaced by the 128×128 box blur SHAP used, so
  the fully blurred image is the baseline. It scores about 0.45, not 0, so DAUC cannot go
  near 0.
- **Curves:** one patch per step (49 steps), with the area taken over the 0–1 fraction of
  patches.
- **AOPC:** measured at 10/20/30/50 %, the same cut-offs as in the text experiments.
- **Pointing game, boxes:** the images have no dataset boxes. One target box per image was
  annotated by the authors, drafted without seeing the SHAP maps and approved on 2026-09-30.
  Call them author-annotated, not benchmark ground truth.
- **Pointing game, rule:** a hit means the centre of the highest-SHAP patch lies inside the box,
  clipped to CLIP's crop, with the standard 15-px tolerance. That gives 9/10 hits.
- **Tolerance matters:** I2 and I9 hit only because of the tolerance, so strict accuracy is 7/10.
  I1 is a clear miss (the burning stump, not the horse).
- **Not class-specific:** the SHAP target is image-only, so the metric shows whether the most
  important patch lies on the main object, not class-specific localization.

The other clip_vision_shap figures per image (`original_and_input`, `SHAP_patch_overlay`
without values, `SHAP_pixel_map`, `SHAP_patch_bar`) were not copied. Neither were the
dataset-level figures of the three experiments, or the D01–D10 sentences. They are still in the
source folders.
