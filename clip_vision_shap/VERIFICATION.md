# Verification record

Date: 2026-09-29 — run `clip_vision_i1_i10`

## Environment

- Conda environment `RFEM` (`C:\Users\bdgou\miniconda3\envs\RFEM`)
- Python 3.11.15, torch 2.14.0+cu126 (CUDA), transformers 5.4.0, SHAP 0.51.0
  (installed into RFEM by `run_full_experiment.bat`, it was missing), NumPy 2.4.6, SciPy 1.17.1
- Model `openai/clip-vit-base-patch32`, vision tower + projection, loaded from the local cache
- Total runtime: about 60 s on GPU

## Automated checks

`python -m unittest discover -s tests -v`: 13/13 passed, including the three
model-parity tests against the real checkpoint:

1. wrapper embedding equals the standard `CLIPImageProcessor` + forward embedding;
2. the unmasked image scores 1;
3. batched and single-image scores agree.

## Full run

- 10/10 images explained with `max_evals=500` (500 model evaluations each)
- Maximum additivity residual: 0 at stored precision (score = base + sum of SHAP)
- Maximum 7x7 patch-aggregation conservation error: 1.1e-16
- Base value (fully blurred image): 0.30-0.56; SHAP sums: +0.44 to +0.70
- All 490 patch values are positive (minimum +0.0009)

## Faithfulness

| Patches | SHAP drop | Random drop | SHAP > random | Sufficiency gap |
|---:|---:|---:|---:|---:|
| 10% | 0.137 | 0.075 | 9/10 images | 0.334 |
| 20% | 0.170 | 0.102 | 9/10 images | 0.236 |
| 30% | 0.213 | 0.135 | 10/10 images | 0.182 |
| 50% | 0.278 | 0.188 | 10/10 images | 0.110 |

## Budget stability (I1, 500 vs 1,000 evaluations)

Spearman 0.982, top-5 overlap 0.80, sign agreement 1.00, same top patch, largest
patch change 0.0027.

## Limitations

- The explained score is similarity to the image's own embedding, so every visible
  region tends to help (all values positive); it shows what CLIP encodes, not a class.
- Blur is the definition of a "missing" region; Partition SHAP blocks are 28 px, not the
  32 px ViT patches, so some patch values are shared across a border.
- Ten images support a qualitative analysis only.
