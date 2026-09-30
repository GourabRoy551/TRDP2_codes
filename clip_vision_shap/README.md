# CLIP Vision SHAP Analysis

Partition SHAP for the CLIP Vision Encoder (`openai/clip-vit-base-patch32`) on the ten
TRDP images `I1`-`I10`. It is the image counterpart of `../bert_shap`: same layout,
same budget (500 evaluations), same faithfulness fractions and the same PNG/PDF
figure conventions. Only SHAP is computed here (no attention, rollout or RFEM).

## Main idea

CLIP's vision encoder outputs an image embedding, not a class score. The explained
scalar is therefore image-only:

```text
score(x') = cosine( CLIP_image_embedding(x'), CLIP_image_embedding(x) )
```

where `x` is the image CLIP sees (224x224 centre crop) and `x'` is a version with some
regions blurred. The full image scores exactly 1. SHAP splits the gap between the fully
blurred image (the base value) and 1 across image regions:

```text
1 = base value + sum of SHAP values
```

- positive SHAP (red): the region helps preserve CLIP's representation of the image;
- negative SHAP (blue): showing the region moves the embedding away from the original.

No text encoder, prompts, captions or labels are used. The images are not connected to
the sentences S1-S10.

## Method

1. CLIP's own `CLIPImageProcessor` resizes the shortest edge to 224 and centre-crops
   224x224; mean/std normalisation is applied inside the model wrapper so SHAP can mask
   real pixels.
2. Missing regions are replaced by a box-blurred copy of the image (kernel 128, the
   same as SHAP's `blur(128,128)` masker, implemented with SciPy so OpenCV is not needed).
3. `shap.maskers.Image` + `PartitionExplainer` (hierarchical image partitions, Owen
   values) with `max_evals=500`.
4. Pixel values are summed over colour channels, then over each 32x32 block to give one
   value per ViT-B/32 patch token (7x7 = 49). Summation preserves SHAP additivity.
5. Faithfulness: blur the top 10/20/30/50% of patches (comprehensiveness), keep only
   them (sufficiency), and compare with blurring the same number of random patches.
6. Stability: `I1` is re-explained with 1,000 evaluations and compared patch by patch.

## Project structure

```text
clip_vision_shap/
├── README.md
├── config.json                 editable experiment defaults
├── requirements.txt
├── run_full_experiment.bat     one double-click: env check, tests, full run, log
├── run_clip_vision_shap.bat    runs src/run_shap.py in the RFEM environment
├── find_python.bat             locates python.exe of the conda env RFEM
├── data/images.csv             I1-I10 paths (../clip_vision_images/images) + SHA-256
├── src/
│   ├── model_wrapper.py        CLIP loading, 224 crop, image -> similarity score
│   ├── patch_aggregation.py    blur baseline, pixel -> 7x7 patches, additivity
│   ├── evaluate_faithfulness.py  comprehensiveness, sufficiency, random, stability
│   ├── make_plots.py           per-image and global figures, run report
│   └── run_shap.py             command-line orchestration only
├── tests/
│   ├── test_additivity.py      aggregation, masking, toy Partition SHAP, faithfulness
│   └── test_model_parity.py    wrapper vs standard CLIP forward (skips if not cached)
└── outputs/
    ├── values/<run-name>/      image_summary.csv, patch_shap_values.csv,
    │                           shap_arrays.npz (pixel + patch maps), config.json
    ├── plots/<run-name>/       I1/ ... I10/, mean_absolute_patch_shap,
    │                           all_images_patch_shap_overview
    └── evaluation/<run-name>/  faithfulness.csv, faithfulness_curves,
                                stability.csv, run_report.md
```

Per image (`plots/<run>/I1/`), every figure as PNG and PDF:

- `I1_original_and_input` - original image with the region CLIP keeps, and the model input;
- `I1_SHAP_patch_overlay` - 7x7 patch SHAP over the model input;
- `I1_SHAP_patch_overlay_values` - same with the value printed in each patch;
- `I1_SHAP_pixel_map` - raw pixel-level SHAP (shows SHAP's partition blocks);
- `I1_SHAP_patch_bar` - the 12 patches with the largest |SHAP|, signed.

## Run the experiment

Easiest: double-click `run_full_experiment.bat`. It finds the conda environment `RFEM`,
installs SHAP/SciPy into it only if they are missing, runs the tests, then runs all ten
images as run `clip_vision_i1_i10`. Everything is written to
`outputs/logs/clip_vision_i1_i10.log`.

Manual runs from `TRDP2` (same environment; set `CLIP_VISION_SHAP_PYTHON` to another
`python.exe` to override):

```powershell
.\clip_vision_shap\run_clip_vision_shap.bat --run-name clip_vision_i1_i10
.\clip_vision_shap\run_clip_vision_shap.bat --image-ids I1 --run-name trial_i1
.\clip_vision_shap\run_clip_vision_shap.bat --max-images 2 --max-evals 100 --no-stability --run-name smoke
```

The model is loaded from the local Hugging Face cache (the TRDP-1 CLIP-Vision pipeline
already downloaded `openai/clip-vit-base-patch32`); if it is missing it is downloaded
once. Run names must be new; existing outputs are never overwritten.

## Tests

From the `clip_vision_shap` directory, in the RFEM environment:

```powershell
python -m unittest discover -s tests -v
```

## Interpretation cautions

- Attributions depend on blurring representing a "missing" region.
- Partition SHAP uses hierarchical image blocks (Owen values), not exhaustive Shapley
  enumeration. Its blocks do not align exactly with the 32x32 ViT patches; a block that
  crosses a patch border shares its value between the patches by pixel count.
- Values explain similarity to the image's own embedding, not a class or a caption.
- CLIP only sees the centre crop; content outside the dashed box cannot receive credit.
- Ten images support a qualitative analysis, not a dataset-level conclusion.
