# Improved CLIP Text Encoder SHAP: Staged Experiment Plan

## Objective

Improve the clarity, sentiment specificity and scientific comparability of the CLIP Text Encoder SHAP experiment while preserving the completed experiment as an unchanged baseline.

The follow-up will answer three separate questions:

1. Can the existing zero-shot CLIP explanation distinguish sentiment direction more clearly using a POS-minus-NEG margin?
2. Can prompt selection and whole-word coalitions improve the stability and readability of the zero-shot explanations?
3. Does a sentiment classifier trained on frozen CLIP text embeddings provide a fairer comparison with the fine-tuned BERT classifier?

These questions will not be mixed into one run. They will be addressed in seven sequential parts.

The dataset source, split counts, leakage controls and final balanced 500-sentence
sampling procedure are specified in `DATASET_PLAN.md`.

## Non-negotiable controls

- `clip_text_shap/` and `bert_shap/` remain read-only.
- The same `[NEG, POS]` class order is used everywhere.
- Partition SHAP remains the explanation algorithm.
- All datasets, splits, prompt sets, seeds and model versions are recorded.
- Prompt selection and model training use training/validation data only; the final test set is not used for selection.
- Raw BPE explanations are preserved even when whole-word explanations are added.
- Every generated numerical plot is saved in PNG and PDF.
- Numerical CSV results are saved for every evaluated sentence, but detailed
  per-sentence plots are generated only for five documented presentation examples.
- Raw BERT logits and CLIP cosine similarities are never compared directly by magnitude.
- Each part finishes with tests, a short results note and an approval checkpoint.

## Planned project structure

```text
TRDP2/
├── bert_shap/                    # existing; read-only
├── clip_text_shap/               # completed baseline; read-only
└── clip_text_shap_improved/      # new staged experiment
    ├── README.md
    ├── EXPERIMENT_PLAN.md
    ├── DATASET_PLAN.md
    ├── plans/
    │   ├── 01_margin_baseline.md
    │   ├── 02_prompt_validation.md
    │   ├── 03_word_level_partition.md
    │   ├── 04_frozen_clip_classifier.md
    │   ├── 05_shap_stability.md
    │   ├── 06_full_evaluation.md
    │   └── 07_comparison_and_report.md
    ├── data/                     # added only when the relevant part begins
    ├── configs/
    ├── src/
    ├── tests/
    └── outputs/
```

## Sequence and approval gates

### Part 1 — Reproduce the baseline and add the discriminative margin

Create an independent minimal implementation that reproduces the existing 20-sentence scores, then add a third explained output:

```text
margin score for sentence x
    = POS similarity for x - NEG similarity for x

s_margin(x) = s_POS(x) - s_NEG(x)
```

and:

```text
margin SHAP value for token i
    = POS SHAP value for token i - NEG SHAP value for token i

phi[i, margin] = phi[i, POS] - phi[i, NEG]
```

The plots will show `NEG`, `POS` and `POS−NEG` together. Directly explained margin values will be compared with the difference of the two class explanations.

**Gate:** Continue only if baseline parity, additivity and margin-linearity tests pass.

### Part 2 — Validate a robust prompt ensemble

Define several balanced prompt families, evaluate them on a validation split, measure accuracy and prompt sensitivity, and freeze one prompt ensemble before using the final evaluation set.

**Gate:** Continue only after reviewing the prompt comparison and approving the frozen prompt set.

### Part 3 — Add whole-word hierarchical masking

Keep raw CLIP BPE explanations, but create a second explanation mode in which all BPE pieces belonging to one word are treated as one coalition. Compare BPE-level and whole-word results for correctness, runtime and stability.

**Gate:** Continue only if no perturbation masks a partial word and word-level additivity is preserved.

### Part 4 — Train a frozen-CLIP sentiment classifier

Freeze the CLIP Text Encoder and train only a small linear NEG/POS classification head on sentiment training data. This produces sentiment logits that are more comparable with BERT while retaining CLIP embeddings.

The zero-shot model and trained-head model remain separate experimental conditions.

**Gate:** Continue only if the saved classifier reproduces its evaluation scores and improves validation performance over the zero-shot baseline.

### Part 5 — Determine a stable SHAP evaluation budget

Run a representative subset with several budgets, initially 500, 1,000 and 2,000 evaluations. Compare word-rank stability, top-k overlap, sign consistency and runtime.

**Gate:** Select the smallest budget that meets the predefined stability threshold before running the larger dataset.

### Part 6 — Run the full evaluation on approximately 500 samples

Create one fixed, balanced and reproducible evaluation set. Run the approved zero-shot-margin and trained-head conditions using the selected SHAP budget.

Calculate classification, faithfulness, stability and efficiency metrics with confidence intervals.

All approximately 500 sentences receive complete CSV results. To avoid producing
thousands of repetitive image files, only five representative sentences receive a
combined presentation overview plot. The selection rule is recorded before inspecting
their SHAP values.

**Gate:** Freeze all result tables and plots before beginning the final model comparison.

### Part 7 — Compare BERT and CLIP and prepare the report

Compare the models using identical examples and scale-independent explanation metrics. Produce final tables, presentation-ready plots, limitations and an academic report.

**Gate:** Final review of conclusions to ensure that every claim is supported by the measured results.

## Planned headline metrics

### Prediction metrics

- accuracy;
- precision, recall and F1 for NEG and POS;
- macro-F1;
- confusion matrix;
- margin distribution and prediction confidence;
- results separated into correct and incorrect predictions.

### Explanation metrics

- class-specific SHAP additivity residual;
- direct-versus-derived margin SHAP difference;
- top-k deletion AOPC;
- random-deletion baseline;
- comprehensiveness and sufficiency;
- rank stability across evaluation budgets;
- prompt sensitivity;
- whole-word versus BPE agreement.

### BERT–CLIP comparison metrics

- Spearman word-rank correlation;
- top-k overlap and Jaccard score;
- contribution-sign agreement;
- prediction agreement;
- faithfulness difference;
- runtime and model-evaluation count.

Raw SHAP magnitudes will only be compared when the explained output scales are genuinely compatible.

## Plot-generation policy

The large evaluation will not generate a complete plot suite for every sentence.
Instead, it will use the following two levels.

### Complete numerical results for every sentence

For every evaluated sentence, CSV files will contain:

- NEG, POS and POS-minus-NEG scores;
- token-level and word-level SHAP values;
- base values, SHAP sums, reconstructed scores and residuals;
- prediction, gold label and correctness;
- faithfulness and stability metrics.

### Five representative presentation examples

Five sentence IDs will be selected using a documented rule before their SHAP values
are inspected:

1. one correctly predicted positive sentence;
2. one correctly predicted negative sentence;
3. one mixed or contrastive sentence;
4. one longer sentence containing several BPE splits;
5. one misclassified or lowest-margin sentence.

If there is no misclassification, the lowest-confidence correct prediction will be
used for the fifth example. The same sentence IDs will be used across comparable
model conditions wherever possible.

Each example produces one combined overview figure containing:

- a word-by-output matrix with `NEG`, `POS` and `POS−NEG` columns;
- the numerical SHAP value in every cell;
- sentence scores, base values, SHAP sums and reconstruction residuals;
- the gold label and model prediction.

This produces five primary example plots rather than hundreds of per-sentence plots.
Each is saved once as PNG and once as PDF.

### Aggregate presentation plots

The experiment will also generate a compact set of approximately five aggregate
figures:

1. classification confusion matrix and metrics;
2. global class and margin word-importance matrix;
3. faithfulness deletion curves with random baselines;
4. SHAP-budget stability and runtime comparison;
5. BERT versus CLIP explanation comparison.

These aggregate plots summarize all evaluated sentences and display the relevant
numerical values directly on the figure.

## Definition of overall success

The project will be considered successful if it produces:

1. valid NEG, POS and margin explanations with verified additivity;
2. clearer sentiment-direction plots than the raw zero-shot baseline;
3. a reproducible prompt-selection procedure without test leakage;
4. whole-word explanations that avoid partial-BPE perturbations;
5. a frozen-CLIP classifier that can be fairly compared with BERT;
6. stable explanations at a documented evaluation budget;
7. a 500-sample evaluation with appropriate uncertainty and limitations;
8. complete CSV results for all samples and numerically annotated PNG/PDF figures
   for the five presentation examples and aggregate results.

## Working rule

Only one part will be implemented at a time. After completing a part, the following will be provided before proceeding:

- the code and configuration created in that part;
- tests and verification results;
- generated plots or tables;
- a concise interpretation;
- any limitations or unexpected findings;
- a request for approval to begin the next part.
