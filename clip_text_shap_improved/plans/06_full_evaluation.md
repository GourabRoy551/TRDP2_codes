# Part 6 — Full Evaluation on Approximately 500 Samples

**Status: completed.** The frozen two-condition CLIP evaluation contains 1,000
sentence explanations, complete per-word and faithfulness tables, bootstrap
intervals, five aggregate figures and five preselected example figures. The Part 6
integrity gate passed and the frozen budget remained 500 evaluations.

## Purpose

Evaluate the approved models and explanation settings on a larger fixed dataset.

## Planned work

1. Construct the fixed 500-sentence SST-2 evaluation set defined in
   `DATASET_PLAN.md`: 250 NEG and 250 POS examples sampled from labelled `dev.tsv`
   with seed 42 after excluding previously inspected rows.
2. Save source identifiers, hashes, class counts and split information.
3. Run the frozen zero-shot-margin condition.
4. Run the frozen-CLIP trained-head condition.
5. Calculate classification and explanation metrics.
6. Estimate uncertainty with bootstrap confidence intervals.
7. Separate results by class, prediction correctness and sentence-length group.
8. Save complete numerical CSV tables for every sentence.
9. Select five representative sentences using the documented rule before inspecting
   their SHAP values.
10. Generate one combined NEG/POS/margin overview plot for each selected sentence.
11. Generate approximately five aggregate presentation plots for the full dataset.

## Metrics

- accuracy, per-class precision/recall/F1 and macro-F1;
- confusion matrix;
- additivity residuals;
- top-k deletion AOPC and random baseline;
- comprehensiveness and sufficiency;
- class-margin statistics;
- explanation sparsity and concentration;
- runtime and evaluation counts.

## Deliverables

- frozen dataset manifest;
- per-sentence and aggregate result tables;
- confidence-interval tables;
- five presentation-ready example plots in PNG and PDF;
- approximately five annotated aggregate plots in PNG and PDF;
- error analysis;
- short Part 6 conclusion.

## Stop condition

Freeze the outputs before beginning BERT comparison or drafting final conclusions.

