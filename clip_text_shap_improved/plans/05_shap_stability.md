# Part 5 — SHAP Budget and Stability Study

**Status: completed.** The gate passed and the smallest qualifying maximum budget
was 500 evaluations per sentence. The final 500-sentence evaluation set was not used.

## Purpose

Choose an evaluation budget using measured explanation stability rather than an arbitrary number.

## Planned work

1. Select a representative subset containing short, medium and long sentences, both classes, correct predictions and errors.
2. Run Partition SHAP at 500, 1,000 and 2,000 evaluations.
3. Record runtime and actual model-evaluation counts.
4. Compare word rankings, top-k sets, signs and values between budgets.
5. Repeat selected runs to verify determinism.
6. Select the smallest budget satisfying the agreed stability threshold.

## Proposed selection criteria

- median Spearman rank correlation with the 2,000-evaluation reference at least 0.95;
- median top-5 overlap at least 0.80;
- margin-contribution sign agreement at least 0.95;
- additivity residual within numerical tolerance;
- no material conclusion change for the inspected sentences.

The thresholds may be adjusted before implementation, but not after viewing the final full-run results.

## Deliverables

- budget-comparison CSV files;
- annotated stability and runtime plots;
- chosen budget with justification;
- short Part 5 conclusion.

## Stop condition

Do not run the 500-sample experiment until the budget has been approved.

