# Part 2 — Prompt Validation and Robustness

## Purpose

Select prompts systematically rather than choosing them from the final test sentences.

## Planned work

1. Define several balanced NEG/POS prompt families.
2. Create fixed training, validation and final-test boundaries.
3. Evaluate each family on validation data only.
4. Measure accuracy, macro-F1, margin separation and prediction stability.
5. Measure whether important-word rankings change across prompt families.
6. Select and freeze one ensemble using predefined criteria.
7. Record rejected prompt sets and selection reasons.

## Required controls

- equal numbers of NEG and POS prompts;
- structurally symmetric wording where possible;
- no prompt selection using final-test results;
- fixed model, tokenizer and random seed;
- no calibrated-probability claim unless calibration is separately validated.

## Deliverables

- prompt registry and configuration files;
- annotated prompt-performance plots;
- prompt-sensitivity tables;
- frozen-prompt manifest;
- short Part 2 conclusion.

## Stop condition

Do not begin whole-word masking until the frozen prompt ensemble has been approved.

