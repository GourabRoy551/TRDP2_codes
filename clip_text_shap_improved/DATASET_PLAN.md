# Dataset Plan

## Primary dataset

The improved experiment will use the Stanford Sentiment Treebank binary task
distributed as SST-2. This is the same sentiment domain used by the existing BERT
checkpoint and the current TRDP experiments.

Local source files:

```text
D:/Research/TRDP_Study Note/TRDP1/trdp/
sst2_quantitative_analysis/data/SST-2/train.tsv

D:/Research/TRDP_Study Note/TRDP1/trdp/
sst2_quantitative_analysis/data/SST-2/dev.tsv
```

Verified local counts:

| Local file | Total | NEG (`0`) | POS (`1`) | Planned role |
|---|---:|---:|---:|---|
| `train.tsv` | 67,349 raw / 67,347 usable | 29,779 usable | 37,568 usable | Training and internal development |
| `dev.tsv` | 872 | 428 | 444 | Final held-out evaluation pool |

Label mapping is fixed as:

```text
SST-2 label 0 = NEG
SST-2 label 1 = POS
```

The SST-2 `test.tsv` file will not be used for quantitative evaluation because its
gold labels are not present locally.

## Role of the existing 20 sentences

The approved `sentences_20_curated.csv` remains the Part 1 regression and presentation
pilot. It will be used to verify that the new implementation reproduces the completed
CLIP baseline before adding the margin output.

It will not be used to train the CLIP classification head, select classifier
hyperparameters, select the final prompt ensemble, or calculate the final 500-sample
performance metrics.

Ten of the current pilot sentences came from SST-2 `dev.tsv`. Their source row IDs are:

```text
29, 59, 129, 181, 212, 327, 429, 504, 591, 603
```

Those already inspected rows will be excluded from the final 500-sentence sample to
reduce example-selection and interpretation bias.

## Training and internal validation split

The 67,347 usable SST-2 training records will be divided deterministically with seed
`42` using stratification by label. Two raw punctuation-only records—row 4246 (`(`,
NEG) and row 8304 (`)`, POS)—have no lexical content after normalization. They will
be preserved in `excluded_train_rows.csv` and will not be used for model fitting.

| Internal split | Total | NEG | POS | Purpose |
|---|---:|---:|---:|---|
| Development training | 60,612 | 26,801 | 33,811 | Fit the frozen-CLIP linear head |
| Internal validation | 6,735 | 2,978 | 3,757 | Prompt and hyperparameter selection |

The internal validation split will be used for:

- selecting the balanced prompt ensemble;
- selecting the linear-head regularization and stopping settings;
- selecting the SHAP evaluation budget;
- development-only masking and stability checks.

After all classifier hyperparameters are frozen, the final linear head may be refitted
once using all 67,347 usable SST-2 training rows. No `dev.tsv` record will be used for fitting
or model selection.

## SHAP stability subset

A fixed 40-sentence subset will be drawn from the internal validation split:

- 20 NEG and 20 POS;
- short, medium and long sentences represented;
- source rows and hashes recorded;
- no overlap with the final evaluation sample because it comes from `train.tsv`.

This subset will be used to compare SHAP budgets such as 500, 1,000 and 2,000 model
evaluations. It is not part of the final reported 500-sample performance estimate.

## Final 500-sentence evaluation set

The final set will be sampled only from labelled SST-2 `dev.tsv` rows after excluding
the ten previously inspected development rows.

Target composition:

| Final group | Count |
|---|---:|
| NEG | 250 |
| POS | 250 |
| Total | 500 |

Sampling will be deterministic with seed `42`. The same 500 sentence IDs, texts and
labels will be used for:

- zero-shot CLIP prototype scoring;
- trained-head CLIP scoring;
- BERT scoring;
- SHAP faithfulness calculations;
- BERT–CLIP explanation comparison.

This is called the **final evaluation set**, not the official SST-2 test set. The
official SST-2 test labels are unavailable, so a frozen subset of the labelled
development split is used as the held-out comparison set.

## Eligibility and data-quality rules

Before sampling, the preparation script will:

1. verify the source-file hashes;
2. remove empty records;
3. detect exact and normalized-text duplicates;
4. exclude the ten previously inspected `dev.tsv` row IDs;
5. check CLIP token length against its 77-token limit;
6. avoid silent truncation by recording and excluding any over-limit candidate before
   the final stratified sample is drawn;
7. preserve the original text, label and source row ID.

Every prepared record will contain at least:

```text
sentence_id
text
gold_label
source_split
source_row_id
selection_seed
clip_bpe_token_count
text_sha256
```

## Reproducibility artifacts

Dataset preparation will produce:

- `train_internal_split_manifest.json`;
- `train_development.csv` and `excluded_train_rows.csv`;
- `prompt_validation.csv`;
- `shap_stability_40.csv`;
- `final_evaluation_500.csv`;
- `final_evaluation_manifest.json`;
- exclusion and duplicate audit tables;
- SHA-256 hashes and class-count checks.

No row will move between roles after the manifests are frozen.

