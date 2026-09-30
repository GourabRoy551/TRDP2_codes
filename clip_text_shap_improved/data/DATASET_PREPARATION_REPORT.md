# Dataset Preparation Report

## Prepared files

- Training development split: 60,612 rows ({'NEG': 26801, 'POS': 33811})
- Prompt/internal validation split: 6,735 rows ({'NEG': 2978, 'POS': 3757})
- SHAP stability subset: 40 rows ({'NEG': 20, 'POS': 20})
- Final evaluation set: 500 rows ({'NEG': 250, 'POS': 250})
- Pilot regression set: 20 rows ({'NEG': 10, 'POS': 10})

The raw SST-2 training file contains 67,349 rows. Two punctuation-only records—source
row 4246 (`(`, NEG) and source row 8304 (`)`, POS)—become empty after text
normalization and are recorded in `excluded_train_rows.csv`. The usable training pool
therefore contains 67,347 rows.

## Leakage and quality checks

- Normalized duplicate groups in SST-2 training: 8,251
- Conflicting-label duplicate groups: 153
- Normalized-text overlap between training and internal validation: 0
- Previously inspected SST-2 development rows in final 500: 0
- Final examples over CLIP's 77-token limit: 0
- Unique normalized texts in final 500: 500

The datasets are prepared only. No model training, prompt selection or SHAP run was performed.
