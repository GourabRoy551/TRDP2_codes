# BERT Dual-Class Partition SHAP

This is an independent extension of the existing `bert_shap` experiment. It
explains both BERT sentiment outputs (`NEG` and `POS`) for every sentence. The
existing implementation, which explains only the scalar margin
`POS logit - NEG logit`, and all of its saved outputs remain unchanged.

The CLIP text and vision encoders are deliberately outside the scope of this
stage.

## Experiment design

The pipeline is:

1. Read a sentence and its gold sentiment.
2. Tokenize it with the same BERT tokenizer used by the classifier.
3. Use `shap.maskers.Text` to define which tokens are present or masked.
4. Use Partition SHAP to generate hierarchy-consistent token coalitions.
5. Send each masked sentence through BERT and retain both class logits in the
   fixed order `[NEG, POS]`.
6. Estimate one SHAP value per token and per class.
7. Aggregate WordPiece tokens to readable words while preserving each class
   sum.
8. Save token, word, sentence, and dataset-level results and plots.

For class \(c\), SHAP additivity is checked as

\[
z_c(x) = \phi_{0,c} + \sum_{i=1}^{M}\phi_{i,c},
\]

where \(z_c(x)\) is the BERT class logit, \(\phi_{0,c}\) is the base value,
and \(\phi_{i,c}\) is the contribution of token \(i\) to class \(c\).

The earlier scalar-margin explanation can be recovered from the two outputs:

\[
z_{POS}(x)-z_{NEG}(x)
= (\phi_{0,POS}-\phi_{0,NEG})
+ \sum_i(\phi_{i,POS}-\phi_{i,NEG}).
\]

Therefore, the per-token margin contribution is
\(\phi_{i,POS}-\phi_{i,NEG}\). A positive SHAP value increases the named
class logit relative to its masked baseline; a negative value decreases it.

## Dataset

`data/sentences_20_curated.csv` contains 20 labeled examples:

- S1-S10 are copied unchanged from the original qualitative experiment.
- D01-D10 are human-selected from the local SST-2 development set. They are
  complete, clearly polarized reviews with presentation-ready capitalization
  and punctuation.
- The completed set is balanced: 10 `NEG` and 10 `POS` sentences.
- The original source and source row identifier are retained for provenance.

The reviewed additions are stored separately in
`data/curated_sst2_additions.csv`. The original S1-S10 set contains 4 negative
and 6 positive examples, so the curated file adds 6 negative and 4 positive
SST-2 examples.

## Files

- `config.json` - model, data, runtime, and output settings.
- `src/prepare_dataset.py` - source validation and reproducible balanced-dataset
  construction from the reviewed additions.
- `src/model_wrapper_dual.py` - batched BERT wrapper returning `[NEG, POS]`
  logits.
- `src/token_aggregation_dual.py` - two-class unpacking, additivity checks,
  WordPiece aggregation, and margin reconstruction.
- `src/make_matrix_plots.py` - sentence- and dataset-level PNG/PDF figures.
- `src/run_dual_class_shap.py` - experiment runner and report generator.
- `tests/` - dataset, mathematics, and direct-model parity tests.
- `outputs/values/` - numerical CSV results.
- `outputs/plots/` - matching PNG and PDF figures.
- `outputs/reports/` - a human-readable summary for each run.

## Running the experiment

Install the parent project's dependencies from `../requirements.txt`. On the
original Windows machine, the convenience launcher uses the existing research
environment automatically. A different interpreter can be selected with the
`BERT_SHAP_PYTHON` environment variable.

Prepare or reproduce the 20-sentence dataset:

```powershell
python src\prepare_dataset.py
```

Run all 20 examples with the configured 500-evaluation budget:

```powershell
.\run_dual_class_shap.bat --run-name bert_dual_20_curated
```

Run selected sentences when testing changes:

```powershell
.\run_dual_class_shap.bat --run-name trial_s1_s2 --sentence-ids S1 S2
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Run names must be unique because the program will not overwrite an existing
values, plots, or reports directory.

## Saved values

Each completed run contains:

- `sentence_summary.csv` - logits, probabilities, base values, SHAP sums,
  reconstructed logits, residuals, prediction, and strongest words.
- `token_shap_values.csv` - one row for every BERT token and target class.
- `word_shap_values.csv` - long-format aggregated word values.
- `word_shap_values_wide.csv` - NEG, POS, and POS-minus-NEG values side by
  side for every word.
- `margin_comparison.csv` - comparison with the saved original scalar-margin
  run for S1-S10.
- `run_config.json` - resolved configuration, package versions, and runtime
  information.

## Saved figures

Five figure types are generated for every sentence in both PNG and PDF:

1. dual-class word bars;
2. token-by-class SHAP matrix;
3. word-by-class SHAP matrix;
4. sentence-summary value matrix; and
5. presentation overview combining word and sentence information.

Five dataset-level figures are also written in both formats:

1. all-sentence summary matrix;
2. sentence-by-class mean absolute SHAP matrix;
3. global word-by-class importance matrix;
4. class additivity-residual matrix; and
5. global signed class-comparison bars.

For 20 sentences this gives 105 PNG files and 105 matching PDF files.

## Completed curated run

The validated run for `sentences_20_curated.csv` is
`bert_dual_20_curated`.

- BERT predictions correct: 20/20;
- token/class pairs missing or duplicated: 0;
- maximum NEG additivity residual: \(1.776\times10^{-15}\);
- maximum POS additivity residual: \(8.882\times10^{-16}\);
- word-level margin identity errors: 0;
- figures generated: 105 PNG and 105 matching PDF files;
- maximum token difference from the earlier scalar-margin run for S1-S10:
  0.0231 (S3); the other nine comparisons differ at approximately
  floating-point precision.

The finite 500-evaluation approximation explains the small S3 margin
difference. Both class-specific additivity equations still close to machine
precision.

The qualitative results identify clear sentiment terms in both directions.
For example, `powerful` increases the POS logit by 1.998 and decreases the NEG
logit by 2.095. Conversely, `awful` increases the NEG logit by 3.601 and
decreases the POS logit by 3.128. These observations describe the selected
qualitative sample and are not population-level performance estimates.
