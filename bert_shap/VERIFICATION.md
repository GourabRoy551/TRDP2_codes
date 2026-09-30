# Verification record

Date: 2026-09-23

## Previous-work alignment

The implementation was checked against the read-only previous project at:

```text
D:\Research\TRDP_Study Note\TRDP1\trdp
```

The following conventions were preserved:

- checkpoint: `textattack/bert-base-uncased-SST-2`;
- qualitative inputs: S1-S10 exactly as defined in `rfem_pipeline.py`;
- labels: index 0 = NEG and index 1 = POS;
- maximum sequence length: 128;
- `[MASK]` replacement for perturbation-based faithfulness;
- special tokens and punctuation excluded from content rankings;
- separate run and per-sentence output directories;
- PDF figures suitable for the report and presentation.

No previous-project files or environments were modified.

## Automated checks

`python -m unittest discover -s tests -v` passes seven tests covering:

1. exact recovery of the ten qualitative examples;
2. equality between the wrapper margin and a direct BERT forward pass;
3. equality between wrapper and direct predicted class;
4. SHAP base-value plus attribution additivity;
5. correct residual magnitude and sign;
6. WordPiece aggregation and signed-value conservation;
7. exclusion of special and punctuation tokens from content rankings.

## Full integration run

Run directory:

```text
outputs/values/s1_s10_shap
outputs/plots/s1_s10_shap
outputs/evaluation/s1_s10_shap
```

Configuration:

- Python 3.12.12
- torch 2.12.0
- transformers 5.9.0
- SHAP 0.52.0
- CPU inference
- `max_evals=500`
- `batch_size=32`
- uncollapsed BERT `[MASK]` tokens

Observed checks:

- 10/10 predictions matched the qualitative gold labels;
- maximum absolute additivity residual: 0 at stored precision;
- total runtime: 33.68 seconds;
- total model text evaluations: 3,814;
- mean decision-score comprehensiveness drop: 5.3107;
- mean decision-score sufficiency gap: 0.8927.

The S1, S7, S8, and global-summary PNGs were inspected visually. Titles,
labels, signs, WordPiece aggregation, and value annotations were legible with
no clipping or overlap.

## Refactor integration check

The modular `src/run_shap.py` entry point was run through
`run_qualitative_shap.bat` for all ten sentences with `max_evals=500`. The run
is stored under the name `modular_s1_s10` in each of the new `values`, `plots`,
and `evaluation` directories.

Observed checks:

- 10/10 predictions matched their gold labels;
- maximum absolute additivity residual: 0 at stored precision;
- 40 faithfulness rows and 32 plot/HTML files were generated;
- all 104 word-level SHAP values exactly matched the pre-refactor run;
- total modular-run time: 39.70 seconds.

## Budget stability check

S3 was rerun with `max_evals=1000` and compared with its 500-evaluation result.
The largest absolute word-level change was 0.0322. All dominant signs and the
ranking of the main sentiment-bearing words were preserved.

## Scientific limitations retained in the documentation

- Attributions are conditional on `[MASK]` representing a missing feature.
- PartitionExplainer uses hierarchical coalitions and therefore estimates
  Owen-style values rather than unrestricted exact Shapley enumeration.
- Masked sentences may differ from BERT's SST-2 fine-tuning distribution.
- The ten hand-written sentences support a qualitative comparison, not a
  dataset-scale conclusion.

