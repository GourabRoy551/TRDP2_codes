# BERT SHAP Analysis

This is an independent, readable implementation of the BERT–SHAP experiment.
It follows the previous TRDP/RFEM project’s checkpoint, S1–S10 sentences,
labels, maximum sequence length, `[MASK]` perturbation convention, and PDF
figure style. The existing `TRDP` project is treated as read-only.

## Main idea

The model output explained by SHAP is:

```text
positive logit - negative logit
```

Consequently, a positive SHAP value pushes the decision toward positive
sentiment, while a negative value pushes it toward negative sentiment. For each
sentence, the following additivity relationship is checked:

```text
model margin ≈ SHAP base value + sum of SHAP values
```

The explainer is SHAP’s text `PartitionExplainer`. It measures the effect of
hiding features in different coalitions and does not interpret BERT attention
weights as explanations.

## Project structure

```text
TRDP2/
├── TRDP/                         existing RFEM project; never modified
└── bert_shap/
    ├── README.md
    ├── requirements.txt
    ├── config.json               editable experiment defaults
    ├── data/
    │   └── sentences.csv         original S1–S10 qualitative set
    ├── src/
    │   ├── model_wrapper.py      BERT loading and scalar margin wrapper
    │   ├── run_shap.py           command-line orchestration
    │   ├── token_aggregation.py  SHAP/BERT alignment and WordPiece merging
    │   ├── evaluate_faithfulness.py
    │   └── make_comparison_plots.py
    ├── tests/
    │   ├── test_model_parity.py
    │   └── test_additivity.py
    └── outputs/
        ├── values/               CSV values and run configuration
        ├── plots/                PDF, PNG, and HTML explanations
        └── evaluation/           faithfulness CSV and run report
```

## What each Python file does

- `model_wrapper.py` loads BERT and exposes one scalar per input. The wrapper
  also reports the original logits, probabilities, and predicted label.
- `token_aggregation.py` aligns SHAP’s features with BERT tokens, joins `##`
  fragments into readable words, and calculates the additivity residual.
- `evaluate_faithfulness.py` masks or retains the highest-ranked whole words
  and computes comprehensiveness and sufficiency.
- `make_comparison_plots.py` creates per-sentence signed bar charts, colored
  HTML explanations, the cross-sentence importance plot, and the run report.
- `run_shap.py` reads configuration and data, calls the other modules, and
  writes all artifacts. It contains no model mathematics or plotting details.

## Setup

From the `bert_shap` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The supplied batch runner can also reuse the sibling TRDP environment. If the
local ignored `.packages` directory exists, it is added to `PYTHONPATH` without
changing the older project’s environment.

## Run the experiment

From `TRDP2`:

```powershell
.\bert_shap\run_qualitative_shap.bat --run-name s1_s10_shap
```

For an offline run after the checkpoint is cached:

```powershell
.\bert_shap\run_qualitative_shap.bat --run-name offline_run --local-files-only
```

Useful smaller runs:

```powershell
.\bert_shap\run_qualitative_shap.bat --sentence-ids S1 S7 S10 --run-name examples
.\bert_shap\run_qualitative_shap.bat --max-sentences 1 --max-evals 100 --run-name smoke
.\bert_shap\run_qualitative_shap.bat --no-faithfulness --run-name shap_only
```

Edit `config.json` to change persistent defaults. Command-line arguments
override those defaults. Use `--config <path>` to select another configuration.

## Output layout

One run name is used in three parallel directories:

```text
outputs/
├── values/<run-name>/
│   ├── config.json
│   ├── sentence_summary.csv
│   ├── token_shap_values.csv
│   └── word_shap_values.csv
├── plots/<run-name>/
│   ├── mean_absolute_word_shap.pdf
│   ├── mean_absolute_word_shap.png
│   └── S1/ ... S10/
└── evaluation/<run-name>/
    ├── faithfulness.csv
    └── run_report.md
```

## Tests

From the `bert_shap` directory, using the environment containing PyTorch and
Transformers:

```powershell
python -m unittest discover -s tests -v
```

`test_model_parity.py` compares the wrapper against a direct BERT forward pass.
It skips cleanly when the checkpoint is not already cached.
`test_additivity.py` tests the SHAP reconstruction formula and WordPiece logic.

## Interpretation cautions

- Results depend on defining a missing word with BERT’s `[MASK]` token.
- Masked sentences may differ from the model’s fine-tuning distribution.
- Partition SHAP uses hierarchical coalitions; it is not exhaustive Shapley
  enumeration over every possible subset.
- Inspect additivity, stability across evaluation budgets, and faithfulness—not
  only visually plausible token colors.
# bert_shap
