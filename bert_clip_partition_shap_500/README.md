# BERT vs CLIP Text Encoder — whole-word Partition SHAP on 500 SST-2 sentences

Self-contained experiment comparing a fine-tuned SST-2 BERT classifier with the frozen,
zero-shot CLIP ViT-B/32 text encoder. Both models score and explain the **same 500
balanced sentences** (250 NEG / 250 POS) with **whole-word Partition SHAP** for three
outputs: NEG, POS and the POS−NEG decision margin. That gives 1,000 model–sentence
explanation records.

No new model training or fine-tuning was performed during this experiment. BERT was
already fine-tuned on SST-2 by its publisher; CLIP is used as a frozen zero-shot encoder
with a prompt family frozen in an earlier experiment.

Results: [outputs/reports/RESULTS_SUMMARY.md](outputs/reports/RESULTS_SUMMARY.md) (short) and
[outputs/reports/FINAL_REPORT.md](outputs/reports/FINAL_REPORT.md) (full). Design and gates:
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

## Relationship to other folders

Everything lives in this folder. The sibling projects `bert_shap/`, `clip_text_shap/` and
`clip_text_shap_improved/` were used as read-only references (model choice, label
mapping, prompt family, masking design, output formats). They were not modified. Two
files are read from `clip_text_shap_improved/` at run time, read-only: the source dataset
(to verify the copy's hash) and the frozen prompt manifest (to verify the prompts in
`config.json`).

## Running

Environment: conda env `rfem` (Python 3.11, torch 2.14 + CUDA 12.6, transformers 5.4,
shap 0.51). Both models load offline from the local Hugging Face cache.

```bat
run_experiment.bat
```

or stage by stage:

```bat
set PY=%USERPROFILE%\miniconda3\envs\rfem\python.exe
%PY% src\run_experiment.py --stage prepare      & rem copy + validate dataset
%PY% src\run_experiment.py --stage score        & rem frozen models, label/prompt/parity checks, case-study selection
%PY% src\run_experiment.py --stage smoke        & rem masking audit + 5-sentence SHAP smoke test
%PY% src\run_experiment.py --stage shap --model bert
%PY% src\run_experiment.py --stage shap --model clip
%PY% src\run_experiment.py --stage metrics
%PY% src\run_experiment.py --stage plots
%PY% src\run_experiment.py --stage report
%PY% -m pytest tests -p no:cacheprovider
```

Each stage refuses to start if the previous gate failed. The SHAP stage appends one
validated JSON record per sentence to `outputs/checkpoints/<model>_shap_records.jsonl`
and resumes from it after an interruption (`--limit N` stops early on purpose).

## Layout

| Path | Content |
|---|---|
| `config.json` | every setting: seed, paths, models, prompts, batch sizes, budget, fractions, tolerances |
| `data/evaluation_500.csv`, `data/dataset_manifest.json` | frozen dataset copy and its verification record |
| `src/` | `data_validation`, `model_outputs`, `bert_backend`, `clip_backend`, `whole_word_masker`, `partition_shap`, `faithfulness`, `evaluation_metrics`, `representative_selection`, `plotting`, `report_builder`, `io_utils`, `run_experiment` |
| `tests/` | dataset, output contract, masking, additivity, completeness and plot tests |
| `outputs/values/` | `sentence_results.csv` (1,000 rows), `word_shap_values.csv`, `faithfulness_by_fraction.csv`, `model_scores.csv`, `representative_selection.csv` |
| `outputs/metrics/` | classification, bootstrap CIs, additivity, pairwise comparison/summary, subgroups, runtime, faithfulness summary, acceptance checks |
| `outputs/plots/` | five aggregate figures + `case_studies/` (five), each PNG + PDF |
| `outputs/plots/word_bars/` | report-ready NEG / POS / POS−NEG word bar charts for four sentences (EV00003, EV00002, EV00091, EV00001): BERT and CLIP from this experiment, zero-shot CLIP and the frozen linear head from `clip_text_shap_improved` (read-only source); stage `--stage wordbars` |
| `outputs/checkpoints/` | per-sentence JSONL records, progress files, model checks, masking audit, smoke tests, run logs |
| `outputs/reports/` | `FINAL_REPORT.md`, `RESULTS_SUMMARY.md`, `latex/` |
