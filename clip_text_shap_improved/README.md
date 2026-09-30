# Improved CLIP Text SHAP Experiment

This folder is reserved for the staged follow-up experiment. It is independent of:

- `../clip_text_shap/` — completed zero-shot CLIP Text Encoder baseline;
- `../bert_shap/` — completed BERT SHAP work.

Neither existing folder will be modified. No experiment code has been implemented here yet.

The complete staged plan is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), and the
proposed SST-2 split and sampling rules are in [DATASET_PLAN.md](DATASET_PLAN.md).
Each part has its own detailed plan under `plans/`. Work stops after every part for
review and approval before the next part begins.

## Dataset preparation status

The datasets have now been prepared under `data/` for review. The principal review
files are:

- `data/dataset_review.xlsx` — formatted workbook containing the full final 500,
  stability 40, pilot 20, prompt-validation split, exclusions and integrity audit;
- `data/DATASET_PREPARATION_REPORT.md` — concise preparation and quality summary;
- `data/final_evaluation_500.csv` — balanced held-out set (250 NEG, 250 POS);
- `data/shap_stability_40.csv` — balanced development-only stability subset;
- `data/train_development.csv` and `data/prompt_validation.csv` — group-aware internal
  split with no normalized-text leakage.

Dataset preparation and all seven experimental parts have been completed. Part 5
selected and froze a maximum budget of 500 evaluations per sentence for whole-word
Partition SHAP. Parts 6 and 7 then used the previously untouched balanced final set
of 500 sentences (250 NEG and 250 POS).

## Parts 1 and 2 implementation

The approved implementation is separated into readable modules under `src/`:

- `clip_backend.py` — local CLIP loading, batched embeddings and `[NEG, POS, MARGIN]`
  scoring;
- `prompting.py` — balanced prompt validation and class-prototype construction;
- `shap_utils.py` — text masking, Partition-SHAP extraction, additivity and BPE-to-word
  aggregation;
- `plotting.py` — numerically annotated PNG/PDF figures;
- `run_part1.py` — baseline parity and direct/derived margin experiment;
- `run_part2.py` — internal-validation prompt comparison and prompt freezing.

`run_parts_1_2.bat` executes Part 1 first and starts Part 2 only if the Part 1 gate
passes. The final 500-sentence evaluation file is not an input to either runner.

Completed results are under:

- `outputs/part1_margin_baseline/`;
- `outputs/part2_prompt_validation/`;
- `outputs/PARTS_1_2_SUMMARY.md`.

The selected prompt family is frozen in `configs/frozen_prompt_manifest.json`.

Parts 3 and 4 add:

- `word_masking.py` — indivisible whole-word coalitions and BPE mapping audits;
- `run_part3.py` — BPE versus direct whole-word Partition-SHAP comparison;
- `classifier_head.py` — deterministic frozen-embedding linear classifier;
- `run_part4.py` — embedding cache, regularization selection, verification and
  trained-logit SHAP;
- `run_parts_3_4.bat` — gated Windows runner.

Their results are under `outputs/part3_word_partition/`,
`outputs/part4_frozen_head/`, and `outputs/PARTS_3_4_SUMMARY.md`.

## Part 5 implementation

Part 5 compared maximum evaluation budgets of 500, 1,000 and 2,000 on all 40
development-only stability sentences for both the selected zero-shot prompt model
and the trained frozen-CLIP linear head. It measures rank stability, top-five
overlap, contribution signs, additivity, deterministic repeats, runtime and actual
model evaluations. `run_part5.bat` runs the study and freezes the smallest passing
budget in `configs/frozen_shap_budget_manifest.json`. The Part 5 gate passed, and
the complete numerical results and paired PNG/PDF figures are under
`outputs/part5_shap_budget/`.

## Parts 6 and 7 completion

Part 6 generated complete zero-shot CLIP and frozen-linear-head explanations for
all 500 held-out sentences, including both `NEG` and `POS` outputs and the derived
`POS−NEG` margin. Classification, bootstrap, subgroup, faithfulness, efficiency,
per-sentence and per-word tables are under `outputs/part6_full_evaluation/`.

Part 7 ran the fine-tuned SST-2 BERT model on exactly the same 500 rows with the
same whole-word masker, Partition-SHAP algorithm and maximum budget of 500. It
compares BERT, zero-shot CLIP and the trained CLIP head using prediction agreement,
rank correlation, top-five overlap, sign agreement, normalized deletion metrics,
runtime and error analysis. Results are frozen in
`configs/frozen_part7_results_manifest.json` and saved under
`outputs/part7_final_comparison/`.

Both parts provide five aggregate figures and five preselected case-study figures,
each in PNG and PDF. `run_parts_6_7.bat` reproduces the computational stages when
the local model files are available; `src/finalize_part7.py` validates and freezes
an already-computed run without recomputing SHAP.

