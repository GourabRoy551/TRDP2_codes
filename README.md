# TRDP II: explaining BERT and CLIP with Partition SHAP

Code for the TRDP II project. It covers Partition SHAP explanations of a BERT sentiment
classifier, of the CLIP text encoder used as a zero-shot sentiment classifier, and of the CLIP
vision encoder. It also covers the evaluation of these explanations with faithfulness and
plausibility metrics.

The project folders contain **code, configurations, documentation and small input data
only**. Generated outputs are not included there: SHAP values, plots, PDFs, checkpoints and
logs are recreated by running the projects, and images and large datasets are also left
out. The exception is `final_report/`, which is kept complete. It holds the collected
results and figures used in the final report, and the compiled report PDF.

## Projects

| Folder | What it does |
|---|---|
| `bert_shap/` | Partition SHAP for the BERT SST-2 classifier (POS−NEG margin) on the ten qualitative sentences S1–S10, with word aggregation and faithfulness; `academic_report/` holds the LaTeX of the first report |
| `bert_shap/dual_class_bert/` | the same BERT explanation for the NEG and POS outputs separately |
| `clip_text_shap/` | CLIP text encoder as a zero-shot sentiment classifier (prompt prototypes), explained for both classes on S1–S10 |
| `clip_vision_shap/` | Partition SHAP for the CLIP vision encoder, 7×7 patch values for images I1–I10 |
| `bert_clip_partition_shap_500/` | whole-word Partition SHAP for BERT and CLIP text on 500 balanced SST-2 sentences |
| `bert_clip_shap_faithfulness_metrics/` | comprehensiveness, sufficiency, deletion AOPC and token-level rationale P/R/F1 (SST lexical proxy) for the 500-sentence explanations |
| `clip_vision_shap_faithfulness_metrics/` | deletion AUC, insertion AUC, AOPC and pointing game for the CLIP vision explanations, including the author-annotated target boxes |
| `bert_clip_partition_shap_report/` | LaTeX report of the 500-sentence experiment |
| `final_report/` | everything used for the final report: metric results (`01_`, `05_`), all S1–S10 and I1–I10 plots (`02_`–`04_`), `MANIFEST.csv`, the collect script, and `report2/` with the LaTeX source and the compiled PDF `TRDP2_SHAP_Report.pdf` |

Every project has its own `README.md` with the method, settings, checks and run commands.

## Running

The projects were run with Python 3.11/3.12, PyTorch 2.12–2.14, Transformers 5.x and SHAP
0.51–0.52 in a conda environment. The exact versions are in each project's
`requirements.txt`. Models are loaded from the Hugging Face cache:
`textattack/bert-base-uncased-SST-2` and `openai/clip-vit-base-patch32`. Each project has a
`run_*.bat` launcher and a `tests/` folder.

Some configuration files contain absolute local paths, for example to the original SST-2
files or to the Python interpreter. Adjust them to your machine before running. The ten
images I1–I10 are not included; `clip_vision_shap/data/images.csv` lists them with their
SHA-256 hashes.

The 500-sentence evaluation set and the CLIP prompt family were prepared in an earlier
project, `clip_text_shap_improved`, which is not part of this repository. The frozen
results are included (`bert_clip_partition_shap_500/data/evaluation_500.csv` and the
prompts in its `config.json`). However, the experiment verifies them against files of that
project at run time, so the stages that need them (`prepare`, `score`, `wordbars`) and the
CLIP prompt check in `bert_clip_shap_faithfulness_metrics` only run where that project is
available.

The metric projects read the stored results of the original experiments and never modify
them, so run the underlying experiment first:
- `bert_clip_partition_shap_500` must run before `bert_clip_shap_faithfulness_metrics`;
- `clip_vision_shap` must run before `clip_vision_shap_faithfulness_metrics`.
