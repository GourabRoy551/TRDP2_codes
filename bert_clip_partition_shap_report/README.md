# BERT--CLIP Partition SHAP report

This folder contains a concise, standalone report based on the completed
`bert_clip_partition_shap_500` experiment. The original report and experiment
outputs are not modified.

## Contents

- `report.tex` -- editable LaTeX source.
- `figures/` -- four selected vector figures copied from the completed experiment.
- `output/pdf/BERT_CLIP_PARTITION_SHAP_REPORT.pdf` -- compiled report.

## Data used in the report

All numbers come from the sibling experiment folder:

```text
../bert_clip_partition_shap_500/
```

The report uses the frozen 500-sentence SST-2 evaluation set, the saved BERT and
CLIP Partition SHAP explanations, and the final CSV metric tables. No model was
retrained and no experiment was rerun for the report.

## Rebuilding

Open `report.tex` in the Codex LaTeX editor to compile it with the built-in
compiler. The figure paths are relative, so the report folder can be moved as a
single unit.
