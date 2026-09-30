# BERT SHAP academic report

This folder contains the editable LaTeX source, generated result tables, copied
experiment figures, and the compiled PDF report.

## Contents

- `main.tex` - complete report source;
- `figures/` - the eleven PDF plots used by the report;
- `tables/` - LaTeX tables generated from the verified experiment outputs;
- `scripts/generate_tables.py` - regenerates tables and result macros from the
  `modular_s1_s10` CSV and JSON files;
- `build_report.bat` - regenerates tables and compiles the PDF;
- `output/pdf/BERT_SHAP_Academic_Report.pdf` - final report.

## Rebuild

From this folder:

```bat
build_report.bat
```

The build requires Python and a LaTeX installation providing `pdflatex`.
Run the command three times internally so that the table of contents, citations,
figure references, and long-table widths are resolved.
