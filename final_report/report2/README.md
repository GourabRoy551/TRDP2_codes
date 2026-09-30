# TRDP II report (LaTeX)

- `main.tex`: report source.
- `TRDP2_SHAP_Report.pdf`: compiled report.
- `build_report.bat`: rebuilds the PDF (MiKTeX `latexmk` + `pdflatex`). The build files go to
  `build/`.

All figures are read directly from the sibling folders `../01_…` to `../05_…`, so the PDF always
uses the current report materials. If you re-collect the materials with
`../collect_report_materials.py`, run `build_report.bat` again.
