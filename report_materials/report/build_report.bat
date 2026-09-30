@echo off
rem Builds the TRDP II report from main.tex (MiKTeX latexmk + pdflatex).
rem Figures are read directly from the report_materials folders (..\01_... to ..\05_...).
setlocal
cd /d "%~dp0"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex || exit /b 1
copy /y build\main.pdf TRDP2_SHAP_Report.pdf >nul
echo Report written to %~dp0TRDP2_SHAP_Report.pdf
endlocal
