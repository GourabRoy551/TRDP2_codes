@echo off
setlocal

set "REPORT_DIR=%~dp0"
pushd "%REPORT_DIR%"

python scripts\generate_tables.py || goto :error
if not exist "output\pdf" mkdir "output\pdf"

for /L %%G in (1,1,3) do (
    pdflatex -jobname=BERT_SHAP_Academic_Report -interaction=nonstopmode -halt-on-error -output-directory=output\pdf main.tex || goto :error
)

echo Report written to output\pdf\BERT_SHAP_Academic_Report.pdf
popd
exit /b 0

:error
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
