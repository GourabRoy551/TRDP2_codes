@echo off
setlocal

set "EXPERIMENT_DIR=%~dp0"
set "BERT_SHAP_DIR=%EXPERIMENT_DIR%.."
set "LEGACY_PYTHON=D:\Research\TRDP_Study Note\TRDP1\trdp\sst2_quantitative_analysis\.venv\Scripts\python.exe"

if exist "%BERT_SHAP_DIR%\.packages" (
    set "PYTHONPATH=%BERT_SHAP_DIR%\.packages;%PYTHONPATH%"
)

if defined BERT_SHAP_PYTHON (
    set "PYTHON_EXE=%BERT_SHAP_PYTHON%"
) else if exist "%LEGACY_PYTHON%" (
    set "PYTHON_EXE=%LEGACY_PYTHON%"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%EXPERIMENT_DIR%src\run_dual_class_shap.py" %*
exit /b %ERRORLEVEL%
