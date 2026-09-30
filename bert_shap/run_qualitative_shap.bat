@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%.packages" (
    set "PYTHONPATH=%SCRIPT_DIR%.packages;%PYTHONPATH%"
)

if defined BERT_SHAP_PYTHON (
    set "PYTHON_EXE=%BERT_SHAP_PYTHON%"
) else if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%..\TRDP\sst2_quantitative_analysis\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%..\TRDP\sst2_quantitative_analysis\.venv\Scripts\python.exe"
) else if exist "D:\Research\TRDP_Study Note\TRDP1\trdp\sst2_quantitative_analysis\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=D:\Research\TRDP_Study Note\TRDP1\trdp\sst2_quantitative_analysis\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%SCRIPT_DIR%src\run_shap.py" %*
exit /b %ERRORLEVEL%

