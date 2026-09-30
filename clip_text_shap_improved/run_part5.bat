@echo off
setlocal

set "EXPERIMENT_DIR=%~dp0"
set "PYTHON_EXE=D:\Research\TRDP_Study Note\TRDP1\trdp\sst2_quantitative_analysis\.venv\Scripts\python.exe"
set "EXTRA_PACKAGES=%EXPERIMENT_DIR%..\bert_shap\.packages"

set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "PYTHONPATH=%EXTRA_PACKAGES%;%EXPERIMENT_DIR%src;%PYTHONPATH%"

"%PYTHON_EXE%" "%EXPERIMENT_DIR%src\run_part5.py"
exit /b %ERRORLEVEL%
