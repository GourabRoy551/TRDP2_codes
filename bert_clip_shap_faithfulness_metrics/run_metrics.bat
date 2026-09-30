@echo off
rem Comprehensiveness, sufficiency, deletion AOPC and rationale F1 for the existing
rem BERT / CLIP-text whole-word POS-NEG margin SHAP (500 sentences), then the tests.
rem Reads ..\bert_clip_partition_shap_500 read-only; writes only to outputs\.
setlocal
set "PROJECT_ROOT=%~dp0"
if defined RFEM_PYTHON (
    set "PYTHON_EXE=%RFEM_PYTHON%"
) else (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\rfem\python.exe"
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTHONHASHSEED=42"
set "PYTHONDONTWRITEBYTECODE=1"

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" src\run_metrics.py || exit /b 1
"%PYTHON_EXE%" -m pytest tests -p no:cacheprovider -q || exit /b 1
endlocal
