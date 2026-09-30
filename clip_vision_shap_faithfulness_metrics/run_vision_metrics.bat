@echo off
rem Deletion AUC, insertion AUC, AOPC and pointing game for the existing CLIP vision
rem patch-level SHAP (I1-I10), then the tests. Reads ..\clip_vision_shap read-only.
setlocal
set "PROJECT_ROOT=%~dp0"
if defined RFEM_PYTHON (
    set "PYTHON_EXE=%RFEM_PYTHON%"
) else (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\rfem\python.exe"
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "MPLBACKEND=Agg"
set "PYTHONDONTWRITEBYTECODE=1"

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" src\run_vision_metrics.py || exit /b 1
"%PYTHON_EXE%" -m pytest tests -p no:cacheprovider -q || exit /b 1
endlocal
