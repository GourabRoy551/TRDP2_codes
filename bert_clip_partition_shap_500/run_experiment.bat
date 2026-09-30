@echo off
rem BERT vs CLIP whole-word Partition SHAP (500 sentences), end to end.
rem Every stage is gated: a failing check stops the pipeline before later stages.
rem The SHAP stage resumes from outputs\checkpoints\*.jsonl if interrupted.
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
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTHONHASHSEED=42"

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" src\run_experiment.py --stage prepare || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage score || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage smoke || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage shap --model bert || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage shap --model clip || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage metrics || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage plots || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage wordbars || exit /b 1
"%PYTHON_EXE%" src\run_experiment.py --stage report || exit /b 1
"%PYTHON_EXE%" -m pytest tests -p no:cacheprovider -q || exit /b 1
endlocal
