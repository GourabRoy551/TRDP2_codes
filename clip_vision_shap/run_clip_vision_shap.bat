@echo off
setlocal
rem CLIP Vision Partition SHAP launcher (conda environment RFEM).
rem Extra arguments go to src\run_shap.py, e.g.
rem     run_clip_vision_shap.bat --run-name clip_vision_i1_i10
set "SCRIPT_DIR=%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
call "%SCRIPT_DIR%find_python.bat"
if not defined PYTHON_EXE (
    echo Could not find the conda environment RFEM.
    echo Set CLIP_VISION_SHAP_PYTHON to its python.exe and run again.
    exit /b 1
)
echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" "%SCRIPT_DIR%src\run_shap.py" %*
exit /b %ERRORLEVEL%
