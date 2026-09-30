@echo off
setlocal
rem One double-click: check the RFEM environment, run the tests, run the
rem full I1-I10 experiment. Everything is logged to outputs\logs\.
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "RUN_NAME=clip_vision_i1_i10"
if not "%~1"=="" set "RUN_NAME=%~1"
set "LOG_DIR=%SCRIPT_DIR%outputs\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG=%LOG_DIR%\%RUN_NAME%.log"

call "%SCRIPT_DIR%find_python.bat"
if not defined PYTHON_EXE (
    echo Could not find the conda environment RFEM. > "%LOG%"
    type "%LOG%"
    pause
    exit /b 1
)

echo [1/4] Environment: %PYTHON_EXE%
echo PYTHON_EXE=%PYTHON_EXE% > "%LOG%"
"%PYTHON_EXE%" -c "import importlib.util as u; missing=[m for m in ('torch','transformers','shap','scipy','matplotlib','PIL') if u.find_spec(m) is None]; print('missing:', ' '.join(missing) if missing else 'none')" >> "%LOG%" 2>&1

"%PYTHON_EXE%" -c "import shap, scipy" >nul 2>nul
if errorlevel 1 (
    echo [2/4] Installing missing SHAP/SciPy into RFEM ...
    "%PYTHON_EXE%" -m pip install "shap>=0.46" "scipy>=1.10" >> "%LOG%" 2>&1
) else (
    echo [2/4] SHAP and SciPy already installed.
)
"%PYTHON_EXE%" -c "import platform, torch, transformers, shap, numpy, scipy, matplotlib; print('versions: python', platform.python_version(), '| torch', torch.__version__, '| transformers', transformers.__version__, '| shap', shap.__version__, '| numpy', numpy.__version__)" >> "%LOG%" 2>&1

echo [3/4] Running tests ...
echo. >> "%LOG%"
echo ===== TESTS ===== >> "%LOG%"
"%PYTHON_EXE%" -m unittest discover -s tests -v >> "%LOG%" 2>&1

echo [4/4] Running CLIP Vision SHAP on I1-I10 (about 10-20 minutes) ...
echo. >> "%LOG%"
echo ===== EXPERIMENT ===== >> "%LOG%"
"%PYTHON_EXE%" "%SCRIPT_DIR%src\run_shap.py" --run-name %RUN_NAME% >> "%LOG%" 2>&1
if errorlevel 1 (
    echo FAILED - see %LOG%
) else (
    echo DONE - results in outputs\values, outputs\plots, outputs\evaluation\%RUN_NAME%
)
echo Log: %LOG%
pause
