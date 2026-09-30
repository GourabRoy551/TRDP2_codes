@echo off
rem Sets PYTHON_EXE to the python.exe of the conda environment RFEM.
rem Override by setting CLIP_VISION_SHAP_PYTHON before calling.
set "PYTHON_EXE="
if defined CLIP_VISION_SHAP_PYTHON (
    set "PYTHON_EXE=%CLIP_VISION_SHAP_PYTHON%"
    goto :eof
)
for %%R in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%ProgramData%\anaconda3"
    "%ProgramData%\miniconda3"
    "%USERPROFILE%\.conda"
) do (
    if not defined PYTHON_EXE if exist "%%~R\envs\RFEM\python.exe" set "PYTHON_EXE=%%~R\envs\RFEM\python.exe"
)
if defined PYTHON_EXE goto :eof
where conda >nul 2>nul
if errorlevel 1 goto :eof
for /f "usebackq delims=" %%P in (`conda run -n RFEM python -c "import sys; print(sys.executable)"`) do set "PYTHON_EXE=%%P"
goto :eof
