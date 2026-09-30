@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=D:\Research\TRDP_Study Note\TRDP1\trdp\sst2_quantitative_analysis\.venv\Scripts\python.exe"
set "PYTHONPATH=D:\ALL_Uni_Documents\UB\Courses\TRDP2\bert_shap\.packages;%PROJECT_ROOT%src"
set "MPLBACKEND=Agg"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" src\run_part6.py || exit /b 1
"%PYTHON_EXE%" src\run_part7.py || exit /b 1
"%PYTHON_EXE%" -m unittest discover -s tests -p "test_*.py" || exit /b 1
endlocal
