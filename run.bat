@echo off
echo =======================================================
echo Intelligent Dataset Quality ^& Auto-Cleaning System
echo =======================================================
echo.

echo [1/2] Starting Ollama server in the background...
start /B "" ollama serve >nul 2>&1

:: Wait a brief moment for the Ollama service to initialize
timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit Application...
python -m streamlit run app.py
