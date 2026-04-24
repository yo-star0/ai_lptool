@echo off
setlocal

echo ================================================
echo  lp2psd setup
echo ================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add to PATH.
    exit /b 1
)

if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing CPU requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    exit /b 1
)

echo.
echo ================================================
echo  Setup completed.
echo  To enable CUDA, follow requirements-gpu.txt
echo  To launch the app, run: run.bat
echo ================================================
endlocal
