@echo off
setlocal

echo ================================================
echo  lp2psd setup
echo ================================================

REM ---- Python launcher (py) を優先し、3.11 → 3.10 の順で探す ----
set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py -3.11"
        goto :py_found
    )
    py -3.10 --version >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py -3.10"
        goto :py_found
    )
)

REM ---- フォールバック: PATH の python が 3.10 or 3.11 か確認 ----
where python >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    echo [INFO] Found python in PATH: %PY_VER%
    echo %PY_VER% | findstr /R "^3\.1[01]\." >nul
    if not errorlevel 1 (
        set "PY_CMD=python"
        goto :py_found
    )
)

echo [ERROR] Python 3.10 or 3.11 not found.
echo         Install Python 3.11 from https://www.python.org/downloads/
echo         and re-run this script. (Python 3.12+ is not yet supported by
echo         PaddlePaddle / PyTorch wheels.)
exit /b 1

:py_found
echo [INFO] Using: %PY_CMD%

if not exist .venv (
    echo [INFO] Creating virtual environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        exit /b 1
    )
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
