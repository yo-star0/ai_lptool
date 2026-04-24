@echo off
setlocal

if not exist .venv (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py %*

endlocal
