@echo off
TITLE Cortex Bet - Dashboard Launcher
COLOR 0B

echo ====================================================
echo      CORTEX BET - INITIALIZING...
echo ====================================================

:: Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"

:: Build path to venv Python
set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"

:: Debug: show the detected path
echo [DEBUG] Project Root: %PROJECT_ROOT%
echo [DEBUG] Python Exe: %PYTHON_EXE%

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment not found at:
    echo         %PYTHON_EXE%
    echo.
    echo Trying to find Python in common locations...
    if exist "C:\Users\Valmont\Documents\GitHub\Cortex_Bet\Cortex_Bet\.venv\Scripts\python.exe" (
        echo [FOUND] Using alternative path
        set "PYTHON_EXE=C:\Users\Valmont\Documents\GitHub\Cortex_Bet\Cortex_Bet\.venv\Scripts\python.exe"
        set "PROJECT_ROOT=C:\Users\Valmont\Documents\GitHub\Cortex_Bet\Cortex_Bet"
    ) else (
        pause
        exit /b 1
    )
)

echo [1/1] Starting unified local stack...
cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" start_system.py

pause
