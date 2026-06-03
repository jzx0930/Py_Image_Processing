@echo off
title Image Processing Tool
echo ========================================
echo  Image Processing Tool - PyQt6
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10 or above:
    echo https://www.python.org/downloads/
    echo Make sure to check Add Python to PATH
    pause
    exit /b 1
)

echo [CHECK] Checking installed packages...

set MISSING=0
python -c "import PyQt6" >nul 2>&1
if %errorlevel% neq 0 set MISSING=1
python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 set MISSING=1
python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 set MISSING=1

if %MISSING%==0 (
    echo [CHECK] All packages installed. Skipping.
    goto RUN
)

echo [INSTALL] Installing missing packages...
pip install PyQt6 opencv-python numpy
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed. Check internet connection.
    pause
    exit /b 1
)
echo [INSTALL] Done.

:RUN
echo.
echo [START] Launching program...
echo.
cd /d "%~dp0Image_Processing"
python ui_main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program failed to start.
    pause
)
