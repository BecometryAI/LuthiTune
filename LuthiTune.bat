@echo off
echo ==========================================
echo LuthiTune Workbench Launcher
echo Humane Fine-Tuning Protocol
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import customtkinter" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Launch the application
echo Starting LuthiTune Workbench...
pythonw LuthiTune.pyw

exit /b 0
