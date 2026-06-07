@echo off
echo ============================================
echo  3rd Year Project - Windows Setup Script
echo ============================================
echo.

:: Check if Python 3.11 is available
echo [1/6] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 not found.
    echo Please download and install it from:
    echo https://www.python.org/downloads/release/python-3117/
    echo Make sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)
py -3.11 --version
echo Python 3.11 found.
echo.

:: Check if ffmpeg is available
echo [2/6] Checking ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg not found in PATH.
    echo Please download ffmpeg from https://www.gyan.dev/ffmpeg/builds/
    echo and add the bin folder to your system PATH.
    echo.
    echo You can continue setup now and add ffmpeg later,
    echo but you will need it before running transcribe.py
    echo.
    pause
) else (
    echo ffmpeg found.
)
echo.

:: Create virtual environment
echo [3/6] Creating virtual environment with Python 3.11...
if exist venv (
    echo Virtual environment already exists, skipping.
) else (
    py -3.11 -m venv venv
    echo Virtual environment created.
)
echo.

:: Activate and install torch CPU
echo [4/6] Installing CPU-only PyTorch...
call venv\Scripts\activate.bat

:: Remove problematic lines from requirements.txt
echo Cleaning requirements.txt for Windows...
powershell -Command "(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^(torch|nvidia|triton|audioop-lts)' } | Set-Content requirements_win.txt"
echo Created requirements_win.txt

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 ^
  --index-url https://download.pytorch.org/whl/cpu ^
  --trusted-host download.pytorch.org ^
  --trusted-host download-r2.pytorch.org ^
  --trusted-host files.pythonhosted.org ^
  --trusted-host pypi.org ^
  --quiet
echo PyTorch CPU installed.
echo.

:: Install remaining dependencies
echo [5/6] Installing remaining dependencies...
pip install -r requirements_win.txt ^
  --trusted-host pypi.org ^
  --trusted-host files.pythonhosted.org ^
  --trusted-host pypi.python.org ^
  --timeout 120 ^
  --quiet
echo Dependencies installed.
echo.

:: Install Whisper
echo [6/6] Installing Whisper...
pip install openai-whisper ^
  --trusted-host pypi.org ^
  --trusted-host files.pythonhosted.org ^
  --trusted-host pypi.python.org ^
  --quiet
echo Whisper installed.
echo.

:: Check for model file
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
if exist personality_model_final.pth (
    echo [OK] personality_model_final.pth found.
) else (
    echo [MISSING] personality_model_final.pth not found.
    echo Please get this file from a teammate and place it in this folder.
)
echo.
echo To run the project:
echo   1. Place your audio file in this folder as audio.wav (must be mono)
echo   2. Convert stereo to mono if needed:
echo      ffmpeg -i your_audio.wav -ac 1 audio.wav
echo   3. Activate the virtual environment:
echo      venv\Scripts\activate
echo   4. Run:
echo      python transcribe.py
echo.
pause
