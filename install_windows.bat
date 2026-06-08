@echo off
:: Overslam / Mei Cartographer - Windows dependency installer
:: Run once from the project directory (where mei_cartographer.py lives).
::
:: Manual prerequisites:
::   1. Python 3.12 (PyTorch has no wheels for 3.13/3.14 yet).
::      Download: https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
::      During install: tick "Add python.exe to PATH".
::      Then re-run this script as: py -3.12 install_windows.bat
::   2. Tesseract-OCR: downloaded and installed automatically below.
::   3. NVIDIA GPU optional -- everything works CPU-only.

echo.
echo ===== overslam: Windows dependency install =====
echo.

:: -------------------------------------------------------------------
:: 0. Find Python 3.12. Use py launcher to pin the version so PyTorch
::    wheels are available regardless of which Python is the default.
:: -------------------------------------------------------------------
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 not found. Please install it from:
    echo   https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
    echo Tick "Add python.exe to PATH" during install, then re-run this script.
    pause
    exit /b 1
)
echo Using Python: & py -3.12 --version
set PY=py -3.12

:: -------------------------------------------------------------------
:: 1. Core scientific stack  (pin numpy<2 for cv2/torchvision compat)
:: -------------------------------------------------------------------
echo [1/7] numpy scipy pillow...
%PY% -m pip install --user --upgrade "numpy>=1.24,<2" scipy pillow

:: -------------------------------------------------------------------
:: 2. PyTorch + torchvision
::    CPU-only build (works on any machine; swap cpu->cu121 in the URL
::    below if you have an NVIDIA GPU for faster depth estimation).
:: -------------------------------------------------------------------
echo.
echo [2/7] PyTorch + torchvision (CPU build)...
%PY% -m pip install --user --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: -------------------------------------------------------------------
:: 3. OpenCV (headless -- no GUI required)
:: -------------------------------------------------------------------
echo.
echo [3/7] OpenCV...
%PY% -m pip install --user --upgrade opencv-python-headless

:: -------------------------------------------------------------------
:: 4. Depth-Anything-V2 via HuggingFace transformers (self-contained,
::    weights download automatically on first run ~100 MB).
:: -------------------------------------------------------------------
echo.
echo [4/7] transformers  accelerate  huggingface_hub...
%PY% -m pip install --user --upgrade transformers accelerate huggingface_hub

:: -------------------------------------------------------------------
:: 5. Screen capture + input automation
:: -------------------------------------------------------------------
echo.
echo [5/7] mss  pyautogui  pydirectinput  keyboard...
%PY% -m pip install --user --upgrade mss pyautogui pydirectinput keyboard

:: -------------------------------------------------------------------
:: 6. OCR: pytesseract python wrapper + Tesseract binary
:: -------------------------------------------------------------------
echo.
echo [6/7] pytesseract + Tesseract-OCR binary...
%PY% -m pip install --user --upgrade pytesseract

:: Download and install Tesseract if not already present.
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo   Tesseract already installed.
) else (
    echo   Downloading Tesseract-OCR installer...
    powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe' -OutFile '%TEMP%\tesseract-setup.exe'"
    echo   Running Tesseract installer...
    start /wait "" "%TEMP%\tesseract-setup.exe"
)

:: -------------------------------------------------------------------
:: 7. Verify all key imports
:: -------------------------------------------------------------------
echo.
echo [7/7] Verifying imports...

%PY% -c "import numpy; print('  numpy           ', numpy.__version__)"
%PY% -c "import scipy; print('  scipy           ', scipy.__version__)"
%PY% -c "import PIL; print('  Pillow          ', PIL.__version__)"
%PY% -c "import torch; print('  torch           ', torch.__version__, '  cuda:', torch.cuda.is_available())"
%PY% -c "import torchvision; print('  torchvision     ', torchvision.__version__)"
%PY% -c "import cv2; print('  cv2             ', cv2.__version__)"
%PY% -c "import transformers; print('  transformers    ', transformers.__version__)"
%PY% -c "import mss; print('  mss             OK')"
%PY% -c "import pyautogui; print('  pyautogui       OK')"
%PY% -c "import pydirectinput; print('  pydirectinput   OK')"
%PY% -c "import keyboard; print('  keyboard        OK')"
%PY% -c "import pytesseract; pytesseract.get_tesseract_version(); print('  pytesseract     OK  (tesseract found)')"

echo.
echo ===== Done. =====
echo.
echo Depth-Anything-V2 weights will be downloaded automatically on first
echo run (~100 MB to %%USERPROFILE%%\.cache\huggingface\hub).
echo.
echo Usage examples:
echo.
echo   Gallery collection (all heroes, default skin + emotes):
echo     python capture_bot.py --capture-root D:\captures --missing
echo.
echo   Passive observation mode (you drive, bot records):
echo     python passive_capture.py --capture-root D:\captures
echo.
echo   Mei Cartographer (map a level on foot):
echo     python mei_cartographer.py --map kingsrow --negatives D:\negatives
echo.
echo     On first run it auto-calibrates mouse sensitivity and move speed
echo     by spinning in place and walking forward for ~2s before exploring.
echo     Calibration is saved to maps\kingsrow\calibration.json and reused
echo     on all future runs. To redo it:
echo       python mei_cartographer.py --map kingsrow --recalibrate
echo.
echo   Localize a gameplay screenshot in a saved map:
echo     python mei_cartographer.py --map kingsrow --localize shot.png
echo.
pause
