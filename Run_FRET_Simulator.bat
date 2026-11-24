@echo off
setlocal
title FRET Monte Carlo Simulator
cd /d "%~dp0"

echo ========================================
echo   FRET Monte Carlo Simulator
echo ========================================
echo.
echo Current folder:
echo %cd%
echo.

echo Looking for required files...
if not exist "exifret_gui.py" (
    echo [ERROR] Cannot find exifret_gui.py
    echo Make sure this .bat file is in the same folder as exifret_gui.py
    goto END
)
if not exist "exifret.py" (
    echo [ERROR] Cannot find exifret.py
    echo Make sure this .bat file is in the same folder as exifret.py
    goto END
)
echo Files found.
echo.

echo Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not in PATH.
    echo Install Python from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during setup.
    goto END
)
python --version
echo.

echo Checking packages (numpy, matplotlib)...
python -c "import numpy, matplotlib; print('Packages OK')" 2>errorlog.txt
if errorlevel 1 (
    echo Packages missing. Trying to install...
    python -m pip install numpy matplotlib
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not install packages.
        echo Open Command Prompt and run:
        echo     pip install numpy matplotlib
        type errorlog.txt
        goto END
    )
)
del errorlog.txt >nul 2>&1
echo.

echo Starting the program...
echo (Leave this window open. Minimize it if you want.)
echo.
python exifret_gui.py
set EXITCODE=%errorlevel%

echo.
echo ----------------------------------------
if %EXITCODE% neq 0 (
    echo Program exited with error code %EXITCODE%.
    echo.
    echo Common causes:
    echo  - File names missing .py extension
    echo  - tkinter not installed
    echo  - Another Python error (see messages above)
) else (
    echo Program finished normally.
)

:END
echo.
echo ========================================
echo Press any key to close this window...
pause >nul
endlocal
