@echo off
rem Install the Python packages V7 Segmenter needs. Run once per computer.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -m pip install -r requirements.txt
) else (
    python -m pip install -r requirements.txt
)
echo.
echo Done. Start the application with "Start V7 GUI.bat".
pause
