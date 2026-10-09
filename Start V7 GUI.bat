@echo off
rem Start V7 Segmenter without a console window.
rem If nothing appears: run "python run_gui.py" in this folder to see errors, or read logs\v7_segmenter.log.
cd /d "%~dp0"
set "PYW=%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe"
if exist "%PYW%" (
    start "" "%PYW%" "%~dp0run_gui.py" %*
    exit /b
)
where pyw >nul 2>nul
if %errorlevel%==0 (
    start "" pyw -3.11 "%~dp0run_gui.py" %*
    exit /b
)
start "" pythonw "%~dp0run_gui.py" %*
