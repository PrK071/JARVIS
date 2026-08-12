@echo off
cd /d "%~dp0"
python main.py 2>nul || python3 main.py 2>nul || py -3 main.py
if errorlevel 1 (
    echo.
    echo Python nao encontrado. Instale Python 3.10+ em https://python.org
    echo Certifique-se de marcar "Add Python to PATH" na instalacao.
    pause
)
