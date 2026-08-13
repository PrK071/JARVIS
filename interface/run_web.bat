@echo off
cd /d "%~dp0"
echo Iniciando T.R.I.A.D.E Web em http://localhost:8000
echo Pressione Ctrl+C para encerrar.
where python >nul 2>nul && (python web_server.py & goto :eof)
where python3 >nul 2>nul && (python3 web_server.py & goto :eof)
where py >nul 2>nul && (py -3 web_server.py & goto :eof)
echo.
echo Python nao encontrado. Instale Python 3.10+ em https://python.org
echo Certifique-se de marcar "Add Python to PATH" na instalacao.
pause
