@echo off
cd /d "%~dp0"
echo Iniciando JARVIS Web em http://localhost:8000
echo Pressione Ctrl+C para encerrar.

rem O venv do projeto tem faster-whisper, PyAV e o pacote tern: sem ele o
rem microfone nao consegue transcrever localmente.
if exist "%~dp0..\.venv\Scripts\python.exe" (
  "%~dp0..\.venv\Scripts\python.exe" web_server.py
  goto :eof
)

echo.
echo Aviso: venv do projeto nao encontrado em ..\.venv
echo O microfone precisa dele. Para criar:
echo     python -m venv .venv
echo     .venv\Scripts\python -m pip install --editable .
echo.
where python >nul 2>nul && (python web_server.py & goto :eof)
where python3 >nul 2>nul && (python3 web_server.py & goto :eof)
where py >nul 2>nul && (py -3 web_server.py & goto :eof)
echo.
echo Python nao encontrado. Instale Python 3.10+ em https://python.org
echo Certifique-se de marcar "Add Python to PATH" na instalacao.
pause
