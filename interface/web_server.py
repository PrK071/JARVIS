#!/usr/bin/env python3
"""Servidor web local e ponte segura para o chat do Synth-Alpha."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
HOST = os.environ.get("TRIADE_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRIADE_PORT", "8000"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
MAX_MESSAGE_LENGTH = 4_000
MAX_HISTORY_ITEMS = 16
MAX_HISTORY_CHARS = 18_000

SYSTEM_INSTRUCTIONS = """Você é o SYNTH-ALPHA, assistente do sistema JARVIS.
Responda sempre em português do Brasil, com clareza e objetividade.
Você conversa livremente, explica assuntos, ajuda com estudos, escrita, ideias e programação.
Mantenha respostas adequadas para leitura em um painel estreito; use texto simples e listas curtas quando ajudarem.
Não afirme ter executado comandos, alterado arquivos, acessado o computador ou consultado dados externos.
O painel chamado Terminal de Resposta é apenas o histórico visual da conversa, não um terminal do sistema operacional.
Quando a pergunta tratar da telemetria do JARVIS, considere que as métricas mostradas pela interface são simuladas.
"""


def _safe_history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    result: list[dict[str, str]] = []
    total_chars = 0
    for item in value[-MAX_HISTORY_ITEMS:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()[:MAX_MESSAGE_LENGTH]
        if not content or total_chars + len(content) > MAX_HISTORY_CHARS:
            continue
        result.append({"role": role, "content": content})
        total_chars += len(content)
    return result


def _extract_output_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _request_openai(message: str, history: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_KEY_MISSING")

    model_input: list[dict[str, str]] = [*history, {"role": "user", "content": message}]
    request_body = json.dumps(
        {
            "model": MODEL,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": model_input,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "medium"},
            "max_output_tokens": 1_200,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"OPENAI_HTTP_{error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError("OPENAI_UNAVAILABLE") from error

    reply = _extract_output_text(payload)
    if not reply:
        raise RuntimeError("EMPTY_MODEL_RESPONSE")
    return reply


class TRIADEWebHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "model_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                    "model": MODEL,
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint não encontrado."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 80_000:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Requisição inválida."})
            return

        try:
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "JSON inválido."})
            return

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, str) or not message.strip():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Envie uma mensagem."})
            return
        message = message.strip()
        if len(message) > MAX_MESSAGE_LENGTH:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"A mensagem deve ter no máximo {MAX_MESSAGE_LENGTH} caracteres."},
            )
            return

        try:
            reply = _request_openai(message, _safe_history(data.get("history")))
        except RuntimeError as error:
            code = str(error)
            if code == "API_KEY_MISSING":
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": "Modelo não configurado. Defina OPENAI_API_KEY e reinicie o servidor.",
                        "code": code,
                    },
                )
            else:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "Não foi possível obter uma resposta do modelo.", "code": code},
                )
            return

        self._send_json(HTTPStatus.OK, {"reply": reply, "model": MODEL})


def _interface_is_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return response.status == HTTPStatus.OK.value and payload.get("ok") is True


def main(*, open_browser: bool = False) -> None:
    browser_host = "127.0.0.1" if HOST in {"0.0.0.0", "::"} else HOST
    url = f"http://{browser_host}:{PORT}"
    if _interface_is_running(url):
        print(f"JARVIS web já está disponível em {url}")
        if open_browser:
            webbrowser.open(url)
        return

    server = ThreadingHTTPServer((HOST, PORT), TRIADEWebHandler)
    print(f"JARVIS web disponível em http://{HOST}:{PORT}")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("Aviso: OPENAI_API_KEY não configurada; comandos locais funcionam, conversa livre exige a chave.")
    if open_browser:
        opener = threading.Timer(0.35, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
