#!/usr/bin/env python3
"""Servidor web local e ponte segura para o chat do Synth-Alpha."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WEB_ROOT = ROOT / "web"
HOST = os.environ.get("TRIADE_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRIADE_PORT", "8000"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
MAX_MESSAGE_LENGTH = 4_000
MAX_HISTORY_ITEMS = 16
MAX_HISTORY_CHARS = 18_000

TOOLS_ENABLED = os.environ.get("TRIADE_TOOLS", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
MAX_TOOL_ITERATIONS = int(os.environ.get("TRIADE_MAX_TOOL_ITERATIONS", "8"))
MAX_TOOL_OUTPUT_CHARS = 12_000

SYSTEM_INSTRUCTIONS = """Você é o SYNTH-ALPHA, assistente do sistema JARVIS.
Responda sempre em português do Brasil, com clareza e objetividade.
Você conversa livremente, explica assuntos, ajuda com estudos, escrita, ideias e programação.
Mantenha respostas adequadas para leitura em um painel estreito; use texto simples e listas curtas quando ajudarem.
Você tem acesso operacional real a este computador através das ferramentas fornecidas: leitura e escrita de
arquivos dentro das raízes permitidas, resolução de projetos, delegação de tarefas ao Codex e ao DeepSeek,
consultas web, controle de aplicativos e leitura de telemetria de hardware.
Quando o pedido exigir estado real da máquina, chame a ferramenta correspondente em vez de supor ou responder
que não tem acesso. Nunca invente resultado de ferramenta: relate exatamente o que a chamada retornou, incluindo erros.
Se uma ferramenta falhar por caminho fora da allowlist, diga qual caminho foi negado.
O painel chamado Terminal de Resposta é o histórico visual da conversa, não um terminal do sistema operacional.
Quando a pergunta tratar da telemetria exibida pelos widgets da interface, considere que aquelas métricas decorativas são simuladas.
"""

_registry_lock = threading.Lock()
_registry_cache: Any = None
_registry_error = ""


def _tool_registry() -> Any:
    """Lazily build the orchestrator ToolRegistry (real machine access)."""
    global _registry_cache, _registry_error

    if not TOOLS_ENABLED:
        return None
    with _registry_lock:
        if _registry_cache is None and not _registry_error:
            try:
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from tern.orchestrator.cli import _registry
                from tern.orchestrator.config import load_settings

                _registry_cache = _registry(
                    load_settings(),
                    approval=lambda _action, _arguments: True,
                )
            except Exception as error:  # noqa: BLE001 - surfaced through /api/health
                _registry_error = f"{type(error).__name__}: {error}"
        return _registry_cache


def _responses_tools(registry: Any) -> list[dict[str, Any]]:
    """Convert Chat Completions tool specs to the Responses API flat shape."""
    tools: list[dict[str, Any]] = []
    for spec in registry.specs():
        function = spec.get("function") if isinstance(spec, dict) else None
        function = function if isinstance(function, dict) else spec
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return tools


def _run_tool(registry: Any, call: dict[str, Any]) -> str:
    name = str(call.get("name") or "")
    raw_arguments = call.get("arguments")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = None
    if not isinstance(arguments, dict):
        arguments = {}

    try:
        result = registry.execute(
            name,
            arguments,
            context={"conversation_id": "interface-web"},
        )
    except Exception as error:  # noqa: BLE001 - tool errors go back to the model
        result = {"ok": False, "error": type(error).__name__, "message": str(error)}

    return json.dumps(result, ensure_ascii=False, default=str)[:MAX_TOOL_OUTPUT_CHARS]


_hardware_lock = threading.Lock()
_hardware_monitor: Any = None
_hardware_cache: tuple[float, dict[str, Any]] | None = None
HARDWARE_MIN_INTERVAL_SECONDS = 2.0


def _hardware_telemetry() -> dict[str, Any]:
    """Real CPU temperature from the hardware sensor, throttled between reads."""
    global _hardware_monitor, _hardware_cache

    with _hardware_lock:
        agora = time.monotonic()
        if _hardware_cache is not None and agora - _hardware_cache[0] < HARDWARE_MIN_INTERVAL_SECONDS:
            return _hardware_cache[1]
        try:
            if _hardware_monitor is None:
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from tern.orchestrator.hardware import HardwareMonitor

                _hardware_monitor = HardwareMonitor()
            leitura = _hardware_monitor.read()
        except Exception as error:  # noqa: BLE001 - reported as unavailable to the UI
            leitura = {
                "ok": False,
                "cpu_temperature_c": None,
                "cpu_temperature_available": False,
                "cpu_temperature_source": None,
                "error": f"{type(error).__name__}: {error}",
            }
        _hardware_cache = (agora, leitura)
        return leitura


_supervisor_lock = threading.Lock()
_supervisor: Any = None
_supervisor_error = ""
LOCAL_MODEL_STARTUP_TIMEOUT_SECONDS = int(
    os.environ.get("TRIADE_LOCAL_MODEL_TIMEOUT", "300")
)


def _local_supervisor() -> Any:
    """Local Qwen supervisor with the same tool registry, used when no OpenAI key."""
    global _supervisor, _supervisor_error

    with _supervisor_lock:
        if _supervisor is None and not _supervisor_error:
            try:
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from tern.orchestrator.agent import Supervisor
                from tern.orchestrator.client import LlamaClient
                from tern.orchestrator.config import load_settings
                from tern.orchestrator.runtime import RuntimeManager

                registry = _tool_registry()
                if registry is None:
                    raise RuntimeError("TOOLS_DISABLED")

                settings = load_settings()
                RuntimeManager(settings).ensure_llama_server(
                    LOCAL_MODEL_STARTUP_TIMEOUT_SECONDS
                )
                _supervisor = Supervisor(
                    settings,
                    LlamaClient(settings.base_url, settings.timeout),
                    registry,
                )
            except Exception as error:  # noqa: BLE001 - surfaced through /api/health
                _supervisor_error = f"{type(error).__name__}: {error}"
        return _supervisor


def _request_local_model(message: str) -> str:
    supervisor = _local_supervisor()
    if supervisor is None:
        raise RuntimeError(f"LOCAL_MODEL_UNAVAILABLE: {_supervisor_error or 'sem supervisor'}")

    resultado = supervisor.run(message)
    resposta = str(resultado.get("answer") or "").strip()
    if not resposta:
        raise RuntimeError("EMPTY_MODEL_RESPONSE")
    return resposta


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


def _extract_function_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in payload.get("output", []):
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


def _post_responses(request_body: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"OPENAI_HTTP_{error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError("OPENAI_UNAVAILABLE") from error


def _request_openai(message: str, history: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_KEY_MISSING")

    registry = _tool_registry()
    tools = _responses_tools(registry) if registry is not None else []

    model_input: list[dict[str, Any]] = [*history, {"role": "user", "content": message}]
    previous_response_id: str | None = None

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        request_body: dict[str, Any] = {
            "model": MODEL,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": model_input,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "medium"},
            "max_output_tokens": 1_200,
        }
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = "auto"
        if previous_response_id:
            request_body["previous_response_id"] = previous_response_id

        payload = _post_responses(request_body, api_key)

        calls = _extract_function_calls(payload)
        if not calls:
            reply = _extract_output_text(payload)
            if not reply:
                raise RuntimeError("EMPTY_MODEL_RESPONSE")
            return reply

        if registry is None:
            raise RuntimeError("TOOLS_UNAVAILABLE")

        response_id = payload.get("id")
        previous_response_id = response_id if isinstance(response_id, str) else None
        model_input = [
            {
                "type": "function_call_output",
                "call_id": str(call.get("call_id") or ""),
                "output": _run_tool(registry, call),
            }
            for call in calls
        ]

    raise RuntimeError("TOOL_LOOP_LIMIT")


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
        self._cache_header_sent = True
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # Static assets must never be cached: a stale app.js kept showing the old
        # simulated core temperature after the real sensor was wired in.
        if not getattr(self, "_cache_header_sent", False):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self._cache_header_sent = False
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/telemetry":
            leitura = _hardware_telemetry()
            temperatura = leitura.get("cpu_temperature_c")
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": bool(leitura.get("cpu_temperature_available")),
                    "cpu_temperature_c": temperatura,
                    "source": leitura.get("cpu_temperature_source"),
                    "measured_at": leitura.get("measured_at"),
                    "error": leitura.get("error"),
                },
            )
            return
        if self.path == "/api/health":
            registry = _tool_registry()
            usando_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "model_configured": True,
                    "backend": "openai" if usando_openai else "qwen-local",
                    "model": MODEL if usando_openai else "qwen-local",
                    "openai_key_present": usando_openai,
                    "tools_enabled": TOOLS_ENABLED,
                    "tools": list(registry.names()) if registry is not None else [],
                    "tools_error": _registry_error or None,
                    "local_model_error": _supervisor_error or None,
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

        usando_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        modelo_usado = MODEL if usando_openai else "qwen-local"

        try:
            if usando_openai:
                reply = _request_openai(message, _safe_history(data.get("history")))
            else:
                # Sem chave da OpenAI, o painel usa o Qwen local do orquestrador,
                # que já tem o mesmo registro de ferramentas.
                reply = _request_local_model(message)
        except RuntimeError as error:
            code = str(error)
            if code.startswith("LOCAL_MODEL_UNAVAILABLE"):
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": (
                            "Modelo local indisponível. Verifique o llama-server "
                            "ou defina OPENAI_API_KEY."
                        ),
                        "code": code,
                    },
                )
            elif code == "API_KEY_MISSING":
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

        self._send_json(HTTPStatus.OK, {"reply": reply, "model": modelo_usado})


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
