#!/usr/bin/env python3
"""Servidor web local e ponte segura para o chat do Synth-Alpha."""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
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
# Cobrado quando o modelo esgota as rodadas de ferramentas sem concluir: pedir
# texto final com o material já coletado vale mais que devolver painel vazio.
TOOL_BUDGET_PROMPT = (
    "O limite de chamadas de ferramenta desta pergunta acabou. Responda agora, "
    "em texto, com o que você já apurou, e diga o que ficou sem verificar."
)
# Transcricao local: o audio do microfone nunca sai da maquina. Usa o mesmo
# FasterWhisperSTT do orquestrador (tern), com o modelo ja baixado em
# models/voice/. Sem faster-whisper ou sem modelo, o botao de microfone some
# da interface em vez de falhar no meio da gravacao.
MAX_AUDIO_BYTES = 8_000_000
STT_SAMPLE_RATE = 16_000
MIN_AUDIO_SAMPLES = STT_SAMPLE_RATE // 5  # 200 ms

_stt_lock = threading.Lock()
_stt_state: dict[str, Any] = {"provider": None, "options": None, "checked": False, "ready": False}


def _import_stt() -> tuple[Any, Any, Any, Any]:
    """Import the tern voice stack, adding the repo root to sys.path if needed."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tern.orchestrator.config import load_settings
    from tern.orchestrator.voice.errors import VoiceError
    from tern.orchestrator.voice.models import AudioData, TranscriptionOptions
    from tern.orchestrator.voice.stt import FasterWhisperSTT
    return load_settings, VoiceError, (AudioData, TranscriptionOptions), FasterWhisperSTT


def _stt_ready() -> bool:
    """Check once whether local transcription can run, without loading the model."""
    with _stt_lock:
        if _stt_state["checked"]:
            return bool(_stt_state["ready"])
        _stt_state["checked"] = True
        try:
            load_settings, _, _, _ = _import_stt()
            import av  # noqa: F401  (decodifica o webm/opus do navegador)
            settings = load_settings()
            ready = (
                settings.voice_stt_provider == "faster_whisper"
                and Path(settings.voice_stt_model).is_dir()
            )
        except Exception:
            ready = False
        _stt_state["ready"] = ready
        return ready


def _stt_provider():
    """Load the STT model once and reuse it across requests."""
    with _stt_lock:
        if _stt_state["provider"] is not None:
            return _stt_state["provider"], _stt_state["options"]
        load_settings, _, models, FasterWhisperSTT = _import_stt()
        _, TranscriptionOptions = models
        settings = load_settings()
        provider = FasterWhisperSTT(
            settings.voice_stt_model,
            device=settings.voice_stt_device,
            compute_type=settings.voice_stt_compute_type,
            threads=settings.voice_stt_threads,
        )
        options = TranscriptionOptions(
            language=settings.voice_stt_language,
            timeout_seconds=settings.voice_stt_timeout_seconds,
        )
        _stt_state["provider"] = provider
        _stt_state["options"] = options
        return provider, options


def _decode_audio(raw: bytes):
    """Decode the browser blob (webm/opus, ogg or mp4) to mono float32 at 16 kHz."""
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    resampler = AudioResampler(format="flt", layout="mono", rate=STT_SAMPLE_RATE)
    chunks: list[Any] = []
    try:
        with av.open(io.BytesIO(raw)) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise RuntimeError("AUDIO_NO_STREAM")
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("AUDIO_DECODE_FAILED") from error

    if not chunks:
        raise RuntimeError("AUDIO_EMPTY")
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    if samples.size < MIN_AUDIO_SAMPLES:
        raise RuntimeError("AUDIO_TOO_SHORT")
    return samples


def _transcribe(raw: bytes) -> str:
    """Transcribe a recorded blob locally and return the recognised text."""
    import numpy as np

    _, VoiceError, models, _ = _import_stt()
    AudioData, _ = models
    samples = _decode_audio(raw)
    provider, options = _stt_provider()

    audio = AudioData(
        samples=samples,
        sample_rate=STT_SAMPLE_RATE,
        duration_seconds=float(samples.size) / STT_SAMPLE_RATE,
        rms=float(np.sqrt(np.mean(np.square(samples)))),
        peak=float(np.max(np.abs(samples))),
    )
    try:
        result = provider.transcribe(audio, options)
    except VoiceError as error:
        raise RuntimeError(f"STT_{error.code.upper()}") from error
    return result.text.strip()


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
            "max_output_tokens": MAX_PROVIDER_OUTPUT_TOKENS,
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



# Conexões de IA ------------------------------------------------------------
# O usuário cadastra provedores pela própria interface. A chave é enviada uma
# única vez, gravada aqui no servidor (arquivo fora do git) e nunca devolvida ao
# navegador: as listagens trazem apenas um resumo mascarado. Assim a chave não
# vira conteúdo do front-end, como o README já exigia.
PROVIDERS_PATH = Path(
    os.environ.get("JARVIS_PROVIDERS_FILE", str(ROOT / "providers.json"))
)
PROVIDERS_LOCK = threading.Lock()
MAX_PROVIDERS = 20
# Modelos de raciocínio (deepseek-v4-pro, o-series) gastam parte do orçamento em
# reasoning_content. Com 1200 o texto final chegava vazio e virava
# EMPTY_MODEL_RESPONSE sem explicação; com 4000 ainda estourava em perguntas que
# encadeiam ferramentas. 8000 fica abaixo do teto de saída da DeepSeek (8192).
MAX_PROVIDER_OUTPUT_TOKENS = int(os.environ.get("TRIADE_PROVIDER_MAX_TOKENS", "16000"))

# Formatos cobrem, na prática, qualquer API de IA relevante:
# - openai-chat     : /chat/completions — OpenAI, DeepSeek, Groq, OpenRouter,
#                     Together, Mistral, Ollama, LM Studio, vLLM, llama.cpp
# - openai-responses: /responses — a Responses API da OpenAI
# - anthropic       : /messages — Claude
# O rótulo cita os provedores conhecidos porque o campo descreve o protocolo, não
# a marca: quem escolhe "DeepSeek" no seletor de provedor precisa reconhecer que
# o protocolo dela é o chat/completions.
PROVIDER_FORMATS = {
    "openai-chat": {
        "label": "Chat Completions — DeepSeek, OpenAI, Groq, Mistral, Ollama, LM Studio",
        "base_url": "https://api.openai.com/v1",
        "path": "/chat/completions",
        "vendors": (
            "DeepSeek",
            "OpenAI",
            "Groq",
            "OpenRouter",
            "Mistral",
            "Together",
            "Gemini (modo OpenAI)",
            "Ollama",
            "LM Studio",
            "vLLM",
            "llama.cpp",
        ),
    },
    "openai-responses": {
        "label": "OpenAI Responses — só OpenAI",
        "base_url": "https://api.openai.com/v1",
        "path": "/responses",
        "vendors": ("OpenAI",),
    },
    "anthropic": {
        "label": "Anthropic Messages — Claude",
        "base_url": "https://api.anthropic.com/v1",
        "path": "/messages",
        "vendors": ("Anthropic (Claude)",),
    },
}

# Presets de provedores conhecidos: o erro mais comum é salvar a chave de um
# provedor com o endpoint de outro, que responde 401 sem dizer o motivo.
PROVIDER_PRESETS = (
    {
        "id": "openai",
        "label": "OpenAI",
        "format": "openai-chat",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "format": "openai-chat",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
    },
    {
        "id": "anthropic",
        "label": "Anthropic (Claude)",
        "format": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-5",
    },
    {
        "id": "groq",
        "label": "Groq",
        "format": "openai-chat",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "format": "openai-chat",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
    },
    {
        "id": "mistral",
        "label": "Mistral",
        "format": "openai-chat",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
    },
    {
        "id": "gemini",
        "label": "Google Gemini",
        "format": "openai-chat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
    },
    {
        "id": "ollama",
        "label": "Ollama (local)",
        "format": "openai-chat",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
    },
    {
        "id": "lmstudio",
        "label": "LM Studio (local)",
        "format": "openai-chat",
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
    },
)


def _slug(value: str) -> str:
    cleaned = [c.lower() if c.isalnum() else "-" for c in value.strip()]
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:40] or "conexao"


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * 6}{key[-4:]}"


def _read_providers() -> dict[str, Any]:
    """Load the provider store, tolerating a missing or corrupt file."""
    if not PROVIDERS_PATH.is_file():
        return {"active": None, "providers": []}
    try:
        data = json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"active": None, "providers": []}
    if not isinstance(data, dict):
        return {"active": None, "providers": []}
    providers = data.get("providers")
    if not isinstance(providers, list):
        providers = []
    return {"active": data.get("active"), "providers": providers}


def _write_providers(store: dict[str, Any]) -> None:
    PROVIDERS_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # A chave fica só para o dono do arquivo quando o sistema permite.
    try:
        os.chmod(PROVIDERS_PATH, 0o600)
    except OSError:
        pass


def _public_provider(provider: dict[str, Any]) -> dict[str, Any]:
    """Provider as the browser may see it: never the raw key."""
    return {
        "id": provider.get("id"),
        "label": provider.get("label"),
        "format": provider.get("format"),
        "base_url": provider.get("base_url"),
        "model": provider.get("model"),
        "has_key": bool(provider.get("api_key")),
        "key_hint": _mask_key(str(provider.get("api_key") or "")),
    }


def _env_provider() -> dict[str, Any] | None:
    """Keep the pre-existing OPENAI_API_KEY setup working with no migration."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return {
        "id": "env",
        "label": "Ambiente (OPENAI_API_KEY)",
        "format": "openai-responses",
        "base_url": "https://api.openai.com/v1",
        "model": MODEL,
        "api_key": api_key,
        "from_env": True,
    }


def _all_providers() -> list[dict[str, Any]]:
    providers = list(_read_providers()["providers"])
    env = _env_provider()
    if env and not any(p.get("id") == "env" for p in providers):
        providers.insert(0, env)
    return providers


def _active_provider() -> dict[str, Any] | None:
    store = _read_providers()
    providers = _all_providers()
    if not providers:
        return None
    active_id = store.get("active")
    for provider in providers:
        if provider.get("id") == active_id:
            return provider
    return providers[0]


def _validate_provider(data: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a submitted connection, returning (provider, error)."""
    if not isinstance(data, dict):
        return None, "Dados inválidos."

    label = str(data.get("label") or "").strip()
    if not label or len(label) > 60:
        return None, "Dê um nome de 1 a 60 caracteres para a conexão."

    fmt = str(data.get("format") or "").strip()
    if fmt not in PROVIDER_FORMATS:
        return None, "Formato de API desconhecido."

    base_url = str(data.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        base_url = PROVIDER_FORMATS[fmt]["base_url"]
    # Colar a URL completa do endpoint e o reflexo natural; guardamos so a raiz,
    # como o cliente DeepSeek do orquestrador ja faz.
    for spec in PROVIDER_FORMATS.values():
        if base_url.endswith(spec["path"]):
            base_url = base_url[: -len(spec["path"])].rstrip("/")
            break
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, "O endpoint deve começar com http:// ou https://."

    model = str(data.get("model") or "").strip()
    if not model or len(model) > 120:
        return None, "Informe o nome do modelo."

    api_key = str(data.get("api_key") or "").strip()
    if len(api_key) > 400:
        return None, "Chave longa demais."

    provider_id = str(data.get("id") or "").strip() or _slug(label)
    if provider_id == "env":
        return None, "Esse identificador é reservado para a chave do ambiente."

    return {
        "id": provider_id,
        "label": label,
        "format": fmt,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
    }, None


def _save_provider(data: Any) -> tuple[dict[str, Any] | None, str | None]:
    provider, error = _validate_provider(data)
    if error or provider is None:
        return None, error

    with PROVIDERS_LOCK:
        store = _read_providers()
        providers = store["providers"]
        existing = next(
            (p for p in providers if p.get("id") == provider["id"]), None
        )
        if existing is None and len(providers) >= MAX_PROVIDERS:
            return None, f"Limite de {MAX_PROVIDERS} conexões atingido."

        if existing is not None:
            # Campo de chave em branco na edição: mantém a chave já gravada.
            if not provider["api_key"]:
                provider["api_key"] = existing.get("api_key", "")
            providers[providers.index(existing)] = provider
        else:
            providers.append(provider)

        store["active"] = provider["id"]
        _write_providers(store)
    return provider, None


def _delete_provider(provider_id: str) -> bool:
    with PROVIDERS_LOCK:
        store = _read_providers()
        remaining = [p for p in store["providers"] if p.get("id") != provider_id]
        if len(remaining) == len(store["providers"]):
            return False
        store["providers"] = remaining
        if store.get("active") == provider_id:
            store["active"] = remaining[0]["id"] if remaining else None
        _write_providers(store)
    return True


def _activate_provider(provider_id: str) -> bool:
    if not any(p.get("id") == provider_id for p in _all_providers()):
        return False
    with PROVIDERS_LOCK:
        store = _read_providers()
        store["active"] = provider_id
        _write_providers(store)
    return True


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP_{error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"UNREACHABLE: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("INVALID_JSON_RESPONSE") from error


def _reply_from_openai_chat(payload: dict[str, Any]) -> str:
    for choice in payload.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
    return ""


def _reply_from_anthropic(payload: dict[str, Any]) -> str:
    chunks = [
        block.get("text", "")
        for block in payload.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _provider_error_message(code: str, *, host: str | None = None) -> str:
    """Turn a provider failure into something readable in the narrow panel."""
    onde = f" por {host}" if host else " pelo provedor"
    if code.startswith("API_KEY_MISSING"):
        return "Essa conexão exige uma chave de API."
    if code.startswith("UNREACHABLE"):
        return "Endpoint inacessível. Confira a URL e a conexão."
    if code.startswith("HTTP_401") or code.startswith("HTTP_403"):
        # Nomear quem recusou evita o erro mais comum: chave de um provedor
        # enviada para o endpoint de outro, que responde 401 sem explicar.
        return (
            f"Chave recusada{onde}. Confira se o endpoint corresponde ao provedor "
            "dessa chave."
        )
    if code.startswith("HTTP_404"):
        return "Endpoint não encontrado. Confira a URL base e o formato."
    if code.startswith("HTTP_429"):
        return "Limite de uso atingido no provedor."
    if code.startswith("HTTP_400") and "model" in code.lower():
        return "Modelo inválido para esse provedor. Confira o nome do modelo."
    if code.startswith("HTTP_"):
        return f"O provedor recusou a requisição ({code.split(':')[0]})."
    if code.startswith("EMPTY_MODEL_RESPONSE"):
        return "O modelo respondeu vazio."
    if code.startswith("TRUNCATED_BY_TOKEN_LIMIT"):
        return (
            "O modelo gastou o orçamento de tokens no raciocínio e não sobrou "
            f"texto final (limite atual: {MAX_PROVIDER_OUTPUT_TOKENS}). Aumente "
            "TRIADE_PROVIDER_MAX_TOKENS ou use um modelo sem raciocínio longo."
        )
    if code.startswith("TOOL_LOOP_LIMIT"):
        return (
            "O modelo encadeou ferramentas além do limite da pergunta "
            f"({MAX_TOOL_ITERATIONS} rodadas) e não fechou uma resposta. Peça algo "
            "mais específico ou aumente TRIADE_MAX_TOOL_ITERATIONS."
        )
    if code.startswith("INVALID_JSON_RESPONSE"):
        return "Resposta do provedor não é JSON válido."
    return "Não foi possível falar com o provedor."


def _provider_host(provider: dict[str, Any] | None) -> str | None:
    if not provider:
        return None
    fmt = provider.get("format", "openai-chat")
    spec = PROVIDER_FORMATS.get(fmt, {})
    base_url = str(provider.get("base_url") or spec.get("base_url") or "")
    try:
        return urllib.parse.urlparse(base_url).hostname
    except ValueError:
        return None


def _chat_tools(registry: Any) -> list[dict[str, Any]]:
    """Especificações no formato chat/completions, aceito por DeepSeek e afins."""
    return list(registry.specs()) if registry is not None else []


def _anthropic_tools(registry: Any) -> list[dict[str, Any]]:
    """Anthropic usa input_schema em vez de parameters."""
    if registry is None:
        return []
    ferramentas = []
    for spec in registry.specs():
        funcao = spec.get("function") if isinstance(spec, dict) else None
        funcao = funcao if isinstance(funcao, dict) else spec
        nome = funcao.get("name")
        if not isinstance(nome, str) or not nome:
            continue
        ferramentas.append(
            {
                "name": nome,
                "description": str(funcao.get("description") or ""),
                "input_schema": funcao.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return ferramentas


def _request_provider_chat(
    url: str,
    headers: dict[str, str],
    model: str,
    turns: list[dict[str, Any]],
    registry: Any,
) -> tuple[str, dict[str, Any]]:
    """Conversa em chat/completions, executando ferramentas quando pedidas."""
    ferramentas = _chat_tools(registry)
    mensagens: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        *turns,
    ]
    payload: dict[str, Any] = {}

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        corpo: dict[str, Any] = {
            "model": model,
            "messages": mensagens,
            "max_tokens": MAX_PROVIDER_OUTPUT_TOKENS,
        }
        if ferramentas:
            corpo["tools"] = ferramentas
            corpo["tool_choice"] = "auto"

        payload = _post_json(url, headers, corpo)
        escolha = next(
            (item for item in payload.get("choices", []) if isinstance(item, dict)),
            {},
        )
        mensagem = escolha.get("message") if isinstance(escolha.get("message"), dict) else {}
        chamadas = [
            item
            for item in (mensagem.get("tool_calls") or [])
            if isinstance(item, dict)
        ]

        if not chamadas:
            return _reply_from_openai_chat(payload), payload

        if registry is None:
            break

        mensagens.append(
            {
                "role": "assistant",
                "content": mensagem.get("content") or "",
                "tool_calls": chamadas,
            }
        )
        for chamada in chamadas:
            funcao = chamada.get("function") if isinstance(chamada.get("function"), dict) else {}
            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": str(chamada.get("id") or ""),
                    "content": _run_tool(
                        registry,
                        {"name": funcao.get("name"), "arguments": funcao.get("arguments")},
                    ),
                }
            )

    if registry is None:
        return "", payload

    # Orçamento de ferramentas esgotado: o modelo pediria mais chamadas para
    # sempre. Uma última rodada sem ferramentas força texto com o que já foi
    # coletado, em vez de devolver resposta vazia ao painel.
    payload = _post_json(
        url,
        headers,
        {
            "model": model,
            "messages": [
                *mensagens,
                {"role": "user", "content": TOOL_BUDGET_PROMPT},
            ],
            "max_tokens": MAX_PROVIDER_OUTPUT_TOKENS,
        },
    )
    return _reply_from_openai_chat(payload), payload


def _request_provider_anthropic(
    url: str,
    headers: dict[str, str],
    model: str,
    turns: list[dict[str, Any]],
    registry: Any,
) -> tuple[str, dict[str, Any]]:
    """Conversa em /messages, executando ferramentas via blocos tool_use."""
    ferramentas = _anthropic_tools(registry)
    mensagens: list[dict[str, Any]] = list(turns)
    payload: dict[str, Any] = {}

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        corpo: dict[str, Any] = {
            "model": model,
            "system": SYSTEM_INSTRUCTIONS,
            "messages": mensagens,
            "max_tokens": MAX_PROVIDER_OUTPUT_TOKENS,
        }
        if ferramentas:
            corpo["tools"] = ferramentas

        payload = _post_json(url, headers, corpo)
        blocos = [item for item in payload.get("content", []) if isinstance(item, dict)]
        usos = [item for item in blocos if item.get("type") == "tool_use"]

        if not usos:
            return _reply_from_anthropic(payload), payload

        if registry is None:
            break

        mensagens.append({"role": "assistant", "content": blocos})
        resultados = []
        for uso in usos:
            resultados.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(uso.get("id") or ""),
                    "content": _run_tool(
                        registry,
                        {"name": uso.get("name"), "arguments": uso.get("input") or {}},
                    ),
                }
            )
        mensagens.append({"role": "user", "content": resultados})

    return "", payload


def _request_provider(
    provider: dict[str, Any],
    message: str,
    history: list[dict[str, str]],
) -> str:
    """Send the conversation to the active connection and return the reply.

    A conexão recebe as mesmas ferramentas do orquestrador. Sem isso o modelo
    remoto respondia que não tinha acesso ao computador, o que era verdade
    apenas porque nenhuma ferramenta era oferecida a ele.
    """
    api_key = str(provider.get("api_key") or "").strip()
    fmt = provider.get("format", "openai-chat")
    spec = PROVIDER_FORMATS.get(fmt)
    if spec is None:
        raise RuntimeError("UNKNOWN_FORMAT")

    base_url = str(provider.get("base_url") or spec["base_url"]).rstrip("/")
    url = f"{base_url}{spec['path']}"
    model = str(provider.get("model") or "")
    turns: list[dict[str, Any]] = [*history, {"role": "user", "content": message}]
    registry = _tool_registry()

    if fmt == "anthropic":
        if not api_key:
            raise RuntimeError("API_KEY_MISSING")
        reply, payload = _request_provider_anthropic(
            url,
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            model,
            turns,
            registry,
        )
    elif fmt == "openai-responses":
        if not api_key:
            raise RuntimeError("API_KEY_MISSING")
        reply, payload = _request_provider_responses(
            url,
            {"Authorization": f"Bearer {api_key}"},
            model,
            turns,
            registry,
        )
    else:
        # Servidores locais (Ollama, LM Studio, llama.cpp) costumam dispensar a
        # chave, entao ela e opcional neste formato.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        reply, payload = _request_provider_chat(url, headers, model, turns, registry)

    if not reply:
        raise RuntimeError(_empty_reply_reason(payload))
    return reply


def _request_provider_responses(
    url: str,
    headers: dict[str, str],
    model: str,
    turns: list[dict[str, Any]],
    registry: Any,
) -> tuple[str, dict[str, Any]]:
    """Conversa na Responses API, executando ferramentas quando pedidas."""
    ferramentas = _responses_tools(registry) if registry is not None else []
    entrada: list[dict[str, Any]] = list(turns)
    anterior: str | None = None
    payload: dict[str, Any] = {}

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        corpo: dict[str, Any] = {
            "model": model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": entrada,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "medium"},
            "max_output_tokens": MAX_PROVIDER_OUTPUT_TOKENS,
        }
        if ferramentas:
            corpo["tools"] = ferramentas
            corpo["tool_choice"] = "auto"
        if anterior:
            corpo["previous_response_id"] = anterior

        payload = _post_json(url, headers, corpo)
        chamadas = _extract_function_calls(payload)
        if not chamadas:
            return _extract_output_text(payload), payload
        if registry is None:
            break

        identificador = payload.get("id")
        anterior = identificador if isinstance(identificador, str) else None
        entrada = [
            {
                "type": "function_call_output",
                "call_id": str(chamada.get("call_id") or ""),
                "output": _run_tool(registry, chamada),
            }
            for chamada in chamadas
        ]

    return "", payload


def _empty_reply_reason(payload: dict[str, Any]) -> str:
    """Distingue resposta vazia de resposta cortada pelo limite de tokens."""
    escolhas = payload.get("choices")
    if isinstance(escolhas, list):
        for escolha in escolhas:
            if not isinstance(escolha, dict):
                continue
            if escolha.get("finish_reason") == "length":
                return "TRUNCATED_BY_TOKEN_LIMIT"
            if escolha.get("finish_reason") == "tool_calls":
                # Parou pedindo ferramenta, não por falta de orçamento: culpar o
                # limite de tokens mandava o usuário mexer na variável errada.
                return "TOOL_LOOP_LIMIT"
            mensagem = escolha.get("message")
            if isinstance(mensagem, dict) and str(mensagem.get("reasoning_content") or "").strip():
                # Raciocinou e não sobrou orçamento para o texto final.
                return "TRUNCATED_BY_TOKEN_LIMIT"
    if payload.get("status") == "incomplete":
        return "TRUNCATED_BY_TOKEN_LIMIT"
    return "EMPTY_MODEL_RESPONSE"


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
            active = _active_provider()
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
                    "model_configured": active is not None,
                    "model": active.get("model") if active else MODEL,
                    "provider": active.get("label") if active else None,
                    "stt_available": _stt_ready(),
                },
            )
            return
        if self.path == "/api/providers":
            store = _read_providers()
            active = _active_provider()
            self._send_json(
                HTTPStatus.OK,
                {
                    "providers": [_public_provider(p) for p in _all_providers()],
                    "active": active.get("id") if active else None,
                    "formats": [
                        {
                            "id": key,
                            "label": spec["label"],
                            "base_url": spec["base_url"],
                            "vendors": list(spec.get("vendors", ())),
                        }
                        for key, spec in PROVIDER_FORMATS.items()
                    ],
                    "presets": list(PROVIDER_PRESETS),
                },
            )
            return
        super().do_GET()

    def _handle_transcribe(self) -> None:
        """Transcribe a microphone recording locally and return the text."""
        if not _stt_ready():
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": "Transcrição local indisponível. Verifique o modelo em models/voice.",
                    "code": "STT_UNAVAILABLE",
                },
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Envie o áudio gravado."})
            return
        if content_length > MAX_AUDIO_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "Gravação longa demais. Fale por até cerca de dois minutos."},
            )
            return

        raw = self.rfile.read(content_length)

        try:
            text = _transcribe(raw)
        except RuntimeError as error:
            code = str(error)
            messages = {
                "AUDIO_EMPTY": "Não captei áudio. Confira o microfone e tente de novo.",
                "AUDIO_TOO_SHORT": "Gravação curta demais. Segure o botão e fale um pouco mais.",
                "AUDIO_NO_STREAM": "O navegador não enviou uma trilha de áudio válida.",
                "AUDIO_DECODE_FAILED": "Não consegui decodificar o áudio gravado.",
                "STT_AUDIO_EMPTY": "Não identifiquei fala na gravação.",
                "STT_STT_TIMEOUT": "A transcrição demorou demais e foi cancelada.",
                "STT_STT_MODEL_NOT_FOUND": "Modelo de transcrição ausente em models/voice.",
            }
            status = (
                HTTPStatus.BAD_REQUEST
                if code.startswith("AUDIO_") or code == "STT_AUDIO_EMPTY"
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            self._send_json(
                status,
                {"error": messages.get(code, "Não foi possível transcrever o áudio."), "code": code},
            )
            return
        except Exception:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Falha inesperada na transcrição.", "code": "STT_FAILED"},
            )
            return

        if not text:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Não identifiquei fala na gravação.", "code": "STT_AUDIO_EMPTY"},
            )
            return

        self._send_json(HTTPStatus.OK, {"text": text[:MAX_MESSAGE_LENGTH]})

    def _read_json_body(self, limit: int = 20_000) -> tuple[Any, str | None]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > limit:
            return None, "Requisição inválida."
        try:
            return json.loads(self.rfile.read(content_length).decode("utf-8")), None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "JSON inválido."

    def _handle_providers(self) -> None:
        """Create, update, activate, remove or test an AI connection."""
        data, error = self._read_json_body()
        if error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": error})
            return

        action = data.get("action") if isinstance(data, dict) else None

        if action == "save":
            provider, message = _save_provider(data.get("provider"))
            if provider is None:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": message})
                return
            self._send_json(
                HTTPStatus.OK,
                {"provider": _public_provider(provider), "active": provider["id"]},
            )
            return

        if action == "activate":
            provider_id = str(data.get("id") or "")
            if not _activate_provider(provider_id):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Conexão não encontrada."})
                return
            self._send_json(HTTPStatus.OK, {"active": provider_id})
            return

        if action == "delete":
            provider_id = str(data.get("id") or "")
            if provider_id == "env":
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "A conexão do ambiente sai removendo OPENAI_API_KEY."},
                )
                return
            if not _delete_provider(provider_id):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Conexão não encontrada."})
                return
            active = _active_provider()
            self._send_json(
                HTTPStatus.OK, {"deleted": provider_id, "active": active.get("id") if active else None}
            )
            return

        if action == "test":
            provider_id = str(data.get("id") or "")
            provider = next(
                (p for p in _all_providers() if p.get("id") == provider_id), None
            )
            if provider is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Conexão não encontrada."})
                return
            try:
                reply = _request_provider(provider, "Responda apenas: ok", [])
            except RuntimeError as error:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {
                        "error": _provider_error_message(
                            str(error), host=_provider_host(provider)
                        ),
                        "code": str(error)[:120],
                    },
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "sample": reply[:200]})
            return

        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Ação desconhecida."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/transcribe":
            self._handle_transcribe()
            return
        if self.path == "/api/providers":
            self._handle_providers()
            return
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

        provider = _active_provider()
        usando_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())

        if provider is None and not usando_openai and _tool_registry() is None:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": "Nenhuma conexão de IA configurada. Cadastre uma em Conexões de IA.",
                    "code": "NO_PROVIDER",
                },
            )
            return

        modelo_usado = MODEL if usando_openai else "qwen-local"
        rotulo_provedor = None

        try:
            if provider is not None:
                # Conexão cadastrada no painel tem precedência: foi escolha explícita.
                reply = _request_provider(
                    provider, message, _safe_history(data.get("history"))
                )
                modelo_usado = provider.get("model")
                rotulo_provedor = provider.get("label")
            elif usando_openai:
                # Caminho com tool-calling das ferramentas do orquestrador.
                reply = _request_openai(message, _safe_history(data.get("history")))
            else:
                # Sem provedor e sem chave: Qwen local, que também tem as ferramentas.
                reply = _request_local_model(message)
        except RuntimeError as error:
            code = str(error)
            if code.startswith("LOCAL_MODEL_UNAVAILABLE"):
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": (
                            "Modelo local indisponível. Verifique o llama-server, "
                            "cadastre uma conexão de IA ou defina OPENAI_API_KEY."
                        ),
                        "code": code,
                    },
                )
                return
            status = (
                HTTPStatus.SERVICE_UNAVAILABLE
                if code == "API_KEY_MISSING"
                else HTTPStatus.BAD_GATEWAY
            )
            self._send_json(
                status,
                {
                    "error": _provider_error_message(code, host=_provider_host(provider)),
                    "code": code[:120],
                },
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {"reply": reply, "model": modelo_usado, "provider": rotulo_provedor},
        )


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
    active = _active_provider()
    if active is None:
        print("Aviso: nenhuma conexão de IA configurada; cadastre uma no painel Conexões de IA.")
    else:
        print(f"Conexão de IA ativa: {active.get('label')} ({active.get('model')}).")
    if _stt_ready():
        print("Microfone: transcrição local ativa (faster-whisper, o áudio não sai da máquina).")
    else:
        print("Aviso: transcrição local indisponível; o botão de microfone ficará oculto.")
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
