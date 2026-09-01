from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_PROVIDER_URLS = {
    "bing_rss": "https://www.bing.com/search",
    "brave": "https://api.search.brave.com/res/v1/web/search",
    "duckduckgo_html": "https://html.duckduckgo.com/html/",
}


@dataclass(frozen=True)
class Backend:
    name: str
    default_model: Path
    runtime: str
    legacy: bool = False


BACKENDS = {
    "qwen35": Backend(
        "qwen35",
        PROJECT_ROOT / "models" / "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        "stock",
    ),
    "qwen25-base": Backend(
        "qwen25-base",
        PROJECT_ROOT / "qwen25-1.5b.base.gguf",
        "stock",
        legacy=True,
    ),
    "qwen25-tq3p": Backend(
        "qwen25-tq3p",
        PROJECT_ROOT / "qwen25-1.5b.tq3p.gguf",
        "sasori",
        legacy=True,
    ),
}


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"booleano invalido: {value!r}")


def _path_list(value: str) -> tuple[Path, ...]:
    return tuple(Path(item).expanduser().resolve() for item in value.split(os.pathsep) if item)


def _domain_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    backend: Backend
    model_path: Path
    server_executable: Path
    context_size: int
    gpu_layers: int
    kv_cache_k: str
    kv_cache_v: str
    flash_attention: str
    parallel_slots: int
    server_host: str
    server_port: int
    timeout: int
    fit: bool
    vram_reserve_mib: int
    reasoning: str
    allowed_roots: tuple[Path, ...]
    max_tool_calls: int
    max_attempts: int
    max_tool_output_bytes: int
    agent_decision_shadow: bool
    agent_decision_fast_path: bool
    agent_decision_context_cache: bool
    agent_decision_semantic_first: bool
    execution_gate_shadow: bool
    execution_gate_authority: str
    orchestration_mode: str
    orchestration_shadow_enabled: bool
    orchestration_fast_path_enabled: bool
    orchestration_decision_cache_enabled: bool
    orchestration_decision_cache_max_entries: int
    orchestration_shadow_max_steps: int
    orchestration_shadow_max_observations: int
    orchestration_shadow_max_action_history: int
    orchestration_shadow_max_context_items: int
    orchestration_shadow_max_repeated_action: int
    orchestration_shadow_max_same_observation: int
    orchestration_shadow_max_failures: int
    orchestration_max_model_calls: int
    orchestration_max_tool_calls: int
    orchestration_max_delegations: int
    orchestration_max_elapsed_seconds: int
    codex_timeout: int
    codex_app_server_endpoint: str
    codex_current_thread_id: str | None
    codex_app_server_start_timeout: int
    codex_quick_wait_timeout_seconds: int
    codex_turn_hard_timeout_seconds: int
    codex_job_retention_days: int
    action_confirmation_timeout_seconds: int
    deepseek_enabled: bool
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    deepseek_auto_escalation: bool
    deepseek_request_timeout_seconds: int
    deepseek_max_retries: int
    deepseek_session_max_recent_turns: int
    state_dir: Path
    env_file: Path
    env_file_loaded: bool
    web_enabled: bool
    web_search_provider: str
    web_search_url: str
    web_search_api_key: str | None
    web_safe_search: str
    web_threat_analysis_enabled: bool
    web_threat_learning_enabled: bool
    web_timeout: int
    web_max_download_bytes: int
    web_max_text_chars: int
    web_max_pdf_pages: int
    web_user_agent: str
    web_allowed_domains: tuple[str, ...]
    web_blocked_domains: tuple[str, ...]
    web_query_expansion_enabled: bool
    web_max_query_variants: int
    web_cross_language_search: bool
    web_default_region: str
    web_min_result_relevance: float
    web_min_source_relevance: float
    web_relevance_top_k: int
    web_max_research_corrections: int
    web_max_total_searches: int
    web_max_total_opens: int
    voice_enabled: bool
    voice_stt_provider: str
    voice_stt_model: Path
    voice_stt_device: str
    voice_stt_compute_type: str
    voice_stt_language: str
    voice_stt_threads: int
    voice_stt_timeout_seconds: int
    voice_tts_provider: str
    voice_mode: str
    voice_piper_voice: str
    voice_piper_requested_voice: str
    voice_piper_config_path: Path
    voice_piper_fallback: bool
    voice_windows_voice_id: str
    voice_windows_rate: float
    voice_windows_volume: int
    voice_fallback_provider: str
    voice_tts_model: Path
    voice_tts_voice: str
    voice_tts_device: str
    voice_tts_rate: float
    voice_tts_volume: float
    voice_tts_timeout_seconds: int
    voice_input_device: str | None
    voice_output_device: str | None
    voice_input_device_name: str | None
    voice_output_device_name: str | None
    voice_tts_streaming: bool
    voice_tts_chunk_min_characters: int
    voice_tts_chunk_max_characters: int
    voice_tts_queue_size: int
    voice_sample_rate: int
    voice_max_recording_seconds: int
    voice_silence_timeout_ms: int
    voice_min_speech_ms: int
    voice_silence_threshold: float
    voice_confirm_transcription: bool
    voice_max_spoken_characters: int
    voice_read_code: bool
    voice_read_urls: bool
    voice_summarize_long_responses: bool
    voice_temp_directory: Path
    voice_keep_recordings: bool
    voice_log_level: str
    voice_debug_transcripts: bool
    voice_interrupt_key: str
    voice_style: str
    voice_sentence_pause_ms: int
    voice_paragraph_pause_ms: int
    voice_post_processing: bool
    voice_normalize_loudness: bool
    voice_light_compression: bool
    voice_light_eq: bool
    voice_translate_common_status_terms: bool
    voice_piper_use_model_default_noise: bool

    @property
    def base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def server_command(self) -> list[str]:
        command = [
            str(self.server_executable),
            "-m",
            str(self.model_path),
            "-c",
            str(self.context_size),
            "-np",
            str(self.parallel_slots),
            "-fa",
            self.flash_attention,
            "-ctk",
            self.kv_cache_k,
            "-ctv",
            self.kv_cache_v,
            "-ngl",
            str(self.gpu_layers),
            "-fit",
            "on" if self.fit else "off",
            "-fitt",
            str(self.vram_reserve_mib),
            "--host",
            self.server_host,
            "--port",
            str(self.server_port),
            "-to",
            str(self.timeout),
            "--jinja",
            "--reasoning",
            self.reasoning,
            "--no-ui",
            "--cors-origins",
            "localhost",
            "--metrics",
            "--slots",
        ]
        return command


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    initial_values = os.environ if env is None else env
    env_file = Path(
        initial_values.get("TERN_ENV_FILE", str(PROJECT_ROOT / ".env"))
    ).expanduser().resolve()
    env_file_loaded = False
    if env is None and env_file.is_file():
        load_dotenv(env_file, override=False)
        env_file_loaded = True
    values = os.environ if env is None else env
    provider = values.get("WEB_SEARCH_PROVIDER", "bing_rss").strip().lower()
    provider_url = SEARCH_PROVIDER_URLS.get(provider, "")
    backend_name = values.get(
        "LOCAL_MODEL_PROVIDER",
        values.get("MODEL_BACKEND", "qwen35"),
    )
    try:
        backend = BACKENDS[backend_name]
    except KeyError as exc:
        allowed = ", ".join(BACKENDS)
        raise ValueError(f"MODEL_BACKEND invalido: {backend_name!r}; use {allowed}") from exc

    model_path = Path(
        values.get(
            "LOCAL_MODEL_PATH",
            values.get("MODEL_PATH", str(backend.default_model)),
        )
    ).expanduser().resolve()
    stock_runtime = PROJECT_ROOT / "runtime" / "llama-b10173-vulkan" / "llama-server.exe"
    sasori_runtime = PROJECT_ROOT.parent / "llama.cpp" / "build" / "bin" / "llama-server.exe"
    runtime_default = stock_runtime if backend.runtime == "stock" else sasori_runtime
    executable = Path(
        values.get(
            "LOCAL_MODEL_RUNTIME",
            values.get("MODEL_SERVER_EXECUTABLE", str(runtime_default)),
        )
    ).expanduser().resolve()
    allowed_default = os.pathsep.join(
        str(path) for path in (PROJECT_ROOT, PROJECT_ROOT.parent / "llama.cpp", PROJECT_ROOT.parent / "sasori_review")
    )
    rate_value = values.get("VOICE_TTS_RATE")
    if rate_value is None:
        rate_value = values.get("VOICE_TTS_SPEED")
        if rate_value is not None:
            warnings.warn(
                "VOICE_TTS_SPEED foi substituida por VOICE_TTS_RATE; "
                "a opcao antiga sera removida futuramente",
                DeprecationWarning,
                stacklevel=2,
            )
    from .voice.voices import resolve_piper_voice

    piper_voice = resolve_piper_voice(values, PROJECT_ROOT)
    settings = Settings(
        backend=backend,
        model_path=model_path,
        server_executable=executable,
        context_size=int(values.get("MODEL_CONTEXT_SIZE", "16384")),
        gpu_layers=int(values.get("MODEL_GPU_LAYERS", "99")),
        kv_cache_k=values.get("MODEL_KV_CACHE_K", "q8_0"),
        kv_cache_v=values.get("MODEL_KV_CACHE_V", "q8_0"),
        flash_attention=values.get("MODEL_FLASH_ATTENTION", "on"),
        parallel_slots=int(values.get("MODEL_PARALLEL_SLOTS", "1")),
        server_host=values.get("MODEL_SERVER_HOST", "127.0.0.1"),
        server_port=int(values.get("MODEL_SERVER_PORT", "8080")),
        # O mesmo valor vira o -to do llama-server (timeout HTTP do servidor) e
        # o timeout do cliente. 180s cortava respostas finais longas no meio da
        # geracao (prompt grande + ~12 tok/s no Vulkan), deixando o painel com
        # "O modelo respondeu vazio." em tarefas reais. 600s da margem segura.
        timeout=int(values.get("MODEL_TIMEOUT", "600")),
        fit=_bool(values.get("MODEL_FIT", "true")),
        vram_reserve_mib=int(values.get("MODEL_VRAM_RESERVE_MIB", "1280")),
        reasoning=values.get("MODEL_REASONING", "off"),
        allowed_roots=_path_list(values.get("MODEL_ALLOWED_ROOTS", allowed_default)),
        max_tool_calls=int(values.get("MODEL_MAX_TOOL_CALLS", "8")),
        max_attempts=int(values.get("MODEL_MAX_ATTEMPTS", "3")),
        max_tool_output_bytes=int(values.get("MODEL_MAX_TOOL_OUTPUT_BYTES", "131072")),
        agent_decision_shadow=_bool(
            values.get("AGENT_DECISION_SHADOW", "false")
        ),
        agent_decision_fast_path=_bool(
            values.get("AGENT_DECISION_FAST_PATH", "true")
        ),
        agent_decision_context_cache=_bool(
            values.get("AGENT_DECISION_CONTEXT_CACHE", "true")
        ),
        agent_decision_semantic_first=_bool(
            values.get("AGENT_DECISION_SEMANTIC_FIRST", "true")
        ),
        execution_gate_shadow=_bool(values.get("EXECUTION_GATE_SHADOW", "true")),
        execution_gate_authority=values.get(
            "EXECUTION_GATE_AUTHORITY", "shadow"
        ).strip().lower(),
        orchestration_mode=values.get(
            "ORCHESTRATION_MODE", "shadow"
        ).strip().lower(),
        orchestration_shadow_enabled=_bool(
            values.get("ORCHESTRATION_SHADOW_ENABLED", "false")
        ),
        orchestration_fast_path_enabled=_bool(
            values.get("ORCHESTRATION_FAST_PATH_ENABLED", "true")
        ),
        orchestration_decision_cache_enabled=_bool(
            values.get("ORCHESTRATION_DECISION_CACHE_ENABLED", "true")
        ),
        orchestration_decision_cache_max_entries=int(
            values.get("ORCHESTRATION_DECISION_CACHE_MAX_ENTRIES", "128")
        ),
        orchestration_shadow_max_steps=int(
            values.get("ORCHESTRATION_SHADOW_MAX_STEPS", "8")
        ),
        orchestration_shadow_max_observations=int(
            values.get("ORCHESTRATION_SHADOW_MAX_OBSERVATIONS", "16")
        ),
        orchestration_shadow_max_action_history=int(
            values.get("ORCHESTRATION_SHADOW_MAX_ACTION_HISTORY", "16")
        ),
        orchestration_shadow_max_context_items=int(
            values.get("ORCHESTRATION_SHADOW_MAX_CONTEXT_ITEMS", "64")
        ),
        orchestration_shadow_max_repeated_action=int(
            values.get("ORCHESTRATION_SHADOW_MAX_REPEATED_ACTION", "2")
        ),
        orchestration_shadow_max_same_observation=int(
            values.get("ORCHESTRATION_SHADOW_MAX_SAME_OBSERVATION", "2")
        ),
        orchestration_shadow_max_failures=int(
            values.get("ORCHESTRATION_SHADOW_MAX_FAILURES", "3")
        ),
        orchestration_max_model_calls=int(
            values.get("ORCHESTRATION_MAX_MODEL_CALLS", "8")
        ),
        orchestration_max_tool_calls=int(
            values.get("ORCHESTRATION_MAX_TOOL_CALLS", "8")
        ),
        orchestration_max_delegations=int(
            values.get("ORCHESTRATION_MAX_DELEGATIONS", "4")
        ),
        orchestration_max_elapsed_seconds=int(
            values.get("ORCHESTRATION_MAX_ELAPSED_SECONDS", "900")
        ),
        codex_timeout=int(values.get("CODEX_TIMEOUT", "1800")),
        codex_app_server_endpoint=values.get(
            "CODEX_APP_SERVER_ENDPOINT", "ws://127.0.0.1:4500"
        ).strip(),
        codex_current_thread_id=(
            values.get("CODEX_THREAD_ID", "").strip() or None
        ),
        codex_app_server_start_timeout=int(
            values.get("CODEX_APP_SERVER_START_TIMEOUT", "30")
        ),
        codex_quick_wait_timeout_seconds=int(
            values.get("CODEX_QUICK_WAIT_TIMEOUT_SECONDS", "60")
        ),
        codex_turn_hard_timeout_seconds=int(
            values.get("CODEX_TURN_HARD_TIMEOUT_SECONDS", "0")
        ),
        codex_job_retention_days=int(
            values.get("CODEX_JOB_RETENTION_DAYS", "7")
        ),
        action_confirmation_timeout_seconds=int(
            values.get("ACTION_CONFIRMATION_TIMEOUT_SECONDS", "300")
        ),
        deepseek_enabled=_bool(values.get("DEEPSEEK_ENABLED", "true")),
        deepseek_api_key=values.get("DEEPSEEK_API_KEY") or None,
        deepseek_base_url=values.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ).strip(),
        deepseek_model=values.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
        deepseek_auto_escalation=_bool(
            values.get("DEEPSEEK_AUTO_ESCALATION", "false")
        ),
        deepseek_request_timeout_seconds=int(
            values.get("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "180")
        ),
        deepseek_max_retries=int(values.get("DEEPSEEK_MAX_RETRIES", "2")),
        deepseek_session_max_recent_turns=int(
            values.get("DEEPSEEK_SESSION_MAX_RECENT_TURNS", "20")
        ),
        state_dir=Path(values.get("MODEL_STATE_DIR", str(PROJECT_ROOT / ".orchestrator"))).resolve(),
        env_file=env_file,
        env_file_loaded=env_file_loaded,
        web_enabled=_bool(values.get("WEB_ENABLED", "true")),
        web_search_provider=provider,
        web_search_url=values.get("WEB_SEARCH_URL", provider_url),
        web_search_api_key=values.get("WEB_SEARCH_API_KEY")
        or values.get("BRAVE_SEARCH_API_KEY"),
        web_safe_search=values.get("WEB_SAFE_SEARCH", "off").strip().lower(),
        web_threat_analysis_enabled=_bool(
            values.get("WEB_THREAT_ANALYSIS_ENABLED", "true")
        ),
        web_threat_learning_enabled=_bool(
            values.get("WEB_THREAT_LEARNING_ENABLED", "true")
        ),
        web_timeout=int(values.get("WEB_TIMEOUT", "20")),
        web_max_download_bytes=int(
            values.get("WEB_MAX_DOWNLOAD_BYTES", str(10 * 1024 * 1024))
        ),
        web_max_text_chars=int(values.get("WEB_MAX_TEXT_CHARS", "65536")),
        web_max_pdf_pages=int(values.get("WEB_MAX_PDF_PAGES", "20")),
        web_user_agent=values.get("WEB_USER_AGENT", "TernLocalResearch/1.0"),
        web_allowed_domains=_domain_list(values.get("WEB_ALLOWED_DOMAINS", "")),
        web_blocked_domains=_domain_list(values.get("WEB_BLOCKED_DOMAINS", "")),
        web_query_expansion_enabled=_bool(
            values.get("WEB_QUERY_EXPANSION_ENABLED", "true")
        ),
        web_max_query_variants=int(
            values.get("WEB_MAX_QUERY_VARIANTS", "4")
        ),
        web_cross_language_search=_bool(
            values.get("WEB_CROSS_LANGUAGE_SEARCH", "true")
        ),
        web_default_region=values.get(
            "WEB_DEFAULT_REGION", "BR"
        ).strip().upper(),
        web_min_result_relevance=float(
            values.get("WEB_MIN_RESULT_RELEVANCE", "0.55")
        ),
        web_min_source_relevance=float(
            values.get("WEB_MIN_SOURCE_RELEVANCE", "0.65")
        ),
        web_relevance_top_k=int(
            values.get("WEB_RELEVANCE_TOP_K", "8")
        ),
        web_max_research_corrections=int(
            values.get("WEB_MAX_RESEARCH_CORRECTIONS", "2")
        ),
        web_max_total_searches=int(
            values.get("WEB_MAX_TOTAL_SEARCHES", "6")
        ),
        web_max_total_opens=int(
            values.get("WEB_MAX_TOTAL_OPENS", "10")
        ),
        voice_enabled=_bool(values.get("VOICE_ENABLED", "true")),
        voice_stt_provider=values.get(
            "VOICE_STT_PROVIDER", "faster_whisper"
        ).strip().lower(),
        voice_stt_model=Path(
            values.get(
                "VOICE_STT_MODEL",
                str(
                    PROJECT_ROOT
                    / "models"
                    / "voice"
                    / "faster-whisper-base"
                ),
            )
        ).expanduser().resolve(),
        voice_stt_device=values.get("VOICE_STT_DEVICE", "cpu").strip().lower(),
        voice_stt_compute_type=values.get(
            "VOICE_STT_COMPUTE_TYPE", "int8"
        ).strip().lower(),
        voice_stt_language=values.get("VOICE_STT_LANGUAGE", "pt").strip(),
        voice_stt_threads=int(
            values.get(
                "VOICE_STT_THREADS",
                str(max(1, min(4, os.cpu_count() or 1))),
            )
        ),
        voice_stt_timeout_seconds=int(
            values.get("VOICE_STT_TIMEOUT_SECONDS", "120")
        ),
        voice_tts_provider=values.get(
            "VOICE_TTS_PROVIDER", "piper"
        ).strip().lower(),
        voice_mode=values.get(
            "VOICE_MODE", values.get("VOICE_TTS_PROVIDER", "piper")
        ).strip().lower(),
        voice_piper_voice=piper_voice.alias,
        voice_piper_requested_voice=piper_voice.requested_alias,
        voice_piper_config_path=piper_voice.config_path,
        voice_piper_fallback=piper_voice.fallback,
        voice_windows_voice_id=values.get(
            "VOICE_WINDOWS_VOICE_ID",
            (
                "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech_OneCore"
                "\\Voices\\Tokens\\MSTTS_V110_ptBR_DanielM"
            ),
        ).strip(),
        voice_windows_rate=float(values.get("VOICE_WINDOWS_RATE", "1.5")),
        voice_windows_volume=int(
            values.get("VOICE_WINDOWS_VOLUME", "100")
        ),
        voice_fallback_provider=values.get(
            "VOICE_FALLBACK_PROVIDER", "piper"
        ).strip().lower(),
        voice_tts_model=piper_voice.model_path,
        voice_tts_voice=piper_voice.alias,
        voice_tts_device=values.get("VOICE_TTS_DEVICE", "cpu").strip().lower(),
        voice_tts_rate=float(rate_value or "0.94"),
        voice_tts_volume=float(values.get("VOICE_TTS_VOLUME", "1.0")),
        voice_tts_timeout_seconds=int(
            values.get("VOICE_TTS_TIMEOUT_SECONDS", "60")
        ),
        voice_input_device=values.get("VOICE_INPUT_DEVICE") or None,
        voice_output_device=values.get("VOICE_OUTPUT_DEVICE") or None,
        voice_input_device_name=values.get(
            "VOICE_INPUT_DEVICE_NAME"
        ) or None,
        voice_output_device_name=values.get(
            "VOICE_OUTPUT_DEVICE_NAME"
        ) or None,
        voice_tts_streaming=_bool(
            values.get("VOICE_TTS_STREAMING", "true")
        ),
        voice_tts_chunk_min_characters=int(
            values.get("VOICE_TTS_CHUNK_MIN_CHARACTERS", "40")
        ),
        voice_tts_chunk_max_characters=int(
            values.get("VOICE_TTS_CHUNK_MAX_CHARACTERS", "280")
        ),
        voice_tts_queue_size=int(
            values.get("VOICE_TTS_QUEUE_SIZE", "3")
        ),
        voice_sample_rate=int(values.get("VOICE_SAMPLE_RATE", "16000")),
        voice_max_recording_seconds=int(
            values.get("VOICE_MAX_RECORDING_SECONDS", "60")
        ),
        voice_silence_timeout_ms=int(
            values.get("VOICE_SILENCE_TIMEOUT_MS", "1200")
        ),
        voice_min_speech_ms=int(values.get("VOICE_MIN_SPEECH_MS", "300")),
        voice_silence_threshold=float(
            values.get("VOICE_SILENCE_THRESHOLD", "0.015")
        ),
        voice_confirm_transcription=_bool(
            values.get("VOICE_CONFIRM_TRANSCRIPTION", "true")
        ),
        voice_max_spoken_characters=int(
            values.get("VOICE_MAX_SPOKEN_CHARACTERS", "1200")
        ),
        voice_read_code=_bool(values.get("VOICE_READ_CODE", "false")),
        voice_read_urls=_bool(values.get("VOICE_READ_URLS", "false")),
        voice_summarize_long_responses=_bool(
            values.get("VOICE_SUMMARIZE_LONG_RESPONSES", "true")
        ),
        voice_temp_directory=Path(
            values.get(
                "VOICE_TEMP_DIRECTORY",
                str(PROJECT_ROOT / ".orchestrator" / "voice-temp"),
            )
        ).expanduser().resolve(),
        voice_keep_recordings=_bool(
            values.get("VOICE_KEEP_RECORDINGS", "false")
        ),
        voice_log_level=values.get("VOICE_LOG_LEVEL", "INFO").strip().upper(),
        voice_debug_transcripts=_bool(
            values.get("VOICE_DEBUG_TRANSCRIPTS", "false")
        ),
        voice_interrupt_key=values.get(
            "VOICE_INTERRUPT_KEY", "esc"
        ).strip().lower(),
        voice_style=values.get("VOICE_STYLE", "clear_adult").strip().lower(),
        voice_sentence_pause_ms=int(
            values.get("VOICE_SENTENCE_PAUSE_MS", "160")
        ),
        voice_paragraph_pause_ms=int(
            values.get("VOICE_PARAGRAPH_PAUSE_MS", "280")
        ),
        voice_post_processing=_bool(
            values.get("VOICE_POST_PROCESSING", "false")
        ),
        voice_normalize_loudness=_bool(
            values.get("VOICE_NORMALIZE_LOUDNESS", "true")
        ),
        voice_light_compression=_bool(
            values.get("VOICE_LIGHT_COMPRESSION", "false")
        ),
        voice_light_eq=_bool(values.get("VOICE_LIGHT_EQ", "false")),
        voice_translate_common_status_terms=_bool(
            values.get("VOICE_TRANSLATE_COMMON_STATUS_TERMS", "true")
        ),
        voice_piper_use_model_default_noise=_bool(
            values.get("VOICE_PIPER_USE_MODEL_DEFAULT_NOISE", "true")
        ),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    if settings.execution_gate_authority not in {"shadow", "explicit_user"}:
        raise ValueError(
            "EXECUTION_GATE_AUTHORITY deve ser shadow ou explicit_user"
        )
    if settings.server_host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("MODEL_SERVER_HOST deve permanecer local")
    if settings.context_size <= 0 or settings.parallel_slots != 1:
        raise ValueError("contexto deve ser positivo e MODEL_PARALLEL_SLOTS deve ser 1")
    if settings.flash_attention not in {"on", "off", "auto"}:
        raise ValueError("MODEL_FLASH_ATTENTION deve ser on, off ou auto")
    if settings.reasoning not in {"on", "off", "auto"}:
        raise ValueError("MODEL_REASONING deve ser on, off ou auto")
    if not settings.deepseek_base_url.startswith(("https://", "http://")):
        raise ValueError("DEEPSEEK_BASE_URL deve ser uma URL HTTP(S)")
    if settings.deepseek_request_timeout_seconds <= 0:
        raise ValueError("DEEPSEEK_REQUEST_TIMEOUT_SECONDS deve ser positivo")
    if settings.deepseek_max_retries < 0:
        raise ValueError("DEEPSEEK_MAX_RETRIES nao pode ser negativo")
    if settings.deepseek_session_max_recent_turns <= 0:
        raise ValueError("DEEPSEEK_SESSION_MAX_RECENT_TURNS deve ser positivo")
    if settings.kv_cache_k not in {"f16", "bf16", "q8_0", "q4_0"}:
        raise ValueError("MODEL_KV_CACHE_K nao suportado")
    if settings.kv_cache_v not in {"f16", "bf16", "q8_0", "q4_0"}:
        raise ValueError("MODEL_KV_CACHE_V nao suportado")
    if settings.max_tool_calls < 1 or settings.max_attempts < 1:
        raise ValueError("limites devem ser positivos")
    if settings.orchestration_mode not in {"shadow", "bounded_live"}:
        raise ValueError("ORCHESTRATION_MODE deve ser shadow ou bounded_live")
    if any(
        value < 1
        for value in (
            settings.orchestration_decision_cache_max_entries,
            settings.orchestration_shadow_max_steps,
            settings.orchestration_shadow_max_observations,
            settings.orchestration_shadow_max_action_history,
            settings.orchestration_shadow_max_context_items,
            settings.orchestration_shadow_max_repeated_action,
            settings.orchestration_shadow_max_same_observation,
            settings.orchestration_shadow_max_failures,
            settings.orchestration_max_model_calls,
            settings.orchestration_max_tool_calls,
            settings.orchestration_max_delegations,
            settings.orchestration_max_elapsed_seconds,
        )
    ):
        raise ValueError("limites de orchestration shadow/live devem ser positivos")
    if (
        not re.fullmatch(
            r"wss?://(?:127\.0\.0\.1|localhost|\[?::1\]?):\d+",
            settings.codex_app_server_endpoint,
            re.IGNORECASE,
        )
        or settings.codex_app_server_start_timeout < 1
        or settings.codex_quick_wait_timeout_seconds < 1
        or settings.codex_turn_hard_timeout_seconds < 0
        or settings.codex_job_retention_days < 1
        or settings.action_confirmation_timeout_seconds < 1
    ):
        raise ValueError("CODEX_APP_SERVER_ENDPOINT deve ser WebSocket local com porta")
    if settings.web_search_provider not in SEARCH_PROVIDER_URLS:
        allowed = ", ".join(SEARCH_PROVIDER_URLS)
        raise ValueError(
            f"WEB_SEARCH_PROVIDER invalido: {settings.web_search_provider!r}; use {allowed}"
        )
    if not settings.web_search_url.startswith("https://"):
        raise ValueError("WEB_SEARCH_URL deve usar HTTPS")
    if settings.web_safe_search not in {"off", "moderate", "strict"}:
        raise ValueError(
            "WEB_SAFE_SEARCH deve ser off, moderate ou strict"
        )
    if (
        settings.web_timeout < 1
        or settings.web_max_download_bytes < 1024
        or settings.web_max_text_chars < 1024
        or settings.web_max_pdf_pages < 1
    ):
        raise ValueError("limites web devem ser positivos")
    if (
        not 1 <= settings.web_max_query_variants <= 10
        or not 0 <= settings.web_min_result_relevance <= 1
        or not 0 <= settings.web_min_source_relevance <= 1
        or not 1 <= settings.web_relevance_top_k <= 20
        or not 0 <= settings.web_max_research_corrections <= 5
        or not 1 <= settings.web_max_total_searches <= 20
        or not 1 <= settings.web_max_total_opens <= 30
        or not re.fullmatch(r"[A-Z]{2}", settings.web_default_region)
    ):
        raise ValueError("configuracao de relevancia web invalida")
    if settings.voice_stt_provider not in {"faster_whisper"}:
        raise ValueError("VOICE_STT_PROVIDER deve ser faster_whisper")
    if settings.voice_tts_provider not in {"piper", "windows_sapi"}:
        raise ValueError(
            "VOICE_TTS_PROVIDER deve ser piper ou windows_sapi"
        )
    if settings.voice_mode not in {"piper", "windows_sapi", "silent"}:
        raise ValueError(
            "VOICE_MODE deve ser piper, windows_sapi ou silent"
        )
    if settings.voice_fallback_provider != "piper":
        raise ValueError("VOICE_FALLBACK_PROVIDER deve ser piper")
    if (
        not settings.voice_windows_voice_id
        or not -10 <= settings.voice_windows_rate <= 10
        or not 0 <= settings.voice_windows_volume <= 100
    ):
        raise ValueError(
            "VOICE_WINDOWS_RATE deve estar entre -10 e 10 e "
            "VOICE_WINDOWS_VOLUME entre 0 e 100"
        )
    if settings.voice_stt_device != "cpu" or settings.voice_tts_device != "cpu":
        raise ValueError("STT e Piper devem usar CPU")
    if settings.voice_stt_compute_type not in {"int8", "int8_float32"}:
        raise ValueError("VOICE_STT_COMPUTE_TYPE deve ser int8 ou int8_float32")
    if (
        settings.voice_stt_threads < 1
        or settings.voice_stt_timeout_seconds < 1
        or settings.voice_tts_timeout_seconds < 1
        or settings.voice_sample_rate < 8000
        or settings.voice_max_recording_seconds < 1
        or settings.voice_silence_timeout_ms < 100
        or settings.voice_min_speech_ms < 50
        or not 0 < settings.voice_silence_threshold < 1
        or settings.voice_max_spoken_characters < 100
        or settings.voice_tts_chunk_min_characters < 1
        or settings.voice_tts_chunk_max_characters
        < settings.voice_tts_chunk_min_characters
        or settings.voice_tts_chunk_max_characters > 2000
        or not 1 <= settings.voice_tts_queue_size <= 10
        or not 0.88 <= settings.voice_tts_rate <= 1.25
        or not 0 <= settings.voice_tts_volume <= 2
        or not 0 <= settings.voice_sentence_pause_ms <= 2000
        or not 0 <= settings.voice_paragraph_pause_ms <= 5000
    ):
        raise ValueError("configuracao numerica de voz invalida")


def assert_runtime_ready(settings: Settings) -> None:
    if not settings.model_path.is_file():
        raise FileNotFoundError(
            f"modelo {settings.backend.name} ausente: {settings.model_path}. "
            "Fallback automatico foi desativado."
        )
    if not settings.server_executable.is_file():
        raise FileNotFoundError(f"llama-server ausente: {settings.server_executable}")
