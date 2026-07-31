from __future__ import annotations

import argparse
import json
import sys

from .agent import Supervisor
from .client import LlamaClient
from .codex import CodexRunner
from .config import load_settings
from .runtime import RuntimeManager
from .security import ActionLogger, PathPolicy
from .tools import ToolRegistry
from .web import WebClient, WebConfig, WebError


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _approval(enabled: bool):
    return (lambda _action, _arguments: True) if enabled else None


def _web_client(settings) -> WebClient:
    return WebClient(
        WebConfig(
            enabled=settings.web_enabled,
            search_provider=settings.web_search_provider,
            search_url=settings.web_search_url,
            search_api_key=settings.web_search_api_key,
            timeout=settings.web_timeout,
            max_download_bytes=settings.web_max_download_bytes,
            max_text_chars=settings.web_max_text_chars,
            max_pdf_pages=settings.web_max_pdf_pages,
            user_agent=settings.web_user_agent,
            allowed_domains=settings.web_allowed_domains,
            blocked_domains=settings.web_blocked_domains,
            query_expansion_enabled=settings.web_query_expansion_enabled,
            max_query_variants=settings.web_max_query_variants,
            cross_language_search=settings.web_cross_language_search,
            default_region=settings.web_default_region,
            min_result_relevance=settings.web_min_result_relevance,
            min_source_relevance=settings.web_min_source_relevance,
            relevance_top_k=settings.web_relevance_top_k,
            max_research_corrections=settings.web_max_research_corrections,
            max_total_searches=settings.web_max_total_searches,
            max_total_opens=settings.web_max_total_opens,
        )
    )


def _voice_stack(settings, *, mode: str | None = None):
    from .voice.audio import SoundDeviceAudio
    from .voice.logging import VoiceLogger
    from .voice.stt import FasterWhisperSTT
    from .voice.tts import PiperTTS, WindowsSpeechTTS

    audio = SoundDeviceAudio()
    stt = FasterWhisperSTT(
        settings.voice_stt_model,
        device=settings.voice_stt_device,
        compute_type=settings.voice_stt_compute_type,
        threads=settings.voice_stt_threads,
    )
    piper = PiperTTS(
        settings.voice_tts_model,
        audio,
        config_path=settings.voice_piper_config_path,
        output_device=settings.voice_output_device,
        output_device_name=settings.voice_output_device_name,
        interrupt_key=settings.voice_interrupt_key,
        sentence_pause_ms=settings.voice_sentence_pause_ms,
        paragraph_pause_ms=settings.voice_paragraph_pause_ms,
        post_processing=settings.voice_post_processing,
        normalize_loudness=settings.voice_normalize_loudness,
        light_compression=settings.voice_light_compression,
        light_eq=settings.voice_light_eq,
    )
    if settings.voice_tts_provider == "windows_sapi":
        tts = WindowsSpeechTTS(
            audio,
            voice_id=settings.voice_windows_voice_id,
            rate=settings.voice_windows_rate,
            volume=settings.voice_windows_volume,
            temp_directory=settings.voice_temp_directory,
            output_device=settings.voice_output_device,
            output_device_name=settings.voice_output_device_name,
            interrupt_key=settings.voice_interrupt_key,
            fallback=piper,
        )
    else:
        tts = piper
    tts.mode = mode or settings.voice_mode
    logger = VoiceLogger(
        settings.state_dir / "voice-actions.jsonl",
        level=settings.voice_log_level,
        debug_transcripts=settings.voice_debug_transcripts,
    )
    return audio, stt, tts, logger


def _registry(settings, *, approval=None) -> ToolRegistry:
    policy = PathPolicy(settings.allowed_roots)
    return ToolRegistry(
        policy=policy,
        logger=ActionLogger(settings.state_dir / "actions.jsonl"),
        codex=CodexRunner(policy, settings.codex_timeout),
        max_output_bytes=settings.max_tool_output_bytes,
        approval=approval,
        web=_web_client(settings),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orquestrador local Qwen3.5")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="mostra configuracao efetiva")
    start = sub.add_parser("start", help="inicia llama-server sem duplicar processo")
    start.add_argument("--wait", type=int, default=240)
    sub.add_parser("stop", help="encerra llama-server")
    sub.add_parser("status", help="mostra estado do servidor")
    sub.add_parser("tools", help="lista schemas das ferramentas")
    diagnose = sub.add_parser(
        "search-diagnose",
        help="testa somente o provedor de busca, sem Qwen",
    )
    diagnose.add_argument("query")
    diagnose.add_argument("--language", default="pt-BR")
    diagnose.add_argument("--max-results", type=int, default=5)
    voice = sub.add_parser("voice", help="sessao push-to-talk local")
    voice.add_argument("--once", action="store_true")
    voice_mode = voice.add_mutually_exclusive_group()
    voice_mode.add_argument(
        "--voice", choices=("piper", "windows_sapi"), dest="voice_mode"
    )
    voice_mode.add_argument(
        "--no-voice", action="store_const", const="silent", dest="voice_mode"
    )
    sub.add_parser("voice-devices", help="lista dispositivos de audio")
    sub.add_parser(
        "voice-windows-list",
        help="lista vozes SAPI, System.Speech e WinRT instaladas",
    )
    light_poc = sub.add_parser(
        "voice-light-ptbr-poc",
        help="gera comparação auditiva local leve pt-BR",
    )
    light_poc.add_argument("--windows-only", action="store_true")
    light_poc.add_argument("--skip-stt", action="store_true")
    light_poc.add_argument("--no-play", action="store_true")
    voice_configure = sub.add_parser(
        "voice-configure",
        help="seleciona e testa dispositivos de audio",
    )
    voice_configure.add_argument("--seconds", type=float, default=3.0)
    sub.add_parser(
        "voice-model-info",
        help="mostra modelo STT local configurado",
    )
    voice_diagnose = sub.add_parser(
        "voice-diagnose",
        help="testa audio, STT e TTS sem chamar Qwen",
    )
    voice_diagnose.add_argument("--seconds", type=float, default=4.0)
    pronunciation = sub.add_parser(
        "voice-pronunciation-test",
        help="gera frases de pronúncia com Piper local",
    )
    pronunciation.add_argument("--output", default=None)
    pronunciation.add_argument("--play", action="store_true")
    sub.add_parser(
        "voice-playback-diagnose",
        help="compara WAV bruto, chunks e reprodução Piper",
    )
    sub.add_parser(
        "voice-phoneme-diagnose",
        help="inspeciona fonemas pt-BR e gera WAVs de diagnóstico",
    )
    sub.add_parser(
        "voice-piper-compare",
        help="compara as vozes pt-BR do Piper disponíveis localmente",
    )
    model_compare = sub.add_parser(
        "voice-compare-models",
        help="gera e reproduz comparação justa entre modelos Piper pt-BR",
    )
    model_compare.add_argument("--no-play", action="store_true")
    selection = model_compare.add_mutually_exclusive_group()
    selection.add_argument(
        "--select",
        choices=("miro", "jeff", "cadu", "dii", "faber"),
    )
    selection.add_argument("--no-select", action="store_true")
    ask = sub.add_parser("ask", help="executa uma solicitacao pelo supervisor")
    ask.add_argument("prompt")
    ask.add_argument("--approve-destructive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        manager = RuntimeManager(settings)
        if args.command == "config":
            _print(
                {
                    "backend": settings.backend.name,
                    "legacy": settings.backend.legacy,
                    "model": settings.model_path,
                    "executable": settings.server_executable,
                    "command": settings.server_command(),
                    "allowed_roots": settings.allowed_roots,
                    "environment": {
                        "env_file": settings.env_file,
                        "env_file_exists": settings.env_file.is_file(),
                        "env_file_loaded": settings.env_file_loaded,
                    },
                    "web": {
                        "enabled": settings.web_enabled,
                        "provider": settings.web_search_provider,
                        "search_url": settings.web_search_url,
                        "api_key_configured": bool(
                            settings.web_search_api_key
                        ),
                        "timeout": settings.web_timeout,
                        "max_download_bytes": settings.web_max_download_bytes,
                        "max_text_chars": settings.web_max_text_chars,
                        "max_pdf_pages": settings.web_max_pdf_pages,
                        "allowed_domains": settings.web_allowed_domains,
                        "blocked_domains": settings.web_blocked_domains,
                        "query_expansion_enabled": settings.web_query_expansion_enabled,
                        "max_query_variants": settings.web_max_query_variants,
                        "cross_language_search": settings.web_cross_language_search,
                        "default_region": settings.web_default_region,
                        "min_result_relevance": settings.web_min_result_relevance,
                        "min_source_relevance": settings.web_min_source_relevance,
                        "relevance_top_k": settings.web_relevance_top_k,
                        "max_research_corrections": settings.web_max_research_corrections,
                        "max_total_searches": settings.web_max_total_searches,
                        "max_total_opens": settings.web_max_total_opens,
                    },
                    "voice": {
                        "enabled": settings.voice_enabled,
                        "stt_provider": settings.voice_stt_provider,
                        "stt_model": settings.voice_stt_model,
                        "stt_device": settings.voice_stt_device,
                        "stt_compute_type": settings.voice_stt_compute_type,
                        "stt_language": settings.voice_stt_language,
                        "stt_threads": settings.voice_stt_threads,
                        "tts_provider": settings.voice_tts_provider,
                        "mode": settings.voice_mode,
                        "tts_model": settings.voice_tts_model,
                        "piper_voice": settings.voice_piper_voice,
                        "piper_requested_voice": (
                            settings.voice_piper_requested_voice
                        ),
                        "piper_config": settings.voice_piper_config_path,
                        "piper_fallback": settings.voice_piper_fallback,
                        "windows_voice_id": settings.voice_windows_voice_id,
                        "windows_rate": settings.voice_windows_rate,
                        "windows_volume": settings.voice_windows_volume,
                        "fallback_provider": (
                            settings.voice_fallback_provider
                        ),
                        "tts_voice": settings.voice_tts_voice,
                        "tts_device": settings.voice_tts_device,
                        "input_device": settings.voice_input_device,
                        "output_device": settings.voice_output_device,
                        "input_device_name": settings.voice_input_device_name,
                        "output_device_name": settings.voice_output_device_name,
                        "sample_rate": settings.voice_sample_rate,
                        "confirm_transcription": settings.voice_confirm_transcription,
                        "keep_recordings": settings.voice_keep_recordings,
                        "tts_streaming": settings.voice_tts_streaming,
                        "tts_rate": settings.voice_tts_rate,
                        "tts_length_scale": (
                            1.0 / settings.voice_tts_rate
                        ),
                        "style": settings.voice_style,
                        "translate_common_status_terms": (
                            settings.voice_translate_common_status_terms
                        ),
                        "piper_use_model_default_noise": (
                            settings.voice_piper_use_model_default_noise
                        ),
                        "sentence_pause_ms": (
                            settings.voice_sentence_pause_ms
                        ),
                        "paragraph_pause_ms": (
                            settings.voice_paragraph_pause_ms
                        ),
                        "post_processing": (
                            settings.voice_post_processing
                        ),
                    },
                }
            )
        elif args.command == "start":
            _print(manager.start(args.wait))
        elif args.command == "stop":
            _print(manager.stop())
        elif args.command == "status":
            _print(manager.status())
        elif args.command == "search-diagnose":
            result = _web_client(settings).search(
                query=args.query,
                max_results=args.max_results,
                language=args.language,
            )
            _print(
                {
                    "configuration": {
                        "provider": settings.web_search_provider,
                        "search_url": settings.web_search_url,
                        "api_key": (
                            "<hidden>"
                            if settings.web_search_api_key
                            else None
                        ),
                        "env_file": settings.env_file,
                        "env_file_loaded": settings.env_file_loaded,
                    },
                    "result": result,
                }
            )
        elif args.command == "voice-devices":
            from .voice.audio import SoundDeviceAudio

            audio = SoundDeviceAudio()
            default_input, default_output = audio.defaults()
            _print(
                {
                    "ok": True,
                    "default_input": default_input,
                    "default_output": default_output,
                    "devices": [
                        item.as_dict() for item in audio.devices()
                    ],
                }
            )
        elif args.command == "voice-configure":
            from .voice.audio import SoundDeviceAudio
            from .voice.configure import VoiceConfigurator
            from .voice.errors import VoiceDisabled
            from .voice.policy import ConsoleIO

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            _print(
                VoiceConfigurator(
                    settings,
                    SoundDeviceAudio(),
                    console=ConsoleIO(),
                ).run(test_seconds=args.seconds)
            )
        elif args.command == "voice-model-info":
            from .voice.configure import voice_model_info

            _print(voice_model_info(settings))
        elif args.command == "voice-diagnose":
            from .voice.diagnostic import VoiceDiagnostic
            from .voice.errors import VoiceDisabled

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            audio, stt, tts, voice_logger = _voice_stack(settings)
            try:
                _print(
                    VoiceDiagnostic(
                        settings, audio, stt, tts, voice_logger
                    ).run(capture_seconds=args.seconds)
                )
            finally:
                tts.close()
        elif args.command == "voice-pronunciation-test":
            from pathlib import Path

            from .voice.audio import SoundDeviceAudio
            from .voice.pronunciation import generate_pronunciation_test
            from .voice.tts import PiperTTS

            audio = SoundDeviceAudio()
            piper = PiperTTS(
                settings.voice_tts_model,
                audio,
                config_path=settings.voice_piper_config_path,
                output_device=settings.voice_output_device,
                output_device_name=settings.voice_output_device_name,
                interrupt_key=settings.voice_interrupt_key,
            )
            try:
                _print(
                    generate_pronunciation_test(
                        settings,
                        piper,
                        audio,
                        output=Path(args.output) if args.output else None,
                        play=args.play,
                    )
                )
            finally:
                piper.close()
        elif args.command == "voice-playback-diagnose":
            from .voice.errors import VoiceDisabled
            from .voice.quality import playback_diagnose

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            audio, stt, tts, voice_logger = _voice_stack(settings)
            del stt, voice_logger
            try:
                _print(playback_diagnose(settings, audio, tts))
            finally:
                tts.close()
        elif args.command == "voice-phoneme-diagnose":
            from .voice.errors import VoiceDisabled
            from .voice.quality import phoneme_diagnose

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            audio, stt, tts, voice_logger = _voice_stack(settings)
            del audio, stt, voice_logger
            try:
                _print(phoneme_diagnose(settings, tts))
            finally:
                tts.close()
        elif args.command == "voice-piper-compare":
            from .voice.errors import VoiceDisabled
            from .voice.quality import piper_compare

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            audio, stt, tts, voice_logger = _voice_stack(settings)
            del voice_logger
            try:
                tts.close()
                _print(piper_compare(settings, audio, stt))
            finally:
                stt.close()
        elif args.command == "voice-compare-models":
            from .voice.errors import VoiceDisabled
            from .voice.model_compare import compare_piper_models
            from .voice.policy import ConsoleIO

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            audio, stt, tts, voice_logger = _voice_stack(settings)
            del voice_logger
            tts.close()
            try:
                _print(
                    compare_piper_models(
                        settings,
                        audio,
                        stt,
                        play=not args.no_play,
                        selection=args.select,
                        prompt_for_selection=not args.no_select,
                        console=ConsoleIO(),
                    )
                )
            finally:
                stt.close()
        elif args.command == "voice-windows-list":
            from .voice.windows_speech import list_windows_voices

            _print(list_windows_voices())
        elif args.command == "voice-light-ptbr-poc":
            from .voice.light_compare import (
                generate_light_comparison,
                play_light_candidates,
            )

            result = generate_light_comparison(
                settings,
                include_stt=not args.skip_stt,
                windows_only=args.windows_only,
            )
            if not args.no_play:
                result["played"] = play_light_candidates(settings, result)
            _print(result)
        elif args.command == "voice":
            from .voice.errors import VoiceDisabled
            from .voice.policy import ConsoleIO, VoiceActionApprover
            from .voice.session import VoiceSession

            if not settings.voice_enabled:
                raise VoiceDisabled("voz desativada por VOICE_ENABLED=false")
            if not manager.status()["healthy"]:
                manager.start(240)
            console = ConsoleIO()
            approval = VoiceActionApprover(console)
            registry = _registry(settings, approval=approval)
            supervisor = Supervisor(
                settings,
                LlamaClient(settings.base_url, settings.timeout),
                registry,
            )
            audio, stt, tts, voice_logger = _voice_stack(
                settings, mode=args.voice_mode
            )
            try:
                result = VoiceSession(
                    settings,
                    supervisor,
                    audio,
                    stt,
                    tts,
                    voice_logger,
                    console=console,
                ).run(once=args.once)
            finally:
                tts.close()
            _print(result)
        else:
            registry = _registry(
                settings,
                approval=_approval(
                    getattr(args, "approve_destructive", False)
                ),
            )
            if args.command == "tools":
                _print(registry.specs())
            elif args.command == "ask":
                client = LlamaClient(settings.base_url, settings.timeout)
                _print(Supervisor(settings, client, registry).run(args.prompt))
        return 0
    except WebError as exc:
        value = {"ok": False, "error": exc.code, "message": str(exc)}
        if exc.details:
            value["details"] = exc.details
        _print(value)
        return 1
    except Exception as exc:
        from .voice.errors import VoiceError

        if isinstance(exc, VoiceError):
            _print(exc.as_dict())
            return 1
        _print({"ok": False, "error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
