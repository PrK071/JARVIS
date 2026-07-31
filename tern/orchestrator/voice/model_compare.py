from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .configure import update_env_values
from .models import AudioData, AudioResult, SynthesisOptions, TranscriptionOptions
from .normalize import normalize_for_speech
from .policy import ConsoleIO
from .quality import _comparison_text, error_rates, read_piper_metadata
from .tts import PiperTTS
from .voices import piper_voice_aliases, validate_piper_voice_pair


MODEL_COMPARISON_PHRASES = (
    ("01-trabalho.wav", "O trabalho foi concluído corretamente."),
    ("02-trabalhando.wav", "Estou trabalhando na análise do projeto."),
    (
        "03-r-forte-fraco.wav",
        "O servidor reiniciou e agora está funcionando normalmente.",
    ),
    (
        "04-assistente.wav",
        "O orquestrador enviou o trabalho ao Codex.",
    ),
    (
        "05-tecnico.wav",
        "A inteligência artificial encontrou um erro no diretório.",
    ),
    (
        "06-frase-longa.wav",
        "O processador terminou a programação e verificou todos os arquivos.",
    ),
    (
        "07-jarvis.wav",
        "Boa noite, senhor. Todos os sistemas estão funcionando normalmente.",
    ),
    (
        "08-perguntas.wav",
        "Você deseja que eu continue o trabalho?",
    ),
    (
        "09-alerta.wav",
        "Atenção. Foi encontrado um problema durante a execução.",
    ),
    (
        "10-status.wav",
        "O hardware, o software e a pesquisa estão operacionais.",
    ),
    (
        "11-flexoes-trabalho.wav",
        "Trabalho, trabalhador, trabalhando, trabalhar e retrabalho.",
    ),
    (
        "12-erres.wav",
        "Rato, carro, porta, correto, ferramenta, servidor e diretório.",
    ),
    (
        "13-varredura.wav",
        "O sistema realizou uma varredura completa no repositório.",
    ),
    ("14-tarefa.wav", "A tarefa foi concluída sem erros."),
    (
        "15-ambiente.wav",
        "Preparando o ambiente de desenvolvimento.",
    ),
)
MODEL_TARGET_WORDS = (
    "trabalho",
    "trabalhador",
    "trabalhando",
    "trabalhar",
    "retrabalho",
    "corretamente",
    "orquestrador",
    "diretório",
    "servidor",
    "ferramenta",
    "programação",
    "inteligência",
    "artificial",
)
PLAYBACK_ORDER = ("miro", "jeff", "cadu", "dii", "faber")


def compare_piper_models(
    settings: Any,
    audio: Any,
    stt: Any,
    *,
    play: bool = True,
    selection: str | None = None,
    prompt_for_selection: bool = True,
    console: ConsoleIO | None = None,
) -> dict[str, Any]:
    console = console or ConsoleIO()
    root = settings.state_dir / "piper-model-comparison"
    root.mkdir(parents=True, exist_ok=True)
    aliases = piper_voice_aliases(Path(__file__).resolve().parents[3])
    lexicon = Path(__file__).with_name("pronunciation_ptbr.json")
    normalized_phrases = tuple(
        (
            filename,
            text,
            normalize_for_speech(
                text,
                "piper",
                settings.voice_style,
                lexicon_path=lexicon,
            ),
        )
        for filename, text in MODEL_COMPARISON_PHRASES
    )
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "normalization_identical": True,
        "normalization": [item[2] for item in normalized_phrases],
        "voices": {},
    }
    available_audio: dict[str, AudioResult] = {}

    for alias in PLAYBACK_ORDER:
        model_path, config_path = aliases[alias]
        voice_report: dict[str, Any] = {
            "alias": alias,
            "model": str(model_path),
            "config": str(config_path),
            "available": model_path.is_file() and config_path.is_file(),
        }
        report["voices"][alias] = voice_report
        if not voice_report["available"]:
            voice_report["rejected"] = True
            voice_report["rejection_reason"] = "modelo ou JSON local ausente"
            continue
        try:
            metadata = validate_piper_voice_pair(model_path, config_path)
        except ValueError as exc:
            voice_report["rejected"] = True
            voice_report["rejection_reason"] = str(exc)
            continue

        voice_root = root / alias
        voice_root.mkdir(parents=True, exist_ok=True)
        provider = PiperTTS(
            model_path,
            audio,
            config_path=config_path,
            output_device=settings.voice_output_device,
            output_device_name=settings.voice_output_device_name,
            interrupt_key=settings.voice_interrupt_key,
        )
        rss_before = _rss_bytes()
        cold_started = time.monotonic()
        try:
            provider._load()
            cold_load_seconds = time.monotonic() - cold_started
            rss_loaded = _rss_bytes()
            items = []
            default_audio: list[np.ndarray] = []
            rates: set[int] = set()
            synthesis_times = []
            for filename, expected, spoken in normalized_phrases:
                started = time.monotonic()
                result = provider.synthesize(
                    spoken,
                    SynthesisOptions(
                        rate=1.0,
                        volume=1.0,
                        timeout_seconds=settings.voice_tts_timeout_seconds,
                    ),
                )
                synthesis_seconds = time.monotonic() - started
                synthesis_times.append(synthesis_seconds)
                rates.add(result.sample_rate)
                path = voice_root / filename
                sf.write(
                    path,
                    result.samples,
                    result.sample_rate,
                    subtype="PCM_16",
                )
                transcript, stt_error = _transcribe(
                    stt, result, settings.voice_stt_timeout_seconds
                )
                wer, cer = error_rates(expected, transcript)
                target_errors = _target_errors(expected, transcript)
                items.append(
                    {
                        "file": str(path),
                        "expected": expected,
                        "spoken": spoken,
                        "transcription": transcript,
                        "wer": wer,
                        "cer": cer,
                        "target_errors": target_errors,
                        "synthesis_seconds": synthesis_seconds,
                        "duration_seconds": result.duration_seconds,
                        "rtf": (
                            synthesis_seconds / result.duration_seconds
                            if result.duration_seconds
                            else None
                        ),
                        "stt_error": stt_error,
                    }
                )
                default_audio.append(
                    np.asarray(result.samples, dtype=np.float32).reshape(-1)
                )
            if len(rates) != 1:
                raise ValueError("sample rate variou dentro da mesma voz")
            sample_rate = rates.pop()
            complete = _join_with_pause(default_audio, sample_rate, 700)
            complete_path = root / f"{alias}-completo.wav"
            sf.write(complete_path, complete, sample_rate, subtype="PCM_16")

            slower_audio = []
            slower_times = []
            for _filename, _expected, spoken in normalized_phrases:
                started = time.monotonic()
                result = provider.synthesize(
                    spoken,
                    SynthesisOptions(
                        rate=0.94,
                        volume=1.0,
                        timeout_seconds=settings.voice_tts_timeout_seconds,
                    ),
                )
                slower_times.append(time.monotonic() - started)
                slower_audio.append(
                    np.asarray(result.samples, dtype=np.float32).reshape(-1)
                )
            slower = _join_with_pause(slower_audio, sample_rate, 700)
            slower_path = root / f"{alias}-rate-094.wav"
            sf.write(slower_path, slower, sample_rate, subtype="PCM_16")

            cancel_event = threading.Event()
            cancel_event.set()
            cancel_started = time.monotonic()
            cancel_result = AudioResult(
                samples=complete[:sample_rate],
                sample_rate=sample_rate,
                duration_seconds=min(1.0, complete.size / sample_rate),
                provider="piper-comparison",
            )
            cancelled = audio.play(
                cancel_result,
                output_device=settings.voice_output_device,
                output_device_name=settings.voice_output_device_name,
                stop_event=cancel_event,
                interrupt_key=settings.voice_interrupt_key,
            )
            cancellation_seconds = time.monotonic() - cancel_started
            mean_wer = sum(item["wer"] for item in items) / len(items)
            mean_cer = sum(item["cer"] for item in items) / len(items)
            target_errors = sorted(
                {
                    error
                    for item in items
                    for error in item["target_errors"]
                }
            )
            absurd_duration = any(
                item["duration_seconds"] < 0.2
                or item["duration_seconds"] > 20
                for item in items
            )
            rejected = (
                absurd_duration
                or (mean_wer > 0.9 and len(target_errors) >= 5)
                or not cancelled
            )
            voice_report.update(
                {
                    "metadata": metadata,
                    "model_size_bytes": model_path.stat().st_size,
                    "model_sha256": _sha256(model_path),
                    "config_sha256": _sha256(config_path),
                    "cold_load_seconds": cold_load_seconds,
                    "first_synthesis_seconds": synthesis_times[0],
                    "warm_synthesis_seconds": (
                        sum(synthesis_times[1:])
                        / max(1, len(synthesis_times) - 1)
                    ),
                    "time_to_first_audio_seconds": (
                        sum(synthesis_times[1:])
                        / max(1, len(synthesis_times) - 1)
                    ),
                    "mean_synthesis_seconds": (
                        sum(synthesis_times) / len(synthesis_times)
                    ),
                    "mean_rate_094_synthesis_seconds": (
                        sum(slower_times) / len(slower_times)
                    ),
                    "duration_seconds": complete.size / sample_rate,
                    "mean_wer": mean_wer,
                    "mean_cer": mean_cer,
                    "target_errors": target_errors,
                    "complete_wav": str(complete_path),
                    "rate_094_wav": str(slower_path),
                    "ram_loaded_bytes": max(0, rss_loaded - rss_before),
                    "vram_bytes": 0,
                    "cancellation_seconds": cancellation_seconds,
                    "cancelled": cancelled,
                    "audio_resumed": False,
                    "rejected": rejected,
                    "rejection_reason": (
                        "duração, reconhecimento ou cancelamento inválido"
                        if rejected
                        else None
                    ),
                    "items": items,
                }
            )
            available_audio[alias] = AudioResult(
                samples=complete,
                sample_rate=sample_rate,
                duration_seconds=complete.size / sample_rate,
                provider=f"piper-{alias}",
            )
        except Exception as exc:
            voice_report["rejected"] = True
            voice_report["rejection_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            provider.close()
            del provider
            gc.collect()

    report["technical_ranking"] = sorted(
        (
            {
                "voice": alias,
                "wer": value["mean_wer"],
                "cer": value["mean_cer"],
            }
            for alias, value in report["voices"].items()
            if value.get("available") and not value.get("rejected")
        ),
        key=lambda item: (item["wer"], item["cer"]),
    )
    report_path = root / "comparison-report.json"
    markdown_path = root / "comparison-report.md"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown_report(report), encoding="utf-8"
    )

    playback_cancelled = False
    if play:
        for alias in PLAYBACK_ORDER:
            result = available_audio.get(alias)
            if result is None:
                continue
            console.write(f"\n[comparação] voz: {alias.upper()}")
            if audio.play(
                result,
                output_device=settings.voice_output_device,
                output_device_name=settings.voice_output_device_name,
                interrupt_key=settings.voice_interrupt_key,
            ):
                playback_cancelled = True
                console.write("[comparação] reprodução interrompida")
                break

    selected = selection
    if selected is None and prompt_for_selection:
        selected = _prompt_selection(console)
    if selected is not None:
        selected = selected.casefold()
        selected_report = report["voices"].get(selected) or {}
        if not selected_report.get("available") or selected_report.get(
            "rejected"
        ):
            raise ValueError(f"voz selecionada indisponível: {selected}")
        update_env_values(
            settings.env_file, {"VOICE_PIPER_VOICE": selected}
        )
        console.write(f"[voz] seleção salva: {selected}")

    return {
        "ok": any(
            value.get("available")
            for value in report["voices"].values()
        ),
        "output": str(root),
        "report_json": str(report_path),
        "report_markdown": str(markdown_path),
        "voices": {
            alias: {
                key: value.get(key)
                for key in (
                    "available",
                    "rejected",
                    "rejection_reason",
                    "model_size_bytes",
                    "metadata",
                    "cold_load_seconds",
                    "first_synthesis_seconds",
                    "warm_synthesis_seconds",
                    "time_to_first_audio_seconds",
                    "mean_wer",
                    "mean_cer",
                    "target_errors",
                    "complete_wav",
                    "rate_094_wav",
                    "ram_loaded_bytes",
                    "vram_bytes",
                    "cancellation_seconds",
                )
            }
            for alias, value in report["voices"].items()
        },
        "technical_ranking": report["technical_ranking"],
        "selected": selected,
        "selection_provisional": selected == "miro",
        "playback_cancelled": playback_cancelled,
    }


def _transcribe(
    stt: Any, audio: AudioResult, timeout_seconds: int
) -> tuple[str, str | None]:
    try:
        result = stt.transcribe(
            AudioData(
                samples=audio.samples,
                sample_rate=audio.sample_rate,
                duration_seconds=audio.duration_seconds,
                rms=float(np.sqrt(np.mean(np.square(audio.samples)))),
                peak=float(np.max(np.abs(audio.samples))),
            ),
            TranscriptionOptions(
                language="pt", timeout_seconds=timeout_seconds
            ),
        )
        return result.text, None
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _target_errors(expected: str, actual: str) -> list[str]:
    expected_words = set(_comparison_text(expected).split())
    actual_words = set(_comparison_text(actual).split())
    return [
        word
        for word in MODEL_TARGET_WORDS
        if _comparison_text(word) in expected_words
        and _comparison_text(word) not in actual_words
    ]


def _join_with_pause(
    values: list[np.ndarray], sample_rate: int, pause_ms: int
) -> np.ndarray:
    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
    joined: list[np.ndarray] = []
    for index, value in enumerate(values):
        if index:
            joined.append(pause)
        joined.append(np.asarray(value, dtype=np.float32).reshape(-1))
    return np.concatenate(joined) if joined else np.empty(0, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return 0


def _prompt_selection(console: ConsoleIO) -> str:
    console.write("\nEscolha a voz:")
    console.write("1 - Miro")
    console.write("2 - Jeff")
    console.write("3 - Cadu")
    console.write("4 - Dii")
    console.write("5 - Manter Faber")
    value = console.read("Escolha: ").strip()
    mapping = {
        "1": "miro",
        "2": "jeff",
        "3": "cadu",
        "4": "dii",
        "5": "faber",
    }
    if value not in mapping:
        raise ValueError("seleção de voz inválida")
    return mapping[value]


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Comparação de modelos Piper pt-BR",
        "",
        "| voz | tamanho | sample rate | síntese | duração | WER | CER | erros-alvo |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for alias in PLAYBACK_ORDER:
        value = report["voices"][alias]
        if not value.get("available"):
            lines.append(
                f"| {alias} | — | — | — | — | — | — | indisponível |"
            )
            continue
        metadata = value.get("metadata") or {}
        lines.append(
            "| {alias} | {size:.1f} MiB | {rate} Hz | {synth:.3f} s | "
            "{duration:.2f} s | {wer:.3f} | {cer:.3f} | {errors} |".format(
                alias=alias,
                size=value["model_size_bytes"] / 1024 / 1024,
                rate=metadata.get("sample_rate", 0),
                synth=value["warm_synthesis_seconds"],
                duration=value["duration_seconds"],
                wer=value["mean_wer"],
                cer=value["mean_cer"],
                errors=", ".join(value["target_errors"]) or "nenhum",
            )
        )
    lines.extend(["", "## WAVs completos", ""])
    for alias in PLAYBACK_ORDER:
        path = report["voices"][alias].get("complete_wav")
        if path:
            lines.append(f"- **{alias}**: `{path}`")
    lines.extend(
        [
            "",
            "A classificação automática é apenas um indicador. A escolha final "
            "deve considerar audição humana.",
            "",
        ]
    )
    return "\n".join(lines)
