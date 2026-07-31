from __future__ import annotations

import gc
import hashlib
import importlib.metadata
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
        "03-trabalhador.wav",
        "O trabalhador verificou o retrabalho antes de continuar.",
    ),
    (
        "04-servidor.wav",
        "O servidor reiniciou e agora está funcionando normalmente.",
    ),
    (
        "05-orquestrador.wav",
        "O orquestrador enviou o trabalho ao Codex.",
    ),
    (
        "06-diretorio.wav",
        "A inteligência artificial encontrou um erro no diretório.",
    ),
    (
        "07-programacao.wav",
        "O processador terminou a programação e verificou todos os arquivos.",
    ),
    (
        "08-assistente.wav",
        "Boa noite, senhor. Todos os sistemas estão funcionando normalmente.",
    ),
    (
        "09-pergunta.wav",
        "Você deseja que eu continue o trabalho?",
    ),
    (
        "10-alerta.wav",
        "Atenção. Foi encontrado um problema durante a execução.",
    ),
    (
        "11-tecnico.wav",
        "O hardware, o software e a pesquisa estão operacionais.",
    ),
    (
        "12-familia-trabalho.wav",
        "Trabalho, trabalhador, trabalhando, trabalhar e retrabalho.",
    ),
    (
        "13-erres.wav",
        "Rato, carro, porta, correto, ferramenta, servidor e diretório.",
    ),
    (
        "14-repositorio.wav",
        "O sistema realizou uma varredura completa no repositório.",
    ),
    ("15-conclusao.wav", "A tarefa foi concluída sem erros."),
    (
        "16-desenvolvimento.wav",
        "Preparando o ambiente de desenvolvimento.",
    ),
    (
        "17-arquivo.wav",
        "O arquivo foi salvo e o diretório permanece disponível.",
    ),
    (
        "18-pesquisa.wav",
        "A pesquisa encontrou três fontes relevantes.",
    ),
    (
        "19-falha.wav",
        "Não foi possível completar a operação solicitada.",
    ),
    (
        "20-pronto.wav",
        "Estou pronto para executar a próxima tarefa.",
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
    "repositório",
    "pesquisa",
)
PLAYBACK_ORDER = ("miro", "jeff", "cadu", "dii", "faber")
PIPER_VOICE_METADATA: dict[str, dict[str, object]] = {
    "faber": {
        "name": "pt_BR-faber-medium",
        "origin": (
            "https://huggingface.co/rhasspy/piper-voices/tree/main/"
            "pt/pt_BR/faber/medium"
        ),
        "license": "MIT (repositório); CC0 (dataset)",
        "license_verified": True,
    },
    "miro": {
        "name": "miro_pt-BR",
        "origin": "https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro",
        "license": "não identificada no repositório/model card",
        "license_verified": False,
    },
    "jeff": {
        "name": "pt_BR-jeff-medium",
        "origin": (
            "https://huggingface.co/rhasspy/piper-voices/tree/main/"
            "pt/pt_BR/jeff/medium"
        ),
        "license": "MIT (repositório); CC0 (dataset)",
        "license_verified": True,
    },
    "cadu": {
        "name": "pt_BR-cadu-medium",
        "origin": (
            "https://huggingface.co/rhasspy/piper-voices/tree/main/"
            "pt/pt_BR/cadu/medium"
        ),
        "license": "MIT (repositório); CC0 (dataset)",
        "license_verified": True,
    },
    "dii": {
        "name": "dii_pt-BR",
        "origin": "https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii",
        "license": "não identificada no repositório/model card",
        "license_verified": False,
    },
}


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
    _remove_stale_wavs(root)
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
        "piper_version": _package_version("piper-tts"),
        "runtime_downloads": False,
        "network_during_synthesis": False,
        "normalization_identical": True,
        "normalization": [item[2] for item in normalized_phrases],
        "voices": {},
    }
    available_audio: dict[str, AudioResult] = {}

    for alias in PLAYBACK_ORDER:
        model_path, config_path = aliases[alias]
        registry = PIPER_VOICE_METADATA[alias]
        voice_report: dict[str, Any] = {
            "alias": alias,
            "name": registry["name"],
            "model": str(model_path),
            "config": str(config_path),
            "origin": registry["origin"],
            "license": registry["license"],
            "license_verified": registry["license_verified"],
            "available": model_path.is_file() and config_path.is_file(),
        }
        report["voices"][alias] = voice_report
        if not voice_report["available"]:
            voice_report["rejected"] = True
            voice_report["rejection_reason"] = "modelo ou JSON local ausente"
            continue
        voice_report["model_size_bytes"] = model_path.stat().st_size
        voice_report["config_size_bytes"] = config_path.stat().st_size
        voice_report["model_sha256"] = _sha256(model_path)
        voice_report["config_sha256"] = _sha256(config_path)
        try:
            metadata = validate_piper_voice_pair(model_path, config_path)
        except ValueError as exc:
            voice_report["rejected"] = True
            voice_report["rejection_reason"] = str(exc)
            continue
        voice_report.update(
            {
                "metadata": metadata,
                "channels": 1,
                "pcm_format": "PCM int16 nos WAVs; float32 interno",
            }
        )
        if not voice_report["license_verified"]:
            voice_report["rejected"] = True
            voice_report["rejection_reason"] = (
                "licença não identificada na origem; modelo preservado, "
                "mas excluído da comparação"
            )
            voice_report["runtime_compatible"] = (
                "não testado nesta rodada devido à licença"
            )
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
                omitted_words, substitutions = _word_differences(
                    expected, transcript
                )
                items.append(
                    {
                        "file": str(path),
                        "expected": expected,
                        "spoken": spoken,
                        "transcription": transcript,
                        "wer": wer,
                        "cer": cer,
                        "target_errors": target_errors,
                        "omitted_words": omitted_words,
                        "substitutions": substitutions,
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
                    "runtime_compatible": True,
                    "cold_load_seconds": cold_load_seconds,
                    "first_synthesis_seconds": synthesis_times[0],
                    "warm_synthesis_seconds": (
                        sum(synthesis_times[1:])
                        / max(1, len(synthesis_times) - 1)
                    ),
                    "time_to_first_audio_seconds": (
                        synthesis_times[0]
                    ),
                    "time_to_first_audio_basis": (
                        "síntese completa da primeira frase; reprodução "
                        "começa imediatamente depois"
                    ),
                    "mean_synthesis_seconds": (
                        sum(synthesis_times) / len(synthesis_times)
                    ),
                    "mean_rtf": (
                        sum(
                            item["rtf"]
                            for item in items
                            if item["rtf"] is not None
                        )
                        / max(
                            1,
                            sum(
                                item["rtf"] is not None
                                for item in items
                            ),
                        )
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

    def play_available() -> bool:
        for alias in PLAYBACK_ORDER:
            result = available_audio.get(alias)
            if result is None:
                continue
            console.write(f"\nREPRODUZINDO: {alias.upper()}")
            if audio.play(
                result,
                output_device=settings.voice_output_device,
                output_device_name=settings.voice_output_device_name,
                interrupt_key=settings.voice_interrupt_key,
            ):
                console.write("[comparação] reprodução interrompida")
                return True
        return False

    playback_cancelled = play_available() if play else False

    selected = selection
    while selected is None and prompt_for_selection:
        selected = _prompt_selection(console)
        if selected == "replay":
            playback_cancelled = play_available()
            selected = None
        elif selected == "none":
            selected = None
            prompt_for_selection = False
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
            value.get("available") and not value.get("rejected")
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


def _word_differences(
    expected: str, actual: str
) -> tuple[list[str], list[dict[str, str]]]:
    expected_words = _comparison_text(expected).split()
    actual_words = _comparison_text(actual).split()
    rows = len(expected_words) + 1
    columns = len(actual_words) + 1
    costs = [[0] * columns for _ in range(rows)]
    for index in range(rows):
        costs[index][0] = index
    for index in range(columns):
        costs[0][index] = index
    for row in range(1, rows):
        for column in range(1, columns):
            substitution = (
                0
                if expected_words[row - 1] == actual_words[column - 1]
                else 1
            )
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + substitution,
            )
    omitted: list[str] = []
    substitutions: list[dict[str, str]] = []
    row, column = len(expected_words), len(actual_words)
    while row or column:
        if (
            row
            and column
            and expected_words[row - 1] == actual_words[column - 1]
            and costs[row][column] == costs[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif (
            row
            and column
            and costs[row][column] == costs[row - 1][column - 1] + 1
        ):
            substitutions.append(
                {
                    "expected": expected_words[row - 1],
                    "actual": actual_words[column - 1],
                }
            )
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            omitted.append(expected_words[row - 1])
            row -= 1
        else:
            column -= 1
    omitted.reverse()
    substitutions.reverse()
    return omitted, substitutions


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


def _remove_stale_wavs(root: Path) -> None:
    """Remove only generated comparison WAVs, never models or directories."""
    if root.name != "piper-model-comparison":
        raise ValueError(f"diretório de comparação inesperado: {root}")
    for path in root.glob("*.wav"):
        path.unlink(missing_ok=True)
    for alias in PLAYBACK_ORDER:
        voice_root = root / alias
        if voice_root.is_dir():
            for path in voice_root.glob("*.wav"):
                path.unlink(missing_ok=True)


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


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "não instalado"


def _prompt_selection(console: ConsoleIO) -> str:
    while True:
        console.write("\nEscolha a voz:")
        console.write("1 - Miro")
        console.write("2 - Jeff")
        console.write("3 - Cadu")
        console.write("4 - Dii")
        console.write("5 - Manter Faber")
        console.write("6 - Reproduzir novamente")
        console.write("7 - Não escolher agora")
        value = console.read("Escolha: ").strip()
        if value == "6":
            return "replay"
        if value == "7":
            return "none"
        if value in {"1", "2", "3", "4", "5"}:
            break
    mapping = {
        "1": "miro",
        "2": "jeff",
        "3": "cadu",
        "4": "dii",
        "5": "faber",
    }
    return mapping[value]


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Comparação de modelos Piper pt-BR",
        "",
        "| voz | origem | licença | tamanho | sample rate | carregamento | primeira síntese | síntese quente | duração | WER | CER | erros-alvo |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for alias in PLAYBACK_ORDER:
        value = report["voices"][alias]
        if not value.get("available") or value.get("rejected"):
            lines.append(
                "| {alias} | {origin} | {license} | {size} | — | — | — | "
                "— | — | — | — | {reason} |".format(
                    alias=alias,
                    origin=value.get("origin", "—"),
                    license=value.get("license", "—"),
                    size=(
                        f"{value['model_size_bytes'] / 1024 / 1024:.1f} MiB"
                        if value.get("model_size_bytes")
                        else "—"
                    ),
                    reason=value.get("rejection_reason", "indisponível"),
                )
            )
            continue
        metadata = value.get("metadata") or {}
        lines.append(
            "| {alias} | {origin} | {license} | {size:.1f} MiB | {rate} Hz | "
            "{load:.3f} s | {first:.3f} s | {warm:.3f} s | "
            "{duration:.2f} s | {wer:.3f} | {cer:.3f} | {errors} |".format(
                alias=alias,
                origin=value["origin"],
                license=value["license"],
                size=value["model_size_bytes"] / 1024 / 1024,
                rate=metadata.get("sample_rate", 0),
                load=value["cold_load_seconds"],
                first=value["first_synthesis_seconds"],
                warm=value["warm_synthesis_seconds"],
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
    lines.extend(["", "## Detalhes", ""])
    for alias in PLAYBACK_ORDER:
        value = report["voices"][alias]
        metadata = value.get("metadata") or {}
        lines.extend(
            [
                f"### {alias}",
                "",
                f"- Nome real: {value.get('name', '—')}",
                f"- Modelo: `{value.get('model', '—')}`",
                f"- Configuração: `{value.get('config', '—')}`",
                f"- Origem: {value.get('origin', '—')}",
                f"- Licença: {value.get('license', '—')}",
                f"- SHA-256 ONNX: `{value.get('model_sha256', '—')}`",
                f"- SHA-256 JSON: `{value.get('config_sha256', '—')}`",
                f"- Sample rate: {metadata.get('sample_rate', '—')}",
                f"- Canais: {value.get('channels', '—')}",
                f"- PCM: {value.get('pcm_format', '—')}",
                f"- Carregamento: {_seconds(value.get('cold_load_seconds'))}",
                f"- Primeira síntese: {_seconds(value.get('first_synthesis_seconds'))}",
                f"- Síntese quente: {_seconds(value.get('warm_synthesis_seconds'))}",
                f"- Primeiro áudio: {_seconds(value.get('time_to_first_audio_seconds'))}",
                f"- RTF médio: {_number(value.get('mean_rtf'))}",
                f"- RAM aproximada: {_mib(value.get('ram_loaded_bytes'))}",
                f"- VRAM adicional: {_mib(value.get('vram_bytes'))}",
                f"- WER: {_number(value.get('mean_wer'))}",
                f"- CER: {_number(value.get('mean_cer'))}",
                "- Erros-alvo: "
                + (", ".join(value.get("target_errors") or []) or "nenhum"),
                f"- WAV completo: `{value.get('complete_wav', '—')}`",
                f"- Estado: {value.get('rejection_reason') or 'válida'}",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "A classificação automática é apenas um indicador. A escolha final "
            "deve considerar audição humana.",
            "",
        ]
    )
    return "\n".join(lines)


def _seconds(value: object) -> str:
    return f"{float(value):.3f} s" if value is not None else "—"


def _number(value: object) -> str:
    return f"{float(value):.3f}" if value is not None else "—"


def _mib(value: object) -> str:
    return f"{int(value) / 1024 / 1024:.1f} MiB" if value is not None else "—"
