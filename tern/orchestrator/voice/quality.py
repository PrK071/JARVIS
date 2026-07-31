from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models import AudioData, AudioResult, SynthesisOptions, TranscriptionOptions
from .normalize import normalize_for_speech
from .tts import PiperTTS, rate_to_length_scale


PLAYBACK_PHRASE = (
    "Eu concluí o trabalho e o sistema está funcionando corretamente."
)
PHONEME_WORDS = (
    "trabalho",
    "trabalhar",
    "trabalhando",
    "trabalhador",
    "diretório",
    "artificial",
    "orquestrador",
    "processador",
    "programação",
    "servidor",
    "ferramenta",
    "corretamente",
    "pesquisa",
    "arquivo",
)
PHONEME_PHRASES = (
    "Eu concluí o trabalho corretamente.",
    "O sistema está trabalhando normalmente.",
    "A inteligência artificial analisou o diretório.",
    "O orquestrador enviou a tarefa ao processador.",
)
COMPARE_PHRASES = (
    "Eu concluí o trabalho corretamente.",
    "O sistema terminou o trabalho e está funcionando corretamente.",
    "O sistema está trabalhando normalmente.",
    "A inteligência artificial analisou o diretório.",
    "O orquestrador enviou a tarefa ao Codex.",
    "O servidor local está funcionando.",
    "O processador terminou a programação.",
    "Boa noite, senhor. Todos os sistemas estão operacionais.",
    "A pesquisa encontrou três fontes relevantes.",
    "O hardware e o software estão funcionando normalmente.",
    "Não foi possível concluir a operação.",
)
TARGET_WORDS = (
    "trabalho",
    "trabalhando",
    "artificial",
    "diretório",
    "orquestrador",
    "processador",
    "programação",
    "servidor",
    "corretamente",
)


def read_piper_metadata(model_path: Path) -> dict[str, Any]:
    config_path = Path(str(model_path) + ".json")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    audio = data.get("audio") or {}
    inference = data.get("inference") or {}
    espeak = data.get("espeak") or {}
    return {
        "config_path": str(config_path),
        "sample_rate": int(audio["sample_rate"]),
        "espeak_voice": str(espeak.get("voice") or ""),
        "phoneme_type": data.get("phoneme_type"),
        "noise_scale": float(inference.get("noise_scale", 0.667)),
        "length_scale": float(inference.get("length_scale", 1.0)),
        "noise_w": float(inference.get("noise_w", 0.8)),
    }


def float_to_pcm16(samples: np.ndarray) -> bytes:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(values, -1.0, 32767.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype("<i2").tobytes()


def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    if len(pcm) % 2:
        raise ValueError("PCM int16 desalinhado")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as stream:
        pcm = stream.readframes(stream.getnframes())
        frames = stream.getnframes()
        rate = stream.getframerate()
        return {
            "path": str(path),
            "sample_rate": rate,
            "channels": stream.getnchannels(),
            "sample_width": stream.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / rate,
            "bytes": len(pcm),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        }


def _collect_piper_chunks(
    voice: Any, text: str, synthesis_config: Any
) -> tuple[bytes, int, list[int]]:
    values: list[bytes] = []
    rates: list[int] = []
    for chunk in voice.synthesize(text, syn_config=synthesis_config):
        raw = bytes(chunk.audio_int16_bytes)
        if len(raw) % 2:
            raise ValueError("chunk PCM int16 desalinhado")
        values.append(raw)
        rates.append(int(chunk.sample_rate))
    if not values:
        raise ValueError("Piper retornou áudio vazio")
    if len(set(rates)) != 1:
        raise ValueError("sample rate variou entre chunks")
    return b"".join(values), rates[0], [len(item) for item in values]


def playback_diagnose(settings: Any, audio: Any, piper: PiperTTS) -> dict[str, Any]:
    from piper import SynthesisConfig

    root = settings.state_dir / "piper-playback-diagnose"
    root.mkdir(parents=True, exist_ok=True)
    metadata = read_piper_metadata(settings.voice_tts_model)
    voice = piper._load()
    config = SynthesisConfig(
        volume=settings.voice_tts_volume,
        length_scale=rate_to_length_scale(settings.voice_tts_rate),
    )
    started = time.monotonic()
    pcm, chunk_rate, chunk_sizes = _collect_piper_chunks(
        voice, PLAYBACK_PHRASE, config
    )
    synthesis_seconds = time.monotonic() - started
    raw_path = root / "piper-raw.wav"
    stream_path = root / "stream-reconstructed.wav"
    write_pcm16_wav(raw_path, pcm, chunk_rate)
    # Rebuild exactly what progressive playback received, preserving frames.
    reconstructed = b"".join(
        pcm[offset : offset + size]
        for offset, size in _offset_sizes(chunk_sizes)
    )
    write_pcm16_wav(stream_path, reconstructed, chunk_rate)
    raw = wav_info(raw_path)
    streamed = wav_info(stream_path)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    result = AudioResult(
        samples=samples,
        sample_rate=chunk_rate,
        duration_seconds=samples.size / chunk_rate,
        provider="piper",
    )
    measured_started = time.monotonic()
    interrupted = audio.play(
        result,
        output_device=settings.voice_output_device,
        output_device_name=settings.voice_output_device_name,
        interrupt_key=settings.voice_interrupt_key,
    )
    measured = time.monotonic() - measured_started
    independent_player = "sounddevice.play"
    try:
        output = audio.output_device(
            settings.voice_output_device,
            settings.voice_output_device_name,
        )
        audio.sd.play(
            samples,
            samplerate=chunk_rate,
            device=output.index,
            blocking=True,
        )
    except (AttributeError, TypeError):
        independent_player = "pipeline-only"
    hashes_match = raw["pcm_sha256"] == streamed["pcm_sha256"]
    return {
        "ok": hashes_match,
        "classification": "D" if hashes_match else "C",
        "phrase": PLAYBACK_PHRASE,
        "model": metadata,
        "raw_wav": raw,
        "stream_wav": streamed,
        "pcm_identical": hashes_match,
        "chunk_sample_rate": chunk_rate,
        "chunk_sizes": chunk_sizes,
        "chunks_int16_aligned": all(size % 2 == 0 for size in chunk_sizes),
        "playback_sample_rate": result.metadata.get(
            "playback_sample_rate", chunk_rate
        ),
        "resampled": result.metadata.get("resampled", False),
        "synthesis_seconds": synthesis_seconds,
        "measured_playback_seconds": measured,
        "relative_speed": (
            measured / raw["duration_seconds"]
            if raw["duration_seconds"]
            else None
        ),
        "interrupted": interrupted,
        "pipeline_player": "sounddevice.OutputStream",
        "independent_player": independent_player,
        "output": str(root),
    }


def _offset_sizes(sizes: Iterable[int]) -> Iterable[tuple[int, int]]:
    offset = 0
    for size in sizes:
        yield offset, size
        offset += size


def phoneme_diagnose(settings: Any, piper: PiperTTS) -> dict[str, Any]:
    import soundfile as sf
    from piper import SynthesisConfig

    root = settings.state_dir / "piper-phoneme-diagnose"
    root.mkdir(parents=True, exist_ok=True)
    voice = piper._load()
    metadata = read_piper_metadata(settings.voice_tts_model)
    synthesis_config = SynthesisConfig(
        volume=settings.voice_tts_volume,
        length_scale=rate_to_length_scale(settings.voice_tts_rate),
    )
    rows = []
    lexicon = Path(__file__).with_name("pronunciation_ptbr.json")
    for index, text in enumerate((*PHONEME_WORDS, *PHONEME_PHRASES), 1):
        normalized = unicodedata.normalize("NFC", text)
        spoken = normalize_for_speech(
            normalized,
            "piper",
            settings.voice_style,
            lexicon_path=lexicon,
        )
        phoneme_sentences = voice.phonemize(spoken)
        flattened = [
            symbol
            for sentence in phoneme_sentences
            for symbol in sentence
        ]
        ids = voice.phonemes_to_ids(flattened)
        synthesized = piper.synthesize(
            spoken,
            SynthesisOptions(
                rate=settings.voice_tts_rate,
                volume=settings.voice_tts_volume,
                timeout_seconds=settings.voice_tts_timeout_seconds,
            ),
        )
        path = root / f"{index:02d}-{_slug(text)}.wav"
        sf.write(
            path,
            synthesized.samples,
            synthesized.sample_rate,
            subtype="PCM_16",
        )
        explicit_path = None
        if text == "trabalho":
            explicit_audio = voice.phoneme_ids_to_audio(
                ids, syn_config=synthesis_config
            )
            explicit_path = root / "01-trabalho-explicit-phonemes.wav"
            sf.write(
                explicit_path,
                np.asarray(explicit_audio, dtype=np.float32),
                metadata["sample_rate"],
                subtype="PCM_16",
            )
        rows.append(
            {
                "original": text,
                "normalized": spoken,
                "language": metadata["espeak_voice"],
                "phonemes": "".join(flattened),
                "phoneme_ids": ids,
                "r_present": any(
                    item in {"r", "ɾ", "ʀ", "ʁ", "ɹ", "x", "ɽ"}
                    for item in flattened
                ),
                "wav": str(path),
                "explicit_phoneme_wav": (
                    str(explicit_path) if explicit_path else None
                ),
            }
        )
    return {
        "ok": metadata["espeak_voice"].casefold() == "pt-br",
        "locale": metadata["espeak_voice"],
        "fallback_en_us": metadata["espeak_voice"].casefold() == "en-us",
        "model": metadata,
        "items": rows,
        "output": str(root),
    }


def _slug(text: str) -> str:
    value = unicodedata.normalize("NFKD", text)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:45]


def _comparison_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _distance(expected: list[str], actual: list[str]) -> int:
    previous = list(range(len(actual) + 1))
    for index, left in enumerate(expected, 1):
        current = [index]
        for offset, right in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def error_rates(expected: str, actual: str) -> tuple[float, float]:
    expected_norm = _comparison_text(expected)
    actual_norm = _comparison_text(actual)
    expected_words = expected_norm.split()
    actual_words = actual_norm.split()
    wer = _distance(expected_words, actual_words) / max(1, len(expected_words))
    cer = _distance(list(expected_norm), list(actual_norm)) / max(
        1, len(expected_norm)
    )
    return wer, cer


def piper_compare(
    settings: Any, audio: Any, stt: Any
) -> dict[str, Any]:
    import soundfile as sf

    root = settings.state_dir / "piper-voice-compare"
    root.mkdir(parents=True, exist_ok=True)
    voice_paths = {
        "faber": settings.voice_tts_model,
        "cadu": settings.voice_tts_model.with_name(
            "pt_BR-cadu-medium.onnx"
        ),
        "jeff": settings.voice_tts_model.with_name(
            "pt_BR-jeff-medium.onnx"
        ),
    }
    report: dict[str, Any] = {
        "ok": True,
        "output": str(root),
        "voices": {},
    }
    for voice_name, model_path in voice_paths.items():
        if not model_path.is_file():
            report["voices"][voice_name] = {
                "available": False,
                "error": "modelo local ausente",
                "download_hint": (
                    "Baixe manualmente o ONNX e o JSON oficiais do Piper "
                    f"para {model_path.parent}; nenhum download foi iniciado."
                ),
            }
            continue
        provider = PiperTTS(model_path, audio)
        voice_root = root / voice_name
        voice_root.mkdir(parents=True, exist_ok=True)
        items = []
        rate_trials = []
        try:
            for index, phrase in enumerate(COMPARE_PHRASES, 1):
                started = time.monotonic()
                result = provider.synthesize(
                    phrase,
                    SynthesisOptions(
                        rate=settings.voice_tts_rate,
                        volume=settings.voice_tts_volume,
                        timeout_seconds=settings.voice_tts_timeout_seconds,
                    ),
                )
                synthesis_seconds = time.monotonic() - started
                path = voice_root / f"{index:02d}.wav"
                sf.write(
                    path,
                    result.samples,
                    result.sample_rate,
                    subtype="PCM_16",
                )
                transcript = ""
                error = None
                try:
                    transcript = stt.transcribe(
                        AudioData(
                            samples=result.samples,
                            sample_rate=result.sample_rate,
                            duration_seconds=result.duration_seconds,
                            rms=float(
                                np.sqrt(np.mean(np.square(result.samples)))
                            ),
                            peak=float(np.max(np.abs(result.samples))),
                        ),
                        TranscriptionOptions(
                            language="pt",
                            timeout_seconds=settings.voice_stt_timeout_seconds,
                        ),
                    ).text
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                wer, cer = error_rates(phrase, transcript)
                expected_words = set(_comparison_text(phrase).split())
                actual_words = set(_comparison_text(transcript).split())
                items.append(
                    {
                        "phrase": phrase,
                        "file": str(path),
                        "sample_rate": result.sample_rate,
                        "synthesis_seconds": synthesis_seconds,
                        "total_analysis_seconds": (
                            time.monotonic() - started
                        ),
                        "duration_seconds": result.duration_seconds,
                        "size_bytes": path.stat().st_size,
                        "transcription": transcript,
                        "wer": wer,
                        "cer": cer,
                        "target_words_divergent": [
                            word
                            for word in TARGET_WORDS
                            if _comparison_text(word) in expected_words
                            and _comparison_text(word) not in actual_words
                        ],
                        "error": error,
                    }
                )
            if voice_name == "faber":
                trial_phrase = COMPARE_PHRASES[0]
                for rate in (0.90, 0.94, 0.97, 1.00):
                    trial_started = time.monotonic()
                    trial_audio = provider.synthesize(
                        trial_phrase,
                        SynthesisOptions(
                            rate=rate,
                            volume=settings.voice_tts_volume,
                            timeout_seconds=settings.voice_tts_timeout_seconds,
                        ),
                    )
                    trial_synthesis_seconds = (
                        time.monotonic() - trial_started
                    )
                    trial_path = voice_root / (
                        f"rate-{str(rate).replace('.', '')}.wav"
                    )
                    sf.write(
                        trial_path,
                        trial_audio.samples,
                        trial_audio.sample_rate,
                        subtype="PCM_16",
                    )
                    transcript = stt.transcribe(
                        AudioData(
                            samples=trial_audio.samples,
                            sample_rate=trial_audio.sample_rate,
                            duration_seconds=trial_audio.duration_seconds,
                            rms=float(
                                np.sqrt(
                                    np.mean(np.square(trial_audio.samples))
                                )
                            ),
                            peak=float(
                                np.max(np.abs(trial_audio.samples))
                            ),
                        ),
                        TranscriptionOptions(
                            language="pt",
                            timeout_seconds=settings.voice_stt_timeout_seconds,
                        ),
                    ).text
                    wer, cer = error_rates(trial_phrase, transcript)
                    rate_trials.append(
                        {
                            "rate": rate,
                            "length_scale": rate_to_length_scale(rate),
                            "file": str(trial_path),
                            "duration_seconds": trial_audio.duration_seconds,
                            "synthesis_seconds": trial_synthesis_seconds,
                            "transcription": transcript,
                            "wer": wer,
                            "cer": cer,
                        }
                    )
        finally:
            provider.close()
        report["voices"][voice_name] = {
            "available": True,
            "model": str(model_path),
            "metadata": read_piper_metadata(model_path),
            "mean_wer": sum(item["wer"] for item in items) / len(items),
            "mean_cer": sum(item["cer"] for item in items) / len(items),
            "rate_trials": rate_trials,
            "parameter_profiles": {
                "model_default": {
                    "noise_scale": read_piper_metadata(model_path)[
                        "noise_scale"
                    ],
                    "noise_w": read_piper_metadata(model_path)["noise_w"],
                    "length_scale": read_piper_metadata(model_path)[
                        "length_scale"
                    ],
                },
                "application": {
                    "noise_scale": None,
                    "noise_w": None,
                    "resolved_to_model_defaults": True,
                },
                "adjusted_noise": {
                    "applied": False,
                    "reason": (
                        "nenhuma evidência justificou alterar os defaults"
                    ),
                },
            },
            "items": items,
        }
    ranking = [
        {
            "voice": name,
            "wer": value["mean_wer"],
            "cer": value["mean_cer"],
        }
        for name, value in report["voices"].items()
        if value.get("available")
    ]
    report["technical_ranking"] = sorted(
        ranking, key=lambda item: (item["wer"], item["cer"])
    )
    report_path = root / "comparison-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report"] = str(report_path)
    return report
