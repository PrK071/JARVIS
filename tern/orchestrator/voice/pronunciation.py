from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from .models import SynthesisOptions
from .normalize import normalize_for_speech


PHRASES = (
    "Boa noite, senhor. Todos os sistemas estão operacionais.",
    "A inteligência artificial está operacional.",
    "A inteligência artificial analisou o diretório.",
    "O orquestrador enviou a tarefa ao Codex.",
    "O processador concluiu a operação.",
    "O servidor local foi reiniciado corretamente.",
    "A pesquisa encontrou três fontes relevantes.",
    "Não foi possível concluir a operação com segurança.",
    "O Qwen três ponto cinco concluiu o planejamento.",
    "O modelo GGUF foi carregado corretamente.",
    "O arquivo JSON foi validado pela API.",
    "Execute o comando no PowerShell.",
)


def generate_pronunciation_test(
    settings,
    piper,
    audio_backend,
    *,
    output: Path | None = None,
    play: bool = False,
) -> dict[str, Any]:
    root = (
        output.expanduser().resolve()
        if output is not None
        else Path(tempfile.mkdtemp(prefix="tern-piper-pronunciation-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    lexicon = Path(__file__).with_name("pronunciation_ptbr.json")
    reports = []
    for index, phrase in enumerate(PHRASES, 1):
        spoken = normalize_for_speech(
            phrase,
            "piper",
            settings.voice_style,
            lexicon_path=lexicon,
        )
        started = time.monotonic()
        result = piper.synthesize(
            spoken,
            SynthesisOptions(
                rate=settings.voice_tts_rate,
                volume=settings.voice_tts_volume,
                timeout_seconds=settings.voice_tts_timeout_seconds,
            ),
        )
        path = root / f"{index:02d}.wav"
        sf.write(
            path,
            result.samples,
            result.sample_rate,
            subtype="PCM_16",
        )
        interrupted = False
        if play:
            interrupted = audio_backend.play(
                result,
                output_device=settings.voice_output_device,
                output_device_name=settings.voice_output_device_name,
                interrupt_key=settings.voice_interrupt_key,
            )
        reports.append(
            {
                "index": index,
                "file": str(path),
                "duration_seconds": result.duration_seconds,
                "synthesis_seconds": time.monotonic() - started,
                "interrupted": interrupted,
            }
        )
        if interrupted:
            break
    return {
        "ok": True,
        "provider": "piper",
        "model": str(settings.voice_tts_model),
        "output": str(root),
        "generated": len(reports),
        "files": reports,
    }
