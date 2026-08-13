"""Benchmark reproducível do Piper completo versus progressivo."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import threading
import time

from tern.orchestrator.config import load_settings
from tern.orchestrator.voice.audio import SoundDeviceAudio
from tern.orchestrator.voice.models import SynthesisOptions
from tern.orchestrator.voice.streaming import segment_for_speech
from tern.orchestrator.voice.tts import PiperTTS


TEXT = (
    "A pesquisa recente encontrou fontes jornalísticas relevantes sobre "
    "inteligência artificial. A primeira matéria descreve investimentos de "
    "empresas de tecnologia em novos sistemas. A segunda explica mudanças "
    "regulatórias e seus possíveis efeitos. Os links completos e as datas "
    "permanecem disponíveis na tela. Este resumo falado evita ler URLs e "
    "listas extensas de fontes."
)


class ImmediateAudio:
    """Consumidor sem hardware; mede síntese real, não duração do alto-falante."""

    def play(self, *_args, **_kwargs) -> bool:
        return False


class CancellingAudio:
    def __init__(self, audio, segment: int, delay: float):
        self.audio = audio
        self.segment = segment
        self.delay = delay
        self.calls = 0

    def play(self, *args, **kwargs) -> bool:
        self.calls += 1
        stop_event = kwargs.get("stop_event")
        if self.calls == self.segment and stop_event is not None:
            threading.Timer(self.delay, stop_event.set).start()
        return self.audio.play(*args, **kwargs)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_rss() -> int:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, os.getpid())
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return 0
        return int(counters.WorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--cancel-on-segment", type=int)
    parser.add_argument("--cancel-delay", type=float, default=0.25)
    parser.add_argument("--text", default=TEXT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    audio = SoundDeviceAudio() if args.play else ImmediateAudio()
    if args.cancel_on_segment:
        audio = CancellingAudio(
            audio, args.cancel_on_segment, args.cancel_delay
        )
    tts = PiperTTS(
        settings.voice_tts_model,
        audio,
        output_device=settings.voice_output_device,
        output_device_name=settings.voice_output_device_name,
        interrupt_key=settings.voice_interrupt_key,
    )
    options = SynthesisOptions(
        rate=settings.voice_tts_rate,
        volume=settings.voice_tts_volume,
        timeout_seconds=settings.voice_tts_timeout_seconds,
    )
    rss_before = process_rss()
    # Aquecimento exclui custo único de carregar ONNX das duas medições.
    tts.synthesize("Teste curto.", options)
    rss_loaded = process_rss()
    started = time.monotonic()
    full = tts.synthesize(args.text, options)
    legacy_first_audio = time.monotonic() - started
    full_duration = full.duration_seconds
    del full
    gc.collect()
    rss_before_streaming = process_rss()
    events: list[tuple[str, dict]] = []
    metrics = tts.speak_streaming(
        args.text,
        options,
        chunk_min_characters=settings.voice_tts_chunk_min_characters,
        chunk_max_characters=settings.voice_tts_chunk_max_characters,
        queue_size=settings.voice_tts_queue_size,
        event_callback=lambda event, values: events.append((event, values)),
    )
    rss_after = process_rss()
    result = {
        "ok": True,
        "real_piper": True,
        "playback": "hardware" if args.play else "immediate_consumer",
        "cancel_on_segment": args.cancel_on_segment,
        "characters": len(args.text),
        "segments_text": segment_for_speech(
            args.text,
            minimum=settings.voice_tts_chunk_min_characters,
            maximum=settings.voice_tts_chunk_max_characters,
        ),
        "legacy": {
            "time_to_first_audio_seconds": legacy_first_audio,
            "audio_duration_seconds": full_duration,
        },
        "progressive": metrics,
        "memory": {
            "piper_loaded_rss_bytes": max(0, rss_loaded - rss_before),
            "streaming_additional_rss_bytes": max(
                0, rss_after - rss_before_streaming
            ),
        },
        "additional_vram_bytes": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
