from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import CaptureOptions, SoundDeviceAudio
from .devices import select_device
from .models import AudioResult
from .policy import ConsoleIO


def update_env_values(path: Path, values: dict[str, str]) -> None:
    lines = (
        path.read_text(encoding="utf-8").splitlines()
        if path.exists()
        else []
    )
    remaining = dict(values)
    result = []
    for line in lines:
        stripped = line.lstrip()
        replaced = False
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                result.append(f"{key}={remaining.pop(key)}")
                replaced = True
        if not replaced:
            result.append(line)
    if remaining:
        if result and result[-1].strip():
            result.append("")
        result.extend(f"{key}={value}" for key, value in remaining.items())
    temporary = path.with_name(path.name + ".voice-configure.tmp")
    temporary.write_text("\n".join(result) + "\n", encoding="utf-8")
    temporary.replace(path)


class VoiceConfigurator:
    def __init__(
        self,
        settings,
        audio: SoundDeviceAudio,
        *,
        console: ConsoleIO | None = None,
    ):
        self.settings = settings
        self.audio = audio
        self.console = console or ConsoleIO()

    def run(self, *, test_seconds: float = 3.0) -> dict[str, Any]:
        devices = self.audio.devices()
        inputs = [item for item in devices if item.input_channels > 0]
        outputs = [item for item in devices if item.output_channels > 0]
        self.console.write("Microfones:")
        for item in inputs:
            self.console.write(
                f"[{item.index}] {item.identity} | "
                f"{item.input_channels} canais | "
                f"{item.default_sample_rate} Hz"
            )
        input_value = self.console.read("Índice do microfone: ").strip()
        input_device = select_device(
            devices, input_value, direction="input"
        )
        self.console.write("Saídas:")
        for item in outputs:
            self.console.write(
                f"[{item.index}] {item.identity} | "
                f"{item.output_channels} canais | "
                f"{item.default_sample_rate} Hz"
            )
        output_value = self.console.read("Índice da saída: ").strip()
        output_device = select_device(
            devices, output_value, direction="output"
        )
        self.console.write(
            f"[voz] teste: fale por {test_seconds:.0f} segundos..."
        )
        captured = self.audio.capture(
            CaptureOptions(
                sample_rate=self.settings.voice_sample_rate,
                fixed_seconds=test_seconds,
                stop_on_key=False,
                allow_empty=False,
                input_device=input_device.index,
                input_device_name=input_device.identity,
                silence_threshold=self.settings.voice_silence_threshold,
                min_speech_ms=self.settings.voice_min_speech_ms,
            )
        )
        self.console.write("[voz] reproduzindo teste...")
        interrupted = self.audio.play(
            AudioResult(
                samples=captured.samples,
                sample_rate=captured.sample_rate,
                duration_seconds=captured.duration_seconds,
                provider="microphone-test",
            ),
            output_device=output_device.index,
            output_device_name=output_device.identity,
            interrupt_key=self.settings.voice_interrupt_key,
        )
        if interrupted:
            self.console.write("[voz] teste interrompido")
        update_env_values(
            self.settings.env_file,
            {
                "VOICE_INPUT_DEVICE_NAME": input_device.identity,
                "VOICE_OUTPUT_DEVICE_NAME": output_device.identity,
                "VOICE_INPUT_DEVICE": str(input_device.index),
                "VOICE_OUTPUT_DEVICE": str(output_device.index),
            },
        )
        self.console.write("[voz] configuração salva")
        return {
            "ok": True,
            "input": input_device.as_dict(),
            "output": output_device.as_dict(),
            "input_selection": "exact_name",
            "output_selection": "exact_name",
            "test_duration_seconds": captured.duration_seconds,
            "interrupted": interrupted,
            "env_file": str(self.settings.env_file),
        }


def voice_model_info(settings) -> dict[str, Any]:
    path = settings.voice_stt_model
    if path.is_dir():
        size = sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )
    elif path.is_file():
        size = path.stat().st_size
    else:
        size = 0
    return {
        "ok": True,
        "provider": settings.voice_stt_provider,
        "model": str(path),
        "exists": path.exists(),
        "size_bytes": size,
        "device": settings.voice_stt_device,
        "compute_type": settings.voice_stt_compute_type,
        "language": settings.voice_stt_language,
        "threads": settings.voice_stt_threads,
    }
