"""Real voice acceptance through a Windows virtual audio cable.

Uses real Piper, sounddevice, faster-whisper, Qwen and tools. No mocks.
The prompt is synthesized into a virtual output captured by the voice CLI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from tern.orchestrator.config import load_settings
from tern.orchestrator.voice.audio import SoundDeviceAudio
from tern.orchestrator.voice.models import SynthesisOptions
from tern.orchestrator.voice.tts import PiperTTS


ROOT = Path(__file__).resolve().parents[1]


class OutputMonitor:
    def __init__(self, stream):
        self.stream = stream
        self.text = ""
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> None:
        while True:
            value = self.stream.read(1)
            if not value:
                break
            print(value, end="", flush=True)
            with self.condition:
                self.text += value
                self.condition.notify_all()

    def wait_for(self, marker: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while marker not in self.text:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"saída não mostrou {marker!r}")
                self.condition.wait(min(remaining, 0.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="aceitação de voz real usando dispositivo loopback"
    )
    parser.add_argument("prompt")
    parser.add_argument("--loopback-input", default="10")
    parser.add_argument("--loopback-output", default="11")
    parser.add_argument("--loopback-channels", type=int, default=2)
    parser.add_argument("--loopback-sample-rate", type=int, default=44100)
    parser.add_argument("--assistant-output", default="3")
    parser.add_argument("--confirmation", default="S")
    parser.add_argument("--sensitive-reply", default="CANCELAR")
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    settings = load_settings()
    environment = os.environ.copy()
    environment.update(
        {
            "MODEL_STATE_DIR": tempfile.gettempdir(),
            "VOICE_INPUT_DEVICE": args.loopback_input,
            "VOICE_OUTPUT_DEVICE": args.assistant_output,
            "VOICE_INPUT_DEVICE_NAME": "",
            "VOICE_OUTPUT_DEVICE_NAME": "",
            "VOICE_MAX_RECORDING_SECONDS": "30",
            "VOICE_SILENCE_THRESHOLD": "0.003",
            "VOICE_MAX_SPOKEN_CHARACTERS": "300",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "tern.orchestrator", "voice", "--once"],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    monitor = OutputMonitor(process.stdout)
    monitor.start()
    try:
        monitor.wait_for("Enter para falar", 30)
        process.stdin.write("\n")
        process.stdin.flush()
        monitor.wait_for("[voz] ouvindo", 30)
        time.sleep(0.5)
        injector = PiperTTS(
            settings.voice_tts_model,
            SoundDeviceAudio(),
            output_device=args.loopback_output,
        )
        injected_audio = injector.synthesize(
            args.prompt,
            SynthesisOptions(
                rate=settings.voice_tts_rate,
                volume=settings.voice_tts_volume,
                timeout_seconds=settings.voice_tts_timeout_seconds,
            ),
        )
        output_frames = round(
            injected_audio.samples.size
            * args.loopback_sample_rate
            / injected_audio.sample_rate
        )
        resampled = np.interp(
            np.arange(output_frames)
            * injected_audio.sample_rate
            / args.loopback_sample_rate,
            np.arange(injected_audio.samples.size),
            injected_audio.samples,
        ).astype(np.float32)
        loopback_samples = np.repeat(
            resampled.reshape(-1, 1),
            args.loopback_channels,
            axis=1,
        )
        injector.audio.sd.play(
            loopback_samples,
            args.loopback_sample_rate,
            device=int(args.loopback_output),
            blocking=True,
        )
        monitor.wait_for("Confirma?", 90)
        process.stdin.write(args.confirmation + "\n")
        process.stdin.flush()
        sensitive_answered = False
        deadline = time.monotonic() + args.timeout
        while process.poll() is None:
            if (
                not sensitive_answered
                and "Digite CONFIRMAR" in monitor.text
            ):
                process.stdin.write(args.sensitive_reply + "\n")
                process.stdin.flush()
                sensitive_answered = True
            if time.monotonic() >= deadline:
                raise TimeoutError("aceitação excedeu limite")
            time.sleep(0.1)
        monitor.thread.join(timeout=5)
        return int(process.returncode or 0)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
