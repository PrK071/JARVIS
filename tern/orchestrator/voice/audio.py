from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .devices import list_devices, resolve_device, select_device
from .errors import (
    AudioCaptureFailed,
    AudioEmpty,
    AudioPlaybackFailed,
    VoiceCancelled,
)
from .models import AudioData, AudioResult, DeviceInfo


@dataclass(frozen=True)
class CaptureOptions:
    sample_rate: int = 16000
    max_seconds: int = 60
    silence_timeout_ms: int = 1200
    min_speech_ms: int = 300
    silence_threshold: float = 0.015
    input_device: str | int | None = None
    input_device_name: str | None = None
    stop_on_key: bool = True
    allow_empty: bool = False
    fixed_seconds: float | None = None


KeyReader = Callable[[], str | None]


class SilenceDetector:
    def __init__(
        self,
        *,
        sample_rate: int,
        threshold: float,
        timeout_ms: int,
        min_speech_ms: int,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.timeout_ms = timeout_ms
        self.min_speech_ms = min_speech_ms
        self.speech_frames = 0
        self.last_speech_at: float | None = None

    @property
    def speech_ms(self) -> int:
        return round(self.speech_frames * 1000 / self.sample_rate)

    def observe(self, chunk: np.ndarray, frames: int, now: float) -> bool:
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        if rms >= self.threshold:
            self.speech_frames += int(frames)
            self.last_speech_at = now
            return False
        return (
            self.last_speech_at is not None
            and self.speech_ms >= self.min_speech_ms
            and (now - self.last_speech_at) * 1000 >= self.timeout_ms
        )


def _windows_key_reader() -> str | None:
    if os.name != "nt":
        return None
    import msvcrt

    if not msvcrt.kbhit():
        return None
    value = msvcrt.getwch()
    if value in {"\x00", "\xe0"} and msvcrt.kbhit():
        msvcrt.getwch()
        return None
    return value


class SoundDeviceAudio:
    def __init__(
        self,
        sounddevice_module: Any | None = None,
        *,
        key_reader: KeyReader | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if sounddevice_module is None:
            try:
                import sounddevice as sounddevice_module
            except ImportError as exc:
                raise AudioCaptureFailed(
                    "sounddevice ausente; instale o extra tern[voice]"
                ) from exc
        self.sd = sounddevice_module
        self.key_reader = key_reader or _windows_key_reader
        self.clock = clock
        self.sleeper = sleeper

    def devices(self) -> list[DeviceInfo]:
        return list_devices(self.sd)

    def defaults(self) -> tuple[int | None, int | None]:
        try:
            values = tuple(self.sd.default.device)
            return int(values[0]), int(values[1])
        except Exception:
            return None, None

    def resolve_input_device(
        self,
        selector: str | int | None,
        preferred_name: str | None = None,
    ) -> tuple[DeviceInfo, str]:
        default_input, _ = self.defaults()
        return resolve_device(
            self.devices(),
            selector,
            direction="input",
            default_index=default_input,
            preferred_name=preferred_name,
        )

    def input_device(
        self,
        selector: str | int | None,
        preferred_name: str | None = None,
    ) -> DeviceInfo:
        return self.resolve_input_device(selector, preferred_name)[0]

    def resolve_output_device(
        self,
        selector: str | int | None,
        preferred_name: str | None = None,
    ) -> tuple[DeviceInfo, str]:
        _, default_output = self.defaults()
        return resolve_device(
            self.devices(),
            selector,
            direction="output",
            default_index=default_output,
            preferred_name=preferred_name,
        )

    def output_device(
        self,
        selector: str | int | None,
        preferred_name: str | None = None,
    ) -> DeviceInfo:
        return self.resolve_output_device(selector, preferred_name)[0]

    def capture(
        self,
        options: CaptureOptions,
        *,
        cancel_event: threading.Event | None = None,
    ) -> AudioData:
        cancel_event = cancel_event or threading.Event()
        device = self.input_device(
            options.input_device, options.input_device_name
        )
        chunks: list[np.ndarray] = []
        callback_error: list[str] = []
        stop_event = threading.Event()
        detector = SilenceDetector(
            sample_rate=options.sample_rate,
            threshold=options.silence_threshold,
            timeout_ms=options.silence_timeout_ms,
            min_speech_ms=options.min_speech_ms,
        )
        started = self.clock()
        stop_reason = "manual"

        def callback(indata, frames, _time_info, status):
            nonlocal stop_reason
            if status:
                callback_error.append(str(status))
            chunk = np.asarray(indata, dtype=np.float32)
            if chunk.ndim == 2:
                chunk = chunk[:, 0]
            chunk = chunk.copy()
            chunks.append(chunk)
            now = self.clock()
            if (
                options.fixed_seconds is None
                and detector.observe(chunk, frames, now)
            ):
                stop_reason = "silence"
                stop_event.set()
            elif options.fixed_seconds is not None:
                detector.observe(chunk, frames, now)

        try:
            self.sd.check_input_settings(
                device=device.index,
                channels=1,
                samplerate=options.sample_rate,
                dtype="float32",
            )
            with self.sd.InputStream(
                samplerate=options.sample_rate,
                device=device.index,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=max(256, options.sample_rate // 20),
            ):
                while not stop_event.is_set():
                    elapsed = self.clock() - started
                    limit = (
                        options.fixed_seconds
                        if options.fixed_seconds is not None
                        else options.max_seconds
                    )
                    if elapsed >= limit:
                        stop_reason = (
                            "fixed_duration"
                            if options.fixed_seconds is not None
                            else "max_duration"
                        )
                        break
                    if cancel_event.is_set():
                        raise VoiceCancelled("captura cancelada")
                    if options.stop_on_key:
                        key = self.key_reader()
                        if key == "\x1b":
                            raise VoiceCancelled("captura cancelada por Esc")
                        if key in {"\r", "\n", " "}:
                            stop_reason = "key"
                            break
                    self.sleeper(0.02)
        except VoiceCancelled:
            raise
        except Exception as exc:
            raise AudioCaptureFailed(
                f"falha ao capturar microfone: {exc}",
                details={"device": device.as_dict()},
            ) from exc
        if callback_error:
            raise AudioCaptureFailed(
                "PortAudio reportou falha durante captura",
                details={"status": callback_error[-3:]},
            )
        samples = (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.empty(0, dtype=np.float32)
        )
        speech_ms = detector.speech_ms
        if not samples.size or (
            speech_ms < options.min_speech_ms and not options.allow_empty
        ):
            raise AudioEmpty(
                "nenhuma fala suficiente detectada",
                details={"speech_ms": speech_ms},
            )
        samples -= float(np.mean(samples)) if samples.size else 0.0
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples *= min(1.0, 0.95 / peak)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = (
            float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        )
        return AudioData(
            samples=samples,
            sample_rate=options.sample_rate,
            duration_seconds=samples.size / options.sample_rate,
            rms=rms,
            peak=peak,
            speech_ms=speech_ms,
            stop_reason=stop_reason,
        )

    def play(
        self,
        audio: AudioResult,
        *,
        output_device: str | int | None = None,
        output_device_name: str | None = None,
        stop_event: threading.Event | None = None,
        interrupt_key: str = "esc",
    ) -> bool:
        stop_event = stop_event or threading.Event()
        device = self.output_device(
            output_device, output_device_name
        )
        samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        if not samples.size:
            raise AudioPlaybackFailed("audio sintetizado vazio")
        interrupted = False
        try:
            self.sd.check_output_settings(
                device=device.index,
                channels=1,
                samplerate=audio.sample_rate,
                dtype="float32",
            )
            with self.sd.OutputStream(
                samplerate=audio.sample_rate,
                device=device.index,
                channels=1,
                dtype="float32",
            ) as stream:
                chunk_size = max(256, audio.sample_rate // 20)
                for offset in range(0, samples.size, chunk_size):
                    key = self.key_reader()
                    key_interrupt = key == "\x1b" or (
                        interrupt_key != "esc"
                        and key is not None
                        and key.casefold() == interrupt_key.casefold()
                    )
                    if stop_event.is_set() or key_interrupt:
                        interrupted = True
                        stop_event.set()
                        break
                    block = samples[offset : offset + chunk_size].reshape(-1, 1)
                    stream.write(block)
        except Exception as exc:
            raise AudioPlaybackFailed(
                f"falha ao reproduzir audio: {exc}",
                details={"device": device.as_dict()},
            ) from exc
        finally:
            try:
                self.sd.stop()
            except Exception:
                pass
        return interrupted

    @staticmethod
    def remove_temporary(audio: AudioData | AudioResult) -> bool:
        path: Path | None = audio.temporary_path
        if path is None or not path.exists():
            return False
        path.unlink()
        audio.temporary_path = None
        return True
