from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import numpy as np

from .audio import SoundDeviceAudio
from .errors import (
    AudioPlaybackFailed,
    TTSChunkSynthesisFailed,
    TTSModelNotFound,
    TTSProviderNotConfigured,
    TTSStreamCancelled,
    TTSStreamPlaybackFailed,
    TTSStreamQueueFailed,
    TTSSynthesisFailed,
    TTSTimeout,
)
from .models import AudioResult, SynthesisOptions
from .postprocess import postprocess_audio
from .streaming import segment_for_speech


def rate_to_length_scale(rate: float) -> float:
    """Convert an intuitive speech rate to Piper's duration scale."""
    if rate <= 0:
        raise ValueError("a taxa de fala deve ser positiva")
    return 1.0 / rate


class TextToSpeechProvider(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> AudioResult:
        raise NotImplementedError

    @abstractmethod
    def speak(
        self,
        text: str,
        options: SynthesisOptions,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool:
        raise NotImplementedError


class PiperTTS(TextToSpeechProvider):
    name = "piper"

    def __init__(
        self,
        model_path: str | Path,
        audio: SoundDeviceAudio,
        *,
        config_path: str | Path | None = None,
        output_device: str | int | None = None,
        output_device_name: str | None = None,
        interrupt_key: str = "esc",
        sentence_pause_ms: int = 0,
        paragraph_pause_ms: int = 0,
        post_processing: bool = False,
        normalize_loudness: bool = True,
        light_compression: bool = False,
        light_eq: bool = False,
        voice_factory: Any | None = None,
        synthesis_config_factory: Any | None = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else Path(str(self.model_path) + ".json")
        )
        self.audio = audio
        self.output_device_selector = output_device
        self.output_device_name = output_device_name
        self.interrupt_key = interrupt_key
        self.sentence_pause_ms = sentence_pause_ms
        self.paragraph_pause_ms = paragraph_pause_ms
        self.post_processing = post_processing
        self.normalize_loudness = normalize_loudness
        self.light_compression = light_compression
        self.light_eq = light_eq
        self.voice_factory = voice_factory
        self.synthesis_config_factory = synthesis_config_factory
        self._voice = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tern-tts"
        )

    def _load(self):
        if self._voice is not None:
            return self._voice
        if not self.model_path.is_file():
            raise TTSModelNotFound(
                f"modelo TTS ausente: {self.model_path}",
                details={"model": str(self.model_path)},
            )
        factory = self.voice_factory
        if factory is None:
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise TTSProviderNotConfigured(
                    "piper-tts ausente; instale tern[voice]"
                ) from exc
            factory = PiperVoice.load
        try:
            self._voice = factory(
                str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=False,
            )
        except TypeError:
            try:
                self._voice = factory(
                    str(self.model_path),
                    config_path=str(self.config_path),
                )
            except TypeError:
                self._voice = factory(str(self.model_path))
        except Exception as exc:
            raise TTSSynthesisFailed(
                f"falha ao carregar voz Piper: {exc}"
            ) from exc
        return self._voice

    def synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> AudioResult:
        text = text.strip()
        if not text:
            raise TTSSynthesisFailed("texto vazio para sintese")
        started = time.monotonic()
        future = self._executor.submit(
            self._synthesize_sync, text, options, None
        )
        try:
            samples, sample_rate = future.result(
                timeout=options.timeout_seconds
            )
        except FutureTimeout as exc:
            future.cancel()
            raise TTSTimeout(
                f"sintese excedeu {options.timeout_seconds}s"
            ) from exc
        except (TTSModelNotFound, TTSSynthesisFailed):
            raise
        except Exception as exc:
            raise TTSSynthesisFailed(f"falha na sintese: {exc}") from exc
        if not samples.size:
            raise TTSSynthesisFailed("Piper retornou audio vazio")
        result = AudioResult(
            samples=samples,
            sample_rate=sample_rate,
            duration_seconds=samples.size / sample_rate,
            provider=self.name,
            metadata={"synthesis_seconds": time.monotonic() - started},
        )
        return self._finalize_audio(result, text)

    def _synthesize_sync(
        self,
        text: str,
        options: SynthesisOptions,
        stop_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, int]:
        voice = self._load()
        config_factory = self.synthesis_config_factory
        if config_factory is None:
            try:
                from piper import SynthesisConfig
            except ImportError as exc:
                raise TTSProviderNotConfigured(
                    "SynthesisConfig do Piper indisponivel"
                ) from exc
            config_factory = SynthesisConfig
        config = config_factory(
            volume=options.volume,
            length_scale=rate_to_length_scale(options.rate),
        )
        chunks = []
        sample_rate = None
        try:
            for chunk in voice.synthesize(text, syn_config=config):
                if stop_event is not None and stop_event.is_set():
                    raise TTSStreamCancelled("sintese progressiva cancelada")
                chunk_sample_rate = int(chunk.sample_rate)
                if sample_rate is None:
                    sample_rate = chunk_sample_rate
                elif chunk_sample_rate != sample_rate:
                    raise TTSSynthesisFailed(
                        "Piper retornou chunks com sample rates diferentes"
                    )
                if hasattr(chunk, "audio_float_array"):
                    values = np.asarray(
                        chunk.audio_float_array, dtype=np.float32
                    )
                elif hasattr(chunk, "audio_int16_array"):
                    values = (
                        np.asarray(chunk.audio_int16_array, dtype=np.float32)
                        / 32768.0
                    )
                else:
                    raw_pcm = chunk.audio_int16_bytes
                    if len(raw_pcm) % np.dtype(np.int16).itemsize:
                        raise TTSSynthesisFailed(
                            "Piper retornou PCM int16 desalinhado"
                        )
                    values = (
                        np.frombuffer(
                            raw_pcm, dtype="<i2"
                        ).astype(np.float32)
                        / 32768.0
                    )
                chunks.append(values.reshape(-1))
        except TTSStreamCancelled:
            raise
        except Exception as exc:
            raise TTSSynthesisFailed(f"Piper falhou: {exc}") from exc
        if not chunks or sample_rate is None:
            raise TTSSynthesisFailed("Piper retornou audio vazio")
        return np.concatenate(chunks), sample_rate

    def _synthesize_stream_chunk(
        self,
        text: str,
        options: SynthesisOptions,
        stop_event: threading.Event,
    ) -> AudioResult:
        started = time.monotonic()
        future = self._executor.submit(
            self._synthesize_sync, text, options, stop_event
        )
        deadline = started + options.timeout_seconds
        while True:
            if stop_event.is_set():
                future.cancel()
                raise TTSStreamCancelled(
                    "sintese pendente cancelada"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise TTSTimeout(
                    f"sintese excedeu {options.timeout_seconds}s"
                )
            try:
                samples, sample_rate = future.result(
                    timeout=min(0.1, remaining)
                )
                break
            except FutureTimeout:
                continue
        result = AudioResult(
            samples=samples,
            sample_rate=sample_rate,
            duration_seconds=samples.size / sample_rate,
            provider=self.name,
            metadata={
                "synthesis_seconds": time.monotonic() - started
            },
        )
        return self._finalize_audio(result, text)

    def _finalize_audio(
        self, result: AudioResult, text: str
    ) -> AudioResult:
        if self.post_processing:
            result = postprocess_audio(
                result,
                normalize_loudness=self.normalize_loudness,
                light_compression=self.light_compression,
                light_eq=self.light_eq,
            )
        pause_ms = self.sentence_pause_ms
        if "\n\n" in text:
            pause_ms = max(pause_ms, self.paragraph_pause_ms)
        if text.rstrip().endswith("?") or any(
            word in text.casefold()
            for word in ("atenção", "alerta", "segurança", "confirmação")
        ):
            pause_ms = max(pause_ms, round(self.sentence_pause_ms * 1.3))
        if pause_ms > 0:
            silence = np.zeros(
                round(result.sample_rate * pause_ms / 1000),
                dtype=np.float32,
            )
            result.samples = np.concatenate(
                [
                    np.asarray(result.samples, dtype=np.float32).reshape(-1),
                    silence,
                ]
            )
            result.duration_seconds = result.samples.size / result.sample_rate
            result.metadata["pause_ms"] = pause_ms
        return result

    def speak(
        self,
        text: str,
        options: SynthesisOptions,
        *,
        stop_event: threading.Event | None = None,
    ) -> bool:
        result = self.synthesize(text, options)
        return self.audio.play(
            result,
            output_device=self.output_device_selector,
            output_device_name=self.output_device_name,
            stop_event=stop_event,
            interrupt_key=self.interrupt_key,
        )

    def speak_streaming(
        self,
        text: str,
        options: SynthesisOptions,
        *,
        chunk_min_characters: int,
        chunk_max_characters: int,
        queue_size: int,
        stop_event: threading.Event | None = None,
        event_callback: Any | None = None,
    ) -> dict[str, Any]:
        stop_event = stop_event or threading.Event()
        segments = segment_for_speech(
            text,
            minimum=chunk_min_characters,
            maximum=chunk_max_characters,
        )
        if not segments:
            raise TTSChunkSynthesisFailed(
                "nenhum segmento disponivel para sintese"
            )
        audio_queue: Queue[Any] = Queue(maxsize=queue_size)
        sentinel = object()
        producer_errors: list[Exception] = []
        synthesis_times: list[float] = []
        peak_queue_size = 0
        started = time.monotonic()

        def emit(event: str, **values: Any) -> None:
            if event_callback is not None:
                event_callback(event, values)

        def put_cancelable(value: Any) -> None:
            while not stop_event.is_set():
                try:
                    audio_queue.put(value, timeout=0.1)
                    return
                except Full:
                    continue
            raise TTSStreamCancelled("fila cancelada")

        def producer() -> None:
            try:
                for index, segment in enumerate(segments):
                    if stop_event.is_set():
                        raise TTSStreamCancelled(
                            "sintese progressiva cancelada"
                        )
                    chunk_started = time.monotonic()
                    audio = self._synthesize_stream_chunk(
                        segment, options, stop_event
                    )
                    duration = time.monotonic() - chunk_started
                    synthesis_times.append(duration)
                    put_cancelable((index, audio))
                    emit(
                        "chunk_synthesized",
                        index=index,
                        duration_seconds=duration,
                        queue_size=audio_queue.qsize(),
                    )
            except Exception as exc:
                producer_errors.append(exc)
            finally:
                try:
                    put_cancelable(sentinel)
                except TTSStreamCancelled:
                    pass

        thread = threading.Thread(
            target=producer,
            name="tern-tts-producer",
            daemon=True,
        )
        thread.start()
        expected_index = 0
        time_to_first_audio: float | None = None
        interrupted = False
        try:
            while True:
                if stop_event.is_set():
                    interrupted = True
                    break
                try:
                    item = audio_queue.get(timeout=0.1)
                except Empty:
                    if not thread.is_alive() and audio_queue.empty():
                        break
                    continue
                peak_queue_size = max(
                    peak_queue_size, audio_queue.qsize()
                )
                if item is sentinel:
                    break
                index, audio = item
                if index != expected_index:
                    raise TTSStreamQueueFailed(
                        "segmentos fora de ordem",
                        details={
                            "expected": expected_index,
                            "received": index,
                        },
                    )
                if time_to_first_audio is None:
                    time_to_first_audio = time.monotonic() - started
                    emit(
                        "first_audio",
                        time_to_first_audio=time_to_first_audio,
                    )
                try:
                    chunk_interrupted = self.audio.play(
                        audio,
                        output_device=self.output_device_selector,
                        output_device_name=self.output_device_name,
                        stop_event=stop_event,
                        interrupt_key=self.interrupt_key,
                    )
                except AudioPlaybackFailed as exc:
                    raise TTSStreamPlaybackFailed(str(exc)) from exc
                expected_index += 1
                emit("chunk_played", index=index)
                if chunk_interrupted or stop_event.is_set():
                    interrupted = True
                    stop_event.set()
                    break
            if producer_errors and not interrupted:
                error = producer_errors[0]
                if isinstance(error, TTSStreamCancelled):
                    raise error
                if isinstance(error, TTSTimeout):
                    raise error
                raise TTSChunkSynthesisFailed(
                    f"falha em segmento: {error}"
                ) from error
        finally:
            if interrupted:
                stop_event.set()
            while True:
                try:
                    audio_queue.get_nowait()
                except Empty:
                    break
            thread.join(timeout=5)
            if thread.is_alive() and not interrupted:
                raise TTSStreamQueueFailed(
                    "produtor TTS nao encerrou"
                )
        total = time.monotonic() - started
        return {
            "interrupted": interrupted,
            "segments": len(segments),
            "segments_played": expected_index,
            "time_to_first_audio": time_to_first_audio,
            "total_seconds": total,
            "average_synthesis_seconds": (
                sum(synthesis_times) / len(synthesis_times)
                if synthesis_times
                else 0.0
            ),
            "peak_queue_size": peak_queue_size,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._voice = None
