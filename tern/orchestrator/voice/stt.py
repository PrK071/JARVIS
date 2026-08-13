from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (
    AudioEmpty,
    STTModelNotFound,
    STTProviderNotConfigured,
    STTTimeout,
    STTTranscriptionFailed,
)
from .models import AudioData, TranscriptionOptions, TranscriptionResult


class SpeechToTextProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: AudioData,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        raise NotImplementedError


class FasterWhisperSTT(SpeechToTextProvider):
    name = "faster_whisper"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        threads: int = 4,
        model_factory: Any | None = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.compute_type = compute_type
        self.threads = threads
        self.model_factory = model_factory
        self._model = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tern-stt"
        )

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_dir():
            raise STTModelNotFound(
                f"modelo STT ausente: {self.model_path}",
                details={"model": str(self.model_path)},
            )
        factory = self.model_factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise STTProviderNotConfigured(
                    "faster-whisper ausente; instale tern[voice]"
                ) from exc
            factory = WhisperModel
        try:
            self._model = factory(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.threads,
                local_files_only=True,
            )
        except TypeError:
            self._model = factory(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.threads,
            )
        except Exception as exc:
            raise STTTranscriptionFailed(
                f"falha ao carregar modelo STT: {exc}"
            ) from exc
        return self._model

    def transcribe(
        self,
        audio: AudioData,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        if not samples.size or float(np.max(np.abs(samples))) < 1e-5:
            raise AudioEmpty("audio vazio para transcricao")
        started = time.monotonic()
        future = self._executor.submit(
            self._transcribe_sync, samples, options.language
        )
        try:
            text, language, confidence, segments = future.result(
                timeout=options.timeout_seconds
            )
        except FutureTimeout as exc:
            future.cancel()
            raise STTTimeout(
                f"transcricao excedeu {options.timeout_seconds}s"
            ) from exc
        except (AudioEmpty, STTTranscriptionFailed):
            raise
        except Exception as exc:
            raise STTTranscriptionFailed(
                f"falha na transcricao: {exc}"
            ) from exc
        duration = time.monotonic() - started
        if not text:
            raise AudioEmpty("STT nao reconheceu fala")
        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            duration_seconds=duration,
            provider=self.name,
            segments=segments,
        )

    def _transcribe_sync(
        self, samples: np.ndarray, language: str
    ) -> tuple[str, str, float | None, tuple[dict[str, Any], ...]]:
        model = self._load()
        try:
            segment_iter, info = model.transcribe(
                samples,
                language=language or None,
                beam_size=3,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segment_values = list(segment_iter)
        except Exception as exc:
            raise STTTranscriptionFailed(
                f"faster-whisper falhou: {exc}"
            ) from exc
        values = []
        log_probabilities = []
        for segment in segment_values:
            text = str(getattr(segment, "text", "")).strip()
            if text:
                values.append(text)
            average = getattr(segment, "avg_logprob", None)
            if isinstance(average, (int, float)):
                log_probabilities.append(float(average))
        confidence = None
        if log_probabilities:
            confidence = max(
                0.0,
                min(
                    1.0,
                    sum(math.exp(value) for value in log_probabilities)
                    / len(log_probabilities),
                ),
            )
        elif isinstance(getattr(info, "language_probability", None), float):
            confidence = float(info.language_probability)
        segments = tuple(
            {
                "start": float(getattr(segment, "start", 0.0)),
                "end": float(getattr(segment, "end", 0.0)),
                "text": str(getattr(segment, "text", "")).strip(),
            }
            for segment in segment_values
        )
        detected = str(getattr(info, "language", language) or language)
        return " ".join(values).strip(), detected, confidence, segments

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
