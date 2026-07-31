from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    input_channels: int
    output_channels: int
    default_sample_rate: int
    host_api: str | None = None

    @property
    def identity(self) -> str:
        return (
            f"{self.name} [{self.host_api}]"
            if self.host_api
            else self.name
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "default_sample_rate": self.default_sample_rate,
            "host_api": self.host_api,
            "identity": self.identity,
        }


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    rms: float
    peak: float
    speech_ms: int = 0
    stop_reason: str = "manual"
    temporary_path: Path | None = None


@dataclass(frozen=True)
class TranscriptionOptions:
    language: str = "pt"
    timeout_seconds: int = 120


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    confidence: float | None
    duration_seconds: float
    provider: str
    segments: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SynthesisOptions:
    rate: float = 1.0
    volume: float = 1.0
    timeout_seconds: int = 60


@dataclass
class AudioResult:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    provider: str
    temporary_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
