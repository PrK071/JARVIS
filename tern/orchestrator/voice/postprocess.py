from __future__ import annotations

import numpy as np

from .models import AudioResult


def postprocess_audio(
    audio: AudioResult,
    *,
    normalize_loudness: bool,
    light_compression: bool,
    light_eq: bool,
) -> AudioResult:
    values = np.asarray(audio.samples, dtype=np.float32).reshape(-1).copy()
    if not values.size:
        return audio
    values -= float(np.mean(values))
    if light_eq and values.size > 1:
        # Filtro DC/rumble leve; sem alteração de pitch.
        previous = np.concatenate(([0.0], values[:-1]))
        values = values - 0.97 * previous
    if light_compression:
        threshold = 0.55
        absolute = np.abs(values)
        over = absolute > threshold
        values[over] = np.sign(values[over]) * (
            threshold + (absolute[over] - threshold) * 0.35
        )
    if normalize_loudness:
        rms = float(np.sqrt(np.mean(np.square(values))))
        target = 10 ** (-20 / 20)
        if rms > 1e-6:
            values *= min(2.5, target / rms)
    peak = float(np.max(np.abs(values)))
    if peak > 0.96:
        values *= 0.96 / peak
    fade = min(values.size // 2, round(audio.sample_rate * 0.006))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        values[:fade] *= ramp
        values[-fade:] *= ramp[::-1]
    audio.samples = values
    audio.duration_seconds = values.size / audio.sample_rate
    audio.metadata["postprocessed"] = True
    return audio
