from __future__ import annotations

import os

import pytest

from tern.orchestrator.config import load_settings
from tern.orchestrator.voice.audio import CaptureOptions, SoundDeviceAudio


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VOICE_INTEGRATION_TESTS", "").lower() != "true",
    reason="defina RUN_VOICE_INTEGRATION_TESTS=true para usar hardware real",
)


def test_real_microphone_capture():
    settings = load_settings()
    audio = SoundDeviceAudio()
    result = audio.capture(
        CaptureOptions(
            sample_rate=settings.voice_sample_rate,
            fixed_seconds=3,
            stop_on_key=False,
            allow_empty=True,
            input_device=settings.voice_input_device,
        )
    )
    assert result.duration_seconds >= 2.5
    assert result.peak >= 0
