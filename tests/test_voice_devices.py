from __future__ import annotations

import threading

import numpy as np
import pytest

from tern.orchestrator.config import load_settings
from tern.orchestrator.voice.configure import (
    VoiceConfigurator,
    update_env_values,
    voice_model_info,
)
from tern.orchestrator.voice.devices import resolve_device, select_device
from tern.orchestrator.voice.diagnostic import VoiceDiagnostic
from tern.orchestrator.voice.errors import AudioInputNotFound
from tern.orchestrator.voice.logging import VoiceLogger
from tern.orchestrator.voice.models import (
    AudioData,
    AudioResult,
    DeviceInfo,
    TranscriptionResult,
)


DEVICES = [
    DeviceInfo(
        5,
        "Microfone Áudio",
        2,
        0,
        44100,
        "Windows DirectSound",
    ),
    DeviceInfo(3, "Fones", 0, 2, 44100, "MME"),
]


def test_device_selection_by_exact_persistent_name():
    device, source = resolve_device(
        DEVICES,
        None,
        direction="input",
        preferred_name="Microfone Áudio [Windows DirectSound]",
    )
    assert device.index == 5 and source == "exact_name"


def test_device_selection_by_normalized_name():
    device, source = resolve_device(
        DEVICES,
        None,
        direction="input",
        preferred_name="microfone audio [windows directsound]",
    )
    assert device.index == 5 and source == "normalized_name"


def test_missing_name_falls_back_to_explicit_id():
    device, source = resolve_device(
        DEVICES,
        "5",
        direction="input",
        preferred_name="Microfone desconectado",
    )
    assert device.index == 5 and source == "explicit_id"


def test_missing_name_and_id_fall_back_to_default():
    device, source = resolve_device(
        DEVICES,
        None,
        direction="input",
        default_index=5,
        preferred_name="Microfone desconectado",
    )
    assert device.index == 5 and source == "default"


def test_missing_device_without_fallback_is_error():
    with pytest.raises(AudioInputNotFound):
        resolve_device(
            DEVICES,
            None,
            direction="input",
            preferred_name="Inexistente",
        )


def test_duplicate_device_names_require_explicit_choice():
    values = [
        DeviceInfo(1, "Microfone", 1, 0, 44100, "MME"),
        DeviceInfo(2, "Microfone", 1, 0, 44100, "WASAPI"),
    ]
    with pytest.raises(AudioInputNotFound) as caught:
        select_device(
            values,
            None,
            direction="input",
            preferred_name="Microfone",
        )
    assert len(caught.value.details["matches"]) == 2


def test_device_selection_is_persisted(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("VOICE_ENABLED=true\n", encoding="utf-8")
    update_env_values(
        env_file,
        {
            "VOICE_INPUT_DEVICE_NAME": DEVICES[0].identity,
            "VOICE_INPUT_DEVICE": "5",
        },
    )
    text = env_file.read_text(encoding="utf-8")
    assert f"VOICE_INPUT_DEVICE_NAME={DEVICES[0].identity}" in text
    assert "VOICE_INPUT_DEVICE=5" in text


def test_env_update_preserves_unrelated_values_and_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# segredo preservado\nWEB_ENABLED=true\nVOICE_INPUT_DEVICE=1\n",
        encoding="utf-8",
    )
    update_env_values(env_file, {"VOICE_INPUT_DEVICE": "5"})
    text = env_file.read_text(encoding="utf-8")
    assert "# segredo preservado" in text
    assert "WEB_ENABLED=true" in text
    assert "VOICE_INPUT_DEVICE=5" in text


def test_same_name_survives_changed_device_id():
    changed = [
        DeviceInfo(
            18,
            "Microfone Áudio",
            2,
            0,
            44100,
            "Windows DirectSound",
        )
    ]
    device, source = resolve_device(
        changed,
        "5",
        direction="input",
        preferred_name="Microfone Áudio [Windows DirectSound]",
    )
    assert device.index == 18 and source == "exact_name"


class FakeConsole:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.values = []

    def write(self, value=""):
        self.values.append(value)

    def read(self, prompt=""):
        self.values.append(prompt)
        return next(self.answers)


class ConfigureAudio:
    def devices(self):
        return DEVICES

    def capture(self, _options):
        return AudioData(
            np.ones(1600, dtype=np.float32),
            16000,
            0.1,
            0.2,
            0.5,
            speech_ms=100,
        )

    def play(self, _audio, **_kwargs):
        return False


def test_interactive_configuration_preserves_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("WEB_ENABLED=true\nSECRET=value\n", encoding="utf-8")
    settings = load_settings({"TERN_ENV_FILE": str(env_file)})
    result = VoiceConfigurator(
        settings,
        ConfigureAudio(),
        console=FakeConsole(["5", "3"]),
    ).run(test_seconds=0.1)
    text = env_file.read_text(encoding="utf-8")
    assert result["ok"]
    assert "SECRET=value" in text
    assert DEVICES[0].identity in text
    assert DEVICES[1].identity in text


class DiagnosticAudio:
    def devices(self):
        return DEVICES

    def input_device(self, *_args):
        return DEVICES[0]

    def output_device(self, *_args):
        return DEVICES[1]

    def resolve_input_device(self, *_args):
        return DEVICES[0], "exact_name"

    def resolve_output_device(self, *_args):
        return DEVICES[1], "exact_name"

    def capture(self, _options):
        return AudioData(
            np.ones(1600, dtype=np.float32),
            16000,
            0.1,
            0.2,
            0.5,
            speech_ms=100,
        )

    def play(self, _audio, *, stop_event=None, **_kwargs):
        if stop_event is None:
            return False
        stop_event.wait(0.4)
        return stop_event.is_set()

    def remove_temporary(self, _audio):
        return False


class DiagnosticSTT:
    def transcribe(self, *_args):
        return TranscriptionResult(
            "teste",
            "pt",
            0.9,
            0.01,
            "fake",
        )


class DiagnosticTTS:
    def synthesize(self, *_args):
        return AudioResult(
            np.ones(1600, dtype=np.float32),
            16000,
            0.1,
            "fake",
            metadata={"synthesis_seconds": 0.01},
        )


def test_diagnostic_reports_resolved_device_selection(tmp_path):
    settings = load_settings(
        {
            "MODEL_STATE_DIR": str(tmp_path),
            "VOICE_INPUT_DEVICE_NAME": DEVICES[0].identity,
            "VOICE_OUTPUT_DEVICE_NAME": DEVICES[1].identity,
        }
    )
    result = VoiceDiagnostic(
        settings,
        DiagnosticAudio(),
        DiagnosticSTT(),
        DiagnosticTTS(),
        VoiceLogger(tmp_path / "voice.jsonl"),
    ).run(capture_seconds=0.1)
    assert result["devices"]["input_selection"] == "exact_name"
    assert result["devices"]["output_selection"] == "exact_name"


def test_optional_small_model_is_reported_without_download(tmp_path):
    model = tmp_path / "faster-whisper-small"
    model.mkdir()
    (model / "model.bin").write_bytes(b"x" * 10)
    settings = load_settings({"VOICE_STT_MODEL": str(model)})
    result = voice_model_info(settings)
    assert result["exists"]
    assert result["model"].endswith("faster-whisper-small")
    assert result["size_bytes"] == 10
