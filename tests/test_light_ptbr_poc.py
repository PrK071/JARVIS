from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tern.orchestrator.cli import build_parser
from tern.orchestrator.config import load_settings
from tern.orchestrator.voice import light_compare, windows_speech
from tern.orchestrator.voice.models import SynthesisOptions
from tern.orchestrator.voice.tts import WindowsSpeechTTS


def test_windows_voice_inventory_filters_pt_br(monkeypatch):
    monkeypatch.setattr(
        windows_speech,
        "_run_helper",
        lambda *_args, **_kwargs: {
            "winrt": [
                {"id": "daniel-id", "locale": "pt-BR"},
                {"id": "david-id", "locale": "en-US"},
            ],
            "sapi": [{"id": "maria-id", "locale": "PT-br"}],
            "system_speech": [],
        },
    )

    result = windows_speech.list_windows_voices()

    assert [voice["id"] for voice in result["pt_br"]] == [
        "daniel-id",
        "maria-id",
    ]
    assert result["winrt"][1]["pt_br"] is False


def test_windows_synthesis_selects_stable_id(monkeypatch, tmp_path):
    captured = {}

    def fake_helper(action, **kwargs):
        captured.update({"action": action, **kwargs})
        return {"metrics": []}

    monkeypatch.setattr(windows_speech, "_run_helper", fake_helper)

    windows_speech.synthesize_windows_voice(
        voice_id="stable-registry-id",
        interface="WinRT",
        output_directory=tmp_path,
        items=[{"filename": "sample.wav", "text": "trabalho"}],
    )

    assert captured["action"] == "synthesize"
    assert captured["voice_id"] == "stable-registry-id"
    assert captured["interface"] == "WinRT"
    assert captured["request"]["output_directory"] == str(
        tmp_path.resolve()
    )


def test_light_poc_text_never_uses_fake_trabalho_spelling():
    texts = " ".join(text for _filename, text in light_compare.LIGHT_PHRASES)
    assert "trabalho" in texts.casefold()
    for forbidden in (
        "trrabalho",
        "trabarro",
        "tabalho",
        "tarabalho",
    ):
        assert forbidden not in texts.casefold()


def test_supertonic_runtime_environment_forces_offline(tmp_path):
    environment = light_compare._supertonic_environment(tmp_path)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["NO_PROXY"] == "*"


def test_light_poc_command_options():
    args = build_parser().parse_args(
        [
            "voice-light-ptbr-poc",
            "--windows-only",
            "--skip-stt",
            "--no-play",
        ]
    )
    assert args.command == "voice-light-ptbr-poc"
    assert args.windows_only is True
    assert args.skip_stt is True
    assert args.no_play is True


def test_report_marks_generic_portuguese_and_human_decision(tmp_path):
    report = {
        "status": "phase-a-audio-ready",
        "windows": {},
        "supertonic": {"styles": {}},
        "rhvoice": {"reason": "licença ausente"},
        "mms": {"reason": "aguarda escolha"},
    }

    light_compare._write_reports(Path(tmp_path), report)
    markdown = (tmp_path / "comparison-report.md").read_text(
        encoding="utf-8"
    )

    assert "português genérico" in markdown
    assert "decisão é auditiva" in markdown


def test_windows_rate_configuration_and_bounds():
    settings = load_settings(
        {
            "VOICE_TTS_PROVIDER": "windows_sapi",
            "VOICE_MODE": "windows_sapi",
            "VOICE_WINDOWS_VOICE_ID": "stable-daniel-id",
            "VOICE_WINDOWS_RATE": "-1",
            "VOICE_WINDOWS_VOLUME": "100",
        }
    )
    assert settings.voice_windows_voice_id == "stable-daniel-id"
    assert settings.voice_windows_rate == -1
    assert settings.voice_windows_volume == 100
    fractional = load_settings(
        {
            "VOICE_TTS_PROVIDER": "windows_sapi",
            "VOICE_WINDOWS_RATE": "1.5",
        }
    )
    assert fractional.voice_windows_rate == 1.5
    with pytest.raises(ValueError, match="VOICE_WINDOWS_RATE"):
        load_settings(
            {
                "VOICE_TTS_PROVIDER": "windows_sapi",
                "VOICE_WINDOWS_RATE": "11",
            }
        )


def test_windows_provider_applies_rate_id_volume_and_cleans_temp(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_synthesize(**kwargs):
        captured.update(kwargs)
        target = kwargs["output_directory"] / kwargs["items"][0]["filename"]
        sf.write(target, np.ones(1600, dtype=np.float32) * 0.1, 16000)
        return {
            "metrics": [
                {
                    "synthesis_seconds": 0.01,
                    "applied_speaking_rate": 0.8959584598407622,
                }
            ]
        }

    monkeypatch.setattr(
        windows_speech,
        "synthesize_windows_voice",
        fake_synthesize,
    )
    provider = WindowsSpeechTTS(
        object(),
        voice_id="stable-daniel-id",
        rate=-1,
        volume=100,
        temp_directory=tmp_path,
    )

    result = provider.synthesize(
        "O trabalho foi concluído corretamente.",
        SynthesisOptions(),
    )

    assert result.provider == "windows_sapi"
    assert result.sample_rate == 16000
    assert result.metadata["configured_rate"] == -1
    assert captured["voice_id"] == "stable-daniel-id"
    assert captured["rate"] == -1
    assert captured["volume"] == 100
    assert list(tmp_path.glob("*.wav")) == []
