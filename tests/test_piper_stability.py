from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from tern.orchestrator.cli import build_parser
from tern.orchestrator.config import Settings, load_settings
from tern.orchestrator.voice.models import AudioResult, SynthesisOptions
from tern.orchestrator.voice.normalize import normalize_for_speech
from tern.orchestrator.voice.pronunciation import (
    PHRASES,
    generate_pronunciation_test,
)
from tern.orchestrator.voice.tts import PiperTTS


class FakeChunk:
    sample_rate = 1000
    audio_float_array = np.ones(100, dtype=np.float32) * 0.1


class FakeVoice:
    def synthesize(self, _text, syn_config=None):
        yield FakeChunk()


class FakeConfig:
    def __init__(self, **values):
        self.values = values


class FakeAudio:
    def play(self, *_args, **_kwargs):
        return False


class FakePiper:
    def synthesize(self, _text, _options):
        return AudioResult(
            np.ones(100, dtype=np.float32) * 0.1,
            1000,
            0.1,
            "piper",
        )


def test_piper_is_the_only_configured_provider():
    settings = load_settings({})
    assert settings.voice_tts_provider == "piper"
    assert settings.voice_mode == "piper"
    with pytest.raises(ValueError):
        load_settings({"VOICE_TTS_PROVIDER": "remote"})


def test_removed_backend_has_no_typed_configuration():
    removed_prefix = "x" + "tts"
    assert not any(
        field.name.casefold().startswith(removed_prefix)
        for field in fields(Settings)
    )


def test_cli_initializes_without_removed_dependency():
    source = (
        Path(__file__).parents[1]
        / "tern"
        / "orchestrator"
        / "cli.py"
    ).read_text(encoding="utf-8")
    removed_package = "coqui" + "-tts"
    assert removed_package not in source.casefold()
    assert build_parser().parse_args(["voice", "--no-voice"]).voice_mode == "silent"


def test_piper_normalization_hides_markdown_urls_code_and_paths():
    original = (
        "**Atenção:** veja https://example.com/a. "
        "Não remova o arquivo. "
        "Arquivo D:\\tern\\src\\orchestrator\\voice.py. "
        "```python\nprint('não falar')\n```"
    )
    spoken = normalize_for_speech(
        original,
        "piper",
        "default",
        lexicon_path=(
            Path(__file__).parents[1]
            / "tern"
            / "orchestrator"
            / "voice"
            / "pronunciation_ptbr.json"
        ),
    )
    assert original.startswith("**Atenção")
    assert "https://" not in spoken
    assert "print" not in spoken
    assert "módulo de voz" in spoken
    assert "não" in spoken.casefold()


def test_piper_lexicon_is_provider_specific():
    lexicon = (
        Path(__file__).parents[1]
        / "tern"
        / "orchestrator"
        / "voice"
        / "pronunciation_ptbr.json"
    )
    spoken = normalize_for_speech(
        "Qwen3.5, GGUF, JSON, API, Codex e PowerShell.",
        "piper",
        "default",
        lexicon_path=lexicon,
    )
    assert "Qwen três ponto cinco" in spoken
    assert "gê gê u éfe" in spoken
    assert "jêison" in spoken
    assert "Códex" in spoken
    assert "Páuer Chél" in spoken


def test_piper_applies_configured_natural_pauses(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    provider = PiperTTS(
        model,
        FakeAudio(),
        voice_factory=lambda *_args, **_kwargs: FakeVoice(),
        synthesis_config_factory=FakeConfig,
        sentence_pause_ms=140,
        paragraph_pause_ms=260,
    )
    sentence = provider.synthesize("Tudo certo.", SynthesisOptions(rate=0.96))
    question = provider.synthesize("Tudo certo?", SynthesisOptions(rate=0.96))
    paragraph = provider.synthesize(
        "Primeiro.\n\nSegundo.", SynthesisOptions(rate=0.96)
    )
    assert sentence.metadata["pause_ms"] == 140
    assert question.metadata["pause_ms"] == 182
    assert paragraph.metadata["pause_ms"] == 260
    provider.close()


def test_pronunciation_command_generates_twelve_local_files(tmp_path):
    settings = load_settings(
        {
            "VOICE_TTS_MODEL": str(tmp_path / "voice.onnx"),
            "VOICE_TTS_SPEED": "0.96",
        }
    )
    result = generate_pronunciation_test(
        settings,
        FakePiper(),
        FakeAudio(),
        output=tmp_path / "pronunciation",
    )
    assert result["provider"] == "piper"
    assert result["generated"] == len(PHRASES) == 12
    assert all(Path(item["file"]).is_file() for item in result["files"])
