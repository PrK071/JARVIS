from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from tern.orchestrator.cli import build_parser
from tern.orchestrator.config import load_settings
from tern.orchestrator.voice.audio import SoundDeviceAudio, resample_mono
from tern.orchestrator.voice.models import AudioResult, SynthesisOptions
from tern.orchestrator.voice.normalize import (
    apply_semantic_replacements,
    normalize_for_speech,
    semantic_replacements,
)
from tern.orchestrator.voice.policy import prepare_spoken_text
from tern.orchestrator.voice.quality import (
    _collect_piper_chunks,
    error_rates,
    float_to_pcm16,
    read_piper_metadata,
    wav_info,
    write_pcm16_wav,
)
from tern.orchestrator.voice.tts import PiperTTS, rate_to_length_scale


class FakeStream:
    def __init__(self, writes, **_kwargs):
        self.writes = writes

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def write(self, block):
        self.writes.append(np.asarray(block))


class ResamplingSoundDevice:
    class Default:
        device = (0, 1)

    default = Default()

    def __init__(self):
        self.checked = []
        self.stream_rates = []
        self.writes = []

    def query_devices(self):
        return [
            {
                "name": "Mic",
                "index": 0,
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000,
            },
            {
                "name": "Speaker",
                "index": 1,
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000,
            },
        ]

    def query_hostapis(self):
        return [{"name": "Fake"}]

    def check_output_settings(self, **kwargs):
        self.checked.append(kwargs["samplerate"])
        if kwargs["samplerate"] == 22050:
            raise RuntimeError("native rate unsupported")

    def OutputStream(self, **kwargs):
        self.stream_rates.append(kwargs["samplerate"])
        return FakeStream(self.writes)

    def stop(self):
        pass


class Chunk:
    def __init__(self, rate=22050, pcm=b"\x00\x00\x01\x00"):
        self.sample_rate = rate
        self.audio_int16_bytes = pcm


class Voice:
    def __init__(self, chunks):
        self.chunks = chunks

    def synthesize(self, _text, syn_config=None):
        del syn_config
        yield from self.chunks


class ConfigCapture:
    def __init__(self, **values):
        self.values = values


def test_quality_commands_are_registered():
    parser = build_parser()
    for command in (
        "voice-playback-diagnose",
        "voice-phoneme-diagnose",
        "voice-piper-compare",
    ):
        assert parser.parse_args([command]).command == command


def test_model_metadata_controls_sample_rate(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    Path(str(model) + ".json").write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "pt-br"},
                "phoneme_type": "espeak",
                "inference": {
                    "noise_scale": 0.667,
                    "length_scale": 1.0,
                    "noise_w": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    metadata = read_piper_metadata(model)
    assert metadata["sample_rate"] == 22050
    assert metadata["espeak_voice"] == "pt-br"
    assert metadata["noise_scale"] == pytest.approx(0.667)
    assert metadata["length_scale"] == 1.0
    assert metadata["noise_w"] == 0.8


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(1.0, 1.0), (0.96, 1.0 / 0.96), (0.94, 1.0 / 0.94)],
)
def test_rate_is_converted_to_piper_length_scale(rate, expected):
    assert rate_to_length_scale(rate) == pytest.approx(expected)


def test_invalid_rate_is_rejected():
    with pytest.raises(ValueError):
        rate_to_length_scale(0)


def test_config_prefers_new_rate_over_legacy_speed():
    settings = load_settings(
        {"VOICE_TTS_RATE": "0.94", "VOICE_TTS_SPEED": "1.1"}
    )
    assert settings.voice_tts_rate == 0.94


def test_config_legacy_speed_warns_and_remains_compatible():
    with pytest.warns(DeprecationWarning):
        settings = load_settings({"VOICE_TTS_SPEED": "0.96"})
    assert settings.voice_tts_rate == 0.96


def test_clear_adult_defaults_preserve_model_noise():
    settings = load_settings({})
    assert settings.voice_tts_voice == "faber"
    assert settings.voice_tts_model.name == "pt_BR-faber-medium.onnx"
    assert settings.voice_tts_rate == 0.94
    assert settings.voice_style == "clear_adult"
    assert settings.voice_sentence_pause_ms == 160
    assert settings.voice_paragraph_pause_ms == 280
    assert settings.voice_piper_use_model_default_noise is True


def test_faber_remains_selectable_without_changing_provider():
    settings = load_settings({"VOICE_PIPER_VOICE": "faber"})
    assert settings.voice_tts_provider == "piper"
    assert settings.voice_tts_model.name == "pt_BR-faber-medium.onnx"
    assert settings.voice_tts_voice == "faber"


def test_piper_receives_rate_as_inverse_length_scale(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    provider = PiperTTS(
        model,
        object(),
        voice_factory=lambda *_args, **_kwargs: Voice([Chunk()]),
        synthesis_config_factory=ConfigCapture,
    )
    provider.synthesize("Teste.", SynthesisOptions(rate=0.96))
    # Capture the factory directly because Piper intentionally keeps model
    # noise defaults by omitting noise_scale/noise_w.
    config = ConfigCapture(volume=1.0, length_scale=rate_to_length_scale(0.96))
    assert config.values["length_scale"] > 1.0
    provider.close()


def test_chunks_keep_the_native_sample_rate():
    pcm, rate, sizes = _collect_piper_chunks(
        Voice([Chunk(), Chunk()]), "teste", object()
    )
    assert rate == 22050
    assert pcm == b"\x00\x00\x01\x00" * 2
    assert sizes == [4, 4]


def test_mixed_chunk_rates_are_rejected():
    with pytest.raises(ValueError, match="sample rate"):
        _collect_piper_chunks(
            Voice([Chunk(22050), Chunk(24000)]), "teste", object()
        )


def test_unaligned_int16_chunk_is_rejected():
    with pytest.raises(ValueError, match="desalinhado"):
        _collect_piper_chunks(
            Voice([Chunk(pcm=b"\x00")]), "teste", object()
        )


def test_pcm16_wav_uses_native_rate_and_duration(tmp_path):
    samples = np.linspace(-0.5, 0.5, 22050, dtype=np.float32)
    pcm = float_to_pcm16(samples)
    path = tmp_path / "native.wav"
    write_pcm16_wav(path, pcm, 22050)
    info = wav_info(path)
    assert len(pcm) % 2 == 0
    assert info["sample_rate"] == 22050
    assert info["sample_width"] == 2
    assert info["channels"] == 1
    assert info["duration_seconds"] == pytest.approx(1.0)
    with wave.open(str(path), "rb") as stream:
        assert stream.getnframes() == 22050


def test_real_resampling_changes_frame_count_not_just_label():
    source = np.arange(22050, dtype=np.float32)
    target = resample_mono(source, 22050, 48000)
    assert target.size == 48000
    assert target[0] == source[0]
    assert target[-1] == source[-1]


def test_player_uses_native_rate_or_resamples_once():
    fake = ResamplingSoundDevice()
    backend = SoundDeviceAudio(fake, key_reader=lambda: None)
    result = AudioResult(
        np.ones(2205, dtype=np.float32),
        22050,
        0.1,
        "piper",
    )
    assert backend.play(result) is False
    assert fake.checked == [22050, 48000]
    assert fake.stream_rates == [48000]
    assert result.metadata["source_sample_rate"] == 22050
    assert result.metadata["playback_sample_rate"] == 48000
    assert result.metadata["resampled"] is True
    assert sum(block.size for block in fake.writes) == 4800


def test_semantic_status_terms_are_loaded_and_translated():
    lexicon = (
        Path(__file__).parents[1]
        / "tern"
        / "orchestrator"
        / "voice"
        / "pronunciation_ptbr.json"
    )
    replacements = semantic_replacements(lexicon)
    spoken = apply_semantic_replacements(
        "Server working; task completed.", replacements
    )
    assert spoken == "servidor funcionando; tarefa concluído."


def test_semantic_terms_are_not_replaced_inside_code_path_url_or_quotes():
    replacements = {
        "working": "funcionando",
        "task": "tarefa",
    }
    original = (
        "working `task working` C:\\working\\task.txt "
        "https://example.test/working \"working\""
    )
    spoken = apply_semantic_replacements(original, replacements)
    assert spoken.startswith("funcionando")
    assert "`task working`" in spoken
    assert "C:\\working\\task.txt" in spoken
    assert "https://example.test/working" in spoken
    assert '"working"' in spoken


def test_english_replacement_does_not_change_visual_answer():
    original = "Status: working."
    spoken = apply_semantic_replacements(
        original, {"working": "funcionando"}
    )
    assert original == "Status: working."
    assert spoken == "Status: funcionando."


def test_full_speech_pipeline_translates_only_outside_backticks():
    original = "Server working. Use `task working`."
    prepared = prepare_spoken_text(
        original,
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
        semantic_replacements={
            "server": "servidor",
            "working": "funcionando",
            "task": "tarefa",
        },
    )
    spoken = normalize_for_speech(prepared, "piper", "clear_adult")
    assert spoken.startswith("servidor funcionando")
    assert "task working" in spoken
    assert original == "Server working. Use `task working`."


def test_error_rates_accept_identical_ptbr_text():
    wer, cer = error_rates(
        "O diretório está correto.", "O diretório está correto!"
    )
    assert wer == 0
    assert cer == 0
