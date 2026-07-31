from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tern.orchestrator.cli import build_parser
from tern.orchestrator.voice import model_compare
from tern.orchestrator.voice.model_compare import (
    MODEL_COMPARISON_PHRASES,
    _join_with_pause,
    compare_piper_models,
)
from tern.orchestrator.voice.models import AudioResult
from tern.orchestrator.voice.tts import PiperTTS
from tern.orchestrator.voice.voices import (
    piper_voice_aliases,
    resolve_piper_voice,
    validate_piper_voice_pair,
)


def write_voice_pair(model: Path, sample_rate: int = 22050) -> tuple[Path, Path]:
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"onnx")
    config = Path(str(model) + ".json")
    config.write_text(
        json.dumps(
            {
                "audio": {"sample_rate": sample_rate},
                "espeak": {"voice": "pt-br"},
                "language": {"code": "pt-br"},
                "num_speakers": 1,
                "inference": {
                    "noise_scale": 0.667,
                    "length_scale": 1.0,
                    "noise_w": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    return model, config


@pytest.mark.parametrize("alias", ["faber", "miro", "jeff", "cadu", "dii"])
def test_resolves_each_portable_alias(tmp_path, alias):
    model, _config = piper_voice_aliases(tmp_path)[alias]
    write_voice_pair(model)
    selected = resolve_piper_voice(
        {"VOICE_PIPER_VOICE": alias}, tmp_path
    )
    assert selected.alias == alias
    assert selected.model_path == model.resolve()
    assert selected.available


def test_explicit_model_and_config_take_priority(tmp_path):
    model, config = write_voice_pair(tmp_path / "custom.onnx")
    selected = resolve_piper_voice(
        {
            "VOICE_PIPER_VOICE": "faber",
            "VOICE_PIPER_MODEL_PATH": str(model),
            "VOICE_PIPER_CONFIG_PATH": str(config),
        },
        tmp_path,
    )
    assert selected.alias == "custom"
    assert selected.model_path == model.resolve()
    assert selected.config_path == config.resolve()


def test_missing_requested_and_default_voice_falls_back_to_faber(tmp_path):
    faber, _config = piper_voice_aliases(tmp_path)["faber"]
    write_voice_pair(faber)
    selected = resolve_piper_voice(
        {"VOICE_PIPER_VOICE": "dii"}, tmp_path
    )
    assert selected.alias == "faber"
    assert selected.requested_alias == "dii"
    assert selected.fallback is True


def test_model_and_json_must_be_corresponding(tmp_path):
    model, _config = write_voice_pair(tmp_path / "one.onnx")
    other = tmp_path / "other.onnx.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="par correspondente"):
        validate_piper_voice_pair(model, other)


def test_sample_rate_is_read_per_voice(tmp_path):
    model, config = write_voice_pair(tmp_path / "voice.onnx", 24000)
    metadata = validate_piper_voice_pair(model, config)
    assert metadata["sample_rate"] == 24000
    assert metadata["language"] == "pt-br"
    assert metadata["num_speakers"] == 1


def test_invalid_language_is_rejected(tmp_path):
    model, config = write_voice_pair(tmp_path / "voice.onnx")
    value = json.loads(config.read_text(encoding="utf-8"))
    value["language"]["code"] = "en-us"
    value["espeak"]["voice"] = "en-us"
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="não é pt-BR"):
        validate_piper_voice_pair(model, config)


def test_piper_loads_only_selected_voice_and_releases_it(tmp_path):
    model, config = write_voice_pair(tmp_path / "voice.onnx")
    loaded = []

    class Voice:
        pass

    def factory(path, **kwargs):
        loaded.append((path, kwargs))
        return Voice()

    provider = PiperTTS(
        model,
        object(),
        config_path=config,
        voice_factory=factory,
    )
    assert provider._load() is provider._load()
    assert len(loaded) == 1
    assert loaded[0][1]["config_path"] == str(config.resolve())
    provider.close()
    assert provider._voice is None


def test_alias_resolution_has_no_download_or_network_code():
    source = (
        Path(__file__).parents[1]
        / "tern"
        / "orchestrator"
        / "voice"
        / "voices.py"
    ).read_text(encoding="utf-8").casefold()
    assert "huggingface" not in source
    assert "urlopen" not in source
    assert "requests" not in source


def test_model_weights_and_new_model_json_are_ignored():
    ignore = (Path(__file__).parents[1] / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert "*.onnx" in ignore
    assert "models/piper/**/*.onnx.json" in ignore


def test_no_checkpoint_is_part_of_aliases_or_downloaded():
    root = Path(__file__).parents[1]
    assert all(
        model.suffix == ".onnx" and ".ckpt" not in str(model)
        for model, _config in piper_voice_aliases(root).values()
    )
    assert not list((root / "models" / "piper").rglob("*.ckpt"))


def test_comparison_command_is_registered():
    args = build_parser().parse_args(
        ["voice-compare-models", "--no-play", "--select", "miro"]
    )
    assert args.command == "voice-compare-models"
    assert args.select == "miro"
    assert args.no_play is True


def test_normalized_comparison_text_is_identical_for_every_voice():
    texts = [text for _filename, text in MODEL_COMPARISON_PHRASES]
    assert len(texts) == 15
    assert len(set(texts)) == 15
    assert all("trrabalho" not in text.casefold() for text in texts)
    assert all("trabarro" not in text.casefold() for text in texts)


def test_continuous_audio_uses_700ms_pause():
    values = [
        np.ones(1000, dtype=np.float32),
        np.ones(1000, dtype=np.float32),
    ]
    result = _join_with_pause(values, 1000, 700)
    assert result.size == 2700
    assert np.all(result[1000:1700] == 0)


def test_comparison_generates_wavs_report_and_persists_alias(
    tmp_path, monkeypatch
):
    aliases = piper_voice_aliases(tmp_path)
    write_voice_pair(aliases["faber"][0], sample_rate=8000)
    monkeypatch.setattr(
        model_compare,
        "piper_voice_aliases",
        lambda _root: aliases,
    )

    class FakeProvider:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def _load(self):
            return object()

        def synthesize(self, _text, options):
            samples = np.ones(
                8000 if options.rate == 1.0 else 8511,
                dtype=np.float32,
            ) * 0.1
            return AudioResult(
                samples=samples,
                sample_rate=8000,
                duration_seconds=samples.size / 8000,
                provider="piper",
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(model_compare, "PiperTTS", FakeProvider)
    transcripts = iter(
        text for _filename, text in MODEL_COMPARISON_PHRASES
    )
    monkeypatch.setattr(
        model_compare,
        "_transcribe",
        lambda *_args, **_kwargs: (next(transcripts), None),
    )

    class FakeAudio:
        def play(self, _audio, **kwargs):
            return bool(kwargs.get("stop_event").is_set())

    env_file = tmp_path / ".env"
    env_file.write_text("KEEP=value\n", encoding="utf-8")
    settings = SimpleNamespace(
        state_dir=tmp_path / ".orchestrator",
        voice_style="clear_adult",
        voice_tts_timeout_seconds=10,
        voice_stt_timeout_seconds=10,
        voice_output_device=None,
        voice_output_device_name=None,
        voice_interrupt_key="esc",
        env_file=env_file,
    )
    result = compare_piper_models(
        settings,
        FakeAudio(),
        object(),
        play=False,
        selection="faber",
        prompt_for_selection=False,
    )
    output = Path(result["output"])
    assert len(list((output / "faber").glob("*.wav"))) == 15
    assert (output / "faber-completo.wav").is_file()
    assert (output / "faber-rate-094.wav").is_file()
    assert Path(result["report_markdown"]).is_file()
    assert "VOICE_PIPER_VOICE=faber" in env_file.read_text(
        encoding="utf-8"
    )
    assert "KEEP=value" in env_file.read_text(encoding="utf-8")
