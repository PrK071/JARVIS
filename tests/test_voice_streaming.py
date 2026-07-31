from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tern.orchestrator.voice.errors import TTSChunkSynthesisFailed
from tern.orchestrator.voice.models import SynthesisOptions
from tern.orchestrator.voice.policy import prepare_research_spoken_text
from tern.orchestrator.voice.streaming import segment_for_speech
from tern.orchestrator.voice.tts import PiperTTS


def test_simple_sentence_segmentation():
    assert segment_for_speech(
        "Primeira frase. Segunda frase!", minimum=1, maximum=100
    ) == ["Primeira frase.", "Segunda frase!"]


def test_abbreviation_does_not_split_sentence():
    values = segment_for_speech(
        "Dr. Silva chegou. Tudo certo?", minimum=1, maximum=100
    )
    assert values[0] == "Dr. Silva chegou."


def test_qwen35_does_not_split_at_decimal_point():
    values = segment_for_speech(
        "Qwen3.5 funciona localmente. Próxima frase.",
        minimum=1,
        maximum=100,
    )
    assert values[0] == "Qwen3.5 funciona localmente."


def test_decimal_number_does_not_split():
    values = segment_for_speech(
        "A versão 1.5 foi publicada. Ela é estável.",
        minimum=1,
        maximum=100,
    )
    assert values[0] == "A versão 1.5 foi publicada."


def test_url_is_not_spoken():
    values = segment_for_speech(
        "Leia https://example.com/a.b?x=1. Depois continue.",
        minimum=1,
        maximum=100,
    )
    assert "https://" not in " ".join(values)
    assert "fonte disponível" in " ".join(values)


def test_code_is_replaced_before_segmentation():
    values = segment_for_speech(
        "Código: ```python\nprint('x')\n``` Resultado pronto.",
        minimum=1,
        maximum=100,
    )
    assert "print" not in " ".join(values)
    assert "Código disponível" in " ".join(values)


def test_inline_tool_identifiers_are_spoken_before_segmentation():
    values = segment_for_speech(
        (
            "Use `codex_delegate`, `codex_continue`, `session_id`, "
            "`working_directory` e `task`."
        ),
        minimum=1,
        maximum=200,
    )
    spoken = " ".join(values)

    assert "código disponível" not in spoken.casefold()
    assert "codex delegate" in spoken
    assert "codex continue" in spoken
    assert "session id" in spoken
    assert "working directory" in spoken
    assert "task" in spoken


def test_lists_keep_spoken_order():
    values = segment_for_speech(
        "- Primeiro item.\n- Segundo item.\n- Terceiro item.",
        minimum=1,
        maximum=100,
    )
    assert values == ["Primeiro item.", "Segundo item.", "Terceiro item."]


def test_short_sentences_are_combined_to_minimum():
    values = segment_for_speech(
        "Um. Dois. Esta frase é maior.",
        minimum=12,
        maximum=100,
    )
    assert values[0].startswith("Um. Dois.")


def test_long_sentence_respects_maximum():
    values = segment_for_speech(
        " ".join(["palavra"] * 100),
        minimum=10,
        maximum=80,
    )
    assert all(len(item) <= 80 for item in values)


class FakeSynthesisConfig:
    def __init__(self, **_kwargs):
        pass


class FakeChunk:
    sample_rate = 16000

    def __init__(self, value: float = 0.1):
        self.audio_float_array = np.ones(800, dtype=np.float32) * value


class FakeVoice:
    def __init__(self, *, delay=0.01, fail_on=None):
        self.delay = delay
        self.fail_on = fail_on
        self.completed: list[str] = []

    def synthesize(self, text, syn_config=None):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("chunk failure")
        time.sleep(self.delay)
        self.completed.append(text)
        yield FakeChunk(len(self.completed) / 10)


class FakeAudio:
    def __init__(self, *, delay=0.02, interrupt_at=None):
        self.delay = delay
        self.interrupt_at = interrupt_at
        self.played: list[float] = []
        self.voice: FakeVoice | None = None
        self.completed_when_first_played = None

    def play(self, audio, *, stop_event=None, **_kwargs):
        if self.completed_when_first_played is None and self.voice:
            self.completed_when_first_played = len(self.voice.completed)
        self.played.append(float(audio.samples[0]))
        time.sleep(self.delay)
        if self.interrupt_at == len(self.played):
            if stop_event is not None:
                stop_event.set()
            return True
        return bool(stop_event and stop_event.is_set())


def provider(tmp_path, *, voice=None, audio=None):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    voice = voice or FakeVoice()
    audio = audio or FakeAudio()
    audio.voice = voice
    value = PiperTTS(
        model,
        audio,
        voice_factory=lambda *_args, **_kwargs: voice,
        synthesis_config_factory=FakeSynthesisConfig,
    )
    return value, voice, audio


def stream(provider_value, text, *, queue_size=2, stop_event=None):
    return provider_value.speak_streaming(
        text,
        SynthesisOptions(timeout_seconds=5),
        chunk_min_characters=1,
        chunk_max_characters=45,
        queue_size=queue_size,
        stop_event=stop_event,
    )


def test_streaming_preserves_segment_order(tmp_path):
    value, _voice, audio = provider(tmp_path)
    result = stream(value, "Primeira frase longa. Segunda frase longa.")
    assert audio.played == pytest.approx([0.1, 0.2])
    assert result["segments_played"] == 2
    value.close()


def test_streaming_queue_is_limited(tmp_path):
    value, _voice, _audio = provider(
        tmp_path, audio=FakeAudio(delay=0.05)
    )
    result = stream(
        value,
        "Primeira frase. Segunda frase. Terceira frase. Quarta frase.",
        queue_size=2,
    )
    assert result["peak_queue_size"] <= 2
    value.close()


def test_first_playback_starts_before_total_synthesis_finishes(tmp_path):
    voice = FakeVoice(delay=0.03)
    value, _voice, audio = provider(
        tmp_path, voice=voice, audio=FakeAudio(delay=0.12)
    )
    result = stream(
        value,
        "Primeira sentença. Segunda sentença. Terceira sentença.",
    )
    assert audio.completed_when_first_played < result["segments"]
    assert result["time_to_first_audio"] < result["total_seconds"]
    value.close()


def test_interrupt_during_first_segment(tmp_path):
    value, _voice, audio = provider(
        tmp_path, audio=FakeAudio(interrupt_at=1)
    )
    result = stream(value, "Primeira frase. Segunda frase.")
    assert result["interrupted"]
    assert len(audio.played) == 1
    value.close()


def test_interrupt_between_segments(tmp_path):
    value, _voice, audio = provider(
        tmp_path, audio=FakeAudio(interrupt_at=2)
    )
    result = stream(
        value, "Primeira frase. Segunda frase. Terceira frase."
    )
    assert result["interrupted"]
    assert len(audio.played) == 2
    value.close()


def test_pending_synthesis_is_cancelled(tmp_path):
    voice = FakeVoice(delay=0.05)
    value, _voice, audio = provider(
        tmp_path, voice=voice, audio=FakeAudio(interrupt_at=1)
    )
    result = stream(
        value,
        "Primeira frase. Segunda frase. Terceira frase. Quarta frase.",
    )
    assert result["interrupted"]
    assert len(voice.completed) < result["segments"]
    value.close()


def test_no_temporary_wav_files_are_created(tmp_path):
    value, _voice, _audio = provider(tmp_path)
    stream(value, "Primeira frase. Segunda frase.")
    assert list(tmp_path.glob("*.wav")) == []
    value.close()


def test_piper_worker_closes_after_stream(tmp_path):
    value, _voice, _audio = provider(tmp_path)
    stream(value, "Uma frase.")
    value.close()
    assert value._executor._shutdown


def test_chunk_error_is_structured(tmp_path):
    value, _voice, _audio = provider(
        tmp_path, voice=FakeVoice(fail_on="Falha")
    )
    with pytest.raises(TTSChunkSynthesisFailed):
        stream(value, "Primeiro segmento. Falha no segundo.")
    value.close()


def test_non_progressive_fallback_still_speaks(tmp_path):
    value, _voice, audio = provider(tmp_path)
    interrupted = value.speak(
        "Resposta completa.",
        SynthesisOptions(timeout_seconds=5),
        stop_event=threading.Event(),
    )
    assert not interrupted and len(audio.played) == 1
    value.close()


def test_stream_can_be_used_again_after_interruption(tmp_path):
    audio = FakeAudio(interrupt_at=1)
    value, _voice, audio = provider(tmp_path, audio=audio)
    assert stream(value, "Primeira. Segunda.")["interrupted"]
    audio.interrupt_at = None
    result = stream(value, "Nova resposta funcional.")
    assert not result["interrupted"]
    value.close()


def test_research_speech_is_summary_with_source_count():
    spoken = prepare_research_spoken_text(
        "Resumo da notícia. Mais detalhes.\n\n"
        "Fontes consultadas:\n- [Fonte](https://example.com)",
        source_count=3,
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    assert "Encontrei 3 fontes relevantes" in spoken


def test_research_speech_does_not_pronounce_urls():
    spoken = prepare_research_spoken_text(
        "Veja https://example.com/noticia.",
        source_count=1,
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    assert "https://" not in spoken


def test_research_sources_remain_in_full_text():
    full = "Resposta.\nFontes consultadas:\nhttps://example.com"
    spoken = prepare_research_spoken_text(
        full,
        source_count=1,
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    assert "https://example.com" in full
    assert "https://example.com" not in spoken


def test_playback_never_occurs_out_of_order(tmp_path):
    value, _voice, audio = provider(
        tmp_path, voice=FakeVoice(delay=0.005)
    )
    stream(value, "Um segmento. Dois segmentos. Três segmentos.")
    assert audio.played == sorted(audio.played)
    value.close()
