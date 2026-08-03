from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tern.orchestrator.cli import build_parser
from tern.orchestrator.config import load_settings
from tern.orchestrator.voice.audio import (
    CaptureOptions,
    SilenceDetector,
    SoundDeviceAudio,
)
from tern.orchestrator.voice.devices import select_device
from tern.orchestrator.voice.errors import (
    AudioEmpty,
    AudioInputNotFound,
    AudioOutputNotFound,
    STTTimeout,
    STTTranscriptionFailed,
    TTSSynthesisFailed,
    TTSTimeout,
    VoiceCancelled,
)
from tern.orchestrator.voice.logging import VoiceLogger
from tern.orchestrator.voice.normalize import normalize_for_speech
from tern.orchestrator.voice.models import (
    AudioData,
    AudioResult,
    DeviceInfo,
    SynthesisOptions,
    TranscriptionOptions,
    TranscriptionResult,
)
from tern.orchestrator.voice.policy import (
    ConfirmationDecision,
    VoiceActionApprover,
    confirm_transcription,
    may_be_sensitive,
    prepare_spoken_text,
)
from tern.orchestrator.voice.session import VoiceSession
from tern.orchestrator.voice.stt import FasterWhisperSTT
from tern.orchestrator.voice.tts import PiperTTS


class FakeConsole:
    def __init__(self, answers=()):
        self.answers = iter(answers)
        self.values: list[str] = []

    def write(self, value=""):
        self.values.append(value)

    def read(self, prompt=""):
        self.values.append(prompt)
        return next(self.answers)


class FakeStream:
    def __init__(self, callback=None, chunks=None, writes=None, **_kwargs):
        self.callback = callback
        self.chunks = chunks or []
        self.writes = writes

    def __enter__(self):
        if self.callback:
            for chunk in self.chunks:
                self.callback(chunk, len(chunk), None, None)
        return self

    def __exit__(self, *_args):
        return False

    def write(self, block):
        if self.writes is not None:
            self.writes.append(np.asarray(block))


class FakeSoundDevice:
    class Default:
        device = (0, 1)

    default = Default()

    def __init__(self, chunks=None):
        self.chunks = chunks or [np.ones((400, 1), dtype=np.float32) * 0.2]
        self.writes = []
        self.stopped = False

    def query_devices(self):
        return [
            {
                "name": "Mic Test",
                "index": 0,
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000,
            },
            {
                "name": "Speaker Test",
                "index": 1,
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 22050,
            },
        ]

    def query_hostapis(self):
        return [{"name": "Fake API"}]

    def check_input_settings(self, **_kwargs):
        return None

    def check_output_settings(self, **_kwargs):
        return None

    def InputStream(self, **kwargs):
        return FakeStream(chunks=self.chunks, **kwargs)

    def OutputStream(self, **kwargs):
        return FakeStream(writes=self.writes, **kwargs)

    def stop(self):
        self.stopped = True


def audio_data() -> AudioData:
    samples = np.ones(1600, dtype=np.float32) * 0.2
    return AudioData(samples, 16000, 0.1, 0.2, 0.2, speech_ms=100)


def test_voice_commands_are_registered():
    parser = build_parser()
    assert parser.parse_args(["voice"]).command == "voice"
    assert parser.parse_args(["voice", "--once"]).once
    assert parser.parse_args(["voice-devices"]).command == "voice-devices"
    assert parser.parse_args(["voice-diagnose"]).command == "voice-diagnose"


def test_voice_configuration_loads():
    settings = load_settings(
        {
            "VOICE_STT_THREADS": "2",
            "VOICE_SAMPLE_RATE": "16000",
            "VOICE_CONFIRM_TRANSCRIPTION": "false",
        }
    )
    assert settings.voice_stt_provider == "faster_whisper"
    assert settings.voice_tts_provider == "piper"
    assert settings.voice_stt_device == settings.voice_tts_device == "cpu"
    assert settings.voice_stt_threads == 2
    assert not settings.voice_confirm_transcription


def test_device_selection_by_index_and_name():
    devices = [
        DeviceInfo(2, "Microfone USB", 1, 0, 16000),
        DeviceInfo(3, "Fone USB", 0, 2, 48000),
    ]
    assert select_device(devices, "2", direction="input").index == 2
    assert select_device(devices, "fone", direction="output").index == 3


def test_missing_microphone_is_structured():
    devices = [DeviceInfo(1, "Speaker", 0, 2, 48000)]
    with pytest.raises(AudioInputNotFound):
        select_device(devices, None, direction="input")


def test_missing_output_is_structured():
    devices = [DeviceInfo(1, "Mic", 1, 0, 16000)]
    with pytest.raises(AudioOutputNotFound):
        select_device(devices, None, direction="output")


def test_capture_can_be_cancelled():
    cancel = threading.Event()
    cancel.set()
    backend = SoundDeviceAudio(
        FakeSoundDevice(), key_reader=lambda: None, sleeper=lambda _n: None
    )
    with pytest.raises(VoiceCancelled):
        backend.capture(
            CaptureOptions(min_speech_ms=10), cancel_event=cancel
        )


def test_capture_stops_at_maximum_duration():
    values = iter([0.0, 0.1, 2.0])
    backend = SoundDeviceAudio(
        FakeSoundDevice(),
        key_reader=lambda: None,
        clock=lambda: next(values),
        sleeper=lambda _n: None,
    )
    result = backend.capture(
        CaptureOptions(max_seconds=1, min_speech_ms=10)
    )
    assert result.stop_reason == "max_duration"


def test_silence_detection_after_speech():
    detector = SilenceDetector(
        sample_rate=1000,
        threshold=0.1,
        timeout_ms=500,
        min_speech_ms=300,
    )
    assert not detector.observe(np.ones(300), 300, 0.0)
    assert not detector.observe(np.zeros(100), 100, 0.2)
    assert detector.observe(np.zeros(100), 100, 0.6)


def test_empty_audio_is_rejected():
    backend = SoundDeviceAudio(
        FakeSoundDevice(chunks=[np.zeros((400, 1), dtype=np.float32)]),
        key_reader=lambda: "\r",
    )
    with pytest.raises(AudioEmpty):
        backend.capture(CaptureOptions(min_speech_ms=10))


class Segment:
    text = "Olá mundo"
    start = 0.0
    end = 1.0
    avg_logprob = -0.1


class Info:
    language = "pt"
    language_probability = 0.99


class WhisperModel:
    def transcribe(self, *_args, **_kwargs):
        return iter([Segment()]), Info()


def model_factory(*_args, **_kwargs):
    return WhisperModel()


def test_valid_transcription(tmp_path):
    provider = FasterWhisperSTT(tmp_path, model_factory=model_factory)
    result = provider.transcribe(
        audio_data(), TranscriptionOptions(language="pt")
    )
    provider.close()
    assert result.text == "Olá mundo"
    assert result.language == "pt"
    assert result.confidence is not None


def test_transcription_failure_is_structured(tmp_path):
    class Broken:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("broken")

    provider = FasterWhisperSTT(
        tmp_path, model_factory=lambda *_a, **_k: Broken()
    )
    with pytest.raises(STTTranscriptionFailed):
        provider.transcribe(audio_data(), TranscriptionOptions())
    provider.close()


def test_transcription_timeout(tmp_path):
    class Slow:
        def transcribe(self, *_args, **_kwargs):
            time.sleep(0.1)
            return iter([Segment()]), Info()

    provider = FasterWhisperSTT(
        tmp_path, model_factory=lambda *_a, **_k: Slow()
    )
    with pytest.raises(STTTimeout):
        provider.transcribe(
            audio_data(), TranscriptionOptions(timeout_seconds=0.01)
        )
    provider.close()


class Chunk:
    sample_rate = 22050
    audio_float_array = np.ones(2205, dtype=np.float32) * 0.1


class Voice:
    def synthesize(self, *_args, **_kwargs):
        yield Chunk()


class SynthesisConfig:
    def __init__(self, **values):
        self.values = values


def test_valid_synthesis(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    provider = PiperTTS(
        model,
        SoundDeviceAudio(FakeSoundDevice()),
        voice_factory=lambda *_a, **_k: Voice(),
        synthesis_config_factory=SynthesisConfig,
    )
    result = provider.synthesize("Olá", SynthesisOptions())
    provider.close()
    assert result.sample_rate == 22050
    assert result.duration_seconds == pytest.approx(0.1)


def test_synthesis_failure_is_structured(tmp_path):
    class Broken:
        def synthesize(self, *_args, **_kwargs):
            raise RuntimeError("broken")

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    provider = PiperTTS(
        model,
        SoundDeviceAudio(FakeSoundDevice()),
        voice_factory=lambda *_a, **_k: Broken(),
        synthesis_config_factory=SynthesisConfig,
    )
    with pytest.raises(TTSSynthesisFailed):
        provider.synthesize("Olá", SynthesisOptions())
    provider.close()


def test_synthesis_timeout(tmp_path):
    class Slow:
        def synthesize(self, *_args, **_kwargs):
            time.sleep(0.1)
            yield Chunk()

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    provider = PiperTTS(
        model,
        SoundDeviceAudio(FakeSoundDevice()),
        voice_factory=lambda *_a, **_k: Slow(),
        synthesis_config_factory=SynthesisConfig,
    )
    with pytest.raises(TTSTimeout):
        provider.synthesize(
            "Olá", SynthesisOptions(timeout_seconds=0.01)
        )
    provider.close()


def test_playback_interruption():
    backend = SoundDeviceAudio(
        FakeSoundDevice(), key_reader=lambda: "\x1b"
    )
    result = backend.play(
        AudioResult(np.ones(1000), 22050, 0.05, "test")
    )
    assert result


def test_temporary_audio_is_removed(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"wave")
    value = audio_data()
    value.temporary_path = path
    assert SoundDeviceAudio.remove_temporary(value)
    assert not path.exists()


def test_transcription_confirmation_send():
    console = FakeConsole(["s"])
    assert (
        confirm_transcription("Olá", required=True, console=console)
        == ConfirmationDecision.SEND
    )


def test_transcription_confirmation_rerecord():
    console = FakeConsole(["r"])
    assert (
        confirm_transcription("Olá", required=True, console=console)
        == ConfirmationDecision.RERECORD
    )


def test_transcription_confirmation_cancel():
    console = FakeConsole(["c"])
    assert (
        confirm_transcription("Olá", required=True, console=console)
        == ConfirmationDecision.CANCEL
    )


def test_safe_action_can_skip_general_confirmation():
    console = FakeConsole([])
    assert (
        confirm_transcription(
            "Que horas são?", required=False, console=console
        )
        == ConfirmationDecision.SEND
    )


def test_destructive_action_requires_separate_confirmation():
    assert may_be_sensitive("Apague a pasta de testes")
    console = FakeConsole(["CONFIRMAR"])
    approver = VoiceActionApprover(console)
    assert approver("delete", {"path": "D:\\tern\\tests"})
    output = "\n".join(console.values)
    assert "filesystem_delete" in output
    assert "reversão" in output


class FakeAudio:
    def __init__(self):
        self.captures = 0

    def capture(self, _options):
        self.captures += 1
        return audio_data()

    def remove_temporary(self, _audio):
        return False


class FakeSTT:
    def __init__(self, text="Pergunta simples"):
        self.text = text

    def transcribe(self, _audio, _options):
        return TranscriptionResult(
            self.text, "pt", 0.95, 0.01, "fake"
        )


class FakeTTS:
    def __init__(self):
        self.values = []

    def speak(self, text, _options, **_kwargs):
        self.values.append(text)
        return False


class FakeSupervisor:
    def __init__(self, answer="Resposta completa"):
        self.answer = answer
        self.requests = []

    def run(self, text, event_callback=None):
        self.requests.append(text)
        return {"ok": True, "answer": self.answer, "web": {"used": False}}


def voice_session(
    tmp_path,
    *,
    console,
    text="Pergunta simples",
    answer="Resposta completa",
    confirm=False,
    supervisor=None,
):
    settings = load_settings(
        {
            "VOICE_CONFIRM_TRANSCRIPTION": "true" if confirm else "false",
            "VOICE_MAX_SPOKEN_CHARACTERS": "200",
        }
    )
    audio = FakeAudio()
    tts = FakeTTS()
    supervisor = supervisor or FakeSupervisor(answer)
    session = VoiceSession(
        settings,
        supervisor,
        audio,
        FakeSTT(text),
        tts,
        VoiceLogger(tmp_path / "voice.jsonl"),
        console=console,
    )
    return session, audio, tts, supervisor


def test_voice_integrates_with_same_ask_flow(tmp_path):
    session, _audio, tts, supervisor = voice_session(
        tmp_path, console=FakeConsole([""])
    )
    result = session.run(once=True)
    assert result["ok"]
    assert supervisor.requests == ["Pergunta simples"]
    assert tts.values


def test_voice_web_state_is_visible(tmp_path):
    class WebSupervisor(FakeSupervisor):
        def run(self, text, event_callback=None):
            event_callback("tool_start", {"name": "web_search"})
            return super().run(text, event_callback)

    console = FakeConsole([""])
    session, *_ = voice_session(
        tmp_path, console=console, supervisor=WebSupervisor()
    )
    session.run(once=True)
    assert "[assistente] pesquisando..." in console.values


def test_voice_codex_state_is_visible(tmp_path):
    class CodexSupervisor(FakeSupervisor):
        def run(self, text, event_callback=None):
            event_callback("tool_start", {"name": "codex_delegate"})
            return super().run(text, event_callback)

    console = FakeConsole([""])
    session, *_ = voice_session(
        tmp_path, console=console, supervisor=CodexSupervisor()
    )
    session.run(once=True)
    assert any("codex_delegate" in value for value in console.values)


def test_voice_codex_history_state_is_visible(tmp_path):
    class HistorySupervisor(FakeSupervisor):
        def run(self, text, event_callback=None):
            event_callback("tool_start", {"name": "review_codex_session"})
            return super().run(text, event_callback)

    console = FakeConsole([""])
    session, *_ = voice_session(
        tmp_path, console=console, supervisor=HistorySupervisor()
    )
    session.run(once=True)
    assert any(
        "consultando a sessao compartilhada do Codex" in value
        for value in console.values
    )


def test_voice_background_job_messages_are_short_and_hide_ids(tmp_path):
    console = FakeConsole([])
    session, _audio, tts, _supervisor = voice_session(
        tmp_path,
        console=console,
    )
    session._interaction_active = True
    session._event(
        "codex_job_started",
        {
            "job_id": "secret-job-id",
            "thread_id": "secret-thread-id",
            "turn_id": "secret-turn-id",
        },
    )
    assert session._spoken_status == "Enviei a tarefa ao Codex."
    session._interaction_active = False
    session._event(
        "codex_job_completed",
        {
            "job_id": "secret-job-id",
            "thread_id": "secret-thread-id",
            "turn_id": "secret-turn-id",
        },
    )
    session._event(
        "codex_job_status",
        {"status": "running", "notify": True, "job_id": "secret-job-id"},
    )
    session._event("codex_job_interrupted", {"job_id": "secret-job-id"})
    session._event("codex_job_failed", {"job_id": "secret-job-id"})
    assert tts.values == [
        "A tarefa do Codex foi concluída.",
        "O Codex ainda está trabalhando.",
        "Tarefa cancelada.",
        "O Codex encontrou um problema durante a execução.",
    ]
    spoken = " ".join(tts.values)
    assert "secret" not in spoken and "job" not in spoken.casefold()


def test_long_response_is_summarized_for_speech():
    text = "Primeira frase. " + ("Detalhe repetido. " * 100) + "Conclusão."
    spoken = prepare_spoken_text(
        text,
        max_characters=200,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    assert len(spoken) <= 200
    assert "Detalhes completos" in spoken


def test_code_between_backticks_is_read():
    spoken = prepare_spoken_text(
        "Use:\n```python\nprint('segredo')\n```",
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    assert "print('segredo')" in spoken
    assert "código disponível" not in spoken.casefold()


def test_inline_code_is_read_through_full_speech_pipeline():
    prepared = prepare_spoken_text(
        (
            "Ferramentas `codex_delegate` e `codex_continue`. "
            "Campos `session_id`, `working_directory` e `task`."
        ),
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    spoken = normalize_for_speech(prepared, "piper", "default")

    assert "código disponível" not in spoken.casefold()
    for expected in (
        "codex delegate",
        "codex continue",
        "session id",
        "working directory",
        "task",
    ):
        assert expected in spoken


def test_url_is_not_read_in_full():
    spoken = prepare_spoken_text(
        "Fonte https://example.com/caminho/muito/longo?token=abc",
        max_characters=300,
        read_code=False,
        read_urls=False,
        summarize_long=True,
    )
    assert "https://" not in spoken
    assert "fonte disponível" in spoken.casefold()


def test_full_text_remains_visible_while_speech_is_short(tmp_path):
    answer = "Início. " + ("conteúdo longo " * 100) + "Fim."
    console = FakeConsole([""])
    session, _audio, tts, _supervisor = voice_session(
        tmp_path, console=console, answer=answer
    )
    session.run(once=True)
    assert answer in console.values
    assert len(tts.values[0]) < len(answer)


def test_unit_session_uses_no_real_audio_hardware(tmp_path):
    session, audio, _tts, _supervisor = voice_session(
        tmp_path, console=FakeConsole([""])
    )
    session.run(once=True)
    assert isinstance(audio, FakeAudio)
    assert audio.captures == 1


def test_rerecord_records_again(tmp_path):
    session, audio, _tts, _supervisor = voice_session(
        tmp_path,
        console=FakeConsole(["", "r", "", "s"]),
        confirm=True,
    )
    result = session.run(once=True)
    assert result["ok"]
    assert audio.captures == 2


def test_voice_cancel_does_not_call_assistant(tmp_path):
    session, _audio, _tts, supervisor = voice_session(
        tmp_path,
        console=FakeConsole(["", "c"]),
        confirm=True,
    )
    result = session.run(once=True)
    assert result["error"] == "voice_cancelled"
    assert not supervisor.requests
