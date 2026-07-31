from __future__ import annotations

import threading
from typing import Any

from .audio import SoundDeviceAudio
from .errors import VoiceError
from .logging import VoiceLogger
from .models import SynthesisOptions
from .stt import SpeechToTextProvider
from .tts import TextToSpeechProvider


class VoiceDiagnostic:
    """Lightweight Piper/output diagnostic. It never calls Qwen or STT."""

    def __init__(
        self,
        settings,
        audio: SoundDeviceAudio,
        stt: SpeechToTextProvider,
        tts: TextToSpeechProvider,
        logger: VoiceLogger,
    ):
        self.settings = settings
        self.audio = audio
        self.stt = stt
        self.tts = tts
        self.logger = logger

    def run(self, *, capture_seconds: float = 4.0) -> dict[str, Any]:
        del capture_seconds
        result: dict[str, Any] = {
            "ok": True,
            "qwen_called": False,
            "stt_called": False,
            "configuration": {
                "provider": self.settings.voice_tts_provider,
                "model": str(self.settings.voice_tts_model),
                "model_available": self.settings.voice_tts_model.is_file(),
                "voice": self.settings.voice_tts_voice,
                "rate": self.settings.voice_tts_rate,
                "length_scale": 1.0 / self.settings.voice_tts_rate,
                "style": self.settings.voice_style,
                "output_device_name": (
                    self.settings.voice_output_device_name
                ),
                "sentence_pause_ms": (
                    self.settings.voice_sentence_pause_ms
                ),
                "paragraph_pause_ms": (
                    self.settings.voice_paragraph_pause_ms
                ),
            },
            "steps": {},
        }
        synthesized = None
        try:
            devices = self.audio.devices()
            input_device, input_selection = self.audio.resolve_input_device(
                self.settings.voice_input_device,
                self.settings.voice_input_device_name,
            )
            output, selection = self.audio.resolve_output_device(
                self.settings.voice_output_device,
                self.settings.voice_output_device_name,
            )
            result["devices"] = {
                "input": input_device.as_dict(),
                "output": output.as_dict(),
                "input_count": sum(
                    item.input_channels > 0 for item in devices
                ),
                "output_count": sum(
                    item.output_channels > 0 for item in devices
                ),
                "input_selection": input_selection,
                "output_selection": selection,
            }
            result["steps"]["output_device"] = {
                "ok": True,
                "selection": selection,
                "device": output.as_dict(),
            }
            phrase = (
                "Boa noite, senhor. Todos os sistemas estão operacionais."
            )
            synthesized = self.tts.synthesize(
                phrase,
                SynthesisOptions(
                    rate=self.settings.voice_tts_rate,
                    volume=self.settings.voice_tts_volume,
                    timeout_seconds=self.settings.voice_tts_timeout_seconds,
                ),
            )
            result["steps"]["synthesis"] = {
                "ok": True,
                "provider": synthesized.provider,
                "duration_seconds": synthesized.duration_seconds,
                "sample_rate": synthesized.sample_rate,
                **synthesized.metadata,
            }
            interrupted = self.audio.play(
                synthesized,
                output_device=self.settings.voice_output_device,
                output_device_name=self.settings.voice_output_device_name,
                interrupt_key=self.settings.voice_interrupt_key,
            )
            result["steps"]["playback"] = {
                "ok": True,
                "interrupted": interrupted,
            }
            stop_event = threading.Event()
            timer = threading.Timer(0.25, stop_event.set)
            timer.start()
            try:
                cancelled = self.audio.play(
                    synthesized,
                    output_device=self.settings.voice_output_device,
                    output_device_name=self.settings.voice_output_device_name,
                    stop_event=stop_event,
                    interrupt_key=self.settings.voice_interrupt_key,
                )
            finally:
                timer.cancel()
            result["steps"]["cancellation"] = {
                "ok": cancelled,
                "audio_did_not_resume": cancelled,
            }
            if not cancelled:
                result["ok"] = False
        except VoiceError as exc:
            result["ok"] = False
            result["error"] = exc.as_dict()
        finally:
            removed = False
            if synthesized is not None:
                removed = self.audio.remove_temporary(synthesized)
            result["steps"]["cleanup"] = {
                "ok": True,
                "temporary_removed": removed,
                "temporary_remaining": bool(
                    synthesized
                    and synthesized.temporary_path
                    and synthesized.temporary_path.exists()
                ),
            }
        self.logger.write("diagnostic_finished", ok=result["ok"])
        return result
