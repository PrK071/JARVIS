from __future__ import annotations

from typing import Any


class VoiceError(RuntimeError):
    code = "voice_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "error": self.code,
            "message": str(self),
        }
        if self.details:
            result["details"] = self.details
        return result


class VoiceDisabled(VoiceError):
    code = "voice_disabled"


class AudioInputNotFound(VoiceError):
    code = "audio_input_not_found"


class AudioOutputNotFound(VoiceError):
    code = "audio_output_not_found"


class AudioCaptureFailed(VoiceError):
    code = "audio_capture_failed"


class AudioPlaybackFailed(VoiceError):
    code = "audio_playback_failed"


class AudioEmpty(VoiceError):
    code = "audio_empty"


class STTProviderNotConfigured(VoiceError):
    code = "stt_provider_not_configured"


class STTModelNotFound(VoiceError):
    code = "stt_model_not_found"


class STTTranscriptionFailed(VoiceError):
    code = "stt_transcription_failed"


class STTTimeout(VoiceError):
    code = "stt_timeout"


class TTSProviderNotConfigured(VoiceError):
    code = "tts_provider_not_configured"


class TTSModelNotFound(VoiceError):
    code = "tts_model_not_found"


class TTSSynthesisFailed(VoiceError):
    code = "tts_synthesis_failed"


class TTSTimeout(VoiceError):
    code = "tts_timeout"


class TTSChunkSynthesisFailed(VoiceError):
    code = "tts_chunk_synthesis_failed"


class TTSStreamCancelled(VoiceError):
    code = "tts_stream_cancelled"


class TTSStreamQueueFailed(VoiceError):
    code = "tts_stream_queue_failed"


class TTSStreamPlaybackFailed(VoiceError):
    code = "tts_stream_playback_failed"


class VoiceCancelled(VoiceError):
    code = "voice_cancelled"


class VoiceConfirmationRequired(VoiceError):
    code = "voice_confirmation_required"
