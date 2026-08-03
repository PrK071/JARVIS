from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..projects import normalize_technical_transcript
from .audio import CaptureOptions, SoundDeviceAudio
from .errors import VoiceCancelled, VoiceError
from .logging import VoiceLogger
from .models import SynthesisOptions, TranscriptionOptions
from .normalize import normalize_for_speech, semantic_replacements
from .policy import (
    ConfirmationDecision,
    ConsoleIO,
    confirm_transcription,
    may_be_sensitive,
    prepare_research_spoken_text,
    prepare_spoken_text,
)
from .stt import SpeechToTextProvider
from .tts import TextToSpeechProvider


class VoiceSession:
    def __init__(
        self,
        settings,
        supervisor,
        audio: SoundDeviceAudio,
        stt: SpeechToTextProvider,
        tts: TextToSpeechProvider,
        logger: VoiceLogger,
        *,
        console: ConsoleIO | None = None,
    ):
        self.settings = settings
        self.supervisor = supervisor
        self.audio = audio
        self.stt = stt
        self.tts = tts
        self.logger = logger
        self.console = console or ConsoleIO()
        self._spoken_status: str | None = None
        self._interaction_active = False
        self._notification_lock = threading.Lock()

    def run(self, *, once: bool = False) -> dict[str, Any]:
        interactions = 0
        last_result: dict[str, Any] = {
            "ok": True,
            "interactions": 0,
        }
        while True:
            self.console.write("[voz] pronto")
            value = self.console.read(
                "Enter para falar; Q para sair: "
            ).strip().casefold()
            if value in {"q", "sair", "exit"}:
                return last_result
            try:
                result = self._interaction()
                if result.get("rerecord"):
                    continue
                interactions += 1
                last_result = {**result, "interactions": interactions}
            except VoiceCancelled as exc:
                self.logger.write("cancelled", code=exc.code)
                self.console.write("[voz] interrompido")
                last_result = {
                    **exc.as_dict(),
                    "interactions": interactions,
                }
            except VoiceError as exc:
                self.logger.write(
                    "error", code=exc.code, message=str(exc)
                )
                self.console.write(f"[erro] {exc.code}: {exc}")
                last_result = {
                    **exc.as_dict(),
                    "interactions": interactions,
                }
            except KeyboardInterrupt:
                self.console.write("\n[voz] interrompido")
                return {
                    "ok": False,
                    "error": "voice_cancelled",
                    "interactions": interactions,
                }
            if once:
                return last_result

    def _interaction(self) -> dict[str, Any]:
        self._interaction_active = True
        self._spoken_status = None
        self.console.write("[voz] ouvindo... Enter/Espaço encerra; Esc cancela")
        self.logger.write(
            "recording_started",
            device=self.settings.voice_input_device,
            sample_rate=self.settings.voice_sample_rate,
        )
        audio_data = self.audio.capture(
            CaptureOptions(
                sample_rate=self.settings.voice_sample_rate,
                max_seconds=self.settings.voice_max_recording_seconds,
                silence_timeout_ms=self.settings.voice_silence_timeout_ms,
                min_speech_ms=self.settings.voice_min_speech_ms,
                silence_threshold=self.settings.voice_silence_threshold,
                input_device=self.settings.voice_input_device,
                input_device_name=self.settings.voice_input_device_name,
            )
        )
        self.logger.write(
            "recording_finished",
            duration_seconds=audio_data.duration_seconds,
            speech_ms=audio_data.speech_ms,
            stop_reason=audio_data.stop_reason,
            rms=audio_data.rms,
            peak=audio_data.peak,
        )
        try:
            self.console.write("[voz] processando áudio...")
            transcription = self.stt.transcribe(
                audio_data,
                TranscriptionOptions(
                    language=self.settings.voice_stt_language,
                    timeout_seconds=self.settings.voice_stt_timeout_seconds,
                ),
            )
            self.logger.write(
                "transcription_finished",
                duration_seconds=transcription.duration_seconds,
                language=transcription.language,
                confidence=transcription.confidence,
                transcript=transcription.text,
            )
            self.console.write(
                f'[voz] transcrição: "{transcription.text}"'
            )
            decision = confirm_transcription(
                transcription.text,
                required=(
                    self.settings.voice_confirm_transcription
                    or may_be_sensitive(transcription.text)
                ),
                console=self.console,
            )
            if decision == ConfirmationDecision.RERECORD:
                return {"ok": False, "rerecord": True}
            if decision == ConfirmationDecision.CANCEL:
                raise VoiceCancelled("transcricao cancelada")
            if decision == ConfirmationDecision.SEND:
                self.console.write("[voz] Transcrição confirmada.")
            routing_text = normalize_technical_transcript(transcription.text)
            self.logger.write(
                "transcription_routing",
                transcript_original=transcription.text,
                transcript_normalized=routing_text,
                changed=routing_text != transcription.text,
            )
            self.console.write("[assistente] pensando...")
            result = self.supervisor.run(
                routing_text,
                event_callback=self._event,
            )
            if not result.get("ok"):
                if result.get("error") == "approval_required":
                    raise VoiceCancelled("acao sensivel cancelada")
                self.console.write(
                    f"[erro] {result.get('error', 'assistant_error')}: "
                    f"{result.get('message', '')}"
                )
                return result
            answer = str(result.get("answer") or "")
            self.console.write("\n[assistente]")
            self.console.write(answer)
            web_result = result.get("web") or {}
            sources = web_result.get("sources") or []
            lexicon_path = Path(__file__).with_name(
                "pronunciation_ptbr.json"
            )
            status_replacements = (
                semantic_replacements(lexicon_path)
                if self.settings.voice_translate_common_status_terms
                else {}
            )
            if web_result.get("used"):
                spoken = prepare_research_spoken_text(
                    answer,
                    source_count=len(sources),
                    max_characters=self.settings.voice_max_spoken_characters,
                    read_code=self.settings.voice_read_code,
                    read_urls=self.settings.voice_read_urls,
                    summarize_long=self.settings.voice_summarize_long_responses,
                    semantic_replacements=status_replacements,
                )
            else:
                spoken = prepare_spoken_text(
                    answer,
                    max_characters=self.settings.voice_max_spoken_characters,
                    read_code=self.settings.voice_read_code,
                    read_urls=self.settings.voice_read_urls,
                    summarize_long=self.settings.voice_summarize_long_responses,
                    semantic_replacements=status_replacements,
                )
            if self._spoken_status:
                spoken = self._spoken_status
            if spoken:
                spoken = normalize_for_speech(
                    spoken,
                    self.settings.voice_tts_provider,
                    self.settings.voice_style,
                    lexicon_path=lexicon_path,
                )
            tts_metrics = None
            if spoken and getattr(self.tts, "mode", None) != "silent":
                self.console.write("[assistente] falando... Esc interrompe")
                self.logger.write(
                    "synthesis_started",
                    spoken_characters=len(spoken),
                )
                synthesis_options = SynthesisOptions(
                    rate=self.settings.voice_tts_rate,
                    volume=self.settings.voice_tts_volume,
                    timeout_seconds=self.settings.voice_tts_timeout_seconds,
                )
                stop_event = threading.Event()
                if (
                    self.settings.voice_tts_streaming
                    and hasattr(self.tts, "speak_streaming")
                ):
                    tts_metrics = self.tts.speak_streaming(
                        spoken,
                        synthesis_options,
                        chunk_min_characters=(
                            self.settings.voice_tts_chunk_min_characters
                        ),
                        chunk_max_characters=(
                            self.settings.voice_tts_chunk_max_characters
                        ),
                        queue_size=self.settings.voice_tts_queue_size,
                        stop_event=stop_event,
                        event_callback=self._tts_event,
                    )
                    interrupted = bool(
                        tts_metrics.get("interrupted")
                    )
                    self.logger.write(
                        "tts_stream_finished",
                        **tts_metrics,
                    )
                else:
                    interrupted = self.tts.speak(
                        spoken,
                        synthesis_options,
                        stop_event=stop_event,
                    )
                self.logger.write(
                    "playback_finished", interrupted=interrupted
                )
                if interrupted:
                    self.console.write("[voz] interrompido")
            return {
                **result,
                "transcription": {
                    "text": transcription.text,
                    "routing_text": routing_text,
                    "language": transcription.language,
                    "confidence": transcription.confidence,
                },
                "spoken_text": spoken,
                "tts": tts_metrics,
            }
        finally:
            self._interaction_active = False
            if not self.settings.voice_keep_recordings:
                removed = self.audio.remove_temporary(audio_data)
                if removed:
                    self.logger.write("temporary_removed")

    def _event(self, event: str, values: dict[str, Any]) -> None:
        if event == "tool_start":
            name = str(values.get("name") or "")
            if name.startswith("web_"):
                self.console.write("[assistente] pesquisando...")
            elif name == "review_codex_session":
                self.console.write(
                    "[assistente] consultando a sessao compartilhada do Codex..."
                )
            elif name != "delegate_to_codex":
                self.console.write(
                    f"[assistente] executando ferramenta: {name}"
                )
        elif event == "codex_sending":
            self.console.write("[assistente] enviando tarefa ao Codex...")
        elif event == "codex_turn_started":
            thread_id = str(values.get("thread_id") or "")[:8]
            turn_id = str(values.get("turn_id") or "")[:8]
            self.console.write(f"[Codex] thread: {thread_id}")
            self.console.write(f"[Codex] turn iniciado: {turn_id}")
        elif event == "codex_job_started":
            job_id = str(values.get("job_id") or "")[:8]
            self.console.write("[Codex] tarefa iniciada.")
            self.console.write(f"[Codex] job: {job_id}")
            self.console.write("[Codex] estado: executando.")
            self._spoken_status = "Enviei a tarefa ao Codex."
        elif event == "codex_working":
            self.console.write("[Codex] trabalhando...")
        elif event in {"codex_job_completed", "codex_result_received"}:
            self.console.write("[Codex] tarefa concluída.")
            self.console.write("[assistente] resultado recebido.")
            self._notify_short(
                "A tarefa do Codex foi concluída.",
                notify=bool(values.get("notify", event == "codex_job_completed")),
            )
        elif event == "codex_job_status":
            status = str(values.get("status") or "")
            if status in {"queued", "starting", "running", "reconnecting"}:
                self.console.write("[Codex] ainda trabalhando.")
                self._notify_short(
                    "O Codex ainda está trabalhando.",
                    notify=bool(values.get("notify")),
                )
        elif event == "codex_job_interrupted":
            self.console.write("[Codex] tarefa cancelada.")
            self._notify_short("Tarefa cancelada.", notify=True)
        elif event == "codex_job_steered":
            self.console.write("[Codex] direção humana aceita.")
        elif event == "codex_job_failed":
            self.console.write("[Codex] tarefa falhou.")
            self._notify_short(
                "O Codex encontrou um problema durante a execução.",
                notify=True,
            )
        elif event == "codex_completed":
            self.console.write("[Codex] tarefa concluída.")
        elif event == "codex_failed":
            self.console.write(
                "[Codex] não foi possível iniciar a tarefa: "
                + str(values.get("error") or values.get("status") or "erro")
            )
        elif event == "assistant_analyzing_result":
            self.console.write("[assistente] analisando o resultado...")
        self.logger.write(
            "assistant_" + event,
            tool=values.get("name"),
            ok=values.get("ok"),
        )

    def _notify_short(self, text: str, *, notify: bool) -> None:
        if not notify:
            return
        if self._interaction_active:
            self._spoken_status = text
            return
        if getattr(self.tts, "mode", None) == "silent":
            return
        with self._notification_lock:
            try:
                self.tts.speak(
                    text,
                    SynthesisOptions(
                        rate=self.settings.voice_tts_rate,
                        volume=self.settings.voice_tts_volume,
                        timeout_seconds=self.settings.voice_tts_timeout_seconds,
                    ),
                    stop_event=threading.Event(),
                )
                self.logger.write(
                    "background_notification_spoken",
                    kind="codex_job",
                )
            except Exception as exc:
                self.logger.write(
                    "background_notification_failed",
                    error=str(exc),
                )

    def _tts_event(self, event: str, values: dict[str, Any]) -> None:
        self.logger.write("tts_" + event, **values)
