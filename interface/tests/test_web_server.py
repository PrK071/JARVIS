"""Testes do servidor web, da validação do chat e da transcrição local."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import web_server


class WebServerHelpersTest(unittest.TestCase):
    def test_safe_history_filters_roles_and_limits_items(self) -> None:
        history = [
            {"role": "system", "content": "ignorar"},
            {"role": "user", "content": "pergunta"},
            {"role": "assistant", "content": "resposta"},
            {"role": "user", "content": "   "},
            "inválido",
        ]

        self.assertEqual(
            web_server._safe_history(history),
            [
                {"role": "user", "content": "pergunta"},
                {"role": "assistant", "content": "resposta"},
            ],
        )

    def test_extract_output_text_uses_all_message_chunks(self) -> None:
        payload = {
            "output": [
                {"type": "reasoning", "content": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Primeira parte."},
                        {"type": "output_text", "text": "Segunda parte."},
                    ],
                },
            ]
        }

        self.assertEqual(
            web_server._extract_output_text(payload),
            "Primeira parte.\nSegunda parte.",
        )

    def test_request_openai_requires_server_side_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "API_KEY_MISSING"):
                web_server._request_openai("Olá", [])


class HardwareTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        web_server._hardware_cache = None
        web_server._hardware_monitor = None

    def tearDown(self) -> None:
        web_server._hardware_cache = None
        web_server._hardware_monitor = None

    def test_telemetry_reports_real_sensor_reading(self) -> None:
        leitura = {
            "ok": True,
            "cpu_temperature_c": 48.0,
            "cpu_temperature_available": True,
            "cpu_temperature_source": "LibreHardwareMonitor/CPU Package",
        }
        web_server._hardware_monitor = type("Fake", (), {"read": lambda self: leitura})()

        self.assertEqual(web_server._hardware_telemetry(), leitura)

    def test_telemetry_is_throttled_between_reads(self) -> None:
        chamadas = []

        class Contador:
            def read(self) -> dict[str, object]:
                chamadas.append(1)
                return {"ok": True, "cpu_temperature_c": 50.0, "cpu_temperature_available": True}

        web_server._hardware_monitor = Contador()
        web_server._hardware_telemetry()
        web_server._hardware_telemetry()

        self.assertEqual(len(chamadas), 1)

    def test_sensor_failure_is_reported_as_unavailable(self) -> None:
        class Quebrado:
            def read(self) -> dict[str, object]:
                raise OSError("sensor offline")

        web_server._hardware_monitor = Quebrado()
        resultado = web_server._hardware_telemetry()

        self.assertFalse(resultado["cpu_temperature_available"])
        self.assertIsNone(resultado["cpu_temperature_c"])
        self.assertIn("sensor offline", resultado["error"])
def _wav_bytes(samples, sample_rate: int = 16_000) -> bytes:
    """Build a small WAV blob; PyAV decodes it the same way it decodes webm."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


class TranscriptionDecodeTest(unittest.TestCase):
    """The decode step is the new code; the STT provider itself is tern's."""

    def setUp(self) -> None:
        try:
            import av  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("PyAV/numpy ausentes")

    def test_decode_resamples_to_the_rate_expected_by_the_stt(self) -> None:
        import numpy as np

        seconds = 1.0
        source_rate = 48_000
        t = np.linspace(0, seconds, int(source_rate * seconds), endpoint=False)
        tone = (np.sin(2 * np.pi * 440 * t) * 12_000).astype(np.int16)

        samples = web_server._decode_audio(_wav_bytes(tone, source_rate))

        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(
            samples.size / web_server.STT_SAMPLE_RATE, seconds, places=1
        )

    def test_decode_rejects_a_recording_shorter_than_the_floor(self) -> None:
        import numpy as np

        tiny = np.zeros(160, dtype=np.int16)  # 10 ms
        with self.assertRaisesRegex(RuntimeError, "AUDIO_TOO_SHORT"):
            web_server._decode_audio(_wav_bytes(tiny))

    def test_decode_rejects_bytes_that_are_not_audio(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "AUDIO_DECODE_FAILED"):
            web_server._decode_audio(b"not audio at all")


class ProviderStoreTest(unittest.TestCase):
    """Connections are configured from the UI, so the store guards the keys."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            web_server, "PROVIDERS_PATH", Path(self._tmp.name) / "providers.json"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def _save(self, **overrides):
        payload = {
            "label": "Teste",
            "format": "openai-chat",
            "base_url": "https://api.exemplo.com/v1",
            "model": "modelo-x",
            "api_key": "chave-super-secreta",
        }
        payload.update(overrides)
        return web_server._save_provider(payload)

    def test_saved_connection_becomes_the_active_one(self) -> None:
        provider, error = self._save()
        self.assertIsNone(error)
        self.assertEqual(provider["id"], "teste")
        active = web_server._active_provider()
        self.assertEqual(active["id"], "teste")

    def test_public_view_never_exposes_the_raw_key(self) -> None:
        provider, _ = self._save()
        public = web_server._public_provider(provider)

        self.assertNotIn("api_key", public)
        self.assertTrue(public["has_key"])
        self.assertNotIn("chave-super-secreta", json.dumps(public))
        self.assertEqual(public["key_hint"], "chav••••••reta")

    def test_editing_without_a_new_key_keeps_the_stored_one(self) -> None:
        self._save()
        updated, error = self._save(model="modelo-y", api_key="")

        self.assertIsNone(error)
        self.assertEqual(updated["model"], "modelo-y")
        self.assertEqual(updated["api_key"], "chave-super-secreta")

    def test_endpoint_must_be_http_or_https(self) -> None:
        _, error = self._save(base_url="ftp://arquivos.exemplo.com")
        self.assertIn("http://", error)

    def test_pasting_the_full_endpoint_keeps_only_the_root(self) -> None:
        provider, error = self._save(
            base_url="https://api.deepseek.com/chat/completions"
        )
        self.assertIsNone(error)
        self.assertEqual(provider["base_url"], "https://api.deepseek.com")

    def test_deepseek_composes_the_endpoint_the_orchestrator_uses(self) -> None:
        provider, _ = self._save(
            label="DeepSeek",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        captured = {}

        def fake_post(url, headers, body):
            captured["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(web_server, "_post_json", fake_post):
            web_server._request_provider(provider, "oi", [])

        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")

    def test_unknown_format_is_rejected(self) -> None:
        _, error = self._save(format="telepatia")
        self.assertIn("Formato", error)

    def test_deleting_the_active_connection_promotes_another(self) -> None:
        self._save(label="Primeira")
        self._save(label="Segunda")
        self.assertEqual(web_server._active_provider()["id"], "segunda")

        self.assertTrue(web_server._delete_provider("segunda"))
        self.assertEqual(web_server._active_provider()["id"], "primeira")

    def test_environment_key_still_provides_a_connection(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "chave-do-ambiente"}):
            active = web_server._active_provider()
        self.assertIsNotNone(active)
        self.assertEqual(active["id"], "env")
        self.assertEqual(active["format"], "openai-responses")


class ProviderRequestTest(unittest.TestCase):
    """Each format has to speak the dialect the provider expects."""

    def test_openai_chat_sends_messages_and_reads_the_choice(self) -> None:
        provider = {
            "format": "openai-chat",
            "base_url": "https://api.exemplo.com/v1",
            "model": "modelo-x",
            "api_key": "abc",
        }
        captured = {}

        def fake_post(url, headers, body):
            captured.update(url=url, headers=headers, body=body)
            return {"choices": [{"message": {"content": " olá "}}]}

        with patch.object(web_server, "_post_json", fake_post):
            reply = web_server._request_provider(provider, "oi", [])

        self.assertEqual(reply, "olá")
        self.assertEqual(captured["url"], "https://api.exemplo.com/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer abc")
        self.assertEqual(captured["body"]["messages"][-1], {"role": "user", "content": "oi"})

    def test_openai_chat_allows_a_local_server_without_a_key(self) -> None:
        provider = {
            "format": "openai-chat",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "llama",
            "api_key": "",
        }
        with patch.object(
            web_server,
            "_post_json",
            lambda url, headers, body: (
                self.assertNotIn("Authorization", headers),
                {"choices": [{"message": {"content": "ok"}}]},
            )[1],
        ):
            self.assertEqual(web_server._request_provider(provider, "oi", []), "ok")

    def test_anthropic_uses_its_own_header_and_body(self) -> None:
        provider = {
            "format": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude",
            "api_key": "sk-ant",
        }
        captured = {}

        def fake_post(url, headers, body):
            captured.update(url=url, headers=headers, body=body)
            return {"content": [{"type": "text", "text": "resposta"}]}

        with patch.object(web_server, "_post_json", fake_post):
            reply = web_server._request_provider(provider, "oi", [])

        self.assertEqual(reply, "resposta")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["headers"]["x-api-key"], "sk-ant")
        self.assertIn("system", captured["body"])

    def test_a_key_less_remote_connection_fails_before_the_request(self) -> None:
        provider = {
            "format": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude",
            "api_key": "",
        }
        with self.assertRaisesRegex(RuntimeError, "API_KEY_MISSING"):
            web_server._request_provider(provider, "oi", [])

    def test_empty_reply_is_reported_instead_of_returned(self) -> None:
        provider = {
            "format": "openai-chat",
            "base_url": "https://api.exemplo.com/v1",
            "model": "m",
            "api_key": "k",
        }
        with patch.object(web_server, "_post_json", lambda *a: {"choices": []}):
            with self.assertRaisesRegex(RuntimeError, "EMPTY_MODEL_RESPONSE"):
                web_server._request_provider(provider, "oi", [])


class ProviderErrorMessageTest(unittest.TestCase):
    def test_rejected_key_names_the_host_that_refused(self) -> None:
        """401 sem o host escondia a causa real: chave certa, endpoint errado."""
        mensagem = web_server._provider_error_message(
            "HTTP_401: incorrect api key", host="api.openai.com"
        )

        self.assertIn("api.openai.com", mensagem)
        self.assertIn("endpoint", mensagem.lower())

    def test_provider_host_comes_from_base_url(self) -> None:
        provider = {"format": "openai-chat", "base_url": "https://api.deepseek.com/v1"}

        self.assertEqual(web_server._provider_host(provider), "api.deepseek.com")

    def test_provider_host_falls_back_to_format_default(self) -> None:
        self.assertEqual(
            web_server._provider_host({"format": "anthropic"}), "api.anthropic.com"
        )

    def test_invalid_model_is_reported_as_model_problem(self) -> None:
        mensagem = web_server._provider_error_message('HTTP_400: {"error":"model not found"}')

        self.assertIn("Modelo inválido", mensagem)


class ProviderPresetsTest(unittest.TestCase):
    def test_presets_cover_known_vendors_with_their_own_endpoints(self) -> None:
        presets = {preset["id"]: preset for preset in web_server.PROVIDER_PRESETS}

        self.assertIn("deepseek", presets)
        self.assertEqual(presets["deepseek"]["base_url"], "https://api.deepseek.com/v1")
        self.assertEqual(presets["openai"]["base_url"], "https://api.openai.com/v1")
        self.assertEqual(presets["anthropic"]["format"], "anthropic")

    def test_every_preset_declares_a_known_format(self) -> None:
        for preset in web_server.PROVIDER_PRESETS:
            self.assertIn(preset["format"], web_server.PROVIDER_FORMATS, preset["id"])

    def test_no_preset_points_to_another_vendor_endpoint(self) -> None:
        for preset in web_server.PROVIDER_PRESETS:
            host = web_server._provider_host(preset) or ""
            if preset["id"] in {"ollama", "lmstudio", "openrouter", "gemini"}:
                continue
            self.assertIn(preset["id"], host, f"{preset['id']} aponta para {host}")


class EmptyReplyDiagnosisTest(unittest.TestCase):
    def test_finish_reason_length_is_reported_as_token_limit(self) -> None:
        payload = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}

        self.assertEqual(web_server._empty_reply_reason(payload), "TRUNCATED_BY_TOKEN_LIMIT")

    def test_reasoning_without_content_is_token_limit(self) -> None:
        """Modelo de raciocinio gastou o orcamento antes do texto final."""
        payload = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "", "reasoning_content": "pensando"}}
            ]
        }

        self.assertEqual(web_server._empty_reply_reason(payload), "TRUNCATED_BY_TOKEN_LIMIT")

    def test_plain_empty_stays_empty(self) -> None:
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}

        self.assertEqual(web_server._empty_reply_reason(payload), "EMPTY_MODEL_RESPONSE")

    def test_token_limit_message_tells_the_user_what_to_change(self) -> None:
        mensagem = web_server._provider_error_message("TRUNCATED_BY_TOKEN_LIMIT")

        self.assertIn("TRIADE_PROVIDER_MAX_TOKENS", mensagem)
        self.assertIn(str(web_server.MAX_PROVIDER_OUTPUT_TOKENS), mensagem)

    def test_output_budget_is_generous_enough_for_reasoning_models(self) -> None:
        self.assertGreaterEqual(web_server.MAX_PROVIDER_OUTPUT_TOKENS, 4_000)


if __name__ == "__main__":
    unittest.main()
