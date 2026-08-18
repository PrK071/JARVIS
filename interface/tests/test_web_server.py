"""Testes do servidor web e da validação do chat."""

from __future__ import annotations

import os
import unittest
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


if __name__ == "__main__":
    unittest.main()
