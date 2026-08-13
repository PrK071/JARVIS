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


if __name__ == "__main__":
    unittest.main()
