from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any


class ServerError(RuntimeError):
    pass


class ServerTimeoutError(ServerError):
    pass


class ServerUnavailableError(ServerError):
    pass


class LlamaClient:
    def __init__(self, base_url: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def props(self) -> dict[str, Any]:
        return self._request("GET", "/props")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": "local",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        return self._request("POST", "/v1/chat/completions", payload)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": "local",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format:
            payload["response_format"] = response_format
        return self._stream_request("/v1/chat/completions", payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4096]
            raise ServerError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ServerTimeoutError(f"llama-server timeout: {exc}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ServerTimeoutError(f"llama-server timeout: {exc}") from exc
            raise ServerUnavailableError(f"llama-server indisponivel: {exc}") from exc
        if not isinstance(result, dict):
            raise ServerError("resposta inesperada do llama-server")
        return result

    def _stream_request(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="strict").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise ServerError("evento SSE inesperado do llama-server")
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    value = json.loads(data)
                    if not isinstance(value, dict):
                        raise ServerError("evento SSE nao e um objeto")
                    yield value
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4096]
            raise ServerError(f"llama-server HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ServerTimeoutError(f"llama-server timeout: {exc}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ServerTimeoutError(f"llama-server timeout: {exc}") from exc
            raise ServerUnavailableError(f"llama-server indisponivel: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServerError(f"evento SSE invalido: {exc}") from exc
