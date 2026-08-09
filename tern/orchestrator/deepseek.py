from __future__ import annotations

import json
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .codex_state import FileMutex


DEEPSEEK_SYSTEM_PROMPT = (
    "Voce e um agente consultivo dentro de um assistente local. Analise problemas, "
    "critique solucoes e proponha abordagens. Voce nao possui acesso direto ao "
    "computador. Nao afirme que executou comandos ou modificou arquivos. Ao receber "
    "contexto de outro agente, diferencie fatos observados de inferencias."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeepSeekError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable


class DeepSeekClient:
    """Small OpenAI-compatible HTTP client with no local tool capabilities."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: int = 180,
        max_retries: int = 2,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.enabled = enabled
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self._opener = opener
        self._sleeper = sleeper

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise DeepSeekError("deepseek_disabled", "DeepSeek esta desativado")
        if not self.api_key:
            raise DeepSeekError(
                "deepseek_api_key_missing", "API key do DeepSeek nao configurada"
            )
        if not self.model:
            raise DeepSeekError(
                "deepseek_model_not_found", "Modelo do DeepSeek nao configurado"
            )
        if cancel_event is not None and cancel_event.is_set():
            raise DeepSeekError("deepseek_cancelled", "Consulta cancelada")

        payload = json.dumps(
            {"model": self.model, "messages": messages, "stream": False},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: DeepSeekError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                if cancel_event is not None and cancel_event.is_set():
                    raise DeepSeekError("deepseek_cancelled", "Consulta cancelada")
                return self._parse_response(raw)
            except HTTPError as exc:
                last_error = self._http_error(exc)
            except (TimeoutError, socket.timeout) as exc:
                last_error = DeepSeekError(
                    "deepseek_timeout", "A consulta ao DeepSeek expirou", retryable=True
                )
            except URLError as exc:
                timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
                last_error = DeepSeekError(
                    "deepseek_timeout" if timed_out else "deepseek_api_error",
                    "A consulta ao DeepSeek expirou"
                    if timed_out
                    else "Nao foi possivel acessar a API do DeepSeek",
                    retryable=True,
                )
            if not last_error.retryable or attempt >= self.max_retries:
                raise last_error
            if cancel_event is not None and cancel_event.is_set():
                raise DeepSeekError("deepseek_cancelled", "Consulta cancelada")
            self._sleeper(min(2.0, 0.5 * (2**attempt)))
        assert last_error is not None
        raise last_error

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        on_delta: Callable[[str], None],
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        """Consume DeepSeek's SSE stream and emit only final-answer text deltas."""
        if not self.enabled:
            raise DeepSeekError("deepseek_disabled", "DeepSeek esta desativado")
        if not self.api_key:
            raise DeepSeekError(
                "deepseek_api_key_missing", "API key do DeepSeek nao configurada"
            )
        if not self.model:
            raise DeepSeekError(
                "deepseek_model_not_found", "Modelo do DeepSeek nao configurado"
            )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        attempts = 0
        while True:
            emitted = False
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    content_parts: list[str] = []
                    usage: dict[str, Any] = {}
                    response_model = self.model
                    finish_reason = None
                    for raw_line in response:
                        if cancel_event is not None and cancel_event.is_set():
                            raise DeepSeekError("deepseek_cancelled", "Consulta cancelada")
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise DeepSeekError(
                                "deepseek_invalid_response", "Stream invalido da API do DeepSeek"
                            ) from exc
                        response_model = str(chunk.get("model") or response_model)
                        if isinstance(chunk.get("usage"), dict):
                            usage = chunk["usage"]
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            emitted = True
                            content_parts.append(text)
                            on_delta(text)
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice.get("finish_reason")
                content = "".join(content_parts).strip()
                if not content:
                    raise DeepSeekError(
                        "deepseek_invalid_response", "Stream do DeepSeek terminou sem resposta"
                    )
                return {
                    "response": content,
                    "model": response_model,
                    "usage": self._normalize_usage(usage),
                    "finish_reason": finish_reason,
                }
            except DeepSeekError:
                raise
            except HTTPError as exc:
                error = self._http_error(exc)
            except (TimeoutError, socket.timeout) as exc:
                error = DeepSeekError(
                    "deepseek_timeout", "A consulta ao DeepSeek expirou", retryable=True
                )
            except URLError as exc:
                timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
                error = DeepSeekError(
                    "deepseek_timeout" if timed_out else "deepseek_api_error",
                    "A consulta ao DeepSeek expirou"
                    if timed_out
                    else "Nao foi possivel acessar a API do DeepSeek",
                    retryable=True,
                )
            if emitted or not error.retryable or attempts >= self.max_retries:
                raise error
            attempts += 1
            self._sleeper(min(2.0, 0.5 * (2 ** (attempts - 1))))

    @staticmethod
    def _safe_error_detail(exc: HTTPError) -> str:
        try:
            raw = exc.read(8192)
            value = json.loads(raw.decode("utf-8", errors="replace"))
            detail = value.get("error") if isinstance(value, dict) else None
            if isinstance(detail, dict):
                return str(detail.get("message") or "")[:500]
            return str(detail or "")[:500]
        except Exception:
            return ""

    def _http_error(self, exc: HTTPError) -> DeepSeekError:
        status = int(exc.code)
        detail = self._safe_error_detail(exc).casefold()
        if status in {401, 403}:
            return DeepSeekError(
                "deepseek_auth_failed", "Autenticacao do DeepSeek falhou", status=status
            )
        if status == 404 or (status == 422 and "model" in detail):
            return DeepSeekError(
                "deepseek_model_not_found", "Modelo do DeepSeek nao encontrado", status=status
            )
        if status == 429:
            return DeepSeekError(
                "deepseek_rate_limited",
                "Limite de requisicoes do DeepSeek atingido",
                status=status,
                retryable=True,
            )
        if status in {413, 422} and any(
            term in detail for term in ("context", "token", "length", "too long")
        ):
            return DeepSeekError(
                "deepseek_context_too_large",
                "Contexto enviado ao DeepSeek excede o limite",
                status=status,
            )
        return DeepSeekError(
            "deepseek_api_error",
            f"API do DeepSeek retornou HTTP {status}",
            status=status,
            retryable=status in {408, 409, 425, 500, 502, 503, 504},
        )

    def _parse_response(self, raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
            choice = value["choices"][0]
            message = choice["message"]
            content = message["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty content")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            raise DeepSeekError(
                "deepseek_invalid_response", "Resposta invalida da API do DeepSeek"
            )
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        normalized_usage = self._normalize_usage(usage)
        return {
            "response": content.strip(),
            "model": str(value.get("model") or self.model),
            "usage": normalized_usage,
            "finish_reason": choice.get("finish_reason"),
        }

    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        return {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
            "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
        }


class DeepSeekSessionManager:
    VERSION = 1

    def __init__(
        self,
        *,
        client: DeepSeekClient,
        state_dir: Path,
        projects: Any,
        logger: Any | None = None,
        max_recent_turns: int = 20,
        max_context_characters: int = 60_000,
        max_message_characters: int = 8_000,
        max_summary_characters: int = 8_000,
    ):
        self.client = client
        self.state_dir = state_dir
        self.projects = projects
        self.logger = logger
        self.max_recent_turns = max(1, max_recent_turns)
        self.max_context_characters = max(4_000, max_context_characters)
        self.max_message_characters = max(500, max_message_characters)
        self.max_summary_characters = max(1_000, max_summary_characters)
        self.path = state_dir / "deepseek-sessions.json"
        self.lock_path = state_dir / "deepseek-sessions.lock"
        self._cancel_events: dict[str, Event] = {}

    def _default(self) -> dict[str, Any]:
        return {"version": self.VERSION, "active_by_project": {}, "sessions": []}

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("sessions"), list):
                raise ValueError("invalid state")
            if not isinstance(value.get("active_by_project"), dict):
                value["active_by_project"] = {}
            return value
        except FileNotFoundError:
            return self._default()
        except (OSError, json.JSONDecodeError, ValueError):
            return self._default()

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".tmp-{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def _project_key(project: str) -> str:
        return str(Path(project).resolve()).casefold()

    def _resolve_project(self, project_path: str | None) -> dict[str, Any]:
        if project_path:
            result = self.projects.resolve(path_hint=project_path)
        else:
            result = self.projects.resolve()
        if not result.get("ok"):
            return result
        return {"ok": True, "project": str(Path(result["root"]).resolve())}

    def _new_session_value(self, project: str) -> dict[str, Any]:
        now = _utc_now()
        return {
            "session_id": str(uuid.uuid4()),
            "project": project,
            "model": self.client.model,
            "created_at": now,
            "updated_at": now,
            "summary": None,
            "summary_message_count": 0,
            "summary_updated_at": None,
            "summary_source_until_message_id": None,
            "messages": [],
            "last_usage": None,
            "usage": {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 0,
                "reasoning_tokens": 0,
            },
            "active_generation": None,
            "last_cancelled_generation_id": None,
        }

    @staticmethod
    def _find_session(state: dict[str, Any], session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        return next(
            (item for item in state["sessions"] if item.get("session_id") == session_id),
            None,
        )

    def new_session(self, project_path: str | None = None) -> dict[str, Any]:
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {"ok": False, "error": resolved.get("error", "project_not_found")}
        project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._new_session_value(project)
            state["sessions"].append(session)
            state["active_by_project"][self._project_key(project)] = session["session_id"]
            self._write_unlocked(state)
        return self._session_result(session)

    @staticmethod
    def _message(source: str, role: str, content: str) -> dict[str, str]:
        return {
            "id": str(uuid.uuid4()),
            "source": source,
            "role": role,
            "content": content,
            "created_at": _utc_now(),
        }

    def _get_or_create_session(
        self,
        state: dict[str, Any],
        project: str,
        *,
        continue_current_session: bool,
    ) -> dict[str, Any]:
        key = self._project_key(project)
        session = None
        if continue_current_session:
            session = self._find_session(state, state["active_by_project"].get(key))
            if (
                session is not None
                and self.client.model
                and session.get("model")
                and session.get("model") != self.client.model
                and session.get("messages")
            ):
                session = None
        if session is None:
            session = self._new_session_value(project)
            state["sessions"].append(session)
            state["active_by_project"][key] = session["session_id"]
        elif self.client.model and not session.get("messages"):
            session["model"] = self.client.model
        return session

    def _compact(self, session: dict[str, Any]) -> bool:
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        recent_count = self.max_recent_turns * 2
        older_count = max(0, len(messages) - recent_count)
        already = int(session.get("summary_message_count") or 0)
        batch_messages = max(4, min(10, self.max_recent_turns))
        if older_count - already < batch_messages:
            return False
        entries: list[str] = []
        if session.get("summary"):
            entries.append(str(session["summary"]))
        for message in messages[already:older_count]:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "").strip().replace("\x00", "")
            content = content[:600]
            entries.append(
                f"[{message.get('source', 'unknown')}/{message.get('role', 'user')}] {content}"
            )
        combined = "\n".join(entries)
        session["summary"] = combined[-self.max_summary_characters :] or None
        session["summary_message_count"] = older_count
        session["summary_updated_at"] = _utc_now()
        session["summary_source_until_message_id"] = (
            messages[older_count - 1].get("id") if older_count else None
        )
        return True

    def _api_messages(
        self,
        session: dict[str, Any],
        *,
        temporary_context: str | None = None,
    ) -> list[dict[str, str]]:
        result = [{"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT}]
        summary = str(session.get("summary") or "").strip()
        if summary:
            summary_budget = min(
                self.max_summary_characters,
                max(500, self.max_context_characters // 4),
            )
            result.append(
                {
                    "role": "system",
                    "content": f"Resumo historico local:\n{summary[-summary_budget:]}",
                }
            )
        if temporary_context and temporary_context.strip():
            context_budget = max(500, self.max_context_characters // 3)
            result.append(
                {
                    "role": "system",
                    "content": (
                        "Contexto temporario para esta solicitacao (nao e historico):\n"
                        + temporary_context.strip()[:context_budget]
                    ),
                }
            )
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        summarized = int(session.get("summary_message_count") or 0)
        candidates = messages[summarized:]
        budget = self.max_context_characters - sum(len(item["content"]) for item in result)
        selected: list[dict[str, str]] = []
        for item in reversed(candidates):
            if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "")[: self.max_message_characters]
            if not content:
                continue
            allowed = max(0, budget - 32)
            if allowed <= 0:
                break
            content = content[-allowed:]
            selected.append({"role": str(item["role"]), "content": content})
            budget -= len(content) + 32
        result.extend(reversed(selected))
        return result

    def _log(self, event: str, **values: Any) -> None:
        writer = getattr(self.logger, "write_event", None)
        if callable(writer):
            writer(event, **values)

    @staticmethod
    def _error_result(error: DeepSeekError, *, session_id: str | None = None) -> dict[str, Any]:
        return {
            "ok": False,
            "session_id": session_id,
            "model": None,
            "response": None,
            "usage": None,
            "error": error.code,
            "message": error.message,
        }

    def delegate(
        self,
        task: str,
        *,
        project_path: str | None = None,
        continue_current_session: bool = True,
        context: str | None = None,
        source: str = "qwen",
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if not self.client.enabled:
            return self._error_result(
                DeepSeekError("deepseek_disabled", "DeepSeek esta desativado")
            )
        if not self.client.api_key:
            return self._error_result(
                DeepSeekError(
                    "deepseek_api_key_missing", "API key do DeepSeek nao configurada"
                )
            )
        task = task.strip()
        if not task:
            return self._error_result(
                DeepSeekError("deepseek_invalid_response", "Tarefa vazia")
            )
        if len(task) > self.max_context_characters - len(DEEPSEEK_SYSTEM_PROMPT) - 512:
            return self._error_result(
                DeepSeekError(
                    "deepseek_context_too_large",
                    "Mensagem atual excede o limite de contexto seguro",
                )
            )
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {
                "ok": False,
                "session_id": None,
                "model": self.client.model or None,
                "response": None,
                "usage": None,
                "error": resolved.get("error", "project_not_found"),
            }
        project = resolved["project"]
        generation_id = str(uuid.uuid4())
        cancel_event = Event()
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._get_or_create_session(
                state, project, continue_current_session=continue_current_session
            )
            active = session.get("active_generation")
            if isinstance(active, dict) and not active.get("cancelled"):
                return self._error_result(
                    DeepSeekError("deepseek_busy", "Ja existe uma consulta em andamento"),
                    session_id=str(session["session_id"]),
                )
            session["messages"].append(self._message(source, "user", task))
            session["active_generation"] = {
                "id": generation_id,
                "started_at": _utc_now(),
                "cancelled": False,
            }
            session["updated_at"] = _utc_now()
            summary_updated = self._compact(session)
            api_messages = self._api_messages(session, temporary_context=context)
            session_id = str(session["session_id"])
            self._write_unlocked(state)
        self._cancel_events[generation_id] = cancel_event
        payload_chars = sum(len(item["content"]) for item in api_messages)
        self._log(
            "deepseek_request_started",
            session_id=session_id,
            project=project,
            model=self.client.model,
            message_count=len(api_messages),
            payload_characters=payload_chars,
        )
        if summary_updated:
            self._log(
                "deepseek_summary_updated",
                session_id=session_id,
                source_until_message_id=session.get("summary_source_until_message_id"),
            )
        try:
            if on_delta is not None:
                response = self.client.stream_chat(
                    api_messages,
                    on_delta=on_delta,
                    cancel_event=cancel_event,
                )
            else:
                response = self.client.chat(api_messages, cancel_event=cancel_event)
        except DeepSeekError as exc:
            self._finish_generation(session_id, generation_id, cancelled=exc.code == "deepseek_cancelled")
            self._log("deepseek_request_failed", session_id=session_id, error=exc.code)
            if exc.code == "deepseek_cancelled":
                self._log("deepseek_request_cancelled", session_id=session_id)
            return self._error_result(exc, session_id=session_id)
        finally:
            self._cancel_events.pop(generation_id, None)

        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(state, session_id)
            active = session.get("active_generation") if session else None
            discarded = (
                session is None
                or not isinstance(active, dict)
                or active.get("id") != generation_id
                or active.get("cancelled")
            )
            if discarded:
                if session is not None and isinstance(active, dict) and active.get("id") == generation_id:
                    session["active_generation"] = None
                    session["last_cancelled_generation_id"] = generation_id
                    session["updated_at"] = _utc_now()
                    self._write_unlocked(state)
                summary_updated = False
            else:
                session["messages"].append(
                    self._message("deepseek", "assistant", response["response"])
                )
                session["model"] = response["model"]
                session["last_usage"] = response["usage"]
                totals = session.get("usage") if isinstance(session.get("usage"), dict) else {}
                totals["requests"] = int(totals.get("requests") or 0) + 1
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cache_hit_tokens",
                    "cache_miss_tokens",
                    "reasoning_tokens",
                ):
                    totals[field] = int(totals.get(field) or 0) + int(
                        response["usage"].get(field) or 0
                    )
                session["usage"] = totals
                session["active_generation"] = None
                session["updated_at"] = _utc_now()
                summary_updated = self._compact(session)
                self._write_unlocked(state)
        if discarded:
            self._log("deepseek_request_cancelled", session_id=session_id, late_response=True)
            return self._error_result(
                DeepSeekError("deepseek_cancelled", "Resposta tardia descartada"),
                session_id=session_id,
            )
        if summary_updated:
            self._log(
                "deepseek_summary_updated",
                session_id=session_id,
                source_until_message_id=session.get("summary_source_until_message_id"),
            )
        self._log(
            "deepseek_request_completed",
            session_id=session_id,
            model=response["model"],
            usage_counts={
                "input": response["usage"].get("input_tokens"),
                "output": response["usage"].get("output_tokens"),
                "total": response["usage"].get("total_tokens"),
                "cache_hit": response["usage"].get("cache_hit_tokens"),
                "reasoning": response["usage"].get("reasoning_tokens"),
            },
        )
        return {
            "ok": True,
            "session_id": session_id,
            "project": project,
            "model": response["model"],
            "response": response["response"],
            "usage": response["usage"],
            "error": None,
        }

    def _finish_generation(self, session_id: str, generation_id: str, *, cancelled: bool) -> None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(state, session_id)
            active = session.get("active_generation") if session else None
            if session and isinstance(active, dict) and active.get("id") == generation_id:
                session["active_generation"] = None
                if cancelled:
                    session["last_cancelled_generation_id"] = generation_id
                session["updated_at"] = _utc_now()
                self._write_unlocked(state)

    def cancel(self, *, project_path: str | None = None) -> dict[str, Any]:
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {"ok": False, "error": resolved.get("error", "project_not_found")}
        key = self._project_key(resolved["project"])
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(state, state["active_by_project"].get(key))
            active = session.get("active_generation") if session else None
            if not session or not isinstance(active, dict):
                return {"ok": False, "error": "deepseek_not_running"}
            generation_id = str(active["id"])
            active["cancelled"] = True
            session["last_cancelled_generation_id"] = generation_id
            session["updated_at"] = _utc_now()
            self._write_unlocked(state)
        event = self._cancel_events.get(generation_id)
        if event is not None:
            event.set()
        return {"ok": True, "session_id": session["session_id"], "error": None}

    @staticmethod
    def _turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                current = {
                    "requested": message.get("content"),
                    "source": message.get("source"),
                    "response": None,
                    "created_at": message.get("created_at"),
                }
                turns.append(current)
            elif message.get("role") == "assistant":
                if current is None or current.get("response") is not None:
                    current = {
                        "requested": None,
                        "source": None,
                        "response": message.get("content"),
                        "created_at": message.get("created_at"),
                    }
                    turns.append(current)
                else:
                    current["response"] = message.get("content")
        return turns

    def review_session(
        self, *, project_path: str | None = None, turn_limit: int | str = 10
    ) -> dict[str, Any]:
        try:
            limit = int(turn_limit)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_turn_limit"}
        if limit <= 0 or limit > 50:
            return {"ok": False, "error": "invalid_turn_limit"}
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {"ok": False, "error": resolved.get("error", "project_not_found")}
        project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(
                state, state["active_by_project"].get(self._project_key(project))
            )
        if session is None:
            return {"ok": False, "error": "deepseek_session_not_found"}
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        turns = self._turns(messages)
        selected = turns[-limit:]
        return {
            "ok": True,
            "session_id": session["session_id"],
            "project": project,
            "model": session.get("model"),
            "turns_available": len(turns),
            "turns_reviewed": len(selected),
            "summary": session.get("summary"),
            "summary_source": selected,
            "last_response": selected[-1].get("response") if selected else None,
            "error": None,
        }

    def status(self, *, project_path: str | None = None) -> dict[str, Any]:
        base = {
            "enabled": self.client.enabled,
            "configured": self.client.configured,
            "model": self.client.model or None,
            "active_session": None,
            "messages": 0,
            "state": (
                "disabled"
                if not self.client.enabled
                else "offline/config error"
                if not self.client.configured
                else "ready"
            ),
        }
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {**base, "error": resolved.get("error", "project_not_found")}
        project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(
                state, state["active_by_project"].get(self._project_key(project))
            )
        if session:
            base["active_session"] = str(session["session_id"])[:8]
            base["messages"] = len(session.get("messages") or [])
            base["summary"] = bool(session.get("summary"))
            base["context_messages"] = min(
                len(session.get("messages") or []), self.max_recent_turns * 2
            )
            base["usage"] = session.get("usage") or {}
            if session.get("active_generation"):
                base["state"] = "generating"
        return {**base, "project": project, "error": None}

    def ensure_session(self, *, project_path: str | None = None) -> dict[str, Any]:
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {"ok": False, "error": resolved.get("error", "project_not_found")}
        project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._get_or_create_session(
                state, project, continue_current_session=True
            )
            self._write_unlocked(state)
        self._log(
            "deepseek_session_opened",
            session_id=session["session_id"],
            project=project,
            message_count=len(session.get("messages") or []),
        )
        return self._session_result(session)

    def history_messages(
        self, *, project_path: str | None = None, limit: int = 40
    ) -> list[dict[str, Any]]:
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return []
        project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(
                state, state["active_by_project"].get(self._project_key(project))
            )
        messages = session.get("messages") if session and isinstance(session.get("messages"), list) else []
        return [dict(item) for item in messages[-max(1, min(limit, 200)) :] if isinstance(item, dict)]

    def list_sessions(self, *, project_path: str | None = None) -> list[dict[str, Any]]:
        project = None
        if project_path is not None:
            resolved = self._resolve_project(project_path)
            if not resolved.get("ok"):
                return []
            project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
        values = []
        for session in state["sessions"]:
            if project and self._project_key(str(session.get("project"))) != self._project_key(project):
                continue
            values.append(
                {
                    "session_id": session.get("session_id"),
                    "project": session.get("project"),
                    "model": session.get("model"),
                    "messages": len(session.get("messages") or []),
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                }
            )
        return sorted(values, key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def switch_session(self, session_id: str) -> dict[str, Any]:
        requested = session_id.strip().casefold()
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            matches = [
                session
                for session in state["sessions"]
                if str(session.get("session_id") or "").casefold().startswith(requested)
            ]
            if len(matches) != 1:
                return {
                    "ok": False,
                    "error": "deepseek_session_ambiguous"
                    if matches
                    else "deepseek_session_not_found",
                }
            session = matches[0]
            state["active_by_project"][self._project_key(str(session["project"]))] = session["session_id"]
            self._write_unlocked(state)
        self._log(
            "deepseek_session_switched",
            session_id=session["session_id"],
            project=session["project"],
        )
        return self._session_result(session)

    def preview_context(
        self,
        *,
        project_path: str | None = None,
        temporary_context: str | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_project(project_path)
        if not resolved.get("ok"):
            return {"ok": False, "error": resolved.get("error", "project_not_found")}
        project = resolved["project"]
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            session = self._find_session(
                state, state["active_by_project"].get(self._project_key(project))
            )
        if session is None:
            return {"ok": False, "error": "deepseek_session_not_found"}
        messages = self._api_messages(session, temporary_context=temporary_context)
        system_chars = len(messages[0]["content"])
        summary_chars = sum(
            len(item["content"])
            for item in messages[1:]
            if item["role"] == "system" and item["content"].startswith("Resumo")
        )
        temporary_chars = sum(
            len(item["content"])
            for item in messages[1:]
            if item["role"] == "system" and item["content"].startswith("Contexto temporario")
        )
        recent = [item for item in messages if item["role"] in {"user", "assistant"}]
        recent_chars = sum(len(item["content"]) for item in recent)
        total_chars = sum(len(item["content"]) for item in messages)
        return {
            "ok": True,
            "system_prompt": {"characters": system_chars, "estimated_tokens": _estimate_tokens(system_chars)},
            "rolling_summary": {"present": bool(summary_chars), "characters": summary_chars, "estimated_tokens": _estimate_tokens(summary_chars)},
            "recent_conversation": {"messages": len(recent), "characters": recent_chars, "estimated_tokens": _estimate_tokens(recent_chars)},
            "temporary_context": {"present": bool(temporary_chars), "characters": temporary_chars, "estimated_tokens": _estimate_tokens(temporary_chars)},
            "estimated_total_tokens": _estimate_tokens(total_chars),
            "estimate": True,
            "error": None,
        }

    @staticmethod
    def _session_result(session: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": session["session_id"],
            "project": session["project"],
            "model": session["model"],
            "messages": len(session.get("messages") or []),
            "error": None,
        }


def _estimate_tokens(characters: int) -> int:
    """Explicit approximation used only before the API returns authoritative usage."""
    return max(0, (int(characters) + 3) // 4)


class DeepSeekService:
    """UI-facing orchestration layer; HTTP and persistence remain in the manager."""

    def __init__(
        self,
        manager: DeepSeekSessionManager,
        *,
        project_path: str,
        codex: Any | None = None,
    ):
        self.manager = manager
        self.project_path = str(Path(project_path).resolve())
        self.codex = codex
        self.temporary_contexts: list[dict[str, Any]] = []
        self.state = "Ready"

    def open(self) -> dict[str, Any]:
        result = self.manager.ensure_session(project_path=self.project_path)
        if not self.manager.client.enabled or not self.manager.client.configured:
            self.state = "Offline/config error"
        else:
            self.state = "Ready"
        return result

    def send(
        self,
        text: str,
        *,
        source: str = "human",
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        pending = [item for item in self.temporary_contexts if not item.get("consumed")]
        context = "\n\n".join(str(item["content"]) for item in pending) or None
        self.state = "Generating"
        result = self.manager.delegate(
            text,
            project_path=self.project_path,
            source=source,
            context=context,
            on_delta=on_delta,
        )
        if result.get("ok"):
            for item in pending:
                item["consumed"] = True
            self.state = "Ready"
        elif result.get("error") == "deepseek_cancelled":
            self.state = "Cancelled"
        elif result.get("error") == "deepseek_rate_limited":
            self.state = "Rate limited"
        elif result.get("error") in {"deepseek_disabled", "deepseek_api_key_missing"}:
            self.state = "Offline/config error"
        else:
            self.state = "Error"
        return result

    def cancel(self) -> dict[str, Any]:
        self.state = "Cancelling"
        return self.manager.cancel(project_path=self.project_path)

    def attach_codex(self, turn_limit: int = 3) -> dict[str, Any]:
        if self.codex is None:
            return {"ok": False, "error": "codex_session_unavailable"}
        result = self.codex.review_session(
            project_path=self.project_path,
            turn_limit=max(1, min(int(turn_limit), 20)),
        )
        if not result.get("ok"):
            return result
        source = result.get("summary_source") if isinstance(result.get("summary_source"), list) else []
        lines = ["Codex recent activity:"]
        for turn in source:
            if not isinstance(turn, dict):
                continue
            requested = str(turn.get("requested") or "")[:600]
            final = str(turn.get("final_response") or turn.get("result") or "")[:900]
            status = str(turn.get("status") or "unknown")
            lines.append(f"- requested: {requested}\n  result: {final}\n  status: {status}")
        content = "\n".join(lines)[:20_000]
        attachment = {
            "id": str(uuid.uuid4()),
            "source": "codex_context",
            "label": f"Codex: last {len(source)} turns",
            "turns": len(source),
            "content": content,
            "estimated_tokens": _estimate_tokens(len(content)),
            "consumed": False,
        }
        self.temporary_contexts.append(attachment)
        self.manager._log(
            "deepseek_context_attached",
            project=self.project_path,
            source="codex_context",
            turns=len(source),
            estimated_tokens=attachment["estimated_tokens"],
        )
        return {"ok": True, **{key: value for key, value in attachment.items() if key != "content"}, "error": None}

    def clear_context(self) -> dict[str, Any]:
        removed = len([item for item in self.temporary_contexts if not item.get("consumed")])
        self.temporary_contexts = []
        return {"ok": True, "removed": removed, "error": None}

    def context_report(self) -> dict[str, Any]:
        pending = [item for item in self.temporary_contexts if not item.get("consumed")]
        context = "\n\n".join(str(item["content"]) for item in pending) or None
        result = self.manager.preview_context(
            project_path=self.project_path,
            temporary_context=context,
        )
        result["attachments"] = [
            {key: value for key, value in item.items() if key != "content"}
            for item in pending
        ]
        return result

    def switch_project(self, query: str) -> dict[str, Any]:
        candidate = Path(query)
        result = self.manager.projects.resolve(
            query=query,
            path_hint=query if candidate.is_absolute() else None,
        )
        if not result.get("ok"):
            return result
        self.project_path = str(Path(result["root"]).resolve())
        self.temporary_contexts = []
        opened = self.open()
        self.manager._log(
            "deepseek_session_switched",
            session_id=opened.get("session_id"),
            project=self.project_path,
        )
        return opened

    def qwen_handoff_prompt(self) -> dict[str, Any]:
        messages = self.manager.history_messages(project_path=self.project_path, limit=10)
        assistant_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("source") == "deepseek"),
            None,
        )
        if assistant_index is None:
            return {"ok": False, "error": "deepseek_recommendation_not_found"}
        recommendation = str(messages[assistant_index].get("content") or "")
        question = next(
            (
                str(messages[index].get("content") or "")
                for index in range(assistant_index - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            "",
        )
        return {
            "ok": True,
            "prompt": (
                "Considere a recomendacao consultiva do DeepSeek abaixo. Avalie-a e, "
                "somente se apropriado ao pedido, use delegate_to_codex.\n\n"
                f"Pergunta relacionada: {question[:2000]}\n\n"
                f"Recomendacao DeepSeek: {recommendation[:6000]}"
            ),
            "error": None,
        }
