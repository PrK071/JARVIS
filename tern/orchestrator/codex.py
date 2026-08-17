from __future__ import annotations

import json
import heapq
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .prompt import CODEX_TASK_SCHEMA
from .schema import validate
from .security import PathPolicy
from .codex_state import SharedCodexState, utc_now
from .codex_jobs import ACTIVE_JOB_STATES, TERMINAL_JOB_STATES, CodexJobStore
from .codex_sessions import (
    CodexSessionRegistry,
    CodexSessionResolution,
    CodexSessionResolver,
    normalize_project_path,
)


def _sandbox_policy(project: Any, *, read_only: bool) -> dict[str, Any]:
    """Structural execution envelope for one turn.

    A read-only turn gets the read-only sandbox, so an authorized read-only
    request cannot gain write capability downstream. This is enforcement by the
    sandbox, not by instructions inside the task text.
    """

    if read_only:
        return {"type": "readOnly"}
    return {
        "type": "workspaceWrite",
        "writableRoots": [str(project)],
        "networkAccess": False,
    }


class CodexError(RuntimeError):
    def __init__(self, message: str, *, layer: str = "bridge"):
        super().__init__(message)
        self.layer = layer


class CodexProtocolError(CodexError):
    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(message, layer="protocol")
        self.code = code


@dataclass(frozen=True)
class CodexResult:
    accepted: bool
    thread_id: str | None
    turn_id: str | None
    status: str
    final_response: str
    error: str | None
    events: int = 0
    human_interventions: tuple[dict[str, Any], ...] = ()
    state_events: tuple[dict[str, Any], ...] = ()
    result_discarded: bool = False
    job_id: str | None = None
    wait_timed_out: bool = False
    session_resolution: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return (
            self.accepted
            and self.status in {"queued", "starting", "running", "completed"}
            and not self.error
        )

    @property
    def session_id(self) -> str | None:
        return self.thread_id

    @property
    def message(self) -> str:
        if self.final_response or self.error:
            return self.final_response or self.error or ""
        if self.status in {"queued", "starting", "running"}:
            return "Tarefa iniciada no Codex."
        return ""

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "final_response": self.final_response,
            "error": self.error,
            "events": self.events,
            "human_interventions": list(self.human_interventions),
            "state_events": list(self.state_events),
            "result_discarded": self.result_discarded,
            "job_id": self.job_id,
            "wait_timed_out": self.wait_timed_out,
            "session_resolution": self.session_resolution,
            # Compatibility for existing Jarvis callers.
            "ok": self.ok,
            "session_id": self.thread_id,
            "message": self.message,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class TurnSnapshot:
    """Stable subset of one App Server turn/read turn."""

    turn_id: str
    status: str | None
    messages: list[dict[str, Any]]
    items: list[dict[str, Any]]
    started_at: int | None
    completed_at: int | None
    duration_ms: int | None
    error: Any = None


@dataclass(frozen=True)
class ThreadSnapshot:
    """Stable subset of an App Server thread/read response."""

    thread_id: str
    status: str | None
    turns: list[TurnSnapshot]
    created_at: int | None
    updated_at: int | None
    cli_version: str | None


class InvalidThreadResponse(ValueError):
    pass


def _optional_counter(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _messages_from_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "userMessage":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                    continue
                messages.append(
                    {
                        "role": "user",
                        "text": part["text"],
                        "item_id": item.get("id"),
                        "client_id": item.get("clientId"),
                    }
                )
        elif item_type == "agentMessage" and isinstance(item.get("text"), str):
            messages.append(
                {
                    "role": "assistant",
                    "text": item["text"],
                    "item_id": item.get("id"),
                    "phase": item.get("phase"),
                }
            )
    return messages


def normalize_thread_read(response: Any) -> ThreadSnapshot:
    """Normalize the real App Server thread/read envelope without coercing types."""
    if not isinstance(response, dict):
        raise InvalidThreadResponse("thread/read response must be an object")
    thread = response.get("thread")
    if not isinstance(thread, dict):
        raise InvalidThreadResponse("thread/read response.thread must be an object")
    thread_id = thread.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        raise InvalidThreadResponse("thread/read response.thread.id must be a string")

    raw_turns = thread.get("turns", [])
    if raw_turns is None:
        raw_turns = []
    if not isinstance(raw_turns, list):
        raise InvalidThreadResponse("thread/read response.thread.turns must be a list")

    turns: list[TurnSnapshot] = []
    for raw_turn in raw_turns:
        if not isinstance(raw_turn, dict):
            continue
        raw_items = raw_turn.get("items", [])
        if raw_items is None:
            raw_items = []
        if not isinstance(raw_items, list):
            raise InvalidThreadResponse("thread/read turn.items must be a list")
        items = [dict(item) for item in raw_items if isinstance(item, dict)]

        raw_messages = raw_turn.get("messages")
        if raw_messages is None:
            messages = _messages_from_items(items)
        elif isinstance(raw_messages, list):
            messages = [dict(item) for item in raw_messages if isinstance(item, dict)]
            messages.extend(_messages_from_items(items))
        else:
            raise InvalidThreadResponse("thread/read turn.messages must be a list")

        status = raw_turn.get("status")
        turns.append(
            TurnSnapshot(
                turn_id=str(raw_turn.get("id") or ""),
                status=status if isinstance(status, str) else None,
                messages=messages,
                items=items,
                started_at=_optional_counter(raw_turn.get("startedAt")),
                completed_at=_optional_counter(raw_turn.get("completedAt")),
                duration_ms=_optional_counter(raw_turn.get("durationMs")),
                error=raw_turn.get("error"),
            )
        )

    raw_status = thread.get("status")
    if isinstance(raw_status, str):
        status = raw_status
    elif isinstance(raw_status, dict) and isinstance(raw_status.get("type"), str):
        status = raw_status["type"]
    else:
        status = None
    cli_version = thread.get("cliVersion")
    return ThreadSnapshot(
        thread_id=thread_id,
        status=status,
        turns=turns,
        created_at=_optional_counter(thread.get("createdAt")),
        updated_at=_optional_counter(thread.get("updatedAt")),
        cli_version=cli_version if isinstance(cli_version, str) else None,
    )


def _utc_now() -> str:
    return utc_now()


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|passwd|secret|token|credential)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._~+/-]{12,})"
)


def _redact(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("<redacted>", value)
    return value


def _safe_summary(value: str, limit: int = 500) -> str:
    normalized = " ".join(value.split())
    redacted = _redact(normalized)
    return str(redacted)[:limit]


class JsonlEventLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: str, **values: Any) -> None:
        record = {"time": _utc_now(), "event": event, **_redact(values)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class CodexAppServerClient:
    """Small synchronous client for Codex App Server JSON-RPC over WebSocket."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: int = 1800,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.event_callback = event_callback
        self._socket: Any = None
        self._next_id = 0

    @property
    def connected(self) -> bool:
        return self._socket is not None and bool(getattr(self._socket, "connected", True))

    def connect(self) -> None:
        if self.connected:
            return
        try:
            import websocket
        except ImportError as exc:
            raise CodexError(
                "dependencia websocket-client ausente",
                layer="websocket",
            ) from exc
        try:
            self._socket = websocket.create_connection(
                self.endpoint,
                timeout=min(self.timeout, 30),
                enable_multithread=True,
                http_proxy_host=None,
                suppress_origin=True,
            )
            self._socket.settimeout(self.timeout)
            self.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "tern_codex_bridge",
                        "title": "Tern Codex Bridge",
                        "version": "0.2.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            self.notify("initialized", {})
        except CodexError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise CodexError(
                f"falha ao conectar ao App Server: {exc}",
                layer="websocket",
            ) from exc

    def close(self) -> None:
        socket = self._socket
        self._socket = None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._socket is None and method != "initialize":
            self.connect()
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"method": method, "id": request_id}
        if params is not None:
            message["params"] = params
        self._send(message)
        while True:
            incoming = self.receive()
            if incoming.get("id") != request_id:
                continue
            error = incoming.get("error")
            if isinstance(error, dict):
                raise CodexProtocolError(
                    str(error.get("message") or error),
                    code=error.get("code"),
                )
            result = incoming.get("result")
            return result if isinstance(result, dict) else {}

    def receive(self) -> dict[str, Any]:
        if self._socket is None:
            raise CodexError("conexao App Server fechada", layer="websocket")
        try:
            raw = self._socket.recv()
            incoming = json.loads(raw)
        except Exception as exc:
            raise CodexError(
                f"falha ao ler evento App Server: {exc}",
                layer="events",
            ) from exc
        if not isinstance(incoming, dict):
            raise CodexProtocolError("mensagem App Server nao e objeto JSON")
        if "method" in incoming and "id" in incoming:
            # Approval policy is `never`; unexpected server requests must not hang.
            self._send(
                {
                    "id": incoming["id"],
                    "error": {
                        "code": -32601,
                        "message": "server request unsupported by Tern bridge",
                    },
                }
            )
        if self.event_callback is not None and "method" in incoming:
            self.event_callback(incoming)
        return incoming

    def _send(self, message: dict[str, Any]) -> None:
        if self._socket is None:
            raise CodexError("conexao App Server fechada", layer="websocket")
        try:
            self._socket.send(json.dumps(message, ensure_ascii=False))
        except Exception as exc:
            raise CodexError(
                f"falha ao enviar ao App Server: {exc}",
                layer="send",
            ) from exc


class CodexSessionManager:
    def __init__(
        self,
        project: Path,
        *,
        endpoint: str = "ws://127.0.0.1:4500",
        timeout: int = 1800,
        executable: str | None = None,
        state_dir: Path | None = None,
        preferred_thread_id: str | None = None,
    ):
        self.project = project.resolve(strict=True)
        self.endpoint = endpoint
        self.timeout = timeout
        self.executable = executable or shutil.which("codex") or "codex"
        self.state_dir = (state_dir or self.project / ".orchestrator").resolve()
        self.preferred_thread_id = (
            str(preferred_thread_id).strip() if preferred_thread_id else None
        )
        self.session_path = self.state_dir / "codex-session.json"
        self.server_path = self.state_dir / "codex-app-server.json"
        self.server_log_path = self.state_dir / "codex-app-server.log"
        self.events_path = self.state_dir / "codex-events.jsonl"
        self.bridge_log = JsonlEventLog(self.state_dir / "codex-bridge.jsonl")
        self.runtime = SharedCodexState(
            self.state_dir,
            self.project,
            self.endpoint,
        )
        self._client: CodexAppServerClient | None = None
        self._queue_condition = threading.Condition()
        self._queue: list[tuple[int, int, object]] = []
        self._queue_sequence = 0
        self._queue_active = False
        self._state_lock = threading.Lock()
        self._active_thread_id: str | None = None
        self._active_turn_id: str | None = None
        self._known_thread_id: str | None = None
        self._thread_created = False
        self._cancelled_turns: set[str] = set()
        self._event_count = 0
        self._final_messages: list[tuple[str | None, str]] = []
        self._completed_turns: dict[str, dict[str, Any]] = {}
        self._client_message_sources: dict[str, str] = {}
        self._seen_user_messages: set[str] = set()
        self._event_observer: Callable[[str, dict[str, Any]], None] | None = None
        self._validate_endpoint()

    @property
    def active_turn_id(self) -> str | None:
        with self._state_lock:
            if self._active_turn_id:
                return self._active_turn_id
        state = self.runtime.read()
        if state.get("state") in {"running", "steering", "cancelling"}:
            value = state.get("turn_id")
            return value if isinstance(value, str) else None
        return None

    def _validate_endpoint(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"ws", "wss"}:
            raise CodexError("endpoint Codex deve usar ws:// ou wss://", layer="endpoint")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise CodexError("App Server compartilhado deve permanecer local", layer="endpoint")
        if parsed.port is None:
            raise CodexError("endpoint Codex sem porta", layer="endpoint")

    @classmethod
    def from_persisted_session(
        cls,
        state_dir: Path,
        *,
        timeout: int = 1800,
        executable: str | None = None,
    ) -> CodexSessionManager:
        state_dir = state_dir.resolve()
        session = cls._read_json(state_dir / "codex-session.json")
        if not session:
            raise CodexError(
                "sessao compartilhada nao encontrada; "
                "use codex-shared-start para recuperar",
                layer="shared_session_not_found",
            )
        project = session.get("project")
        endpoint = session.get("server_endpoint")
        thread_id = session.get("thread_id")
        if (
            not isinstance(project, str)
            or not isinstance(endpoint, str)
            or not isinstance(thread_id, str)
            or not thread_id
        ):
            raise CodexError(
                "codex-session.json nao contem project, endpoint e thread_id validos",
                layer="invalid_shared_session",
            )
        return cls(
            Path(project),
            endpoint=endpoint,
            timeout=timeout,
            executable=executable,
            state_dir=state_dir,
        )

    def readiness_url(self) -> str:
        parsed = urlparse(self.endpoint)
        scheme = "https" if parsed.scheme == "wss" else "http"
        host = f"[{parsed.hostname}]" if ":" in str(parsed.hostname) else parsed.hostname
        return f"{scheme}://{host}:{parsed.port}/readyz"

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.readiness_url(), timeout=1) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def start_server(self, wait_seconds: int = 30) -> dict[str, Any]:
        if self.is_ready():
            self.bridge_log.write("server_reused", endpoint=self.endpoint)
            return {"started": False, "ready": True, "endpoint": self.endpoint}
        executable_path = shutil.which(self.executable) or self.executable
        if shutil.which(executable_path) is None and not Path(executable_path).is_file():
            raise CodexError("executavel Codex nao encontrado", layer="executable")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        log_handle = self.server_log_path.open("a", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                [executable_path, "app-server", "--listen", self.endpoint],
                cwd=str(self.project),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                shell=False,
                close_fds=True,
            )
        except OSError as exc:
            log_handle.close()
            raise CodexError(f"falha ao iniciar App Server: {exc}", layer="server_start") from exc
        log_handle.close()
        self._write_json(
            self.server_path,
            {
                "pid": process.pid,
                "endpoint": self.endpoint,
                "project": str(self.project),
                "started_at": _utc_now(),
            },
        )
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = self._tail(self.server_log_path, 4096)
                raise CodexError(
                    f"App Server encerrou com codigo {process.returncode}: {tail}",
                    layer="server_start",
                )
            if self.is_ready():
                self.bridge_log.write(
                    "server_started",
                    endpoint=self.endpoint,
                    pid=process.pid,
                )
                return {
                    "started": True,
                    "ready": True,
                    "endpoint": self.endpoint,
                    "pid": process.pid,
                }
            time.sleep(0.1)
        raise CodexError(
            f"App Server nao ficou pronto em {wait_seconds}s",
            layer="readiness",
        )

    def stop_server(self) -> dict[str, Any]:
        state = self._read_json(self.server_path)
        pid = state.get("pid") if isinstance(state, dict) else None
        if not isinstance(pid, int):
            return {"stopped": False, "reason": "processo gerenciado nao encontrado"}
        if state.get("endpoint") != self.endpoint:
            raise CodexError("estado do servidor pertence a outro endpoint", layer="shutdown")
        try:
            if os.name == "nt":
                command = (
                    f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; "
                    "if($null -eq $p){exit 3}; "
                    "if($p.CommandLine -notmatch 'app-server'){exit 4}; "
                    f"Stop-Process -Id {pid} -Force"
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode not in {0, 3}:
                    raise CodexError(
                        f"recusa ao encerrar PID {pid}: verificacao do processo falhou",
                        layer="shutdown",
                    )
            else:
                os.kill(pid, signal.SIGTERM)
        finally:
            self.close()
        self.bridge_log.write("server_stopped", endpoint=self.endpoint, pid=pid)
        return {"stopped": True, "pid": pid, "endpoint": self.endpoint}

    def connect(self) -> CodexAppServerClient:
        self.start_server()
        if self._client is not None and self._client.connected:
            return self._client
        self._client = CodexAppServerClient(
            self.endpoint,
            timeout=self.timeout,
            event_callback=self._on_event,
        )
        self._client.connect()
        self.runtime.update(
            bridge_connected=True,
            bridge_pid=os.getpid(),
            last_event_at=_utc_now(),
        )
        self.bridge_log.write("protocol_initialized", endpoint=self.endpoint)
        return self._client

    def reconnect(self) -> CodexAppServerClient:
        self.close()
        client = self.connect()
        state = self._load_session()
        thread_id = state.get("thread_id") if state else None
        if isinstance(thread_id, str):
            client.request(
                "thread/resume",
                {"threadId": thread_id, "cwd": str(self.project)},
            )
        self.bridge_log.write("reconnected", endpoint=self.endpoint, thread_id=thread_id)
        return client

    @staticmethod
    def _provider_session_record(
        thread: dict[str, Any],
        *,
        project: Path,
        endpoint: str,
        visible_thread_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        thread_id = str(thread.get("id") or "").strip()
        status = thread.get("status")
        state = (
            str(status.get("type") or "unknown")
            if isinstance(status, dict)
            else "unknown"
        )
        source = str(thread.get("source") or "unknown")
        # appServer threads are surfaced by the Jarvis UI through the canonical
        # project binding; cli/vscode threads are also visible in Codex history.
        visible = source in {"cli", "vscode", "appServer"} or (
            visible_thread_ids is not None and thread_id in visible_thread_ids
        )
        return {
            "thread_id": thread_id,
            "session_id": str(thread.get("sessionId") or thread_id),
            "project": str(project),
            "project_key": normalize_project_path(project),
            "endpoint": endpoint,
            "state": state,
            "source": source,
            "visible": visible,
            "recoverable": bool(thread_id) and not bool(thread.get("ephemeral")),
            "ephemeral": bool(thread.get("ephemeral")),
            "created_at": thread.get("createdAt"),
            "updated_at": thread.get("updatedAt"),
            "name": thread.get("name"),
        }

    def list_project_threads(self) -> list[dict[str, Any]]:
        """Discover provider-native, interactive threads for this exact project."""
        client = self.connect()
        cursor: str | None = None
        values: list[dict[str, Any]] = []
        visible_values = self._known_tui_thread_ids()
        visible_thread_ids = (
            set(visible_values) if isinstance(visible_values, list) else None
        )
        while True:
            params: dict[str, Any] = {
                "limit": 100,
                "sortKey": "updated_at",
                "sortDirection": "desc",
                "sourceKinds": ["cli", "vscode", "appServer"],
                "archived": False,
                "cwd": str(self.project),
            }
            if cursor:
                params["cursor"] = cursor
            result = client.request("thread/list", params)
            data = result.get("data")
            if not isinstance(data, list):
                raise CodexProtocolError("thread/list nao retornou data")
            for thread in data:
                if not isinstance(thread, dict):
                    continue
                cwd = thread.get("cwd")
                if isinstance(cwd, str) and normalize_project_path(cwd) != normalize_project_path(
                    self.project
                ):
                    continue
                record = self._provider_session_record(
                    thread,
                    project=self.project,
                    endpoint=self.endpoint,
                    visible_thread_ids=visible_thread_ids,
                )
                if record["thread_id"]:
                    values.append(record)
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        return values

    def read_session_record(
        self,
        thread_id: str,
        *,
        require_project_match: bool = True,
    ) -> dict[str, Any]:
        client = self.connect()
        result = client.request(
            "thread/read",
            {"threadId": thread_id, "includeTurns": False},
        )
        thread = result.get("thread")
        if not isinstance(thread, dict) or thread.get("id") != thread_id:
            raise CodexProtocolError("thread/read nao retornou a thread solicitada")
        cwd = thread.get("cwd")
        if (
            require_project_match
            and isinstance(cwd, str)
            and normalize_project_path(cwd) != normalize_project_path(self.project)
        ):
            raise CodexError(
                "thread Codex pertence a outro project_path",
                layer="cross_project_session",
            )
        visible_values = self._known_tui_thread_ids()
        return self._provider_session_record(
            thread,
            project=self.project,
            endpoint=self.endpoint,
            visible_thread_ids=(
                set(visible_values) if isinstance(visible_values, list) else None
            ),
        )

    def adopt_thread(self, thread_id: str) -> dict[str, Any]:
        record = self.read_session_record(thread_id)
        client = self.connect()
        result = client.request(
            "thread/resume",
            {"threadId": thread_id, "cwd": str(self.project)},
        )
        thread = result.get("thread")
        if not isinstance(thread, dict) or thread.get("id") != thread_id:
            raise CodexProtocolError("thread/resume nao retomou a thread solicitada")
        self._known_thread_id = thread_id
        self._thread_created = False
        self._persist_session(thread_id)
        self.runtime.update(thread_id=thread_id)
        return record

    def create_thread(self) -> dict[str, Any]:
        client = self.connect()
        result = client.request(
            "thread/start",
            {
                "cwd": str(self.project),
                "approvalPolicy": "never",
                "sandbox": "workspace-write",
                "ephemeral": False,
                "serviceName": "jarvis",
            },
        )
        thread = result.get("thread")
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexProtocolError("thread/start nao retornou thread.id")
        self._known_thread_id = thread_id
        self._thread_created = True
        self._persist_session(thread_id)
        self.runtime.update(thread_id=thread_id, turn_id=None, state="idle")
        record = self.read_session_record(thread_id)
        self.bridge_log.write(
            "thread_started",
            project=str(self.project),
            thread_id=thread_id,
        )
        return record

    def ensure_thread(
        self,
        *,
        continue_current_thread: bool = True,
        target_thread_id: str | None = None,
    ) -> str:
        client = self.connect()
        if target_thread_id:
            if self._known_thread_id != target_thread_id:
                self.adopt_thread(target_thread_id)
            return target_thread_id
        preferred_thread_id = (
            self.preferred_thread_id if continue_current_thread else None
        )
        if preferred_thread_id and self._known_thread_id == preferred_thread_id:
            return preferred_thread_id
        if preferred_thread_id:
            previous = self._load_session() or {}
            previous_thread_id = previous.get("thread_id")
            try:
                read_result = client.request(
                    "thread/read",
                    {"threadId": preferred_thread_id, "includeTurns": False},
                )
                read_thread = read_result.get("thread")
                if (
                    not isinstance(read_thread, dict)
                    or read_thread.get("id") != preferred_thread_id
                ):
                    raise CodexProtocolError(
                        "thread/read nao retornou a thread visivel solicitada"
                    )
                resume_result = client.request(
                    "thread/resume",
                    {"threadId": preferred_thread_id, "cwd": str(self.project)},
                )
                resumed_thread = resume_result.get("thread")
                if (
                    not isinstance(resumed_thread, dict)
                    or resumed_thread.get("id") != preferred_thread_id
                ):
                    raise CodexProtocolError(
                        "thread/resume nao retomou a thread visivel solicitada"
                    )
            except CodexError as exc:
                # Remember the user-visible target for the next startup even
                # when its current standalone owner still holds the writer lock.
                self._persist_session(preferred_thread_id)
                active_writer = "active writer" in str(exc).casefold()
                failure_event = (
                    "preferred_thread_active_writer"
                    if active_writer
                    else "preferred_thread_unavailable"
                )
                self.bridge_log.write(
                    failure_event,
                    project=str(self.project),
                    thread_id=preferred_thread_id,
                    error=str(exc),
                )
                if active_writer:
                    raise CodexError(
                        "a sessao atual do Codex ainda esta aberta em modo "
                        "standalone. Feche esta tela e execute `jarvis codex` "
                        "para reabri-la no modo compartilhado; depois envie a "
                        "tarefa novamente. Nenhuma sessao alternativa foi criada",
                        layer="preferred_thread_active_writer",
                    ) from exc
                raise CodexError(
                    "nao foi possivel acessar a sessao atual do Codex; "
                    "nenhuma sessao alternativa foi criada",
                    layer="preferred_thread_unavailable",
                ) from exc
            self._known_thread_id = preferred_thread_id
            self._thread_created = False
            self._persist_session(preferred_thread_id)
            self.runtime.update(thread_id=preferred_thread_id)
            self.bridge_log.write(
                "preferred_thread_adopted",
                project=str(self.project),
                previous_thread_id=previous_thread_id,
                thread_id=preferred_thread_id,
            )
            return preferred_thread_id
        if continue_current_thread and self._known_thread_id:
            return self._known_thread_id
        state = self._load_session() if continue_current_thread else None
        thread_id = state.get("thread_id") if state else None
        if isinstance(thread_id, str) and thread_id:
            try:
                client.request("thread/read", {"threadId": thread_id, "includeTurns": False})
                client.request(
                    "thread/resume",
                    {"threadId": thread_id, "cwd": str(self.project)},
                )
                self.bridge_log.write(
                    "thread_resumed",
                    project=str(self.project),
                    thread_id=thread_id,
                )
                self._known_thread_id = thread_id
                self._thread_created = False
                self._persist_session(thread_id)
                self.runtime.update(thread_id=thread_id)
                return thread_id
            except CodexProtocolError as exc:
                self.bridge_log.write(
                    "thread_invalid",
                    project=str(self.project),
                    thread_id=thread_id,
                    error=str(exc),
                )
        previous_thread_id = thread_id if isinstance(thread_id, str) else None
        result = client.request(
            "thread/start",
            {
                "cwd": str(self.project),
                "approvalPolicy": "never",
                "sandbox": "workspace-write",
                "ephemeral": False,
            },
        )
        thread = result.get("thread")
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexProtocolError("thread/start nao retornou thread.id")
        self._persist_session(thread_id)
        self.runtime.update(thread_id=thread_id, turn_id=None, state="idle")
        self._known_thread_id = thread_id
        self._thread_created = True
        self.bridge_log.write(
            "thread_started",
            project=str(self.project),
            thread_id=thread_id,
        )
        if previous_thread_id and previous_thread_id != thread_id:
            self.bridge_log.write(
                "thread_replaced",
                previous_thread_id=previous_thread_id,
                reason="thread/read failed; stored thread unavailable",
                new_thread_id=thread_id,
                time=_utc_now(),
            )
        return thread_id

    def run_turn(
        self,
        task: str,
        *,
        origin: str = "qwen",
        continue_current_thread: bool = True,
        target_thread_id: str | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        read_only: bool = False,
    ) -> CodexResult:
        if origin not in {"human", "qwen", "system"}:
            raise CodexError("origem deve ser human, qwen ou system", layer="queue")
        if not task.strip():
            raise CodexError("task obrigatoria", layer="validation")
        queued_epoch = int(self.runtime.read().get("queue_epoch") or 0)
        ticket = self._queue_acquire(origin)
        try:
            with self.runtime.turn_mutex(timeout=self.timeout):
                thread_id: str | None = None
                turn_id: str | None = None
                self._event_count = 0
                self._final_messages = []
                self._event_observer = event_callback
                try:
                    shared = self.runtime.read()
                    if int(shared.get("queue_epoch") or 0) != queued_epoch:
                        return CodexResult(
                            accepted=False,
                            thread_id=shared.get("thread_id"),
                            turn_id=None,
                            status="cancelled",
                            final_response="",
                            error="queue_cleared",
                            result_discarded=True,
                        )
                    ensure_arguments: dict[str, Any] = {
                        "continue_current_thread": continue_current_thread
                    }
                    if target_thread_id is not None:
                        ensure_arguments["target_thread_id"] = target_thread_id
                    thread_id = self.ensure_thread(**ensure_arguments)
                    existing_turn_id = (
                        None
                        if self._thread_created
                        else self._server_active_turn(thread_id)
                    )
                    if existing_turn_id:
                        raise CodexError(
                            f"thread ja possui turn ativo {existing_turn_id}",
                            layer="queue",
                        )
                    client = self.connect()
                    client_message_id = f"tern-{origin}-{uuid.uuid4()}"
                    self._client_message_sources[client_message_id] = origin
                    self.runtime.update(
                        thread_id=thread_id,
                        turn_id=None,
                        state="starting",
                        last_instruction_source=origin,
                        bridge_connected=True,
                        bridge_pid=os.getpid(),
                        qwen_connected=origin == "qwen",
                        qwen_pid=os.getpid() if origin == "qwen" else None,
                        active_client_message_id=client_message_id,
                        result_discarded=False,
                    )
                    self.bridge_log.write(
                        "message_queued",
                        source=origin,
                        operation="turn/start",
                        message_summary=_safe_summary(task),
                        state="starting",
                        project=str(self.project),
                        thread_id=thread_id,
                        turn_id=None,
                        client_message_id=client_message_id,
                    )
                    response = client.request(
                        "turn/start",
                        {
                            "threadId": thread_id,
                            "input": [{"type": "text", "text": task.strip()}],
                            "clientUserMessageId": client_message_id,
                            "cwd": str(self.project),
                            "approvalPolicy": "never",
                            "sandboxPolicy": _sandbox_policy(
                                self.project,
                                read_only=read_only,
                            ),
                        },
                    )
                    turn = response.get("turn")
                    turn_id = turn.get("id") if isinstance(turn, dict) else None
                    if not isinstance(turn_id, str) or not turn_id:
                        raise CodexProtocolError("turn/start nao retornou turn.id")
                    with self._state_lock:
                        self._active_thread_id = thread_id
                        self._active_turn_id = turn_id
                    self.runtime.update(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        state="running",
                        last_instruction_source=origin,
                    )
                    self.runtime.append_state_event(
                        "turn started",
                        source=origin,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        state_result="running",
                        summary=_safe_summary(task),
                    )
                    self.bridge_log.write(
                        "turn_started",
                        source=origin,
                        operation="turn/start",
                        message_summary=_safe_summary(task),
                        state="running",
                        project=str(self.project),
                        thread_id=thread_id,
                        turn_id=turn_id,
                        client_message_id=client_message_id,
                    )
                    if event_callback is not None:
                        event_callback(
                            "codex_turn_started",
                            {"thread_id": thread_id, "turn_id": turn_id},
                        )
                        event_callback(
                            "codex_working",
                            {"thread_id": thread_id, "turn_id": turn_id},
                        )
                    completed = self._wait_for_completion(thread_id, turn_id)
                    turn_value = completed.get("turn")
                    status = (
                        str(turn_value.get("status"))
                        if isinstance(turn_value, dict)
                        else "failed"
                    )
                    error_value = (
                        turn_value.get("error")
                        if isinstance(turn_value, dict)
                        else None
                    )
                    error = (
                        None
                        if error_value is None or error_value == ""
                        else str(error_value)
                    )
                    final = self._select_final_message(turn_id)
                    shared = self.runtime.read()
                    cancelled = (
                        turn_id in self._cancelled_turns
                        or turn_id in (shared.get("cancelled_turn_ids") or [])
                        or status == "interrupted"
                    )
                    result_discarded = False
                    if cancelled:
                        final = ""
                        error = "cancelled"
                        status = "interrupted"
                        result_discarded = True
                        for name in (
                            "turn interrupted",
                            "result discarded",
                            "session ready",
                        ):
                            self.runtime.append_state_event(
                                name,
                                source="system",
                                thread_id=thread_id,
                                turn_id=turn_id,
                                state_result=(
                                    "idle" if name == "session ready" else status
                                ),
                            )
                        self.runtime.update(
                            state="idle",
                            last_terminal_state="interrupted",
                            turn_id=None,
                            queue_length=0,
                            qwen_connected=False,
                            qwen_pid=None,
                            result_discarded=True,
                        )
                    else:
                        self.runtime.update(
                            state="completed" if status == "completed" else "failed",
                            last_terminal_state=status,
                            turn_id=None,
                            qwen_connected=False,
                            qwen_pid=None,
                            result_discarded=False,
                        )
                    interventions = tuple(self.runtime.interventions_for(turn_id))
                    state_events = tuple(self.runtime.state_events_for(turn_id))
                    self.bridge_log.write(
                        "turn_completed",
                        source=origin,
                        operation="turn/completed",
                        state=status,
                        project=str(self.project),
                        thread_id=thread_id,
                        turn_id=turn_id,
                        final_response=final,
                        interventions=len(interventions),
                        result_discarded=result_discarded,
                        error=error,
                        events=self._event_count,
                    )
                    self._thread_created = False
                    self._persist_session(thread_id)
                    if event_callback is not None:
                        event_callback(
                            (
                                "codex_completed"
                                if status == "completed" and not error
                                else "codex_failed"
                            ),
                            {
                                "thread_id": thread_id,
                                "turn_id": turn_id,
                                "status": status,
                                "error": error,
                            },
                        )
                    return CodexResult(
                        accepted=True,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        status=status,
                        final_response=final,
                        error=error,
                        events=self._event_count,
                        human_interventions=interventions,
                        state_events=state_events,
                        result_discarded=result_discarded,
                    )
                except CodexError as exc:
                    self.runtime.update(
                        state="failed",
                        last_terminal_state="failed",
                        turn_id=None,
                        qwen_connected=False,
                        qwen_pid=None,
                    )
                    self.bridge_log.write(
                        "turn_failed",
                        source=origin,
                        operation="turn/start" if turn_id is None else "turn/wait",
                        state="failed",
                        project=str(self.project),
                        thread_id=thread_id,
                        turn_id=turn_id,
                        layer=exc.layer,
                        error=str(exc),
                    )
                    if event_callback is not None:
                        event_callback(
                            "codex_failed",
                            {
                                "thread_id": thread_id,
                                "turn_id": turn_id,
                                "status": "failed",
                                "error": str(exc),
                            },
                        )
                    return CodexResult(
                        accepted=turn_id is not None,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        status="failed",
                        final_response="",
                        error=f"{exc.layer}: {exc}",
                        events=self._event_count,
                    )
                finally:
                    self._event_observer = None
                    with self._state_lock:
                        self._active_thread_id = None
                        self._active_turn_id = None
        finally:
            self._queue_release(ticket)

    def steer(self, task: str, *, origin: str = "human") -> dict[str, Any]:
        if origin != "human":
            raise CodexError("turn/steer externo exige origem human", layer="steer")
        if not task.strip():
            raise CodexError("instrucao de steer obrigatoria", layer="steer")
        with self.runtime.control_mutex():
            thread_id, turn_id = self._resolve_active_turn()
            client_message_id = f"tern-human-{uuid.uuid4()}"
            self.runtime.update(
                thread_id=thread_id,
                turn_id=turn_id,
                state="steering",
                last_instruction_source="human",
            )
            self.bridge_log.write(
                "message_queued",
                source="human",
                operation="turn/steer",
                message_summary=_safe_summary(task),
                thread_id=thread_id,
                turn_id=turn_id,
                state="steering",
                client_message_id=client_message_id,
            )
            client = CodexAppServerClient(self.endpoint, timeout=self.timeout)
            try:
                client.connect()
                result = client.request(
                    "turn/steer",
                    {
                        "threadId": thread_id,
                        "expectedTurnId": turn_id,
                        "input": [{"type": "text", "text": task.strip()}],
                        "clientUserMessageId": client_message_id,
                    },
                )
            except CodexError as exc:
                self.runtime.update(state="running")
                self.bridge_log.write(
                    "turn_steer_failed",
                    source="human",
                    operation="turn/steer",
                    message_summary=_safe_summary(task),
                    thread_id=thread_id,
                    turn_id=turn_id,
                    state="failed",
                    error=str(exc),
                )
                raise CodexError(str(exc), layer="steer") from exc
            finally:
                client.close()
            accepted_turn = result.get("turnId")
            if accepted_turn != turn_id:
                self.runtime.update(state="running")
                raise CodexProtocolError(
                    "turn/steer retornou turnId diferente do turn ativo"
                )
            intervention = {
                "timestamp": _utc_now(),
                "source": "human",
                "operation": "turn/steer",
                "thread_id": thread_id,
                "turn_id": turn_id,
                "client_message_id": client_message_id,
                "summary": _safe_summary(task),
                "state": "accepted",
            }
            self.runtime.append_intervention(intervention)
            self.runtime.append_state_event(
                "steer accepted",
                source="human",
                thread_id=thread_id,
                turn_id=turn_id,
                state_result="running",
                summary=_safe_summary(task),
            )
            self.runtime.update(state="running", last_instruction_source="human")
            self.bridge_log.write(
                "turn_steered",
                source="human",
                operation="turn/steer",
                message_summary=_safe_summary(task),
                thread_id=thread_id,
                turn_id=turn_id,
                state="accepted",
                client_message_id=client_message_id,
            )
            return {
                "accepted": True,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "source": "human",
                "status": "steer accepted",
            }

    def cancel(self) -> dict[str, Any]:
        with self.runtime.control_mutex():
            try:
                thread_id, turn_id = self._resolve_active_turn()
            except CodexError:
                return {"cancelled": False, "reason": "nenhum turn ativo"}
            self._cancelled_turns.add(turn_id)

            def mark_cancelled(state: dict[str, Any]) -> None:
                cancelled = list(state.get("cancelled_turn_ids") or [])
                if turn_id not in cancelled:
                    cancelled.append(turn_id)
                state["cancelled_turn_ids"] = cancelled[-100:]
                state["state"] = "cancelling"
                state["last_instruction_source"] = "human"
                state["queue_epoch"] = int(state.get("queue_epoch") or 0) + 1
                state["queue_length"] = 0

            self.runtime.mutate(mark_cancelled)
            client = CodexAppServerClient(
                self.endpoint,
                timeout=min(self.timeout, 30),
            )
            try:
                client.connect()
                client.request(
                    "turn/interrupt",
                    {"threadId": thread_id, "turnId": turn_id},
                )
            finally:
                client.close()
            self.runtime.append_state_event(
                "interrupt sent",
                source="human",
                thread_id=thread_id,
                turn_id=turn_id,
                state_result="cancelling",
            )
            self.bridge_log.write(
                "turn_cancelled",
                source="human",
                operation="turn/interrupt",
                thread_id=thread_id,
                turn_id=turn_id,
                state="cancelling",
            )
            return {
                "cancelled": True,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "source": "human",
                "status": "interrupt sent",
            }

    def _resolve_active_turn(self) -> tuple[str, str]:
        shared = self.runtime.read()
        thread_id = shared.get("thread_id")
        turn_id = shared.get("turn_id")
        if (
            shared.get("state") in {"running", "steering", "cancelling"}
            and isinstance(thread_id, str)
            and isinstance(turn_id, str)
        ):
            actual_turn_id = self._server_active_turn(thread_id)
            if actual_turn_id == turn_id:
                return thread_id, turn_id
            self.runtime.update(thread_id=thread_id, turn_id=None, state="idle")
            raise CodexError(
                "turn ativo mudou ou ja concluiu; steer nao enviado",
                layer="steer",
            )
        session = self._load_session()
        thread_id = session.get("thread_id") if session else None
        if not isinstance(thread_id, str):
            raise CodexError("thread persistida ausente", layer="steer")
        turn_id = self._server_active_turn(thread_id)
        if turn_id:
            self.runtime.update(
                thread_id=thread_id,
                turn_id=turn_id,
                state="running",
            )
            return thread_id, turn_id
        self.runtime.update(
            thread_id=thread_id,
            turn_id=None,
            state="idle",
        )
        raise CodexError("nenhum turn ativo para direcionar", layer="steer")

    def _server_active_turn(self, thread_id: str) -> str | None:
        client = CodexAppServerClient(self.endpoint, timeout=min(self.timeout, 30))
        try:
            client.connect()
            response = client.request(
                "thread/read",
                {"threadId": thread_id, "includeTurns": True},
            )
        finally:
            client.close()
        thread = response.get("thread")
        if not isinstance(thread, dict):
            raise CodexError("thread/read sem thread", layer="steer")
        status = thread.get("status")
        if not isinstance(status, dict) or status.get("type") != "active":
            return None
        turns = thread.get("turns")
        for turn in reversed(turns if isinstance(turns, list) else []):
            if isinstance(turn, dict) and turn.get("status") == "inProgress":
                turn_id = turn.get("id")
                if isinstance(turn_id, str):
                    return turn_id
        return None

    def shared_start(self) -> dict[str, Any]:
        server = self.start_server()
        thread_id = self.ensure_thread()
        if self._thread_created:
            bootstrap = self.run_turn(
                "Inicialize esta thread compartilhada. Responda apenas READY.",
                origin="system",
            )
            if not bootstrap.ok:
                raise CodexError(
                    bootstrap.error or "falha ao persistir thread compartilhada",
                    layer="thread_bootstrap",
                )
        return {
            "ok": True,
            "endpoint": self.endpoint,
            "thread_id": thread_id,
            "server_started": server.get("started", False),
            "terminal_command": f"codex --remote {self.endpoint}",
            "terminal_same_thread_command": (
                subprocess.list2cmdline(self._shared_tui_command(thread_id))
            ),
            "tui_command": "python -m tern.orchestrator codex-shared-tui",
            "events_command": "python -m tern.orchestrator codex-shared-events --follow",
            "status_command": "python -m tern.orchestrator codex-shared-status",
            "steer_command": "python -m tern.orchestrator codex-steer \"instrucao\"",
            "interrupt_command": "python -m tern.orchestrator codex-interrupt",
        }

    def _shared_tui_command(self, thread_id: str) -> list[str]:
        return [
            self.executable,
            "resume",
            "--remote",
            self.endpoint,
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(self.project),
            thread_id,
        ]

    def prepare_shared_tui(self) -> dict[str, Any]:
        """Validate the persisted thread and build a TUI command without a new turn."""
        session = self._load_session()
        thread_id = session.get("thread_id") if session else None
        if not isinstance(thread_id, str) or not thread_id:
            return {
                "ok": False,
                "error": "shared_session_not_found",
                "message": (
                    "sessao compartilhada nao encontrada; "
                    "use codex-shared-start para recuperar"
                ),
                "new_thread_started": False,
            }
        try:
            self.start_server()
        except CodexError as exc:
            return {
                "ok": False,
                "error": "codex_server_unavailable",
                "message": str(exc),
                "thread_id": thread_id,
                "new_thread_started": False,
            }

        client = CodexAppServerClient(
            self.endpoint,
            timeout=min(self.timeout, 60),
        )
        try:
            client.connect()
            response = client.request(
                "thread/read",
                {"threadId": thread_id, "includeTurns": False},
            )
        except CodexProtocolError as exc:
            return {
                "ok": False,
                "error": "thread_not_found",
                "message": (
                    f"thread persistida {thread_id} nao existe mais: {exc}. "
                    "Use codex-shared-start para criar uma recuperacao explicita."
                ),
                "thread_id": thread_id,
                "new_thread_started": False,
            }
        except CodexError as exc:
            return {
                "ok": False,
                "error": "thread_read_failed",
                "message": str(exc),
                "thread_id": thread_id,
                "new_thread_started": False,
            }
        finally:
            client.close()
        thread = response.get("thread")
        if not isinstance(thread, dict) or thread.get("id") != thread_id:
            return {
                "ok": False,
                "error": "invalid_thread_response",
                "message": "thread/read nao confirmou a thread persistida",
                "thread_id": thread_id,
                "new_thread_started": False,
            }
        command = self._shared_tui_command(thread_id)
        return {
            "ok": True,
            "title": "Codex shared TUI",
            "project": str(self.project),
            "thread_id": thread_id,
            "endpoint": self.endpoint,
            "permissions": "dangerously-bypass-approvals-and-sandbox",
            "command": command,
            "command_line": subprocess.list2cmdline(command),
            "new_thread_started": False,
        }

    def open_shared_tui(
        self,
        *,
        launcher: Callable[[list[str], Path], Any] | None = None,
    ) -> dict[str, Any]:
        prepared = self.prepare_shared_tui()
        if not prepared.get("ok"):
            return prepared
        command = list(prepared["command"])
        try:
            if launcher is not None:
                launched = launcher(command, self.project)
            else:
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                launched = subprocess.Popen(
                    command,
                    cwd=str(self.project),
                    creationflags=flags,
                    shell=False,
                )
        except OSError as exc:
            return {
                **prepared,
                "ok": False,
                "error": "tui_launch_failed",
                "message": str(exc),
            }
        pid = getattr(
            launched,
            "pid",
            launched if isinstance(launched, int) else None,
        )
        self.bridge_log.write(
            "shared_tui_opened",
            source="human",
            operation="tui/open",
            project=str(self.project),
            thread_id=prepared["thread_id"],
            endpoint=self.endpoint,
            permissions=prepared["permissions"],
            pid=pid,
        )
        return {**prepared, "tui_pid": pid}

    def shared_status(self) -> dict[str, Any]:
        session = self._load_session() or {}
        shared = self.runtime.read()
        thread_id = session.get("thread_id") or shared.get("thread_id")
        actual_thread_status: dict[str, Any] | None = None
        actual_turn_id: str | None = None
        last_turn_status: str | None = None
        error: str | None = None
        if self.is_ready() and isinstance(thread_id, str):
            client = CodexAppServerClient(
                self.endpoint,
                timeout=min(self.timeout, 30),
            )
            try:
                client.connect()
                response = client.request(
                    "thread/read",
                    {"threadId": thread_id, "includeTurns": True},
                )
                thread = response.get("thread")
                if isinstance(thread, dict):
                    status_value = thread.get("status")
                    if isinstance(status_value, dict):
                        actual_thread_status = status_value
                    turns = thread.get("turns")
                    for turn in reversed(turns if isinstance(turns, list) else []):
                        if not isinstance(turn, dict):
                            continue
                        last_turn_status = str(turn.get("status") or "") or None
                        if turn.get("status") == "inProgress":
                            value = turn.get("id")
                            actual_turn_id = value if isinstance(value, str) else None
                        break
            except CodexError as exc:
                error = str(exc)
            finally:
                client.close()
        state = str(shared.get("state") or "idle")
        turn_id = shared.get("turn_id")
        if actual_turn_id:
            turn_id = actual_turn_id
            if state not in {"steering", "cancelling"}:
                state = "running"
        elif state in {"starting", "running", "steering", "cancelling"}:
            state = (
                "completed"
                if last_turn_status == "completed"
                else "failed"
                if last_turn_status == "failed"
                else "idle"
            )
            turn_id = None
            self.runtime.update(
                state=state,
                turn_id=None,
                qwen_connected=False,
                qwen_pid=None,
            )
        last_event = shared.get("last_event_at")
        age_seconds: float | None = None
        if isinstance(last_event, str):
            try:
                observed = datetime.fromisoformat(last_event)
                age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - observed).total_seconds(),
                )
            except ValueError:
                pass
        bridge_pid = shared.get("bridge_pid")
        qwen_pid = shared.get("qwen_pid")
        tui_processes = self._known_tui_processes()
        if tui_processes is None:
            tui_clients = None
            standalone_tui_count = None
        else:
            target_thread = str(thread_id) if thread_id else None
            shared_processes = [
                item
                for item in tui_processes
                if item.get("remote_endpoint") == self.endpoint
                and item.get("thread_id") == target_thread
            ]
            tui_clients = len(shared_processes)
            standalone_tui_count = len(tui_processes) - tui_clients
        shared_tui_connected = (
            bool(tui_clients) if tui_clients is not None else None
        )
        standalone_warning = None
        if standalone_tui_count:
            standalone_warning = (
                "A standalone Codex TUI is running. Messages entered there may not "
                "appear in the Jarvis shared thread."
            )
        return {
            "title": "Codex shared session",
            "endpoint": self.endpoint,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "state": state,
            "last_instruction_source": shared.get("last_instruction_source"),
            "queue_length": int(shared.get("queue_length") or 0),
            "last_event_at": last_event,
            "last_event_age_seconds": age_seconds,
            "app_server_ready": self.is_ready(),
            "app_server_thread_status": actual_thread_status,
            "tui_clients_known": tui_clients,
            "tui_connected": shared_tui_connected,
            "shared_tui_connected": shared_tui_connected,
            "standalone_tui_count": standalone_tui_count,
            "standalone_tui_detected": (
                bool(standalone_tui_count)
                if standalone_tui_count is not None
                else None
            ),
            "tui_warning": standalone_warning,
            "qwen_connected": bool(shared.get("qwen_connected"))
            and self._pid_alive(qwen_pid),
            "bridge_connected": bool(shared.get("bridge_connected"))
            and self._pid_alive(bridge_pid),
            "last_terminal_state": shared.get("last_terminal_state"),
            "result_discarded": bool(shared.get("result_discarded")),
            "client_count_note": (
                "somente clientes TUI locais conhecidos por inspecao de processo; "
                "App Server nao fornece contagem total"
            ),
            "error": error,
        }

    def _known_tui_clients(self, thread_id: str | None) -> int | None:
        values = self._known_tui_thread_ids()
        if values is None:
            return None
        if thread_id is None:
            return len(values)
        return sum(1 for value in values if value == thread_id)

    def _known_tui_processes(self) -> list[dict[str, Any]] | None:
        if os.name != "nt":
            return None
        command = (
            "$items=Get-CimInstance Win32_Process | "
            "Where-Object {$_.Name -eq 'codex.exe'} | "
            "Select-Object ProcessId,CommandLine; "
            "$items | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            values = json.loads(result.stdout)
            if isinstance(values, dict):
                values = [values]
            if not isinstance(values, list):
                return []
            processes: list[dict[str, Any]] = []
            non_tui_commands = re.compile(
                r"(?i)^\s*(?:app-server|exec|review|mcp-server|remote-control|"
                r"doctor|sandbox|completion|update|login|logout)\b"
            )
            remote_pattern = re.compile(
                r"(?i)(?:^|\s)--remote(?:=|\s+)(?:\"([^\"]+)\"|(\S+))"
            )
            thread_pattern = re.compile(
                r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
                re.IGNORECASE,
            )
            for item in values:
                if not isinstance(item, dict):
                    continue
                command_line = str(item.get("CommandLine") or "")
                executable_match = re.search(
                    r"(?i)codex\.exe\"?\s*(.*)$",
                    command_line,
                )
                executable_tail = (
                    executable_match.group(1) if executable_match else command_line
                )
                if non_tui_commands.search(executable_tail):
                    continue
                remote_match = remote_pattern.search(command_line)
                remote_endpoint = (
                    (remote_match.group(1) or remote_match.group(2))
                    if remote_match
                    else None
                )
                resume_match = re.search(r"(?i)\bresume\b(.*)$", command_line)
                thread_match = (
                    thread_pattern.search(resume_match.group(1))
                    if resume_match
                    else None
                )
                processes.append(
                    {
                        "pid": item.get("ProcessId"),
                        "command_line": command_line,
                        "remote_endpoint": remote_endpoint,
                        "thread_id": (
                            thread_match.group(0) if thread_match else None
                        ),
                    }
                )
            return processes
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return None

    def _known_tui_thread_ids(self) -> list[str] | None:
        processes = self._known_tui_processes()
        if processes is None:
            return None
        return [
            str(item["thread_id"])
            for item in processes
            if item.get("remote_endpoint") == self.endpoint
            and isinstance(item.get("thread_id"), str)
        ]

    def read_turn(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        """Read one existing turn without starting or changing it."""
        self.start_server()
        client = CodexAppServerClient(
            self.endpoint,
            timeout=min(self.timeout, 60),
        )
        try:
            client.connect()
            response = client.request(
                "thread/read",
                {"threadId": thread_id, "includeTurns": True},
            )
        finally:
            client.close()
        thread = response.get("thread")
        if not isinstance(thread, dict) or thread.get("id") != thread_id:
            raise CodexError("thread/read nao retornou a thread esperada", layer="thread/read")
        turns = thread.get("turns")
        for turn in reversed(turns if isinstance(turns, list) else []):
            if not isinstance(turn, dict) or turn.get("id") != turn_id:
                continue
            messages = [
                str(item.get("text"))
                for item in turn.get("items") or []
                if isinstance(item, dict)
                and item.get("type") == "agentMessage"
                and isinstance(item.get("text"), str)
            ]
            raw_status = str(turn.get("status") or "failed")
            status = "running" if raw_status == "inProgress" else raw_status
            return {
                "thread_id": thread_id,
                "turn_id": turn_id,
                "status": status,
                "raw_status": raw_status,
                "final_response": messages[-1] if messages else "",
                "error": turn.get("error"),
                "turn": turn,
            }
        raise CodexError("turn_id nao encontrado em thread/read", layer="thread/read")

    def resume_turn(
        self,
        thread_id: str,
        turn_id: str,
        *,
        origin: str = "qwen",
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> CodexResult:
        """Reconnect to an already-created turn; never invokes turn/start."""
        ticket = self._queue_acquire(origin)
        try:
            with self.runtime.turn_mutex(timeout=self.timeout):
                self._event_count = 0
                self._final_messages = []
                self._event_observer = event_callback
                with self._state_lock:
                    self._active_thread_id = thread_id
                    self._active_turn_id = turn_id
                self.runtime.update(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    state="reconnecting",
                    bridge_connected=True,
                    bridge_pid=os.getpid(),
                    qwen_connected=origin == "qwen",
                    qwen_pid=os.getpid() if origin == "qwen" else None,
                )
                self._emit_observer(
                    "codex_reconnecting",
                    {"thread_id": thread_id, "turn_id": turn_id},
                )
                try:
                    completed = self._recover_active_turn(thread_id, turn_id)
                    if completed is None:
                        self.runtime.update(state="running")
                        self._emit_observer(
                            "codex_reconnected",
                            {"thread_id": thread_id, "turn_id": turn_id},
                        )
                        completed = self._wait_for_completion(thread_id, turn_id)
                    return self._result_for_existing_turn(
                        thread_id,
                        turn_id,
                        completed,
                        event_callback=event_callback,
                    )
                except CodexError as exc:
                    self.runtime.update(
                        state="disconnected",
                        last_terminal_state="disconnected",
                        qwen_connected=False,
                        qwen_pid=None,
                    )
                    return CodexResult(
                        accepted=True,
                        thread_id=thread_id,
                        turn_id=turn_id,
                        status="disconnected",
                        final_response="",
                        error=f"{exc.layer}: {exc}",
                        events=self._event_count,
                    )
                finally:
                    self._event_observer = None
                    with self._state_lock:
                        self._active_thread_id = None
                        self._active_turn_id = None
        finally:
            self._queue_release(ticket)

    def _result_for_existing_turn(
        self,
        thread_id: str,
        turn_id: str,
        completed: dict[str, Any],
        *,
        event_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> CodexResult:
        turn = completed.get("turn")
        status = str(turn.get("status")) if isinstance(turn, dict) else "failed"
        error_value = turn.get("error") if isinstance(turn, dict) else None
        error = None if error_value in {None, ""} else str(error_value)
        final = self._select_final_message(turn_id)
        shared = self.runtime.read()
        cancelled = (
            turn_id in self._cancelled_turns
            or turn_id in (shared.get("cancelled_turn_ids") or [])
            or status == "interrupted"
        )
        discarded = bool(cancelled)
        if cancelled:
            status = "interrupted"
            final = ""
            error = "cancelled"
            self.runtime.update(
                state="idle",
                last_terminal_state=status,
                turn_id=None,
                qwen_connected=False,
                qwen_pid=None,
                result_discarded=True,
            )
        else:
            self.runtime.update(
                state="completed" if status == "completed" else "failed",
                last_terminal_state=status,
                turn_id=None,
                qwen_connected=False,
                qwen_pid=None,
                result_discarded=False,
            )
        interventions = tuple(self.runtime.interventions_for(turn_id))
        state_events = tuple(self.runtime.state_events_for(turn_id))
        if event_callback is not None:
            event_callback(
                "codex_completed" if status == "completed" and not error else "codex_failed",
                {
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "status": status,
                    "error": error,
                },
            )
        return CodexResult(
            accepted=True,
            thread_id=thread_id,
            turn_id=turn_id,
            status=status,
            final_response=final,
            error=error,
            events=self._event_count,
            human_interventions=interventions,
            state_events=state_events,
            result_discarded=discarded,
        )

    def review_session(self, *, turn_limit: int | str = 10) -> dict[str, Any]:
        """Read persisted Codex history without starting or resuming a turn."""
        if isinstance(turn_limit, str) and re.fullmatch(r"\s*\d+\s*", turn_limit):
            normalized_limit: Any = int(turn_limit)
        else:
            normalized_limit = turn_limit
        if (
            not isinstance(normalized_limit, int)
            or isinstance(normalized_limit, bool)
            or not 1 <= normalized_limit <= 50
        ):
            return self._session_review_error(
                "invalid_turn_limit",
                "turn_limit deve ser um inteiro entre 1 e 50",
                thread_id=None,
            )
        turn_limit = normalized_limit
        session = self._load_session()
        thread_id = session.get("thread_id") if session else None
        if not isinstance(thread_id, str) or not thread_id:
            return self._session_review_error(
                "thread_not_found",
                "thread compartilhada nao encontrada",
                thread_id=None,
            )
        try:
            self.start_server()
        except CodexError as exc:
            return self._session_review_error(
                "codex_server_unavailable",
                "servidor Codex indisponivel",
                thread_id=thread_id,
                technical=exc,
            )
        client = CodexAppServerClient(
            self.endpoint,
            timeout=min(self.timeout, 60),
        )
        try:
            client.connect()
            response = client.request(
                "thread/read",
                {"threadId": thread_id, "includeTurns": True},
            )
        except CodexProtocolError as exc:
            message = str(exc).casefold()
            error = (
                "thread_not_found"
                if any(value in message for value in ("not found", "no rollout", "unknown thread"))
                else "thread_read_failed"
            )
            return self._session_review_error(
                error,
                (
                    "thread compartilhada nao encontrada"
                    if error == "thread_not_found"
                    else "nao foi possivel ler a thread Codex"
                ),
                thread_id=thread_id,
                technical=exc,
            )
        except CodexError as exc:
            error = (
                "codex_server_unavailable"
                if exc.layer in {"websocket", "send", "events"}
                else "thread_read_failed"
            )
            return self._session_review_error(
                error,
                (
                    "servidor Codex indisponivel"
                    if error == "codex_server_unavailable"
                    else "nao foi possivel ler a thread Codex"
                ),
                thread_id=thread_id,
                technical=exc,
            )
        finally:
            client.close()
        try:
            snapshot = normalize_thread_read(response)
        except InvalidThreadResponse as exc:
            return self._session_review_error(
                "invalid_thread_response",
                "resposta invalida ao ler a thread Codex",
                thread_id=thread_id,
                technical=exc,
            )
        if snapshot.thread_id != thread_id:
            return self._session_review_error(
                "invalid_thread_response",
                "thread retornada nao corresponde a thread compartilhada",
                thread_id=thread_id,
            )
        all_turns = snapshot.turns
        selected = all_turns[-turn_limit:]
        turns = [self._summarize_history_turn(item) for item in selected]
        tui_thread_ids = self._known_tui_thread_ids()
        known_tui_threads = (
            [value for value in tui_thread_ids if isinstance(value, str)]
            if isinstance(tui_thread_ids, list)
            else []
        )
        threads_match = (
            None
            if not known_tui_threads
            else all(value == thread_id for value in known_tui_threads)
        )
        warning = None
        if threads_match is False:
            warning = (
                f"Thread do assistente: {thread_id}\n"
                f"Thread da TUI: {', '.join(known_tui_threads)}\n"
                "As sessoes nao sao iguais."
            )
        summary = "\n".join(
            (
                f"Turn {item['turn_id']} [{item['status']}]. "
                f"Tarefa: {_safe_summary(item['requested']) or 'nao registrada'}. "
                f"Resposta: {_safe_summary(item['final_response']) or 'sem resposta final'}."
            )
            for item in turns
        )
        last_turn = turns[-1] if turns else None
        self.bridge_log.write(
            "session_reviewed",
            source="qwen",
            operation="thread/read",
            project=str(self.project),
            thread_id=thread_id,
            turn_id=last_turn.get("turn_id") if last_turn else None,
            state=last_turn.get("status") if last_turn else "empty",
            turns_read=len(turns),
            turn_limit=turn_limit,
        )
        return {
            "ok": True,
            "operation": "thread/read",
            "project": str(self.project),
            "thread_id": thread_id,
            "tui_thread_id": (
                known_tui_threads[0] if len(known_tui_threads) == 1 else None
            ),
            "tui_thread_ids": known_tui_threads,
            "threads_match": threads_match,
            "thread_warning": warning,
            "thread_status": snapshot.status,
            "codex_cli_version": snapshot.cli_version,
            "turns_available": len(all_turns),
            "turns_reviewed": len(turns),
            "last_turn_id": last_turn.get("turn_id") if last_turn else None,
            "last_status": last_turn.get("status") if last_turn else None,
            "summary_source": turns,
            "error": None,
            "turn_count_total": len(all_turns),
            "turns_read": len(turns),
            "turn_limit": turn_limit,
            "last_turn": last_turn,
            "last_turn_state": last_turn.get("status") if last_turn else "empty",
            "conversation_summary": summary,
            "new_turn_started": False,
        }

    @staticmethod
    def _summarize_history_turn(turn: TurnSnapshot) -> dict[str, Any]:
        tasks: list[str] = []
        responses: list[str] = []
        final_responses: list[str] = []
        human_interventions: list[str] = []
        events: list[str] = []
        files: list[str] = []
        tests: list[str] = []
        errors: list[str] = []

        def add_unique(values: list[str], value: str, *, limit: int = 1000) -> None:
            safe = str(_redact(value)).strip()[:limit]
            if safe and safe not in values:
                values.append(safe)

        def mentioned_paths(text: str) -> None:
            for value in re.findall(
                r"(?i)(?:[A-Z]:\\[^\s\]\[()'\"`]+\."
                r"(?:py|md|json|jsonl|toml|txt|yaml|yml|ini|cfg)(?::\d+)?|"
                r"[A-Za-z0-9_.\\/-]+\.(?:py|md|json|jsonl|toml|txt|yaml|yml|ini|cfg)(?::\d+)?)",
                text[:4000],
            ):
                add_unique(files, value.rstrip(".,:;"), limit=1000)

        for message in turn.messages:
            text = message.get("text")
            if not isinstance(text, str):
                continue
            if message.get("role") == "user":
                add_unique(tasks, text, limit=600)
                mentioned_paths(text)
                client_id = str(message.get("client_id") or "")
                if "human" in client_id.casefold():
                    add_unique(human_interventions, text, limit=500)
            elif message.get("role") == "assistant":
                add_unique(responses, text)
                if message.get("phase") == "final_answer":
                    add_unique(final_responses, text)
                mentioned_paths(text)
                for line in text.splitlines():
                    if re.search(
                        r"(?i)(?:\b(?:pytest|unittest|tox)\b|"
                        r"\b\d+\s+(?:tests?|testes?)\b|"
                        r"\btests?\s+(?:passed|failed|executed|run)\b|"
                        r"\btestes?\s+(?:passaram|falharam|executados?)\b)",
                        line,
                    ):
                        add_unique(tests, line, limit=400)

        for item in turn.items:
            item_type = item.get("type")
            if item_type == "fileChange":
                for change in item.get("changes") or []:
                    if isinstance(change, dict) and isinstance(change.get("path"), str):
                        add_unique(files, change["path"], limit=1000)
            elif item_type == "commandExecution":
                command = str(item.get("command") or "")
                if re.search(r"(?i)\b(?:pytest|unittest|tox|test)\b", command):
                    add_unique(tests, command, limit=3000)
                status = str(item.get("status") or "")
                exit_code = item.get("exitCode")
                if status.casefold() in {"failed", "error"} or (
                    isinstance(exit_code, int) and exit_code != 0
                ):
                    add_unique(
                        errors,
                        f"command status={status or 'unknown'} exit_code={exit_code}: {command}",
                        limit=800,
                    )
                add_unique(
                    events,
                    f"commandExecution status={status or 'unknown'}: {command}",
                    limit=600,
                )
            elif item_type not in {"userMessage", "agentMessage", "contextCompaction"}:
                item_status = item.get("status")
                add_unique(
                    events,
                    f"{item_type or 'unknown'}"
                    + (f" status={item_status}" if item_status is not None else ""),
                    limit=300,
                )
        status = turn.status or "unknown"
        error = turn.error
        if error:
            detail = (
                json.dumps(error, ensure_ascii=False)
                if isinstance(error, dict)
                else str(error)
            )
            add_unique(errors, detail)
        if status in {
            "interrupted",
            "failed",
            "cancelled",
            "canceled",
            "inProgress",
            "in_progress",
        }:
            add_unique(errors, f"turn status={status}")
        final_response = (
            final_responses[-1]
            if final_responses
            else (responses[-1] if responses else "")
        )
        return {
            "turn_id": turn.turn_id,
            "status": status,
            "requested": tasks[0] if tasks else "",
            "additional_instructions": tasks[1:2],
            "final_response": final_response,
            "human_interventions": human_interventions,
            "events": events[:8],
            "files_mentioned_or_changed": files,
            "tests_mentioned_or_executed": tests,
            "errors_cancellations_or_pending": errors,
            "started_at": turn.started_at,
            "completed_at": turn.completed_at,
            "duration_ms": turn.duration_ms,
        }

    def _session_review_error(
        self,
        error: str,
        message: str,
        *,
        thread_id: str | None,
        technical: Exception | None = None,
    ) -> dict[str, Any]:
        self.bridge_log.write(
            "session_review_failed",
            source="qwen",
            operation="thread/read",
            project=str(self.project),
            thread_id=thread_id,
            state="failed",
            error=error,
            technical_type=type(technical).__name__ if technical else None,
            technical_detail=str(technical) if technical else None,
        )
        return {
            "ok": False,
            "operation": "thread/read",
            "project": str(self.project),
            "thread_id": thread_id,
            "turns_available": 0,
            "turns_reviewed": 0,
            "last_turn_id": None,
            "last_status": None,
            "summary_source": [],
            "error": error,
            "message": message,
            "new_turn_started": False,
        }

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        if not isinstance(pid, int) or pid <= 0:
            return False
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        state = self.runtime.read()
        if state.get("bridge_pid") == os.getpid():
            self.runtime.update(
                bridge_connected=False,
                bridge_pid=None,
                qwen_connected=False,
                qwen_pid=None,
            )

    def _queue_acquire(self, origin: str) -> object:
        ticket = object()
        priority = {"human": 0, "system": 1, "qwen": 2}[origin]
        with self._queue_condition:
            self._queue_sequence += 1
            heapq.heappush(
                self._queue,
                (priority, self._queue_sequence, ticket),
            )
            self.runtime.update(queue_length=len(self._queue))
            while self._queue_active or self._queue[0][2] is not ticket:
                self._queue_condition.wait()
            heapq.heappop(self._queue)
            self._queue_active = True
            self.runtime.update(queue_length=len(self._queue))
        return ticket

    def _queue_release(self, _ticket: object) -> None:
        with self._queue_condition:
            self._queue_active = False
            self.runtime.update(queue_length=len(self._queue))
            self._queue_condition.notify_all()

    def _wait_for_completion(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        client = self.connect()
        cached = self._completed_turns.pop(turn_id, None)
        if cached is not None:
            return cached
        while True:
            try:
                message = client.receive()
            except CodexError as exc:
                self._emit_observer(
                    "codex_disconnected",
                    {
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        "error": str(exc),
                    },
                )
                self.bridge_log.write(
                    "websocket_disconnected",
                    source="system",
                    operation="reconnect",
                    thread_id=thread_id,
                    turn_id=turn_id,
                    state="running",
                    error=str(exc),
                )
                self._emit_observer(
                    "codex_reconnecting",
                    {"thread_id": thread_id, "turn_id": turn_id},
                )
                recovered = self._recover_active_turn(thread_id, turn_id)
                if recovered is not None:
                    return recovered
                client = self.connect()
                self._emit_observer(
                    "codex_reconnected",
                    {"thread_id": thread_id, "turn_id": turn_id},
                )
                continue
            cached = self._completed_turns.pop(turn_id, None)
            if cached is not None:
                return cached
            if message.get("method") != "turn/completed":
                continue
            params = message.get("params")
            if not isinstance(params, dict):
                continue
            turn = params.get("turn")
            observed_turn_id = turn.get("id") if isinstance(turn, dict) else None
            if params.get("threadId") == thread_id and observed_turn_id == turn_id:
                return params

    def _recover_active_turn(
        self,
        thread_id: str,
        turn_id: str,
    ) -> dict[str, Any] | None:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                client = self.reconnect()
                response = client.request(
                    "thread/read",
                    {"threadId": thread_id, "includeTurns": True},
                )
                thread = response.get("thread")
                turns = thread.get("turns") if isinstance(thread, dict) else []
                for turn in reversed(turns if isinstance(turns, list) else []):
                    if not isinstance(turn, dict) or turn.get("id") != turn_id:
                        continue
                    for item in turn.get("items") or []:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "agentMessage"
                            and isinstance(item.get("text"), str)
                        ):
                            self._final_messages.append((turn_id, item["text"]))
                    if turn.get("status") in {"completed", "failed", "interrupted"}:
                        self.bridge_log.write(
                            "turn_recovered",
                            source="system",
                            operation="thread/read",
                            thread_id=thread_id,
                            turn_id=turn_id,
                            state=turn.get("status"),
                        )
                        return {"threadId": thread_id, "turn": turn}
                    self.bridge_log.write(
                        "turn_reconnected",
                        source="system",
                        operation="thread/resume",
                        thread_id=thread_id,
                        turn_id=turn_id,
                        state="running",
                    )
                    return None
                raise CodexProtocolError("turn ativo nao encontrado em thread/read")
            except CodexError as exc:
                last_error = exc
                time.sleep(0.2 * (attempt + 1))
        raise CodexError(
            f"falha ao recuperar WebSocket durante turn: {last_error}",
            layer="reconnect",
        )

    def _on_event(self, message: dict[str, Any]) -> None:
        method = str(message.get("method") or "")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        turn = params.get("turn")
        if turn_id is None and isinstance(turn, dict):
            turn_id = turn.get("id")
        item = params.get("item")
        item_type = item.get("type") if isinstance(item, dict) else None
        status = turn.get("status") if isinstance(turn, dict) else None
        if method == "turn/completed" and isinstance(turn_id, str):
            self._completed_turns[turn_id] = params
        source = "system"
        summary: dict[str, Any] = {
            "method": method,
            "operation": method,
            "source": source,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "item_type": item_type,
            "status": status,
        }
        if method in {"item/started", "item/completed"} and isinstance(item, dict):
            if item.get("type") == "userMessage":
                client_id = item.get("clientId")
                item_id = str(item.get("id") or "")
                text_values = []
                for content in item.get("content") or []:
                    if isinstance(content, dict) and content.get("type") == "text":
                        text = content.get("text")
                        if isinstance(text, str):
                            text_values.append(text)
                message_text = "\n".join(text_values)
                source = self._client_message_sources.get(str(client_id), "")
                shared = self.runtime.read()
                if not source and client_id == shared.get("active_client_message_id"):
                    source = str(shared.get("last_instruction_source") or "system")
                if not source:
                    source = "human"
                is_active_direction = (
                    source == "human"
                    and shared.get("turn_id") == turn_id
                    and shared.get("state")
                    in {"running", "steering", "cancelling"}
                )
                summary.update(
                    source=source,
                    operation=(
                        "turn/steer" if is_active_direction else "turn/start"
                    ),
                    client_message_id=client_id,
                    message_summary=_safe_summary(message_text),
                )
                if (
                    is_active_direction
                    and isinstance(turn_id, str)
                    and item_id
                    and item_id not in self._seen_user_messages
                ):
                    self._seen_user_messages.add(item_id)
                    self.runtime.append_intervention(
                        {
                            "timestamp": _utc_now(),
                            "source": "human",
                            "operation": "turn/steer",
                            "thread_id": thread_id,
                            "turn_id": turn_id,
                            "client_message_id": client_id or item_id,
                            "summary": _safe_summary(message_text),
                            "state": "observed",
                        }
                    )
                    self.runtime.append_state_event(
                        "human steer observed",
                        source="human",
                        thread_id=str(thread_id) if thread_id else None,
                        turn_id=turn_id,
                        state_result="running",
                        summary=_safe_summary(message_text),
                    )
            elif item.get("type") == "agentMessage" and method == "item/completed":
                text = item.get("text")
                if isinstance(text, str):
                    self._final_messages.append((str(turn_id) if turn_id else None, text))
                    summary["text"] = text
                    summary["phase"] = item.get("phase")
            elif item.get("type") == "commandExecution":
                summary["command"] = _safe_summary(str(item.get("command") or ""))
                summary["command_status"] = item.get("status")
                summary["exit_code"] = item.get("exitCode")
            elif item.get("type") == "fileChange":
                summary["changed_paths"] = [
                    change.get("path")
                    for change in item.get("changes") or []
                    if isinstance(change, dict)
                ]
        elif method == "item/agentMessage/delta":
            summary["delta"] = params.get("delta")
        elif method == "error":
            summary["error"] = params.get("error")
        if method == "thread/status/changed":
            thread_status = params.get("status")
            summary["thread_status"] = thread_status
            if isinstance(thread_status, dict):
                status_type = thread_status.get("type")
                shared = self.runtime.read()
                if status_type == "idle" and shared.get("turn_id") is None:
                    self.runtime.update(state="idle")
        self.runtime.update(last_event_at=_utc_now())
        self._event_count += 1
        JsonlEventLog(self.events_path).write("codex_event", **summary)
        self._emit_observer("codex_event", summary)

    def _emit_observer(self, event: str, values: dict[str, Any]) -> None:
        callback = self._event_observer
        if callback is None:
            return
        try:
            callback(event, values)
        except Exception as exc:
            self.bridge_log.write(
                "event_observer_failed",
                source="system",
                operation=event,
                state="failed",
                error=str(exc),
            )

    def _select_final_message(self, turn_id: str) -> str:
        matching = [text for observed, text in self._final_messages if observed == turn_id]
        return matching[-1] if matching else ""

    def _load_session(self) -> dict[str, Any] | None:
        state = self._read_json(self.session_path)
        if not state:
            return None
        if state.get("project") != str(self.project):
            return None
        if state.get("server_endpoint") != self.endpoint:
            return None
        return state

    def _persist_session(self, thread_id: str) -> None:
        self._write_json(
            self.session_path,
            {
                "project": str(self.project),
                "thread_id": thread_id,
                "server_endpoint": self.endpoint,
                "updated_at": _utc_now(),
            },
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _tail(path: Path, limit: int) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""


class CodexRunner:
    """Compatibility facade used by ToolRegistry and direct diagnostics."""

    def __init__(
        self,
        policy: PathPolicy,
        timeout: int = 1800,
        executable: str | None = None,
        *,
        endpoint: str = "ws://127.0.0.1:4500",
        state_dir: Path | None = None,
        preferred_thread_id: str | None = None,
        quick_wait_timeout: int = 60,
        hard_timeout: int = 0,
        job_retention_days: int = 7,
    ):
        self.policy = policy
        self.timeout = timeout
        self.executable = executable or shutil.which("codex") or "codex"
        self.endpoint = endpoint
        self.state_dir = state_dir
        self.preferred_thread_id = (
            str(preferred_thread_id).strip() if preferred_thread_id else None
        )
        default_job_state = (
            policy.roots[0] / ".orchestrator"
            if policy.roots
            else Path(__file__).resolve().parents[2] / ".orchestrator"
        )
        job_state_dir = Path(state_dir or default_job_state)
        self.quick_wait_timeout = quick_wait_timeout
        self.hard_timeout = hard_timeout
        self.jobs = CodexJobStore(
            job_state_dir,
            retention_days=job_retention_days,
        )
        self.sessions = CodexSessionRegistry(job_state_dir)
        self.session_resolver = CodexSessionResolver(self.sessions)
        legacy_session = CodexSessionManager._read_json(
            job_state_dir / "codex-session.json"
        )
        self.sessions.import_legacy(
            session=legacy_session,
            jobs=self.jobs.list(),
        )
        self._managers: dict[Path, CodexSessionManager] = {}
        self._job_lock = threading.Lock()
        self._job_threads: dict[str, threading.Thread] = {}
        self._job_events: dict[str, tuple[threading.Event, threading.Event]] = {}

    def manager_for(self, project: Path) -> CodexSessionManager:
        resolved = project.resolve(strict=True)
        manager = self._managers.get(resolved)
        if manager is None:
            shared_project = Path(__file__).resolve().parents[2]
            uses_shared_state = resolved == shared_project
            state_dir = (
                self.state_dir
                if uses_shared_state
                else resolved / ".orchestrator"
            )
            manager = CodexSessionManager(
                resolved,
                endpoint=self.endpoint,
                timeout=self.timeout,
                executable=self.executable,
                state_dir=state_dir,
                preferred_thread_id=(
                    self.preferred_thread_id if uses_shared_state else None
                ),
            )
            self._managers[resolved] = manager
        return manager

    def shared_project(self) -> Path | None:
        state_dir = self.state_dir or Path(__file__).resolve().parents[2] / ".orchestrator"
        session_path = Path(state_dir) / "codex-session.json"
        try:
            value = json.loads(session_path.read_text(encoding="utf-8"))
            project = value.get("project") if isinstance(value, dict) else None
            if isinstance(project, str) and project:
                resolved = Path(project).resolve(strict=True)
                return resolved if resolved.is_dir() else None
        except (OSError, json.JSONDecodeError):
            return None
        return None

    def delegate_to_codex(
        self,
        *,
        task: str,
        project_path: str,
        continue_current_thread: bool = True,
        thread_id: str | None = None,
        focused_thread_id: str | None = None,
        conversation_id: str | None = None,
        allow_create: bool = True,
        origin: str = "qwen",
        wait: bool = True,
        execution_mode: str | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> CodexResult:
        project = self.policy.resolve(project_path)
        if not project.is_dir():
            raise CodexError("project_path nao e diretorio", layer="validation")
        request_id = str(uuid.uuid4())
        resolution = self.resolve_session(
            project,
            request_id=request_id,
            explicit_thread_id=thread_id,
            focused_thread_id=focused_thread_id or self.preferred_thread_id,
            conversation_id=conversation_id,
            # Session identity is resolved independently from turn-continuation
            # preference. A new thread is allowed only after proving that no
            # reusable session exists.
            force_new=False,
            allow_create=allow_create,
        )
        if not resolution.ok:
            return CodexResult(
                accepted=False,
                thread_id=resolution.thread_id,
                turn_id=None,
                status="failed",
                final_response="",
                error=resolution.reason_code,
                session_resolution=resolution.as_dict(),
            )
        job = self.jobs.create(
            project=str(project),
            task_summary=_safe_summary(task),
            source=origin,
            wait=wait,
            thread_id=resolution.thread_id,
            session_resolution=resolution.as_dict(),
            request_id=request_id,
            execution_mode=execution_mode,
        )
        self.manager_for(project).bridge_log.write(
            "codex_delegation_started",
            request_id=request_id,
            requested_agent="codex",
            project_path=str(project),
            session_resolution_attempted=True,
            candidate_session_count=resolution.candidate_count,
            reusable_candidate_session_count=(
                resolution.reusable_candidate_count
            ),
            selected_session_id=resolution.session_id,
            selected_thread_id=resolution.thread_id,
            session_binding_source=resolution.binding_source,
            session_state=resolution.state,
            session_reused=resolution.reused,
            session_created=resolution.created,
            session_registered=resolution.registered,
            session_recoverable=resolution.recoverable,
            session_visible=resolution.visible,
            active_job_id=resolution.active_job_id,
            delegation_started=True,
            delegation_job_id=job["job_id"],
            resolution_reason_code=resolution.reason_code,
        )
        self.sessions.update(
            str(resolution.thread_id),
            active_job_id=job["job_id"],
            last_used_at=utc_now(),
        )
        start_event, done_event = self._start_job_worker(
            job,
            task=task,
            continue_current_thread=continue_current_thread,
            event_callback=event_callback,
        )
        if wait:
            if done_event.wait(self.quick_wait_timeout):
                result = self._result_from_job(job["job_id"])
                self.jobs.update(job["job_id"], result_delivered=True)
                return result
            value = self.jobs.update(job["job_id"], wait_timed_out=True)
            return CodexResult(
                accepted=True,
                thread_id=value.get("thread_id"),
                turn_id=value.get("turn_id"),
                status=str(value.get("status") or "running"),
                final_response="",
                error=None,
                job_id=job["job_id"],
                wait_timed_out=True,
                session_resolution=resolution.as_dict(),
            )
        start_event.wait(min(10, self.quick_wait_timeout))
        value = self.jobs.get(job["job_id"]) or job
        if value.get("status") in TERMINAL_JOB_STATES:
            result = self._result_from_job(job["job_id"])
            self.jobs.update(job["job_id"], result_delivered=True)
            return result
        return CodexResult(
            accepted=True,
            thread_id=value.get("thread_id"),
            turn_id=value.get("turn_id"),
            status=str(value.get("status") or "queued"),
            final_response="",
            error=None,
            job_id=job["job_id"],
            session_resolution=resolution.as_dict(),
        )

    def resolve_session(
        self,
        project: Path,
        *,
        request_id: str,
        explicit_thread_id: str | None = None,
        focused_thread_id: str | None = None,
        conversation_id: str | None = None,
        force_new: bool = False,
        allow_create: bool = True,
    ) -> CodexSessionResolution:
        manager = self.manager_for(project)
        started = time.perf_counter()
        with self.sessions.resolution_mutex(project, timeout=min(self.timeout, 60)):
            try:
                provider_threads = manager.list_project_threads()
            except CodexError as exc:
                resolution = CodexSessionResolution(
                    "UNAVAILABLE", "SESSION_UNAVAILABLE"
                )
                manager.bridge_log.write(
                    "codex_session_resolution",
                    request_id=request_id,
                    requested_agent="codex",
                    project_path=str(project),
                    session_resolution_attempted=True,
                    candidate_session_count=0,
                    reusable_candidate_session_count=0,
                    selected_session_id=None,
                    selected_thread_id=None,
                    session_binding_source=None,
                    session_state="unavailable",
                    session_reused=False,
                    session_created=False,
                    session_registered=False,
                    session_recoverable=False,
                    session_visible=False,
                    active_job_id=None,
                    delegation_started=False,
                    delegation_job_id=None,
                    resolution_reason_code=resolution.reason_code,
                    resolution_latency_ms=round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                    error=str(exc),
                )
                return resolution
            candidates = self.sessions.reconcile(project, provider_threads)
            resolution = self.session_resolver.resolve(
                project=project,
                candidates=candidates,
                explicit_thread_id=explicit_thread_id,
                focused_thread_id=focused_thread_id,
                conversation_id=conversation_id,
                force_new=force_new,
            )
            if resolution.status == "NONE" and allow_create:
                try:
                    provider = manager.create_thread()
                    registered = self.sessions.register(
                        {
                            **provider,
                            "origin": "jarvis_created",
                            "recoverable": bool(provider.get("recoverable")),
                        }
                    )
                except Exception as exc:
                    resolution = CodexSessionResolution(
                        "UNAVAILABLE",
                        "SESSION_REGISTRATION_FAILED",
                        candidate_count=len(candidates),
                        reusable_candidate_count=(
                            resolution.reusable_candidate_count
                        ),
                    )
                    manager.bridge_log.write(
                        "codex_session_registration_failed",
                        request_id=request_id,
                        project_path=str(project),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    if not registered.get("recoverable"):
                        resolution = CodexSessionResolution(
                            "UNAVAILABLE",
                            "SESSION_NOT_RECOVERABLE",
                            thread_id=registered.get("thread_id"),
                            candidate_count=len(candidates),
                            reusable_candidate_count=(
                                resolution.reusable_candidate_count
                            ),
                            created=True,
                            registered=True,
                        )
                    else:
                        resolution = CodexSessionResolution(
                            "RESOLVED",
                            "NEW_SESSION_CREATED",
                            thread_id=str(registered["thread_id"]),
                            session_id=str(
                                registered.get("session_id")
                                or registered["thread_id"]
                            ),
                            binding_source="newly_created",
                            state=str(registered.get("state") or "idle"),
                            candidate_count=len(candidates),
                            reusable_candidate_count=(
                                resolution.reusable_candidate_count
                            ),
                            created=True,
                            registered=True,
                            recoverable=True,
                            visible=bool(registered.get("visible")),
                        )
            elif resolution.ok:
                try:
                    provider = manager.adopt_thread(str(resolution.thread_id))
                    previous = self.sessions.get(str(resolution.thread_id)) or {}
                    self.sessions.register(
                        {
                            **previous,
                            **provider,
                            "origin": previous.get("origin") or "user_existing",
                            "active_job_id": previous.get("active_job_id"),
                        }
                    )
                except CodexError as exc:
                    reason = (
                        "SESSION_BUSY"
                        if "active writer" in str(exc).casefold()
                        else "SESSION_UNAVAILABLE"
                    )
                    resolution = CodexSessionResolution(
                        "UNAVAILABLE",
                        reason,
                        thread_id=resolution.thread_id,
                        session_id=resolution.session_id,
                        binding_source=resolution.binding_source,
                        state=resolution.state,
                        candidate_count=resolution.candidate_count,
                        reusable_candidate_count=(
                            resolution.reusable_candidate_count
                        ),
                        registered=True,
                        recoverable=resolution.recoverable,
                        visible=resolution.visible,
                        active_job_id=resolution.active_job_id,
                    )
            if resolution.ok:
                self.sessions.bind_project(project, str(resolution.thread_id))
                self.sessions.bind_conversation(
                    conversation_id or "", str(resolution.thread_id), project
                )
            manager.bridge_log.write(
                "codex_session_resolution",
                request_id=request_id,
                requested_agent="codex",
                project_path=str(project),
                session_resolution_attempted=True,
                candidate_session_count=resolution.candidate_count,
                reusable_candidate_session_count=(
                    resolution.reusable_candidate_count
                ),
                selected_session_id=resolution.session_id,
                selected_thread_id=resolution.thread_id,
                session_binding_source=resolution.binding_source,
                session_state=resolution.state,
                session_reused=resolution.reused,
                session_created=resolution.created,
                session_registered=resolution.registered,
                session_recoverable=resolution.recoverable,
                session_visible=resolution.visible,
                active_job_id=resolution.active_job_id,
                delegation_started=False,
                delegation_job_id=None,
                resolution_reason_code=resolution.reason_code,
                resolution_latency_ms=round(
                    (time.perf_counter() - started) * 1000, 3
                ),
            )
            return resolution

    def _start_job_worker(
        self,
        job: dict[str, Any],
        *,
        task: str | None,
        continue_current_thread: bool = True,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        resume: bool = False,
    ) -> tuple[threading.Event, threading.Event]:
        job_id = str(job["job_id"])
        with self._job_lock:
            existing = self._job_events.get(job_id)
            thread = self._job_threads.get(job_id)
            if existing and thread and thread.is_alive():
                return existing
            start_event = threading.Event()
            done_event = threading.Event()
            token = str(uuid.uuid4())
            if not self.jobs.claim_monitor(job_id, token):
                start_event.set()
                return start_event, done_event
            worker = threading.Thread(
                target=self._run_job_worker,
                kwargs={
                    "job_id": job_id,
                    "monitor_token": token,
                    "task": task,
                    "continue_current_thread": continue_current_thread,
                    "event_callback": event_callback,
                    "start_event": start_event,
                    "done_event": done_event,
                    "resume": resume,
                },
                name=f"codex-job-{job_id[:8]}",
                daemon=True,
            )
            self._job_events[job_id] = (start_event, done_event)
            self._job_threads[job_id] = worker
            worker.start()
            return start_event, done_event

    def _run_job_worker(
        self,
        *,
        job_id: str,
        monitor_token: str,
        task: str | None,
        continue_current_thread: bool,
        event_callback: Callable[[str, dict[str, Any]], None] | None,
        start_event: threading.Event,
        done_event: threading.Event,
        resume: bool,
    ) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            done_event.set()
            return
        manager = self.manager_for(Path(str(job["project"])))
        timer_holder: list[threading.Timer | None] = [None]

        def observed(event: str, values: dict[str, Any]) -> None:
            payload = {**values, "job_id": job_id}
            if event == "codex_turn_started":
                self.jobs.update(
                    job_id,
                    status="running",
                    thread_id=values.get("thread_id"),
                    turn_id=values.get("turn_id"),
                    monitor_pid=os.getpid(),
                    monitor_token=monitor_token,
                )
                start_event.set()
                if self.hard_timeout > 0 and timer_holder[0] is None:
                    timer_holder[0] = threading.Timer(
                        self.hard_timeout,
                        lambda: manager.cancel(),
                    )
                    timer_holder[0].daemon = True
                    timer_holder[0].start()
                if event_callback is not None:
                    event_callback("codex_job_started", payload)
                    event_callback(event, payload)
                return
            if event == "codex_event":
                method = str(values.get("method") or "")
                updates: dict[str, Any] = {"last_event": method}
                if method == "error":
                    updates["last_event_error"] = _safe_summary(str(values.get("error") or ""))
                self.jobs.update(job_id, **updates)
                return
            state = {
                "codex_disconnected": "disconnected",
                "codex_reconnecting": "reconnecting",
                "codex_reconnected": "running",
            }.get(event)
            if state:
                self.jobs.update(job_id, status=state)
            if event not in {"codex_completed", "codex_failed"} and event_callback is not None:
                event_callback(event, payload)

        try:
            self.jobs.update(job_id, status="reconnecting" if resume else "starting")
            if resume:
                start_event.set()
                result = manager.resume_turn(
                    str(job["thread_id"]),
                    str(job["turn_id"]),
                    origin=str(job.get("source") or "qwen"),
                    event_callback=observed,
                )
            else:
                if task is None:
                    raise CodexError("tarefa ausente para novo job", layer="job")
                result = manager.run_turn(
                    task,
                    origin=str(job.get("source") or "qwen"),
                    continue_current_thread=continue_current_thread,
                    target_thread_id=str(job["thread_id"]),
                    event_callback=observed,
                    read_only=str(job.get("execution_mode") or "") == "READ_ONLY",
                )
            self._finish_job(job_id, result, event_callback=event_callback)
        except Exception as exc:
            result = CodexResult(
                accepted=False,
                thread_id=job.get("thread_id"),
                turn_id=job.get("turn_id"),
                status="failed",
                final_response="",
                error=f"{type(exc).__name__}: {exc}",
            )
            self._finish_job(job_id, result, event_callback=event_callback)
        finally:
            if timer_holder[0] is not None:
                timer_holder[0].cancel()
            start_event.set()
            done_event.set()
            with self._job_lock:
                self._job_threads.pop(job_id, None)
                self._job_events.pop(job_id, None)

    def _finish_job(
        self,
        job_id: str,
        result: CodexResult,
        *,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        current = self.jobs.get(job_id)
        if (
            current
            and current.get("status") == "interrupted"
            and result.status != "interrupted"
        ):
            result = CodexResult(
                accepted=True,
                thread_id=result.thread_id or current.get("thread_id"),
                turn_id=result.turn_id or current.get("turn_id"),
                status="interrupted",
                final_response="",
                error="cancelled",
                result_discarded=True,
            )
        status = result.status
        if status == "cancelled":
            status = "interrupted"
        result_available = status in {"completed", "failed"}
        if status == "disconnected":
            result_available = False
        value = self.jobs.update(
            job_id,
            thread_id=result.thread_id,
            turn_id=result.turn_id,
            status=status,
            result=_redact(result.as_dict()) if result_available else None,
            result_available=result_available,
            error=result.error,
            human_interventions=list(result.human_interventions),
            result_discarded=result.result_discarded,
            monitor_pid=None,
            monitor_token=None,
        )
        if result.thread_id:
            record = self.sessions.get(result.thread_id)
            if record is not None and record.get("active_job_id") == job_id:
                self.sessions.update(
                    result.thread_id,
                    active_job_id=None,
                    state=("idle" if status in TERMINAL_JOB_STATES else status),
                    last_used_at=utc_now(),
                )
        if event_callback is not None:
            payload = {
                "job_id": job_id,
                "thread_id": result.thread_id,
                "turn_id": result.turn_id,
                "status": status,
                "error": result.error,
            }
            if status == "completed" and self.jobs.claim_notification(job_id, "completed"):
                event_callback("codex_job_completed", payload)
            elif status == "failed" and self.jobs.claim_notification(job_id, "failed"):
                event_callback("codex_job_failed", payload)
            elif status == "interrupted":
                event_callback("codex_job_interrupted", payload)
        return value

    def _result_from_job(self, job_id: str) -> CodexResult:
        job = self.jobs.get(job_id)
        if job is None:
            raise CodexError("job Codex ausente", layer="job")
        stored = job.get("result")
        if isinstance(stored, dict):
            return CodexResult(
                accepted=bool(stored.get("accepted")),
                thread_id=stored.get("thread_id"),
                turn_id=stored.get("turn_id"),
                status=str(stored.get("status") or job.get("status") or "failed"),
                final_response=str(stored.get("final_response") or ""),
                error=stored.get("error"),
                events=int(stored.get("events") or 0),
                human_interventions=tuple(stored.get("human_interventions") or []),
                state_events=tuple(stored.get("state_events") or []),
                result_discarded=bool(stored.get("result_discarded")),
                job_id=job_id,
                wait_timed_out=bool(job.get("wait_timed_out")),
                session_resolution=(
                    dict(job.get("session_resolution"))
                    if isinstance(job.get("session_resolution"), dict)
                    else None
                ),
            )
        return CodexResult(
            accepted=True,
            thread_id=job.get("thread_id"),
            turn_id=job.get("turn_id"),
            status=str(job.get("status") or "running"),
            final_response="",
            error=job.get("error"),
            job_id=job_id,
            wait_timed_out=bool(job.get("wait_timed_out")),
            session_resolution=(
                dict(job.get("session_resolution"))
                if isinstance(job.get("session_resolution"), dict)
                else None
            ),
        )

    def reconcile_jobs(self) -> list[dict[str, Any]]:
        reconciled: list[dict[str, Any]] = []
        for job in self.jobs.list():
            if job.get("status") not in ACTIVE_JOB_STATES:
                continue
            monitor_pid = job.get("monitor_pid")
            if monitor_pid and (
                int(monitor_pid) == os.getpid()
                or CodexSessionManager._pid_alive(monitor_pid)
            ):
                continue
            thread_id = job.get("thread_id")
            turn_id = job.get("turn_id")
            if not thread_id or not turn_id:
                reconciled.append(
                    self.jobs.update(
                        str(job["job_id"]),
                        status="failed",
                        error="client_restarted_before_turn_start",
                        result_available=True,
                        result={
                            "accepted": False,
                            "status": "failed",
                            "error": "client_restarted_before_turn_start",
                        },
                        monitor_pid=None,
                        monitor_token=None,
                    )
                )
                continue
            manager = self.manager_for(Path(str(job["project"])))
            try:
                snapshot = manager.read_turn(str(thread_id), str(turn_id))
            except Exception as exc:
                reconciled.append(
                    self.jobs.update(
                        str(job["job_id"]),
                        status="disconnected",
                        error=f"{type(exc).__name__}: {exc}",
                        monitor_pid=None,
                        monitor_token=None,
                    )
                )
                continue
            if snapshot["status"] == "running":
                reconciled.append(
                    self.jobs.update(str(job["job_id"]), status="reconnecting", error=None)
                )
                self._start_job_worker(job, task=None, resume=True)
                continue
            result = CodexResult(
                accepted=True,
                thread_id=str(thread_id),
                turn_id=str(turn_id),
                status=str(snapshot["status"]),
                final_response=str(snapshot.get("final_response") or ""),
                error=(None if snapshot.get("error") in {None, ""} else str(snapshot["error"])),
                human_interventions=tuple(manager.runtime.interventions_for(str(turn_id))),
                result_discarded=snapshot["status"] == "interrupted",
            )
            reconciled.append(self._finish_job(str(job["job_id"]), result))
        return reconciled

    def get_job_status(
        self,
        *,
        job_id: str | None = None,
        latest: bool = False,
    ) -> dict[str, Any]:
        self.reconcile_jobs()
        job = self.jobs.get(job_id, latest=latest or job_id is None)
        if job is None:
            return {"ok": False, "error": "codex_job_not_found"}
        started = datetime.fromisoformat(str(job["started_at"]))
        end_value = job.get("completed_at") or utc_now()
        ended = datetime.fromisoformat(str(end_value))
        return {
            "ok": True,
            "job_id": job.get("job_id"),
            "request_id": job.get("request_id"),
            "status": job.get("status"),
            "task_summary": job.get("task_summary"),
            "duration_seconds": max(0.0, (ended - started).total_seconds()),
            "thread_id": job.get("thread_id"),
            "turn_id": job.get("turn_id"),
            "session_resolution": job.get("session_resolution"),
            "human_interventions": len(job.get("human_interventions") or []),
            "result_available": bool(job.get("result_available")),
            "result_delivered": bool(job.get("result_delivered")),
            "wait_timed_out": bool(job.get("wait_timed_out")),
            "error": job.get("error"),
            "updated_at": job.get("updated_at"),
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        self.reconcile_jobs()
        return self.jobs.list()

    def list_sessions(self, *, project_path: str | None = None) -> list[dict[str, Any]]:
        project = self.policy.resolve(project_path) if project_path else None
        return self.sessions.list(project=project)

    def session_metrics(self) -> dict[str, Any]:
        jobs = self.jobs.list()
        delegated = [item for item in jobs if item.get("thread_id")]
        resolutions = [
            item.get("session_resolution")
            for item in delegated
            if isinstance(item.get("session_resolution"), dict)
        ]
        measured = [
            item
            for item in delegated
            if isinstance(item.get("session_resolution"), dict)
        ]
        correct = [
            item
            for item in measured
            if item["session_resolution"].get("thread_id") == item.get("thread_id")
        ]
        reused = [item for item in resolutions if item.get("reused")]
        unnecessary_created = [
            item
            for item in resolutions
            if item.get("created")
            and int(item.get("reusable_candidate_count") or 0) > 0
        ]
        recoverable = [item for item in resolutions if item.get("recoverable")]
        visible = [item for item in resolutions if item.get("visible")]
        ghosts = [
            item
            for item in delegated
            if not (
                (record := self.sessions.get(str(item.get("thread_id"))))
                and record.get("recoverable")
            )
        ]

        def rate(numerator: int, denominator: int) -> float:
            return round(numerator / denominator, 4) if denominator else 0.0

        return {
            "delegations": len(delegated),
            "measured_delegations": len(measured),
            "correct_session_rate": rate(len(correct), len(measured)),
            "existing_session_reuse_rate": rate(len(reused), len(resolutions)),
            "unnecessary_new_session_rate": rate(
                len(unnecessary_created), len(resolutions)
            ),
            "new_codex_session_created_when_reusable_session_exists": len(
                unnecessary_created
            ),
            "ghost_session_rate": rate(len(ghosts), len(delegated)),
            "ghost_sessions": len(ghosts),
            "session_recoverability_rate": rate(
                len(recoverable), len(resolutions)
            ),
            "session_visibility_rate": rate(len(visible), len(resolutions)),
            "session_resolution_success_rate": rate(
                len(
                    [
                        item
                        for item in resolutions
                        if item.get("status") == "RESOLVED"
                    ]
                ),
                len(resolutions),
            ),
            "historical_resolution_coverage_rate": rate(
                len(resolutions), len(jobs)
            ),
            "delegation_success_rate": rate(
                len([item for item in measured if item.get("status") == "completed"]),
                len(measured),
            ),
        }

    def cancel_job(self, *, job_id: str | None = None, latest: bool = False) -> dict[str, Any]:
        job = self.jobs.get(job_id, latest=latest or job_id is None, active_only=True)
        if job is None:
            return {"ok": False, "error": "active_codex_job_not_found"}
        if not job.get("turn_id"):
            return {"ok": False, "error": "codex_job_not_started", "job_id": job["job_id"]}
        self.jobs.update(str(job["job_id"]), status="cancelling")
        result = self.manager_for(Path(str(job["project"]))).cancel()
        if not result.get("cancelled"):
            return {"ok": False, "error": result.get("reason"), "job_id": job["job_id"]}
        self.jobs.update(
            str(job["job_id"]),
            status="interrupted",
            result_available=False,
            result_discarded=True,
            error="cancelled",
        )
        return {"ok": True, "job_id": job["job_id"], **result, "status": "interrupted"}

    def steer_job(
        self,
        instruction: str,
        *,
        job_id: str | None = None,
        latest: bool = False,
    ) -> dict[str, Any]:
        job = self.jobs.get(job_id, latest=latest or job_id is None, active_only=True)
        if job is None:
            return {"ok": False, "error": "active_codex_job_not_found"}
        if not job.get("turn_id"):
            return {"ok": False, "error": "codex_job_not_started", "job_id": job["job_id"]}
        self.jobs.update(str(job["job_id"]), status="steering")
        result = self.manager_for(Path(str(job["project"]))).steer(instruction)
        interventions = list(job.get("human_interventions") or [])
        interventions.append(
            {
                "timestamp": utc_now(),
                "source": "human",
                "summary": _safe_summary(instruction),
                "turn_id": job.get("turn_id"),
            }
        )
        self.jobs.update(
            str(job["job_id"]),
            status="running",
            human_interventions=interventions,
        )
        return {"ok": True, "job_id": job["job_id"], **result}

    def claim_completed_results(self) -> list[dict[str, Any]]:
        self.reconcile_jobs()
        return self.jobs.claim_results()

    def acknowledge_result(self, job_id: str, token: str) -> bool:
        return self.jobs.acknowledge_delivery(job_id, token)

    def release_result(self, job_id: str, token: str) -> None:
        self.jobs.release_delivery(job_id, token)

    def review_session(
        self,
        *,
        project_path: str,
        turn_limit: int = 10,
    ) -> dict[str, Any]:
        project = self.policy.resolve(project_path)
        if not project.is_dir():
            raise CodexError("project_path nao e diretorio", layer="validation")
        return self.manager_for(project).review_session(turn_limit=turn_limit)

    def delegate(self, task: dict[str, Any]) -> CodexResult:
        validate(task, CODEX_TASK_SCHEMA)
        prompt = json.dumps(task, ensure_ascii=False, indent=2)
        return self.delegate_to_codex(
            task=prompt,
            project_path=task["working_directory"],
            continue_current_thread=True,
        )

    def continue_session(
        self,
        *,
        session_id: str,
        working_directory: str,
        task: str,
    ) -> CodexResult:
        if not session_id.strip() or not task.strip():
            raise CodexError("session_id e task sao obrigatorios", layer="validation")
        return self.delegate_to_codex(
            task=task,
            project_path=working_directory,
            continue_current_thread=True,
            thread_id=session_id,
        )
