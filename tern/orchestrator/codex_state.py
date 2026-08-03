from __future__ import annotations

import json
import os
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FileMutex(AbstractContextManager["FileMutex"]):
    """Small cross-process mutex backed by one locked byte."""

    def __init__(self, path: Path, *, timeout: float = 30):
        self.path = path
        self.timeout = timeout
        self._handle: Any = None

    def __enter__(self) -> "FileMutex":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._lock()
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError(f"timeout ao obter lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *_args: object) -> None:
        if self._handle is None:
            return
        try:
            self._unlock()
        finally:
            self._handle.close()
            self._handle = None

    def _lock(self) -> None:
        assert self._handle is not None
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(self) -> None:
        assert self._handle is not None
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)


class SharedCodexState:
    def __init__(self, state_dir: Path, project: Path, endpoint: str):
        self.state_dir = state_dir
        self.project = project
        self.endpoint = endpoint
        self.path = state_dir / "codex-runtime.json"
        self.lock_path = state_dir / "codex-runtime.lock"
        self.turn_lock_path = state_dir / "codex-turn.lock"
        self.control_lock_path = state_dir / "codex-control.lock"

    def default(self) -> dict[str, Any]:
        return {
            "project": str(self.project),
            "endpoint": self.endpoint,
            "thread_id": None,
            "turn_id": None,
            "state": "idle",
            "last_terminal_state": None,
            "last_instruction_source": None,
            "queue_length": 0,
            "queue_epoch": 0,
            "last_event_at": None,
            "bridge_connected": False,
            "bridge_pid": None,
            "qwen_connected": False,
            "qwen_pid": None,
            "active_client_message_id": None,
            "cancelled_turn_ids": [],
            "interventions": [],
            "state_events": [],
            "result_discarded": False,
            "updated_at": utc_now(),
        }

    def read(self) -> dict[str, Any]:
        with FileMutex(self.lock_path):
            return self._read_unlocked()

    def update(self, **values: Any) -> dict[str, Any]:
        return self.mutate(lambda state: state.update(values))

    def mutate(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            callback(state)
            state["project"] = str(self.project)
            state["endpoint"] = self.endpoint
            state["updated_at"] = utc_now()
            self._write_unlocked(state)
            return state

    def append_state_event(
        self,
        name: str,
        *,
        source: str,
        thread_id: str | None,
        turn_id: str | None,
        state_result: str,
        summary: str = "",
    ) -> dict[str, Any]:
        event = {
            "timestamp": utc_now(),
            "name": name,
            "source": source,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "state": state_result,
            "summary": summary[:500],
        }

        def add(state: dict[str, Any]) -> None:
            events = list(state.get("state_events") or [])
            events.append(event)
            state["state_events"] = events[-100:]
            state["last_event_at"] = event["timestamp"]

        return self.mutate(add)

    def append_intervention(self, intervention: dict[str, Any]) -> dict[str, Any]:
        def add(state: dict[str, Any]) -> None:
            values = list(state.get("interventions") or [])
            identity = (
                intervention.get("turn_id"),
                intervention.get("client_message_id"),
                intervention.get("summary"),
            )
            if not any(
                (
                    item.get("turn_id"),
                    item.get("client_message_id"),
                    item.get("summary"),
                )
                == identity
                for item in values
            ):
                values.append(intervention)
            state["interventions"] = values[-100:]
            state["last_instruction_source"] = intervention.get("source")
            state["last_event_at"] = intervention.get("timestamp") or utc_now()

        return self.mutate(add)

    def interventions_for(self, turn_id: str) -> list[dict[str, Any]]:
        state = self.read()
        return [
            item
            for item in state.get("interventions") or []
            if item.get("turn_id") == turn_id
        ]

    def state_events_for(self, turn_id: str) -> list[dict[str, Any]]:
        state = self.read()
        return [
            item
            for item in state.get("state_events") or []
            if item.get("turn_id") == turn_id
        ]

    def turn_mutex(self, *, timeout: float) -> FileMutex:
        return FileMutex(self.turn_lock_path, timeout=timeout)

    def control_mutex(self, *, timeout: float = 30) -> FileMutex:
        return FileMutex(self.control_lock_path, timeout=timeout)

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.default()
        if not isinstance(value, dict):
            return self.default()
        default = self.default()
        default.update(value)
        return default

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
