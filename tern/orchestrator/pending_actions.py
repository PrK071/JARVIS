from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .codex_state import FileMutex, utc_now


class PendingActionStore:
    """Persistent single-flight state for tool confirmation and execution."""

    def __init__(self, state_dir: Path, *, timeout_seconds: int = 300):
        self.path = state_dir / "pending-actions.json"
        self.lock_path = state_dir / "pending-actions.lock"
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def fingerprints(
        tool: str,
        arguments: dict[str, Any],
        *,
        project: str | None,
        risk: str,
        turn_id: str,
    ) -> tuple[str, str]:
        operation = {
            "tool": tool,
            "arguments": arguments,
            "project": project,
            "risk": risk,
        }
        canonical = json.dumps(
            operation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        operation_fingerprint = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        request_fingerprint = hashlib.sha256(
            f"{canonical}\nturn={turn_id}".encode("utf-8")
        ).hexdigest()
        return request_fingerprint, operation_fingerprint

    def prepare(
        self,
        *,
        action_id: str,
        tool: str,
        arguments: dict[str, Any],
        project: str | None,
        risk: str,
        confirmation_required: bool,
        turn_id: str,
    ) -> tuple[dict[str, Any], str]:
        request_fingerprint, operation_fingerprint = self.fingerprints(
            tool,
            arguments,
            project=project,
            risk=risk,
            turn_id=turn_id,
        )
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            self._expire_unlocked(state)
            pending = state.get("pending")
            if isinstance(pending, dict):
                if pending.get("operation_fingerprint") == operation_fingerprint:
                    return pending, "duplicate_pending"
                return pending, "another_action_pending"
            for item in reversed(state.get("history") or []):
                if (
                    isinstance(item, dict)
                    and item.get("request_fingerprint") == request_fingerprint
                    and item.get("status")
                    in {"executing", "completed", "failed", "cancelled"}
                ):
                    return item, "duplicate_same_turn"
            now = utc_now()
            record = {
                "action_id": action_id,
                "tool": tool,
                "arguments": arguments,
                "project": project,
                "risk": risk,
                "confirmation_required": confirmation_required,
                "status": (
                    "awaiting_confirmation"
                    if confirmation_required
                    else "prepared"
                ),
                "created_at": now,
                "updated_at": now,
                "expires_at": (
                    datetime.now(timezone.utc)
                    + timedelta(seconds=self.timeout_seconds)
                ).isoformat(),
                "request_fingerprint": request_fingerprint,
                "operation_fingerprint": operation_fingerprint,
                "turn_id": turn_id,
                "owner_pid": os.getpid(),
                "confirmation_presented": False,
                "result": None,
                "error": None,
            }
            state["pending"] = record
            self._write_unlocked(state)
            return record, "created"

    def pending(self) -> dict[str, Any] | None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            changed = self._expire_unlocked(state)
            if changed:
                self._write_unlocked(state)
            value = state.get("pending")
            return dict(value) if isinstance(value, dict) else None

    def mark_presented(self, action_id: str) -> dict[str, Any]:
        return self._update_active(
            action_id,
            status="awaiting_confirmation",
            confirmation_presented=True,
        )

    def claim_execution(self, action_id: str) -> tuple[dict[str, Any], bool]:
        """Atomically claim an action so confirmations cannot execute it twice."""
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            self._expire_unlocked(state)
            pending = state.get("pending")
            if not isinstance(pending, dict) or pending.get("action_id") != action_id:
                if state.get("pending") is None:
                    self._write_unlocked(state)
                for item in reversed(state.get("history") or []):
                    if isinstance(item, dict) and item.get("action_id") == action_id:
                        return dict(item), False
                raise KeyError(f"acao pendente ausente: {action_id}")
            if pending.get("status") == "executing":
                return dict(pending), False
            if pending.get("status") not in {
                "prepared",
                "awaiting_confirmation",
            }:
                return dict(pending), False
            pending["status"] = "executing"
            pending["updated_at"] = utc_now()
            self._write_unlocked(state)
            return dict(pending), True

    def complete(
        self,
        action_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed", "cancelled", "expired"}:
            raise ValueError("estado terminal invalido")
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            pending = state.get("pending")
            if not isinstance(pending, dict) or pending.get("action_id") != action_id:
                for item in reversed(state.get("history") or []):
                    if isinstance(item, dict) and item.get("action_id") == action_id:
                        return dict(item)
                raise KeyError(f"acao pendente ausente: {action_id}")
            pending["status"] = status
            pending["updated_at"] = utc_now()
            pending["result"] = result
            pending["error"] = error
            history = list(state.get("history") or [])
            history.append(pending)
            state["history"] = history[-100:]
            state["pending"] = None
            self._write_unlocked(state)
            return dict(pending)

    def history(self) -> list[dict[str, Any]]:
        with FileMutex(self.lock_path):
            return [
                dict(item)
                for item in self._read_unlocked().get("history") or []
                if isinstance(item, dict)
            ]

    def _update_active(
        self,
        action_id: str,
        *,
        status: str,
        confirmation_presented: bool | None = None,
    ) -> dict[str, Any]:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            pending = state.get("pending")
            if not isinstance(pending, dict) or pending.get("action_id") != action_id:
                raise KeyError(f"acao pendente ausente: {action_id}")
            pending["status"] = status
            pending["updated_at"] = utc_now()
            if confirmation_presented is not None:
                pending["confirmation_presented"] = confirmation_presented
            self._write_unlocked(state)
            return dict(pending)

    def _expire_unlocked(self, state: dict[str, Any]) -> bool:
        pending = state.get("pending")
        if not isinstance(pending, dict):
            return False
        try:
            expires_at = datetime.fromisoformat(str(pending["expires_at"]))
        except (KeyError, ValueError):
            expires_at = datetime.now(timezone.utc)
        if datetime.now(timezone.utc) < expires_at:
            return False
        status = str(pending.get("status") or "")
        if status == "executing" and self._process_is_alive(
            pending.get("owner_pid")
        ):
            return False
        if status not in {"awaiting_confirmation", "prepared", "executing"}:
            return False
        pending["status"] = (
            "expired" if status == "awaiting_confirmation" else "failed"
        )
        pending["updated_at"] = utc_now()
        if status == "prepared":
            pending["error"] = "prepared_action_expired"
        elif status == "executing":
            pending["error"] = "orphaned_execution"
        history = list(state.get("history") or [])
        history.append(pending)
        state["history"] = history[-100:]
        state["pending"] = None
        return True

    @staticmethod
    def _process_is_alive(value: Any) -> bool:
        try:
            pid = int(value)
            if pid <= 0:
                return False
            if pid == os.getpid():
                return True
            os.kill(pid, 0)
        except (TypeError, ValueError, ProcessLookupError):
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"pending": None, "history": []}
        if not isinstance(value, dict):
            return {"pending": None, "history": []}
        value.setdefault("pending", None)
        value.setdefault("history", [])
        return value

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
