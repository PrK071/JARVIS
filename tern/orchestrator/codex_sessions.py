from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .codex_state import FileMutex, utc_now


REUSABLE_THREAD_STATES = frozenset({"idle", "notLoaded"})


def normalize_project_path(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve()))


@dataclass(frozen=True)
class CodexSessionResolution:
    status: str
    reason_code: str
    thread_id: str | None = None
    session_id: str | None = None
    binding_source: str | None = None
    state: str | None = None
    candidate_count: int = 0
    reusable_candidate_count: int = 0
    reused: bool = False
    created: bool = False
    registered: bool = False
    recoverable: bool = False
    visible: bool = False
    active_job_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "RESOLVED" and bool(self.thread_id)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodexSessionRegistry:
    """Canonical Jarvis registry for provider-native Codex thread identities."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir.resolve()
        self.path = self.state_dir / "codex-sessions.json"
        self.lock_path = self.state_dir / "codex-sessions.lock"

    def resolution_mutex(self, project: str | Path, *, timeout: float = 30) -> FileMutex:
        identity = normalize_project_path(project).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:16]
        return FileMutex(
            self.state_dir / f"codex-session-resolution-{digest}.lock",
            timeout=timeout,
        )

    def list(self, *, project: str | Path | None = None) -> list[dict[str, Any]]:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
        records = [dict(item) for item in state["sessions"] if isinstance(item, dict)]
        if project is None:
            return records
        normalized = normalize_project_path(project)
        return [item for item in records if item.get("project_key") == normalized]

    def get(self, thread_id: str) -> dict[str, Any] | None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            value = self._find_unlocked(state, thread_id)
            return dict(value) if value is not None else None

    def register(self, record: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(record.get("thread_id") or "").strip()
        project = str(record.get("project") or "").strip()
        if not thread_id or not project:
            raise ValueError("thread_id e project sao obrigatorios")
        now = utc_now()
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            existing = self._find_unlocked(state, thread_id)
            value = dict(existing or {})
            value.update(record)
            value.update(
                {
                    "thread_id": thread_id,
                    "session_id": str(record.get("session_id") or thread_id),
                    "project": str(Path(project).resolve()),
                    "project_key": normalize_project_path(project),
                    "created_at": value.get("created_at") or now,
                    "last_used_at": record.get("last_used_at") or now,
                    "updated_at": now,
                }
            )
            if existing is None:
                state["sessions"].append(value)
            else:
                existing.clear()
                existing.update(value)
            self._write_unlocked(state)
            return dict(value)

    def update(self, thread_id: str, **values: Any) -> dict[str, Any]:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            record = self._find_unlocked(state, thread_id)
            if record is None:
                raise KeyError(f"sessao Codex ausente: {thread_id}")
            record.update(values)
            record["updated_at"] = utc_now()
            self._write_unlocked(state)
            return dict(record)

    def bind_conversation(
        self,
        conversation_id: str,
        thread_id: str,
        project: str | Path,
    ) -> None:
        if not conversation_id:
            return
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            state["conversation_bindings"][conversation_id] = {
                "thread_id": thread_id,
                "project_key": normalize_project_path(project),
                "updated_at": utc_now(),
            }
            self._write_unlocked(state)

    def conversation_binding(
        self,
        conversation_id: str | None,
        project: str | Path,
    ) -> str | None:
        if not conversation_id:
            return None
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            value = state["conversation_bindings"].get(conversation_id)
        if not isinstance(value, dict):
            return None
        if value.get("project_key") != normalize_project_path(project):
            return None
        thread_id = value.get("thread_id")
        return str(thread_id) if isinstance(thread_id, str) and thread_id else None

    def bind_project(self, project: str | Path, thread_id: str) -> None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            state["project_bindings"][normalize_project_path(project)] = {
                "thread_id": thread_id,
                "updated_at": utc_now(),
            }
            self._write_unlocked(state)

    def project_binding(self, project: str | Path) -> str | None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            value = state["project_bindings"].get(normalize_project_path(project))
        thread_id = value.get("thread_id") if isinstance(value, dict) else None
        return str(thread_id) if isinstance(thread_id, str) and thread_id else None

    def reconcile(
        self,
        project: str | Path,
        provider_threads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        project_path = str(Path(project).resolve())
        project_key = normalize_project_path(project)
        observed: set[str] = set()
        now = utc_now()
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            for provider in provider_threads:
                thread_id = str(provider.get("thread_id") or provider.get("id") or "").strip()
                if not thread_id:
                    continue
                observed.add(thread_id)
                record = self._find_unlocked(state, thread_id)
                if record is None:
                    record = {
                        "thread_id": thread_id,
                        "created_at": provider.get("created_at") or now,
                        "origin": "user_existing",
                    }
                    state["sessions"].append(record)
                record.update(
                    {
                        "session_id": str(provider.get("session_id") or thread_id),
                        "project": project_path,
                        "project_key": project_key,
                        "endpoint": provider.get("endpoint"),
                        "state": provider.get("state") or "unknown",
                        "source": provider.get("source") or "unknown",
                        "visible": bool(provider.get("visible")),
                        "recoverable": bool(provider.get("recoverable", True)),
                        "ephemeral": bool(provider.get("ephemeral", False)),
                        "last_used_at": provider.get("updated_at") or now,
                        "updated_at": now,
                    }
                )
            for record in state["sessions"]:
                if (
                    isinstance(record, dict)
                    and record.get("project_key") == project_key
                    and record.get("thread_id") not in observed
                ):
                    record.update(
                        {
                            "state": "stale",
                            "recoverable": False,
                            "visible": False,
                            "updated_at": now,
                        }
                    )
            self._write_unlocked(state)
            return [
                dict(item)
                for item in state["sessions"]
                if isinstance(item, dict) and item.get("project_key") == project_key
            ]

    def import_legacy(
        self,
        *,
        session: dict[str, Any] | None,
        jobs: list[dict[str, Any]],
    ) -> None:
        candidates: list[tuple[str, str, str, str | None]] = []
        if isinstance(session, dict):
            thread_id = session.get("thread_id")
            project = session.get("project")
            if isinstance(thread_id, str) and isinstance(project, str):
                candidates.append(
                    (thread_id, project, "user_existing", session.get("updated_at"))
                )
        for job in jobs:
            thread_id = job.get("thread_id")
            project = job.get("project")
            if isinstance(thread_id, str) and thread_id and isinstance(project, str):
                candidates.append(
                    (thread_id, project, "jarvis_created", job.get("updated_at"))
                )
        for thread_id, project, origin, last_used in candidates:
            existing = self.get(thread_id)
            self.register(
                {
                    **(existing or {}),
                    "thread_id": thread_id,
                    "session_id": thread_id,
                    "project": project,
                    "origin": (existing or {}).get("origin") or origin,
                    "state": (existing or {}).get("state") or "unknown",
                    "recoverable": bool((existing or {}).get("recoverable", True)),
                    "visible": bool((existing or {}).get("visible", False)),
                    "last_used_at": last_used or utc_now(),
                }
            )
        latest_by_project: dict[str, tuple[str, str]] = {}
        for thread_id, project, _origin, last_used in candidates:
            key = normalize_project_path(project)
            value = str(last_used or "")
            if key not in latest_by_project or value >= latest_by_project[key][0]:
                latest_by_project[key] = (value, thread_id)
        for key, (_updated, thread_id) in latest_by_project.items():
            with FileMutex(self.lock_path):
                state = self._read_unlocked()
                state["project_bindings"][key] = {
                    "thread_id": thread_id,
                    "updated_at": utc_now(),
                }
                self._write_unlocked(state)

    @staticmethod
    def _find_unlocked(
        state: dict[str, Any], thread_id: str
    ) -> dict[str, Any] | None:
        for record in state["sessions"]:
            if isinstance(record, dict) and record.get("thread_id") == thread_id:
                return record
        return None

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        state.setdefault("version", 1)
        state.setdefault("sessions", [])
        state.setdefault("conversation_bindings", {})
        state.setdefault("project_bindings", {})
        return state

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = utc_now()
        temporary = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class CodexSessionResolver:
    def __init__(self, registry: CodexSessionRegistry):
        self.registry = registry

    @staticmethod
    def _selectable(record: dict[str, Any]) -> bool:
        return (
            bool(record.get("recoverable"))
            and not bool(record.get("ephemeral"))
            and record.get("state") in REUSABLE_THREAD_STATES
        )

    def resolve(
        self,
        *,
        project: str | Path,
        candidates: list[dict[str, Any]],
        explicit_thread_id: str | None = None,
        focused_thread_id: str | None = None,
        conversation_id: str | None = None,
        force_new: bool = False,
    ) -> CodexSessionResolution:
        project_key = normalize_project_path(project)
        compatible = [
            item for item in candidates if item.get("project_key") == project_key
        ]
        by_id = {
            str(item.get("thread_id")): item
            for item in compatible
            if item.get("thread_id")
        }
        reusable_candidate_count = sum(
            1 for item in compatible if self._selectable(item)
        )

        def selected(thread_id: str | None, source: str) -> CodexSessionResolution | None:
            if not thread_id:
                return None
            record = by_id.get(thread_id)
            if record is None:
                if source == "explicit_session":
                    return CodexSessionResolution(
                        "UNAVAILABLE",
                        "SESSION_UNAVAILABLE",
                        thread_id=thread_id,
                        binding_source=source,
                        candidate_count=len(compatible),
                        reusable_candidate_count=reusable_candidate_count,
                    )
                return None
            state = str(record.get("state") or "unknown")
            if state == "active":
                active_job_id = record.get("active_job_id")
                if active_job_id:
                    return CodexSessionResolution(
                        "RESOLVED",
                        "SESSION_BUSY_QUEUED",
                        thread_id=thread_id,
                        session_id=str(record.get("session_id") or thread_id),
                        binding_source=source,
                        state=state,
                        candidate_count=len(compatible),
                        reusable_candidate_count=reusable_candidate_count,
                        reused=True,
                        registered=True,
                        recoverable=True,
                        visible=bool(record.get("visible")),
                        active_job_id=str(active_job_id),
                    )
                return CodexSessionResolution(
                    "UNAVAILABLE",
                    "SESSION_BUSY",
                    thread_id=thread_id,
                    session_id=str(record.get("session_id") or thread_id),
                    binding_source=source,
                    state=state,
                    candidate_count=len(compatible),
                    reusable_candidate_count=reusable_candidate_count,
                    registered=True,
                    recoverable=True,
                    visible=bool(record.get("visible")),
                )
            if not self._selectable(record):
                if source != "explicit_session":
                    return None
                return CodexSessionResolution(
                    "UNAVAILABLE",
                    "SESSION_STALE",
                    thread_id=thread_id,
                    session_id=str(record.get("session_id") or thread_id),
                    binding_source=source,
                    state=state,
                    candidate_count=len(compatible),
                    reusable_candidate_count=reusable_candidate_count,
                    registered=True,
                    recoverable=bool(record.get("recoverable")),
                    visible=bool(record.get("visible")),
                )
            return CodexSessionResolution(
                "RESOLVED",
                {
                    "explicit_session": "EXPLICIT_SESSION_MATCH",
                    "focused_ui_session": "FOCUSED_SESSION_MATCH",
                    "conversation_affinity": "CONVERSATION_AFFINITY_MATCH",
                    "project_affinity": "PROJECT_AFFINITY_MATCH",
                    "unique_project_session": "UNIQUE_PROJECT_SESSION_MATCH",
                }[source],
                thread_id=thread_id,
                session_id=str(record.get("session_id") or thread_id),
                binding_source=source,
                state=state,
                candidate_count=len(compatible),
                reusable_candidate_count=reusable_candidate_count,
                reused=True,
                registered=True,
                recoverable=True,
                visible=bool(record.get("visible")),
                active_job_id=record.get("active_job_id"),
            )

        if force_new:
            return CodexSessionResolution(
                "NONE",
                "NO_REUSABLE_SESSION",
                candidate_count=len(compatible),
                reusable_candidate_count=reusable_candidate_count,
            )
        for thread_id, source in (
            (explicit_thread_id, "explicit_session"),
            (focused_thread_id, "focused_ui_session"),
            (
                self.registry.conversation_binding(conversation_id, project),
                "conversation_affinity",
            ),
            (self.registry.project_binding(project), "project_affinity"),
        ):
            result = selected(thread_id, source)
            if result is not None:
                return result
        reusable = [item for item in compatible if self._selectable(item)]
        if len(reusable) == 1:
            return selected(
                str(reusable[0]["thread_id"]), "unique_project_session"
            ) or CodexSessionResolution("NONE", "NO_REUSABLE_SESSION")
        if len(reusable) > 1:
            return CodexSessionResolution(
                "AMBIGUOUS",
                "AMBIGUOUS_SESSION",
                candidate_count=len(reusable),
                reusable_candidate_count=len(reusable),
            )
        busy = [item for item in compatible if item.get("state") == "active"]
        if busy:
            return CodexSessionResolution(
                "UNAVAILABLE",
                "SESSION_BUSY",
                candidate_count=len(busy),
                reusable_candidate_count=0,
            )
        return CodexSessionResolution(
            "NONE",
            "NO_REUSABLE_SESSION",
            candidate_count=len(compatible),
            reusable_candidate_count=0,
        )
