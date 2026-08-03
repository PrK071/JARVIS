from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .codex_state import FileMutex, utc_now


ACTIVE_JOB_STATES = frozenset(
    {
        "queued",
        "starting",
        "running",
        "steering",
        "cancelling",
        "disconnected",
        "reconnecting",
    }
)
TERMINAL_JOB_STATES = frozenset({"interrupted", "completed", "failed"})


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _pid_alive(value: Any) -> bool:
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


class CodexJobStore:
    """Atomic persistent registry and delivery queue for Codex turns."""

    def __init__(self, state_dir: Path, *, retention_days: int = 7):
        self.path = state_dir / "codex-jobs.json"
        self.lock_path = state_dir / "codex-jobs.lock"
        self.retention_days = retention_days

    def create(
        self,
        *,
        project: str,
        task_summary: str,
        source: str,
        wait: bool,
    ) -> dict[str, Any]:
        now = utc_now()
        job = {
            "job_id": str(uuid.uuid4()),
            "thread_id": None,
            "turn_id": None,
            "project": project,
            "task_summary": task_summary,
            "source": source,
            "status": "queued",
            "wait": wait,
            "wait_timed_out": False,
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "result_available": False,
            "result_delivered": False,
            "result": None,
            "error": None,
            "human_interventions": [],
            "result_discarded": False,
            "monitor_pid": None,
            "monitor_token": None,
            "delivery_claim": None,
            "completion_notified": False,
            "failure_notified": False,
            "progress_notified_at": None,
        }
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            self._prune_unlocked(state)
            state["jobs"].append(job)
            self._write_unlocked(state)
        return dict(job)

    def update(self, job_id: str, **values: Any) -> dict[str, Any]:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            job = self._find_unlocked(state, job_id)
            if job is None:
                raise KeyError(f"job Codex ausente: {job_id}")
            job.update(values)
            job["updated_at"] = utc_now()
            if job.get("status") in TERMINAL_JOB_STATES and not job.get(
                "completed_at"
            ):
                job["completed_at"] = job["updated_at"]
            self._write_unlocked(state)
            return dict(job)

    def get(
        self,
        job_id: str | None = None,
        *,
        latest: bool = False,
        active_only: bool = False,
    ) -> dict[str, Any] | None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            changed = self._prune_unlocked(state)
            jobs = [item for item in state["jobs"] if isinstance(item, dict)]
            if active_only:
                jobs = [item for item in jobs if item.get("status") in ACTIVE_JOB_STATES]
            value = self._find_unlocked(state, job_id) if job_id else None
            if value is not None and active_only and value.get("status") not in ACTIVE_JOB_STATES:
                value = None
            if value is None and (latest or job_id is None) and jobs:
                value = jobs[-1]
            if changed:
                self._write_unlocked(state)
            return dict(value) if isinstance(value, dict) else None

    def list(self) -> list[dict[str, Any]]:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            if self._prune_unlocked(state):
                self._write_unlocked(state)
            return [dict(item) for item in state["jobs"] if isinstance(item, dict)]

    def claim_monitor(self, job_id: str, token: str) -> bool:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            job = self._find_unlocked(state, job_id)
            if job is None or job.get("status") not in ACTIVE_JOB_STATES:
                return False
            owner = job.get("monitor_pid")
            existing = job.get("monitor_token")
            if existing and existing != token and (
                int(owner or 0) == os.getpid() or _pid_alive(owner)
            ):
                return False
            job["monitor_pid"] = os.getpid()
            job["monitor_token"] = token
            job["updated_at"] = utc_now()
            self._write_unlocked(state)
            return True

    def claim_results(self, *, limit: int = 10) -> list[dict[str, Any]]:
        token = str(uuid.uuid4())
        claimed: list[dict[str, Any]] = []
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            for job in state["jobs"]:
                if len(claimed) >= limit or not isinstance(job, dict):
                    break
                if not job.get("result_available") or job.get("result_delivered"):
                    continue
                previous = job.get("delivery_claim")
                if isinstance(previous, dict) and _pid_alive(previous.get("pid")):
                    claimed_at = _parse_time(previous.get("claimed_at"))
                    if (
                        claimed_at is not None
                        and (datetime.now(timezone.utc) - claimed_at).total_seconds() < 60
                    ):
                        continue
                job["delivery_claim"] = {
                    "token": token,
                    "pid": os.getpid(),
                    "claimed_at": utc_now(),
                }
                value = dict(job)
                value["delivery_token"] = token
                claimed.append(value)
            if claimed:
                self._write_unlocked(state)
        return claimed

    def acknowledge_delivery(self, job_id: str, token: str) -> bool:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            job = self._find_unlocked(state, job_id)
            claim = job.get("delivery_claim") if isinstance(job, dict) else None
            if not isinstance(claim, dict) or claim.get("token") != token:
                return False
            job["result_delivered"] = True
            job["delivery_claim"] = None
            job["updated_at"] = utc_now()
            self._write_unlocked(state)
            return True

    def release_delivery(self, job_id: str, token: str) -> None:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            job = self._find_unlocked(state, job_id)
            claim = job.get("delivery_claim") if isinstance(job, dict) else None
            if isinstance(claim, dict) and claim.get("token") == token:
                job["delivery_claim"] = None
                job["updated_at"] = utc_now()
                self._write_unlocked(state)

    def claim_notification(self, job_id: str, kind: str) -> bool:
        field = {
            "completed": "completion_notified",
            "failed": "failure_notified",
        }.get(kind)
        if field is None:
            raise ValueError("tipo de notificacao Codex invalido")
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            job = self._find_unlocked(state, job_id)
            if job is None or job.get(field):
                return False
            job[field] = True
            job["updated_at"] = utc_now()
            self._write_unlocked(state)
            return True

    def mark_progress_notified(self, job_id: str, *, interval_seconds: int = 60) -> bool:
        with FileMutex(self.lock_path):
            state = self._read_unlocked()
            job = self._find_unlocked(state, job_id)
            if job is None:
                return False
            previous = _parse_time(job.get("progress_notified_at"))
            now = datetime.now(timezone.utc)
            if previous and (now - previous).total_seconds() < interval_seconds:
                return False
            job["progress_notified_at"] = now.isoformat()
            job["updated_at"] = job["progress_notified_at"]
            self._write_unlocked(state)
            return True

    def _prune_unlocked(self, state: dict[str, Any]) -> bool:
        if self.retention_days <= 0:
            return False
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        before = len(state["jobs"])
        state["jobs"] = [
            job
            for job in state["jobs"]
            if not isinstance(job, dict)
            or job.get("status") in ACTIVE_JOB_STATES
            or (_parse_time(job.get("completed_at") or job.get("updated_at")) or cutoff)
            >= cutoff
        ]
        return len(state["jobs"]) != before

    @staticmethod
    def _find_unlocked(
        state: dict[str, Any], job_id: str | None
    ) -> dict[str, Any] | None:
        if not job_id:
            return None
        for job in state["jobs"]:
            if isinstance(job, dict) and job.get("job_id") == job_id:
                return job
        return None

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {"version": 1, "jobs": []}
        if not isinstance(state, dict):
            state = {"version": 1, "jobs": []}
        state.setdefault("version", 1)
        state.setdefault("jobs", [])
        return state

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = utc_now()
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
