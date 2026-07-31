from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompt import CODEX_TASK_SCHEMA
from .schema import validate
from .security import PathPolicy


class CodexError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodexResult:
    ok: bool
    session_id: str | None
    message: str
    events: int
    exit_code: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "session_id": self.session_id,
            "message": self.message,
            "events": self.events,
            "exit_code": self.exit_code,
        }


class CodexRunner:
    def __init__(self, policy: PathPolicy, timeout: int = 1800, executable: str | None = None):
        self.policy = policy
        self.timeout = timeout
        self.executable = executable or shutil.which("codex") or "codex"

    def delegate(self, task: dict[str, Any]) -> CodexResult:
        validate(task, CODEX_TASK_SCHEMA)
        working_directory = self.policy.resolve(task["working_directory"])
        if not working_directory.is_dir():
            raise CodexError("working_directory nao e diretorio")
        prompt = json.dumps(task, ensure_ascii=False, indent=2)
        command = [
            self.executable,
            "exec",
            "--json",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "-C",
            str(working_directory),
            prompt,
        ]
        return self._run(command, working_directory)

    def continue_session(
        self,
        *,
        session_id: str,
        working_directory: str,
        task: str,
    ) -> CodexResult:
        if not session_id.strip() or not task.strip():
            raise CodexError("session_id e task sao obrigatorios")
        directory = self.policy.resolve(working_directory)
        command = [
            self.executable,
            "exec",
            "resume",
            "--json",
            session_id,
            task,
        ]
        return self._run(command, directory)

    def _run(self, command: list[str], cwd: Path) -> CodexResult:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.run(
                command,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                creationflags=creationflags,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexError(f"Codex excedeu timeout de {self.timeout}s") from exc
        except OSError as exc:
            raise CodexError(f"falha ao iniciar Codex: {exc}") from exc

        events: list[dict[str, Any]] = []
        session_id = None
        messages: list[str] = []
        for line in process.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if event.get("type") == "thread.started":
                session_id = event.get("thread_id")
            item = event.get("item")
            if event.get("type") == "item.completed" and isinstance(item, dict):
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    messages.append(item["text"])
        message = messages[-1] if messages else process.stderr.strip()[-8192:]
        return CodexResult(
            ok=process.returncode == 0,
            session_id=session_id,
            message=message,
            events=len(events),
            exit_code=process.returncode,
        )
