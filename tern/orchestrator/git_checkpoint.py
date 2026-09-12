from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence


DEFAULT_MESSAGE = "chore: checkpoint before session limit"
DEFAULT_REPOSITORY = "github.com/PrK071/JARVIS"
MAX_OUTPUT_CHARS = 2_000

GitRun = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
Sleep = Callable[[float], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class CheckpointPlan:
    root: Path
    remote: str
    branch: str
    message: str
    paths: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "remote": self.remote,
            "branch": self.branch,
            "message": self.message,
            "paths": list(self.paths),
        }


class GitCheckpointService:
    """Creates a narrow, explicit git checkpoint without changing git policy."""

    def __init__(
        self,
        root: str | Path,
        *,
        remote: str = "origin",
        branch: str = "main",
        expected_repository: str = DEFAULT_REPOSITORY,
        runner: GitRun | None = None,
        sleeper: Sleep = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.remote = remote
        self.branch = branch
        self.expected_repository = expected_repository
        self._runner = runner or self._default_run
        self._sleeper = sleeper
        self._clock = clock

    @staticmethod
    def _default_run(
        command: Sequence[str], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    @staticmethod
    def _brief(value: str) -> str:
        compact = value.strip()
        return compact[:MAX_OUTPUT_CHARS]

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self._runner(("git", "-C", str(self.root), *args), self.root)

    @staticmethod
    def _normalize_repository(value: str) -> str:
        normalized = value.strip().casefold().replace("\\", "/")
        for prefix in ("https://", "http://", "ssh://", "git@"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        normalized = normalized.replace(":", "/", 1)
        return normalized.rstrip("/").removesuffix(".git")

    @classmethod
    def _is_protected(cls, path: str) -> bool:
        normalized = path.replace("\\", "/").lstrip("/")
        parts = PurePosixPath(normalized).parts
        if not parts or ".." in parts:
            return True
        first = parts[0].casefold()
        name = parts[-1].casefold()
        if first in {".git", "_arquivo", "runtime", "models"}:
            return True
        if name == ".env" or name.startswith(".env.") or name.endswith(".gguf"):
            return True
        return normalized.casefold() == "interface/providers.json"

    def _error(self, code: str, **details: object) -> dict[str, object]:
        return {"ok": False, "error": code, "root": str(self.root), **details}

    def _check_result(
        self, result: subprocess.CompletedProcess[str], code: str
    ) -> dict[str, object] | None:
        if result.returncode == 0:
            return None
        return self._error(
            code,
            returncode=result.returncode,
            stdout=self._brief(result.stdout or ""),
            stderr=self._brief(result.stderr or ""),
        )

    def plan(self, *, message: str = DEFAULT_MESSAGE) -> CheckpointPlan | dict[str, object]:
        if not self.root.is_dir():
            return self._error("project_not_found")
        root_result = self._run("rev-parse", "--show-toplevel")
        error = self._check_result(root_result, "not_a_git_repository")
        if error:
            return error
        git_root = Path(root_result.stdout.strip()).resolve()
        if git_root != self.root:
            return self._error("project_root_mismatch", git_root=str(git_root))
        branch_result = self._run("branch", "--show-current")
        error = self._check_result(branch_result, "git_branch_unavailable")
        if error:
            return error
        current_branch = branch_result.stdout.strip()
        if current_branch != self.branch:
            return self._error(
                "unexpected_branch", branch=current_branch or None, expected=self.branch
            )
        remote_result = self._run("remote", "get-url", self.remote)
        error = self._check_result(remote_result, "remote_not_found")
        if error:
            return error
        remote_url = remote_result.stdout.strip()
        if self._normalize_repository(remote_url) != self._normalize_repository(
            self.expected_repository
        ):
            return self._error(
                "unexpected_remote", remote=self.remote, remote_url=remote_url,
                expected_repository=self.expected_repository,
            )
        staged = self._run("diff", "--cached", "--name-only", "-z")
        error = self._check_result(staged, "git_status_failed")
        if error:
            return error
        staged_paths = tuple(item for item in staged.stdout.split("\0") if item)
        if staged_paths:
            return self._error("preexisting_staged_changes", paths=list(staged_paths))
        tracked = self._run("diff", "--name-only", "-z", "HEAD", "--")
        error = self._check_result(tracked, "git_status_failed")
        if error:
            return error
        untracked = self._run("ls-files", "--others", "--exclude-standard", "-z")
        error = self._check_result(untracked, "git_status_failed")
        if error:
            return error
        paths = tuple(sorted({
            *(item for item in tracked.stdout.split("\0") if item),
            *(item for item in untracked.stdout.split("\0") if item),
        }))
        protected = [path for path in paths if self._is_protected(path)]
        if protected:
            return self._error("protected_path_detected", paths=protected)
        return CheckpointPlan(
            root=self.root,
            remote=self.remote,
            branch=self.branch,
            message=message.strip() or DEFAULT_MESSAGE,
            paths=paths,
        )

    def commit_and_push(
        self, *, message: str = DEFAULT_MESSAGE, run_tests: bool = False,
        dry_run: bool = False,
    ) -> dict[str, object]:
        plan = self.plan(message=message)
        if isinstance(plan, dict):
            return plan
        value = {"ok": True, "plan": plan.as_dict(), "committed": False, "pushed": False}
        if not plan.paths:
            return {**value, "status": "clean"}
        check = self._run("diff", "--check")
        error = self._check_result(check, "diff_check_failed")
        if error:
            return error
        if run_tests:
            tests = self._runner((sys.executable, "-m", "pytest", "-q"), self.root)
            error = self._check_result(tests, "tests_failed")
            if error:
                return error
        if dry_run:
            return {**value, "status": "dry_run"}
        add = self._run("add", "--", *plan.paths)
        error = self._check_result(add, "git_add_failed")
        if error:
            return error
        commit = self._run("commit", "-m", plan.message)
        error = self._check_result(commit, "git_commit_failed")
        if error:
            return {**error, "staged": True}
        value = {**value, "committed": True, "commit_output": self._brief(commit.stdout or "")}
        push = self._run("push", plan.remote, plan.branch)
        error = self._check_result(push, "git_push_failed")
        if error:
            return {**error, "committed": True, "pushed": False}
        return {**value, "pushed": True, "status": "checkpoint_pushed"}

    def watch_and_checkpoint(
        self, *, session_seconds: int = 18_000, lead_seconds: int = 90,
        message: str = DEFAULT_MESSAGE, run_tests: bool = False,
        dry_run: bool = False,
    ) -> dict[str, object]:
        if session_seconds <= 0 or lead_seconds <= 0 or lead_seconds >= session_seconds:
            return self._error("invalid_watch_duration")
        wait_seconds = session_seconds - lead_seconds
        if dry_run:
            plan = self.plan(message=message)
            if isinstance(plan, dict):
                return plan
            return {
                "ok": True,
                "status": "watch_planned",
                "wait_seconds": wait_seconds,
                "lead_seconds": lead_seconds,
                "plan": plan.as_dict(),
            }
        deadline = self._clock() + wait_seconds
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            self._sleeper(min(remaining, 60.0))
        result = self.commit_and_push(message=message, run_tests=run_tests)
        return {**result, "wait_seconds": wait_seconds, "lead_seconds": lead_seconds}
