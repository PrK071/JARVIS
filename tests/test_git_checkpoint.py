from __future__ import annotations

import subprocess
from pathlib import Path

from tern.orchestrator.cli import build_parser
from tern.orchestrator.git_checkpoint import GitCheckpointService
from tern.orchestrator.text import TextSession


class FakeGit:
    def __init__(
        self, root: Path, *, paths: tuple[str, ...] = ("tern/example.py",),
        remote: str = "https://github.com/PrK071/JARVIS.git",
        staged: tuple[str, ...] = (),
    ) -> None:
        self.root = root
        self.paths = paths
        self.remote = remote
        self.staged = staged
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command, _cwd):
        values = tuple(command)
        self.calls.append(values)
        args = values[3:] if values[0] == "git" else values
        stdout = ""
        if args == ("rev-parse", "--show-toplevel"):
            stdout = str(self.root)
        elif args == ("branch", "--show-current"):
            stdout = "main"
        elif args == ("remote", "get-url", "origin"):
            stdout = self.remote
        elif args == ("diff", "--cached", "--name-only", "-z"):
            stdout = "\0".join(self.staged) + ("\0" if self.staged else "")
        elif args == ("diff", "--name-only", "-z", "HEAD", "--"):
            stdout = "\0".join(self.paths) + ("\0" if self.paths else "")
        elif args == ("ls-files", "--others", "--exclude-standard", "-z"):
            stdout = ""
        elif args == ("diff", "--check"):
            stdout = ""
        elif args[:2] == ("commit", "-m"):
            stdout = "[main abc123] checkpoint"
        return subprocess.CompletedProcess(values, 0, stdout=stdout, stderr="")


def test_checkpoint_commits_and_pushes_only_checked_paths(tmp_path):
    runner = FakeGit(tmp_path)
    service = GitCheckpointService(tmp_path, runner=runner)

    result = service.commit_and_push(message="chore: checkpoint")

    assert result["ok"] is True
    assert result["committed"] is True
    assert result["pushed"] is True
    assert ("git", "-C", str(tmp_path), "add", "--", "tern/example.py") in runner.calls
    assert ("git", "-C", str(tmp_path), "push", "origin", "main") in runner.calls


def test_checkpoint_fails_closed_for_protected_paths(tmp_path):
    service = GitCheckpointService(tmp_path, runner=FakeGit(tmp_path, paths=(".env",)))

    result = service.commit_and_push()

    assert result["ok"] is False
    assert result["error"] == "protected_path_detected"


def test_checkpoint_never_consumes_existing_staged_changes(tmp_path):
    service = GitCheckpointService(
        tmp_path, runner=FakeGit(tmp_path, staged=("unrelated.py",))
    )

    result = service.commit_and_push()

    assert result["ok"] is False
    assert result["error"] == "preexisting_staged_changes"


def test_checkpoint_rejects_an_unexpected_remote(tmp_path):
    service = GitCheckpointService(
        tmp_path, runner=FakeGit(tmp_path, remote="https://github.com/example/other.git")
    )

    result = service.commit_and_push()

    assert result["ok"] is False
    assert result["error"] == "unexpected_remote"


def test_watch_dry_run_reports_the_time_based_plan_without_waiting(tmp_path):
    runner = FakeGit(tmp_path)
    service = GitCheckpointService(
        tmp_path, runner=runner, sleeper=lambda _seconds: (_ for _ in ()).throw(AssertionError())
    )

    result = service.watch_and_checkpoint(
        session_seconds=18_000, lead_seconds=90, dry_run=True
    )

    assert result["ok"] is True
    assert result["status"] == "watch_planned"
    assert result["wait_seconds"] == 17_910
    assert not any(call[-1] == "main" and "push" in call for call in runner.calls)


def test_cli_exposes_immediate_and_elapsed_time_checkpoint_commands():
    parser = build_parser()

    immediate = parser.parse_args(["git-checkpoint-push", "--dry-run"])
    watch = parser.parse_args([
        "git-checkpoint-watch", "--session-seconds", "18000", "--lead-seconds", "90",
        "--dry-run",
    ])

    assert immediate.command == "git-checkpoint-push"
    assert watch.command == "git-checkpoint-watch"
    assert watch.session_seconds - watch.lead_seconds == 17_910


def test_text_session_executes_the_explicit_checkpoint_slash_command():
    class Console:
        def __init__(self):
            self.prompts = iter(("/git-checkpoint-push chore: checkpoint",))
            self.writes: list[str] = []

        def read(self, _prompt):
            return next(self.prompts)

        def write(self, value):
            self.writes.append(value)

    class Supervisor:
        def run(self, *_args, **_kwargs):
            raise AssertionError("slash command must bypass the supervisor")

    console = Console()
    session = TextSession(
        Supervisor(),
        console=console,
        slash_commands={"git-checkpoint-push": lambda message: {"ok": True, "message": message}},
    )

    result = session.run(once=True)

    assert result["ok"] is True
    assert result["interactions"] == 1
    assert "chore: checkpoint" in console.writes[-1]
