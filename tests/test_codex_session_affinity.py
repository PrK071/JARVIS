from __future__ import annotations

import json
import threading
from pathlib import Path

from tern.orchestrator.codex import CodexResult, CodexRunner
from tern.orchestrator.codex_sessions import (
    CodexSessionRegistry,
    CodexSessionResolver,
    normalize_project_path,
)
from tern.orchestrator.security import PathPolicy


def session(
    project: Path,
    thread_id: str,
    *,
    state: str = "idle",
    visible: bool = True,
    recoverable: bool = True,
    origin: str = "user_existing",
    active_job_id: str | None = None,
) -> dict:
    return {
        "thread_id": thread_id,
        "session_id": thread_id,
        "project": str(project.resolve()),
        "project_key": normalize_project_path(project),
        "state": state,
        "source": "cli" if visible else "appServer",
        "visible": visible,
        "recoverable": recoverable,
        "ephemeral": False,
        "origin": origin,
        "active_job_id": active_job_id,
    }


def resolver(tmp_path: Path):
    registry = CodexSessionRegistry(tmp_path / "state")
    return registry, CodexSessionResolver(registry)


def test_single_existing_session_reused(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _registry, value = resolver(tmp_path)
    result = value.resolve(project=project, candidates=[session(project, "thread-a")])
    assert result.ok and result.thread_id == "thread-a"
    assert result.reason_code == "UNIQUE_PROJECT_SESSION_MATCH"
    assert result.reused and not result.created


def test_explicit_then_focused_session_precedence(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _registry, value = resolver(tmp_path)
    candidates = [session(project, "thread-a"), session(project, "thread-b")]
    explicit = value.resolve(
        project=project,
        candidates=candidates,
        explicit_thread_id="thread-a",
        focused_thread_id="thread-b",
    )
    focused = value.resolve(
        project=project,
        candidates=candidates,
        focused_thread_id="thread-b",
    )
    assert explicit.thread_id == "thread-a"
    assert explicit.reason_code == "EXPLICIT_SESSION_MATCH"
    assert focused.thread_id == "thread-b"
    assert focused.reason_code == "FOCUSED_SESSION_MATCH"


def test_conversation_affinity_wins_over_project_binding(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    registry, value = resolver(tmp_path)
    for record in [session(project, "thread-a"), session(project, "thread-b")]:
        registry.register(record)
    registry.bind_project(project, "thread-a")
    registry.bind_conversation("conversation-1", "thread-b", project)
    result = value.resolve(
        project=project,
        candidates=registry.list(project=project),
        conversation_id="conversation-1",
    )
    assert result.thread_id == "thread-b"
    assert result.reason_code == "CONVERSATION_AFFINITY_MATCH"


def test_ambiguous_sessions_do_not_guess_or_create(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _registry, value = resolver(tmp_path)
    result = value.resolve(
        project=project,
        candidates=[session(project, "thread-a"), session(project, "thread-b")],
    )
    assert result.status == "AMBIGUOUS"
    assert result.reason_code == "AMBIGUOUS_SESSION"
    assert result.thread_id is None and not result.created


def test_different_project_and_stale_session_not_reused(tmp_path):
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    _registry, value = resolver(tmp_path)
    different = value.resolve(
        project=project_b,
        candidates=[session(project_a, "thread-a")],
    )
    stale = value.resolve(
        project=project_b,
        candidates=[
            session(
                project_b,
                "thread-stale",
                state="stale",
                recoverable=False,
            )
        ],
    )
    assert different.status == "NONE"
    assert stale.status == "NONE"
    assert not different.reused and not stale.reused


def test_busy_session_queues_only_when_owned_job_exists(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _registry, value = resolver(tmp_path)
    external = value.resolve(
        project=project,
        candidates=[session(project, "thread-a", state="active")],
        focused_thread_id="thread-a",
    )
    owned = value.resolve(
        project=project,
        candidates=[
            session(
                project,
                "thread-a",
                state="active",
                active_job_id="job-a",
            )
        ],
        focused_thread_id="thread-a",
    )
    assert external.status == "UNAVAILABLE"
    assert external.reason_code == "SESSION_BUSY"
    assert owned.ok and owned.reason_code == "SESSION_BUSY_QUEUED"


def test_restart_restores_project_affinity(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    first = CodexSessionRegistry(tmp_path / "state")
    first.register(session(project, "thread-a"))
    first.bind_project(project, "thread-a")
    second = CodexSessionRegistry(tmp_path / "state")
    result = CodexSessionResolver(second).resolve(
        project=project,
        candidates=second.list(project=project),
    )
    assert result.thread_id == "thread-a"
    assert result.reason_code == "PROJECT_AFFINITY_MATCH"


def test_stale_focused_session_falls_back_to_valid_project_affinity(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    registry, value = resolver(tmp_path)
    registry.register(
        session(project, "stale", state="stale", recoverable=False)
    )
    registry.register(session(project, "valid"))
    registry.bind_project(project, "valid")
    result = value.resolve(
        project=project,
        candidates=registry.list(project=project),
        focused_thread_id="stale",
    )
    assert result.thread_id == "valid"
    assert result.reason_code == "PROJECT_AFFINITY_MATCH"


class FakeBridgeLog:
    def __init__(self):
        self.records: list[tuple[str, dict]] = []

    def write(self, event, **values):
        self.records.append((event, values))


class FakeRuntime:
    def interventions_for(self, _turn_id):
        return []


class FakeSessionManager:
    def __init__(self, project: Path, threads: list[dict] | None = None):
        self.project = project.resolve()
        self.threads = list(threads or [])
        self.create_count = 0
        self.adopted: list[str] = []
        self.tasks: list[str] = []
        self.bridge_log = FakeBridgeLog()
        self.runtime = FakeRuntime()
        self._turn = 0

    def list_project_threads(self):
        return [dict(item) for item in self.threads]

    def create_thread(self):
        self.create_count += 1
        value = session(
            self.project,
            f"created-{self.create_count}",
            visible=True,
            origin="jarvis_created",
        )
        self.threads.append(value)
        return dict(value)

    def adopt_thread(self, thread_id):
        self.adopted.append(thread_id)
        return dict(next(item for item in self.threads if item["thread_id"] == thread_id))

    def run_turn(self, task, **kwargs):
        self.tasks.append(task)
        self._turn += 1
        thread_id = kwargs["target_thread_id"]
        callback = kwargs.get("event_callback")
        if callback:
            callback(
                "codex_turn_started",
                {"thread_id": thread_id, "turn_id": f"turn-{self._turn}"},
            )
        return CodexResult(
            True,
            thread_id,
            f"turn-{self._turn}",
            "completed",
            "done",
            None,
        )

    def read_turn(self, thread_id, turn_id):
        return {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "status": "completed",
            "final_response": "done",
            "error": None,
        }


def integration_runner(tmp_path: Path, manager: FakeSessionManager) -> CodexRunner:
    value = CodexRunner(
        PathPolicy((manager.project,)),
        state_dir=tmp_path / "state",
        quick_wait_timeout=2,
    )
    value._managers[manager.project] = manager
    return value


def test_new_session_registered_recoverable_then_reused_payload_unchanged(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    manager = FakeSessionManager(project)
    value = integration_runner(tmp_path, manager)
    first = value.delegate_to_codex(
        task="PAYLOAD-EXATO",
        project_path=str(project),
        conversation_id="conversation-1",
    )
    second = value.delegate_to_codex(
        task="PAYLOAD-SEGUNDO",
        project_path=str(project),
        conversation_id="conversation-1",
    )
    record = value.sessions.get(str(first.thread_id))
    assert first.ok and second.ok
    assert first.thread_id == second.thread_id == "created-1"
    assert manager.create_count == 1
    assert manager.tasks == ["PAYLOAD-EXATO", "PAYLOAD-SEGUNDO"]
    assert record and record["recoverable"] and record["origin"] == "jarvis_created"
    assert record["visible"]
    assert all(job["thread_id"] == "created-1" for job in value.jobs.list())
    metrics = value.session_metrics()
    assert metrics["correct_session_rate"] == 1.0
    assert metrics["ghost_session_rate"] == 0.0
    assert metrics["session_visibility_rate"] == 1.0
    assert metrics["new_codex_session_created_when_reusable_session_exists"] == 0


def test_registration_failure_sends_no_user_work(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    manager = FakeSessionManager(project)
    value = integration_runner(tmp_path, manager)

    def failed(_record):
        raise OSError("registry unavailable")

    monkeypatch.setattr(value.sessions, "register", failed)
    result = value.delegate_to_codex(task="NAO-ENVIAR", project_path=str(project))
    assert not result.ok
    assert result.error == "SESSION_REGISTRATION_FAILED"
    assert manager.create_count == 1
    assert manager.tasks == []
    assert value.jobs.list() == []


def test_concurrent_resolution_creates_at_most_one_session(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    manager = FakeSessionManager(project)
    value = integration_runner(tmp_path, manager)
    results = []

    def resolve(number):
        results.append(
            value.resolve_session(
                project,
                request_id=f"request-{number}",
                conversation_id="conversation-1",
            )
        )

    first = threading.Thread(target=resolve, args=(1,))
    second = threading.Thread(target=resolve, args=(2,))
    first.start()
    second.start()
    first.join(5)
    second.join(5)
    assert len(results) == 2
    assert {item.thread_id for item in results} == {"created-1"}
    assert manager.create_count == 1


def test_concurrent_delegation_creates_once_and_keeps_both_tasks(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    manager = FakeSessionManager(project)
    value = integration_runner(tmp_path, manager)
    results = []

    def delegate(number):
        results.append(
            value.delegate_to_codex(
                task=f"payload-{number}",
                project_path=str(project),
                conversation_id="conversation-1",
            )
        )

    first = threading.Thread(target=delegate, args=(1,))
    second = threading.Thread(target=delegate, args=(2,))
    first.start()
    second.start()
    first.join(5)
    second.join(5)
    assert len(results) == 2 and all(item.ok for item in results)
    assert {item.thread_id for item in results} == {"created-1"}
    assert manager.create_count == 1
    assert sorted(manager.tasks) == ["payload-1", "payload-2"]


def test_user_created_session_reused_without_destructive_ownership(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    existing = session(project, "user-thread", origin="user_existing")
    manager = FakeSessionManager(project, [existing])
    value = integration_runner(tmp_path, manager)
    result = value.delegate_to_codex(task="reuse", project_path=str(project))
    record = value.sessions.get("user-thread")
    assert result.thread_id == "user-thread" and manager.create_count == 0
    assert record and record["origin"] == "user_existing"


def test_diagnostic_corpus_is_separate_and_complete():
    path = Path(__file__).parent / "data" / "codex_session_affinity_diagnostic.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["id"] for item in cases] == [f"C{number}" for number in range(1, 13)]
