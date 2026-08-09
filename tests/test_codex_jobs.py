from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tern.orchestrator.agent import Supervisor, _codex_job_intent
from tern.orchestrator import cli as cli_module
from tern.orchestrator.codex import CodexError, CodexResult, CodexRunner
from tern.orchestrator.codex_jobs import CodexJobStore
from tern.orchestrator.config import load_settings
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry


THREAD = "thread-shared"


class Runtime:
    def interventions_for(self, _turn_id):
        return []


class FakeManager:
    def __init__(self, *, blocked: bool = False, fail: bool = False):
        self.blocked = blocked
        self.fail = fail
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.started = threading.Event()
        self.calls: list[dict] = []
        self.resume_calls = 0
        self.cancel_calls = 0
        self.steer_calls: list[str] = []
        self.read_status = "running"
        self.read_response = "offline result"
        self.runtime = Runtime()
        self._serial = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run_turn(self, task, **kwargs):
        with self._serial:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            turn_id = f"turn-{len(self.calls) + 1}"
            self.calls.append({"task": task, **kwargs, "turn_id": turn_id})
            callback = kwargs.get("event_callback")
            if callback:
                callback(
                    "codex_turn_started",
                    {"thread_id": THREAD, "turn_id": turn_id},
                )
                callback(
                    "codex_event",
                    {"method": "item/started", "turn_id": turn_id},
                )
            self.started.set()
            assert self.release.wait(5)
            self.active -= 1
            if self.fail:
                return CodexResult(True, THREAD, turn_id, "failed", "", "turn_failed")
            return CodexResult(True, THREAD, turn_id, "completed", "done", None)

    def read_turn(self, thread_id, turn_id):
        if self.read_status == "unavailable":
            raise CodexError("server unavailable", layer="websocket")
        return {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "status": self.read_status,
            "final_response": self.read_response,
            "error": None,
        }

    def resume_turn(self, thread_id, turn_id, **kwargs):
        self.resume_calls += 1
        callback = kwargs.get("event_callback")
        if callback:
            callback("codex_reconnected", {"thread_id": thread_id, "turn_id": turn_id})
        assert self.release.wait(5)
        if self.read_status == "disconnected":
            return CodexResult(True, thread_id, turn_id, "disconnected", "", "network")
        return CodexResult(True, thread_id, turn_id, "completed", self.read_response, None)

    def cancel(self):
        self.cancel_calls += 1
        return {"cancelled": True, "thread_id": THREAD, "turn_id": "turn-1"}

    def steer(self, instruction):
        self.steer_calls.append(instruction)
        return {
            "accepted": True,
            "thread_id": THREAD,
            "turn_id": "turn-1",
            "status": "steer accepted",
        }


def runner(tmp_path: Path, manager: FakeManager, **kwargs) -> CodexRunner:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    value = CodexRunner(
        PathPolicy((project,)),
        state_dir=tmp_path / "state",
        quick_wait_timeout=kwargs.pop("quick_wait_timeout", 1),
        **kwargs,
    )
    value._managers[project.resolve()] = manager
    return value


def wait_for_status(value: CodexRunner, job_id: str, status: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = value.jobs.get(job_id)
        if job and job.get("status") == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {status}")


def test_quick_wait_returns_completed_result(tmp_path):
    manager = FakeManager()
    value = runner(tmp_path, manager)
    result = value.delegate_to_codex(
        task="read one file",
        project_path=str(tmp_path / "project"),
        wait=True,
    )
    assert result.ok and result.status == "completed" and result.job_id
    assert len(manager.calls) == 1
    assert value.jobs.get(str(result.job_id))["result_delivered"] is True


def test_background_returns_running_and_persists_job(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    started = time.monotonic()
    result = value.delegate_to_codex(
        task="long task",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    assert result.accepted and result.status == "running"
    assert time.monotonic() - started < 0.5
    assert result.thread_id == THREAD and result.turn_id == "turn-1"
    persisted = json.loads(value.jobs.path.read_text(encoding="utf-8"))
    assert persisted["jobs"][0]["job_id"] == result.job_id
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if value.jobs.get(str(result.job_id)).get("last_event") == "item/started":
            break
        time.sleep(0.01)
    assert value.jobs.get(str(result.job_id))["last_event"] == "item/started"
    manager.release.set()
    wait_for_status(value, str(result.job_id), "completed")


def test_client_wait_timeout_does_not_cancel_turn(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager, quick_wait_timeout=0.05)
    result = value.delegate_to_codex(
        task="slow quick task",
        project_path=str(tmp_path / "project"),
        wait=True,
    )
    assert result.accepted and result.status == "running"
    assert result.wait_timed_out and manager.cancel_calls == 0
    assert manager.release.is_set() is False
    manager.release.set()
    wait_for_status(value, str(result.job_id), "completed")


def test_later_completion_is_delivered_once(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    result = value.delegate_to_codex(
        task="background",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    manager.release.set()
    wait_for_status(value, str(result.job_id), "completed")
    first = value.claim_completed_results()
    assert [job["job_id"] for job in first] == [result.job_id]
    assert value.acknowledge_result(first[0]["job_id"], first[0]["delivery_token"])
    assert value.claim_completed_results() == []


def test_real_turn_failure_is_not_wait_timeout(tmp_path):
    manager = FakeManager(fail=True)
    value = runner(tmp_path, manager)
    result = value.delegate_to_codex(
        task="fail",
        project_path=str(tmp_path / "project"),
        wait=True,
    )
    assert result.status == "failed"
    assert result.error == "turn_failed"
    assert not result.wait_timed_out


def test_status_cancel_and_steer_use_same_job_thread_and_turn(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    result = value.delegate_to_codex(
        task="active",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    status = value.get_job_status(job_id=result.job_id)
    assert status["status"] == "running"
    steer = value.steer_job("only warnings", job_id=result.job_id)
    assert steer["thread_id"] == THREAD and steer["turn_id"] == "turn-1"
    assert manager.steer_calls == ["only warnings"]
    cancelled = value.cancel_job(job_id=result.job_id)
    assert cancelled["status"] == "interrupted" and manager.cancel_calls == 1
    manager.release.set()


def test_cancel_latest_job_does_not_start_another_turn(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    value.delegate_to_codex(
        task="active",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    result = value.cancel_job(latest=True)
    assert result["ok"] and len(manager.calls) == 1
    manager.release.set()


def test_two_background_jobs_are_serialized_without_duplicate_turn(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager, quick_wait_timeout=0.05)
    first = value.delegate_to_codex(
        task="first",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    second = value.delegate_to_codex(
        task="second",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    assert first.turn_id == "turn-1"
    assert second.status in {"queued", "starting"}
    assert len(manager.calls) == 1
    manager.release.set()
    wait_for_status(value, str(first.job_id), "completed")
    wait_for_status(value, str(second.job_id), "completed")
    assert len(manager.calls) == 2 and manager.max_active == 1


def test_restart_recovers_completed_turn_without_starting_new_one(tmp_path):
    manager = FakeManager()
    value = runner(tmp_path, manager)
    job = value.jobs.create(
        project=str(tmp_path / "project"),
        task_summary="offline",
        source="qwen",
        wait=False,
    )
    value.jobs.update(
        job["job_id"],
        status="running",
        thread_id=THREAD,
        turn_id="turn-offline",
        monitor_pid=99999999,
    )
    manager.read_status = "completed"
    reconciled = value.reconcile_jobs()
    assert reconciled[-1]["status"] == "completed"
    assert reconciled[-1]["result"]["final_response"] == "offline result"
    assert manager.calls == []


def test_restart_reconnects_to_running_turn_without_turn_start(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    job = value.jobs.create(
        project=str(tmp_path / "project"),
        task_summary="resume",
        source="qwen",
        wait=False,
    )
    value.jobs.update(
        job["job_id"],
        status="running",
        thread_id=THREAD,
        turn_id="turn-resume",
        monitor_pid=99999999,
    )
    value.reconcile_jobs()
    deadline = time.monotonic() + 2
    while manager.resume_calls < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.resume_calls == 1 and manager.calls == []
    manager.release.set()
    wait_for_status(value, job["job_id"], "completed")


def test_app_server_unavailable_marks_job_disconnected(tmp_path):
    manager = FakeManager()
    manager.read_status = "unavailable"
    value = runner(tmp_path, manager)
    job = value.jobs.create(
        project=str(tmp_path / "project"),
        task_summary="offline",
        source="qwen",
        wait=False,
    )
    value.jobs.update(
        job["job_id"],
        status="running",
        thread_id=THREAD,
        turn_id="turn-offline",
        monitor_pid=99999999,
    )
    value.reconcile_jobs()
    assert value.jobs.get(job["job_id"])["status"] == "disconnected"


def test_old_terminal_jobs_are_pruned(tmp_path):
    store = CodexJobStore(tmp_path, retention_days=7)
    job = store.create(project=str(tmp_path), task_summary="old", source="qwen", wait=False)
    store.update(job["job_id"], status="completed", result_available=True)
    state = json.loads(store.path.read_text(encoding="utf-8"))
    state["jobs"][0]["completed_at"] = (
        datetime.now(timezone.utc) - timedelta(days=8)
    ).isoformat()
    store.path.write_text(json.dumps(state), encoding="utf-8")
    assert store.list() == []


def test_delivery_claim_can_be_recovered_before_ack(tmp_path):
    store = CodexJobStore(tmp_path)
    job = store.create(project=str(tmp_path), task_summary="done", source="qwen", wait=False)
    store.update(job["job_id"], status="completed", result_available=True, result={"ok": True})
    first = store.claim_results()[0]
    state = json.loads(store.path.read_text(encoding="utf-8"))
    state["jobs"][0]["delivery_claim"]["pid"] = 99999999
    store.path.write_text(json.dumps(state), encoding="utf-8")
    second = CodexJobStore(tmp_path).claim_results()[0]
    assert first["job_id"] == second["job_id"]
    assert first["delivery_token"] != second["delivery_token"]


def test_job_intents_select_status_cancel_and_steer():
    assert _codex_job_intent("O Codex já terminou?") == "get_codex_job_status"
    assert _codex_job_intent("Cancele a tarefa do Codex.") == "cancel_codex_job"
    assert _codex_job_intent("Avise ao Codex para mostrar só avisos.") == "steer_codex_job"


def test_job_timeout_and_retention_configuration():
    settings = load_settings(
        {
            "CODEX_QUICK_WAIT_TIMEOUT_SECONDS": "12",
            "CODEX_TURN_HARD_TIMEOUT_SECONDS": "0",
            "CODEX_JOB_RETENTION_DAYS": "9",
        }
    )
    assert settings.codex_quick_wait_timeout_seconds == 12
    assert settings.codex_turn_hard_timeout_seconds == 0
    assert settings.codex_job_retention_days == 9


def test_cli_lists_status_and_result_for_jobs(monkeypatch, capsys):
    job = {
        "job_id": "job-12345678",
        "status": "completed",
        "task_summary": "validate bridge",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result_available": True,
        "result": {"final_response": "done"},
        "error": None,
    }

    class Jobs:
        def get(self, job_id):
            return job if job_id == job["job_id"] else None

    class Codex:
        jobs = Jobs()

        def list_jobs(self):
            return [job]

        def get_job_status(self, **_kwargs):
            return {"ok": True, "job_id": job["job_id"], "status": "completed"}

        def reconcile_jobs(self):
            return []

    class Registry:
        codex = Codex()

    monkeypatch.setattr(cli_module, "_registry", lambda *_args, **_kwargs: Registry())
    assert cli_module.main(["codex-jobs"]) == 0
    assert cli_module.main(["codex-job-status", job["job_id"]]) == 0
    assert cli_module.main(["codex-job-result", job["job_id"]]) == 0
    output = capsys.readouterr().out
    assert "Codex jobs" in output
    assert "job-123" in output
    assert "completed" in output and "done" in output


class FinalClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **_kwargs):
        self.messages = list(messages)
        return {
            "choices": [{"message": {"role": "assistant", "content": "resultado recebido"}}],
            "usage": {},
        }


class ToolCallingClient:
    def __init__(self, tool_call):
        self.tool_call = tool_call
        self.tools_seen = []
        self.calls = 0

    def chat(self, _messages, **kwargs):
        self.calls += 1
        self.tools_seen.append(kwargs.get("tools"))
        if self.calls == 1:
            return {
                "choices": [{"message": {"role": "assistant", "tool_calls": [self.tool_call]}}],
                "usage": {},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": "ainda executando"}}],
            "usage": {},
        }


def test_status_question_uses_job_tool_without_filesystem_or_new_turn(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    result = value.delegate_to_codex(
        task="active",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    tools = ToolRegistry(
        policy=value.policy,
        logger=ActionLogger(tmp_path / "state" / "actions.jsonl"),
        codex=value,
        max_output_bytes=131072,
    )
    call = {
        "id": "status-one",
        "type": "function",
        "function": {
            "name": "get_codex_job_status",
            "arguments": json.dumps({"job_id": result.job_id}),
        },
    }
    client = ToolCallingClient(call)
    answer = Supervisor(
        load_settings({"AGENT_DECISION_FAST_PATH": "false"}),
        client,
        tools,
    ).run("O Codex já terminou?")
    assert answer["ok"] and answer["tool_calls"] == 1
    assert [item["function"]["name"] for item in client.tools_seen[0]] == [
        "get_codex_job_status"
    ]
    assert len(manager.calls) == 1
    manager.release.set()


def test_completed_result_is_injected_into_qwen_and_acknowledged(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager)
    result = value.delegate_to_codex(
        task="done",
        project_path=str(tmp_path / "project"),
        wait=False,
    )
    manager.release.set()
    wait_for_status(value, str(result.job_id), "completed")
    tools = ToolRegistry(
        policy=value.policy,
        logger=ActionLogger(tmp_path / "state" / "actions.jsonl"),
        codex=value,
        max_output_bytes=131072,
    )
    client = FinalClient()
    answer = Supervisor(load_settings({}), client, tools).run("Olá")
    assert answer["ok"]
    system_values = [item["content"] for item in client.messages if item["role"] == "system"]
    assert any(result.job_id in item and "codex_job_completed" in item for item in system_values)
    assert value.jobs.get(str(result.job_id))["result_delivered"] is True


def test_delegate_wait_is_selected_from_task_intent(tmp_path):
    manager = FakeManager(blocked=True)
    value = runner(tmp_path, manager, quick_wait_timeout=0.05)
    value.shared_project = lambda: (tmp_path / "project").resolve()
    tools = ToolRegistry(
        policy=value.policy,
        logger=ActionLogger(tmp_path / "state" / "actions.jsonl"),
        codex=value,
        max_output_bytes=131072,
    )
    result = tools.execute(
        "delegate_to_codex",
        {"task": "Implemente uma funcionalidade", "project_path": str(tmp_path / "project")},
        context={
            "turn_id": "auto",
            "user_text": f"Implemente em segundo plano no projeto {tmp_path / 'project'}",
        },
    )
    assert result.get("status") == "running" and result.get("job_id"), result
    manager.release.set()
