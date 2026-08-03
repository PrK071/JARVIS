from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import tern.orchestrator.cli as cli_module
import tern.orchestrator.codex as codex_module
from tern.orchestrator.codex import (
    CodexError,
    CodexProtocolError,
    CodexSessionManager,
)


THREAD_ID = "thread-shared"


class FakeProtocol:
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active_turn_id: str | None = None
        self.turn_number = 0
        self.started = threading.Event()
        self.start_calls: list[dict] = []
        self.steer_calls: list[dict] = []
        self.interrupt_calls: list[dict] = []
        self.completed: dict[str, tuple[str, str]] = {}
        self.turns: list[dict] = []
        self.steer_delay = 0.0
        self.steer_concurrency = 0
        self.max_steer_concurrency = 0
        self.thread_exists = True
        self.request_calls: list[tuple[str, dict]] = []

    def start(self, params: dict) -> dict:
        with self.condition:
            if self.active_turn_id is not None:
                raise CodexProtocolError("turn already active")
            self.turn_number += 1
            turn_id = f"turn-{self.turn_number}"
            self.active_turn_id = turn_id
            self.start_calls.append(params)
            self.turns.append(
                {"id": turn_id, "status": "inProgress", "items": []}
            )
            self.started.set()
            self.condition.notify_all()
            return {
                "turn": {"id": turn_id, "status": "inProgress", "items": []}
            }

    def steer(self, params: dict) -> dict:
        with self.lock:
            if params["expectedTurnId"] != self.active_turn_id:
                raise CodexProtocolError("expected turn is not active")
            self.steer_concurrency += 1
            self.max_steer_concurrency = max(
                self.max_steer_concurrency, self.steer_concurrency
            )
        try:
            if self.steer_delay:
                time.sleep(self.steer_delay)
            self.steer_calls.append(params)
            return {"turnId": params["expectedTurnId"]}
        finally:
            with self.lock:
                self.steer_concurrency -= 1

    def interrupt(self, params: dict) -> dict:
        with self.condition:
            if params["turnId"] != self.active_turn_id:
                raise CodexProtocolError("turn is not active")
            self.interrupt_calls.append(params)
            self._finish_unlocked(params["turnId"], "interrupted", "late result")
            return {}

    def finish(
        self,
        status: str = "completed",
        final: str = "done",
        *,
        turn_id: str | None = None,
    ) -> str:
        with self.condition:
            value = turn_id or self.active_turn_id
            assert value is not None
            self._finish_unlocked(value, status, final)
            return value

    def _finish_unlocked(self, turn_id: str, status: str, final: str) -> None:
        self.completed[turn_id] = (status, final)
        for turn in self.turns:
            if turn["id"] == turn_id:
                turn["status"] = status
                turn["items"] = [
                    {
                        "id": f"message-{turn_id}",
                        "type": "agentMessage",
                        "text": final,
                    }
                ]
        if self.active_turn_id == turn_id:
            self.active_turn_id = None
        self.condition.notify_all()

    def read_thread(self) -> dict:
        if not self.thread_exists:
            raise CodexProtocolError("no rollout found")
        return {
            "thread": {
                "id": THREAD_ID,
                "status": {
                    "type": "active" if self.active_turn_id else "idle",
                    **(
                        {"activeFlags": []}
                        if self.active_turn_id
                        else {}
                    ),
                },
                "turns": [dict(turn) for turn in self.turns],
            }
        }


class FakeClient:
    def __init__(self, protocol: FakeProtocol, *_args, **_kwargs):
        self.protocol = protocol
        self.connected = True

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def notify(self, *_args, **_kwargs) -> None:
        return None

    def request(self, method: str, params: dict | None = None) -> dict:
        params = params or {}
        self.protocol.request_calls.append((method, params))
        if method == "initialize":
            return {}
        if method == "turn/start":
            return self.protocol.start(params)
        if method == "turn/steer":
            return self.protocol.steer(params)
        if method == "turn/interrupt":
            return self.protocol.interrupt(params)
        if method == "thread/read":
            return self.protocol.read_thread()
        if method == "thread/resume":
            return self.protocol.read_thread()
        if method == "thread/start":
            self.protocol.thread_exists = True
            return {"thread": {"id": THREAD_ID}}
        raise AssertionError(method)


class RunManager(CodexSessionManager):
    def __init__(self, project: Path, protocol: FakeProtocol):
        super().__init__(project, timeout=10, state_dir=project / ".orchestrator")
        self.protocol = protocol
        self.fake_client = FakeClient(protocol)

    def start_server(self, wait_seconds: int = 30) -> dict:
        return {"started": False, "ready": True, "endpoint": self.endpoint}

    def is_ready(self) -> bool:
        return True

    def connect(self) -> FakeClient:
        self._client = self.fake_client
        return self.fake_client

    def ensure_thread(self, *, continue_current_thread: bool = True) -> str:
        self._known_thread_id = THREAD_ID
        self._thread_created = False
        self._persist_session(THREAD_ID)
        self.runtime.update(thread_id=THREAD_ID)
        return THREAD_ID

    def _wait_for_completion(self, thread_id: str, turn_id: str) -> dict:
        with self.protocol.condition:
            assert thread_id == THREAD_ID
            deadline = time.monotonic() + 8
            while turn_id not in self.protocol.completed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fake completion timeout")
                self.protocol.condition.wait(remaining)
            status, final = self.protocol.completed[turn_id]
        self._final_messages.append((turn_id, final))
        return {
            "threadId": thread_id,
            "turn": {
                "id": turn_id,
                "status": status,
                "error": None,
                "items": [],
            },
        }


@pytest.fixture
def protocol_client(monkeypatch):
    protocol = FakeProtocol()

    def factory(*args, **kwargs):
        return FakeClient(protocol, *args, **kwargs)

    monkeypatch.setattr(codex_module, "CodexAppServerClient", factory)
    return protocol


def start_qwen_turn(manager: RunManager, task: str = "original task"):
    values = {}

    def run():
        values["result"] = manager.run_turn(task, origin="qwen")

    worker = threading.Thread(target=run)
    worker.start()
    assert manager.protocol.started.wait(3)
    return worker, values


def test_human_steer_keeps_thread_and_turn_and_returns_to_qwen(
    tmp_path, protocol_client
):
    manager = RunManager(tmp_path, protocol_client)
    worker, values = start_qwen_turn(manager)
    active_turn = protocol_client.active_turn_id
    instruction = "Use somente testes do bridge e inclua reconexao."
    steer = RunManager(tmp_path, protocol_client).steer(instruction)
    protocol_client.finish(final="Relatorio com secao sobre reconexao.")
    worker.join(5)

    result = values["result"]
    assert steer["thread_id"] == result.thread_id == THREAD_ID
    assert steer["turn_id"] == result.turn_id == active_turn
    assert protocol_client.steer_calls[0]["input"][0]["text"] == instruction
    assert "origin:" not in protocol_client.start_calls[0]["input"][0]["text"]
    assert result.human_interventions[0]["source"] == "human"
    assert result.human_interventions[0]["summary"] == instruction
    assert "reconexao" in result.final_response


def test_new_qwen_turn_waits_while_human_steers(tmp_path, protocol_client):
    first = RunManager(tmp_path, protocol_client)
    second = RunManager(tmp_path, protocol_client)
    first_worker, _first_values = start_qwen_turn(first, "first")
    second_values = {}
    second_worker = threading.Thread(
        target=lambda: second_values.setdefault(
            "result", second.run_turn("second", origin="qwen")
        )
    )
    second_worker.start()
    time.sleep(0.15)
    assert len(protocol_client.start_calls) == 1
    RunManager(tmp_path, protocol_client).steer("human priority")
    assert len(protocol_client.start_calls) == 1
    protocol_client.finish(final="first done")
    first_worker.join(5)
    deadline = time.monotonic() + 3
    while len(protocol_client.start_calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(protocol_client.start_calls) == 2
    protocol_client.finish(final="second done")
    second_worker.join(5)
    assert second_values["result"].ok


def test_message_origin_uses_metadata_not_prompt_prefix(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    worker, values = start_qwen_turn(manager, "technical content unchanged")
    params = protocol_client.start_calls[0]
    assert params["input"][0]["text"] == "technical content unchanged"
    assert params["clientUserMessageId"].startswith("tern-qwen-")
    protocol_client.finish()
    worker.join(5)
    assert values["result"].ok


def test_steer_without_active_turn_and_after_completion(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    manager.ensure_thread()
    with pytest.raises(CodexError, match="nenhum turn ativo"):
        manager.steer("late")
    worker, _values = start_qwen_turn(manager)
    protocol_client.finish()
    worker.join(5)
    with pytest.raises(CodexError, match="nenhum turn ativo"):
        RunManager(tmp_path, protocol_client).steer("too late")


def test_interrupt_after_completion_is_rejected(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    worker, _values = start_qwen_turn(manager)
    protocol_client.finish()
    worker.join(5)
    result = RunManager(tmp_path, protocol_client).cancel()
    assert result == {"cancelled": False, "reason": "nenhum turn ativo"}


def test_two_concurrent_steers_are_serialized(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    worker, _values = start_qwen_turn(manager)
    protocol_client.steer_delay = 0.1
    errors = []

    def steer(value):
        try:
            RunManager(tmp_path, protocol_client).steer(value)
        except Exception as exc:  # pragma: no cover - assertion captures it
            errors.append(exc)

    first = threading.Thread(target=steer, args=("first steer",))
    second = threading.Thread(target=steer, args=("second steer",))
    first.start()
    second.start()
    first.join(3)
    second.join(3)
    assert not errors
    assert protocol_client.max_steer_concurrency == 1
    assert len(protocol_client.steer_calls) == 2
    protocol_client.finish()
    worker.join(5)


def test_cancel_after_steer_discards_late_result_and_clears_queue(
    tmp_path, protocol_client
):
    manager = RunManager(tmp_path, protocol_client)
    worker, values = start_qwen_turn(manager)
    controller = RunManager(tmp_path, protocol_client)
    controller.steer("correction")
    cancel = controller.cancel()
    worker.join(5)
    result = values["result"]
    state = manager.runtime.read()
    names = {item["name"] for item in result.state_events}
    assert cancel["cancelled"]
    assert result.status == "interrupted"
    assert result.result_discarded
    assert result.final_response == ""
    assert state["state"] == "idle"
    assert state["queue_length"] == 0
    assert {"turn interrupted", "result discarded", "session ready"} <= names


def test_invalid_turn_id_is_not_steered(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    manager.ensure_thread()
    protocol_client.active_turn_id = "actual-turn"
    protocol_client.turns.append(
        {"id": "actual-turn", "status": "inProgress", "items": []}
    )
    manager.runtime.update(
        thread_id=THREAD_ID,
        turn_id="stale-turn",
        state="running",
    )
    with pytest.raises(CodexError, match="mudou ou ja concluiu"):
        manager.steer("must fail")
    assert not protocol_client.steer_calls
    assert manager.runtime.read()["state"] == "idle"


def test_tui_user_message_event_becomes_qwen_intervention(
    tmp_path, protocol_client
):
    manager = RunManager(tmp_path, protocol_client)
    manager.runtime.update(
        thread_id=THREAD_ID,
        turn_id="turn-tui",
        state="running",
        last_instruction_source="qwen",
        active_client_message_id="tern-qwen-original",
    )
    manager._on_event(
        {
            "method": "item/started",
            "params": {
                "threadId": THREAD_ID,
                "turnId": "turn-tui",
                "item": {
                    "id": "human-item",
                    "clientId": "tui-client-message",
                    "type": "userMessage",
                    "content": [
                        {"type": "text", "text": "analyze only three files"}
                    ],
                },
            },
        }
    )
    interventions = manager.runtime.interventions_for("turn-tui")
    assert interventions == [
        {
            "timestamp": interventions[0]["timestamp"],
            "source": "human",
            "operation": "turn/steer",
            "thread_id": THREAD_ID,
            "turn_id": "turn-tui",
            "client_message_id": "tui-client-message",
            "summary": "analyze only three files",
            "state": "observed",
        }
    ]


def test_reconnect_recovers_completed_turn_without_duplicate(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    manager.ensure_thread()
    protocol_client.active_turn_id = "turn-reconnect"
    protocol_client.turns.append(
        {"id": "turn-reconnect", "status": "inProgress", "items": []}
    )
    protocol_client.finish(
        turn_id="turn-reconnect",
        final="recovered once",
    )
    recovered = manager._recover_active_turn(THREAD_ID, "turn-reconnect")
    assert recovered["turn"]["status"] == "completed"
    assert manager._select_final_message("turn-reconnect") == "recovered once"


def test_restart_resumes_same_thread_and_no_old_result_duplication(
    tmp_path, protocol_client
):
    first = RunManager(tmp_path, protocol_client)
    first.ensure_thread()
    protocol_client.turns.append(
        {
            "id": "old-turn",
            "status": "completed",
            "items": [
                {"id": "old-message", "type": "agentMessage", "text": "old"}
            ],
        }
    )
    second = RunManager(tmp_path, protocol_client)
    assert second.ensure_thread() == THREAD_ID
    worker, values = start_qwen_turn(second, "new task")
    protocol_client.finish(final="new result")
    worker.join(5)
    assert values["result"].thread_id == THREAD_ID
    assert values["result"].final_response == "new result"
    assert "old" not in values["result"].final_response


def test_shared_status_repairs_stale_running_state(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    manager.ensure_thread()
    manager.runtime.update(
        thread_id=THREAD_ID,
        turn_id="old-turn",
        state="running",
        qwen_connected=True,
        qwen_pid=999999,
    )
    manager._known_tui_clients = lambda _thread_id: 1
    status = manager.shared_status()
    assert status["thread_id"] == THREAD_ID
    assert status["turn_id"] is None
    assert status["state"] == "idle"
    assert status["tui_clients_known"] == 1
    assert not status["qwen_connected"]


def test_cli_codex_steer_status_and_interrupt(monkeypatch, capsys):
    class Manager:
        def __init__(self, *_args, **_kwargs):
            pass

        def steer(self, instruction, origin="human"):
            assert instruction == "focus"
            assert origin == "human"
            return {"thread_id": THREAD_ID, "turn_id": "turn-1"}

        def cancel(self):
            return {
                "cancelled": True,
                "thread_id": THREAD_ID,
                "turn_id": "turn-1",
            }

        def shared_status(self):
            return {
                "endpoint": "ws://127.0.0.1:4500",
                "thread_id": THREAD_ID,
                "turn_id": "turn-1",
                "state": "running",
                "last_instruction_source": "human",
                "queue_length": 0,
                "tui_clients_known": 1,
                "qwen_connected": True,
                "last_event_age_seconds": 1.0,
                "error": None,
            }

    monkeypatch.setattr(cli_module, "CodexSessionManager", Manager)
    assert cli_module.main(["codex-steer", "focus"]) == 0
    assert "Steer enviado" in capsys.readouterr().out
    assert cli_module.main(["codex-interrupt"]) == 0
    assert "Interrupt enviado" in capsys.readouterr().out
    assert cli_module.main(["codex-shared-status"]) == 0
    output = capsys.readouterr().out
    assert "State: running" in output
    assert "TUI clients known: 1" in output


def test_missing_thread_is_replaced_and_reason_logged(tmp_path, protocol_client):
    manager = CodexSessionManager(
        tmp_path,
        timeout=10,
        state_dir=tmp_path / ".orchestrator",
    )
    manager.start_server = lambda wait_seconds=30: {
        "started": False,
        "ready": True,
    }
    manager._persist_session("missing-thread")
    protocol_client.thread_exists = False
    assert manager.ensure_thread() == THREAD_ID
    records = [
        json.loads(line)
        for line in manager.bridge_log.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    replacement = [item for item in records if item["event"] == "thread_replaced"]
    assert replacement[0]["previous_thread_id"] == "missing-thread"
    assert replacement[0]["new_thread_id"] == THREAD_ID


def test_disconnected_tui_does_not_stop_bridge_turn(tmp_path, protocol_client):
    manager = RunManager(tmp_path, protocol_client)
    worker, values = start_qwen_turn(manager)
    tui = FakeClient(protocol_client)
    tui.close()
    protocol_client.finish(final="bridge survived")
    worker.join(5)
    assert values["result"].ok
    assert values["result"].final_response == "bridge survived"


def test_app_server_unavailable_reports_executable_layer(tmp_path):
    manager = CodexSessionManager(
        tmp_path,
        endpoint="ws://127.0.0.1:4599",
        executable=str(tmp_path / "missing-codex.exe"),
        state_dir=tmp_path / ".orchestrator",
    )
    manager.is_ready = lambda: False
    with pytest.raises(CodexError) as failure:
        manager.start_server()
    assert failure.value.layer == "executable"


def test_review_session_reads_persisted_thread_without_starting_turn(
    tmp_path, protocol_client
):
    manager = RunManager(tmp_path, protocol_client)
    manager.ensure_thread()
    protocol_client.turns.extend(
        [
            {
                "id": "history-1",
                "status": "interrupted",
                "error": None,
                "items": [
                    {
                        "type": "userMessage",
                        "content": [{"type": "text", "text": "run tests"}],
                    }
                ],
            },
            {
                "id": "history-2",
                "status": "completed",
                "error": None,
                "items": [
                    {
                        "type": "userMessage",
                        "content": [
                            {"type": "text", "text": "read codex.py"}
                        ],
                    },
                    {
                        "type": "agentMessage",
                        "text": "codex.py reviewed; 5 tests passed",
                    },
                ],
            },
        ]
    )
    manager._known_tui_thread_ids = lambda: [THREAD_ID]
    before = len(protocol_client.start_calls)
    result = manager.review_session(turn_limit=2)
    after = len(protocol_client.start_calls)
    assert result["thread_id"] == THREAD_ID
    assert result["turns_read"] == 2
    assert result["last_turn"]["turn_id"] == "history-2"
    assert result["last_turn_state"] == "completed"
    assert result["threads_match"]
    assert result["last_turn"]["files_mentioned_or_changed"] == ["codex.py"]
    assert any(
        "5 tests passed" in value
        for value in result["last_turn"]["tests_mentioned_or_executed"]
    )
    assert before == after == 0
    assert any(
        method == "thread/read" and params.get("includeTurns") is True
        for method, params in protocol_client.request_calls
    )


def test_review_session_warns_when_tui_uses_other_thread(
    tmp_path, protocol_client
):
    manager = RunManager(tmp_path, protocol_client)
    manager.ensure_thread()
    manager._known_tui_thread_ids = lambda: ["different-thread-id-12345"]
    result = manager.review_session(turn_limit=1)
    assert result["threads_match"] is False
    assert "Thread do assistente: thread-shared" in result["thread_warning"]
    assert "Thread da TUI: different-thread-id-12345" in result["thread_warning"]
    assert "As sessoes nao sao iguais." in result["thread_warning"]
