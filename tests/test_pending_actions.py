from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.codex import CodexResult
from tern.orchestrator.config import load_settings
from tern.orchestrator.pending_actions import PendingActionStore
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry
from tern.orchestrator.voice.policy import (
    ConfirmationDecision,
    VoiceActionApprover,
    confirm_transcription,
)


class FakeCodex:
    timeout = 30

    def __init__(self, project: Path, *, fail: bool = False):
        self.project = project.resolve()
        self.fail = fail
        self.calls: list[dict] = []

    def shared_project(self):
        return self.project

    def delegate_to_codex(self, **arguments):
        self.calls.append(arguments)
        if self.fail:
            raise RuntimeError("codex unavailable")
        return CodexResult(
            accepted=True,
            thread_id="shared-thread",
            turn_id="turn-one",
            status="completed",
            final_response="done",
            error=None,
        )

    def review_session(self, **_arguments):
        return {"ok": True, "operation": "thread/read"}


class BlockingCodex(FakeCodex):
    def __init__(self, project: Path):
        super().__init__(project)
        self.entered = threading.Event()
        self.release = threading.Event()

    def delegate_to_codex(self, **arguments):
        self.calls.append(arguments)
        self.entered.set()
        assert self.release.wait(3)
        return CodexResult(
            accepted=True,
            thread_id="shared-thread",
            turn_id="turn-one",
            status="completed",
            final_response="done",
            error=None,
        )


class Responses:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        return next(self.values)


class Console:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.values: list[str] = []

    def write(self, value=""):
        self.values.append(value)

    def read(self, prompt=""):
        self.values.append(prompt)
        return next(self.answers)


def response(message):
    return {"choices": [{"message": message}], "usage": {"total_tokens": 1}}


def delegate_call(project: Path, *, task: str = "Implemente a melhoria"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "delegate-one",
                "type": "function",
                "function": {
                    "name": "delegate_to_codex",
                    "arguments": json.dumps(
                        {
                            "task": task,
                            "project_path": str(project),
                            "continue_current_thread": True,
                        }
                    ),
                },
            }
        ],
    }


def make_registry(
    state_dir: Path,
    project: Path,
    *,
    approval=None,
    codex: FakeCodex | None = None,
    roots: tuple[Path, ...] | None = None,
    timeout: int = 300,
):
    codex = codex or FakeCodex(project)
    return ToolRegistry(
        policy=PathPolicy(roots or (project,)),
        logger=ActionLogger(state_dir / "actions.jsonl"),
        codex=codex,
        max_output_bytes=131072,
        approval=approval,
        confirmation_timeout_seconds=timeout,
    ), codex


def test_confirmation_blocks_agent_and_is_shown_once(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    entered = threading.Event()
    release = threading.Event()
    approvals = []

    def approval(action, arguments):
        approvals.append((action, arguments))
        entered.set()
        assert release.wait(3)
        return True

    tools, codex = make_registry(tmp_path, project, approval=approval)
    client = Responses(
        [
            response(delegate_call(project)),
            response({"role": "assistant", "content": "completed"}),
        ]
    )
    values = {}
    worker = threading.Thread(
        target=lambda: values.setdefault(
            "result",
            Supervisor(load_settings({}), client, tools).run(
                "Cuide desta tarefa para mim."
            ),
        )
    )
    worker.start()
    assert entered.wait(3)
    assert client.calls == 1
    assert len(approvals) == 1
    assert not codex.calls
    pending = tools.pending_actions.pending()
    assert pending["status"] == "awaiting_confirmation"
    duplicate = tools.execute(
        "delegate_to_codex",
        {
            "task": "Implemente a melhoria",
            "project_path": str(project),
            "continue_current_thread": True,
        },
        context={
            "turn_id": pending["turn_id"],
            "user_text": "Cuide desta tarefa para mim.",
        },
    )
    assert duplicate["error"] == "duplicate_tool_call_blocked"
    assert len(approvals) == 1
    release.set()
    worker.join(5)
    assert values["result"]["ok"]
    assert client.calls == 2
    assert len(codex.calls) == 1
    assert tools.pending_actions.pending() is None


def test_cancelled_first_tool_stops_remaining_calls_in_same_plan(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tools, codex = make_registry(
        tmp_path,
        project,
        approval=lambda *_values: False,
    )
    first = delegate_call(project)
    first["tool_calls"].append(
        {
            "id": "delegate-two",
            "type": "function",
            "function": {
                "name": "delegate_to_codex",
                "arguments": json.dumps(
                    {
                        "task": "Implemente outra melhoria",
                        "project_path": str(project),
                        "continue_current_thread": True,
                    }
                ),
            },
        }
    )
    client = Responses(
        [
            response(first),
            response({"role": "assistant", "content": "cancelled"}),
        ]
    )
    result = Supervisor(load_settings({}), client, tools).run(
        "Avalie mudanças possíveis."
    )
    assert result["ok"]
    assert result["tool_calls"] == 1
    assert not codex.calls


def test_confirm_executes_stored_arguments_once(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tools, codex = make_registry(tmp_path, project)
    arguments = {
        "task": "original exact task",
        "project_path": str(project),
        "continue_current_thread": True,
    }
    record, state = tools.pending_actions.prepare(
        action_id="action-confirm",
        tool="delegate_to_codex",
        arguments=arguments,
        project=str(project),
        risk="codex_modify_files",
        confirmation_required=True,
        turn_id="turn-confirm",
    )
    assert state == "created"
    result = tools.confirm_pending_action(record["action_id"])
    assert result["ok"]
    assert len(codex.calls) == 1
    assert codex.calls[0]["task"] == "original exact task"
    duplicate = tools.confirm_pending_action(record["action_id"])
    assert duplicate["error"] == "duplicate_tool_call_blocked"
    assert len(codex.calls) == 1


def test_two_confirmations_cannot_claim_same_action(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    codex = BlockingCodex(project)
    tools, _ = make_registry(tmp_path, project, codex=codex)
    record, _ = tools.pending_actions.prepare(
        action_id="atomic-confirm",
        tool="delegate_to_codex",
        arguments={
            "task": "stored task",
            "project_path": str(project),
            "continue_current_thread": True,
        },
        project=str(project),
        risk="codex_modify_files",
        confirmation_required=True,
        turn_id="atomic-turn",
    )
    first_result = {}
    worker = threading.Thread(
        target=lambda: first_result.update(
            tools.confirm_pending_action(record["action_id"])
        )
    )
    worker.start()
    assert codex.entered.wait(3)
    second = tools.confirm_pending_action(record["action_id"])
    assert second["error"] == "duplicate_tool_call_blocked"
    assert second["status"] == "executing"
    codex.release.set()
    worker.join(5)
    assert first_result["ok"]
    assert len(codex.calls) == 1


@pytest.mark.parametrize("answer", ["CANCELAR", "nao", "S", ""])
def test_non_confirm_text_cancels_action_once(tmp_path, answer):
    project = tmp_path / "project"
    project.mkdir()
    console = Console([answer])
    tools, codex = make_registry(
        tmp_path,
        project,
        approval=VoiceActionApprover(console),
    )
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "Implemente algo",
            "project_path": str(project),
            "continue_current_thread": True,
        },
        context={"turn_id": f"turn-{answer}", "user_text": "avalie isto"},
    )
    assert result["error"] == "ActionCancelled"
    assert not codex.calls
    assert sum(value == "[ação pendente]" for value in console.values) == 1
    assert tools.pending_actions.pending() is None


def test_voice_transcription_confirmation_does_not_approve_action():
    transcription_console = Console(["S"])
    assert (
        confirm_transcription(
            "corrija o projeto",
            required=True,
            console=transcription_console,
        )
        == ConfirmationDecision.SEND
    )
    action_console = Console(["S"])
    assert not VoiceActionApprover(action_console)(
        "codex_modify_files",
        {
            "action_id": "one",
            "tool": "delegate_to_codex",
            "path": r"D:\tern",
        },
    )


def test_interrupted_confirmation_is_discarded_on_exit(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    def interrupted(*_values):
        raise KeyboardInterrupt

    tools, codex = make_registry(tmp_path, project, approval=interrupted)
    with pytest.raises(KeyboardInterrupt):
        tools.execute(
            "delegate_to_codex",
            {
                "task": "Implemente algo",
                "project_path": str(project),
                "continue_current_thread": True,
            },
            context={"turn_id": "exit", "user_text": "Avalie isto"},
        )
    assert tools.pending_actions.pending() is None
    assert tools.pending_actions.history()[-1]["status"] == "cancelled"
    assert not codex.calls


def test_passive_read_calls_are_left_to_progress_tracker(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tools, _codex = make_registry(tmp_path, project)
    arguments = {"path": str(project)}
    context = {"turn_id": "same-turn", "user_text": "liste"}
    assert tools.execute("filesystem_list", arguments, context=context)["ok"]
    second = tools.execute("filesystem_list", arguments, context=context)
    assert second["ok"]


def test_pending_action_survives_registry_reconnection(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    first, _ = make_registry(tmp_path, project)
    arguments = {
        "task": "stored",
        "project_path": str(project),
        "continue_current_thread": True,
    }
    first.pending_actions.prepare(
        action_id="reconnect-action",
        tool="delegate_to_codex",
        arguments=arguments,
        project=str(project),
        risk="codex_modify_files",
        confirmation_required=True,
        turn_id="original-turn",
    )
    second_codex = FakeCodex(project)
    second, _ = make_registry(tmp_path, project, codex=second_codex)
    result = second.confirm_pending_action("reconnect-action")
    assert result["ok"]
    assert len(second_codex.calls) == 1


@pytest.mark.parametrize("task", ["Revise o codigo", "Analise a arquitetura"])
def test_read_only_codex_delegation_needs_no_confirmation(tmp_path, task):
    project = tmp_path / "project"
    project.mkdir()
    approvals = []
    tools, codex = make_registry(
        tmp_path,
        project,
        approval=lambda *values: approvals.append(values) or False,
    )
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": task,
            "project_path": str(project),
            "continue_current_thread": True,
        },
        context={"turn_id": task, "user_text": task},
    )
    assert result["ok"]
    assert not approvals
    assert len(codex.calls) == 1


def test_explicit_project_modification_needs_no_redundant_confirmation(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    approvals = []
    tools, codex = make_registry(
        tmp_path,
        project,
        approval=lambda *values: approvals.append(values) or False,
    )
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "Corrija o bug e adicione testes",
            "project_path": str(project),
            "continue_current_thread": True,
        },
        context={
            "turn_id": "explicit-modification",
            "user_text": "Peça ao Codex para corrigir esse bug no Jarvis.",
        },
    )
    assert result["ok"]
    assert not approvals
    assert len(codex.calls) == 1


def test_outside_project_requires_one_confirmation(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    approvals = []
    tools, codex = make_registry(
        tmp_path,
        project,
        roots=(project, outside),
        approval=lambda action, args: approvals.append((action, args)) or True,
    )
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "Modifique o arquivo permitido",
            "project_path": str(outside),
            "continue_current_thread": True,
        },
        context={
            "turn_id": "outside",
            "user_text": f"Modifique o projeto {outside}",
        },
    )
    assert result["ok"]
    assert [value[0] for value in approvals] == ["outside_project"]
    assert len(codex.calls) == 1
    assert codex.calls[0]["project_path"] == str(outside.resolve())


def test_shared_project_replaces_unmentioned_home_directory(tmp_path):
    project = Path(r"D:\tern")
    tools, codex = make_registry(tmp_path, project)
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "Analise a repeticao de ferramentas",
            "project_path": r"C:\Users\User",
            "continue_current_thread": True,
        },
        context={
            "turn_id": "default-project",
            "user_text": "Peça ao Codex para analisar a repetição de ferramentas.",
        },
    )
    assert result["ok"]
    assert codex.calls[0]["project_path"] == str(project.resolve())
    assert codex.calls[0]["project_path"] != str(Path.home())


def test_codex_failure_clears_pending_state(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    codex = FakeCodex(project, fail=True)
    tools, _ = make_registry(tmp_path, project, codex=codex)
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "Analise o bridge",
            "project_path": str(project),
            "continue_current_thread": True,
        },
        context={"turn_id": "failure", "user_text": "Analise o bridge"},
    )
    assert result["error"] == "RuntimeError"
    assert tools.pending_actions.pending() is None
    assert tools.pending_actions.history()[-1]["status"] == "failed"


def test_pending_confirmation_expires(tmp_path):
    store = PendingActionStore(tmp_path, timeout_seconds=1)
    store.prepare(
        action_id="expires",
        tool="delegate_to_codex",
        arguments={"task": "x"},
        project=str(tmp_path),
        risk="outside_project",
        confirmation_required=True,
        turn_id="turn",
    )
    time.sleep(1.05)
    assert store.pending() is None
    assert store.history()[-1]["status"] == "expired"


def test_confirmation_timeout_is_configurable():
    assert (
        load_settings({"ACTION_CONFIRMATION_TIMEOUT_SECONDS": "17"})
        .action_confirmation_timeout_seconds
        == 17
    )


def test_expired_confirmation_is_not_executed(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tools, codex = make_registry(tmp_path, project, timeout=1)
    record, _ = tools.pending_actions.prepare(
        action_id="expired-confirm",
        tool="delegate_to_codex",
        arguments={
            "task": "stored task",
            "project_path": str(project),
            "continue_current_thread": True,
        },
        project=str(project),
        risk="outside_project",
        confirmation_required=True,
        turn_id="expired-turn",
    )
    time.sleep(1.05)
    result = tools.confirm_pending_action(record["action_id"])
    assert result["error"] == "ActionExpired"
    assert not codex.calls


def test_orphaned_execution_is_retired_without_reexecution(tmp_path):
    store = PendingActionStore(tmp_path, timeout_seconds=1)
    record, _ = store.prepare(
        action_id="orphaned",
        tool="delegate_to_codex",
        arguments={"task": "already started"},
        project=str(tmp_path),
        risk="normal",
        confirmation_required=False,
        turn_id="crashed-turn",
    )
    store.claim_execution(record["action_id"])
    state = json.loads(store.path.read_text(encoding="utf-8"))
    state["pending"]["owner_pid"] = 99999999
    store.path.write_text(json.dumps(state), encoding="utf-8")
    time.sleep(1.05)
    assert store.pending() is None
    retired = store.history()[-1]
    assert retired["status"] == "failed"
    assert retired["error"] == "orphaned_execution"
