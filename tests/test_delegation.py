from __future__ import annotations

import json
from pathlib import Path

from tern.orchestrator.codex import CodexResult
from tern.orchestrator.delegation import (
    DELEGATION_REQUEST_SCHEMA,
    DelegationRequest,
)
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry


def test_delegation_request_serialization_is_stable_and_preserves_exact_task():
    original = "  Edite apenas auth.py.\nNão altere os testes.  "
    context = {
        "original_user_text": original,
        "delegation_constraints": ["read_only", "no_tests", "read_only"],
        "delegation_references": ["auth.py", "README.md"],
        "execution_mode": "READ_ONLY",
        "delegation_action": "REMOVE_COMPONENT",
        "requested_agent_source": "explicit_user",
    }

    first = DelegationRequest.build(
        requested_agent="codex",
        submitted_task="reescreva o projeto inteiro",
        project_path=r"D:\JARVIS",
        context=context,
    )
    second = DelegationRequest.build(
        requested_agent="codex",
        submitted_task="outra invenção do roteador",
        project_path=r"D:\JARVIS",
        context=context,
    )

    assert first.serialize() == second.serialize()
    payload = json.loads(first.serialize())
    assert payload["schema"] == DELEGATION_REQUEST_SCHEMA
    assert payload["task"] == original
    assert payload["constraints"] == ["no_tests", "read_only"]
    assert payload["action"] == "REMOVE_COMPONENT"
    assert payload["references"] == ["README.md", "auth.py"]
    assert payload["source"] == "original_user_text"
    assert "reescreva o projeto inteiro" not in first.serialize()


def test_delegation_request_falls_back_to_direct_tool_argument():
    request = DelegationRequest.build(
        requested_agent="deepseek",
        submitted_task="analise o desenho",
        project_path=None,
        context={},
    )

    assert request.task == "analise o desenho"
    assert request.source == "tool_argument"


def test_codex_receives_original_request_instead_of_generated_task(tmp_path: Path):
    captured: list[dict[str, object]] = []

    class CapturingCodex:
        timeout = 1

        def shared_project(self):
            return tmp_path

        def delegate_to_codex(self, **arguments):
            captured.append(arguments)
            return CodexResult(
                accepted=True,
                thread_id="thread-1",
                turn_id="turn-1",
                status="completed",
                final_response="feito",
                error=None,
                events=1,
            )

    tools = ToolRegistry(
        policy=PathPolicy((tmp_path,)),
        logger=ActionLogger(tmp_path / "actions.jsonl"),
        codex=CapturingCodex(),
        max_output_bytes=131072,
        approval=lambda _action, _arguments: True,
    )
    original = "Implemente somente auth.py e não altere nenhum teste."

    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "reescreva todos os módulos",
            "project_path": str(tmp_path),
            "wait": True,
        },
        context={
            "user_text": original,
            "original_user_text": original,
            "delegation_constraints": ["forbid_test_execution"],
            "delegation_action": "REMOVE_COMPONENT",
            "turn_id": "turn-preservation",
        },
    )

    assert result["accepted"]
    payload = json.loads(str(captured[0]["task"]))
    assert payload["requested_agent"] == "codex"
    assert payload["action"] == "REMOVE_COMPONENT"
    assert payload["task"] == original
    assert payload["scope"] == {"project_path": str(tmp_path.resolve())}
    assert "reescreva todos os módulos" not in str(captured[0]["task"])
