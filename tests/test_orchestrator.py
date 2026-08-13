from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.codex import CodexRunner
from tern.orchestrator.config import load_settings
from tern.orchestrator.security import AccessDenied, ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry


class FakeCodex:
    timeout = 1

    def delegate(self, _task):
        raise AssertionError("not expected")

    def continue_session(self, **_arguments):
        raise AssertionError("not expected")


class Result:
    def __init__(self, values):
        self.values = iter(values)

    def chat(self, _messages, **_kwargs):
        return next(self.values)


def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(
        policy=PathPolicy((tmp_path,)),
        logger=ActionLogger(tmp_path / "actions.jsonl"),
        codex=FakeCodex(),
        max_output_bytes=131072,
    )


def response(message):
    return {"choices": [{"message": message}], "usage": {"total_tokens": 1}}


def test_qwen35_is_default_and_no_missing_model_fallback():
    settings = load_settings({})
    assert settings.backend.name == "qwen35"
    assert settings.model_path.name == "Qwen_Qwen3.5-4B-Q4_K_M.gguf"
    assert settings.context_size == 16384
    assert settings.gpu_layers == 99
    assert settings.kv_cache_k == settings.kv_cache_v == "q8_0"
    assert settings.parallel_slots == 1
    command = settings.server_command()
    assert "-fa" in command and command[command.index("-fa") + 1] == "on"
    assert "--jinja" in command


@pytest.mark.parametrize("backend", ["qwen35", "qwen25-base", "qwen25-tq3p"])
def test_backends_are_explicit(backend):
    settings = load_settings({"MODEL_BACKEND": backend})
    assert settings.backend.name == backend


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="MODEL_BACKEND"):
        load_settings({"MODEL_BACKEND": "automatic"})


def test_path_policy_allows_root_and_blocks_parent(tmp_path):
    policy = PathPolicy((tmp_path,))
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("ok", encoding="utf-8")
    assert policy.resolve(str(allowed)) == allowed.resolve()
    with pytest.raises(AccessDenied):
        policy.resolve(str(tmp_path / ".." / "outside.txt"), must_exist=False)


def test_unknown_tool_and_invalid_arguments(tmp_path):
    tools = registry(tmp_path)
    assert tools.execute("shell", {})["error"] == "unknown_tool"
    invalid = tools.execute("filesystem_read_text", {"path": str(tmp_path)})
    assert invalid["error"] == "invalid_arguments"


def test_allowed_read_and_outside_block(tmp_path):
    tools = registry(tmp_path)
    sample = tmp_path / "sample.txt"
    sample.write_text("ola", encoding="utf-8")
    good = tools.execute("filesystem_read_text", {"path": str(sample), "max_bytes": 4096})
    assert good["ok"] and good["content"] == "ola"
    bad = tools.execute(
        "filesystem_read_text",
        {"path": str(tmp_path / ".." / "secret.txt"), "max_bytes": 4096},
    )
    assert not bad["ok"] and bad["error"] == "AccessDenied"


def test_overwrite_requires_confirmation(tmp_path):
    tools = registry(tmp_path)
    sample = tmp_path / "sample.txt"
    sample.write_text("old", encoding="utf-8")
    result = tools.execute(
        "filesystem_write_text",
        {"directory": str(tmp_path), "name": "sample.txt", "content": "new"},
    )
    assert result["error"] == "ApprovalRequired"
    assert sample.read_text(encoding="utf-8") == "old"


def test_simple_answer_without_tool(tmp_path):
    settings = load_settings({"MODEL_MAX_TOOL_CALLS": "2"})
    client = Result([response({"role": "assistant", "content": "Ola!"})])
    result = Supervisor(settings, client, registry(tmp_path)).run("Oi")
    assert result["ok"] and result["answer"] == "Ola!"
    assert result["tool_calls"] == 0


def test_structured_tool_call(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("conteudo", encoding="utf-8")
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "one",
                "type": "function",
                "function": {
                    "name": "filesystem_read_text",
                    "arguments": json.dumps({"path": str(sample), "max_bytes": 4096}),
                },
            }
        ],
    }
    client = Result([response(call), response({"role": "assistant", "content": "Confirmado"})])
    result = Supervisor(load_settings({}), client, registry(tmp_path)).run("Leia")
    assert result["ok"] and result["tool_calls"] == 1


def test_repeated_call_prevents_loop(tmp_path):
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "same",
                "type": "function",
                "function": {
                    "name": "filesystem_list",
                    "arguments": json.dumps({"path": str(tmp_path)}),
                },
            }
        ],
    }
    client = Result([response(call), response(call)])
    result = Supervisor(load_settings({}), client, registry(tmp_path)).run("Repita")
    assert result["error"] == "loop_prevented"


def test_codex_does_not_inherit_voice_stdin(tmp_path, monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        stdout = "\n".join(
            [
                json.dumps(
                    {"type": "thread.started", "thread_id": "session-1"}
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "87",
                        },
                    }
                ),
            ]
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CodexRunner(
        PathPolicy((tmp_path,)), executable="codex"
    ).delegate(
        {
            "working_directory": str(tmp_path),
            "task": "contar testes",
            "context": [],
            "constraints": ["nao alterar arquivos"],
            "acceptance_criteria": ["informar total"],
            "validation": ["pytest --collect-only"],
        }
    )
    assert observed["stdin"] is subprocess.DEVNULL
    assert result.ok and result.session_id == "session-1"


def test_denied_destructive_action_stops_tool_loop(tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("keep", encoding="utf-8")
    tools = registry(tmp_path)
    tools.approval = lambda _action, _arguments: False
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "delete-one",
                "type": "function",
                "function": {
                    "name": "filesystem_delete",
                    "arguments": json.dumps({"path": str(target)}),
                },
            }
        ],
    }
    result = Supervisor(
        load_settings({}), Result([response(call)]), tools
    ).run("apague")
    assert result["error"] == "approval_required"
    assert target.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("task", "action"),
    [
        ("Instale este pacote", "install_software"),
        ("Desinstale o programa", "remove_software"),
        ("Altere a configuração do sistema", "system_change"),
        ("Execute como administrador", "administrative"),
        ("Corrija o código", "codex_modify_files"),
    ],
)
def test_codex_sensitive_tasks_are_classified(task, action):
    assert ToolRegistry._codex_sensitive_action(task) == action


def test_read_only_codex_task_needs_no_extra_approval():
    assert (
        ToolRegistry._codex_sensitive_action(
            "Revise os testes e não alterar arquivos"
        )
        is None
    )
