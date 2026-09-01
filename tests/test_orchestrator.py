from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tern.orchestrator.agent import Supervisor, _is_codex_history_request
from tern.orchestrator.codex import CodexResult, CodexRunner, CodexSessionManager
from tern.orchestrator.config import load_settings
from tern.orchestrator.security import AccessDenied, ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry


class FakeCodex:
    timeout = 1

    def delegate_to_codex(self, **_arguments):
        return CodexResult(
            accepted=True,
            thread_id="thread-1",
            turn_id="turn-1",
            status="completed",
            final_response="feito",
            error=None,
            events=3,
        )

    def delegate(self, _task):
        raise AssertionError("not expected")

    def continue_session(self, **_arguments):
        raise AssertionError("not expected")

    def review_session(self, **_arguments):
        return {
            "ok": True,
            "operation": "thread/read",
            "thread_id": "thread-1",
            "turns_read": 1,
            "last_turn_state": "completed",
            "conversation_summary": "feito",
            "new_turn_started": False,
        }


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


def orchestration_response(action, **overrides):
    payload = {
        "action": action,
        "target_agent": None,
        "target": None,
        "tool_name": None,
        "arguments": {},
        "objective": "continue",
        "execution_mode": None,
        "required_capabilities": [],
        "reason_code": "SUFFICIENT_INFORMATION",
        "evidence_refs": [],
        "expected_observation": None,
        "confidence": 0.9,
        "short_horizon_hint": None,
    }
    payload.update(overrides)
    return response({"role": "assistant", "content": json.dumps(payload)})


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


def test_codex_visible_thread_id_is_loaded_from_runtime_environment():
    settings = load_settings({"CODEX_THREAD_ID": "thread-visible"})

    assert settings.codex_current_thread_id == "thread-visible"


@pytest.mark.parametrize("backend", ["qwen35", "qwen25-base", "qwen25-tq3p"])
def test_backends_are_explicit(backend):
    settings = load_settings({"MODEL_BACKEND": backend})
    assert settings.backend.name == backend


def test_local_model_aliases_are_model_agnostic(tmp_path):
    model = tmp_path / "candidate.gguf"
    runtime = tmp_path / "llama-server.exe"
    settings = load_settings(
        {
            "LOCAL_MODEL_PROVIDER": "qwen35",
            "LOCAL_MODEL_PATH": str(model),
            "LOCAL_MODEL_RUNTIME": str(runtime),
        }
    )
    assert settings.backend.name == "qwen35"
    assert settings.model_path == model.resolve()
    assert settings.server_executable == runtime.resolve()


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="MODEL_BACKEND"):
        load_settings({"MODEL_BACKEND": "automatic"})


def test_invalid_orchestration_mode_is_rejected():
    with pytest.raises(ValueError, match="ORCHESTRATION_MODE"):
        load_settings({"ORCHESTRATION_MODE": "unbounded"})


def test_supervisor_bounded_live_uses_new_loop_and_real_read(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("conteudo real", encoding="utf-8")
    client = Result(
        [
            orchestration_response(
                "INSPECT",
                target="sample.txt",
                tool_name="filesystem_read_text",
                arguments={"path": str(sample), "max_bytes": 4096},
                objective="read the requested file",
                execution_mode="READ_ONLY",
                reason_code="REPOSITORY_INSPECTION_REQUIRED",
            ),
            orchestration_response(
                "RESPOND",
                objective="O arquivo contém conteúdo real.",
                reason_code="GOAL_COMPLETED",
            ),
        ]
    )
    settings = load_settings(
        {
            "ORCHESTRATION_MODE": "bounded_live",
            "ORCHESTRATION_SHADOW_ENABLED": "true",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "3",
        }
    )

    result = Supervisor(settings, client, registry(tmp_path)).run(
        "analise o arquivo sample.txt"
    )

    assert result["ok"] is True
    assert result["tool_calls"] == 1
    assert result["orchestration"]["mode"] == "BOUNDED_LIVE"
    assert result["orchestration"]["termination_reason"] == "GOAL_COMPLETED"
    assert result["orchestration"]["effect_counts"]["tools_executed"] == 1
    assert result["answer"].endswith("O arquivo contém conteúdo real.")


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


def test_filesystem_list_can_consolidate_recursive_discovery(tmp_path):
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    (nested / "nested.txt").write_text("nested", encoding="utf-8")
    result = registry(tmp_path).execute(
        "filesystem_list",
        {"path": str(tmp_path), "recursive": True, "max_depth": 2},
    )
    assert result["ok"] and result["recursive"]
    paths = {item["relative_path"] for item in result["entries"]}
    assert {"root.txt", "one", "one/two", "one/two/nested.txt"} <= paths


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


def test_bounded_live_authority_avoids_redundant_overwrite_confirmation(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("old", encoding="utf-8")
    client = Result(
        [
            orchestration_response(
                "EXECUTE",
                target="sample.txt",
                tool_name="filesystem_write_text",
                arguments={
                    "directory": str(tmp_path),
                    "name": "sample.txt",
                    "content": "new",
                },
                objective="corrigir o conteudo do arquivo",
                execution_mode="MUTATION",
                reason_code="CODE_MUTATION_REQUIRED",
            )
        ]
    )
    settings = load_settings(
        {
            "ORCHESTRATION_MODE": "bounded_live",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "1",
        }
    )

    result = Supervisor(settings, client, registry(tmp_path)).run(
        "corrija o arquivo sample.txt e altere o conteudo para new"
    )

    record = result["orchestration"]["records"][0]
    assert record["authority_shadow_result"]["allowed"] is True
    assert record["authority_shadow_result"]["mutation_authorized"] is True
    assert record["observation"]["status"] == "SUCCESS"
    assert sample.read_text(encoding="utf-8") == "new"


def test_bounded_live_reruns_tests_after_authorized_mutation(tmp_path):
    module = tmp_path / "calculator.py"
    module.write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    client = Result(
        [
            orchestration_response(
                "EXECUTE",
                target="tests",
                tool_name="run_project_tests",
                arguments={"project_path": str(tmp_path)},
                objective="observar a falha dos testes",
                execution_mode="READ_ONLY",
                reason_code="REPOSITORY_INSPECTION_REQUIRED",
            ),
            orchestration_response(
                "EXECUTE",
                target="calculator.py",
                tool_name="filesystem_write_text",
                arguments={
                    "directory": str(tmp_path),
                    "name": "calculator.py",
                    "content": "def add(left, right):\n    return left + right\n",
                },
                objective="corrigir a soma",
                execution_mode="MUTATION",
                reason_code="CODE_MUTATION_REQUIRED",
            ),
            orchestration_response(
                "EXECUTE",
                target="tests",
                tool_name="run_project_tests",
                arguments={"project_path": str(tmp_path)},
                objective="verificar a correcao com os mesmos testes",
                execution_mode="READ_ONLY",
                reason_code="SUFFICIENT_INFORMATION",
            ),
        ]
    )
    settings = load_settings(
        {
            "ORCHESTRATION_MODE": "bounded_live",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "4",
        }
    )

    result = Supervisor(settings, client, registry(tmp_path)).run(
        f"no projeto {tmp_path}, descubra por que o teste falha, corrija e verifique"
    )

    records = result["orchestration"]["records"]
    assert [item["next_action"]["action"] for item in records] == [
        "EXECUTE",
        "EXECUTE",
        "EXECUTE",
        "RESPOND",
    ]
    assert records[0]["observation"]["verification_status"] == "FAILED"
    assert records[2]["observation"]["verification_status"] == "VERIFIED"
    assert records[3]["decision_source"] == "FAST_PATH"
    assert result["orchestration"]["termination_reason"] == "GOAL_COMPLETED"
    assert result["tool_calls"] == 3


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


def test_repeated_call_without_progress_forces_replan(tmp_path):
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
    client = Result(
        [
            response(call),
            response(call),
            response({"role": "assistant", "content": "Usei o que ja encontrei."}),
        ]
    )
    result = Supervisor(load_settings({}), client, registry(tmp_path)).run("Repita")
    assert result["ok"]
    assert result["tool_calls"] == 2


def test_second_filesystem_list_with_different_path_can_make_progress(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()

    def list_call(identifier, path):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": identifier,
                    "type": "function",
                    "function": {
                        "name": "filesystem_list",
                        "arguments": json.dumps({"path": str(path)}),
                    },
                }
            ],
        }

    client = Result(
        [
            response(list_call("root", tmp_path)),
            response(list_call("nested", nested)),
            response({"role": "assistant", "content": "Exploração concluída"}),
        ]
    )
    result = Supervisor(load_settings({}), client, registry(tmp_path)).run(
        "Explore repetidamente"
    )
    assert result["ok"]
    assert result["tool_calls"] == 2


def test_tool_call_after_access_denied_is_not_executed(tmp_path):
    outside = tmp_path.parent

    def list_call(identifier, path):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": identifier,
                    "type": "function",
                    "function": {
                        "name": "filesystem_list",
                        "arguments": json.dumps({"path": str(path)}),
                    },
                }
            ],
        }

    tools = registry(tmp_path)
    client = Result(
        [
            response(list_call("denied", outside)),
            response(list_call("must-not-run", tmp_path)),
            response({"role": "assistant", "content": "Fluxo encerrado."}),
        ]
    )
    result = Supervisor(load_settings({}), client, tools).run("Liste pastas")
    assert result["ok"]
    assert result["answer"] == "Fluxo encerrado."
    assert result["tool_calls"] == 1
    records = [
        json.loads(line)
        for line in tools.logger.path.read_text(encoding="utf-8").splitlines()
    ]
    executed = [item for item in records if item.get("event") == "tool_result"]
    assert len(executed) == 1
    assert executed[0]["result"]["error"] == "AccessDenied"


def test_batched_calls_after_denial_are_acknowledged_but_not_executed(tmp_path):
    batch = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "denied",
                "type": "function",
                "function": {
                    "name": "filesystem_list",
                    "arguments": json.dumps({"path": str(tmp_path.parent)}),
                },
            },
            {
                "id": "skipped",
                "type": "function",
                "function": {
                    "name": "filesystem_list",
                    "arguments": json.dumps({"path": str(tmp_path)}),
                },
            },
        ],
    }

    class CapturingClient:
        def __init__(self):
            self.calls = 0
            self.final_messages = None

        def chat(self, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return response(batch)
            self.final_messages = list(messages)
            return response({"role": "assistant", "content": "Bloqueado com seguranca"})

    tools = registry(tmp_path)
    client = CapturingClient()
    result = Supervisor(load_settings({}), client, tools).run("Liste pastas")
    assert result["ok"] and result["tool_calls"] == 1
    tool_messages = [
        item for item in client.final_messages if item.get("role") == "tool"
    ]
    assert [item["tool_call_id"] for item in tool_messages] == [
        "denied",
        "skipped",
    ]
    assert json.loads(tool_messages[1]["content"])["error"] == "tools_disabled"
    records = [
        json.loads(line)
        for line in tools.logger.path.read_text(encoding="utf-8").splitlines()
    ]
    executed = [item for item in records if item.get("event") == "tool_result"]
    assert len(executed) == 1


def test_codex_uses_project_session_manager(tmp_path):
    observed = {}

    class Manager:
        class Log:
            def write(self, *_args, **_kwargs):
                return None

        bridge_log = Log()

        def list_project_threads(self):
            return []

        def create_thread(self):
            return {
                "thread_id": "thread-1",
                "session_id": "thread-1",
                "project": str(tmp_path),
                "state": "idle",
                "source": "appServer",
                "visible": False,
                "recoverable": True,
                "ephemeral": False,
            }

        def adopt_thread(self, thread_id):
            assert thread_id == "thread-1"
            return self.create_thread()

        def run_turn(self, task, **kwargs):
            observed.update(task=task, **kwargs)
            return CodexResult(
                accepted=True,
                thread_id="thread-1",
                turn_id="turn-1",
                status="completed",
                final_response="87",
                error=None,
                events=7,
            )

    runner = CodexRunner(PathPolicy((tmp_path,)), executable="codex")
    runner._managers[tmp_path.resolve()] = Manager()
    result = runner.delegate_to_codex(
        task="contar testes",
        project_path=str(tmp_path),
        continue_current_thread=True,
    )
    assert observed["task"] == "contar testes"
    assert observed["origin"] == "qwen"
    assert observed["continue_current_thread"] is True
    assert callable(observed["event_callback"])
    assert result.ok and result.thread_id == "thread-1"


def test_qwen_codex_tool_is_small_and_returns_thread(tmp_path):
    tools = registry(tmp_path)
    specs = {item["function"]["name"]: item for item in tools.specs()}
    assert "delegate_to_codex" in specs
    assert "codex_delegate" not in specs
    assert set(
        specs["delegate_to_codex"]["function"]["parameters"]["properties"]
    ) == {
        "task",
        "project_path",
        "continue_current_thread",
        "thread_id",
        "wait",
    }
    result = tools.execute(
        "delegate_to_codex",
        {
            "task": "Leia o README",
            "project_path": str(tmp_path),
            "continue_current_thread": True,
        },
    )
    assert result["accepted"]
    assert result["thread_id"] == "thread-1"
    assert result["turn_id"] == "turn-1"


def test_session_identity_changes_do_not_rewrite_delegation_payload(tmp_path):
    captured = []

    class CapturingCodex(FakeCodex):
        def delegate_to_codex(self, **arguments):
            captured.append(arguments)
            return super().delegate_to_codex(**arguments)

    tools = ToolRegistry(
        policy=PathPolicy((tmp_path,)),
        logger=ActionLogger(tmp_path / "actions.jsonl"),
        codex=CapturingCodex(),
        max_output_bytes=131072,
        approval=lambda _action, _arguments: True,
    )
    for number, thread_id in enumerate(("thread-a", "thread-b"), 1):
        result = tools.execute(
            "delegate_to_codex",
            {
                "task": "implemente PAYLOAD-IDÊNTICO",
                "project_path": str(tmp_path),
                "thread_id": thread_id,
            },
            context={
                "turn_id": f"turn-{number}",
                "conversation_id": "conversation-1",
            },
        )
        assert result["ok"], result
    assert captured[0]["task"] == captured[1]["task"]
    assert captured[0]["thread_id"] == "thread-a"
    assert captured[1]["thread_id"] == "thread-b"


def test_review_codex_session_tool_has_safe_defaults(tmp_path):
    tools = registry(tmp_path)
    specs = {item["function"]["name"]: item for item in tools.specs()}
    schema = specs["review_codex_session"]["function"]["parameters"]
    assert schema["required"] == []
    assert set(schema["properties"]) == {"project_path", "turn_limit"}
    result = tools.execute(
        "review_codex_session",
        {"project_path": str(tmp_path), "turn_limit": 10},
    )
    assert result["operation"] == "thread/read"
    assert not result["new_turn_started"]


@pytest.mark.parametrize(
    "user_text",
    [
        "Leia as ultimas informacoes do Codex.",
        "O que aconteceu na sessao do Codex?",
        "Resuma os ultimos turns do Codex.",
        "Faca uma revisao da ultima sessao do Codex.",
        "O que o Codex fez por ultimo?",
        "De uma olhada na tarefa do Codex.",
        "Faca uma vistoria do trabalho do Codex.",
        "Revise o resultado da tarefa do Codex.",
    ],
)
def test_codex_history_intent_is_detected(user_text):
    assert _is_codex_history_request(user_text)


@pytest.mark.parametrize(
    "user_text",
    [
        "Peca ao Codex para revisar codex.py.",
        "Use o Codex para executar os testes.",
        "Codex, analise e corrija este bug.",
        "Resuma este texto sobre o Codex.",
        "Pesquise noticias sobre o Codex.",
    ],
)
def test_normal_codex_actions_are_not_history_requests(user_text):
    assert not _is_codex_history_request(user_text)


def test_explicit_codex_delegate_bypasses_qwen_and_calls_codex_directly(tmp_path):
    observed_tools = []

    class CapturingClient(Result):
        def chat(self, _messages, **kwargs):
            observed_tools.extend(
                item["function"]["name"] for item in kwargs["tools"]
            )
            return next(self.values)

    client = CapturingClient(
        [response({"role": "assistant", "content": "Posso ajudar."})]
    )
    result = Supervisor(load_settings({}), client, registry(tmp_path)).run(
        "Peca ao Codex para revisar codex.py."
    )
    assert result["ok"]
    assert observed_tools == []
    assert result["tool_calls"] == 1
    assert result["answer"] == "feito"
    assert result["decision"]["fast_path"] is True


def test_codex_history_exposes_only_review_tool_and_does_not_repeat(tmp_path):
    observed_tools = []
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "review-one",
                "type": "function",
                "function": {
                    "name": "review_codex_session",
                    "arguments": json.dumps(
                        {"project_path": str(tmp_path), "turn_limit": 10}
                    ),
                },
            }
        ],
    }

    class CapturingClient(Result):
        def chat(self, _messages, **kwargs):
            tools = kwargs.get("tools")
            observed_tools.append(
                None
                if tools is None
                else [item["function"]["name"] for item in tools]
            )
            return next(self.values)

    client = CapturingClient(
        [
            response(call),
            response({"role": "assistant", "content": "Resumo real"}),
        ]
    )
    result = Supervisor(load_settings({}), client, registry(tmp_path)).run(
        "Quero que voce leia as ultimas informacoes do Codex e me forneca um resumo."
    )
    assert result["ok"] and result["tool_calls"] == 1
    assert observed_tools == [["review_codex_session"], None]


def test_human_waiter_has_priority_over_qwen_waiter(tmp_path):
    manager = CodexSessionManager(tmp_path)
    active = manager._queue_acquire("qwen")
    order = []

    def wait(origin):
        ticket = manager._queue_acquire(origin)
        order.append(origin)
        manager._queue_release(ticket)

    qwen = threading.Thread(target=wait, args=("qwen",))
    human = threading.Thread(target=wait, args=("human",))
    qwen.start()
    deadline = time.monotonic() + 2
    while len(manager._queue) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    human.start()
    deadline = time.monotonic() + 2
    while len(manager._queue) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    manager._queue_release(active)
    qwen.join(2)
    human.join(2)
    assert order == ["human", "qwen"]


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
        load_settings({}),
        Result(
            [
                response(call),
                response(
                    {
                        "role": "assistant",
                        "content": "Ação cancelada pelo usuário.",
                    }
                ),
            ]
        ),
        tools,
    ).run("apague")
    assert result["ok"]
    assert "cancelada" in result["answer"]
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
