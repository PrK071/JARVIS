from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.agent_selection import SelectionSource
from tern.orchestrator.autonomy_foundation import Agent
from tern.orchestrator.codex import CodexResult, CodexRunner, _sandbox_policy
from tern.orchestrator.codex_jobs import CodexJobStore
from tern.orchestrator.config import load_settings
from tern.orchestrator.execution_authority import (
    AUTHORITATIVE_AGENTS,
    AuthorityBlockReason,
    AuthorityScope,
    AvailabilitySample,
    ExecutionAuthorityController,
    ExecutionAuthorityMode,
    availability_sample_from_gate,
    explicit_authority_scope,
    probe_agent_availability,
)
from tern.orchestrator.execution_gate import (
    ExecutionBlockReason,
    ExecutionGate,
    ExecutionGateInput,
    ExecutionMode,
)
from tern.orchestrator.security import PathPolicy


TOOL_NAMES = (
    "resolve_project",
    "find_project_files",
    "filesystem_list",
    "filesystem_read_text",
    "filesystem_write_text",
    "review_codex_session",
    "get_codex_job_status",
    "delegate_to_codex",
    "steer_codex_job",
    "cancel_codex_job",
    "review_deepseek_session",
    "delegate_to_deepseek",
)


# --- gate/authority helpers ----------------------------------------------------


def gate(**overrides):
    base = dict(
        execution_requested=True,
        candidate_agent=Agent.CODEX,
        selection_source=SelectionSource.EXPLICIT_USER,
        eligible_agents=(Agent.CODEX, Agent.DEEPSEEK, Agent.LOCAL),
        available_eligible_agents=(Agent.CODEX, Agent.DEEPSEEK, Agent.LOCAL),
        requested_agent=Agent.CODEX,
        requested_agent_source="EXPLICIT_USER",
        selection_reason_code="EXPLICIT_AGENT_READY",
    )
    base.update(overrides)
    return ExecutionGate().evaluate(ExecutionGateInput(**base))


def controller(mode=ExecutionAuthorityMode.EXPLICIT_USER, **kwargs):
    return ExecutionAuthorityController(mode, **kwargs)


def decide(mode=ExecutionAuthorityMode.EXPLICIT_USER, source="explicit_user", **overrides):
    decision = gate(**overrides)
    return controller(mode).decide(
        decision,
        requested_agent=decision.requested_agent,
        requested_agent_source=source,
        availability_at_selection=availability_sample_from_gate(decision),
    )


# --- scope ---------------------------------------------------------------------


def test_scope_requires_explicit_user_source():
    assert (
        explicit_authority_scope(Agent.CODEX, "explicit_user")
        is AuthorityScope.EXPLICIT_USER
    )
    assert (
        explicit_authority_scope(Agent.CODEX, "semantic") is AuthorityScope.OUT_OF_SCOPE
    )
    assert explicit_authority_scope(Agent.CODEX, None) is AuthorityScope.OUT_OF_SCOPE
    assert explicit_authority_scope(None, "explicit_user") is AuthorityScope.OUT_OF_SCOPE


def test_local_executor_never_enters_phase_one_scope():
    assert Agent.LOCAL not in AUTHORITATIVE_AGENTS
    assert (
        explicit_authority_scope(Agent.LOCAL, "explicit_user")
        is AuthorityScope.OUT_OF_SCOPE
    )


def test_shadow_mode_never_becomes_authoritative():
    decision = decide(mode=ExecutionAuthorityMode.SHADOW)
    assert decision.authoritative is False
    assert decision.dispatch_allowed is False
    assert decision.mode is ExecutionAuthorityMode.SHADOW


def test_authority_mode_parsing_and_rollback_value():
    assert ExecutionAuthorityMode.parse("shadow") is ExecutionAuthorityMode.SHADOW
    assert (
        ExecutionAuthorityMode.parse("explicit_user")
        is ExecutionAuthorityMode.EXPLICIT_USER
    )
    assert ExecutionAuthorityMode.parse(None) is ExecutionAuthorityMode.SHADOW
    with pytest.raises(ValueError):
        ExecutionAuthorityMode.parse("single_eligible")


def test_configuration_rejects_unknown_authority():
    with pytest.raises(ValueError):
        load_settings({"EXECUTION_GATE_AUTHORITY": "everything"})


def test_configuration_defaults_to_shadow():
    assert load_settings({}).execution_gate_authority == "shadow"


# --- authority decisions -------------------------------------------------------


def test_dispatch_requires_the_recheck():
    decision = decide()
    assert decision.authoritative is True
    assert decision.execution_allowed is True
    assert decision.recheck_performed is False
    assert decision.dispatch_allowed is False  # no recheck, no dispatch


def test_recheck_allows_dispatch_when_still_available():
    decision = controller().recheck(
        decide(),
        AvailabilitySample(Agent.CODEX, True, None, "dispatch"),
    )
    assert decision.recheck_performed is True
    assert decision.dispatch_allowed is True
    assert decision.availability_changed is False
    assert decision.block_reason is None


def test_recheck_blocks_when_availability_changed():
    decision = controller().recheck(
        decide(),
        AvailabilitySample(Agent.CODEX, False, "tool_not_registered", "dispatch"),
    )
    assert decision.dispatch_allowed is False
    assert decision.availability_changed is True
    assert (
        decision.block_reason
        == AuthorityBlockReason.AVAILABILITY_CHANGED_BEFORE_DISPATCH.value
    )


def test_substitution_is_rejected_by_authority():
    decision = gate(candidate_agent=Agent.DEEPSEEK, requested_agent=Agent.CODEX)
    value = controller().decide(
        decision,
        requested_agent=Agent.CODEX,
        requested_agent_source="explicit_user",
        availability_at_selection=availability_sample_from_gate(decision),
    )
    assert value.execution_allowed is False
    assert (
        value.block_reason == AuthorityBlockReason.AGENT_OUTSIDE_AUTHORITY_SCOPE.value
    )


def test_read_only_without_structural_enforcement_is_blocked():
    value = controller(read_only_enforceable=frozenset()).decide(
        gate(),
        requested_agent=Agent.CODEX,
        requested_agent_source="explicit_user",
        availability_at_selection=AvailabilitySample(Agent.CODEX, True, None, "selection"),
    )
    assert value.execution_mode is ExecutionMode.READ_ONLY
    assert value.execution_allowed is False
    assert (
        value.block_reason
        == AuthorityBlockReason.READ_ONLY_ENFORCEMENT_UNAVAILABLE.value
    )


def test_gate_block_reasons_are_preserved_by_authority():
    for overrides, expected in (
        ({"execution_requested": False}, ExecutionBlockReason.EXECUTION_NOT_REQUESTED),
        (
            {
                "eligible_agents": (Agent.DEEPSEEK,),
                "available_eligible_agents": (Agent.DEEPSEEK,),
            },
            ExecutionBlockReason.REQUESTED_AGENT_INELIGIBLE,
        ),
        (
            {"available_eligible_agents": (Agent.DEEPSEEK, Agent.LOCAL)},
            ExecutionBlockReason.REQUESTED_AGENT_UNAVAILABLE,
        ),
        (
            {"execution_safe": False},
            ExecutionBlockReason.EXECUTION_SAFETY_UNRESOLVED,
        ),
        (
            {"constraint_violation": "FORBID_DELEGATION"},
            ExecutionBlockReason.CONSTRAINT_VIOLATION,
        ),
    ):
        decision = decide(**overrides)
        assert decision.execution_allowed is False
        assert decision.block_reason == expected.value
        assert decision.dispatch_allowed is False


def test_explicit_deepseek_survives_auto_escalation_policy():
    decision = decide(
        candidate_agent=Agent.DEEPSEEK,
        requested_agent=Agent.DEEPSEEK,
    )
    assert decision.candidate_agent is Agent.DEEPSEEK
    assert decision.execution_allowed is True


def test_provenance_record_answers_why_this_agent_executed():
    decision = controller().recheck(
        decide(mutation_requested=True, agent_can_mutate=True),
        AvailabilitySample(Agent.CODEX, True, None, "dispatch"),
    )
    record = ExecutionAuthorityController.mark_dispatched(decision).provenance_record()
    for field in (
        "requested_agent",
        "requested_agent_source",
        "selected_agent",
        "selection_source",
        "selection_factors",
        "execution_gate_result",
        "block_reason",
        "availability_at_selection",
        "availability_at_dispatch",
        "execution_mode",
        "mutation_authorized",
        "dispatched",
        "recheck_performed",
    ):
        assert field in record
    assert record["selected_agent"] == "codex"
    assert record["execution_gate_result"] == "ALLOW"
    assert record["execution_mode"] == "MUTATION"
    assert record["dispatched"] is True
    assert record["availability_at_dispatch"]["available"] is True


# --- live availability probe ---------------------------------------------------


class ProbeRegistry:
    def __init__(self, *, tools=TOOL_NAMES, deepseek=True, configured=True, codex=True):
        self._tools = tuple(tools)
        self.deepseek = SimpleNamespace(
            status=lambda **_kwargs: {
                "enabled": deepseek,
                "configured": configured,
            }
        )
        self.codex = SimpleNamespace() if codex else None

    def names(self):
        return self._tools


def test_probe_reports_available_agents():
    sample = probe_agent_availability(ProbeRegistry(), Agent.CODEX)
    assert sample.available is True and sample.source == "dispatch"


def test_probe_detects_unregistered_tool():
    registry = ProbeRegistry(tools=("filesystem_read_text",))
    assert probe_agent_availability(registry, Agent.CODEX).reason == "tool_not_registered"


def test_probe_detects_disabled_and_unconfigured_deepseek():
    assert (
        probe_agent_availability(ProbeRegistry(deepseek=False), Agent.DEEPSEEK).reason
        == "agent_disabled"
    )
    assert (
        probe_agent_availability(ProbeRegistry(configured=False), Agent.DEEPSEEK).reason
        == "agent_not_configured"
    )


def test_probe_without_candidate_is_unavailable():
    assert probe_agent_availability(ProbeRegistry(), None).available is False


# --- read-only structural enforcement -----------------------------------------


def test_sandbox_policy_is_read_only_when_requested():
    assert _sandbox_policy(Path("C:/project"), read_only=True) == {"type": "readOnly"}


def test_sandbox_policy_keeps_workspace_write_by_default():
    policy = _sandbox_policy(Path("C:/project"), read_only=False)
    assert policy["type"] == "workspaceWrite"
    assert policy["networkAccess"] is False


def test_job_record_carries_execution_mode(tmp_path):
    jobs = CodexJobStore(tmp_path)
    job = jobs.create(
        project=str(tmp_path),
        task_summary="read one file",
        source="qwen",
        wait=False,
        execution_mode="READ_ONLY",
    )
    assert job["execution_mode"] == "READ_ONLY"
    assert jobs.get(job["job_id"])["execution_mode"] == "READ_ONLY"


class Runtime:
    def __init__(self):
        self.values = {"queue_epoch": 0}

    def read(self):
        return dict(self.values)

    def update(self, **values):
        self.values.update(values)

    def append_state_event(self, *_args, **_kwargs):
        return None

    def turn_mutex(self, timeout=None):
        class Lock:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_args):
                return False

        return Lock()


class CaptureManager:
    """Minimal manager that records what run_turn received."""

    def __init__(self):
        self.calls: list[dict] = []
        self.runtime = Runtime()
        self.bridge_log = SimpleNamespace(write=lambda *_a, **_k: None)
        self.project: Path | None = None

    def _session(self):
        return {
            "thread_id": "thread-1",
            "session_id": "thread-1",
            "project": str(self.project),
            "state": "idle",
            "source": "cli",
            "visible": True,
            "recoverable": True,
            "ephemeral": False,
        }

    def list_project_threads(self):
        return [self._session()]

    def adopt_thread(self, thread_id):
        return self._session()

    def create_thread(self):
        return self._session()

    def run_turn(self, task, **kwargs):
        self.calls.append({"task": task, **kwargs})
        return CodexResult(True, "thread-1", "turn-1", "completed", "done", None)

    def read_turn(self, thread_id, turn_id):
        return {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "status": "completed",
            "final_response": "done",
            "error": None,
        }


def codex_runner(tmp_path: Path, manager: CaptureManager) -> CodexRunner:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    value = CodexRunner(
        PathPolicy((project,)),
        state_dir=tmp_path / "state",
        quick_wait_timeout=2,
    )
    value._managers[project.resolve()] = manager
    manager.project = project.resolve()
    return value


def test_read_only_mode_reaches_the_codex_turn(tmp_path):
    manager = CaptureManager()
    runner = codex_runner(tmp_path, manager)
    runner.delegate_to_codex(
        task="analise este arquivo",
        project_path=str(tmp_path / "project"),
        wait=True,
        execution_mode="READ_ONLY",
    )
    deadline = time.monotonic() + 5
    while not manager.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.calls, "run_turn was never called"
    assert manager.calls[0]["read_only"] is True


def test_mutation_mode_keeps_write_capability(tmp_path):
    manager = CaptureManager()
    runner = codex_runner(tmp_path, manager)
    runner.delegate_to_codex(
        task="corrija o arquivo",
        project_path=str(tmp_path / "project"),
        wait=True,
        execution_mode="MUTATION",
    )
    deadline = time.monotonic() + 5
    while not manager.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.calls and manager.calls[0]["read_only"] is False


# --- live supervisor integration ----------------------------------------------


class Logger:
    def __init__(self):
        self.events = []

    def write_event(self, event, **values):
        self.events.append((event, values))

    def find(self, event):
        return [values for name, values in self.events if name == event]


class Jobs:
    def __init__(self):
        self.created = 0

    def list(self):
        return []

    def create(self, **_kwargs):
        self.created += 1
        return {"job_id": "job-1"}


class Codex:
    def __init__(self):
        self.jobs = Jobs()
        self.sessions_resolved = 0

    def shared_project(self):
        return r"D:\tern"

    def claim_completed_results(self):
        return []

    def resolve_session(self, *_args, **_kwargs):
        self.sessions_resolved += 1
        raise AssertionError("session resolved without dispatch authority")


class LiveRegistry:
    """Registry that records dispatches and can change availability mid-turn."""

    def __init__(self, *, deepseek=True, configured=True, tools=TOOL_NAMES):
        self.logger = Logger()
        self.web = SimpleNamespace(begin_research=lambda _text: None)
        self.projects = SimpleNamespace(
            context=lambda: {
                "active_project": {"id": "tern", "root": r"D:\tern"},
                "codex_thread_project": {"id": "tern"},
            },
            context_text=lambda: "Active project: tern",
            projects=lambda: [{"id": "tern", "root": r"D:\tern"}],
        )
        self.codex = Codex()
        self._deepseek_enabled = deepseek
        self._deepseek_configured = configured
        self.deepseek = SimpleNamespace(
            client=SimpleNamespace(enabled=deepseek, configured=configured),
            status=lambda **_kwargs: {
                "enabled": self._deepseek_enabled,
                "configured": self._deepseek_configured,
                "active_session": "ds-1",
            },
        )
        self.pending_actions = SimpleNamespace(pending=lambda: None)
        self._tools = list(tools)
        self.calls: list[tuple[str, dict, dict]] = []
        self.before_execute = None

    def names(self):
        return tuple(self._tools)

    def specs(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "project_path": {"type": "string"},
                        },
                    },
                },
            }
            for name in self._tools
        ]

    def drop_tool(self, name):
        if name in self._tools:
            self._tools.remove(name)

    def disable_deepseek(self):
        self._deepseek_enabled = False

    def execute(self, name, arguments, *, context=None, event_callback=None):
        if self.before_execute is not None:
            self.before_execute(self)
        self.calls.append((name, dict(arguments), dict(context or {})))
        if name == "delegate_to_codex":
            return {
                "ok": True,
                "job_id": "job-1",
                "thread_id": "thread-1",
                "status": "running",
                "session_resolution": {"session_id": "thread-1", "reused": True},
            }
        if name == "delegate_to_deepseek":
            return {"ok": True, "response": "analise pronta"}
        return {"ok": True}

    def delegated(self, name):
        return [item for item in self.calls if item[0] == name]


class DirectClient:
    """Answers without tool calls; delegations can only come from the fast path."""

    supports_structured_output = False

    def chat(self, _messages, **_kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class DelegatingClient:
    """Asks for one delegation through the tool loop."""

    supports_structured_output = False

    def __init__(self, tool="delegate_to_codex", arguments=None):
        self.tool = tool
        self.arguments = arguments or {
            "task": "corrigir",
            "project_path": r"D:\tern",
        }
        self.turn = 0

    def chat(self, _messages, **kwargs):
        self.turn += 1
        if self.turn == 1:
            import json

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": self.tool,
                                        "arguments": json.dumps(self.arguments),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def authority_settings(**overrides):
    values = {
        "EXECUTION_GATE_AUTHORITY": "explicit_user",
        "AGENT_DECISION_FAST_PATH": "true",
        "AGENT_DECISION_SEMANTIC_FIRST": "false",
        "MODEL_MAX_TOOL_CALLS": "4",
    }
    values.update(overrides)
    return load_settings(values)


def run_live(text, *, registry=None, client=None, **settings_overrides):
    registry = registry or LiveRegistry()
    agent = Supervisor(
        authority_settings(**settings_overrides),
        client or DirectClient(),
        registry,
    )
    result = agent.run(text)
    return registry, result


def authority_record(registry, event="execution_gate_authority"):
    records = registry.logger.find(event)
    return records[-1] if records else None


def test_explicit_codex_mutation_is_dispatched_once():
    registry, result = run_live(
        "use o Codex para corrigir o bug e alterar o arquivo do modulo"
    )
    assert result["ok"] is True
    delegations = registry.delegated("delegate_to_codex")
    assert len(delegations) == 1
    context = delegations[0][2]
    assert context["execution_mode"] == "MUTATION"
    assert context["mutation_authorized"] is True
    assert context["execution_authority"] == "explicit_user"
    dispatch = authority_record(registry, "execution_authority_dispatch")
    assert dispatch is not None
    assert dispatch["selected_agent"] == "codex"
    assert dispatch["dispatched"] is True
    assert dispatch["recheck_performed"] is True
    assert dispatch["availability_at_dispatch"]["available"] is True


def test_explicit_codex_read_only_is_dispatched_in_read_only_mode():
    registry, result = run_live(
        "use o Codex para investigar a causa sem modificar nenhum arquivo"
    )
    assert result["ok"] is True
    delegations = registry.delegated("delegate_to_codex")
    assert len(delegations) == 1
    context = delegations[0][2]
    assert context["execution_mode"] == "READ_ONLY"
    assert context["mutation_authorized"] is False


def test_explicit_codex_unavailable_blocks_without_substitution():
    registry = LiveRegistry(tools=[n for n in TOOL_NAMES if n != "delegate_to_codex"])
    registry, result = run_live(
        "use o Codex para corrigir o bug e alterar o arquivo", registry=registry
    )
    assert registry.delegated("delegate_to_codex") == []
    assert registry.delegated("delegate_to_deepseek") == []
    assert "substituto" in result["answer"]
    blocked = authority_record(registry, "execution_authority_blocked")
    assert blocked is not None
    assert blocked["selected_agent"] == "codex"
    assert blocked["dispatch_allowed"] is False


class VanishingRegistry(LiveRegistry):
    """Codex is available at selection and gone at the pre-dispatch recheck."""

    def __init__(self):
        super().__init__()
        self.name_reads = 0

    def names(self):
        self.name_reads += 1
        if self.name_reads <= 1:
            return tuple(self._tools)
        return tuple(item for item in self._tools if item != "delegate_to_codex")


def test_toctou_availability_change_blocks_before_dispatch():
    registry = VanishingRegistry()
    agent = Supervisor(authority_settings(), DirectClient(), registry)
    result = agent.run("use o Codex para corrigir o bug e alterar o arquivo")
    assert registry.delegated("delegate_to_codex") == []
    assert registry.codex.jobs.created == 0
    assert registry.codex.sessions_resolved == 0
    assert "substituto" in result["answer"]
    recheck = authority_record(registry, "execution_authority_recheck")
    assert recheck is not None
    assert recheck["recheck_performed"] is True
    assert recheck["availability_at_selection"]["available"] is True
    assert recheck["availability_at_dispatch"]["available"] is False
    assert recheck["availability_changed_before_dispatch"] is True
    assert (
        recheck["block_reason"]
        == AuthorityBlockReason.AVAILABILITY_CHANGED_BEFORE_DISPATCH.value
    )
    assert registry.logger.find("execution_authority_dispatch") == []


def test_toctou_detected_by_recheck_reports_change():
    registry = LiveRegistry()
    controller_value = ExecutionAuthorityController(
        ExecutionAuthorityMode.EXPLICIT_USER
    )
    decision = decide()
    registry.drop_tool("delegate_to_codex")
    rechecked = controller_value.recheck(
        decision,
        probe_agent_availability(registry, Agent.CODEX),
    )
    assert rechecked.availability_changed is True
    assert rechecked.dispatch_allowed is False
    assert (
        rechecked.block_reason
        == AuthorityBlockReason.AVAILABILITY_CHANGED_BEFORE_DISPATCH.value
    )
    assert registry.codex.jobs.created == 0


def test_meta_question_naming_codex_never_delegates():
    registry, result = run_live("o Codex seria melhor para isso?")
    assert registry.delegated("delegate_to_codex") == []
    assert registry.codex.jobs.created == 0
    assert registry.codex.sessions_resolved == 0


def test_explicit_binding_without_execution_request_blocks():
    registry, _ = run_live("como eu uso o Codex para tarefas grandes?")
    assert registry.delegated("delegate_to_codex") == []


def test_unknown_requirements_fail_closed():
    registry, result = run_live("use o Codex para resolver isso")
    record = authority_record(registry)
    assert record is not None
    if record["execution_gate_result"] == "BLOCK":
        assert registry.delegated("delegate_to_codex") == []
        assert record["block_reason"] in {
            "EXECUTION_SAFETY_UNRESOLVED",
            "EXECUTION_NOT_REQUESTED",
        }


def test_blocked_gate_resolves_no_session_and_creates_no_job():
    registry = LiveRegistry(deepseek=False)
    registry, _ = run_live(
        "use o DeepSeek para revisar esta arquitetura", registry=registry
    )
    assert registry.delegated("delegate_to_deepseek") == []
    assert registry.codex.sessions_resolved == 0
    assert registry.codex.jobs.created == 0


def test_explicit_deepseek_is_never_replaced_by_codex():
    registry = LiveRegistry(deepseek=False)
    registry, result = run_live(
        "use o DeepSeek para explicar esse conceito", registry=registry
    )
    assert registry.delegated("delegate_to_codex") == []
    assert "substituto" in result["answer"]


def test_tool_loop_cannot_swap_the_explicit_agent():
    registry = LiveRegistry()
    client = DelegatingClient(
        tool="delegate_to_deepseek", arguments={"task": "analisar"}
    )
    agent = Supervisor(
        authority_settings(AGENT_DECISION_FAST_PATH="false"), client, registry
    )
    result = agent.run("use o Codex para corrigir o bug e alterar o arquivo")
    assert registry.delegated("delegate_to_deepseek") == []
    assert "substituto" in result["answer"]
    blocked = authority_record(registry, "execution_authority_blocked")
    assert blocked is not None
    assert blocked["block_reason"] == "AGENT_OUTSIDE_AUTHORITY_SCOPE"


def test_tool_loop_dispatch_carries_the_execution_envelope():
    registry = LiveRegistry()
    client = DelegatingClient(
        arguments={"task": "corrigir o modulo", "project_path": r"D:\tern"}
    )
    agent = Supervisor(
        authority_settings(AGENT_DECISION_FAST_PATH="false"), client, registry
    )
    agent.run("use o Codex para corrigir o bug e alterar o arquivo do modulo")
    delegations = registry.delegated("delegate_to_codex")
    assert len(delegations) == 1
    assert delegations[0][2]["execution_mode"] == "MUTATION"


def test_single_authoritative_decision_dispatches_at_most_once():
    registry = LiveRegistry()
    registry, _ = run_live(
        "use o Codex para corrigir o bug e alterar o arquivo do modulo",
        registry=registry,
    )
    assert len(registry.delegated("delegate_to_codex")) == 1
    dispatches = registry.logger.find("execution_authority_dispatch")
    assert len(dispatches) == 1


def test_executor_failure_before_job_creation_does_not_fall_back():
    class Failing(LiveRegistry):
        def execute(self, name, arguments, *, context=None, event_callback=None):
            self.calls.append((name, dict(arguments), dict(context or {})))
            return {"ok": False, "error": "codex_unreachable", "message": "offline"}

    registry, result = run_live(
        "use o Codex para corrigir o bug e alterar o arquivo",
        registry=Failing(),
    )
    assert len(registry.delegated("delegate_to_codex")) == 1
    assert registry.delegated("delegate_to_deepseek") == []


def test_non_explicit_request_keeps_legacy_authority():
    registry = LiveRegistry()
    client = DelegatingClient(
        arguments={"task": "corrigir", "project_path": r"D:\tern"}
    )
    agent = Supervisor(
        authority_settings(AGENT_DECISION_FAST_PATH="false"), client, registry
    )
    agent.run("corrija o bug do modulo de autenticacao")
    delegations = registry.delegated("delegate_to_codex")
    assert len(delegations) == 1
    context = delegations[0][2]
    # legacy dispatch carries no authority envelope
    assert "execution_mode" not in context
    assert "execution_authority" not in context
    record = authority_record(registry)
    assert record is not None
    assert record["authoritative"] is False
    assert record["authority_scope"] == "OUT_OF_SCOPE"


def test_shadow_rollback_restores_previous_behaviour():
    registry = LiveRegistry(tools=[n for n in TOOL_NAMES if n != "delegate_to_codex"])
    agent = Supervisor(
        authority_settings(EXECUTION_GATE_AUTHORITY="shadow"),
        DirectClient(),
        registry,
    )
    result = agent.run("use o Codex para corrigir o bug e alterar o arquivo")
    assert result["ok"] is True
    record = authority_record(registry)
    assert record is not None
    assert record["authority"] == "shadow"
    assert record["authoritative"] is False
    assert registry.logger.find("execution_authority_dispatch") == []


def test_authority_records_stay_free_of_task_text():
    import json

    registry, _ = run_live(
        "use o Codex para corrigir o token SUPERSECRETVALUE e alterar o arquivo"
    )
    for event in (
        "execution_gate_authority",
        "execution_authority_recheck",
        "execution_authority_dispatch",
    ):
        for record in registry.logger.find(event):
            assert "SUPERSECRETVALUE" not in json.dumps(record, default=str)


# --- A/B: legacy authority versus explicit-user authority ----------------------

AB_CASES = (
    # id, request, availability, expected new-authority outcome
    ("explicit-codex-mutation", "use o Codex para corrigir o bug e alterar o arquivo do modulo", {}, "allow", "codex", "MUTATION"),
    ("explicit-codex-read-only", "use o Codex para investigar a causa sem modificar nenhum arquivo", {}, "allow", "codex", "READ_ONLY"),
    ("explicit-codex-unavailable", "use o Codex para corrigir o bug e alterar o arquivo", {"drop": "delegate_to_codex"}, "block", None, None),
    ("explicit-deepseek-conceptual", "use o DeepSeek para explicar esse conceito de arquitetura", {}, "any", None, None),
    ("explicit-deepseek-disabled", "use o DeepSeek para explicar esse conceito", {"deepseek": False}, "block", None, None),
    ("explicit-deepseek-repository", "use o DeepSeek para revisar o modulo do repositorio", {}, "block", None, None),
    ("meta-codex", "o Codex seria melhor para isso?", {}, "block", None, None),
    ("meta-deepseek", "por que voce escolheria o DeepSeek?", {}, "block", None, None),
    ("explicit-codex-pronoun-target", "use o Codex para resolver isso", {}, "any", None, None),
    ("explicit-codex-no-execution", "como eu uso o Codex para tarefas grandes?", {}, "block", None, None),
)


def build_registry(availability):
    tools = list(TOOL_NAMES)
    drop = availability.get("drop")
    if drop:
        tools = [item for item in tools if item != drop]
    return LiveRegistry(
        tools=tools,
        deepseek=bool(availability.get("deepseek", True)),
        configured=bool(availability.get("configured", True)),
    )


def run_mode(text, availability, mode):
    registry = build_registry(availability)
    agent = Supervisor(
        authority_settings(EXECUTION_GATE_AUTHORITY=mode),
        DirectClient(),
        registry,
    )
    result = agent.run(text)
    codex = registry.delegated("delegate_to_codex")
    deepseek = registry.delegated("delegate_to_deepseek")
    delegations = codex + deepseek
    agent_identity = (
        "codex" if codex else "deepseek" if deepseek else None
    )
    context = delegations[0][2] if delegations else {}
    return {
        "delegations": len(delegations),
        "agent": agent_identity,
        "execution_mode": context.get("execution_mode"),
        "mutation_authorized": context.get("mutation_authorized"),
        "jobs": registry.codex.jobs.created,
        "sessions": registry.codex.sessions_resolved,
        "authority_record": authority_record(registry),
        "dispatch_record": authority_record(registry, "execution_authority_dispatch"),
        "ok": bool(result.get("ok")),
    }


def test_ab_legacy_versus_new_authority(tmp_path):
    import json

    rows = []
    metrics = {
        "cases": 0,
        "allow_expected": 0,
        "allow_correct": 0,
        "block_expected": 0,
        "block_correct": 0,
        "selection_to_executor_matches": 0,
        "selection_to_executor_total": 0,
        "provenance_complete": 0,
        "recheck_performed": 0,
        "recheck_expected": 0,
        "exactly_once": 0,
        "read_only_correct": 0,
        "read_only_total": 0,
        "legacy_delegations": 0,
        "new_delegations": 0,
        "divergences": 0,
    }
    safety = {
        "unexpected_execution": 0,
        "execution_without_request": 0,
        "mutation_without_authorization": 0,
        "wrong_agent_execution": 0,
        "explicit_agent_override": 0,
        "ineligible_agent_execution": 0,
        "unavailable_agent_execution": 0,
        "availability_recheck_bypass": 0,
        "silent_agent_substitution": 0,
        "policy_excluded_execution": 0,
        "duplicate_initial_delegation": 0,
        "selection_provenance_loss": 0,
        "read_only_violation": 0,
    }

    for case_id, text, availability, expectation, agent_expected, mode_expected in AB_CASES:
        legacy = run_mode(text, availability, "shadow")
        new = run_mode(text, availability, "explicit_user")
        metrics["cases"] += 1
        metrics["legacy_delegations"] += legacy["delegations"]
        metrics["new_delegations"] += new["delegations"]
        record = new["authority_record"] or {}
        dispatch = new["dispatch_record"]

        if expectation == "allow":
            metrics["allow_expected"] += 1
            metrics["allow_correct"] += int(new["delegations"] == 1)
            metrics["recheck_expected"] += 1
            metrics["recheck_performed"] += int(
                bool(dispatch and dispatch.get("recheck_performed"))
            )
        elif expectation == "block":
            metrics["block_expected"] += 1
            metrics["block_correct"] += int(new["delegations"] == 0)

        if agent_expected is not None and new["delegations"]:
            metrics["selection_to_executor_total"] += 1
            metrics["selection_to_executor_matches"] += int(
                new["agent"] == agent_expected
            )
        if mode_expected is not None and new["delegations"]:
            metrics["read_only_total"] += 1
            metrics["read_only_correct"] += int(
                new["execution_mode"] == mode_expected
            )

        metrics["provenance_complete"] += int(bool(record.get("provenance_complete")))
        metrics["exactly_once"] += int(new["delegations"] <= 1)
        if legacy["delegations"] != new["delegations"] or legacy["agent"] != new["agent"]:
            metrics["divergences"] += 1

        # hard safety accounting on the authoritative side
        if new["delegations"] > 1:
            safety["duplicate_initial_delegation"] += 1
        if new["delegations"] and record.get("execution_requested") is False:
            safety["execution_without_request"] += 1
        if new["delegations"] and record.get("execution_gate_result") == "BLOCK":
            safety["unexpected_execution"] += 1
        if new["delegations"] and new["execution_mode"] == "MUTATION" and not new[
            "mutation_authorized"
        ]:
            safety["mutation_without_authorization"] += 1
        if new["delegations"] and record.get("requested_agent") and new["agent"] != record[
            "requested_agent"
        ]:
            safety["silent_agent_substitution"] += 1
            safety["wrong_agent_execution"] += 1
        if new["delegations"] and record.get("selected_agent") not in (
            record.get("eligible_agents") or []
        ):
            safety["ineligible_agent_execution"] += 1
        if new["delegations"] and record.get("selected_agent") not in (
            record.get("available_eligible_agents") or []
        ):
            safety["unavailable_agent_execution"] += 1
        if new["delegations"] and not (dispatch or {}).get("recheck_performed"):
            safety["availability_recheck_bypass"] += 1
        if new["delegations"] and not record.get("provenance_complete"):
            safety["selection_provenance_loss"] += 1
        if (
            new["delegations"]
            and new["execution_mode"] == "READ_ONLY"
            and new["mutation_authorized"]
        ):
            safety["read_only_violation"] += 1
        if new["delegations"] and record.get("requested_agent_source") not in {
            "EXPLICIT_USER",
            "explicit_user",
            None,
        }:
            safety["explicit_agent_override"] += 1

        rows.append(
            {
                "id": case_id,
                "input": text,
                "expectation": expectation,
                "legacy": {
                    "delegations": legacy["delegations"],
                    "agent": legacy["agent"],
                },
                "new": {
                    "delegations": new["delegations"],
                    "agent": new["agent"],
                    "execution_mode": new["execution_mode"],
                    "block_reason": record.get("block_reason"),
                    "gate_result": record.get("execution_gate_result"),
                },
            }
        )

    assert safety == {key: 0 for key in safety}, safety
    assert metrics["block_correct"] == metrics["block_expected"]
    assert metrics["allow_correct"] == metrics["allow_expected"]
    assert metrics["recheck_performed"] == metrics["recheck_expected"]
    assert metrics["exactly_once"] == metrics["cases"]
    if metrics["selection_to_executor_total"]:
        assert (
            metrics["selection_to_executor_matches"]
            == metrics["selection_to_executor_total"]
        )
    if metrics["read_only_total"]:
        assert metrics["read_only_correct"] == metrics["read_only_total"]

    report = {"metrics": metrics, "safety": safety, "rows": rows}
    output = Path(".orchestrator/execution-authority-ab.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
