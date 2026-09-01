from __future__ import annotations

import json

from tern.orchestrator.autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentRuntimeAvailability,
    Capability,
    CapabilityBaseline,
)
from tern.orchestrator.execution_gate import ExecutionMode
from tern.orchestrator.orchestration_contracts import (
    AgentSource,
    NextAction,
    NextActionType,
    OrchestrationReasonCode,
    UserGoal,
)
from tern.orchestrator.orchestration_policy import (
    NextActionValidator,
    QwenOrchestrationPolicy,
)
from tern.orchestrator.orchestration_state import WorldStateBuilder


def _state(*, codex_available: bool = True):
    profile = AgentCapabilityProfile(
        Agent.CODEX,
        frozenset({Capability.REPOSITORY_READ, Capability.CODE_ANALYSIS}),
        (),
    )
    baseline = CapabilityBaseline(
        {Agent.CODEX: profile},
        {
            Agent.CODEX: AgentRuntimeAvailability(
                Agent.CODEX, codex_available, True, True
            )
        },
    )
    goal = UserGoal("g", "investigue", "diagnóstico")
    return goal, WorldStateBuilder().build(
        goal, baseline=baseline, tool_names=("inspect_repo",)
    ).state


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "choices": [
                {"message": {"content": json.dumps(self.payload)}}
            ]
        }


class _SequenceClient:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"choices": [{"message": {"content": next(self.contents)}}]}


def test_qwen_policy_uses_closed_structured_output_and_one_action() -> None:
    goal, state = _state()
    client = _Client(
        {
            "action": "INSPECT",
            "target_agent": None,
                "target": "authentication files",
                "tool_name": None,
                "arguments": {},
                "objective": "inspect authentication initialization",
            "execution_mode": "READ_ONLY",
            "required_capabilities": ["repository_read"],
            "reason_code": "REPOSITORY_INSPECTION_REQUIRED",
            "evidence_refs": [],
            "expected_observation": "authentication facts",
            "confidence": None,
            "short_horizon_hint": None,
        }
    )

    action = QwenOrchestrationPolicy(client).decide(goal, state)

    assert action.action is NextActionType.INSPECT
    assert client.calls[0][1]["tools"] is None
    schema = client.calls[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["action"]["enum"]) == {
        item.value for item in NextActionType
    }
    assert (
        OrchestrationReasonCode.INVALID_ACTION.value
        not in schema["properties"]["reason_code"]["enum"]
    )
    assert (
        OrchestrationReasonCode.BUDGET_EXHAUSTED.value
        not in schema["properties"]["reason_code"]["enum"]
    )


def test_qwen_policy_removes_irrelevant_agent_from_tool_action() -> None:
    goal, state = _state()
    client = _Client(
        {
            "action": "INSPECT",
            "target_agent": "codex",
            "target": "repository",
            "tool_name": "inspect_repo",
            "arguments": {},
            "objective": "inspect repository",
            "execution_mode": "READ_ONLY",
            "reason_code": "REPOSITORY_INSPECTION_REQUIRED",
        }
    )

    action = QwenOrchestrationPolicy(client).decide(goal, state)

    assert action.action is NextActionType.INSPECT
    assert action.target_agent is None


def test_qwen_policy_canonicalizes_safe_read_only_tool_arguments(tmp_path) -> None:
    goal, state = _state()
    state = state.evolve(
        project=state.project.__class__(path=str(tmp_path)),
        tools=("filesystem_read_text", "run_project_tests"),
    )
    file_path = tmp_path / "sample.py"
    read = QwenOrchestrationPolicy._parse(
        {
            "action": "INSPECT",
            "target": str(file_path),
            "tool_name": "filesystem_read_text",
            "arguments": {"max_bytes": 10000},
            "objective": "read sample",
            "execution_mode": "READ_ONLY",
            "reason_code": "INSUFFICIENT_INFORMATION",
        },
        goal,
        state,
        "bounded_live",
    )
    tests = QwenOrchestrationPolicy._parse(
        {
            "action": "EXECUTE",
            "target": "tests",
            "tool_name": "run_project_tests",
            "arguments": {},
            "objective": "run tests",
            "execution_mode": "READ_ONLY",
            "reason_code": "REPOSITORY_INSPECTION_REQUIRED",
        },
        goal,
        state,
        "bounded_live",
    )

    assert dict(read.arguments) == {"path": str(file_path), "max_bytes": 16384}
    assert dict(tests.arguments) == {"project_path": str(tmp_path)}


def test_qwen_policy_accepts_compact_wire_action_and_retries_invalid_json() -> None:
    goal, state = _state()
    compact = json.dumps(
        {
            "action": "INSPECT",
            "target_agent": None,
            "target": "authentication",
            "tool_name": None,
            "arguments": {},
            "objective": "inspect authentication",
            "execution_mode": "READ_ONLY",
            "reason_code": "REPOSITORY_INSPECTION_REQUIRED",
        }
    )
    client = _SequenceClient(["not-json", compact])

    action = QwenOrchestrationPolicy(client).decide(goal, state)

    assert action.action is NextActionType.INSPECT
    assert action.required_capabilities == ()
    assert client.calls == 2
    assert action.action_id.startswith("shadow-")


def test_validator_preserves_explicit_and_forbidden_agents() -> None:
    _, state = _state()
    explicit = UserGoal(
        "g",
        "manda pro Codex",
        "diagnóstico",
        explicit_agent=Agent.CODEX,
        agent_source=AgentSource.EXPLICIT_USER,
    )
    wrong = NextAction(
        "a",
        NextActionType.DELEGATE,
        "analisar",
        OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
        target_agent=Agent.DEEPSEEK,
        execution_mode=ExecutionMode.READ_ONLY,
    )
    result = NextActionValidator().validate(wrong, explicit, state)
    assert "IGNORED_EXPLICIT_AGENT" in result.critical_violations
    assert "SILENT_AGENT_SUBSTITUTION" in result.critical_violations

    forbidden = UserGoal(
        "g",
        "não use Codex",
        "diagnóstico",
        forbidden_agents=(Agent.CODEX,),
    )
    codex = NextAction(
        "a2",
        NextActionType.DELEGATE,
        "analisar",
        OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
        target_agent=Agent.CODEX,
        execution_mode=ExecutionMode.READ_ONLY,
    )
    blocked = NextActionValidator().validate(codex, forbidden, state)
    assert "USED_FORBIDDEN_AGENT" in blocked.critical_violations


def test_validator_detects_read_only_and_unavailable_agent() -> None:
    _, state = _state(codex_available=False)
    goal = UserGoal(
        "g",
        "analise sem alterar",
        "diagnóstico",
        mutation_forbidden=True,
    )
    action = NextAction(
        "a",
        NextActionType.DELEGATE,
        "corrigir",
        OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
        target_agent=Agent.CODEX,
        execution_mode=ExecutionMode.MUTATION,
    )
    result = NextActionValidator().validate(action, goal, state)
    assert "AGENT_UNAVAILABLE" in result.violations
    assert "VIOLATED_READ_ONLY" in result.critical_violations


def test_respond_rejects_incompatible_agent_fields() -> None:
    goal, state = _state()
    action = NextAction(
        "a",
        NextActionType.RESPOND,
        "responder",
        OrchestrationReasonCode.SUFFICIENT_INFORMATION,
        target_agent=Agent.CODEX,
    )
    result = NextActionValidator().validate(action, goal, state)
    assert result.valid is False
    assert "RESPOND_FORBIDS_AGENT" in result.violations


def test_validator_rejects_invalid_reason_code() -> None:
    goal, state = _state()
    action = NextAction(
        "a-invalid-reason",
        NextActionType.INSPECT,
        "inspect",
        OrchestrationReasonCode.INVALID_ACTION,
        target="repository",
    )

    result = NextActionValidator().validate(action, goal, state)

    assert result.valid is False
    assert "INVALID_REASON_CODE" in result.violations


def test_mutation_without_semantic_requirement_is_critical() -> None:
    goal, state = _state()
    action = NextAction(
        "a",
        NextActionType.EXECUTE,
        "change files",
        OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
        tool_name="inspect_repo",
        execution_mode=ExecutionMode.MUTATION,
    )
    result = NextActionValidator().validate(action, goal, state)
    assert "MUTATION_WITHOUT_REQUIREMENT_OR_AUTHORITY" in result.critical_violations
    assert "UNSAFE_EXECUTE" in result.critical_violations
