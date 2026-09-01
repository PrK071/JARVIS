from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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
    Observation,
    ObservationSource,
    ObservationStatus,
    OrchestrationBudget,
    OrchestrationReasonCode,
    UserGoal,
    VerificationStatus,
)
from tern.orchestrator.orchestration_fast_path import (
    ActionSpaceBuilder,
    OrchestrationDecisionCache,
    OrchestrationFastPath,
)
from tern.orchestrator.orchestration_live import (
    BoundedLiveOrchestrationRunner,
    LiveActionSink,
)
from tern.orchestrator.orchestration_loop import OrchestrationLoop
from tern.orchestrator.orchestration_policy import ScriptedOrchestrationPolicy
from tern.orchestrator.orchestration_state import WorldStateBuilder
from tern.orchestrator.task_requirement_grounding import RequirementValue
from tern.orchestrator.user_goal import UserGoalBuilder


class _PathPolicy:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, value: str, must_exist: bool = True) -> Path:
        path = Path(value).resolve()
        if path != self.root and self.root not in path.parents:
            raise PermissionError(path)
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path


class _Codex:
    def __init__(self):
        self.is_available = True

    def available(self) -> bool:
        return self.is_available


class _DeepSeekClient:
    enabled = True
    configured = True


class _DeepSeek:
    client = _DeepSeekClient()

    def status(self):
        return {"enabled": True, "configured": True}


class _Logger:
    def __init__(self):
        self.events = []

    def write_event(self, event, **values):
        self.events.append((event, values))


class _Registry:
    def __init__(self, root: Path):
        self.policy = _PathPolicy(root)
        self.codex = _Codex()
        self.deepseek = _DeepSeek()
        self.logger = _Logger()
        self.calls = []
        self.results = {}
        self._names = (
            "resolve_project",
            "filesystem_read_text",
            "find_project_files",
            "get_project_git_state",
            "run_project_tests",
            "filesystem_write_text",
            "delegate_to_codex",
            "delegate_to_deepseek",
            "get_codex_job_status",
        )

    def names(self):
        return self._names

    def specs(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "additionalProperties": True},
                },
            }
            for name in self._names
        ]

    def execute(self, name, arguments, *, context=None, event_callback=None):
        self.calls.append((name, dict(arguments), dict(context or {})))
        result = self.results.get(name)
        if callable(result):
            return result(arguments, context or {})
        if result is not None:
            return dict(result)
        if name == "run_project_tests":
            return {"ok": True, "returncode": 0, "output": "2 passed"}
        if name.startswith("delegate_to_"):
            return {"ok": True, "response": f"{name} completed", "session_id": "s-1"}
        return {"ok": True, "path": arguments.get("path"), "result": f"{name} completed"}


def _baseline() -> CapabilityBaseline:
    profiles = {
        Agent.LOCAL: AgentCapabilityProfile(
            Agent.LOCAL,
            frozenset(
                {
                    Capability.REPOSITORY_READ,
                    Capability.FILESYSTEM_READ,
                    Capability.FILESYSTEM_WRITE,
                    Capability.REPOSITORY_WRITE,
                    Capability.CODE_EDIT,
                    Capability.TEST_EXECUTION,
                    Capability.MUTATION,
                }
            ),
            (),
        ),
        Agent.CODEX: AgentCapabilityProfile(
            Agent.CODEX,
            frozenset(
                {
                    Capability.REPOSITORY_READ,
                    Capability.REPOSITORY_WRITE,
                    Capability.CODE_ANALYSIS,
                    Capability.CODE_EDIT,
                    Capability.TEST_EXECUTION,
                    Capability.MUTATION,
                    Capability.READ_ONLY,
                }
            ),
            (),
        ),
        Agent.DEEPSEEK: AgentCapabilityProfile(
            Agent.DEEPSEEK,
            frozenset(
                {
                    Capability.GENERAL_REASONING,
                    Capability.CODE_ANALYSIS,
                    Capability.READ_ONLY,
                }
            ),
            (),
        ),
    }
    availability = {
        agent: AgentRuntimeAvailability(agent, True, True, True)
        for agent in profiles
    }
    return CapabilityBaseline(profiles, availability)


def _state(goal: UserGoal, root: Path, registry: _Registry, *, max_steps: int = 6):
    return WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        project_snapshot={"project_path": str(root)},
        tool_names=registry.names(),
        budget=OrchestrationBudget(max_steps=max_steps),
        authority_facts=("BOUNDED_LIVE",),
    ).state


def _action(index: int, action: NextActionType, **values) -> NextAction:
    return NextAction(
        action_id=f"live-{index}",
        action=action,
        objective=values.pop("objective", f"step {index}"),
        reason_code=values.pop(
            "reason_code", OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED
        ),
        **values,
    )


def _loop(goal, state, registry, actions, *, fast_path=False):
    sink = LiveActionSink(
        registry,
        goal,
        run_id="run-1",
        conversation_id="conversation-1",
        original_user_text=goal.summary,
    )
    return OrchestrationLoop(
        ScriptedOrchestrationPolicy(actions),
        sink=sink,
        action_space_builder=ActionSpaceBuilder(),
        tool_specs=registry.specs(),
        fast_path=OrchestrationFastPath() if fast_path else None,
    ).run(goal, state)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("filesystem_read_text", {"path": "{file}"}),
        ("find_project_files", {"project_id": "project", "query": "auth"}),
        ("get_project_git_state", {"project_path": "{root}"}),
        ("run_project_tests", {"project_path": "{root}", "target": "tests"}),
    ],
)
def test_low_risk_actions_execute_real_registry_calls(
    tmp_path: Path, tool_name: str, arguments: dict[str, str]
) -> None:
    target = tmp_path / "auth.py"
    target.write_text("token = True", encoding="utf-8")
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("investigue a autenticação")
    resolved = {
        key: value.format(file=str(target), root=str(tmp_path))
        for key, value in arguments.items()
    }
    actions = [
        _action(
            1,
            NextActionType.INSPECT,
            tool_name=tool_name,
            target="auth",
            arguments=resolved,
            execution_mode=ExecutionMode.READ_ONLY,
        ),
        _action(
            2,
            NextActionType.RESPOND,
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
    ]

    result = _loop(goal, _state(goal, tmp_path, registry), registry, actions)

    assert registry.calls[0][0] == tool_name
    assert result.records[0].observation.source is ObservationSource.LIVE_TOOL
    assert result.records[0].observation.status is ObservationStatus.SUCCESS
    assert result.termination_reason == "GOAL_COMPLETED"


@pytest.mark.parametrize("agent", [Agent.CODEX, Agent.DEEPSEEK])
def test_read_only_delegation_executes_selected_agent(tmp_path: Path, agent: Agent) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("analise a falha")
    actions = [
        _action(
            1,
            NextActionType.DELEGATE,
            target_agent=agent,
            execution_mode=ExecutionMode.READ_ONLY,
            arguments={
                "task": "model override",
                "path": str(tmp_path / "auth.py"),
                "max_bytes": 999999,
            },
            required_capabilities=(Capability.CODE_ANALYSIS,),
            reason_code=OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
        ),
        _action(
            2,
            NextActionType.RESPOND,
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
    ]

    result = _loop(goal, _state(goal, tmp_path, registry), registry, actions)

    assert registry.calls[0][0] == f"delegate_to_{agent.value}"
    assert registry.calls[0][2]["execution_mode"] == "READ_ONLY"
    assert registry.calls[0][1]["task"] == "step 1"
    assert "path" not in registry.calls[0][1]
    assert "max_bytes" not in registry.calls[0][1]
    assert result.records[0].observation.source is ObservationSource.LIVE_AGENT


def test_reaching_delegation_budget_still_allows_terminal_response(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoal(
        "g-budget-agent",
        "manda pro Codex analisar",
        "diagnostico",
        execution_requested=True,
        explicit_agent=Agent.CODEX,
        agent_source=AgentSource.EXPLICIT_USER,
    )
    state = WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        project_snapshot={"project_path": str(tmp_path)},
        tool_names=registry.names(),
        budget=OrchestrationBudget(max_steps=2, max_delegations=1),
        authority_facts=("BOUNDED_LIVE",),
    ).state
    actions = [
        _action(
            1,
            NextActionType.DELEGATE,
            target_agent=Agent.CODEX,
            execution_mode=ExecutionMode.READ_ONLY,
            reason_code=OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
        ),
        _action(
            2,
            NextActionType.RESPOND,
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
    ]

    result = _loop(goal, state, registry, actions)

    assert result.final_state.delegations == 1
    assert result.termination_reason == OrchestrationReasonCode.GOAL_COMPLETED.value
    assert [item.next_action.action for item in result.records] == [
        NextActionType.DELEGATE,
        NextActionType.RESPOND,
    ]


def test_codex_mutation_executes_then_verifies(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("descobre por que o teste falha e corrige")
    assert goal.mutation_required is RequirementValue.TRUE
    actions = [
        _action(
            1,
            NextActionType.DELEGATE,
            target_agent=Agent.CODEX,
            execution_mode=ExecutionMode.MUTATION,
            required_capabilities=(Capability.CODE_EDIT,),
            reason_code=OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
        ),
        _action(
            2,
            NextActionType.EXECUTE,
            tool_name="run_project_tests",
            arguments={"project_path": str(tmp_path)},
            execution_mode=ExecutionMode.READ_ONLY,
            reason_code=OrchestrationReasonCode.SUFFICIENT_INFORMATION,
        ),
    ]

    result = _loop(goal, _state(goal, tmp_path, registry), registry, actions)

    assert [call[0] for call in registry.calls] == [
        "delegate_to_codex",
        "run_project_tests",
    ]
    assert result.effect_counts["filesystem_mutations"] == 1
    assert result.records[-1].observation.verification_status is VerificationStatus.VERIFIED
    assert result.records[-1].observation.goal_completed is True
    assert result.termination_reason == "GOAL_COMPLETED"


def test_read_only_forbidden_and_explicit_agent_constraints_block(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    read_only = UserGoalBuilder().build("analisa sem alterar nada")
    mutate = _action(
        1,
        NextActionType.DELEGATE,
        target_agent=Agent.CODEX,
        execution_mode=ExecutionMode.MUTATION,
        reason_code=OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
    )
    blocked = _loop(read_only, _state(read_only, tmp_path, registry), registry, [mutate])
    assert blocked.records[0].observation.status is ObservationStatus.BLOCKED
    assert "READ_ONLY_CONFLICT" in blocked.records[0].observation.errors
    assert not registry.calls

    forbidden = UserGoalBuilder().build("não use o Codex para resolver isso")
    denied = _loop(forbidden, _state(forbidden, tmp_path, registry), registry, [mutate])
    assert "FORBIDDEN_AGENT" in denied.records[0].observation.errors
    assert not registry.calls

    explicit = UserGoal(
        "g-explicit",
        "usa DeepSeek",
        "diagnóstico",
        explicit_agent=Agent.DEEPSEEK,
        agent_source=AgentSource.EXPLICIT_USER,
    )
    substituted = _action(
        2,
        NextActionType.DELEGATE,
        target_agent=Agent.CODEX,
        execution_mode=ExecutionMode.READ_ONLY,
        reason_code=OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
    )
    denied = _loop(explicit, _state(explicit, tmp_path, registry), registry, [substituted])
    assert "EXPLICIT_AGENT_NOT_PRESERVED" in denied.records[0].observation.errors
    assert not registry.calls


def test_explicit_agent_cannot_be_bypassed_by_local_inspection(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoal(
        "g-explicit-use",
        "manda pro Codex analisar",
        "diagnóstico do Codex",
        execution_requested=True,
        explicit_agent=Agent.CODEX,
        agent_source=AgentSource.EXPLICIT_USER,
    )
    actions = [
        _action(
            1,
            NextActionType.INSPECT,
            tool_name="get_project_git_state",
            target="project",
            arguments={"project_path": str(tmp_path)},
            execution_mode=ExecutionMode.READ_ONLY,
        ),
        _action(
            2,
            NextActionType.RESPOND,
            objective="diagnóstico local",
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
    ]

    result = _loop(goal, _state(goal, tmp_path, registry), registry, actions)

    assert result.records[-1].observation.authority_outcome == "EXPLICIT_AGENT_NOT_USED"
    assert [call[0] for call in registry.calls] == ["get_project_git_state"]


def test_project_resolution_updates_factual_world_state(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    registry.results["resolve_project"] = {
        "ok": True,
        "root": str(tmp_path),
        "project_id": "project-1",
        "files": ["auth.py"],
    }
    goal = UserGoalBuilder().build("localize o projeto")
    action = _action(
        1,
        NextActionType.INSPECT,
        tool_name="resolve_project",
        target="project",
        arguments={"query": "project"},
        execution_mode=ExecutionMode.READ_ONLY,
    )

    result = _loop(goal, _state(goal, tmp_path, registry, max_steps=1), registry, [action])

    assert result.final_state.project.path == str(tmp_path)
    assert result.final_state.project.project_id == "project-1"
    assert str(tmp_path) in result.records[0].observation.artifacts


def test_permitted_agent_is_not_mandatory_and_generic_mutation_does_not_force_codex(
    tmp_path: Path,
) -> None:
    registry = _Registry(tmp_path)
    permitted = UserGoalBuilder().build("analisa isso e usa o Codex se precisar")
    inspect = _action(
        1,
        NextActionType.INSPECT,
        objective="inspect project",
        tool_name="get_project_git_state",
        target="project",
        arguments={"project_path": str(tmp_path)},
        execution_mode=ExecutionMode.READ_ONLY,
    )
    result = _loop(permitted, _state(permitted, tmp_path, registry), registry, [inspect])
    assert registry.calls[0][0] == "get_project_git_state"

    generic = UserGoalBuilder().build("crie o arquivo necessário")
    local_write = _action(
        2,
        NextActionType.EXECUTE,
        tool_name="filesystem_write_text",
        arguments={"directory": str(tmp_path), "name": "x.py", "content": "x = 1"},
        execution_mode=ExecutionMode.MUTATION,
        reason_code=OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
    )
    result = _loop(generic, _state(generic, tmp_path, registry), registry, [local_write])
    assert registry.calls[-1][0] == "filesystem_write_text"
    assert all(call[0] != "delegate_to_codex" for call in registry.calls[-1:])


def test_authority_block_becomes_observation_and_policy_replans(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("resolva o problema")
    premature = _action(
        1,
        NextActionType.RESPOND,
        reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        objective="corrigido",
    )
    inspect = _action(
        2,
        NextActionType.INSPECT,
        tool_name="get_project_git_state",
        target="project",
        arguments={"project_path": str(tmp_path)},
        execution_mode=ExecutionMode.READ_ONLY,
    )

    result = _loop(goal, _state(goal, tmp_path, registry), registry, [premature, inspect])

    assert len(result.records) == 2
    assert result.records[0].observation.source is ObservationSource.EXECUTION_AUTHORITY
    assert result.records[0].observation.authority_outcome == "PREMATURE_RESPONSE"
    assert registry.calls[0][0] == "get_project_git_state"


def test_goal_completed_reason_on_inspect_does_not_terminate_before_respond(
    tmp_path: Path,
) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("inspecione o estado Git")
    actions = [
        _action(
            1,
            NextActionType.INSPECT,
            tool_name="get_project_git_state",
            target="project",
            arguments={"project_path": str(tmp_path)},
            execution_mode=ExecutionMode.READ_ONLY,
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
        _action(
            2,
            NextActionType.RESPOND,
            objective="estado Git observado",
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
    ]

    result = _loop(goal, _state(goal, tmp_path, registry), registry, actions)

    assert [item.next_action.action for item in result.records] == [
        NextActionType.INSPECT,
        NextActionType.RESPOND,
    ]
    assert result.records[0].observation.verification_status is VerificationStatus.VERIFIED
    assert result.termination_reason == "GOAL_COMPLETED"


def test_tool_error_availability_change_and_duplicate_are_observations(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    registry.results["get_project_git_state"] = {
        "ok": False,
        "error": "TOOL_FAILED",
        "message": "git unavailable",
    }
    goal = UserGoalBuilder().build("investigue")
    inspect = _action(
        1,
        NextActionType.INSPECT,
        objective="inspect project",
        tool_name="get_project_git_state",
        target="project",
        arguments={"project_path": str(tmp_path)},
        execution_mode=ExecutionMode.READ_ONLY,
    )
    failed = _loop(goal, _state(goal, tmp_path, registry), registry, [inspect])
    assert failed.records[0].observation.status is ObservationStatus.FAILURE
    assert failed.records[0].observation.errors[0] == "TOOL_FAILED"

    registry = _Registry(tmp_path)
    state = _state(goal, tmp_path, registry)
    registry.codex.is_available = False
    delegate = _action(
        2,
        NextActionType.DELEGATE,
        target_agent=Agent.CODEX,
        execution_mode=ExecutionMode.READ_ONLY,
        reason_code=OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
    )
    unavailable = _loop(goal, state, registry, [delegate])
    assert unavailable.records[0].observation.authority_outcome == (
        "AVAILABILITY_CHANGED_BEFORE_DISPATCH"
    )
    assert not registry.calls

    registry = _Registry(tmp_path)
    repeated = [inspect, _action(
        3,
        NextActionType.INSPECT,
        objective="inspect project",
        tool_name="get_project_git_state",
        target="project",
        arguments={"project_path": str(tmp_path)},
        execution_mode=ExecutionMode.READ_ONLY,
    )]
    duplicate = _loop(goal, _state(goal, tmp_path, registry), registry, repeated)
    assert len(registry.calls) == 1
    assert duplicate.records[-1].observation.authority_outcome == "DUPLICATE_ACTION"


def test_runner_emits_live_telemetry_and_preserves_original_request(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    actions = [
        _action(
            1,
            NextActionType.DELEGATE,
            target_agent=Agent.CODEX,
            execution_mode=ExecutionMode.MUTATION,
            reason_code=OrchestrationReasonCode.CODE_MUTATION_REQUIRED,
        ),
        _action(
            2,
            NextActionType.EXECUTE,
            tool_name="run_project_tests",
            arguments={"project_path": str(tmp_path)},
            execution_mode=ExecutionMode.READ_ONLY,
            reason_code=OrchestrationReasonCode.SUFFICIENT_INFORMATION,
        ),
    ]
    runner = BoundedLiveOrchestrationRunner(
        object(),
        registry,
        budget=OrchestrationBudget(max_steps=4),
        policy=ScriptedOrchestrationPolicy(actions),
        fast_path_enabled=True,
        decision_cache_enabled=True,
    )
    context = SimpleNamespace(
        project_root=str(tmp_path),
        focused_project_root=None,
        codex_job_id=None,
        codex_job_status=None,
        focused_agent="codex",
        focused_session="thread-1",
        active_project=tmp_path.name,
        known_projects=(),
    )
    user_text = "descobre por que o teste falha e corrige"

    result = runner.run(
        user_text,
        runtime_context=context,
        conversation_id="conversation-1",
    )

    telemetry = result["orchestration"]["telemetry"]
    assert result["orchestration"]["mode"] == "BOUNDED_LIVE"
    assert telemetry["successful_completion"] is True
    assert telemetry["verified_goal_completion"] is True
    assert telemetry["model_calls"] == 0
    assert telemetry["tool_calls"] == 2
    assert telemetry["authority_blocks"] == 0
    assert registry.calls[0][2]["original_user_text"] == user_text
    assert registry.calls[0][2]["focused_codex_thread_id"] == "thread-1"
    assert any(event == "orchestration_bounded_live" for event, _ in registry.logger.events)


def test_action_space_prunes_mutation_and_unavailable_agents_without_hiding_state(
    tmp_path: Path,
) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("analise sem alterar arquivos e não use o Codex")
    state = _state(goal, tmp_path, registry)

    allowed = ActionSpaceBuilder().build(
        goal,
        state,
        tool_specs=registry.specs(),
    )

    assert Agent.CODEX not in allowed.candidate_agents
    assert "filesystem_write_text" not in allowed.tools
    assert "delegate_to_codex" not in allowed.tools
    assert "delegate_to_deepseek" not in allowed.tools
    assert "get_codex_job_status" not in allowed.tools
    assert "READ_ONLY" in allowed.active_constraints
    assert Agent.CODEX in state.agents
    assert state.agents[Agent.CODEX].available is True


def test_verified_completion_uses_fast_path_without_a_model_call(
    tmp_path: Path,
) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("verifique o projeto")
    verified = Observation(
        observation_id="verified-1",
        source=ObservationSource.LIVE_TOOL,
        action_id="prior-action",
        status=ObservationStatus.SUCCESS,
        summary="Verificação concluída.",
        verification_status=VerificationStatus.VERIFIED,
        goal_completed=True,
    )
    state = _state(goal, tmp_path, registry).evolve(
        observations=(verified,),
        step=1,
        state_version=1,
    )
    policy = ScriptedOrchestrationPolicy([])
    result = OrchestrationLoop(
        policy,
        sink=LiveActionSink(
            registry,
            goal,
            run_id="run-fast",
            conversation_id="conversation-fast",
        ),
        action_space_builder=ActionSpaceBuilder(),
        tool_specs=registry.specs(),
        fast_path=OrchestrationFastPath(),
    ).run(goal, state)

    assert result.termination_reason == OrchestrationReasonCode.GOAL_COMPLETED.value
    assert result.fast_path_decisions == 1
    assert result.model_calls == 0
    assert result.records[0].next_action.action is NextActionType.RESPOND


def test_semantically_identical_state_reuses_cached_decision(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("preciso informar um valor externo")
    state = _state(goal, tmp_path, registry)
    policy = ScriptedOrchestrationPolicy(
        [
            _action(
                1,
                NextActionType.ASK_USER,
                reason_code=OrchestrationReasonCode.USER_INPUT_REQUIRED,
            )
        ]
    )
    cache = OrchestrationDecisionCache()

    first = OrchestrationLoop(
        policy,
        action_space_builder=ActionSpaceBuilder(),
        tool_specs=registry.specs(),
        decision_cache=cache,
    ).run(goal, state)
    second = OrchestrationLoop(
        policy,
        action_space_builder=ActionSpaceBuilder(),
        tool_specs=registry.specs(),
        decision_cache=cache,
    ).run(goal, state)

    assert first.records[0].decision_source == "POLICY"
    assert second.model_calls == 0
    assert second.decision_cache_hits == 1
    assert second.decision_cache_misses == 0
    assert second.records[0].decision_source == "CACHE"


def test_read_only_tool_rejects_mutation_execution_mode(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    goal = UserGoalBuilder().build("corrija o teste")
    action = _action(
        1,
        NextActionType.EXECUTE,
        tool_name="run_project_tests",
        arguments={"project_path": str(tmp_path)},
        execution_mode=ExecutionMode.MUTATION,
        reason_code=OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED,
    )

    result = _loop(goal, _state(goal, tmp_path, registry), registry, [action])

    assert "TOOL_MODE_MISMATCH" in result.records[0].validation.violations
    assert not registry.calls


def test_failing_tests_are_observed_results_not_tool_failures(tmp_path: Path) -> None:
    registry = _Registry(tmp_path)
    registry.results["run_project_tests"] = {
        "ok": False,
        "returncode": 1,
        "output": "1 failed",
        "path": str(tmp_path),
    }
    goal = UserGoalBuilder().build("descubra por que o teste falha e corrija")
    action = _action(
        1,
        NextActionType.EXECUTE,
        tool_name="run_project_tests",
        arguments={"project_path": str(tmp_path)},
        execution_mode=ExecutionMode.READ_ONLY,
        reason_code=OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED,
    )

    result = _loop(goal, _state(goal, tmp_path, registry), registry, [action])

    observation = result.records[0].observation
    assert observation.status is ObservationStatus.SUCCESS
    assert observation.verification_status is VerificationStatus.FAILED
    assert result.final_state.failure_count == 0


def test_explicit_project_path_discards_unrelated_focused_job(tmp_path: Path) -> None:
    project = tmp_path / "isolated"
    project.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    registry = _Registry(tmp_path)

    class CapturingPolicy(ScriptedOrchestrationPolicy):
        def __init__(self):
            super().__init__(
                [
                    _action(
                        1,
                        NextActionType.ASK_USER,
                        reason_code=OrchestrationReasonCode.USER_INPUT_REQUIRED,
                    )
                ]
            )
            self.states = []

        def decide(self, user_goal, world_state, allowed_actions=None):
            self.states.append(world_state)
            return super().decide(user_goal, world_state, allowed_actions)

    policy = CapturingPolicy()
    runner = BoundedLiveOrchestrationRunner(
        object(),
        registry,
        budget=OrchestrationBudget(max_steps=2),
        policy=policy,
        fast_path_enabled=False,
        decision_cache_enabled=False,
    )
    context = SimpleNamespace(
        project_root=str(unrelated),
        focused_project_root=str(unrelated),
        codex_job_id="job-from-another-project",
        codex_job_status="running",
        focused_agent="codex",
        focused_session="thread-from-another-project",
        active_project=unrelated.name,
        known_projects=(),
    )

    runner.run(
        f"No projeto {project}, investigue o teste",
        runtime_context=context,
        conversation_id="conversation-isolated",
    )

    assert policy.states[0].project.path == str(project.resolve())
    assert policy.states[0].jobs == ()
