from __future__ import annotations

import pytest

from tern.orchestrator.autonomy_foundation import Agent, Capability
from tern.orchestrator.execution_gate import ExecutionMode
from tern.orchestrator.orchestration_contracts import (
    AgentSource,
    AgentState,
    NextAction,
    NextActionType,
    Observation,
    ObservationSource,
    ObservationStatus,
    OrchestrationBudget,
    OrchestrationReasonCode,
    ProjectState,
    SemanticAction,
    UserGoal,
    WorldState,
)
from tern.orchestrator.task_requirement_grounding import RequirementValue
from tern.orchestrator.config import load_settings


def test_user_goal_is_versioned_and_separates_semantics_from_executor() -> None:
    goal = UserGoal(
        goal_id="goal-1",
        summary="delete o arquivo temporário",
        desired_outcome="arquivo temporário inexistente",
        semantic_action=SemanticAction.DELETE_OBJECT,
        execution_requested=True,
        mutation_required=RequirementValue.TRUE,
    )

    payload = goal.as_dict()
    assert payload["schema_version"] == "1"
    assert payload["semantic_action"] == "DELETE_OBJECT"
    assert payload["executor"]["explicit_agent"] is None
    assert payload["mutation"] == {"required": "TRUE", "forbidden": False}


def test_explicit_agent_requires_explicit_source() -> None:
    with pytest.raises(ValueError, match="set together"):
        UserGoal(
            goal_id="g",
            summary="manda pro Codex",
            desired_outcome="feito",
            explicit_agent=Agent.CODEX,
        )

    goal = UserGoal(
        goal_id="g",
        summary="manda pro Codex",
        desired_outcome="feito",
        explicit_agent=Agent.CODEX,
        agent_source=AgentSource.EXPLICIT_USER,
    )
    assert goal.as_dict()["executor"]["agent_source"] == "explicit_user"


def test_user_goal_rejects_conflicting_agent_and_mutation_constraints() -> None:
    with pytest.raises(ValueError, match="both permitted and forbidden"):
        UserGoal(
            goal_id="g",
            summary="x",
            desired_outcome="y",
            permitted_agents=(Agent.CODEX,),
            forbidden_agents=(Agent.CODEX,),
        )
    with pytest.raises(ValueError, match="both required and forbidden"):
        UserGoal(
            goal_id="g",
            summary="x",
            desired_outcome="y",
            mutation_required=RequirementValue.TRUE,
            mutation_forbidden=True,
        )


def test_next_action_is_closed_versioned_data_without_execution_handle() -> None:
    action = NextAction(
        action_id="a-1",
        action=NextActionType.DELEGATE,
        target_agent=Agent.CODEX,
        objective="investigar autenticação",
        execution_mode=ExecutionMode.READ_ONLY,
        required_capabilities=(Capability.REPOSITORY_READ,),
        reason_code=OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED,
    )

    payload = action.as_dict()
    assert payload["schema_version"] == "1"
    assert payload["action"] == "DELEGATE"
    assert payload["execution_mode"] == "READ_ONLY"
    assert not hasattr(action, "execute")
    assert not hasattr(action, "tool_registry")


def test_observation_and_world_state_are_bounded_structured_contracts() -> None:
    observation = Observation(
        observation_id="o-1",
        source=ObservationSource.REPLAY,
        action_id="a-1",
        status=ObservationStatus.SUCCESS,
        summary="Firebase inicializado duas vezes",
        facts=("firebase_init_duplicated",),
    )
    agent = AgentState(
        agent=Agent.CODEX,
        availability_known=True,
        available=True,
        eligible=True,
        capabilities=(Capability.REPOSITORY_READ,),
        execution_modes=(ExecutionMode.READ_ONLY,),
    )
    state = WorldState(
        goal_id="g-1",
        state_version=0,
        project=ProjectState(project_id="jarvis", working_tree="dirty"),
        agents={Agent.CODEX: agent},
        tools=("inspect_repo",),
        observations=(observation,),
        jobs=(),
        previous_actions=(),
        current_facts=("firebase_init_duplicated",),
        unresolved_questions=(),
        authority_facts=("shadow_only",),
        budget=OrchestrationBudget(max_steps=3),
    )

    payload = state.as_dict()
    assert payload["schema_version"] == "1"
    assert payload["budget"]["max_steps"] == 3
    assert payload["agents"]["codex"]["availability_known"] is True


def test_orchestration_budget_rejects_magic_zero_limits() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        OrchestrationBudget(max_steps=0)


def test_shadow_orchestration_configuration_is_opt_in_and_bounded() -> None:
    settings = load_settings({})
    assert settings.orchestration_shadow_enabled is False
    assert settings.orchestration_shadow_max_steps == 8
    configured = load_settings(
        {
            "ORCHESTRATION_SHADOW_ENABLED": "true",
            "ORCHESTRATION_SHADOW_MAX_STEPS": "3",
            "ORCHESTRATION_SHADOW_MAX_CONTEXT_ITEMS": "20",
        }
    )
    assert configured.orchestration_shadow_enabled is True
    assert configured.orchestration_shadow_max_steps == 3
    assert configured.orchestration_shadow_max_context_items == 20

    with pytest.raises(ValueError, match="orchestration shadow"):
        load_settings({"ORCHESTRATION_SHADOW_MAX_STEPS": "0"})
