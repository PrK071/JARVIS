from __future__ import annotations

from tern.orchestrator.autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentRuntimeAvailability,
    Capability,
    CapabilityBaseline,
)
from tern.orchestrator.orchestration_contracts import (
    NextAction,
    NextActionType,
    Observation,
    ObservationSource,
    ObservationStatus,
    OrchestrationBudget,
    OrchestrationReasonCode,
    SemanticAction,
    UserGoal,
)
from tern.orchestrator.orchestration_state import WorldStateBuilder, WorldStateReducer


def _goal() -> UserGoal:
    return UserGoal(
        goal_id="g-1",
        summary="investigue o login",
        desired_outcome="causa identificada",
        semantic_action=SemanticAction.ANALYZE,
    )


def _baseline() -> CapabilityBaseline:
    profile = AgentCapabilityProfile(
        agent=Agent.CODEX,
        capabilities=frozenset(
            {Capability.REPOSITORY_READ, Capability.CODE_ANALYSIS}
        ),
        evidence=(),
    )
    return CapabilityBaseline(
        profiles={Agent.CODEX: profile},
        availability={
            Agent.CODEX: AgentRuntimeAvailability(
                agent=Agent.CODEX,
                available=True,
                enabled=True,
                configured=True,
            )
        },
    )


def test_world_state_builder_uses_compact_references_and_pure_agent_snapshots() -> None:
    budget = OrchestrationBudget(max_context_items=2)
    built = WorldStateBuilder().build(
        _goal(),
        baseline=_baseline(),
        project_snapshot={
            "project_path": "D:/JARVIS",
            "languages": ["Python", "TypeScript"],
            "git_branch": "main",
            "git_status": "dirty",
            "important_files": ["a.py", "b.py", "c.py"],
            "entry_points": ["main.py"],
        },
        tool_names=("inspect_repo", "inspect_repo", "run_tests"),
        jobs=({"job_id": "j-1", "status": "running", "task_summary": "x"},),
        budget=budget,
    )

    state = built.state
    assert state.project.project_id == "JARVIS"
    assert len(state.project.context_refs) == 2
    assert state.tools == ("inspect_repo", "run_tests")
    assert state.agents[Agent.CODEX].available is True
    assert not hasattr(state.agents[Agent.CODEX], "executor")
    assert built.build_ms >= 0
    import pytest
    with pytest.raises(TypeError):
        state.agents[Agent.CODEX] = state.agents[Agent.CODEX]


def test_reducer_is_deterministic_and_compacts_history() -> None:
    state = WorldStateBuilder().build(
        _goal(),
        baseline=_baseline(),
        budget=OrchestrationBudget(
            max_steps=4,
            max_observations=1,
            max_action_history=1,
            max_context_items=2,
        ),
    ).state
    reducer = WorldStateReducer()

    for index in range(2):
        action = NextAction(
            action_id=f"a-{index}",
            action=NextActionType.INSPECT,
            objective=f"inspect {index}",
            target="repo",
            reason_code=OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED,
        )
        observation = Observation(
            observation_id=f"o-{index}",
            source=ObservationSource.REPLAY,
            action_id=action.action_id,
            status=ObservationStatus.SUCCESS,
            summary=f"fact {index}",
            facts=(f"fact:{index}",),
        )
        state = reducer.reduce(state, action, observation)

    assert state.step == 2
    assert state.state_version == 2
    assert len(state.observations) == 1
    assert len(state.previous_actions) == 1
    assert state.current_facts == ("fact:0", "fact:1")


def test_reducer_rejects_fabricated_observation_binding() -> None:
    state = WorldStateBuilder().build(_goal(), baseline=_baseline()).state
    action = NextAction(
        action_id="a",
        action=NextActionType.RESPOND,
        objective="responder",
        reason_code=OrchestrationReasonCode.SUFFICIENT_INFORMATION,
    )
    observation = Observation(
        observation_id="o",
        source=ObservationSource.SYNTHETIC,
        action_id="different",
        status=ObservationStatus.PROPOSED,
        summary="shadow",
    )

    import pytest

    with pytest.raises(ValueError, match="does not match"):
        WorldStateReducer().reduce(state, action, observation)
