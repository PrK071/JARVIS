from __future__ import annotations

from tern.orchestrator.autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentRuntimeAvailability,
    Capability,
    CapabilityBaseline,
)
from tern.orchestrator.execution_gate import ExecutionMode
from tern.orchestrator.orchestration_contracts import (
    NextAction,
    NextActionType,
    Observation,
    ObservationSource,
    ObservationStatus,
    OrchestrationBudget,
    OrchestrationReasonCode,
)
from tern.orchestrator.orchestration_loop import (
    OrchestrationLoop,
    ReplayObservationProvider,
    ShadowActionSink,
)
from tern.orchestrator.orchestration_policy import ScriptedOrchestrationPolicy
from tern.orchestrator.orchestration_state import WorldStateBuilder
from tern.orchestrator.user_goal import UserGoalBuilder


def _baseline() -> CapabilityBaseline:
    profiles = {
        Agent.CODEX: AgentCapabilityProfile(
            Agent.CODEX,
            frozenset(
                {
                    Capability.REPOSITORY_READ,
                    Capability.CODE_ANALYSIS,
                    Capability.CODE_EDIT,
                    Capability.MUTATION,
                }
            ),
            (),
        ),
        Agent.DEEPSEEK: AgentCapabilityProfile(
            Agent.DEEPSEEK,
            frozenset({Capability.GENERAL_REASONING, Capability.READ_ONLY}),
            (),
        ),
    }
    availability = {
        agent: AgentRuntimeAvailability(agent, True, True, True)
        for agent in profiles
    }
    return CapabilityBaseline(profiles, availability)


def _action(index: int, action: NextActionType, **kwargs) -> NextAction:
    objective = kwargs.pop("objective", f"step {index}")
    defaults = {
        "reason_code": OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED,
        "target": "authentication",
    }
    defaults.update(kwargs)
    return NextAction(
        action_id=f"a-{index}",
        action=action,
        objective=objective,
        **defaults,
    )


def _observation(index: int, summary: str, **kwargs) -> Observation:
    return Observation(
        observation_id=f"o-{index}",
        source=ObservationSource.REPLAY,
        action_id="fixture",
        status=ObservationStatus.SUCCESS,
        summary=summary,
        **kwargs,
    )


def test_multi_step_replay_observes_reduces_and_replans() -> None:
    goal = UserGoalBuilder().build("corrija o login")
    actions = [
        _action(1, NextActionType.INSPECT),
        _action(
            2,
            NextActionType.DELEGATE,
            target=None,
            target_agent=Agent.CODEX,
            execution_mode=ExecutionMode.READ_ONLY,
            required_capabilities=(Capability.REPOSITORY_READ,),
            reason_code=OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED,
        ),
        _action(
            3,
            NextActionType.RESPOND,
            target=None,
            reason_code=OrchestrationReasonCode.GOAL_COMPLETED,
        ),
    ]
    observations = [
        _observation(1, "Firebase initialization appears duplicated", facts=("firebase_init_duplicated",)),
        _observation(2, "Codex confirms duplicated initialization", facts=("duplicate_confirmed",)),
        _observation(3, "diagnosis ready", goal_completed=True),
    ]
    state = WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        budget=OrchestrationBudget(max_steps=5),
    ).state
    sink = ShadowActionSink(ReplayObservationProvider(observations))
    result = OrchestrationLoop(
        ScriptedOrchestrationPolicy(actions), sink=sink
    ).run(goal, state)

    assert [record.next_action.action for record in result.records] == [
        NextActionType.INSPECT,
        NextActionType.DELEGATE,
        NextActionType.RESPOND,
    ]
    assert result.final_state.step == 3
    assert "duplicate_confirmed" in result.final_state.current_facts
    assert result.termination_reason == "GOAL_COMPLETED"
    assert all(value == 0 for value in result.effect_counts.values())


def test_repeated_action_and_same_observation_terminate_no_progress() -> None:
    goal = UserGoalBuilder().build("investigue o erro")
    repeated = [_action(index, NextActionType.INSPECT) for index in range(1, 6)]
    same = [_observation(index, "same fact") for index in range(1, 6)]
    state = WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        budget=OrchestrationBudget(
            max_steps=6,
            max_repeated_action=2,
            max_same_observation=2,
        ),
    ).state
    result = OrchestrationLoop(
        ScriptedOrchestrationPolicy(repeated),
        sink=ShadowActionSink(ReplayObservationProvider(same)),
    ).run(goal, state)

    assert result.termination_reason in {"REPEATED_ACTION", "NO_PROGRESS"}
    assert len(result.records) == 3


def test_alternating_loop_is_detected() -> None:
    goal = UserGoalBuilder().build("investigue")
    actions = [
        _action(1, NextActionType.INSPECT, target="A", objective="inspect"),
        _action(2, NextActionType.INSPECT, target="B", objective="inspect"),
        _action(3, NextActionType.INSPECT, target="A", objective="inspect"),
        _action(4, NextActionType.INSPECT, target="B", objective="inspect"),
    ]
    observations = [
        _observation(1, "A"),
        _observation(2, "B"),
        _observation(3, "A"),
        _observation(4, "B"),
    ]
    state = WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        budget=OrchestrationBudget(max_steps=5, max_same_observation=4),
    ).state
    result = OrchestrationLoop(
        ScriptedOrchestrationPolicy(actions),
        sink=ShadowActionSink(ReplayObservationProvider(observations)),
    ).run(goal, state)
    assert result.termination_reason == "LOOP_DETECTED"


def test_max_steps_and_max_failures_terminate() -> None:
    goal = UserGoalBuilder().build("investigue")
    state = WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        budget=OrchestrationBudget(max_steps=2, max_failures=1),
    ).state
    invalid = _action(
        1,
        NextActionType.DELEGATE,
        target=None,
        target_agent=Agent.DEEPSEEK,
        execution_mode=ExecutionMode.MUTATION,
        required_capabilities=(Capability.FILESYSTEM_WRITE,),
    )
    result = OrchestrationLoop(ScriptedOrchestrationPolicy([invalid])).run(goal, state)
    assert result.termination_reason == "MAX_FAILURES"
    assert "FABRICATED_CAPABILITY" in result.critical_shadow_violations

    state2 = WorldStateBuilder().build(
        goal,
        baseline=_baseline(),
        budget=OrchestrationBudget(max_steps=1),
    ).state
    budgeted = OrchestrationLoop(
        ScriptedOrchestrationPolicy([_action(2, NextActionType.INSPECT)])
    ).run(goal, state2)
    assert budgeted.termination_reason == "BUDGET_EXHAUSTED"


def test_shadow_sink_has_no_execution_capability() -> None:
    sink = ShadowActionSink()
    assert not hasattr(sink, "tools")
    assert not hasattr(sink, "registry")
    assert not hasattr(sink, "executor")
    assert set(sink.effect_counts.values()) == {0}
