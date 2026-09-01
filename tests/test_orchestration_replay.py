from __future__ import annotations

from pathlib import Path

from tern.orchestrator.execution_gate import ExecutionMode
from tern.orchestrator.orchestration_contracts import (
    NextAction,
    NextActionType,
    OrchestrationReasonCode,
)
from tern.orchestrator.orchestration_policy import ScriptedOrchestrationPolicy
from tern.orchestrator.orchestration_replay import (
    ShadowReplayEvaluator,
    compare_policy_reports,
    load_replay_scenarios,
)
from tern.orchestrator.user_goal import UserGoalBuilder


FIXTURE = Path(__file__).parent / "fixtures" / "shadow_orchestration_scenarios.json"


def _reference_policy(scenario):
    goal = UserGoalBuilder().build(scenario.user_input)
    actions = []
    for index, step in enumerate(scenario.steps, 1):
        action_type = step.acceptable_actions[0]
        agent = (
            step.acceptable_agents[0]
            if action_type is NextActionType.DELEGATE and step.acceptable_agents
            else None
        )
        if action_type is NextActionType.DELEGATE and agent is None:
            agent = goal.explicit_agent
        mode = None
        target = None
        tool_name = None
        reason = OrchestrationReasonCode.SUFFICIENT_INFORMATION
        if action_type is NextActionType.INSPECT:
            target = "relevant project evidence"
            mode = ExecutionMode.READ_ONLY
            reason = OrchestrationReasonCode.REPOSITORY_INSPECTION_REQUIRED
        elif action_type is NextActionType.DELEGATE:
            mode = ExecutionMode.READ_ONLY
            reason = (
                OrchestrationReasonCode.EXPLICIT_AGENT_REQUIRED
                if goal.explicit_agent
                else OrchestrationReasonCode.EXPERT_ANALYSIS_REQUIRED
            )
        elif action_type is NextActionType.ASK_USER:
            reason = OrchestrationReasonCode.USER_INPUT_REQUIRED
        elif action_type is NextActionType.STOP:
            reason = OrchestrationReasonCode.GOAL_IMPOSSIBLE
        actions.append(
            NextAction(
                action_id=f"{scenario.scenario_id}-{index}",
                action=action_type,
                objective="advance the replay goal",
                reason_code=reason,
                target_agent=agent,
                target=target,
                tool_name=tool_name,
                execution_mode=mode,
            )
        )
    return ScriptedOrchestrationPolicy(actions)


def test_required_single_and_multi_step_scenarios_load() -> None:
    scenarios = load_replay_scenarios(FIXTURE)
    assert len(scenarios) == 14
    assert sum(len(item.steps) > 1 for item in scenarios) >= 3
    ids = {item.scenario_id for item in scenarios}
    assert {
        "generic_repair_no_executor",
        "explicit_codex_preserved",
        "conditional_codex_permission",
        "read_only_analysis",
        "delete_without_executor",
        "login_explanation_only",
        "repository_inspection_first",
        "codex_unavailable",
        "deepseek_filesystem_ineligible",
        "sufficient_information_no_delegation",
        "multistep_unknown_bug",
        "multistep_insufficient_then_ask",
        "multistep_agent_unavailable",
        "multistep_no_progress",
    } == ids


def test_replay_metrics_are_non_exact_match_and_report_safety() -> None:
    report = ShadowReplayEvaluator().run(
        load_replay_scenarios(FIXTURE), _reference_policy
    )
    metrics = report.metrics
    assert metrics.valid_action_rate == 1.0
    assert metrics.unsafe_action_rate == 0.0
    assert metrics.agent_capability_match == 1.0
    assert metrics.explicit_agent_preservation == 1.0
    assert metrics.constraint_preservation == 1.0
    assert metrics.unnecessary_delegation_rate == 0.0
    assert metrics.premature_response_rate == 0.0
    assert metrics.premature_mutation_rate == 0.0
    assert metrics.ask_user_when_inspection_possible == 0.0
    assert metrics.loop_rate == 1 / 14
    assert metrics.critical_shadow_violations == ()
    assert metrics.real_side_effects == 0


def test_same_scenarios_can_compare_two_policy_versions() -> None:
    scenarios = load_replay_scenarios(FIXTURE)
    evaluator = ShadowReplayEvaluator()
    a = evaluator.run(scenarios, _reference_policy)
    b = evaluator.run(scenarios, _reference_policy)
    comparison = compare_policy_reports(a, b)
    assert comparison.outcome == "agreement"
