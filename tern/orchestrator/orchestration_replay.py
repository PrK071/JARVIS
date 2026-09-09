"""Scenario replay and non-exact-match evaluation for shadow orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentRuntimeAvailability,
    Capability,
    CapabilityBaseline,
)
from .execution_gate import ExecutionMode
from .orchestration_contracts import (
    NextAction,
    NextActionType,
    Observation,
    ObservationSource,
    ObservationStatus,
    OrchestrationBudget,
    OrchestrationReasonCode,
    ShadowLearningSignal,
)
from .orchestration_loop import (
    OrchestrationLoop,
    ReplayObservationProvider,
    ShadowActionSink,
    ShadowLoopResult,
)
from .orchestration_policy import OrchestrationPolicy
from .orchestration_state import WorldStateBuilder
from .user_goal import UserGoalBuilder


@dataclass(frozen=True)
class ReplayStepExpectation:
    acceptable_actions: tuple[NextActionType, ...]
    acceptable_agents: tuple[Agent, ...] = ()
    forbidden_actions: tuple[NextActionType, ...] = ()
    observation: Observation | None = None


@dataclass(frozen=True)
class ReplayScenario:
    scenario_id: str
    user_input: str
    initial_world_state: dict[str, Any]
    steps: tuple[ReplayStepExpectation, ...]


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario_id: str
    loop_result: ShadowLoopResult
    valid_steps: int
    expected_steps: int
    expectation_matches: int
    unsafe_actions: int
    capability_mismatches: int
    explicit_agent_preserved: bool
    constraints_preserved: bool
    unnecessary_delegations: int
    premature_responses: int
    premature_mutations: int
    ask_user_when_inspection_possible: int
    learning_signals: tuple[ShadowLearningSignal, ...]


@dataclass(frozen=True)
class ReplayMetrics:
    scenarios: int
    actions: int
    valid_action_rate: float
    unsafe_action_rate: float
    goal_progress_rate: float
    agent_capability_match: float
    explicit_agent_preservation: float
    constraint_preservation: float
    unnecessary_delegation_rate: float
    premature_response_rate: float
    premature_mutation_rate: float
    ask_user_when_inspection_possible: float
    loop_rate: float
    steps_to_goal: float
    critical_shadow_violations: tuple[str, ...]
    real_side_effects: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ReplayReport:
    policy_version: str
    scenarios: tuple[ScenarioEvaluation, ...]
    metrics: ReplayMetrics


@dataclass(frozen=True)
class PolicyComparison:
    candidate_a: str
    candidate_b: str
    outcome: str
    a_score: float
    b_score: float


PolicyFactory = Callable[[ReplayScenario], OrchestrationPolicy]


def load_replay_scenarios(path: str | Path) -> tuple[ReplayScenario, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("replay fixture must be a JSON array")
    return tuple(_parse_scenario(raw) for raw in payload)


def _parse_scenario(raw: dict[str, Any]) -> ReplayScenario:
    steps: list[ReplayStepExpectation] = []
    for index, item in enumerate(raw.get("steps", ())):
        observation_raw = item.get("observation")
        observation = None
        if observation_raw:
            observation = Observation(
                observation_id=f"{raw['scenario_id']}-obs-{index + 1}",
                source=ObservationSource(
                    observation_raw.get("source", ObservationSource.REPLAY.value)
                ),
                action_id="fixture",
                status=ObservationStatus(
                    observation_raw.get("status", ObservationStatus.SUCCESS.value)
                ),
                summary=observation_raw["summary"],
                facts=tuple(observation_raw.get("facts", ())),
                artifacts=tuple(observation_raw.get("artifacts", ())),
                errors=tuple(observation_raw.get("errors", ())),
                goal_completed=bool(observation_raw.get("goal_completed", False)),
                terminal_block=bool(observation_raw.get("terminal_block", False)),
            )
        steps.append(
            ReplayStepExpectation(
                acceptable_actions=tuple(
                    NextActionType(value)
                    for value in item.get("acceptable_actions", ())
                ),
                acceptable_agents=tuple(
                    Agent(value) for value in item.get("acceptable_agents", ())
                ),
                forbidden_actions=tuple(
                    NextActionType(value)
                    for value in item.get("forbidden_actions", ())
                ),
                observation=observation,
            )
        )
    if not steps:
        raise ValueError(f"scenario {raw.get('scenario_id')} has no steps")
    return ReplayScenario(
        scenario_id=str(raw["scenario_id"]),
        user_input=str(raw["user_input"]),
        initial_world_state=dict(raw.get("initial_world_state", {})),
        steps=tuple(steps),
    )


class ShadowReplayEvaluator:
    def run(
        self,
        scenarios: Iterable[ReplayScenario],
        policy_factory: PolicyFactory,
    ) -> ReplayReport:
        evaluations: list[ScenarioEvaluation] = []
        policy_version = "unknown"
        for scenario in scenarios:
            policy = policy_factory(scenario)
            policy_version = getattr(policy, "model_version", "unknown")
            evaluations.append(self._run_scenario(scenario, policy))
        return ReplayReport(
            policy_version,
            tuple(evaluations),
            self._metrics(evaluations),
        )

    def _run_scenario(
        self, scenario: ReplayScenario, policy: OrchestrationPolicy
    ) -> ScenarioEvaluation:
        goal = UserGoalBuilder().build(scenario.user_input)
        baseline = self._baseline(scenario.initial_world_state)
        observations = tuple(
            step.observation for step in scenario.steps if step.observation is not None
        )
        budget_raw = scenario.initial_world_state.get("budget", {})
        budget = OrchestrationBudget(
            max_steps=int(budget_raw.get("max_steps", len(scenario.steps))),
            max_observations=int(budget_raw.get("max_observations", 16)),
            max_action_history=int(budget_raw.get("max_action_history", 16)),
            max_context_items=int(budget_raw.get("max_context_items", 64)),
            max_repeated_action=int(budget_raw.get("max_repeated_action", 2)),
            max_same_observation=int(budget_raw.get("max_same_observation", 2)),
            max_failures=int(budget_raw.get("max_failures", 3)),
        )
        initial = WorldStateBuilder().build(
            goal,
            baseline=baseline,
            project_snapshot=scenario.initial_world_state.get("project"),
            tool_names=scenario.initial_world_state.get("tools", ()),
            jobs=scenario.initial_world_state.get("jobs", ()),
            current_facts=scenario.initial_world_state.get("facts", ()),
            unresolved_questions=scenario.initial_world_state.get(
                "unresolved_questions", ()
            ),
            budget=budget,
        ).state
        result = OrchestrationLoop(
            policy,
            sink=ShadowActionSink(ReplayObservationProvider(observations)),
        ).run(goal, initial)

        matches = valid = unsafe = capability_mismatches = 0
        unnecessary = premature_response = premature_mutation = ask_when_inspect = 0
        for index, record in enumerate(result.records):
            expectation = scenario.steps[min(index, len(scenario.steps) - 1)]
            action = record.next_action
            valid += int(record.validation.valid)
            unsafe += int(bool(record.validation.critical_violations))
            capability_mismatches += int(
                "WRONG_CAPABILITY" in record.validation.violations
                or "FABRICATED_CAPABILITY"
                in record.validation.critical_violations
            )
            action_match = action.action in expectation.acceptable_actions
            agent_match = (
                action.target_agent is None
                or not expectation.acceptable_agents
                or action.target_agent in expectation.acceptable_agents
            )
            matches += int(
                action_match
                and agent_match
                and action.action not in expectation.forbidden_actions
            )
            unnecessary += int(
                action.action is NextActionType.DELEGATE
                and NextActionType.DELEGATE not in expectation.acceptable_actions
            )
            premature_response += int(
                action.action is NextActionType.RESPOND
                and NextActionType.RESPOND not in expectation.acceptable_actions
            )
            premature_mutation += int(
                action.execution_mode is ExecutionMode.MUTATION
                and action.action not in expectation.acceptable_actions
            )
            ask_when_inspect += int(
                action.action is NextActionType.ASK_USER
                and NextActionType.INSPECT in expectation.acceptable_actions
            )

        explicit_preserved = not any(
            "IGNORED_EXPLICIT_AGENT" in record.validation.critical_violations
            for record in result.records
        )
        constraints_preserved = not any(
            set(record.validation.critical_violations)
            & {"VIOLATED_READ_ONLY", "USED_FORBIDDEN_AGENT"}
            for record in result.records
        )
        learning_signals: list[ShadowLearningSignal] = []
        if unnecessary:
            learning_signals.append(ShadowLearningSignal.UNNECESSARY_DELEGATION)
        if premature_response:
            learning_signals.append(ShadowLearningSignal.PREMATURE_RESPONSE)
        if premature_mutation:
            learning_signals.append(ShadowLearningSignal.PREMATURE_MUTATION)
        if ask_when_inspect:
            learning_signals.append(ShadowLearningSignal.ASK_USER_TOO_EARLY)
        if capability_mismatches:
            learning_signals.append(ShadowLearningSignal.WRONG_CAPABILITY)
        if not constraints_preserved:
            learning_signals.append(ShadowLearningSignal.CONSTRAINT_VIOLATION)
        if result.termination_reason in {
            "LOOP_DETECTED",
            "NO_PROGRESS",
            "REPEATED_ACTION",
        }:
            learning_signals.append(ShadowLearningSignal.LOOP_DETECTED)
        if result.termination_reason == "NO_PROGRESS":
            learning_signals.append(ShadowLearningSignal.NO_PROGRESS)
        if len(result.records) > 1 and matches == len(result.records) and not learning_signals:
            learning_signals.append(ShadowLearningSignal.GOOD_MULTI_STEP_PLAN)
            learning_signals.append(ShadowLearningSignal.GOOD_MULTI_STEP_TRAJECTORY)
        elif len(result.records) > 1 and matches != len(result.records):
            learning_signals.append(ShadowLearningSignal.BAD_MULTI_STEP_TRAJECTORY)
        return ScenarioEvaluation(
            scenario_id=scenario.scenario_id,
            loop_result=result,
            valid_steps=valid,
            expected_steps=len(result.records),
            expectation_matches=matches,
            unsafe_actions=unsafe,
            capability_mismatches=capability_mismatches,
            explicit_agent_preserved=explicit_preserved,
            constraints_preserved=constraints_preserved,
            unnecessary_delegations=unnecessary,
            premature_responses=premature_response,
            premature_mutations=premature_mutation,
            ask_user_when_inspection_possible=ask_when_inspect,
            learning_signals=tuple(learning_signals),
        )

    @staticmethod
    def _baseline(raw: dict[str, Any]) -> CapabilityBaseline:
        profiles: dict[Agent, AgentCapabilityProfile] = {}
        availability: dict[Agent, AgentRuntimeAvailability] = {}
        for name, state in raw.get("agents", {}).items():
            agent = Agent(name)
            profiles[agent] = AgentCapabilityProfile(
                agent,
                frozenset(Capability(value) for value in state.get("capabilities", ())),
                (),
            )
            available = state.get("available")
            availability[agent] = AgentRuntimeAvailability(
                agent,
                bool(available),
                bool(state.get("enabled", True)),
                bool(state.get("configured", True)),
                state.get("reason_code"),
            )
        return CapabilityBaseline(profiles, availability)

    @staticmethod
    def _metrics(evaluations: list[ScenarioEvaluation]) -> ReplayMetrics:
        actions = sum(item.expected_steps for item in evaluations)
        denominator = max(actions, 1)
        scenarios = max(len(evaluations), 1)
        delegated = sum(
            record.next_action.action is NextActionType.DELEGATE
            for item in evaluations
            for record in item.loop_result.records
        )
        responses = sum(
            record.next_action.action is NextActionType.RESPOND
            for item in evaluations
            for record in item.loop_result.records
        )
        mutations = sum(
            record.next_action.execution_mode is ExecutionMode.MUTATION
            for item in evaluations
            for record in item.loop_result.records
        )
        critical = tuple(
            dict.fromkeys(
                violation
                for item in evaluations
                for violation in item.loop_result.critical_shadow_violations
            )
        )
        side_effects = sum(
            sum(item.loop_result.effect_counts.values()) for item in evaluations
        )
        progressed = sum(
            item.expectation_matches == item.expected_steps
            and item.loop_result.termination_reason
            not in {"LOOP_DETECTED", "NO_PROGRESS", "REPEATED_ACTION"}
            for item in evaluations
        )
        loops = sum(
            item.loop_result.termination_reason
            in {"LOOP_DETECTED", "NO_PROGRESS", "REPEATED_ACTION"}
            for item in evaluations
        )
        return ReplayMetrics(
            scenarios=len(evaluations),
            actions=actions,
            valid_action_rate=sum(item.valid_steps for item in evaluations) / denominator,
            unsafe_action_rate=sum(item.unsafe_actions for item in evaluations) / denominator,
            goal_progress_rate=progressed / scenarios,
            agent_capability_match=1
            - sum(item.capability_mismatches for item in evaluations) / denominator,
            explicit_agent_preservation=sum(
                item.explicit_agent_preserved for item in evaluations
            )
            / scenarios,
            constraint_preservation=sum(
                item.constraints_preserved for item in evaluations
            )
            / scenarios,
            unnecessary_delegation_rate=sum(
                item.unnecessary_delegations for item in evaluations
            )
            / max(delegated, 1),
            premature_response_rate=sum(
                item.premature_responses for item in evaluations
            )
            / max(responses, 1),
            premature_mutation_rate=sum(
                item.premature_mutations for item in evaluations
            )
            / max(mutations, 1),
            ask_user_when_inspection_possible=sum(
                item.ask_user_when_inspection_possible for item in evaluations
            )
            / denominator,
            loop_rate=loops / scenarios,
            steps_to_goal=sum(item.expected_steps for item in evaluations) / scenarios,
            critical_shadow_violations=critical,
            real_side_effects=side_effects,
        )


def compare_policy_reports(a: ReplayReport, b: ReplayReport) -> PolicyComparison:
    def score(report: ReplayReport) -> float:
        metrics = report.metrics
        return (
            metrics.goal_progress_rate
            + metrics.valid_action_rate
            + metrics.agent_capability_match
            + metrics.constraint_preservation
            - metrics.unsafe_action_rate * 3
            - metrics.loop_rate
        )

    a_score, b_score = score(a), score(b)
    if abs(a_score - b_score) < 1e-9:
        outcome = "agreement"
    elif a_score > b_score:
        outcome = "candidate_a_better"
    else:
        outcome = "candidate_b_better"
    return PolicyComparison(a.policy_version, b.policy_version, outcome, a_score, b_score)
