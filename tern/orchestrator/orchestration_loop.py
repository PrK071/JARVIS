"""Receding-horizon orchestration shared by shadow and bounded-live modes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .orchestration_contracts import (
    NextAction,
    NextActionType,
    Observation,
    ObservationSource,
    ObservationStatus,
    OrchestrationReasonCode,
    UserGoal,
    VerificationStatus,
    WorldState,
)
from .orchestration_policy import (
    ActionValidationResult,
    NextActionValidator,
    OrchestrationPolicy,
    PolicyOutputError,
)
from .orchestration_fast_path import (
    ActionSpaceBuilder,
    OrchestrationDecisionCache,
    OrchestrationFastPath,
)
from .orchestration_state import WorldStateReducer


class ShadowObservationProvider(Protocol):
    def observe(self, action: NextAction, state: WorldState) -> Observation:
        """Return replay/fixture/synthetic facts. It must not perform the action."""


@dataclass(frozen=True)
class ShadowAuthorityResult:
    candidate_effect: bool
    would_allow: bool | None
    reason_code: str
    live_authority: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_effect": self.candidate_effect,
            "would_allow": self.would_allow,
            "reason_code": self.reason_code,
            "authority": "SHADOW",
            "live_authority": self.live_authority,
        }


@dataclass(frozen=True)
class ShadowDecisionRecord:
    goal_id: str
    step: int
    world_state_version: int
    model_version: str
    next_action: NextAction
    validation: ActionValidationResult
    authority_shadow_result: Any
    observation: Observation
    capability_snapshot: dict[str, tuple[str, ...]]
    available_agents: tuple[str, ...]
    policy_inference_ms: float
    validation_ms: float
    state_reduce_ms: float
    authority_ms: float = 0.0
    execution_ms: float = 0.0
    decision_source: str = "POLICY"
    fast_path_reason: str | None = None
    total_step_ms: float = 0.0
    policy_stats: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "step": self.step,
            "world_state_version": self.world_state_version,
            "model_version": self.model_version,
            "next_action": self.next_action.as_dict(),
            "reason_code": self.next_action.reason_code.value,
            "evidence_refs": list(self.next_action.evidence_refs),
            "available_agents": list(self.available_agents),
            "capability_snapshot": {
                key: list(value) for key, value in self.capability_snapshot.items()
            },
            "authority_shadow_result": self.authority_shadow_result.as_dict(),
            "authority_result": self.authority_shadow_result.as_dict(),
            "validation": self.validation.as_dict(),
            "observation": self.observation.as_dict(),
            "timing": {
                "policy_inference_ms": self.policy_inference_ms,
                "validation_ms": self.validation_ms,
                "authority_ms": self.authority_ms,
                "execution_ms": self.execution_ms,
                "state_reduce_ms": self.state_reduce_ms,
                "total_step_ms": self.total_step_ms,
            },
            "decision_source": self.decision_source,
            "fast_path_reason": self.fast_path_reason,
            "policy_stats": dict(self.policy_stats or {}),
        }


class SyntheticObservationProvider:
    """Record a proposal as a proposal; never fabricate an action result."""

    def observe(self, action: NextAction, state: WorldState) -> Observation:
        return Observation(
            observation_id=f"synthetic-{action.action_id}",
            source=ObservationSource.SYNTHETIC,
            action_id=action.action_id,
            status=ObservationStatus.PROPOSED,
            summary="Shadow proposal recorded; no action was executed.",
        )


class ReplayObservationProvider:
    """Bounded deterministic observations for multi-step replay."""

    def __init__(self, observations: list[Observation] | tuple[Observation, ...]):
        self._observations = list(observations)

    def observe(self, action: NextAction, state: WorldState) -> Observation:
        if not self._observations:
            return SyntheticObservationProvider().observe(action, state)
        fixture = self._observations.pop(0)
        return Observation(
            observation_id=fixture.observation_id,
            source=fixture.source,
            action_id=action.action_id,
            status=fixture.status,
            summary=fixture.summary,
            facts=fixture.facts,
            artifacts=fixture.artifacts,
            state_changes=fixture.state_changes,
            errors=fixture.errors,
            authority_outcome=fixture.authority_outcome,
            tool_name=fixture.tool_name,
            agent=fixture.agent,
            verification_status=fixture.verification_status,
            goal_completed=fixture.goal_completed,
            terminal_block=fixture.terminal_block,
        )


class ShadowActionSink:
    """The loop's only action sink: records proposals and owns no executor handles."""

    __slots__ = ("_provider", "_records")

    def __init__(self, provider: ShadowObservationProvider | None = None):
        self._provider = provider or SyntheticObservationProvider()
        self._records: list[tuple[NextAction, Observation]] = []

    @property
    def effect_counts(self) -> dict[str, int]:
        return {
            "tools_executed": 0,
            "delegations": 0,
            "jobs_created": 0,
            "sessions_created": 0,
            "filesystem_mutations": 0,
            "git_mutations": 0,
            "external_effects": 0,
            "live_decisions_changed": 0,
        }

    @property
    def records(self) -> tuple[tuple[NextAction, Observation], ...]:
        return tuple(self._records)

    def record(
        self,
        action: NextAction,
        state: WorldState,
        validation: ActionValidationResult,
    ) -> tuple[Observation, ShadowAuthorityResult]:
        candidate_effect = action.action in {
            NextActionType.DELEGATE,
            NextActionType.EXECUTE,
        }
        if candidate_effect:
            would_allow = validation.valid
            reason = "WOULD_ALLOW" if would_allow else (
                validation.violations[0] if validation.violations else "WOULD_BLOCK"
            )
        else:
            would_allow = None
            reason = "NOT_APPLICABLE"
        authority = ShadowAuthorityResult(candidate_effect, would_allow, reason)
        if validation.valid:
            observation = self._provider.observe(action, state)
        else:
            observation = Observation(
                observation_id=f"validation-{action.action_id}",
                source=ObservationSource.SHADOW_VALIDATOR,
                action_id=action.action_id,
                status=ObservationStatus.BLOCKED,
                summary="Shadow action failed deterministic validation.",
                errors=validation.violations,
                terminal_block=False,
            )
        self._records.append((action, observation))
        return observation, authority


@dataclass(frozen=True)
class ShadowLoopResult:
    goal_id: str
    initial_state_version: int
    final_state: WorldState
    records: tuple[ShadowDecisionRecord, ...]
    termination_reason: str
    shadow_total_ms: float
    model_calls: int
    critical_shadow_violations: tuple[str, ...]
    effect_counts: dict[str, int]
    mode: str = "SHADOW"
    fast_path_decisions: int = 0
    decision_cache_hits: int = 0
    decision_cache_misses: int = 0
    telemetry: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "mode": self.mode,
            "initial_state_version": self.initial_state_version,
            "final_state_version": self.final_state.state_version,
            "termination_reason": self.termination_reason,
            "records": [item.as_dict() for item in self.records],
            "shadow_total_ms": self.shadow_total_ms,
            "model_calls": self.model_calls,
            "critical_shadow_violations": list(
                self.critical_shadow_violations
            ),
            "critical_violations": list(self.critical_shadow_violations),
            "effect_counts": dict(self.effect_counts),
            "fast_path_decisions": self.fast_path_decisions,
            "decision_cache_hits": self.decision_cache_hits,
            "decision_cache_misses": self.decision_cache_misses,
            "telemetry": dict(self.telemetry or {}),
        }


class OrchestrationLoop:
    """Choose one action, pass it to a sink, reduce factual state, and replan."""

    def __init__(
        self,
        policy: OrchestrationPolicy,
        *,
        validator: NextActionValidator | None = None,
        reducer: WorldStateReducer | None = None,
        sink: ShadowActionSink | None = None,
        action_space_builder: ActionSpaceBuilder | None = None,
        tool_specs: Iterable[Mapping[str, Any]] = (),
        fast_path: OrchestrationFastPath | None = None,
        decision_cache: OrchestrationDecisionCache | None = None,
    ):
        self.policy = policy
        self.validator = validator or NextActionValidator()
        self.reducer = reducer or WorldStateReducer()
        self.sink = sink or ShadowActionSink()
        self.action_space_builder = action_space_builder
        self.tool_specs = tuple(tool_specs)
        self.fast_path = fast_path
        self.decision_cache = decision_cache

    def run(self, goal: UserGoal, initial_state: WorldState) -> ShadowLoopResult:
        if initial_state.goal_id != goal.goal_id:
            raise ValueError("WorldState goal_id does not match UserGoal")
        started = time.perf_counter()
        state = initial_state
        records: list[ShadowDecisionRecord] = []
        action_fingerprints: list[str] = []
        observation_fingerprints: list[str] = []
        model_calls = 0
        fast_path_decisions = 0
        policy_inference_total_ms = 0.0
        validation_total_ms = 0.0
        authority_total_ms = 0.0
        execution_total_ms = 0.0
        state_reduce_total_ms = 0.0
        terminal_critical: list[str] = []
        termination = OrchestrationReasonCode.BUDGET_EXHAUSTED.value
        initial_cache_hits = self.decision_cache.hits if self.decision_cache else 0
        initial_cache_misses = self.decision_cache.misses if self.decision_cache else 0

        while state.step < state.budget.max_steps:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if elapsed_ms >= state.budget.max_elapsed_seconds * 1000:
                termination = OrchestrationReasonCode.BUDGET_EXHAUSTED.value
                break
            allowed = (
                self.action_space_builder.build(
                    goal, state, tool_specs=self.tool_specs
                )
                if self.action_space_builder
                else None
            )
            step_started = time.perf_counter()
            inference_started = time.perf_counter()
            decision_source = "POLICY"
            fast_path_reason = None
            cache_key = None
            action = None
            try:
                if self.fast_path is not None and allowed is not None:
                    fast = self.fast_path.decide(goal, state, allowed)
                    action = fast.action
                    fast_path_reason = fast.reason
                    if action is not None:
                        decision_source = "FAST_PATH"
                        fast_path_decisions += 1
                if action is None and self.decision_cache is not None and allowed is not None:
                    cache_key = self.decision_cache.key(
                        goal,
                        state,
                        getattr(self.policy, "model_version", "unknown"),
                        allowed,
                    )
                    action = self.decision_cache.get(cache_key)
                    if action is not None:
                        decision_source = "CACHE"
                if action is None:
                    if state.model_calls >= state.budget.max_model_calls:
                        termination = OrchestrationReasonCode.BUDGET_EXHAUSTED.value
                        break
                    action = self.policy.decide(goal, state, allowed)
                    if self.decision_cache is not None and cache_key is not None:
                        self.decision_cache.put(cache_key, action)
            except PolicyOutputError:
                termination = "POLICY_OUTPUT_INVALID"
                terminal_critical.append("ACTION_OUTSIDE_AVAILABLE_ACTIONS")
                break
            except Exception:
                termination = "POLICY_FAILURE"
                break
            inference_ms = round((time.perf_counter() - inference_started) * 1000, 3)
            call_stats = getattr(self.policy, "last_call_stats", None)
            step_model_calls = (
                0
                if decision_source in {"FAST_PATH", "CACHE"}
                else int(getattr(call_stats, "model_calls", 1))
            )
            model_calls += step_model_calls
            policy_inference_total_ms += inference_ms
            effects_before = dict(self.sink.effect_counts)
            validation = self.validator.validate(action, goal, state, allowed)
            observation, authority = self.sink.record(action, state, validation)
            sink_timing = getattr(self.sink, "last_timing", None)
            authority_ms = float(getattr(sink_timing, "authority_ms", 0.0))
            execution_ms = float(getattr(sink_timing, "execution_ms", 0.0))
            effects_after = dict(self.sink.effect_counts)
            reduce_started = time.perf_counter()
            updated = self.reducer.reduce(state, action, observation)
            reduce_ms = round((time.perf_counter() - reduce_started) * 1000, 3)
            validation_total_ms += validation.validation_ms
            authority_total_ms += authority_ms
            execution_total_ms += execution_ms
            state_reduce_total_ms += reduce_ms
            tool_delta = max(
                0,
                int(effects_after.get("tools_executed", 0))
                - int(effects_before.get("tools_executed", 0)),
            )
            delegation_delta = max(
                0,
                int(effects_after.get("delegations", 0))
                - int(effects_before.get("delegations", 0)),
            )
            updated = updated.evolve(
                model_calls=state.model_calls + step_model_calls,
                tool_calls=state.tool_calls + tool_delta,
                delegations=state.delegations + delegation_delta,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            stats_dict = (
                call_stats.as_dict()
                if decision_source == "POLICY" and hasattr(call_stats, "as_dict")
                else None
            )
            records.append(
                ShadowDecisionRecord(
                    goal_id=goal.goal_id,
                    step=state.step + 1,
                    world_state_version=state.state_version,
                    model_version=getattr(self.policy, "model_version", "unknown"),
                    next_action=action,
                    validation=validation,
                    authority_shadow_result=authority,
                    observation=observation,
                    capability_snapshot={
                        agent.value: tuple(
                            capability.value for capability in agent_state.capabilities
                        )
                        for agent, agent_state in state.agents.items()
                    },
                    available_agents=tuple(
                        agent.value
                        for agent, agent_state in state.agents.items()
                        if agent_state.availability_known and agent_state.available
                    ),
                    policy_inference_ms=inference_ms,
                    validation_ms=validation.validation_ms,
                    state_reduce_ms=reduce_ms,
                    authority_ms=authority_ms,
                    execution_ms=execution_ms,
                    decision_source=decision_source,
                    fast_path_reason=fast_path_reason,
                    total_step_ms=round(
                        (time.perf_counter() - step_started) * 1000, 3
                    ),
                    policy_stats=stats_dict,
                )
            )
            state = updated
            action_fingerprints.append(action.fingerprint())
            observation_fingerprints.append(observation.fingerprint())

            if (
                observation.goal_completed
                and action.action is not NextActionType.RESPOND
                and self.fast_path is not None
                and observation.verification_status is VerificationStatus.VERIFIED
                and state.step < state.budget.max_steps
            ):
                # Let the deterministic verified-completion fast path emit the
                # final RESPOND without paying for another model call.
                continue
            if observation.goal_completed or (
                action.action is NextActionType.RESPOND
                and
                action.reason_code is OrchestrationReasonCode.GOAL_COMPLETED
                and observation.status is ObservationStatus.SUCCESS
            ):
                termination = OrchestrationReasonCode.GOAL_COMPLETED.value
                break
            if observation.terminal_block:
                termination = OrchestrationReasonCode.SAFETY_TERMINAL.value
                break
            if action.action is NextActionType.STOP:
                termination = (
                    action.reason_code.value
                    if action.reason_code is not OrchestrationReasonCode.GOAL_COMPLETED
                    else OrchestrationReasonCode.INVALID_ACTION.value
                )
                break
            if action.action is NextActionType.ASK_USER:
                termination = OrchestrationReasonCode.USER_INPUT_REQUIRED.value
                break
            if action.action is NextActionType.WAIT and observation.status is ObservationStatus.PENDING:
                termination = OrchestrationReasonCode.DEPENDENCY_IN_PROGRESS.value
                break
            if state.failure_count >= state.budget.max_failures:
                termination = "MAX_FAILURES"
                break
            if self._count_tail(action_fingerprints) > state.budget.max_repeated_action:
                termination = OrchestrationReasonCode.REPEATED_ACTION.value
                break
            if self._count_tail(observation_fingerprints) > state.budget.max_same_observation:
                termination = OrchestrationReasonCode.NO_PROGRESS.value
                break
            if self._alternating_loop(action_fingerprints, observation_fingerprints):
                termination = OrchestrationReasonCode.LOOP_DETECTED.value
                break

        critical = tuple(
            dict.fromkeys(
                (
                    *terminal_critical,
                    *(
                        violation
                        for record in records
                        for violation in record.validation.critical_violations
                    ),
                )
            )
        )
        return ShadowLoopResult(
            goal_id=goal.goal_id,
            initial_state_version=initial_state.state_version,
            final_state=state,
            records=tuple(records),
            termination_reason=termination,
            shadow_total_ms=round((time.perf_counter() - started) * 1000, 3),
            model_calls=model_calls,
            critical_shadow_violations=critical,
            effect_counts=self.sink.effect_counts,
            mode=str(getattr(self.sink, "mode", "SHADOW")).upper(),
            fast_path_decisions=fast_path_decisions,
            decision_cache_hits=(
                self.decision_cache.hits - initial_cache_hits
                if self.decision_cache
                else 0
            ),
            decision_cache_misses=(
                self.decision_cache.misses - initial_cache_misses
                if self.decision_cache
                else 0
            ),
            telemetry={
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                "policy_inference_ms": round(policy_inference_total_ms, 3),
                "validation_ms": round(validation_total_ms, 3),
                "authority_ms": round(authority_total_ms, 3),
                "execution_ms": round(execution_total_ms, 3),
                "state_reduce_ms": round(state_reduce_total_ms, 3),
                "model_calls": model_calls,
            },
        )

    @staticmethod
    def _count_tail(values: list[str]) -> int:
        if not values:
            return 0
        current = values[-1]
        count = 0
        for value in reversed(values):
            if value != current:
                break
            count += 1
        return count

    @staticmethod
    def _alternating_loop(actions: list[str], observations: list[str]) -> bool:
        return (
            len(actions) >= 4
            and actions[-4] == actions[-2]
            and actions[-3] == actions[-1]
            and actions[-4] != actions[-3]
            and len(set(observations[-4:])) <= 2
        )
