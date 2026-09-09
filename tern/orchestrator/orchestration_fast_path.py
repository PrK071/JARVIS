"""Action-space pruning, semantic decision caching and trivial fast paths."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .autonomy_foundation import Agent
from .execution_authority import BoundedLiveRiskMatrix, RiskDisposition
from .execution_gate import ExecutionMode
from .orchestration_contracts import (
    NextAction,
    NextActionType,
    ObservationStatus,
    OrchestrationReasonCode,
    UserGoal,
    VerificationStatus,
    WorldState,
)


@dataclass(frozen=True)
class AllowedActionSpace:
    action_types: tuple[NextActionType, ...]
    candidate_agents: tuple[Agent, ...]
    agent_modes: Mapping[Agent, tuple[ExecutionMode, ...]]
    tools: tuple[str, ...]
    tool_schemas: Mapping[str, Mapping[str, Any]]
    active_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_modes", MappingProxyType(dict(self.agent_modes)))
        object.__setattr__(
            self,
            "tool_schemas",
            MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in self.tool_schemas.items()}
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_types": [item.value for item in self.action_types],
            "candidate_agents": [item.value for item in self.candidate_agents],
            "agent_modes": {
                agent.value: [mode.value for mode in modes]
                for agent, modes in self.agent_modes.items()
            },
            "tools": [
                {
                    "name": name,
                    "arguments_schema": dict(self.tool_schemas.get(name, {})),
                }
                for name in self.tools
            ],
            "active_constraints": list(self.active_constraints),
        }

    def as_prompt_dict(self) -> dict[str, Any]:
        """Compact tool contracts for inference; full schemas remain in-memory."""

        tools: list[str] = []
        for name in self.tools:
            schema = self.tool_schemas.get(name, {})
            properties = schema.get("properties") if isinstance(schema, Mapping) else {}
            required = schema.get("required") if isinstance(schema, Mapping) else ()
            required_names = set(required or ())
            argument_names = [
                key if key in required_names else f"{key}?"
                for key in (properties if isinstance(properties, Mapping) else ())
            ]
            effect_mode = (
                "MUTATION"
                if BoundedLiveRiskMatrix.tool_policy(name).mutation
                else "READ_ONLY"
            )
            tools.append(
                f"{name}[{effect_mode}]({','.join(argument_names)})"
            )
        return {
            "action_types": [item.value for item in self.action_types],
            "candidate_agents": [item.value for item in self.candidate_agents],
            "agent_modes": {
                agent.value: [mode.value for mode in modes]
                for agent, modes in self.agent_modes.items()
            },
            "tools": tools,
            "active_constraints": list(self.active_constraints),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


class ActionSpaceBuilder:
    """Remove impossible actions without choosing strategy."""

    def build(
        self,
        goal: UserGoal,
        state: WorldState,
        *,
        tool_specs: Iterable[Mapping[str, Any]] = (),
    ) -> AllowedActionSpace:
        candidates: list[Agent] = []
        modes: dict[Agent, tuple[ExecutionMode, ...]] = {}
        delegation_budget_available = (
            state.delegations < state.budget.max_delegations
        )
        for agent, agent_state in state.agents.items():
            if not delegation_budget_available:
                continue
            if agent not in {Agent.CODEX, Agent.DEEPSEEK}:
                continue
            if (
                not agent_state.availability_known
                or not agent_state.available
                or agent_state.eligible is False
                or agent in goal.forbidden_agents
            ):
                continue
            if goal.explicit_agent is not None and agent is not goal.explicit_agent:
                continue
            allowed_modes = tuple(
                mode
                for mode in agent_state.execution_modes
                if not (
                    mode is ExecutionMode.MUTATION and goal.mutation_forbidden
                )
            )
            if not allowed_modes:
                continue
            candidates.append(agent)
            modes[agent] = allowed_modes

        specs_by_name: dict[str, Mapping[str, Any]] = {}
        for item in tool_specs:
            function = item.get("function") if isinstance(item, Mapping) else None
            if not isinstance(function, Mapping):
                continue
            name = str(function.get("name") or "")
            parameters = function.get("parameters")
            if name and isinstance(parameters, Mapping):
                specs_by_name[name] = parameters
        tools: list[str] = []
        tool_budget_available = state.tool_calls < state.budget.max_tool_calls
        for name in state.tools:
            if not tool_budget_available:
                continue
            if name.startswith("delegate_to_"):
                continue
            if name == "get_codex_job_status" and not state.jobs:
                continue
            if (
                name == "review_codex_session"
                and "FOCUSED_CODEX_SESSION" not in state.authority_facts
            ):
                continue
            if (
                name == "review_deepseek_session"
                and "FOCUSED_DEEPSEEK_SESSION" not in state.authority_facts
            ):
                continue
            policy = BoundedLiveRiskMatrix.tool_policy(name)
            if policy.disposition is RiskDisposition.RESTRICTED:
                continue
            if policy.mutation and goal.mutation_forbidden:
                continue
            tools.append(name)

        action_types = [
            NextActionType.ASK_USER,
            NextActionType.RESPOND,
            NextActionType.STOP,
        ]
        if tools:
            action_types[:0] = [NextActionType.INSPECT, NextActionType.EXECUTE]
        if candidates:
            action_types.insert(1, NextActionType.DELEGATE)
        if any(
            job.status.casefold()
            in {"queued", "starting", "running", "steering", "reconnecting"}
            for job in state.jobs
        ):
            action_types.insert(0, NextActionType.WAIT)

        constraints = list(goal.constraints)
        if goal.mutation_forbidden:
            constraints.append("READ_ONLY")
        if goal.explicit_agent:
            constraints.append(f"EXPLICIT_AGENT:{goal.explicit_agent.value}")
        constraints.extend(f"FORBIDDEN_AGENT:{item.value}" for item in goal.forbidden_agents)
        return AllowedActionSpace(
            action_types=tuple(dict.fromkeys(action_types)),
            candidate_agents=tuple(candidates),
            agent_modes=modes,
            tools=tuple(tools),
            tool_schemas={name: specs_by_name.get(name, {}) for name in tools},
            active_constraints=tuple(dict.fromkeys(constraints)),
        )


@dataclass(frozen=True)
class FastPathDecision:
    action: NextAction | None
    reason: str | None


class OrchestrationFastPath:
    """Only decisions mechanically proven by current state belong here."""

    def decide(
        self,
        goal: UserGoal,
        state: WorldState,
        allowed: AllowedActionSpace,
    ) -> FastPathDecision:
        action_id = f"fast-{goal.goal_id}-{state.step + 1}"
        if state.step >= state.budget.max_steps:
            return FastPathDecision(
                NextAction(
                    action_id,
                    NextActionType.STOP,
                    "Stop because the configured step budget is exhausted.",
                    OrchestrationReasonCode.BUDGET_EXHAUSTED,
                ),
                "MAX_STEPS",
            )
        last = state.observations[-1] if state.observations else None
        if last and last.authority_outcome == "USER_CONFIRMATION_REQUIRED":
            return FastPathDecision(
                NextAction(
                    action_id,
                    NextActionType.ASK_USER,
                    "Confirme a ação protegida necessária para continuar.",
                    OrchestrationReasonCode.USER_INPUT_REQUIRED,
                    evidence_refs=(last.observation_id,),
                ),
                "AUTHORITY_CONFIRMATION",
            )
        if last and last.terminal_block:
            return FastPathDecision(
                NextAction(
                    action_id,
                    NextActionType.STOP,
                    "Stop after a terminal safety block.",
                    OrchestrationReasonCode.SAFETY_TERMINAL,
                    evidence_refs=(last.observation_id,),
                ),
                "TERMINAL_SAFETY_BLOCK",
            )
        if last and last.goal_completed and last.verification_status is VerificationStatus.VERIFIED:
            return FastPathDecision(
                NextAction(
                    action_id,
                    NextActionType.RESPOND,
                    "O objetivo foi concluído e verificado.",
                    OrchestrationReasonCode.GOAL_COMPLETED,
                    evidence_refs=(last.observation_id,),
                ),
                "VERIFIED_GOAL_COMPLETION",
            )
        active = next(
            (
                job
                for job in state.jobs
                if job.status.casefold()
                in {"queued", "starting", "running", "steering", "reconnecting"}
            ),
            None,
        )
        if active and (last is None or last.status is ObservationStatus.PENDING):
            return FastPathDecision(
                NextAction(
                    action_id,
                    NextActionType.WAIT,
                    "Observe the active job without starting duplicate work.",
                    OrchestrationReasonCode.DEPENDENCY_IN_PROGRESS,
                    target=active.job_id,
                    evidence_refs=(f"job:{active.job_id}",),
                ),
                "ACTIVE_JOB_NO_NEW_INFORMATION",
            )
        return FastPathDecision(None, None)


class OrchestrationDecisionCache:
    def __init__(self, max_entries: int = 128):
        self.max_entries = max(1, max_entries)
        self._values: OrderedDict[str, NextAction] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        goal: UserGoal,
        state: WorldState,
        policy_version: str,
        allowed: AllowedActionSpace,
    ) -> str:
        payload = {
            "goal_schema": goal.schema_version,
            "goal_id": goal.goal_id,
            "state": state.semantic_hash(),
            "policy": policy_version,
            "allowed": allowed.fingerprint(),
            "constraints": goal.constraints,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def get(self, key: str) -> NextAction | None:
        value = self._values.get(key)
        if value is None:
            self.misses += 1
            return None
        self._values.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, action: NextAction) -> None:
        self._values[key] = action
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
