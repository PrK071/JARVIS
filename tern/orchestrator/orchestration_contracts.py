"""Versioned, side-effect-free contracts for Phase 1.75 orchestration.

The types in this module deliberately contain data only.  In particular, they
cannot carry a tool registry, executor, session, job store or filesystem handle.
That boundary lets the shadow loop describe what *should* happen without gaining
the authority or ability to make it happen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .autonomy_foundation import Agent, Capability
from .execution_gate import ExecutionMode
from .task_requirement_grounding import RequirementValue


ORCHESTRATION_SCHEMA_VERSION = "1"


def _bounded(value: str | None, limit: int, field_name: str) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} must not be blank")
    if len(clean) > limit:
        raise ValueError(f"{field_name} exceeds {limit} characters")
    return clean


class SemanticAction(str, Enum):
    UNKNOWN = "UNKNOWN"
    ANALYZE = "ANALYZE"
    REPAIR = "REPAIR"
    REWRITE = "REWRITE"
    CREATE = "CREATE"
    IMPROVE = "IMPROVE"
    EXPLAIN = "EXPLAIN"
    CONTINUE = "CONTINUE"
    DELETE_OBJECT = "DELETE_OBJECT"
    REMOVE_COMPONENT = "REMOVE_COMPONENT"
    CLEAR_CONTENT = "CLEAR_CONTENT"
    RESET_STATE = "RESET_STATE"


class AgentSource(str, Enum):
    EXPLICIT_USER = "explicit_user"


@dataclass(frozen=True)
class UserGoal:
    """Semantic understanding output; never an executor-selection result."""

    goal_id: str
    summary: str
    desired_outcome: str
    semantic_action: SemanticAction = SemanticAction.UNKNOWN
    execution_requested: bool = False
    explicit_agent: Agent | None = None
    agent_source: AgentSource | None = None
    permitted_agents: tuple[Agent, ...] = ()
    forbidden_agents: tuple[Agent, ...] = ()
    constraints: tuple[str, ...] = ()
    mutation_required: RequirementValue = RequirementValue.UNKNOWN
    mutation_forbidden: bool = False
    references: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    schema_version: str = field(default=ORCHESTRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _bounded(self.goal_id, 160, "goal_id"))
        object.__setattr__(self, "summary", _bounded(self.summary, 1000, "summary"))
        object.__setattr__(
            self,
            "desired_outcome",
            _bounded(self.desired_outcome, 1000, "desired_outcome"),
        )
        if (self.explicit_agent is None) != (self.agent_source is None):
            raise ValueError("explicit_agent and agent_source must be set together")
        if self.explicit_agent in self.forbidden_agents:
            raise ValueError("explicit agent cannot also be forbidden")
        if set(self.permitted_agents) & set(self.forbidden_agents):
            raise ValueError("an agent cannot be both permitted and forbidden")
        if self.mutation_forbidden and self.mutation_required is RequirementValue.TRUE:
            raise ValueError("mutation cannot be both required and forbidden")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "goal": {
                "summary": self.summary,
                "desired_outcome": self.desired_outcome,
            },
            "semantic_action": self.semantic_action.value,
            "execution": {"requested": self.execution_requested},
            "executor": {
                "explicit_agent": (
                    self.explicit_agent.value if self.explicit_agent else None
                ),
                "agent_source": self.agent_source.value if self.agent_source else None,
                "permitted_agents": [item.value for item in self.permitted_agents],
                "forbidden_agents": [item.value for item in self.forbidden_agents],
            },
            "constraints": list(self.constraints),
            "mutation": {
                "required": self.mutation_required.value,
                "forbidden": self.mutation_forbidden,
            },
            "references": list(self.references),
            "evidence": list(self.evidence),
        }


class NextActionType(str, Enum):
    INSPECT = "INSPECT"
    DELEGATE = "DELEGATE"
    EXECUTE = "EXECUTE"
    ASK_USER = "ASK_USER"
    WAIT = "WAIT"
    RESPOND = "RESPOND"
    STOP = "STOP"


class OrchestrationReasonCode(str, Enum):
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    REPOSITORY_INSPECTION_REQUIRED = "REPOSITORY_INSPECTION_REQUIRED"
    EXPERT_ANALYSIS_REQUIRED = "EXPERT_ANALYSIS_REQUIRED"
    CODE_MUTATION_REQUIRED = "CODE_MUTATION_REQUIRED"
    USER_INPUT_REQUIRED = "USER_INPUT_REQUIRED"
    GOAL_COMPLETED = "GOAL_COMPLETED"
    GOAL_IMPOSSIBLE = "GOAL_IMPOSSIBLE"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    AGENT_INELIGIBLE = "AGENT_INELIGIBLE"
    AUTHORITY_BLOCKED = "AUTHORITY_BLOCKED"
    NO_PROGRESS = "NO_PROGRESS"
    LOOP_DETECTED = "LOOP_DETECTED"
    REPEATED_ACTION = "REPEATED_ACTION"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    SUFFICIENT_INFORMATION = "SUFFICIENT_INFORMATION"
    EXPLICIT_AGENT_REQUIRED = "EXPLICIT_AGENT_REQUIRED"
    DEPENDENCY_IN_PROGRESS = "DEPENDENCY_IN_PROGRESS"
    SAFETY_TERMINAL = "SAFETY_TERMINAL"
    INVALID_ACTION = "INVALID_ACTION"


class ShadowLearningSignal(str, Enum):
    """Evaluation labels only; Phase 1.75 performs no training or promotion."""

    ORCHESTRATION_AGENT_DISAGREEMENT = "ORCHESTRATION_AGENT_DISAGREEMENT"
    UNNECESSARY_DELEGATION = "UNNECESSARY_DELEGATION"
    PREMATURE_EXECUTION = "PREMATURE_EXECUTION"
    PREMATURE_MUTATION = "PREMATURE_MUTATION"
    PREMATURE_RESPONSE = "PREMATURE_RESPONSE"
    ASK_USER_TOO_EARLY = "ASK_USER_TOO_EARLY"
    WRONG_AGENT = "WRONG_AGENT"
    LOOP_DETECTED = "LOOP_DETECTED"
    NO_PROGRESS = "NO_PROGRESS"
    WRONG_CAPABILITY = "WRONG_CAPABILITY"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    AUTHORITY_BLOCK_RECOVERED = "AUTHORITY_BLOCK_RECOVERED"
    AUTHORITY_BLOCK_NOT_RECOVERED = "AUTHORITY_BLOCK_NOT_RECOVERED"
    EXPLICIT_AGENT_PRESERVATION = "EXPLICIT_AGENT_PRESERVATION"
    GOOD_MULTI_STEP_PLAN = "GOOD_MULTI_STEP_PLAN"
    GOOD_MULTI_STEP_TRAJECTORY = "GOOD_MULTI_STEP_TRAJECTORY"
    BAD_MULTI_STEP_TRAJECTORY = "BAD_MULTI_STEP_TRAJECTORY"


ORCHESTRATION_ISSUE_SEVERITY: Mapping[str, str] = MappingProxyType(
    {
        "SUBOPTIMAL_INSPECTION": "LOW",
        "EXTRA_INSPECTION": "LOW",
        "UNNECESSARY_DELEGATION": "MEDIUM",
        "ASK_USER_TOO_EARLY": "MEDIUM",
        "PREMATURE_RESPONSE": "HIGH",
        "WRONG_AGENT": "HIGH",
        "WRONG_CAPABILITY": "HIGH",
        "PREMATURE_MUTATION": "CRITICAL",
        "VIOLATED_READ_ONLY": "CRITICAL",
        "USED_FORBIDDEN_AGENT": "CRITICAL",
        "UNAUTHORIZED_EXECUTION": "CRITICAL",
    }
)


@dataclass(frozen=True)
class NextAction:
    """A closed proposal type.  It is intentionally not executable."""

    action_id: str
    action: NextActionType
    objective: str
    reason_code: OrchestrationReasonCode
    target_agent: Agent | None = None
    target: str | None = None
    tool_name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    execution_mode: ExecutionMode | None = None
    required_capabilities: tuple[Capability, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    expected_observation: str | None = None
    confidence: float | None = None
    short_horizon_hint: str | None = None
    schema_version: str = field(default=ORCHESTRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _bounded(self.action_id, 160, "action_id"))
        object.__setattr__(self, "objective", _bounded(self.objective, 1200, "objective"))
        for name, value, limit in (
            ("target", self.target, 600),
            ("tool_name", self.tool_name, 160),
            ("expected_observation", self.expected_observation, 1000),
            ("short_horizon_hint", self.short_horizon_hint, 600),
        ):
            if value is not None:
                object.__setattr__(self, name, _bounded(value, limit, name))
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be an object")
        arguments = dict(self.arguments)
        if len(arguments) > 24:
            raise ValueError("arguments exceeds 24 fields")
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("arguments must contain JSON values") from exc
        if len(encoded.encode("utf-8")) > 32_768:
            raise ValueError("arguments exceeds 32768 bytes")
        object.__setattr__(self, "arguments", MappingProxyType(arguments))

    def fingerprint(self) -> str:
        payload = self.as_dict().copy()
        payload.pop("action_id", None)
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "action": self.action.value,
            "target_agent": self.target_agent.value if self.target_agent else None,
            "target": self.target,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "objective": self.objective,
            "execution_mode": (
                self.execution_mode.value if self.execution_mode else None
            ),
            "required_capabilities": [
                item.value for item in self.required_capabilities
            ],
            "reason_code": self.reason_code.value,
            "evidence_refs": list(self.evidence_refs),
            "expected_observation": self.expected_observation,
            "confidence": self.confidence,
            "short_horizon_hint": self.short_horizon_hint,
        }


class ObservationSource(str, Enum):
    REPLAY = "replay"
    FIXTURE = "fixture"
    SYNTHETIC = "synthetic"
    EXISTING_LOG = "existing_log"
    SHADOW_VALIDATOR = "shadow_validator"
    EXECUTION_AUTHORITY = "execution_authority"
    LIVE_TOOL = "live_tool"
    LIVE_AGENT = "live_agent"
    FAST_PATH = "fast_path"


class ObservationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"
    PROPOSED = "PROPOSED"
    NO_PROGRESS = "NO_PROGRESS"


class VerificationStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: ObservationSource
    action_id: str
    status: ObservationStatus
    summary: str
    facts: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    state_changes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    authority_outcome: str | None = None
    tool_name: str | None = None
    agent: Agent | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    goal_completed: bool = False
    terminal_block: bool = False
    schema_version: str = field(default=ORCHESTRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _bounded(self.observation_id, 160, "observation_id")
        )
        object.__setattr__(self, "action_id", _bounded(self.action_id, 160, "action_id"))
        object.__setattr__(self, "summary", _bounded(self.summary, 1600, "summary"))

    def fingerprint(self) -> str:
        payload = {
            "status": self.status.value,
            "summary": self.summary,
            "facts": self.facts,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "authority_outcome": self.authority_outcome,
            "verification_status": self.verification_status.value,
            "goal_completed": self.goal_completed,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "source": self.source.value,
            "action_id": self.action_id,
            "status": self.status.value,
            "summary": self.summary,
            "facts": list(self.facts),
            "artifacts": list(self.artifacts),
            "state_changes": list(self.state_changes),
            "errors": list(self.errors),
            "authority_outcome": self.authority_outcome,
            "tool_name": self.tool_name,
            "agent": self.agent.value if self.agent else None,
            "verification_status": self.verification_status.value,
            "goal_completed": self.goal_completed,
            "terminal_block": self.terminal_block,
        }


@dataclass(frozen=True)
class ProjectState:
    project_id: str | None = None
    path: str | None = None
    project_type: str | None = None
    branch: str | None = None
    working_tree: str = "unknown"
    test_state: str = "unknown"
    context_refs: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.project_id,
            "path": self.path,
            "type": self.project_type,
            "branch": self.branch,
            "working_tree": self.working_tree,
            "tests": self.test_state,
            "context_refs": list(self.context_refs),
        }


@dataclass(frozen=True)
class AgentState:
    agent: Agent
    availability_known: bool
    available: bool
    eligible: bool | None
    capabilities: tuple[Capability, ...] = ()
    execution_modes: tuple[ExecutionMode, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available if self.availability_known else None,
            "availability_known": self.availability_known,
            "eligible": self.eligible,
            "capabilities": [item.value for item in self.capabilities],
            "execution_modes": [item.value for item in self.execution_modes],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class JobState:
    job_id: str
    agent: Agent
    status: str
    objective_ref: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "agent": self.agent.value,
            "status": self.status,
            "objective_ref": self.objective_ref,
        }


@dataclass(frozen=True)
class OrchestrationBudget:
    max_steps: int = 8
    max_observations: int = 16
    max_action_history: int = 16
    max_context_items: int = 64
    max_repeated_action: int = 2
    max_same_observation: int = 2
    max_failures: int = 3
    max_model_calls: int = 8
    max_tool_calls: int = 8
    max_delegations: int = 4
    max_elapsed_seconds: int = 900

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def as_dict(
        self,
        *,
        step: int,
        model_calls: int = 0,
        tool_calls: int = 0,
        delegations: int = 0,
        elapsed_ms: float = 0.0,
    ) -> dict[str, int | float]:
        return {
            "step": step,
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "delegations": delegations,
            "elapsed_ms": round(elapsed_ms, 3),
            **self.__dict__,
        }


@dataclass(frozen=True)
class WorldState:
    goal_id: str
    state_version: int
    project: ProjectState
    agents: Mapping[Agent, AgentState]
    tools: tuple[str, ...]
    observations: tuple[Observation, ...]
    jobs: tuple[JobState, ...]
    previous_actions: tuple[NextAction, ...]
    current_facts: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    authority_facts: tuple[str, ...]
    budget: OrchestrationBudget
    step: int = 0
    failure_count: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    delegations: int = 0
    elapsed_ms: float = 0.0
    schema_version: str = field(default=ORCHESTRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.state_version,
                self.step,
                self.failure_count,
                self.model_calls,
                self.tool_calls,
                self.delegations,
                self.elapsed_ms,
            )
        ):
            raise ValueError("state counters cannot be negative")
        for key, state in self.agents.items():
            if key is not state.agent:
                raise ValueError("agent map key must match AgentState.agent")
        object.__setattr__(self, "agents", MappingProxyType(dict(self.agents)))

    def evolve(self, **changes: Any) -> "WorldState":
        return replace(self, **changes)

    def semantic_hash(self) -> str:
        """Hash only facts that can change the next orchestration decision."""

        payload = {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "project": self.project.as_dict(),
            "agents": {
                agent.value: state.as_dict()
                for agent, state in sorted(
                    self.agents.items(), key=lambda item: item[0].value
                )
            },
            "tools": self.tools,
            "observations": [item.fingerprint() for item in self.observations],
            "jobs": [item.as_dict() for item in self.jobs],
            "facts": self.current_facts,
            "unresolved": self.unresolved_questions,
            "authority": self.authority_facts,
            "failures": self.failure_count,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "state_version": self.state_version,
            "project": self.project.as_dict(),
            "agents": {
                agent.value: state.as_dict()
                for agent, state in sorted(
                    self.agents.items(), key=lambda item: item[0].value
                )
            },
            "tools": list(self.tools),
            "observations": [item.as_dict() for item in self.observations],
            "jobs": [item.as_dict() for item in self.jobs],
            "history": {
                "previous_actions": [
                    item.as_dict() for item in self.previous_actions
                ]
            },
            "current_facts": list(self.current_facts),
            "unresolved_questions": list(self.unresolved_questions),
            "authority_facts": list(self.authority_facts),
            "budget": self.budget.as_dict(
                step=self.step,
                model_calls=self.model_calls,
                tool_calls=self.tool_calls,
                delegations=self.delegations,
                elapsed_ms=self.elapsed_ms,
            ),
            "failure_count": self.failure_count,
        }
