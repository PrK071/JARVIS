"""Controlled authority transfer, phase 1: explicit user agent only.

Authority is granted to the ExecutionGate exactly when the user named the agent:

    requested_agent is not None and requested_agent_source == EXPLICIT_USER

Everything else stays shadow and keeps the legacy live behaviour. Inside the
authoritative scope there is no fallback: if the gate blocks, nothing executes,
and no other agent is ever substituted for the requested one.

This module decides. It does not execute: dispatch keeps using the existing
tools.execute, pending actions, confirmation, PathPolicy, session resolver and
job lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from .autonomy_foundation import Agent
from .execution_gate import (
    ExecutionAuthority,
    ExecutionBlockReason,
    ExecutionMode,
    ShadowExecutionDecision,
)
from .orchestration_contracts import (
    NextAction,
    NextActionType,
    ObservationStatus,
    UserGoal,
    WorldState,
)
from .task_requirement_grounding import RequirementValue


EXPLICIT_USER_SOURCE = "explicit_user"


class ExecutionAuthorityMode(str, Enum):
    """Single knob. SHADOW is the immediate rollback value."""

    SHADOW = "shadow"
    EXPLICIT_USER = "explicit_user"

    @classmethod
    def parse(cls, value: str | None) -> "ExecutionAuthorityMode":
        normalized = (value or "shadow").strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(
            "EXECUTION_GATE_AUTHORITY deve ser shadow ou explicit_user"
        )


class AuthorityScope(str, Enum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    EXPLICIT_USER = "EXPLICIT_USER"


class AuthorityBlockReason(str, Enum):
    AVAILABILITY_CHANGED_BEFORE_DISPATCH = "AVAILABILITY_CHANGED_BEFORE_DISPATCH"
    READ_ONLY_ENFORCEMENT_UNAVAILABLE = "READ_ONLY_ENFORCEMENT_UNAVAILABLE"
    AGENT_OUTSIDE_AUTHORITY_SCOPE = "AGENT_OUTSIDE_AUTHORITY_SCOPE"


class OrchestrationMode(str, Enum):
    SHADOW = "shadow"
    BOUNDED_LIVE = "bounded_live"

    @classmethod
    def parse(cls, value: str | None) -> "OrchestrationMode":
        normalized = (value or cls.SHADOW.value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                "ORCHESTRATION_MODE deve ser shadow ou bounded_live"
            ) from exc


class EffectRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskDisposition(str, Enum):
    AUTO = "AUTO"
    AUTHORITY = "AUTHORITY"
    CONFIRM = "CONFIRM"
    RESTRICTED = "RESTRICTED"


@dataclass(frozen=True)
class LiveActionPolicy:
    risk: EffectRisk
    disposition: RiskDisposition
    mutation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk.value,
            "disposition": self.disposition.value,
            "mutation": self.mutation,
        }


class BoundedLiveRiskMatrix:
    """One auditable action matrix; strategy remains outside this class."""

    TOOL_POLICIES: Mapping[str, LiveActionPolicy] = {
        "list_available_agents": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "get_hardware_telemetry": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "list_installed_applications": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "resolve_project": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "discover_project": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "find_project_files": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "filesystem_list": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "filesystem_read_text": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "review_codex_session": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "review_deepseek_session": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "get_codex_job_status": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "web_search": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "web_open": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "web_extract": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "get_project_git_state": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "run_project_tests": LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO),
        "filesystem_write_text": LiveActionPolicy(
            EffectRisk.MEDIUM, RiskDisposition.AUTHORITY, mutation=True
        ),
        "filesystem_delete": LiveActionPolicy(
            EffectRisk.HIGH, RiskDisposition.CONFIRM, mutation=True
        ),
        "open_application": LiveActionPolicy(EffectRisk.MEDIUM, RiskDisposition.CONFIRM),
        "schedule_application": LiveActionPolicy(EffectRisk.HIGH, RiskDisposition.CONFIRM),
        "web_open_browser": LiveActionPolicy(EffectRisk.MEDIUM, RiskDisposition.CONFIRM),
        "cancel_codex_job": LiveActionPolicy(EffectRisk.MEDIUM, RiskDisposition.CONFIRM),
        "steer_codex_job": LiveActionPolicy(EffectRisk.MEDIUM, RiskDisposition.CONFIRM),
    }

    @classmethod
    def tool_policy(cls, tool_name: str | None) -> LiveActionPolicy:
        if not tool_name:
            return LiveActionPolicy(EffectRisk.HIGH, RiskDisposition.RESTRICTED)
        return cls.TOOL_POLICIES.get(
            tool_name,
            LiveActionPolicy(EffectRisk.HIGH, RiskDisposition.RESTRICTED),
        )

    @staticmethod
    def delegation_policy(
        agent: Agent | None, execution_mode: ExecutionMode | None
    ) -> LiveActionPolicy:
        if agent is Agent.DEEPSEEK and execution_mode is ExecutionMode.READ_ONLY:
            return LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO)
        if agent is Agent.CODEX and execution_mode is ExecutionMode.READ_ONLY:
            return LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO)
        if agent is Agent.CODEX and execution_mode is ExecutionMode.MUTATION:
            return LiveActionPolicy(
                EffectRisk.MEDIUM, RiskDisposition.AUTHORITY, mutation=True
            )
        return LiveActionPolicy(EffectRisk.HIGH, RiskDisposition.RESTRICTED)


class BoundedAuthorityBlockReason(str, Enum):
    MODE_IS_SHADOW = "MODE_IS_SHADOW"
    STRUCTURAL_VALIDATION_FAILED = "STRUCTURAL_VALIDATION_FAILED"
    ACTION_NOT_IN_LIVE_MATRIX = "ACTION_NOT_IN_LIVE_MATRIX"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    PATH_POLICY_BLOCKED = "PATH_POLICY_BLOCKED"
    DUPLICATE_ACTION = "DUPLICATE_ACTION"
    EXECUTION_NOT_REQUESTED = "EXECUTION_NOT_REQUESTED"
    MUTATION_NOT_AUTHORIZED = "MUTATION_NOT_AUTHORIZED"
    READ_ONLY_CONSTRAINT = "READ_ONLY_CONSTRAINT"
    FORBIDDEN_AGENT = "FORBIDDEN_AGENT"
    EXPLICIT_AGENT_MISMATCH = "EXPLICIT_AGENT_MISMATCH"
    EXPLICIT_AGENT_NOT_USED = "EXPLICIT_AGENT_NOT_USED"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    AGENT_INELIGIBLE = "AGENT_INELIGIBLE"
    EXECUTION_MODE_UNSUPPORTED = "EXECUTION_MODE_UNSUPPORTED"
    USER_CONFIRMATION_REQUIRED = "USER_CONFIRMATION_REQUIRED"
    PREMATURE_RESPONSE = "PREMATURE_RESPONSE"
    GOAL_NOT_VERIFIED = "GOAL_NOT_VERIFIED"


@dataclass(frozen=True)
class BoundedAuthorityFacts:
    structural_valid: bool
    tool_available: bool = True
    path_allowed: bool = True
    duplicate_action: bool = False
    goal_evidence_sufficient: bool = False
    goal_verified: bool = False


@dataclass(frozen=True)
class BoundedAuthorityDecision:
    action_id: str
    mode: OrchestrationMode
    allowed: bool
    policy: LiveActionPolicy
    block_reason: BoundedAuthorityBlockReason | None
    confirmation_required: bool
    requires_availability_recheck: bool
    mutation_authorized: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "mode": self.mode.value,
            "allowed": self.allowed,
            "policy": self.policy.as_dict(),
            "block_reason": self.block_reason.value if self.block_reason else None,
            "confirmation_required": self.confirmation_required,
            "requires_availability_recheck": self.requires_availability_recheck,
            "mutation_authorized": self.mutation_authorized,
        }


class BoundedLiveExecutionAuthority:
    """Deterministic permission boundary for orchestration proposals."""

    def __init__(
        self,
        mode: OrchestrationMode,
        *,
        risk_matrix: type[BoundedLiveRiskMatrix] = BoundedLiveRiskMatrix,
    ):
        self.mode = mode
        self.risk_matrix = risk_matrix

    def evaluate(
        self,
        action: NextAction,
        goal: UserGoal,
        state: WorldState,
        facts: BoundedAuthorityFacts,
    ) -> BoundedAuthorityDecision:
        policy = (
            self.risk_matrix.delegation_policy(
                action.target_agent, action.execution_mode
            )
            if action.action is NextActionType.DELEGATE
            else self.risk_matrix.tool_policy(action.tool_name)
            if action.action in {NextActionType.INSPECT, NextActionType.EXECUTE}
            else LiveActionPolicy(EffectRisk.LOW, RiskDisposition.AUTO)
        )
        block: BoundedAuthorityBlockReason | None = None
        if self.mode is not OrchestrationMode.BOUNDED_LIVE:
            block = BoundedAuthorityBlockReason.MODE_IS_SHADOW
        elif not facts.structural_valid:
            block = BoundedAuthorityBlockReason.STRUCTURAL_VALIDATION_FAILED
        elif facts.duplicate_action:
            block = BoundedAuthorityBlockReason.DUPLICATE_ACTION
        elif policy.disposition is RiskDisposition.RESTRICTED:
            block = BoundedAuthorityBlockReason.ACTION_NOT_IN_LIVE_MATRIX
        elif action.action in {NextActionType.INSPECT, NextActionType.EXECUTE}:
            if not facts.tool_available:
                block = BoundedAuthorityBlockReason.TOOL_UNAVAILABLE
            elif not facts.path_allowed:
                block = BoundedAuthorityBlockReason.PATH_POLICY_BLOCKED
        if block is None and action.action is NextActionType.DELEGATE:
            agent = action.target_agent
            agent_state = state.agents.get(agent) if agent else None
            if agent in goal.forbidden_agents:
                block = BoundedAuthorityBlockReason.FORBIDDEN_AGENT
            elif goal.explicit_agent and agent is not goal.explicit_agent:
                block = BoundedAuthorityBlockReason.EXPLICIT_AGENT_MISMATCH
            elif agent_state is None or not agent_state.availability_known or not agent_state.available:
                block = BoundedAuthorityBlockReason.AGENT_UNAVAILABLE
            elif agent_state.eligible is False:
                block = BoundedAuthorityBlockReason.AGENT_INELIGIBLE
            elif action.execution_mode not in agent_state.execution_modes:
                block = BoundedAuthorityBlockReason.EXECUTION_MODE_UNSUPPORTED
        if block is None and policy.mutation:
            if goal.mutation_forbidden:
                block = BoundedAuthorityBlockReason.READ_ONLY_CONSTRAINT
            elif not goal.execution_requested:
                block = BoundedAuthorityBlockReason.EXECUTION_NOT_REQUESTED
            elif goal.mutation_required is not RequirementValue.TRUE:
                block = BoundedAuthorityBlockReason.USER_CONFIRMATION_REQUIRED
        if block is None and policy.disposition is RiskDisposition.CONFIRM:
            block = BoundedAuthorityBlockReason.USER_CONFIRMATION_REQUIRED
        if block is None and action.action is NextActionType.RESPOND:
            explicit_agent_used = any(
                observation.agent is goal.explicit_agent
                and observation.status is ObservationStatus.SUCCESS
                for observation in state.observations
            )
            mutation_happened = any(
                observation.state_changes for observation in state.observations
            )
            if goal.explicit_agent is not None and not explicit_agent_used:
                block = BoundedAuthorityBlockReason.EXPLICIT_AGENT_NOT_USED
            elif mutation_happened and not facts.goal_verified:
                block = BoundedAuthorityBlockReason.GOAL_NOT_VERIFIED
            elif goal.execution_requested and not facts.goal_evidence_sufficient:
                block = BoundedAuthorityBlockReason.PREMATURE_RESPONSE
        return BoundedAuthorityDecision(
            action_id=action.action_id,
            mode=self.mode,
            allowed=block is None,
            policy=policy,
            block_reason=block,
            confirmation_required=(
                block is BoundedAuthorityBlockReason.USER_CONFIRMATION_REQUIRED
            ),
            requires_availability_recheck=action.action is NextActionType.DELEGATE,
            mutation_authorized=bool(policy.mutation and block is None),
        )


# Phase 1 only transfers authority for agents the user can name explicitly.
AUTHORITATIVE_AGENTS = frozenset({Agent.CODEX, Agent.DEEPSEEK})

# Read-only can be enforced structurally for these agents:
#   codex    -> per-turn sandboxPolicy {"type": "readOnly"}
#   deepseek -> remote text generation with no filesystem or repository access
READ_ONLY_ENFORCEABLE_AGENTS = frozenset({Agent.CODEX, Agent.DEEPSEEK})


@dataclass(frozen=True)
class AvailabilitySample:
    """One availability observation with its origin, kept for TOCTOU auditing."""

    agent: Agent | None
    available: bool
    reason: str | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value if self.agent else None,
            "available": self.available,
            "reason": self.reason,
            "source": self.source,
        }


@dataclass(frozen=True)
class AuthoritativeExecutionDecision:
    """Gate decision plus authority, recheck state and dispatch permission."""

    mode: ExecutionAuthorityMode
    scope: AuthorityScope
    gate: ShadowExecutionDecision
    availability_at_selection: AvailabilitySample
    requested_agent_source: str | None = None
    availability_at_dispatch: AvailabilitySample | None = None
    authority_block_reason: AuthorityBlockReason | None = None
    recheck_performed: bool = False
    dispatched: bool = False

    @property
    def authoritative(self) -> bool:
        return (
            self.mode is ExecutionAuthorityMode.EXPLICIT_USER
            and self.scope is AuthorityScope.EXPLICIT_USER
        )

    @property
    def candidate_agent(self) -> Agent | None:
        return self.gate.candidate_agent

    @property
    def execution_mode(self) -> ExecutionMode:
        return self.gate.execution_mode

    @property
    def execution_allowed(self) -> bool:
        """Gate permission before the immediate recheck."""

        return self.gate.execution_allowed and self.authority_block_reason is None

    @property
    def dispatch_allowed(self) -> bool:
        """Permission to hand the request to the existing executor."""

        return bool(
            self.authoritative
            and self.gate.execution_allowed
            and self.authority_block_reason is None
            and self.recheck_performed
            and self.availability_at_dispatch is not None
            and self.availability_at_dispatch.available
        )

    @property
    def block_reason(self) -> str | None:
        if self.authority_block_reason is not None:
            return self.authority_block_reason.value
        if self.gate.block_reason is not None:
            return self.gate.block_reason.value
        return None

    @property
    def availability_changed(self) -> bool:
        return bool(
            self.availability_at_dispatch is not None
            and self.availability_at_dispatch.available
            != self.availability_at_selection.available
        )

    def dispatch_context(self) -> dict[str, Any]:
        """Structural execution envelope for the existing executor."""

        return {
            "execution_mode": self.gate.execution_mode.value,
            "mutation_authorized": self.gate.mutation_authorized,
            "execution_authority": self.mode.value,
            "selection_source": self.gate.selection_source.value,
            "requested_agent": (
                self.gate.requested_agent.value if self.gate.requested_agent else None
            ),
        }

    def provenance_record(self) -> dict[str, Any]:
        gate = self.gate
        return {
            "authority": self.mode.value,
            "authority_scope": self.scope.value,
            "authoritative": self.authoritative,
            "requested_agent": (
                gate.requested_agent.value if gate.requested_agent else None
            ),
            "requested_agent_source": (
                self.requested_agent_source or gate.requested_agent_source
            ),
            "selected_agent": gate.candidate_agent.value if gate.candidate_agent else None,
            "selection_source": gate.selection_source.value,
            "selection_factors": [item.value for item in gate.selection_factors],
            "selection_reason_code": gate.selection_reason_code,
            "execution_requested": gate.execution_requested,
            "execution_gate_result": "ALLOW" if gate.execution_allowed else "BLOCK",
            "execution_allowed": self.execution_allowed,
            "dispatch_allowed": self.dispatch_allowed,
            "dispatched": self.dispatched,
            "block_reason": self.block_reason,
            "block_reasons": [item.value for item in gate.block_reasons],
            "execution_mode": gate.execution_mode.value,
            "mutation_requested": gate.mutation_requested,
            "mutation_authorized": gate.mutation_authorized,
            "mutation_block_reason": (
                gate.mutation_block_reason.value if gate.mutation_block_reason else None
            ),
            "eligible_agents": [item.value for item in gate.eligible_agents],
            "available_eligible_agents": [
                item.value for item in gate.available_eligible_agents
            ],
            "availability_at_selection": self.availability_at_selection.as_dict(),
            "availability_at_dispatch": (
                self.availability_at_dispatch.as_dict()
                if self.availability_at_dispatch
                else None
            ),
            "availability_changed_before_dispatch": self.availability_changed,
            "recheck_performed": self.recheck_performed,
            "provenance_complete": gate.provenance_complete,
        }


def explicit_authority_scope(
    requested_agent: Agent | None,
    requested_agent_source: str | None,
) -> AuthorityScope:
    """Only a real explicit user binding enters the authoritative scope."""

    if requested_agent is None:
        return AuthorityScope.OUT_OF_SCOPE
    if (requested_agent_source or "").strip().lower() != EXPLICIT_USER_SOURCE:
        return AuthorityScope.OUT_OF_SCOPE
    if requested_agent not in AUTHORITATIVE_AGENTS:
        return AuthorityScope.OUT_OF_SCOPE
    return AuthorityScope.EXPLICIT_USER


class ExecutionAuthorityController:
    """Pure decisions about authority, read-only enforceability and recheck."""

    def __init__(
        self,
        mode: ExecutionAuthorityMode = ExecutionAuthorityMode.SHADOW,
        *,
        read_only_enforceable: frozenset[Agent] = READ_ONLY_ENFORCEABLE_AGENTS,
    ):
        self.mode = mode
        self.read_only_enforceable = read_only_enforceable

    def scope_of(
        self,
        requested_agent: Agent | None,
        requested_agent_source: str | None,
    ) -> AuthorityScope:
        return explicit_authority_scope(requested_agent, requested_agent_source)

    def decide(
        self,
        gate: ShadowExecutionDecision,
        *,
        requested_agent: Agent | None,
        requested_agent_source: str | None,
        availability_at_selection: AvailabilitySample,
    ) -> AuthoritativeExecutionDecision:
        scope = self.scope_of(requested_agent, requested_agent_source)
        block: AuthorityBlockReason | None = None
        authoritative = (
            self.mode is ExecutionAuthorityMode.EXPLICIT_USER
            and scope is AuthorityScope.EXPLICIT_USER
        )
        candidate = gate.candidate_agent
        if authoritative:
            # the explicit choice is sovereign: no substitution is ever accepted
            if candidate is not None and candidate != requested_agent:
                block = AuthorityBlockReason.AGENT_OUTSIDE_AUTHORITY_SCOPE
            elif candidate is not None and candidate not in AUTHORITATIVE_AGENTS:
                block = AuthorityBlockReason.AGENT_OUTSIDE_AUTHORITY_SCOPE
            elif (
                gate.execution_allowed
                and gate.execution_mode is ExecutionMode.READ_ONLY
                and candidate is not None
                and candidate not in self.read_only_enforceable
            ):
                # never pretend read-only enforcement downstream
                block = AuthorityBlockReason.READ_ONLY_ENFORCEMENT_UNAVAILABLE
        return AuthoritativeExecutionDecision(
            mode=self.mode if authoritative else ExecutionAuthorityMode.SHADOW,
            scope=scope,
            gate=gate,
            availability_at_selection=availability_at_selection,
            requested_agent_source=requested_agent_source,
            authority_block_reason=block,
        )

    def recheck(
        self,
        decision: AuthoritativeExecutionDecision,
        availability_at_dispatch: AvailabilitySample,
    ) -> AuthoritativeExecutionDecision:
        """Immediate pre-dispatch revalidation. Divergence is terminal."""

        block = decision.authority_block_reason
        if not availability_at_dispatch.available and block is None:
            block = AuthorityBlockReason.AVAILABILITY_CHANGED_BEFORE_DISPATCH
        return replace(
            decision,
            availability_at_dispatch=availability_at_dispatch,
            recheck_performed=True,
            authority_block_reason=block,
        )

    @staticmethod
    def mark_dispatched(
        decision: AuthoritativeExecutionDecision,
    ) -> AuthoritativeExecutionDecision:
        return replace(decision, dispatched=True)


def probe_agent_availability(
    registry: Any,
    agent: Agent | None,
    *,
    source: str = "dispatch",
) -> AvailabilitySample:
    """Live availability from the same operational facts the legacy path uses.

    It reads tool registration and the agent's own enabled/configured status.
    Nothing is cached, so this is a real recheck rather than a replay.
    """

    if agent is None:
        return AvailabilitySample(None, False, "NO_CANDIDATE_AGENT", source)
    tool = f"delegate_to_{agent.value}"
    names = getattr(registry, "names", None)
    registered = tool in set(names() if callable(names) else ())
    if not registered:
        return AvailabilitySample(agent, False, "tool_not_registered", source)
    if agent is Agent.DEEPSEEK:
        status: Mapping[str, Any] = {}
        probe = getattr(getattr(registry, "deepseek", None), "status", None)
        if callable(probe):
            try:
                value = probe()
                if isinstance(value, Mapping):
                    status = value
            except Exception:
                return AvailabilitySample(agent, False, "agent_probe_failed", source)
        if not bool(status.get("enabled")):
            return AvailabilitySample(agent, False, "agent_disabled", source)
        if not bool(status.get("configured")):
            return AvailabilitySample(agent, False, "agent_not_configured", source)
    if agent is Agent.CODEX:
        codex = getattr(registry, "codex", None)
        if codex is None:
            return AvailabilitySample(agent, False, "agent_not_configured", source)
        probe = getattr(codex, "available", None)
        if callable(probe):
            try:
                if not bool(probe()):
                    return AvailabilitySample(
                        agent, False, "agent_runtime_unavailable", source
                    )
            except Exception:
                return AvailabilitySample(agent, False, "agent_probe_failed", source)
    return AvailabilitySample(agent, True, None, source)


def availability_sample_from_gate(
    gate: ShadowExecutionDecision,
    *,
    source: str = "selection",
) -> AvailabilitySample:
    candidate = gate.candidate_agent
    if candidate is None:
        return AvailabilitySample(None, False, "NO_CANDIDATE_AGENT", source)
    available = candidate in gate.available_eligible_agents
    return AvailabilitySample(
        candidate,
        available,
        None if available else ExecutionBlockReason.AGENT_UNAVAILABLE.value,
        source,
    )


__all__ = [
    "AUTHORITATIVE_AGENTS",
    "READ_ONLY_ENFORCEABLE_AGENTS",
    "AuthorityBlockReason",
    "AuthorityScope",
    "AuthoritativeExecutionDecision",
    "AvailabilitySample",
    "ExecutionAuthority",
    "ExecutionAuthorityController",
    "ExecutionAuthorityMode",
    "availability_sample_from_gate",
    "explicit_authority_scope",
    "probe_agent_availability",
]
