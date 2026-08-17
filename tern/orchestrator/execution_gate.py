"""Execution gate: turns existing facts into one structured execution decision.

Three questions stay separate on purpose:

    who should execute      -> selection (agent_selection.AgentSelectionProposal)
    should execution occur  -> ExecutionGate.execution_allowed
    may it mutate           -> ExecutionGate.mutation_authorized

The gate is a pure function. It owns no tools, resolves no session, creates no
job and never mutates anything. In this stage its authority is SHADOW, so the
legacy pipeline keeps every operational decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .agent_selection import (
    AgentSelectionProposal,
    SelectionFactor,
    SelectionFactorType,
    SelectionPolicy,
    SelectionSource,
)
from .autonomy_foundation import Agent, AgentCapabilityProfile, Capability
from .task_requirement_grounding import (
    MUTATION_REQUIRED,
    READ_ONLY_REQUIRED,
    GroundedAgentEligibility,
    GroundedTaskRequirements,
    RequirementValue,
)


class ExecutionAuthority(str, Enum):
    """Only SHADOW is produced in this stage; LIVE exists for the next one."""

    SHADOW = "SHADOW"
    LIVE = "LIVE"


class ExecutionMode(str, Enum):
    READ_ONLY = "READ_ONLY"
    MUTATION = "MUTATION"


class ExecutionBlockReason(str, Enum):
    EXECUTION_NOT_REQUESTED = "EXECUTION_NOT_REQUESTED"
    NO_ELIGIBLE_AGENT = "NO_ELIGIBLE_AGENT"
    NO_AVAILABLE_ELIGIBLE_AGENT = "NO_AVAILABLE_ELIGIBLE_AGENT"
    SELECTION_UNRESOLVED = "SELECTION_UNRESOLVED"
    INELIGIBLE_AGENT = "INELIGIBLE_AGENT"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    REQUESTED_AGENT_UNAVAILABLE = "REQUESTED_AGENT_UNAVAILABLE"
    REQUESTED_AGENT_INELIGIBLE = "REQUESTED_AGENT_INELIGIBLE"
    POLICY_EXCLUDED_FROM_AUTO_SELECTION = "POLICY_EXCLUDED_FROM_AUTO_SELECTION"
    EXECUTION_SAFETY_UNRESOLVED = "EXECUTION_SAFETY_UNRESOLVED"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    SELECTION_PROVENANCE_MISSING = "SELECTION_PROVENANCE_MISSING"


class MutationBlockReason(str, Enum):
    MUTATION_NOT_REQUESTED = "MUTATION_NOT_REQUESTED"
    READ_ONLY_TASK = "READ_ONLY_TASK"
    MUTATION_FORBIDDEN_BY_CONSTRAINT = "MUTATION_FORBIDDEN_BY_CONSTRAINT"
    EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
    AGENT_CANNOT_MUTATE = "AGENT_CANNOT_MUTATE"
    MUTATION_REQUIREMENT_UNRESOLVED = "MUTATION_REQUIREMENT_UNRESOLVED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    PATH_POLICY_NOT_SATISFIED = "PATH_POLICY_NOT_SATISFIED"


_UNRESOLVED_SOURCES = frozenset(
    {
        SelectionSource.UNRESOLVED,
        SelectionSource.INVALID_SELECTION,
    }
)
_MISSING_CANDIDATE_SOURCES = frozenset(
    {
        SelectionSource.NO_ELIGIBLE_AGENT,
        SelectionSource.NO_AVAILABLE_ELIGIBLE_AGENT,
    }
)


@dataclass(frozen=True)
class ExecutionGateInput:
    """Facts only. Nothing here is resolved lazily and nothing is executable."""

    execution_requested: bool
    candidate_agent: Agent | None
    selection_source: SelectionSource
    eligible_agents: tuple[Agent, ...] = ()
    available_eligible_agents: tuple[Agent, ...] = ()
    requested_agent: Agent | None = None
    requested_agent_source: str | None = None
    selection_reason_code: str = ""
    selection_factors: tuple[SelectionFactorType, ...] = ()
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)
    # requirement-derived mutation facts
    mutation_requested: bool = False
    mutation_requirement_unresolved: bool = False
    read_only_required: bool = False
    forbid_mutation: bool = False
    agent_can_mutate: bool = False
    # execution-safety facts owned by grounded eligibility
    execution_safe: bool = True
    unresolved_safety_requirements: tuple[str, ...] = ()
    conflict_requirements: tuple[str, ...] = ()
    # legacy operational facts, consumed read-only and never rewritten here
    constraint_violation: str | None = None
    confirmation_required: bool = False
    path_policy_satisfied: bool = True

    @property
    def explicit(self) -> bool:
        return self.selection_source is SelectionSource.EXPLICIT_USER

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_requested": self.execution_requested,
            "candidate_agent": self.candidate_agent.value if self.candidate_agent else None,
            "selection_source": self.selection_source.value,
            "eligible_agents": [item.value for item in self.eligible_agents],
            "available_eligible_agents": [
                item.value for item in self.available_eligible_agents
            ],
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "requested_agent_source": self.requested_agent_source,
            "selection_reason_code": self.selection_reason_code,
            "selection_factors": [item.value for item in self.selection_factors],
            "policy": {"deepseek_auto_escalation": self.policy.deepseek_auto_escalation},
            "mutation_requested": self.mutation_requested,
            "mutation_requirement_unresolved": self.mutation_requirement_unresolved,
            "read_only_required": self.read_only_required,
            "forbid_mutation": self.forbid_mutation,
            "agent_can_mutate": self.agent_can_mutate,
            "execution_safe": self.execution_safe,
            "unresolved_safety_requirements": list(self.unresolved_safety_requirements),
            "conflict_requirements": list(self.conflict_requirements),
            "constraint_violation": self.constraint_violation,
            "confirmation_required": self.confirmation_required,
            "path_policy_satisfied": self.path_policy_satisfied,
        }


@dataclass(frozen=True)
class ShadowExecutionDecision:
    """Structured decision with zero authority in this stage."""

    candidate_agent: Agent | None
    execution_requested: bool
    agent_eligible: bool
    agent_available: bool
    selection_valid: bool
    selection_supported: bool
    execution_allowed: bool
    execution_mode: ExecutionMode
    mutation_requested: bool
    mutation_authorized: bool
    block_reason: ExecutionBlockReason | None
    block_reasons: tuple[ExecutionBlockReason, ...]
    mutation_block_reason: MutationBlockReason | None
    selection_source: SelectionSource
    selection_factors: tuple[SelectionFactorType, ...]
    selection_reason_code: str
    requested_agent: Agent | None
    requested_agent_source: str | None
    eligible_agents: tuple[Agent, ...]
    available_eligible_agents: tuple[Agent, ...]

    # Shadow invariants: this object can never authorize or perform anything.
    authority: ExecutionAuthority = field(default=ExecutionAuthority.SHADOW, init=False)
    live_authority: bool = field(default=False, init=False)
    mode: str = field(default="SHADOW", init=False)
    delegations: int = field(default=0, init=False)
    jobs_created: int = field(default=0, init=False)
    sessions_resolved: int = field(default=0, init=False)
    filesystem_mutations: int = field(default=0, init=False)

    @property
    def provenance_complete(self) -> bool:
        if self.selection_source in _MISSING_CANDIDATE_SOURCES:
            return self.candidate_agent is None
        if self.selection_source in _UNRESOLVED_SOURCES:
            return True
        return bool(self.selection_reason_code) and self.candidate_agent is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "authority": self.authority.value,
            "live_authority": False,
            "candidate_agent": self.candidate_agent.value if self.candidate_agent else None,
            "execution_requested": self.execution_requested,
            "agent_eligible": self.agent_eligible,
            "agent_available": self.agent_available,
            "selection_valid": self.selection_valid,
            "selection_supported": self.selection_supported,
            "execution_allowed": self.execution_allowed,
            "execution_mode": self.execution_mode.value,
            "mutation_requested": self.mutation_requested,
            "mutation_authorized": self.mutation_authorized,
            "block_reason": self.block_reason.value if self.block_reason else None,
            "block_reasons": [item.value for item in self.block_reasons],
            "mutation_block_reason": (
                self.mutation_block_reason.value if self.mutation_block_reason else None
            ),
            "selection_source": self.selection_source.value,
            "selection_factors": [item.value for item in self.selection_factors],
            "selection_reason_code": self.selection_reason_code,
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "requested_agent_source": self.requested_agent_source,
            "eligible_agents": [item.value for item in self.eligible_agents],
            "available_eligible_agents": [
                item.value for item in self.available_eligible_agents
            ],
            "provenance_complete": self.provenance_complete,
            "delegations": 0,
            "jobs_created": 0,
            "sessions_resolved": 0,
            "filesystem_mutations": 0,
        }


class ExecutionGate:
    """Deterministic, side-effect free evaluation. Fails closed by construction."""

    authority = ExecutionAuthority.SHADOW

    def evaluate(self, facts: ExecutionGateInput) -> ShadowExecutionDecision:
        candidate = facts.candidate_agent
        eligible = tuple(facts.eligible_agents)
        available = tuple(facts.available_eligible_agents)
        agent_eligible = bool(candidate is not None and candidate in eligible)
        agent_available = bool(candidate is not None and candidate in available)
        selection_valid = (
            candidate is not None
            and facts.selection_source not in _UNRESOLVED_SOURCES
            and facts.selection_source not in _MISSING_CANDIDATE_SOURCES
        )
        selection_supported = bool(
            selection_valid
            and (
                facts.explicit
                or facts.selection_factors
                or facts.selection_source
                in {
                    SelectionSource.SINGLE_ELIGIBLE_AGENT,
                    SelectionSource.ONLY_AVAILABLE_ELIGIBLE_AGENT,
                }
            )
        )

        reasons: list[ExecutionBlockReason] = []

        # 1. execution intent is owned by the existing frame, never re-parsed here.
        if not facts.execution_requested:
            reasons.append(ExecutionBlockReason.EXECUTION_NOT_REQUESTED)

        # 2. candidate-set facts.
        if not eligible:
            reasons.append(ExecutionBlockReason.NO_ELIGIBLE_AGENT)
        elif not available:
            reasons.append(ExecutionBlockReason.NO_AVAILABLE_ELIGIBLE_AGENT)

        # 3. selection facts.
        if facts.selection_source in _UNRESOLVED_SOURCES or candidate is None:
            if eligible and available:
                reasons.append(ExecutionBlockReason.SELECTION_UNRESOLVED)
            elif candidate is None and not eligible:
                pass  # already reported as NO_ELIGIBLE_AGENT
        else:
            if not agent_eligible:
                reasons.append(
                    ExecutionBlockReason.REQUESTED_AGENT_INELIGIBLE
                    if facts.explicit
                    else ExecutionBlockReason.INELIGIBLE_AGENT
                )
            elif not agent_available:
                reasons.append(
                    ExecutionBlockReason.REQUESTED_AGENT_UNAVAILABLE
                    if facts.explicit
                    else ExecutionBlockReason.AGENT_UNAVAILABLE
                )

        # 4. policy may remove an agent from automatic candidacy without ever
        #    touching eligibility, and never overrides an explicit user request.
        if (
            candidate is not None
            and not facts.explicit
            and candidate in facts.policy.agents_requiring_explicit_request()
        ):
            reasons.append(ExecutionBlockReason.POLICY_EXCLUDED_FROM_AUTO_SELECTION)

        # 5. grounded execution safety (UNKNOWN/CONFLICT never authorizes).
        if not facts.execution_safe:
            reasons.append(ExecutionBlockReason.EXECUTION_SAFETY_UNRESOLVED)

        # 6. explicit semantic constraints from the legacy frame.
        if facts.constraint_violation:
            reasons.append(ExecutionBlockReason.CONSTRAINT_VIOLATION)

        # 7. provenance must exist for any allowed execution.
        if selection_valid and not facts.selection_reason_code:
            reasons.append(ExecutionBlockReason.SELECTION_PROVENANCE_MISSING)

        ordered = tuple(dict.fromkeys(reasons))
        execution_allowed = not ordered

        mutation_block = self._mutation_block_reason(facts, execution_allowed)
        mutation_authorized = mutation_block is None
        execution_mode = (
            ExecutionMode.MUTATION if mutation_authorized else ExecutionMode.READ_ONLY
        )

        return ShadowExecutionDecision(
            candidate_agent=candidate,
            execution_requested=facts.execution_requested,
            agent_eligible=agent_eligible,
            agent_available=agent_available,
            selection_valid=selection_valid,
            selection_supported=selection_supported,
            execution_allowed=execution_allowed,
            execution_mode=execution_mode,
            mutation_requested=facts.mutation_requested,
            mutation_authorized=mutation_authorized,
            block_reason=ordered[0] if ordered else None,
            block_reasons=ordered,
            mutation_block_reason=mutation_block,
            selection_source=facts.selection_source,
            selection_factors=tuple(facts.selection_factors),
            selection_reason_code=facts.selection_reason_code,
            requested_agent=facts.requested_agent,
            requested_agent_source=facts.requested_agent_source,
            eligible_agents=eligible,
            available_eligible_agents=available,
        )

    @staticmethod
    def _mutation_block_reason(
        facts: ExecutionGateInput,
        execution_allowed: bool,
    ) -> MutationBlockReason | None:
        """Delegation is never mutation authorization."""

        if not facts.mutation_requested:
            return MutationBlockReason.MUTATION_NOT_REQUESTED
        if facts.forbid_mutation:
            return MutationBlockReason.MUTATION_FORBIDDEN_BY_CONSTRAINT
        if facts.read_only_required:
            return MutationBlockReason.READ_ONLY_TASK
        if facts.mutation_requirement_unresolved:
            return MutationBlockReason.MUTATION_REQUIREMENT_UNRESOLVED
        if not execution_allowed:
            return MutationBlockReason.EXECUTION_BLOCKED
        if not facts.agent_can_mutate:
            return MutationBlockReason.AGENT_CANNOT_MUTATE
        if facts.confirmation_required:
            return MutationBlockReason.CONFIRMATION_REQUIRED
        if not facts.path_policy_satisfied:
            return MutationBlockReason.PATH_POLICY_NOT_SATISFIED
        return None


def gate_input_from_proposal(
    proposal: AgentSelectionProposal,
    requirements: GroundedTaskRequirements,
    *,
    execution_requested: bool,
    evaluations: Mapping[Agent, GroundedAgentEligibility],
    capability_profiles: Mapping[Agent, AgentCapabilityProfile],
    policy: SelectionPolicy,
    requested_agent_source: str | None = None,
    forbid_mutation: bool = False,
    constraint_violation: str | None = None,
    confirmation_required: bool = False,
    path_policy_satisfied: bool = True,
) -> ExecutionGateInput:
    """Adapt already-computed facts. This function derives nothing new."""

    candidate = proposal.proposed_agent
    mutation = requirements.requirements[MUTATION_REQUIRED]
    read_only = requirements.requirements[READ_ONLY_REQUIRED]
    evaluation = evaluations.get(candidate) if candidate is not None else None
    profile = capability_profiles.get(candidate) if candidate is not None else None
    agent_can_mutate = bool(
        profile is not None
        and profile.has(Capability.MUTATION)
        and requirements.mutation_authorized_by_requirements
    )
    return ExecutionGateInput(
        execution_requested=execution_requested,
        candidate_agent=candidate,
        selection_source=proposal.selection_source,
        eligible_agents=tuple(proposal.eligible_agents),
        available_eligible_agents=tuple(proposal.available_eligible_agents),
        requested_agent=proposal.requested_agent,
        requested_agent_source=requested_agent_source,
        selection_reason_code=proposal.reason_code,
        selection_factors=_factor_types(proposal.factors, proposal.task_factors),
        policy=policy,
        mutation_requested=mutation.value is RequirementValue.TRUE,
        mutation_requirement_unresolved=mutation.value
        in {RequirementValue.UNKNOWN, RequirementValue.CONFLICT},
        read_only_required=read_only.value is RequirementValue.TRUE,
        forbid_mutation=forbid_mutation,
        agent_can_mutate=agent_can_mutate,
        execution_safe=bool(evaluation.execution_safe) if evaluation else False,
        unresolved_safety_requirements=(
            tuple(evaluation.unresolved_safety_requirements) if evaluation else ()
        ),
        conflict_requirements=tuple(evaluation.conflict_requirements) if evaluation else (),
        constraint_violation=constraint_violation,
        confirmation_required=confirmation_required,
        path_policy_satisfied=path_policy_satisfied,
    )


def _factor_types(
    *groups: Sequence[SelectionFactor],
) -> tuple[SelectionFactorType, ...]:
    seen: dict[SelectionFactorType, None] = {}
    for group in groups:
        for item in group:
            seen.setdefault(item.type, None)
    return tuple(seen)
