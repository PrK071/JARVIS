"""Shadow bridge between the new selection pipeline and the legacy live one.

The legacy pipeline keeps every operational decision. This module only observes:
it derives grounded requirements, capabilities, eligibility, availability and one
agent selection proposal from facts the live request already produced, evaluates
the ExecutionGate in SHADOW authority and compares the result with the legacy
decision.

It calls no tool, delegates to no agent, resolves no session, creates no job and
writes nothing to the filesystem. The deterministic path performs zero model
calls; a semantic selector may be injected explicitly and is measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .agent_selection import (
    AgentSelectionEngine,
    AgentSelectionProposal,
    OperationalFacts,
    SelectionPolicy,
    SelectionProjectContext,
    SelectionSource,
    SemanticAgentSelector,
)
from .autonomy_foundation import (
    Agent,
    AgentRuntimeAvailability,
    CapabilityBaseline,
    CapabilityProfileBuilder,
    RiskLevel,
)
from .decision_policy import SideEffect, TOOL_EFFECTS
from .execution_gate import (
    ExecutionAuthority,
    ExecutionGate,
    ExecutionGateInput,
    ExecutionMode,
    ShadowExecutionDecision,
    gate_input_from_proposal,
)
from .intent_semantics import Constraint, IntentFrame
from .task_requirement_grounding import (
    MUTATION_REQUIRED,
    READ_ONLY_REQUIRED,
    REQUIREMENT_DIMENSIONS,
    ExplicitTaskEvidence,
    ExplicitTaskEvidenceExtractor,
    GroundedEligibilityEngine,
    GroundedRequirement,
    GroundedTaskRequirements,
    RequirementValue,
)


_AGENT_BY_TOOL: Mapping[str, Agent] = {
    "delegate_to_codex": Agent.CODEX,
    "steer_codex_job": Agent.CODEX,
    "cancel_codex_job": Agent.CODEX,
    "delegate_to_deepseek": Agent.DEEPSEEK,
}
_EXECUTION_EFFECTS = frozenset(
    {
        SideEffect.CODE_EXECUTION,
        SideEffect.REMOTE_GENERATION,
        SideEffect.LOCAL_MUTATION,
    }
)


class DivergenceCode(str, Enum):
    AGENT_MISMATCH = "AGENT_MISMATCH"
    LEGACY_EXECUTES_SHADOW_BLOCKS = "LEGACY_EXECUTES_SHADOW_BLOCKS"
    LEGACY_BLOCKS_SHADOW_ALLOWS = "LEGACY_BLOCKS_SHADOW_ALLOWS"
    MUTATION_POLICY_MISMATCH = "MUTATION_POLICY_MISMATCH"
    ELIGIBILITY_MISMATCH = "ELIGIBILITY_MISMATCH"
    AVAILABILITY_MISMATCH = "AVAILABILITY_MISMATCH"
    EXPLICIT_AGENT_MISMATCH = "EXPLICIT_AGENT_MISMATCH"
    POLICY_EXCLUSION_MISMATCH = "POLICY_EXCLUSION_MISMATCH"
    SELECTION_PROVENANCE_MISSING = "SELECTION_PROVENANCE_MISSING"


def grounded_requirements_from_evidence(
    evidence: ExplicitTaskEvidence,
    *,
    ambiguity_material: bool | None = None,
    risk_level: RiskLevel | None = None,
) -> GroundedTaskRequirements:
    """Deterministic requirements: unseen dimensions stay UNKNOWN, never guessed."""

    seeded = dict(evidence.requirements)
    requirements: dict[str, GroundedRequirement] = {
        name: seeded.get(name, GroundedRequirement.unknown(name))
        for name in REQUIREMENT_DIMENSIONS
    }
    mutation = requirements[MUTATION_REQUIRED].value
    inferred_risk = risk_level or (
        RiskLevel.MEDIUM if mutation is RequirementValue.TRUE else RiskLevel.LOW
    )
    return GroundedTaskRequirements(
        requirements=requirements,
        target_scope=evidence.target_scope,
        risk_level=inferred_risk,
        ambiguity_material=(
            bool(evidence.contradiction_refs)
            if ambiguity_material is None
            else ambiguity_material
        ),
        expected_files=evidence.expected_files,
        forbidden_files=evidence.forbidden_files,
        tests_requested=evidence.tests_requested,
        prohibitions=evidence.prohibitions,
        requested_agent=evidence.requested_agent,
        requested_agent_source=evidence.requested_agent_source,
        requested_agent_evidence_ref=evidence.requested_agent_evidence_ref,
    )


@dataclass(frozen=True)
class LegacyDecisionFacts:
    """What the live pipeline decided, read without changing it."""

    agent: Agent | None
    selected_action: str | None
    side_effect: SideEffect | None
    execution_allowed: bool
    mutation_behavior: bool
    requested_agent: Agent | None
    constraint_violation: str | None
    intent: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value if self.agent else None,
            "selected_action": self.selected_action,
            "side_effect": self.side_effect.value if self.side_effect else None,
            "execution_allowed": self.execution_allowed,
            "mutation_behavior": self.mutation_behavior,
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "constraint_violation": self.constraint_violation,
            "intent": self.intent,
        }


def legacy_facts_from_decision(decision: Any) -> LegacyDecisionFacts:
    tool = getattr(decision, "selected_action", None)
    effect = TOOL_EFFECTS.get(tool) if tool else None
    requested = getattr(decision, "requested_agent", None)
    explicit_allowed = getattr(decision, "execution_allowed", None)
    constraint = getattr(decision, "constraint_violation", None)
    effects = tuple(getattr(decision, "side_effects", ()) or ())
    agent = _AGENT_BY_TOOL.get(tool or "")
    if agent is None and tool:
        agent = Agent.LOCAL
    execution_allowed = bool(
        tool
        and constraint is None
        and effect in _EXECUTION_EFFECTS
        and explicit_allowed is not False
    )
    mutation_behavior = bool(
        effect is SideEffect.LOCAL_MUTATION
        or SideEffect.LOCAL_MUTATION in effects
        or effect is SideEffect.CODE_EXECUTION
    ) and constraint is None
    return LegacyDecisionFacts(
        agent=agent,
        selected_action=tool,
        side_effect=effect,
        execution_allowed=execution_allowed,
        mutation_behavior=mutation_behavior,
        requested_agent=Agent(requested) if requested else None,
        constraint_violation=constraint,
        intent=getattr(getattr(decision, "intent", None), "value", None),
    )


@dataclass(frozen=True)
class LegacyVsShadowDecision:
    legacy_agent: Agent | None
    shadow_agent: Agent | None
    legacy_execution_allowed: bool
    shadow_execution_allowed: bool
    legacy_mutation_behavior: bool
    shadow_mutation_authorized: bool
    agent_agreement: bool
    execution_agreement: bool
    mutation_agreement: bool
    divergence_codes: tuple[DivergenceCode, ...]

    @property
    def agreement(self) -> bool:
        return not self.divergence_codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "legacy_agent": self.legacy_agent.value if self.legacy_agent else None,
            "shadow_agent": self.shadow_agent.value if self.shadow_agent else None,
            "legacy_execution_allowed": self.legacy_execution_allowed,
            "shadow_execution_allowed": self.shadow_execution_allowed,
            "legacy_mutation_behavior": self.legacy_mutation_behavior,
            "shadow_mutation_authorized": self.shadow_mutation_authorized,
            "agent_agreement": self.agent_agreement,
            "execution_agreement": self.execution_agreement,
            "mutation_agreement": self.mutation_agreement,
            "agreement": self.agreement,
            "divergence_codes": [item.value for item in self.divergence_codes],
        }


def compare_legacy_and_shadow(
    legacy: LegacyDecisionFacts,
    shadow: ShadowExecutionDecision,
    *,
    policy: SelectionPolicy | None = None,
) -> LegacyVsShadowDecision:
    """Divergence is recorded, never repaired."""

    policy = policy or SelectionPolicy()
    codes: list[DivergenceCode] = []
    agent_agreement = legacy.agent == shadow.candidate_agent
    if not agent_agreement and legacy.agent is not None and shadow.candidate_agent is not None:
        codes.append(DivergenceCode.AGENT_MISMATCH)
    elif not agent_agreement:
        codes.append(DivergenceCode.AGENT_MISMATCH)
    if legacy.execution_allowed and not shadow.execution_allowed:
        codes.append(DivergenceCode.LEGACY_EXECUTES_SHADOW_BLOCKS)
    if shadow.execution_allowed and not legacy.execution_allowed:
        codes.append(DivergenceCode.LEGACY_BLOCKS_SHADOW_ALLOWS)
    if legacy.mutation_behavior != shadow.mutation_authorized:
        codes.append(DivergenceCode.MUTATION_POLICY_MISMATCH)
    if legacy.agent is not None and legacy.agent not in shadow.eligible_agents:
        codes.append(DivergenceCode.ELIGIBILITY_MISMATCH)
    elif (
        legacy.agent is not None
        and legacy.agent not in shadow.available_eligible_agents
    ):
        codes.append(DivergenceCode.AVAILABILITY_MISMATCH)
    if (
        legacy.requested_agent is not None
        and shadow.candidate_agent is not None
        and legacy.requested_agent != shadow.candidate_agent
    ):
        codes.append(DivergenceCode.EXPLICIT_AGENT_MISMATCH)
    if (
        legacy.agent in policy.agents_requiring_explicit_request()
        and legacy.requested_agent is None
        and legacy.execution_allowed
    ):
        codes.append(DivergenceCode.POLICY_EXCLUSION_MISMATCH)
    if not shadow.provenance_complete:
        codes.append(DivergenceCode.SELECTION_PROVENANCE_MISSING)
    return LegacyVsShadowDecision(
        legacy_agent=legacy.agent,
        shadow_agent=shadow.candidate_agent,
        legacy_execution_allowed=legacy.execution_allowed,
        shadow_execution_allowed=shadow.execution_allowed,
        legacy_mutation_behavior=legacy.mutation_behavior,
        shadow_mutation_authorized=shadow.mutation_authorized,
        agent_agreement=agent_agreement,
        execution_agreement=legacy.execution_allowed == shadow.execution_allowed,
        mutation_agreement=legacy.mutation_behavior == shadow.mutation_authorized,
        divergence_codes=tuple(dict.fromkeys(codes)),
    )


@dataclass(frozen=True)
class ShadowObservation:
    """One shadow pass over a live request. Carries no authority."""

    requirements: GroundedTaskRequirements
    proposal: AgentSelectionProposal
    gate_input: ExecutionGateInput
    decision: ShadowExecutionDecision
    comparison: LegacyVsShadowDecision | None
    availability_snapshot: Mapping[Agent, AgentRuntimeAvailability]
    model_calls: int
    deterministic_latency_ms: float
    semantic_latency_ms: float
    total_latency_ms: float
    authority: ExecutionAuthority = field(default=ExecutionAuthority.SHADOW, init=False)

    def provenance_record(self) -> dict[str, Any]:
        """Structured provenance only; no prompt or task text is recorded."""

        decision = self.decision
        return {
            "mode": "SHADOW",
            "authority": self.authority.value,
            "live_authority": False,
            "selection_source": decision.selection_source.value,
            "selection_factors": [item.value for item in decision.selection_factors],
            "selection_reason_code": decision.selection_reason_code,
            "requested_agent": (
                decision.requested_agent.value if decision.requested_agent else None
            ),
            "requested_agent_source": decision.requested_agent_source,
            "eligible_agents": [item.value for item in decision.eligible_agents],
            "available_eligible_agents": [
                item.value for item in decision.available_eligible_agents
            ],
            "candidate_agent": (
                decision.candidate_agent.value if decision.candidate_agent else None
            ),
            "execution_requested": decision.execution_requested,
            "execution_allowed_shadow": decision.execution_allowed,
            "execution_mode_shadow": decision.execution_mode.value,
            "mutation_requested": decision.mutation_requested,
            "mutation_authorized_shadow": decision.mutation_authorized,
            "block_reason": decision.block_reason.value if decision.block_reason else None,
            "block_reasons": [item.value for item in decision.block_reasons],
            "mutation_block_reason": (
                decision.mutation_block_reason.value
                if decision.mutation_block_reason
                else None
            ),
            "provenance_complete": decision.provenance_complete,
            "unknown_requirement_dimensions": list(self.requirements.unknown_dimensions),
            "conflict_requirement_dimensions": list(self.requirements.conflict_dimensions),
            "model_calls": self.model_calls,
            "deterministic_latency_ms": self.deterministic_latency_ms,
            "semantic_latency_ms": self.semantic_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "delegations": 0,
            "jobs_created": 0,
            "sessions_resolved": 0,
            "filesystem_mutations": 0,
            "comparison": self.comparison.as_dict() if self.comparison else None,
        }


class _ReadOnlyRegistryView:
    """Exposes only what capability derivation needs: tool names and agent flags.

    Executors are deliberately not forwarded, so an observer cannot reach them.
    """

    def __init__(self, registry: Any):
        self._names = _tool_names(registry)
        self.codex = getattr(registry, "codex", None)
        self.deepseek = getattr(registry, "deepseek", None)

    def names(self) -> tuple[str, ...]:
        return self._names


def _tool_names(registry: Any) -> tuple[str, ...]:
    names = getattr(registry, "names", None)
    if callable(names):
        return tuple(str(item) for item in names())
    specs = getattr(registry, "specs", None)
    if callable(specs):
        return tuple(
            str((item.get("function") or {}).get("name") or "")
            for item in specs()
        )
    return ()


def capability_baseline_from_registry(
    registry: Any,
    *,
    local_model_available: bool = True,
    codex_available: bool = True,
) -> CapabilityBaseline:
    """Read-only capability/availability snapshot from the live tool registry."""

    return CapabilityProfileBuilder.from_registry(
        _ReadOnlyRegistryView(registry),
        local_model_available=local_model_available,
        codex_available=codex_available,
    )


class ShadowExecutionObserver:
    """Runs the new pipeline for observation only."""

    authority = ExecutionAuthority.SHADOW

    def __init__(
        self,
        *,
        policy: SelectionPolicy | None = None,
        semantic_selector: SemanticAgentSelector | None = None,
        extractor: ExplicitTaskEvidenceExtractor | None = None,
    ):
        self.policy = policy or SelectionPolicy()
        self.semantic_selector = semantic_selector
        self.extractor = extractor or ExplicitTaskEvidenceExtractor()
        self.eligibility = GroundedEligibilityEngine()
        self.gate = ExecutionGate()

    def observe(
        self,
        task: str,
        *,
        baseline: CapabilityBaseline,
        execution_requested: bool,
        intent_frame: IntentFrame | None = None,
        legacy: LegacyDecisionFacts | None = None,
        operational: OperationalFacts | None = None,
        project_context: SelectionProjectContext | None = None,
        project_snapshot: Mapping[str, Any] | None = None,
        availability_override: Mapping[Agent, bool] | None = None,
        confirmation_required: bool = False,
        path_policy_satisfied: bool = True,
        constraint_violation: str | None = None,
        requirements: GroundedTaskRequirements | None = None,
    ) -> ShadowObservation:
        started = time.perf_counter()
        if requirements is None:
            evidence = self.extractor.extract(
                task,
                project_snapshot=project_snapshot,
                intent_frame=intent_frame,
            )
            requirements = grounded_requirements_from_evidence(evidence)
        availability = _availability_snapshot(baseline.availability, availability_override)
        evaluations = self.eligibility.evaluate(
            requirements,
            baseline.profiles,
            availability,
        )
        engine = AgentSelectionEngine(
            policy=self.policy,
            semantic_selector=self.semantic_selector,
        )
        proposal = engine.propose(
            requirements,
            evaluations,
            capability_profiles=baseline.profiles,
            availability=availability,
            operational=operational or OperationalFacts(),
            project_context=project_context,
        )
        forbid_mutation = bool(
            intent_frame is not None
            and (
                Constraint.FORBID_MUTATION in intent_frame.constraints
                or Constraint.READ_ONLY in intent_frame.constraints
            )
        )
        gate_input = gate_input_from_proposal(
            proposal,
            requirements,
            execution_requested=execution_requested,
            evaluations=evaluations,
            capability_profiles=baseline.profiles,
            policy=self.policy,
            requested_agent_source=(
                requirements.requested_agent_source.value
                if requirements.requested_agent_source
                else None
            ),
            forbid_mutation=forbid_mutation,
            constraint_violation=constraint_violation,
            confirmation_required=confirmation_required,
            path_policy_satisfied=path_policy_satisfied,
        )
        decision = self.gate.evaluate(gate_input)
        comparison = (
            compare_legacy_and_shadow(legacy, decision, policy=self.policy)
            if legacy is not None
            else None
        )
        total = round((time.perf_counter() - started) * 1000, 3)
        return ShadowObservation(
            requirements=requirements,
            proposal=proposal,
            gate_input=gate_input,
            decision=decision,
            comparison=comparison,
            availability_snapshot=availability,
            model_calls=proposal.model_calls,
            deterministic_latency_ms=proposal.deterministic_latency_ms,
            semantic_latency_ms=proposal.model_latency_ms,
            total_latency_ms=total,
        )


def _availability_snapshot(
    baseline_availability: Mapping[Agent, AgentRuntimeAvailability],
    overrides: Mapping[Agent, bool] | None,
) -> dict[Agent, AgentRuntimeAvailability]:
    if not overrides:
        return dict(baseline_availability)
    snapshot: dict[Agent, AgentRuntimeAvailability] = {}
    for agent, state in baseline_availability.items():
        available = bool(overrides.get(agent, state.available))
        snapshot[agent] = AgentRuntimeAvailability(
            agent,
            available,
            state.enabled,
            state.configured,
            state.reason_code if available else f"{agent.value.upper()}_UNAVAILABLE",
        )
    return snapshot


__all__ = [
    "DivergenceCode",
    "ExecutionMode",
    "LegacyDecisionFacts",
    "LegacyVsShadowDecision",
    "ShadowExecutionObserver",
    "ShadowObservation",
    "capability_baseline_from_registry",
    "compare_legacy_and_shadow",
    "grounded_requirements_from_evidence",
    "legacy_facts_from_decision",
]
