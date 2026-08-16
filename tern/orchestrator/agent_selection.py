"""Automatic agent selection as a dry-run proposal with full provenance.

Selection answers one question only: among the agents that are genuinely able to
execute a task, which executor is the most appropriate proposal? It never
executes, never delegates, never resolves or creates a Codex session, never
grants permission and never mutates the layers below it.

Layering contract:

    task -> grounded requirements -> capability profiles -> eligibility
         -> availability -> selection factors -> proposal + provenance

Capability answers ``CAN DO?``. A selection factor answers ``AMONG ELIGIBLE
AGENTS, WHAT MAKES THIS AGENT A BETTER FIT?``. A capability alone never becomes
a preference, availability never rewrites eligibility, and no keyword, regex,
substring or task-name lookup participates in the decision: the semantic
selector does not even receive the raw task text.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentRuntimeAvailability,
    Capability,
)
from .task_requirement_grounding import (
    MUTATION_REQUIRED,
    READ_ONLY_REQUIRED,
    GroundedAgentEligibility,
    GroundedTaskRequirements,
    RequirementValue,
)


WRITE_CAPABILITIES = (
    Capability.REPOSITORY_WRITE,
    Capability.FILESYSTEM_WRITE,
    Capability.CODE_EDIT,
    Capability.MUTATION,
)
REPOSITORY_CAPABILITIES = (
    Capability.REPOSITORY_READ,
    Capability.REPOSITORY_WRITE,
    Capability.FILESYSTEM_READ,
    Capability.FILESYSTEM_WRITE,
    Capability.CODE_EDIT,
    Capability.TEST_EXECUTION,
)


class SelectionSource(str, Enum):
    """Mandatory provenance for every proposal."""

    EXPLICIT_USER = "EXPLICIT_USER"
    SINGLE_ELIGIBLE_AGENT = "SINGLE_ELIGIBLE_AGENT"
    ONLY_AVAILABLE_ELIGIBLE_AGENT = "ONLY_AVAILABLE_ELIGIBLE_AGENT"
    DETERMINISTIC_SELECTION = "DETERMINISTIC_SELECTION"
    SEMANTIC_MULTI_AGENT = "SEMANTIC_MULTI_AGENT"
    UNRESOLVED = "UNRESOLVED"
    NO_ELIGIBLE_AGENT = "NO_ELIGIBLE_AGENT"
    NO_AVAILABLE_ELIGIBLE_AGENT = "NO_AVAILABLE_ELIGIBLE_AGENT"
    INVALID_SELECTION = "INVALID_SELECTION"


class SelectionConfidence(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    SUPPORTED = "SUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class SelectionFactorType(str, Enum):
    """Factors derived from real requirements, capabilities and runtime facts."""

    # requirement-derived task shape
    MUTATION_REQUIRED = "MUTATION_REQUIRED"
    READ_ONLY_TASK = "READ_ONLY_TASK"
    TEST_EXECUTION_REQUIRED = "TEST_EXECUTION_REQUIRED"
    REPOSITORY_SCOPE_REQUIRED = "REPOSITORY_SCOPE_REQUIRED"
    NO_REPOSITORY_ACCESS_REQUIRED = "NO_REPOSITORY_ACCESS_REQUIRED"
    LONG_RUNNING_JOB_REQUIRED = "LONG_RUNNING_JOB_REQUIRED"
    AMBIGUOUS_REQUIREMENTS = "AMBIGUOUS_REQUIREMENTS"
    # agent-side operational fit
    IMPLEMENTATION_SUPPORT = "IMPLEMENTATION_SUPPORT"
    TEST_EXECUTION_SUPPORT = "TEST_EXECUTION_SUPPORT"
    LONG_RUNNING_JOB_SUPPORT = "LONG_RUNNING_JOB_SUPPORT"
    STRUCTURAL_READ_ONLY_GUARANTEE = "STRUCTURAL_READ_ONLY_GUARANTEE"
    LOCAL_EXECUTION_NO_REMOTE_SIDE_EFFECT = "LOCAL_EXECUTION_NO_REMOTE_SIDE_EFFECT"
    EXISTING_REUSABLE_SESSION = "EXISTING_REUSABLE_SESSION"
    PROJECT_AFFINITY = "PROJECT_AFFINITY"
    EXPLICIT_REQUEST_REQUIRED_BY_POLICY = "EXPLICIT_REQUEST_REQUIRED_BY_POLICY"


class SelectionFactorSource(str, Enum):
    REQUIREMENT_FACT = "REQUIREMENT_FACT"
    CAPABILITY_FACT = "CAPABILITY_FACT"
    RUNTIME_FACT = "RUNTIME_FACT"
    SESSION_FACT = "SESSION_FACT"
    PROJECT_FACT = "PROJECT_FACT"
    POLICY_FACT = "POLICY_FACT"


class SelectionFactorStrength(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    OPERATIONAL = "OPERATIONAL"
    SUPPORTING = "SUPPORTING"


class SelectionFactorPolarity(str, Enum):
    SUPPORT = "SUPPORT"
    EXCLUDE = "EXCLUDE"


@dataclass(frozen=True)
class SelectionFactor:
    type: SelectionFactorType
    source: SelectionFactorSource
    applies_to: Agent | None
    evidence: tuple[str, ...]
    strength: SelectionFactorStrength
    polarity: SelectionFactorPolarity = SelectionFactorPolarity.SUPPORT

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("a selection factor requires evidence")

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "source": self.source.value,
            "applies_to": self.applies_to.value if self.applies_to else None,
            "evidence": list(self.evidence),
            "strength": self.strength.value,
            "polarity": self.polarity.value,
        }


@dataclass(frozen=True)
class SelectionPolicy:
    """Operational policy facts that constrain automatic candidacy."""

    deepseek_auto_escalation: bool = False

    def agents_requiring_explicit_request(self) -> tuple[Agent, ...]:
        return () if self.deepseek_auto_escalation else (Agent.DEEPSEEK,)


@dataclass(frozen=True)
class OperationalFacts:
    """Read-only snapshot of runtime facts; nothing here is resolved lazily."""

    reusable_codex_session: bool = False
    codex_project_affinity: bool = False
    deepseek_project_session: bool = False
    project_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reusable_codex_session": self.reusable_codex_session,
            "codex_project_affinity": self.codex_project_affinity,
            "deepseek_project_session": self.deepseek_project_session,
            "project_id": self.project_id,
        }


@dataclass(frozen=True)
class SelectionProjectContext:
    """Deterministic project facts only; building it costs zero inference."""

    project_id: str | None = None
    languages: tuple[str, ...] = ()
    test_roots: tuple[str, ...] = ()
    git_branch: str | None = None
    modified_file_count: int = 0

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> "SelectionProjectContext":
        return cls(
            getattr(snapshot, "project_id", None),
            tuple(getattr(snapshot, "languages", ()) or ()),
            tuple(getattr(snapshot, "test_roots", ()) or ()),
            getattr(snapshot, "git_branch", None),
            len(tuple(getattr(snapshot, "modified_files", ()) or ())),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "languages": list(self.languages),
            "test_roots": list(self.test_roots),
            "git_branch": self.git_branch,
            "modified_file_count": self.modified_file_count,
        }


@dataclass(frozen=True)
class AgentSelectionProfile:
    """Preference-relevant facts for one agent, derived per task."""

    agent: Agent
    factors: tuple[SelectionFactor, ...]

    @property
    def support_factors(self) -> tuple[SelectionFactor, ...]:
        return tuple(
            item for item in self.factors
            if item.polarity is SelectionFactorPolarity.SUPPORT
        )

    @property
    def exclusion_factors(self) -> tuple[SelectionFactor, ...]:
        return tuple(
            item for item in self.factors
            if item.polarity is SelectionFactorPolarity.EXCLUDE
        )

    @property
    def justified(self) -> bool:
        return bool(self.support_factors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value,
            "factors": [item.as_dict() for item in self.factors],
            "justified": self.justified,
        }


def _requirement_true(requirements: GroundedTaskRequirements, name: str) -> bool:
    return requirements.requirements[name].value is RequirementValue.TRUE


def _requirement_refs(requirements: GroundedTaskRequirements, name: str) -> tuple[str, ...]:
    item = requirements.requirements[name]
    return tuple(f"requirement:{item.name}={item.value.value}:{ref}" for ref in item.evidence_refs)


def _capability_refs(profile: AgentCapabilityProfile, capabilities: Sequence[Capability]) -> tuple[str, ...]:
    wanted = frozenset(capabilities)
    refs = tuple(
        f"capability:{item.capability.value}:{item.source_kind}:{item.source}"
        for item in profile.evidence
        if item.capability in wanted
    )
    return refs or tuple(
        f"capability:{item.value}:profile" for item in capabilities if profile.has(item)
    )


def derive_task_factors(requirements: GroundedTaskRequirements) -> tuple[SelectionFactor, ...]:
    """Task-shape factors; every one of them cites requirement evidence."""

    factors: list[SelectionFactor] = []

    def add(kind: SelectionFactorType, dimension: str, strength: SelectionFactorStrength) -> None:
        factors.append(
            SelectionFactor(
                kind,
                SelectionFactorSource.REQUIREMENT_FACT,
                None,
                _requirement_refs(requirements, dimension),
                strength,
            )
        )

    if _requirement_true(requirements, MUTATION_REQUIRED):
        add(SelectionFactorType.MUTATION_REQUIRED, MUTATION_REQUIRED, SelectionFactorStrength.DETERMINISTIC)
    if _requirement_true(requirements, READ_ONLY_REQUIRED):
        add(SelectionFactorType.READ_ONLY_TASK, READ_ONLY_REQUIRED, SelectionFactorStrength.DETERMINISTIC)
    if _requirement_true(requirements, Capability.TEST_EXECUTION.value):
        add(
            SelectionFactorType.TEST_EXECUTION_REQUIRED,
            Capability.TEST_EXECUTION.value,
            SelectionFactorStrength.DETERMINISTIC,
        )
    elif requirements.tests_requested:
        factors.append(
            SelectionFactor(
                SelectionFactorType.TEST_EXECUTION_REQUIRED,
                SelectionFactorSource.REQUIREMENT_FACT,
                None,
                tuple(f"tests_requested:{item}" for item in requirements.tests_requested),
                SelectionFactorStrength.DETERMINISTIC,
            )
        )
    repository_dimensions = tuple(
        item.value for item in REPOSITORY_CAPABILITIES if _requirement_true(requirements, item.value)
    )
    if repository_dimensions or requirements.expected_files:
        evidence = tuple(
            ref for name in repository_dimensions for ref in _requirement_refs(requirements, name)
        ) + tuple(f"expected_file:{item}" for item in requirements.expected_files)
        factors.append(
            SelectionFactor(
                SelectionFactorType.REPOSITORY_SCOPE_REQUIRED,
                SelectionFactorSource.REQUIREMENT_FACT,
                None,
                evidence,
                SelectionFactorStrength.DETERMINISTIC,
            )
        )
    else:
        factors.append(
            SelectionFactor(
                SelectionFactorType.NO_REPOSITORY_ACCESS_REQUIRED,
                SelectionFactorSource.REQUIREMENT_FACT,
                None,
                tuple(
                    f"requirement:{item.value}=FALSE_OR_UNKNOWN"
                    for item in REPOSITORY_CAPABILITIES
                ),
                SelectionFactorStrength.DETERMINISTIC,
            )
        )
    if _requirement_true(requirements, Capability.LONG_RUNNING_JOB.value):
        add(
            SelectionFactorType.LONG_RUNNING_JOB_REQUIRED,
            Capability.LONG_RUNNING_JOB.value,
            SelectionFactorStrength.DETERMINISTIC,
        )
    unresolved = tuple(
        name for name in requirements.unknown_dimensions
        if name in {MUTATION_REQUIRED, READ_ONLY_REQUIRED}
    )
    if requirements.ambiguity_material or unresolved or requirements.conflict_dimensions:
        evidence = ("requirement:ambiguity_material=TRUE",) if requirements.ambiguity_material else ()
        evidence += tuple(f"requirement:{name}=UNKNOWN" for name in unresolved)
        evidence += tuple(f"requirement:{name}=CONFLICT" for name in requirements.conflict_dimensions)
        factors.append(
            SelectionFactor(
                SelectionFactorType.AMBIGUOUS_REQUIREMENTS,
                SelectionFactorSource.REQUIREMENT_FACT,
                None,
                evidence,
                SelectionFactorStrength.DETERMINISTIC,
            )
        )
    return tuple(factors)


class AgentSelectionProfileBuilder:
    """Build per-task selection profiles from facts that already exist.

    A capability by itself is never a preference. Only a real requirement match,
    a structural guarantee, a persisted operational fact or a policy fact becomes
    a factor.
    """

    def build(
        self,
        requirements: GroundedTaskRequirements,
        *,
        agents: Sequence[Agent],
        profiles: Mapping[Agent, AgentCapabilityProfile],
        policy: SelectionPolicy,
        operational: OperationalFacts,
        task_factors: Sequence[SelectionFactor] | None = None,
    ) -> dict[Agent, AgentSelectionProfile]:
        shape = {item.type for item in (task_factors or derive_task_factors(requirements))}
        explicit_only = frozenset(policy.agents_requiring_explicit_request())
        result: dict[Agent, AgentSelectionProfile] = {}
        for agent in agents:
            profile = profiles[agent]
            factors: list[SelectionFactor] = []
            if SelectionFactorType.MUTATION_REQUIRED in shape and all(
                profile.has(item) for item in (Capability.CODE_EDIT, Capability.MUTATION)
            ):
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.IMPLEMENTATION_SUPPORT,
                        SelectionFactorSource.CAPABILITY_FACT,
                        agent,
                        _capability_refs(profile, (Capability.CODE_EDIT, Capability.MUTATION)),
                        SelectionFactorStrength.DETERMINISTIC,
                    )
                )
            if SelectionFactorType.TEST_EXECUTION_REQUIRED in shape and profile.has(
                Capability.TEST_EXECUTION
            ):
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.TEST_EXECUTION_SUPPORT,
                        SelectionFactorSource.CAPABILITY_FACT,
                        agent,
                        _capability_refs(profile, (Capability.TEST_EXECUTION,)),
                        SelectionFactorStrength.DETERMINISTIC,
                    )
                )
            if SelectionFactorType.LONG_RUNNING_JOB_REQUIRED in shape and profile.has(
                Capability.LONG_RUNNING_JOB
            ):
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.LONG_RUNNING_JOB_SUPPORT,
                        SelectionFactorSource.CAPABILITY_FACT,
                        agent,
                        _capability_refs(profile, (Capability.LONG_RUNNING_JOB,)),
                        SelectionFactorStrength.DETERMINISTIC,
                    )
                )
            if SelectionFactorType.READ_ONLY_TASK in shape and not any(
                profile.has(item) for item in WRITE_CAPABILITIES
            ):
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.STRUCTURAL_READ_ONLY_GUARANTEE,
                        SelectionFactorSource.CAPABILITY_FACT,
                        agent,
                        tuple(f"capability_absent:{item.value}" for item in WRITE_CAPABILITIES),
                        SelectionFactorStrength.DETERMINISTIC,
                    )
                )
            if agent is Agent.LOCAL:
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.LOCAL_EXECUTION_NO_REMOTE_SIDE_EFFECT,
                        SelectionFactorSource.RUNTIME_FACT,
                        agent,
                        ("runtime:in_process_tools", "runtime:no_remote_job_or_session_created"),
                        SelectionFactorStrength.OPERATIONAL,
                    )
                )
            repository_task = SelectionFactorType.REPOSITORY_SCOPE_REQUIRED in shape
            if repository_task and agent is Agent.CODEX and operational.reusable_codex_session:
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.EXISTING_REUSABLE_SESSION,
                        SelectionFactorSource.SESSION_FACT,
                        agent,
                        ("session_registry:reusable_codex_session=true",),
                        SelectionFactorStrength.OPERATIONAL,
                    )
                )
            if repository_task and agent is Agent.CODEX and operational.codex_project_affinity:
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.PROJECT_AFFINITY,
                        SelectionFactorSource.PROJECT_FACT,
                        agent,
                        ("session_registry:codex_project_affinity=true",),
                        SelectionFactorStrength.OPERATIONAL,
                    )
                )
            if agent is Agent.DEEPSEEK and operational.deepseek_project_session:
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.PROJECT_AFFINITY,
                        SelectionFactorSource.SESSION_FACT,
                        agent,
                        ("deepseek_sessions:project_session=true",),
                        SelectionFactorStrength.OPERATIONAL,
                    )
                )
            if agent in explicit_only:
                factors.append(
                    SelectionFactor(
                        SelectionFactorType.EXPLICIT_REQUEST_REQUIRED_BY_POLICY,
                        SelectionFactorSource.POLICY_FACT,
                        agent,
                        ("policy:deepseek_auto_escalation=false",),
                        SelectionFactorStrength.DETERMINISTIC,
                        SelectionFactorPolarity.EXCLUDE,
                    )
                )
            result[agent] = AgentSelectionProfile(agent, tuple(factors))
        return result


@dataclass(frozen=True)
class SemanticSelectionOutcome:
    proposed_agent: Agent | None
    factors: tuple[SelectionFactorType, ...]
    uncertainty: str
    reason_code: str
    latency_ms: float
    calls: int
    valid: bool
    failure_code: str | None = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AgentSelectionProposal:
    requested_agent: Agent | None
    eligible_agents: tuple[Agent, ...]
    available_eligible_agents: tuple[Agent, ...]
    candidate_agents: tuple[Agent, ...]
    proposed_agent: Agent | None
    selection_source: SelectionSource
    confidence: SelectionConfidence
    reason_code: str
    factors: tuple[SelectionFactor, ...]
    task_factors: tuple[SelectionFactor, ...]
    profiles: Mapping[Agent, AgentSelectionProfile] = field(repr=False, default_factory=dict)
    eligible_but_unavailable: tuple[Agent, ...] = ()
    excluded_by_policy: tuple[Agent, ...] = ()
    execution_possible: bool = False
    model_calls: int = 0
    model_latency_ms: float = 0.0
    deterministic_latency_ms: float = 0.0
    errors: tuple[str, ...] = ()
    session_resolved: bool = False

    # Dry-run invariants: selection never authorizes or performs anything.
    execution_authorized: bool = field(default=False, init=False)
    dry_run: bool = field(default=True, init=False)
    jobs_created: int = field(default=0, init=False)
    delegations: int = field(default=0, init=False)
    filesystem_mutations: int = field(default=0, init=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "eligible_agents": [item.value for item in self.eligible_agents],
            "available_eligible_agents": [item.value for item in self.available_eligible_agents],
            "candidate_agents": [item.value for item in self.candidate_agents],
            "eligible_but_unavailable": [item.value for item in self.eligible_but_unavailable],
            "excluded_by_policy": [item.value for item in self.excluded_by_policy],
            "proposed_agent": self.proposed_agent.value if self.proposed_agent else None,
            "selection_source": self.selection_source.value,
            "confidence": self.confidence.value,
            "reason_code": self.reason_code,
            "factors": [item.as_dict() for item in self.factors],
            "task_factors": [item.as_dict() for item in self.task_factors],
            "selection_profiles": {
                agent.value: profile.as_dict() for agent, profile in self.profiles.items()
            },
            "execution_possible": self.execution_possible,
            "execution_authorized": False,
            "dry_run": True,
            "jobs_created": 0,
            "delegations": 0,
            "filesystem_mutations": 0,
            "session_resolved": self.session_resolved,
            "model_calls": self.model_calls,
            "model_latency_ms": self.model_latency_ms,
            "deterministic_latency_ms": self.deterministic_latency_ms,
            "errors": list(self.errors),
        }


def selection_json_schema(candidates: Sequence[Agent], factors: Sequence[SelectionFactorType]) -> dict[str, Any]:
    """Constrained output: the model can only name an allowed candidate."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposed_agent": {
                "type": "string",
                "enum": [item.value for item in candidates] + ["UNRESOLVED"],
            },
            "selection_factors": {
                "type": "array",
                "maxItems": 4,
                "items": {"type": "string", "enum": [item.value for item in factors]},
            },
            "uncertainty": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
            "reason_code": {
                "type": "string",
                "enum": [
                    "BEST_FACTOR_FIT",
                    "EQUAL_FIT",
                    "INSUFFICIENT_BASIS",
                ],
            },
        },
        "required": ["proposed_agent", "selection_factors", "uncertainty", "reason_code"],
    }


class SemanticAgentSelector:
    """One bounded inference over facts only; never sees the raw task text."""

    system_prompt = (
        "Choose the most appropriate executor among the listed candidates only. "
        "Decide from grounded requirements, capability facts, selection factors, "
        "availability and project facts. Never name an agent outside the candidate "
        "list. Answer UNRESOLVED when the facts do not support one candidate over "
        "the others. Do not execute, plan or explain in prose."
    )

    def __init__(self, runtime: Any, *, max_tokens: int = 160, temperature: float = 0.0):
        self.runtime = runtime
        self.max_tokens = max_tokens
        self.temperature = temperature

    def select(
        self,
        requirements: GroundedTaskRequirements,
        *,
        candidates: Sequence[Agent],
        profiles: Mapping[Agent, AgentSelectionProfile],
        capability_profiles: Mapping[Agent, AgentCapabilityProfile],
        availability: Mapping[Agent, AgentRuntimeAvailability],
        task_factors: Sequence[SelectionFactor],
        project_context: SelectionProjectContext | None,
    ) -> SemanticSelectionOutcome:
        allowed = tuple(candidates)
        factor_types = tuple(
            dict.fromkeys(
                [item.type for item in task_factors]
                + [
                    factor.type
                    for agent in allowed
                    for factor in profiles[agent].support_factors
                ]
            )
        )
        schema = selection_json_schema(allowed, factor_types or tuple(SelectionFactorType))
        payload = {
            "grounded_requirements": {
                name: item.value.value for name, item in sorted(requirements.requirements.items())
            },
            "target_scope": requirements.target_scope,
            "risk_level": requirements.risk_level.value,
            "prohibitions": list(requirements.prohibitions),
            "tests_requested": list(requirements.tests_requested),
            "task_factors": [item.type.value for item in task_factors],
            "candidates": [
                {
                    "agent": agent.value,
                    "capabilities": sorted(item.value for item in capability_profiles[agent].capabilities),
                    "selection_factors": [
                        {
                            "type": factor.type.value,
                            "source": factor.source.value,
                            "strength": factor.strength.value,
                            "evidence": list(factor.evidence[:3]),
                        }
                        for factor in profiles[agent].support_factors
                    ],
                    "runtime_available": availability[agent].available,
                }
                for agent in allowed
            ],
            "project_context": project_context.as_dict() if project_context else None,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        started = time.perf_counter()
        try:
            result = self.runtime.generate_structured(
                messages,
                schema=schema,
                schema_name="agent_selection_proposal",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # transport, schema, truncation, validator
            code = getattr(exc, "code", None)
            return SemanticSelectionOutcome(
                None,
                (),
                "HIGH",
                "MODEL_PARSE_FAILURE",
                round((time.perf_counter() - started) * 1000, 3),
                1,
                False,
                getattr(code, "value", None) or type(exc).__name__,
            )
        latency = getattr(getattr(result, "observation", None), "latency_ms", None)
        latency_ms = float(latency) if latency is not None else round((time.perf_counter() - started) * 1000, 3)
        raw = result.parsed if result.parsed is not None else json.loads(result.content)
        name = str(raw["proposed_agent"])
        agent = None if name == "UNRESOLVED" else Agent(name)
        factors = tuple(
            dict.fromkeys(SelectionFactorType(item) for item in raw.get("selection_factors") or ())
        )
        return SemanticSelectionOutcome(
            agent,
            factors,
            str(raw.get("uncertainty") or "HIGH"),
            str(raw.get("reason_code") or "EQUAL_FIT"),
            latency_ms,
            1,
            True,
            None,
            raw,
        )


class AgentSelectionEngine:
    """Deterministic precedence first; one bounded inference only when needed.

    Precedence: explicit user agent, single eligible agent, only available
    eligible agent, deterministic selection fact, semantic multi-agent
    selection, unresolved.
    """

    def __init__(
        self,
        *,
        policy: SelectionPolicy | None = None,
        semantic_selector: SemanticAgentSelector | None = None,
        profile_builder: AgentSelectionProfileBuilder | None = None,
    ):
        self.policy = policy or SelectionPolicy()
        self.semantic_selector = semantic_selector
        self.profile_builder = profile_builder or AgentSelectionProfileBuilder()

    def propose(
        self,
        requirements: GroundedTaskRequirements,
        evaluations: Mapping[Agent, GroundedAgentEligibility],
        *,
        capability_profiles: Mapping[Agent, AgentCapabilityProfile],
        availability: Mapping[Agent, AgentRuntimeAvailability],
        operational: OperationalFacts | None = None,
        project_context: SelectionProjectContext | None = None,
        requested_agent: Agent | None = None,
    ) -> AgentSelectionProposal:
        started = time.perf_counter()
        operational = operational or OperationalFacts()
        requested = requested_agent if requested_agent is not None else requirements.requested_agent
        eligible = tuple(
            agent for agent in evaluations if evaluations[agent].eligible
        )
        available_eligible = tuple(
            agent for agent in eligible if availability[agent].available
        )
        unavailable = tuple(agent for agent in eligible if agent not in available_eligible)
        task_factors = derive_task_factors(requirements)
        profiles = self.profile_builder.build(
            requirements,
            agents=tuple(evaluations),
            profiles=capability_profiles,
            policy=self.policy,
            operational=operational,
            task_factors=task_factors,
        )

        def finish(
            proposed: Agent | None,
            source: SelectionSource,
            confidence: SelectionConfidence,
            reason: str,
            *,
            candidates: tuple[Agent, ...] = (),
            factors: tuple[SelectionFactor, ...] = (),
            excluded: tuple[Agent, ...] = (),
            semantic: SemanticSelectionOutcome | None = None,
            errors: tuple[str, ...] = (),
        ) -> AgentSelectionProposal:
            execution_possible = bool(
                proposed is not None
                and proposed in eligible
                and availability[proposed].available
                and evaluations[proposed].executable_now
            )
            model_calls = semantic.calls if semantic else 0
            model_latency = semantic.latency_ms if semantic else 0.0
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return AgentSelectionProposal(
                requested_agent=requested,
                eligible_agents=eligible,
                available_eligible_agents=available_eligible,
                candidate_agents=candidates,
                proposed_agent=proposed,
                selection_source=source,
                confidence=confidence,
                reason_code=reason,
                factors=factors,
                task_factors=task_factors,
                profiles=profiles,
                eligible_but_unavailable=unavailable,
                excluded_by_policy=excluded,
                execution_possible=execution_possible,
                model_calls=model_calls,
                model_latency_ms=model_latency,
                deterministic_latency_ms=round(max(elapsed - model_latency, 0.0), 3),
                errors=errors,
            )

        # 1. explicit user agent stays sovereign; no inference, no substitution.
        if requested is not None:
            evaluation = evaluations.get(requested)
            if evaluation is None:
                return finish(
                    requested,
                    SelectionSource.EXPLICIT_USER,
                    SelectionConfidence.DETERMINISTIC,
                    "REQUESTED_AGENT_UNKNOWN",
                )
            reason = (
                "EXPLICIT_AGENT_READY"
                if evaluation.executable_now and availability[requested].available
                else "REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS"
                if not evaluation.eligible
                else "REQUESTED_AGENT_UNAVAILABLE"
                if not availability[requested].available
                else "REQUESTED_AGENT_EXECUTION_BLOCKED"
            )
            return finish(
                requested,
                SelectionSource.EXPLICIT_USER,
                SelectionConfidence.DETERMINISTIC,
                reason,
                candidates=(requested,),
                factors=tuple(profiles[requested].support_factors),
            )

        # 2. no eligible agent: never invent one, never fall back.
        if not eligible:
            return finish(
                None,
                SelectionSource.NO_ELIGIBLE_AGENT,
                SelectionConfidence.DETERMINISTIC,
                "NO_ELIGIBLE_AGENT",
            )

        # 3. single eligible agent: no probabilistic decision exists.
        if len(eligible) == 1:
            only = eligible[0]
            return finish(
                only,
                SelectionSource.SINGLE_ELIGIBLE_AGENT,
                SelectionConfidence.DETERMINISTIC,
                "SINGLE_ELIGIBLE_AGENT"
                if availability[only].available
                else "SINGLE_ELIGIBLE_AGENT_UNAVAILABLE",
                candidates=(only,),
                factors=tuple(profiles[only].support_factors),
            )

        # 4. availability is separate and never rewrites eligibility.
        if not available_eligible:
            return finish(
                None,
                SelectionSource.NO_AVAILABLE_ELIGIBLE_AGENT,
                SelectionConfidence.DETERMINISTIC,
                "NO_AVAILABLE_ELIGIBLE_AGENT",
            )
        if len(available_eligible) == 1:
            only = available_eligible[0]
            return finish(
                only,
                SelectionSource.ONLY_AVAILABLE_ELIGIBLE_AGENT,
                SelectionConfidence.DETERMINISTIC,
                "ONLY_AVAILABLE_ELIGIBLE_AGENT",
                candidates=(only,),
                factors=tuple(profiles[only].support_factors),
            )

        # 5. deterministic selection facts.
        ambiguous = any(
            item.type is SelectionFactorType.AMBIGUOUS_REQUIREMENTS for item in task_factors
        )
        if ambiguous:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "AMBIGUOUS_REQUIREMENTS",
                candidates=available_eligible,
            )
        excluded = tuple(
            agent for agent in available_eligible if profiles[agent].exclusion_factors
        )
        candidates = tuple(agent for agent in available_eligible if agent not in excluded)
        if not candidates:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "ALL_CANDIDATES_REQUIRE_EXPLICIT_REQUEST",
                candidates=(),
                excluded=excluded,
            )
        if len(candidates) == 1:
            only = candidates[0]
            return finish(
                only,
                SelectionSource.DETERMINISTIC_SELECTION,
                SelectionConfidence.DETERMINISTIC,
                "POLICY_RESTRICTED_AUTOMATIC_CANDIDATES",
                candidates=candidates,
                factors=tuple(profiles[only].support_factors),
                excluded=excluded,
            )
        justified = tuple(agent for agent in candidates if profiles[agent].justified)
        if not justified:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "NO_JUSTIFIED_CANDIDATE",
                candidates=candidates,
                excluded=excluded,
            )
        if len(justified) == 1:
            only = justified[0]
            return finish(
                only,
                SelectionSource.DETERMINISTIC_SELECTION,
                SelectionConfidence.DETERMINISTIC,
                "UNIQUE_JUSTIFIED_CANDIDATE",
                candidates=candidates,
                factors=tuple(profiles[only].support_factors),
                excluded=excluded,
            )

        # 6. genuine multi-agent decision: at most one bounded inference.
        if self.semantic_selector is None:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "SEMANTIC_SELECTION_UNAVAILABLE",
                candidates=justified,
                excluded=excluded,
            )
        outcome = self.semantic_selector.select(
            requirements,
            candidates=justified,
            profiles=profiles,
            capability_profiles=capability_profiles,
            availability=availability,
            task_factors=task_factors,
            project_context=project_context,
        )
        if not outcome.valid:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "MODEL_PARSE_FAILURE",
                candidates=justified,
                excluded=excluded,
                semantic=outcome,
                errors=(outcome.failure_code or "MODEL_PARSE_FAILURE",),
            )
        if outcome.proposed_agent is None:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "SEMANTIC_UNRESOLVED",
                candidates=justified,
                excluded=excluded,
                semantic=outcome,
            )
        proposed = outcome.proposed_agent
        # Hard invariant: no silent substitution when the pick is not allowed.
        if proposed not in justified:
            return finish(
                None,
                SelectionSource.INVALID_SELECTION,
                SelectionConfidence.UNRESOLVED,
                "PROPOSED_AGENT_OUTSIDE_CANDIDATE_SET",
                candidates=justified,
                excluded=excluded,
                semantic=outcome,
                errors=("INELIGIBLE_AGENT_SELECTED",),
            )
        actual = tuple(profiles[proposed].support_factors)
        if not actual:
            return finish(
                None,
                SelectionSource.UNRESOLVED,
                SelectionConfidence.UNRESOLVED,
                "UNJUSTIFIED_SEMANTIC_SELECTION",
                candidates=justified,
                excluded=excluded,
                semantic=outcome,
                errors=("UNJUSTIFIED_SELECTION",),
            )
        actual_types = {item.type for item in actual} | {item.type for item in task_factors}
        bad = tuple(item.value for item in outcome.factors if item not in actual_types)
        stronger = max(
            (len(profiles[agent].support_factors) for agent in justified if agent is not proposed),
            default=0,
        )
        confidence = (
            SelectionConfidence.SUPPORTED
            if outcome.uncertainty == "LOW"
            and outcome.reason_code == "BEST_FACTOR_FIT"
            and len(actual) > stronger
            else SelectionConfidence.AMBIGUOUS
        )
        return finish(
            proposed,
            SelectionSource.SEMANTIC_MULTI_AGENT,
            confidence,
            outcome.reason_code,
            candidates=justified,
            factors=actual,
            excluded=excluded,
            semantic=outcome,
            errors=tuple(f"BAD_SELECTION_FACTOR:{item}" for item in bad),
        )
