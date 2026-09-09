from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping


class Agent(str, Enum):
    LOCAL = "local"
    CODEX = "codex"
    DEEPSEEK = "deepseek"


class Capability(str, Enum):
    REPOSITORY_READ = "repository_read"
    REPOSITORY_WRITE = "repository_write"
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    CODE_ANALYSIS = "code_analysis"
    CODE_EDIT = "code_edit"
    TEST_EXECUTION = "test_execution"
    LONG_RUNNING_JOB = "long_running_job"
    PERSISTENT_SESSION = "persistent_session"
    GENERAL_REASONING = "general_reasoning"
    CODE_REVIEW = "code_review"
    WEB_ACCESS = "web_access"
    MUTATION = "mutation_capable"
    READ_ONLY = "read_only_capable"
    SEMANTIC_INTERPRETATION = "semantic_interpretation"
    TASK_REQUIREMENT_EXTRACTION = "task_requirement_extraction"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class CapabilityEvidence:
    capability: Capability
    source_kind: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "capability": self.capability.value,
            "source_kind": self.source_kind,
            "source": self.source,
        }


@dataclass(frozen=True)
class AgentCapabilityProfile:
    agent: Agent
    capabilities: frozenset[Capability]
    evidence: tuple[CapabilityEvidence, ...]

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value,
            "capabilities": sorted(item.value for item in self.capabilities),
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class AgentRuntimeAvailability:
    agent: Agent
    available: bool
    enabled: bool
    configured: bool
    reason_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value,
            "available": self.available,
            "enabled": self.enabled,
            "configured": self.configured,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class PermissionProfile:
    agent: Agent
    execution_allowed: bool = True
    denied_capabilities: frozenset[Capability] = frozenset()
    reason_code: str | None = None


@dataclass(frozen=True)
class CapabilityBaseline:
    profiles: Mapping[Agent, AgentCapabilityProfile]
    availability: Mapping[Agent, AgentRuntimeAvailability]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profiles": {
                agent.value: profile.as_dict() for agent, profile in self.profiles.items()
            },
            "availability": {
                agent.value: state.as_dict() for agent, state in self.availability.items()
            },
        }


class CapabilityProfileBuilder:
    """Derive executor profiles from registered tools and runtime facts.

    Tool presence establishes capability. Runtime health is recorded separately
    and never changes the capability set.
    """

    _LOCAL_TOOL_CAPABILITIES: Mapping[str, tuple[Capability, ...]] = {
        "filesystem_list": (Capability.FILESYSTEM_READ, Capability.REPOSITORY_READ),
        "filesystem_read_text": (Capability.FILESYSTEM_READ, Capability.REPOSITORY_READ),
        "find_project_files": (Capability.FILESYSTEM_READ, Capability.REPOSITORY_READ),
        "get_project_git_state": (
            Capability.FILESYSTEM_READ,
            Capability.REPOSITORY_READ,
        ),
        "run_project_tests": (
            Capability.REPOSITORY_READ,
            Capability.TEST_EXECUTION,
        ),
        "filesystem_write_text": (
            Capability.FILESYSTEM_WRITE,
            Capability.REPOSITORY_WRITE,
            Capability.CODE_EDIT,
            Capability.MUTATION,
        ),
        "filesystem_delete": (
            Capability.FILESYSTEM_WRITE,
            Capability.REPOSITORY_WRITE,
            Capability.MUTATION,
        ),
        "web_search": (Capability.WEB_ACCESS,),
        "web_open": (Capability.WEB_ACCESS,),
        "web_extract": (Capability.WEB_ACCESS,),
    }

    _CODEX_CAPABILITIES = frozenset(
        {
            Capability.REPOSITORY_READ,
            Capability.REPOSITORY_WRITE,
            Capability.FILESYSTEM_READ,
            Capability.FILESYSTEM_WRITE,
            Capability.CODE_ANALYSIS,
            Capability.CODE_EDIT,
            Capability.TEST_EXECUTION,
            Capability.GENERAL_REASONING,
            Capability.CODE_REVIEW,
            Capability.MUTATION,
            Capability.READ_ONLY,
        }
    )

    _DEEPSEEK_CAPABILITIES = frozenset(
        {
            Capability.GENERAL_REASONING,
            Capability.CODE_ANALYSIS,
            Capability.CODE_REVIEW,
            Capability.READ_ONLY,
        }
    )

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        *,
        local_model_available: bool,
        codex_available: bool,
    ) -> CapabilityBaseline:
        names = frozenset(registry.names())
        profiles: dict[Agent, AgentCapabilityProfile] = {}

        local_capabilities: set[Capability] = set()
        local_evidence: list[CapabilityEvidence] = []
        for tool, capabilities in cls._LOCAL_TOOL_CAPABILITIES.items():
            if tool not in names:
                continue
            for capability in capabilities:
                local_capabilities.add(capability)
                local_evidence.append(CapabilityEvidence(capability, "registered_tool", tool))
        if local_model_available:
            for capability in (
                Capability.GENERAL_REASONING,
                Capability.CODE_ANALYSIS,
                Capability.CODE_REVIEW,
                Capability.READ_ONLY,
                Capability.SEMANTIC_INTERPRETATION,
                Capability.TASK_REQUIREMENT_EXTRACTION,
            ):
                local_capabilities.add(capability)
                local_evidence.append(
                    CapabilityEvidence(capability, "runtime_fact", "local_model_healthy")
                )
        profiles[Agent.LOCAL] = AgentCapabilityProfile(
            Agent.LOCAL, frozenset(local_capabilities), tuple(local_evidence)
        )

        codex_evidence: list[CapabilityEvidence] = []
        codex_capabilities: set[Capability] = set()
        if "delegate_to_codex" in names:
            codex_capabilities.update(cls._CODEX_CAPABILITIES)
            codex_evidence.extend(
                CapabilityEvidence(item, "registered_tool", "delegate_to_codex")
                for item in cls._CODEX_CAPABILITIES
            )
        if "get_codex_job_status" in names:
            codex_capabilities.add(Capability.LONG_RUNNING_JOB)
            codex_evidence.append(
                CapabilityEvidence(
                    Capability.LONG_RUNNING_JOB,
                    "registered_tool",
                    "get_codex_job_status",
                )
            )
        sessions = getattr(getattr(registry, "codex", None), "sessions", None)
        if sessions is not None:
            codex_capabilities.add(Capability.PERSISTENT_SESSION)
            codex_evidence.append(
                CapabilityEvidence(
                    Capability.PERSISTENT_SESSION,
                    "manager_fact",
                    type(sessions).__name__,
                )
            )
        profiles[Agent.CODEX] = AgentCapabilityProfile(
            Agent.CODEX, frozenset(codex_capabilities), tuple(codex_evidence)
        )

        deepseek_evidence: list[CapabilityEvidence] = []
        deepseek_capabilities: set[Capability] = set()
        if "delegate_to_deepseek" in names:
            deepseek_capabilities.update(cls._DEEPSEEK_CAPABILITIES)
            deepseek_evidence.extend(
                CapabilityEvidence(item, "registered_tool", "delegate_to_deepseek")
                for item in cls._DEEPSEEK_CAPABILITIES
            )
        if "review_deepseek_session" in names:
            deepseek_capabilities.add(Capability.PERSISTENT_SESSION)
            deepseek_evidence.append(
                CapabilityEvidence(
                    Capability.PERSISTENT_SESSION,
                    "registered_tool",
                    "review_deepseek_session",
                )
            )
        profiles[Agent.DEEPSEEK] = AgentCapabilityProfile(
            Agent.DEEPSEEK, frozenset(deepseek_capabilities), tuple(deepseek_evidence)
        )

        deepseek = getattr(registry, "deepseek", None)
        deepseek_client = getattr(deepseek, "client", None)
        deepseek_enabled = bool(getattr(deepseek_client, "enabled", False))
        deepseek_configured = bool(getattr(deepseek_client, "configured", False))
        deepseek_registered = "delegate_to_deepseek" in names
        deepseek_available = deepseek_registered and deepseek_enabled and deepseek_configured
        availability = {
            Agent.LOCAL: AgentRuntimeAvailability(
                Agent.LOCAL,
                local_model_available,
                True,
                True,
                None if local_model_available else "LOCAL_MODEL_UNAVAILABLE",
            ),
            Agent.CODEX: AgentRuntimeAvailability(
                Agent.CODEX,
                "delegate_to_codex" in names and codex_available,
                "delegate_to_codex" in names,
                bool(getattr(registry, "codex", None)),
                None
                if "delegate_to_codex" in names and codex_available
                else "CODEX_RUNTIME_UNAVAILABLE",
            ),
            Agent.DEEPSEEK: AgentRuntimeAvailability(
                Agent.DEEPSEEK,
                deepseek_available,
                deepseek_enabled,
                deepseek_configured,
                None
                if deepseek_available
                else (
                    "DEEPSEEK_TOOL_NOT_REGISTERED"
                    if not deepseek_registered
                    else "DEEPSEEK_DISABLED"
                    if not deepseek_enabled
                    else "DEEPSEEK_NOT_CONFIGURED"
                ),
            ),
        }
        return CapabilityBaseline(profiles, availability)


@dataclass(frozen=True)
class TaskRequirements:
    capabilities: frozenset[Capability]
    mutation_required: bool
    read_only_required: bool
    target_scope: str
    risk_level: RiskLevel
    expected_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()
    tests_requested: tuple[str, ...] = ()
    ambiguity_material: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskRequirements":
        capabilities = frozenset(Capability(item) for item in value["capabilities"])
        return cls(
            capabilities=capabilities,
            mutation_required=bool(value["mutation_required"]),
            read_only_required=bool(value["read_only_required"]),
            target_scope=str(value["target_scope"]),
            risk_level=RiskLevel(value["risk_level"]),
            expected_files=tuple(value.get("expected_files") or ()),
            forbidden_files=tuple(value.get("forbidden_files") or ()),
            tests_requested=tuple(value.get("tests_requested") or ()),
            ambiguity_material=bool(value.get("ambiguity_material", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "capabilities": sorted(item.value for item in self.capabilities),
            "mutation_required": self.mutation_required,
            "read_only_required": self.read_only_required,
            "target_scope": self.target_scope,
            "risk_level": self.risk_level.value,
            "expected_files": list(self.expected_files),
            "forbidden_files": list(self.forbidden_files),
            "tests_requested": list(self.tests_requested),
            "ambiguity_material": self.ambiguity_material,
        }


def task_requirement_json_schema() -> dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}, "maxItems": 20}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "capabilities": {
                "type": "array",
                "items": {"type": "string", "enum": [item.value for item in Capability]},
                "uniqueItems": True,
                "maxItems": len(Capability),
            },
            "mutation_required": {"type": "boolean"},
            "read_only_required": {"type": "boolean"},
            "target_scope": {"type": "string", "maxLength": 500},
            "risk_level": {"type": "string", "enum": [item.value for item in RiskLevel]},
            "expected_files": string_array,
            "forbidden_files": string_array,
            "tests_requested": string_array,
            "ambiguity_material": {"type": "boolean"},
        },
        "required": [
            "capabilities",
            "mutation_required",
            "read_only_required",
            "target_scope",
            "risk_level",
            "expected_files",
            "forbidden_files",
            "tests_requested",
            "ambiguity_material",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "task_requirements", "strict": True, "schema": schema},
    }


@dataclass(frozen=True)
class RequirementAnalysisResult:
    requirements: TaskRequirements | None
    valid: bool
    first_pass_valid: bool
    attempts: int
    latency_ms: float
    finish_reason: str | None
    prompt_tokens: int | None
    generated_tokens: int | None
    error_code: str | None = None


class TaskRequirementAnalyzer:
    system_prompt = (
        "Derive only the capabilities and constraints required by the task. "
        "Do not choose, rank, recommend, or name any executor. Use only facts in "
        "the task and project snapshot. Mark material ambiguity instead of guessing."
    )

    def __init__(self, client: Any):
        self.client = client

    def analyze(
        self,
        task: str,
        *,
        project_snapshot: Mapping[str, Any] | None = None,
    ) -> RequirementAnalysisResult:
        started = time.perf_counter()
        schema = task_requirement_json_schema()
        payload = {"task": task, "project_snapshot": project_snapshot or None}
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        first_pass_valid = False
        last_error = "TASK_REQUIREMENT_INVALID"
        for attempt in range(1, 3):
            try:
                response = self.client.chat(
                    messages,
                    response_format=schema,
                    temperature=0.0,
                    max_tokens=512,
                )
            except Exception as exc:
                last_error = type(exc).__name__
                continue
            choice = (response.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            usage = response.get("usage") or {}
            try:
                raw = json.loads(choice["message"]["content"])
                requirements = TaskRequirements.from_dict(raw)
                first_pass_valid = attempt == 1
                return RequirementAnalysisResult(
                    requirements,
                    True,
                    first_pass_valid,
                    attempt,
                    round((time.perf_counter() - started) * 1000, 3),
                    finish_reason,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous object was invalid. Return one object matching "
                            f"the same schema exactly. Validation error: {last_error}."
                        ),
                    }
                )
        return RequirementAnalysisResult(
            None,
            False,
            False,
            2,
            round((time.perf_counter() - started) * 1000, 3),
            None,
            None,
            None,
            last_error,
        )


@dataclass(frozen=True)
class AgentEligibility:
    agent: Agent
    capability_eligible: bool
    permission_eligible: bool
    eligible: bool
    runtime_available: bool
    executable_now: bool
    missing_capabilities: tuple[Capability, ...]
    denied_capabilities: tuple[Capability, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.value,
            "capability_eligible": self.capability_eligible,
            "permission_eligible": self.permission_eligible,
            "eligible": self.eligible,
            "runtime_available": self.runtime_available,
            "executable_now": self.executable_now,
            "missing_capabilities": [item.value for item in self.missing_capabilities],
            "denied_capabilities": [item.value for item in self.denied_capabilities],
            "reason_codes": list(self.reason_codes),
        }


class EligibilityEngine:
    def evaluate(
        self,
        requirements: TaskRequirements,
        profiles: Mapping[Agent, AgentCapabilityProfile],
        availability: Mapping[Agent, AgentRuntimeAvailability],
        permissions: Mapping[Agent, PermissionProfile] | None = None,
    ) -> dict[Agent, AgentEligibility]:
        permissions = permissions or {}
        results: dict[Agent, AgentEligibility] = {}
        for agent, profile in profiles.items():
            permission = permissions.get(agent, PermissionProfile(agent))
            runtime = availability[agent]
            missing = tuple(sorted(requirements.capabilities - profile.capabilities, key=str))
            denied = tuple(
                sorted(requirements.capabilities.intersection(permission.denied_capabilities), key=str)
            )
            capability_eligible = not missing
            permission_eligible = permission.execution_allowed and not denied
            eligible = capability_eligible and permission_eligible
            executable_now = eligible and runtime.available
            prefix = agent.value.upper()
            reasons: list[str] = []
            reasons.extend(f"{prefix}_MISSING_{item.value.upper()}" for item in missing)
            reasons.extend(f"{prefix}_PERMISSION_DENIED_{item.value.upper()}" for item in denied)
            if not permission.execution_allowed:
                reasons.append(permission.reason_code or f"{prefix}_EXECUTION_NOT_ALLOWED")
            if eligible:
                reasons.append(f"{prefix}_ELIGIBLE")
            if not runtime.available:
                reasons.append(runtime.reason_code or f"{prefix}_UNAVAILABLE")
            results[agent] = AgentEligibility(
                agent,
                capability_eligible,
                permission_eligible,
                eligible,
                runtime.available,
                executable_now,
                missing,
                denied,
                tuple(reasons),
            )
        return results


@dataclass(frozen=True)
class AgentSelectionProposal:
    requested_agent: Agent | None
    selected_agent: Agent | None
    proposed_agent: Agent | None
    selection_source: str
    eligible_agents: tuple[Agent, ...]
    executable_agents: tuple[Agent, ...]
    execution_currently_possible: bool
    execution_authorized: bool
    dry_run: bool
    reason_code: str
    evaluations: Mapping[Agent, AgentEligibility] = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "selected_agent": self.selected_agent.value if self.selected_agent else None,
            "proposed_agent": self.proposed_agent.value if self.proposed_agent else None,
            "selection_source": self.selection_source,
            "eligible_agents": [item.value for item in self.eligible_agents],
            "executable_agents": [item.value for item in self.executable_agents],
            "execution_currently_possible": self.execution_currently_possible,
            "execution_authorized": False,
            "dry_run": True,
            "reason_code": self.reason_code,
            "evaluations": {
                agent.value: result.as_dict() for agent, result in self.evaluations.items()
            },
        }


def propose_agent_selection(
    evaluations: Mapping[Agent, AgentEligibility],
    *,
    requested_agent: Agent | None,
) -> AgentSelectionProposal:
    eligible = tuple(agent for agent, result in evaluations.items() if result.eligible)
    executable = tuple(agent for agent, result in evaluations.items() if result.executable_now)
    if requested_agent is not None:
        result = evaluations[requested_agent]
        reason = (
            "EXPLICIT_AGENT_READY"
            if result.executable_now
            else "REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS"
            if not result.eligible
            else "REQUESTED_AGENT_UNAVAILABLE"
        )
        return AgentSelectionProposal(
            requested_agent,
            requested_agent,
            None,
            "explicit_user",
            eligible,
            executable,
            result.executable_now,
            False,
            True,
            reason,
            evaluations,
        )
    if len(eligible) == 1:
        proposed = eligible[0]
        return AgentSelectionProposal(
            None,
            None,
            proposed,
            "deterministic_single_candidate",
            eligible,
            executable,
            evaluations[proposed].executable_now,
            False,
            True,
            "SINGLE_ELIGIBLE_AGENT",
            evaluations,
        )
    return AgentSelectionProposal(
        None,
        None,
        None,
        "none",
        eligible,
        executable,
        False,
        False,
        True,
        "NO_ELIGIBLE_AGENT" if not eligible else "MULTIPLE_ELIGIBLE_AGENTS",
        evaluations,
    )


@dataclass(frozen=True)
class VerificationExpectation:
    expected_files: frozenset[str] = frozenset()
    forbidden_files: frozenset[str] = frozenset()
    tests_requested: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationResult:
    expected_files_changed: tuple[str, ...]
    actual_files_changed: tuple[str, ...]
    tests_requested: tuple[str, ...]
    tests_executed: tuple[str, ...]
    test_exit_code: int | None
    forbidden_files_touched: tuple[str, ...]
    scope_violation: bool
    unexpected_mutation: bool
    objective_satisfied: bool | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def verify_facts(
    expectation: VerificationExpectation,
    *,
    actual_files_changed: Iterable[str],
    tests_executed: Iterable[str] = (),
    test_exit_code: int | None = None,
    objective_satisfied: bool | None = None,
) -> VerificationResult:
    actual = frozenset(actual_files_changed)
    forbidden = actual.intersection(expectation.forbidden_files)
    unexpected = bool(expectation.expected_files and actual - expectation.expected_files)
    expected_changed = actual.intersection(expectation.expected_files)
    tests = tuple(tests_executed)
    tests_missing = bool(expectation.tests_requested and not tests)
    deterministic_failure = bool(forbidden or unexpected or tests_missing or test_exit_code not in (None, 0))
    status = (
        "failed"
        if deterministic_failure or objective_satisfied is False
        else "passed"
        if objective_satisfied is True and not deterministic_failure
        else "inconclusive"
    )
    return VerificationResult(
        tuple(sorted(expected_changed)),
        tuple(sorted(actual)),
        expectation.tests_requested,
        tests,
        test_exit_code,
        tuple(sorted(forbidden)),
        bool(forbidden or unexpected),
        unexpected,
        objective_satisfied,
        status,
    )


@dataclass(frozen=True)
class AgentResultProvenance:
    agent: Agent
    task: str
    status: str
    session_id: str | None = None
    job_id: str | None = None
    run_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    artifacts: tuple[str, ...] = ()
    verification_state: str = "not_verified"


class AutonomyState(str, Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    SELECT = "select"
    DELEGATE = "delegate"
    WAIT_OBSERVE = "wait_observe"
    VERIFY = "verify"
    COMPLETE = "complete"
    REPLAN = "replan"
    HUMAN_ESCALATION = "human_escalation"


@dataclass(frozen=True)
class AutonomyBudgets:
    max_plan_steps: int = 12
    max_delegations: int = 4
    max_replans: int = 2
    max_failures_per_agent: int = 2
    max_project_reads: int = 80
    max_runtime_seconds: int = 1800
    max_token_budget: int = 100_000
