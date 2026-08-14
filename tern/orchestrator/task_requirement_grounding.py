from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .autonomy_foundation import (
    Agent,
    AgentCapabilityProfile,
    AgentEligibility,
    AgentRuntimeAvailability,
    Capability,
    EligibilityEngine,
    PermissionProfile,
    RiskLevel,
    TaskRequirements,
)
from .explicit_agent_binding import detect_explicit_agent_binding
from .intent_semantics import Constraint, IntentFrame, IntentFrameBuilder, SpeechAct


MUTATION_REQUIRED = "mutation_required"
READ_ONLY_REQUIRED = "read_only_required"
REQUIREMENT_DIMENSIONS = tuple(item.value for item in Capability) + (
    MUTATION_REQUIRED,
    READ_ONLY_REQUIRED,
)
MODEL_INFERENCE_DIMENSIONS = tuple(
    name
    for name in REQUIREMENT_DIMENSIONS
    if name
    not in {
        Capability.LONG_RUNNING_JOB.value,
        Capability.PERSISTENT_SESSION.value,
        Capability.SEMANTIC_INTERPRETATION.value,
        Capability.TASK_REQUIREMENT_EXTRACTION.value,
    }
)
_SCOPE_BOUND_ACCESS = frozenset(
    {
        Capability.REPOSITORY_READ.value,
        Capability.REPOSITORY_WRITE.value,
        Capability.FILESYSTEM_READ.value,
        Capability.FILESYSTEM_WRITE.value,
        Capability.CODE_EDIT.value,
        Capability.TEST_EXECUTION.value,
    }
)


class RequirementValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class RequirementEvidenceSource(str, Enum):
    EXPLICIT_USER = "EXPLICIT_USER"
    EXPLICIT_USER_NEGATION = "EXPLICIT_USER_NEGATION"
    REQUESTED_AGENT = "REQUESTED_AGENT"
    PROJECT_FACT = "PROJECT_FACT"
    RUNTIME_FACT = "RUNTIME_FACT"
    TASK_IMPLICATION = "TASK_IMPLICATION"
    SEMANTIC_INFERENCE = "SEMANTIC_INFERENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class RequirementSafetyClass(str, Enum):
    ELIGIBILITY_RELEVANT = "ELIGIBILITY_RELEVANT"
    EXECUTION_SAFETY_CRITICAL = "EXECUTION_SAFETY_CRITICAL"
    OPTIONAL = "OPTIONAL"


_SOURCE_STRENGTH = {
    RequirementEvidenceSource.EXPLICIT_USER: 4,
    RequirementEvidenceSource.EXPLICIT_USER_NEGATION: 4,
    RequirementEvidenceSource.REQUESTED_AGENT: 4,
    RequirementEvidenceSource.PROJECT_FACT: 3,
    RequirementEvidenceSource.RUNTIME_FACT: 3,
    RequirementEvidenceSource.TASK_IMPLICATION: 2,
    RequirementEvidenceSource.SEMANTIC_INFERENCE: 1,
    RequirementEvidenceSource.INSUFFICIENT_EVIDENCE: 0,
    RequirementEvidenceSource.CONFLICTING_EVIDENCE: 5,
}


_SAFETY_CLASS = {
    MUTATION_REQUIRED: RequirementSafetyClass.EXECUTION_SAFETY_CRITICAL,
    Capability.MUTATION.value: RequirementSafetyClass.EXECUTION_SAFETY_CRITICAL,
    Capability.REPOSITORY_WRITE.value: RequirementSafetyClass.EXECUTION_SAFETY_CRITICAL,
    Capability.FILESYSTEM_WRITE.value: RequirementSafetyClass.EXECUTION_SAFETY_CRITICAL,
    Capability.CODE_EDIT.value: RequirementSafetyClass.EXECUTION_SAFETY_CRITICAL,
    Capability.TEST_EXECUTION.value: RequirementSafetyClass.ELIGIBILITY_RELEVANT,
    Capability.WEB_ACCESS.value: RequirementSafetyClass.ELIGIBILITY_RELEVANT,
}


def requirement_safety_class(name: str) -> RequirementSafetyClass:
    if name in _SAFETY_CLASS:
        return _SAFETY_CLASS[name]
    if name == READ_ONLY_REQUIRED:
        return RequirementSafetyClass.EXECUTION_SAFETY_CRITICAL
    if name in {item.value for item in Capability}:
        return RequirementSafetyClass.ELIGIBILITY_RELEVANT
    return RequirementSafetyClass.OPTIONAL


@dataclass(frozen=True)
class GroundedRequirement:
    name: str
    value: RequirementValue
    source: RequirementEvidenceSource
    evidence_refs: tuple[str, ...]
    safety_class: RequirementSafetyClass

    def __post_init__(self) -> None:
        if self.name not in REQUIREMENT_DIMENSIONS:
            raise ValueError(f"unknown requirement dimension: {self.name}")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must not be empty")

    @classmethod
    def unknown(cls, name: str) -> "GroundedRequirement":
        return cls(
            name,
            RequirementValue.UNKNOWN,
            RequirementEvidenceSource.INSUFFICIENT_EVIDENCE,
            ("evidence:insufficient",),
            requirement_safety_class(name),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value.value,
            "source": self.source.value,
            "evidence_refs": list(self.evidence_refs),
            "safety_class": self.safety_class.value,
        }


def grounded_requirement(
    name: str,
    value: RequirementValue,
    source: RequirementEvidenceSource,
    evidence_ref: str,
) -> GroundedRequirement:
    return GroundedRequirement(
        name,
        value,
        source,
        (evidence_ref,),
        requirement_safety_class(name),
    )


def merge_grounded_requirements(
    current: GroundedRequirement,
    candidate: GroundedRequirement,
) -> GroundedRequirement:
    """Merge evidence without allowing weaker evidence to replace stronger facts."""

    if current.name != candidate.name:
        raise ValueError("cannot merge different requirement dimensions")
    if current.value is RequirementValue.CONFLICT:
        return current
    if candidate.value is RequirementValue.CONFLICT:
        return candidate
    current_strength = _SOURCE_STRENGTH[current.source]
    candidate_strength = _SOURCE_STRENGTH[candidate.source]
    if candidate_strength > current_strength:
        return candidate
    if candidate_strength < current_strength:
        return current
    if current.value == candidate.value:
        return GroundedRequirement(
            current.name,
            current.value,
            current.source,
            tuple(dict.fromkeys((*current.evidence_refs, *candidate.evidence_refs))),
            current.safety_class,
        )
    if RequirementValue.UNKNOWN in {current.value, candidate.value}:
        return candidate if current.value is RequirementValue.UNKNOWN else current
    return GroundedRequirement(
        current.name,
        RequirementValue.CONFLICT,
        RequirementEvidenceSource.CONFLICTING_EVIDENCE,
        tuple(dict.fromkeys((*current.evidence_refs, *candidate.evidence_refs))),
        current.safety_class,
    )


@dataclass(frozen=True)
class ExplicitTaskEvidence:
    requirements: Mapping[str, GroundedRequirement]
    requested_agent: Agent | None = None
    requested_agent_source: RequirementEvidenceSource | None = None
    requested_agent_evidence_ref: str | None = None
    target_scope: str = "unknown"
    expected_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()
    tests_requested: tuple[str, ...] = ()
    prohibitions: tuple[str, ...] = ()
    output_requirements: tuple[str, ...] = ()
    contradiction_refs: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirements": {
                name: item.as_dict() for name, item in sorted(self.requirements.items())
            },
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "requested_agent_source": (
                self.requested_agent_source.value if self.requested_agent_source else None
            ),
            "requested_agent_evidence_ref": self.requested_agent_evidence_ref,
            "target_scope": self.target_scope,
            "expected_files": list(self.expected_files),
            "forbidden_files": list(self.forbidden_files),
            "tests_requested": list(self.tests_requested),
            "prohibitions": list(self.prohibitions),
            "output_requirements": list(self.output_requirements),
            "contradiction_refs": list(self.contradiction_refs),
        }


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(character for character in normalized if not unicodedata.combining(character)).split()
    )


def _safe_task_ref(value: str) -> str:
    return "task:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


class ExplicitTaskEvidenceExtractor:
    """Conservatively preserves explicit facts; it never selects an agent.

    The bounded clause recognizers below only identify unequivocal permissions,
    prohibitions, and requested operations. All other semantics stay unresolved
    for the existing single model call.
    """

    _READ_ONLY = re.compile(
        r"\b(?:so|somente)\s+(?:de\s+)?leitura\b|"
        r"\bsem\s+(?:modificar|alterar|editar|mudar|mexer)(?:\s+nada)?\b|"
        r"\bnao\s+(?:modifique|altere|edite|mude|mexa)\s+"
        r"(?:nada|(?:o|os|este|esse|esta|essa)\s+(?:codigo|arquivos?|modulo|projeto))\b"
    )
    _NO_TESTS = re.compile(
        r"\bsem\s+(?:rodar|executar)\s+(?:os\s+)?testes\b|"
        r"\bnao\s+(?:rode|execute|rodar|executar)\s+(?:os\s+)?testes\b"
    )
    _TESTS = re.compile(
        r"\b(?:rode|roda|rodar|execute|executa|executar)\s+(?:os\s+)?testes\b"
    )
    _NO_WEB = re.compile(
        r"\bnao\s+(?:use|acesse|consulte|pesquise)(?:\s+(?:a|na))?\s+web\b|"
        r"\bsem\s+(?:usar|acessar|consultar|pesquisar)(?:\s+(?:a|na))?\s+web\b|"
        r"\bsomente\s+com\s+(?:o\s+)?codigo\s+local\b"
    )
    _WEB = re.compile(
        r"\b(?:use|acesse|consulte|pesquise|pesquisar)(?:\s+(?:a|na))?\s+web\b|"
        r"\bpesquise\s+(?:a\s+)?documentacao\s+oficial\b"
    )
    _NO_COMMIT = re.compile(
        r"\bnao\s+(?:faca|crie|realize)\s+(?:o\s+)?commit\b|\bsem\s+commit\b"
    )
    _NO_PUBLIC_API = re.compile(
        r"\bnao\s+(?:altere|mude|modifique)\s+(?:a\s+)?api\s+publica\b"
    )
    _MUTATION = re.compile(
        r"\b(?:corrija|corrige|corrigir|edite|edita|editar|altere|altera|alterar|"
        r"implemente|implementa|implementar)\b"
    )
    _REPOSITORY_ANALYSIS = re.compile(
        r"\b(?:analise|analisar|revise|revisar|inspecione|inspecionar)\b.*"
        r"\b(?:repositorio|projeto|modulo|codigo|arquivo)\b|"
        r"\b(?:repositorio|projeto|modulo|codigo|arquivo)\b.*"
        r"\b(?:analise|analisar|revise|revisar|inspecione|inspecionar)\b"
    )
    _REVIEW = re.compile(r"\b(?:revise|revisar|review)\b")
    _ANALYSIS_ONLY = re.compile(
        r"\b(?:analise|analisar|investigue|investigar|encontre|encontrar)\b"
    )
    _PATH = re.compile(
        r"(?<![\w.-])(?P<path>(?:[\w.-]+[\\/])+[\w.-]+\.[A-Za-z0-9]+)(?![\w.-])"
    )

    def extract(
        self,
        task: str,
        *,
        project_snapshot: Mapping[str, Any] | None = None,
        intent_frame: IntentFrame | None = None,
    ) -> ExplicitTaskEvidence:
        normalized = _plain(task)
        task_ref = _safe_task_ref(normalized)
        if intent_frame is None:
            project_path = str((project_snapshot or {}).get("project_path") or "")
            project_id = Path(project_path).name if project_path else None
            context = SimpleNamespace(
                active_project=project_id,
                project_root=project_path or None,
                known_projects=(
                    ({"id": project_id, "root": project_path},)
                    if project_id and project_path
                    else ()
                ),
            )
            intent_frame, _ = IntentFrameBuilder().build(normalized, context)
        values: dict[str, GroundedRequirement] = {}

        def add(
            name: str,
            value: RequirementValue,
            source: RequirementEvidenceSource,
            ref: str,
        ) -> None:
            item = grounded_requirement(name, value, source, ref)
            values[name] = merge_grounded_requirements(values.get(name, GroundedRequirement.unknown(name)), item)

        binding = detect_explicit_agent_binding(task)
        requested_agent = Agent(binding.requested_agent) if binding else None
        read_only = bool(self._READ_ONLY.search(normalized)) or any(
            item in intent_frame.constraints
            for item in (Constraint.READ_ONLY, Constraint.FORBID_MUTATION)
        )
        no_tests = bool(self._NO_TESTS.search(normalized))
        tests = bool(self._TESTS.search(normalized)) and not no_tests
        no_web = bool(self._NO_WEB.search(normalized))
        web = bool(self._WEB.search(normalized)) and not no_web
        mutation = bool(self._MUTATION.search(normalized))
        repository_analysis = bool(self._REPOSITORY_ANALYSIS.search(normalized)) or (
            intent_frame.operation in {"read", "search"} and bool(intent_frame.target)
        )
        review = bool(self._REVIEW.search(normalized))

        prohibitions: list[str] = []
        forbidden_files: list[str] = []
        tests_requested: list[str] = []

        if read_only:
            for name in (
                MUTATION_REQUIRED,
                Capability.MUTATION.value,
                Capability.REPOSITORY_WRITE.value,
                Capability.FILESYSTEM_WRITE.value,
                Capability.CODE_EDIT.value,
            ):
                add(
                    name,
                    RequirementValue.FALSE,
                    RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
                    "constraint:forbid_mutation",
                )
            add(
                READ_ONLY_REQUIRED,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
                "constraint:read_only",
            )
            add(
                Capability.READ_ONLY.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:read_only_capable",
            )
            prohibitions.append("mutation")
        if mutation:
            add(
                MUTATION_REQUIRED,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
                "action:mutation",
            )
            if not read_only:
                add(
                    READ_ONLY_REQUIRED,
                    RequirementValue.FALSE,
                    RequirementEvidenceSource.TASK_IMPLICATION,
                    "implication:mutation_not_read_only",
                )
                for name in (
                    Capability.MUTATION.value,
                    Capability.REPOSITORY_WRITE.value,
                    Capability.CODE_EDIT.value,
                    Capability.REPOSITORY_READ.value,
                    Capability.CODE_ANALYSIS.value,
                ):
                    add(
                        name,
                        RequirementValue.TRUE,
                        RequirementEvidenceSource.TASK_IMPLICATION,
                        "implication:code_mutation",
                    )

        analysis_only = bool(self._ANALYSIS_ONLY.search(normalized)) and not mutation
        if (analysis_only or web or repository_analysis) and not read_only:
            add(
                MUTATION_REQUIRED,
                RequirementValue.FALSE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:analysis_without_mutation",
            )
            add(
                READ_ONLY_REQUIRED,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:analysis_read_only",
            )
            add(
                Capability.READ_ONLY.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:analysis_read_only_capable",
            )
        if analysis_only and re.search(
            r"\b(?:bug|codigo|modulo|arquitetura|repositorio|projeto)\b",
            normalized,
        ):
            add(
                Capability.CODE_ANALYSIS.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:code_analysis",
            )

        if no_tests:
            add(
                Capability.TEST_EXECUTION.value,
                RequirementValue.FALSE,
                RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
                "constraint:forbid_tests",
            )
            prohibitions.append("test_execution")
        elif tests:
            add(
                Capability.TEST_EXECUTION.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
                "action:test_execution",
            )
            tests_requested.append("explicit:test_execution")

        if no_web:
            add(
                Capability.WEB_ACCESS.value,
                RequirementValue.FALSE,
                RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
                "constraint:forbid_web",
            )
            prohibitions.append("web_access")
        elif web:
            add(
                Capability.WEB_ACCESS.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.EXPLICIT_USER,
                "action:web_access",
            )

        if repository_analysis:
            for name in (Capability.REPOSITORY_READ.value, Capability.CODE_ANALYSIS.value):
                add(
                    name,
                    RequirementValue.TRUE,
                    RequirementEvidenceSource.TASK_IMPLICATION,
                    "implication:repository_analysis",
                )
            if review:
                add(
                    Capability.CODE_REVIEW.value,
                    RequirementValue.TRUE,
                    RequirementEvidenceSource.TASK_IMPLICATION,
                    "implication:code_review",
                )

        informational = (
            intent_frame.execution_requested is False
            and intent_frame.speech_act
            in {
                SpeechAct.QUESTION,
                SpeechAct.EXPLANATION_REQUEST,
            }
            and not mutation
        )
        if informational:
            add(
                MUTATION_REQUIRED,
                RequirementValue.FALSE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:informational_no_mutation",
            )
            add(
                READ_ONLY_REQUIRED,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:informational_read_only",
            )
            add(
                Capability.READ_ONLY.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:informational_read_only_capable",
            )

        known_non_mutating = bool(
            not mutation
            and (read_only or analysis_only or web or informational or repository_analysis)
        )
        if known_non_mutating:
            add(
                Capability.GENERAL_REASONING.value,
                RequirementValue.TRUE,
                RequirementEvidenceSource.TASK_IMPLICATION,
                "implication:reasoning_task",
            )
            for name in (
                Capability.MUTATION.value,
                Capability.REPOSITORY_WRITE.value,
                Capability.FILESYSTEM_WRITE.value,
                Capability.CODE_EDIT.value,
            ):
                add(
                    name,
                    RequirementValue.FALSE,
                    RequirementEvidenceSource.TASK_IMPLICATION,
                    "implication:non_mutating_task",
                )

        expected_files: list[str] = []
        known_paths = {
            str(item.get("path"))
            for item in (project_snapshot or {}).get("repo_map", ())
            if isinstance(item, Mapping) and item.get("path")
        }
        known_paths.update((project_snapshot or {}).get("important_files", ()))
        for match in self._PATH.finditer(task.replace("\\", "/")):
            path = match.group("path").replace("\\", "/")
            if path in known_paths:
                expected_files.append(path)
                add(
                    Capability.REPOSITORY_READ.value,
                    RequirementValue.TRUE,
                    RequirementEvidenceSource.PROJECT_FACT,
                    "project:target_path_exists",
                )
                add(
                    Capability.CODE_ANALYSIS.value,
                    RequirementValue.TRUE,
                    RequirementEvidenceSource.TASK_IMPLICATION,
                    "implication:inspect_code_target",
                )
                if review:
                    add(
                        Capability.CODE_REVIEW.value,
                        RequirementValue.TRUE,
                        RequirementEvidenceSource.TASK_IMPLICATION,
                        "implication:code_review",
                    )

        if self._NO_COMMIT.search(normalized):
            prohibitions.append("commit")
        if self._NO_PUBLIC_API.search(normalized):
            prohibitions.append("public_api_change")
            forbidden_files.append("scope:public_api")

        target_scope = (
            "files"
            if expected_files
            else "repository"
            if repository_analysis or mutation
            else "web"
            if web
            else "informational"
            if informational
            else "analysis"
            if analysis_only
            else "unknown"
        )
        contradiction_refs = tuple(
            f"intent:{index}"
            for index, _ in enumerate(intent_frame.contradictory_constraints, start=1)
        )
        return ExplicitTaskEvidence(
            requirements=values,
            requested_agent=requested_agent,
            requested_agent_source=(
                RequirementEvidenceSource.REQUESTED_AGENT if binding else None
            ),
            requested_agent_evidence_ref=(
                f"binding:{binding.evidence}" if binding else None
            ),
            target_scope=target_scope,
            expected_files=tuple(dict.fromkeys(expected_files)),
            forbidden_files=tuple(dict.fromkeys(forbidden_files)),
            tests_requested=tuple(dict.fromkeys(tests_requested)),
            prohibitions=tuple(dict.fromkeys(prohibitions)),
            contradiction_refs=contradiction_refs,
            output_requirements=(task_ref,),
        )


@dataclass(frozen=True)
class GroundedTaskRequirements:
    requirements: Mapping[str, GroundedRequirement]
    target_scope: str
    risk_level: RiskLevel
    ambiguity_material: bool
    expected_files: tuple[str, ...] = ()
    forbidden_files: tuple[str, ...] = ()
    tests_requested: tuple[str, ...] = ()
    prohibitions: tuple[str, ...] = ()
    requested_agent: Agent | None = None
    requested_agent_source: RequirementEvidenceSource | None = None
    requested_agent_evidence_ref: str | None = None

    def __post_init__(self) -> None:
        if set(self.requirements) != set(REQUIREMENT_DIMENSIONS):
            raise ValueError("grounded requirements must contain every dimension")

    @property
    def true_capabilities(self) -> frozenset[Capability]:
        return frozenset(
            item
            for item in Capability
            if self.requirements[item.value].value is RequirementValue.TRUE
        )

    @property
    def unknown_dimensions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, item in self.requirements.items()
            if item.value is RequirementValue.UNKNOWN
        )

    @property
    def conflict_dimensions(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, item in self.requirements.items()
            if item.value is RequirementValue.CONFLICT
        )

    @property
    def mutation_authorized_by_requirements(self) -> bool:
        mutation = self.requirements[MUTATION_REQUIRED]
        return mutation.value is RequirementValue.TRUE

    def to_legacy(self) -> TaskRequirements:
        mutation = self.requirements[MUTATION_REQUIRED].value
        read_only = self.requirements[READ_ONLY_REQUIRED].value
        return TaskRequirements(
            capabilities=self.true_capabilities,
            mutation_required=mutation is RequirementValue.TRUE,
            read_only_required=read_only is RequirementValue.TRUE,
            target_scope=self.target_scope,
            risk_level=self.risk_level,
            expected_files=self.expected_files,
            forbidden_files=self.forbidden_files,
            tests_requested=self.tests_requested,
            ambiguity_material=self.ambiguity_material,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirements": {
                name: item.as_dict() for name, item in sorted(self.requirements.items())
            },
            "target_scope": self.target_scope,
            "risk_level": self.risk_level.value,
            "ambiguity_material": self.ambiguity_material,
            "expected_files": list(self.expected_files),
            "forbidden_files": list(self.forbidden_files),
            "tests_requested": list(self.tests_requested),
            "prohibitions": list(self.prohibitions),
            "requested_agent": self.requested_agent.value if self.requested_agent else None,
            "requested_agent_source": (
                self.requested_agent_source.value if self.requested_agent_source else None
            ),
            "requested_agent_evidence_ref": self.requested_agent_evidence_ref,
        }


def grounded_requirement_json_schema(unresolved: Iterable[str]) -> dict[str, Any]:
    names = tuple(unresolved)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "resolved": {
                "type": "array",
                "maxItems": len(names),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "enum": list(names)},
                        "value": {"type": "string", "enum": ["TRUE", "FALSE"]},
                    },
                    "required": ["name", "value"],
                },
            },
            "target_scope": {"type": "string", "maxLength": 200},
            "risk_level": {"type": "string", "enum": [item.value for item in RiskLevel]},
            "ambiguity_material": {"type": "boolean"},
        },
        "required": ["resolved", "target_scope", "risk_level", "ambiguity_material"],
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "grounded_task_requirements", "strict": True, "schema": schema},
    }


@dataclass(frozen=True)
class GroundedRequirementAnalysisResult:
    requirements: GroundedTaskRequirements | None
    valid: bool
    first_pass_valid: bool
    attempts: int
    inference_count: int
    latency_ms: float
    finish_reason: str | None
    prompt_tokens: int | None
    generated_tokens: int | None
    error_code: str | None = None


class GroundedTaskRequirementAnalyzer:
    system_prompt = (
        "Resolve only the listed unresolved task requirement dimensions. "
        "Return TRUE or FALSE only when the request provides sufficient evidence; "
        "omit uncertain dimensions so they remain UNKNOWN. Do not choose or name an "
        "executor. Known evidence is immutable and must not be regenerated."
    )

    def __init__(self, client: Any, *, extractor: ExplicitTaskEvidenceExtractor | None = None):
        self.client = client
        self.extractor = extractor or ExplicitTaskEvidenceExtractor()

    def analyze(
        self,
        task: str,
        *,
        project_snapshot: Mapping[str, Any] | None = None,
        explicit_evidence: ExplicitTaskEvidence | None = None,
    ) -> GroundedRequirementAnalysisResult:
        started = time.perf_counter()
        evidence = explicit_evidence or self.extractor.extract(
            task,
            project_snapshot=project_snapshot,
        )
        seeded = dict(evidence.requirements)
        unresolved = tuple(
            name
            for name in MODEL_INFERENCE_DIMENSIONS
            if name not in seeded
        )
        schema = grounded_requirement_json_schema(unresolved)
        payload = {
            "task": task,
            "known_evidence": evidence.as_dict(),
            "unresolved_dimensions": list(unresolved),
            "project_snapshot": project_snapshot or None,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = "GROUNDED_REQUIREMENT_INVALID"
        for attempt in range(1, 3):
            try:
                response = self.client.chat(
                    messages,
                    response_format=schema,
                    temperature=0.0,
                    max_tokens=512,
                )
                choice = (response.get("choices") or [{}])[0]
                usage = response.get("usage") or {}
                raw = json.loads(choice["message"]["content"])
                resolved = raw["resolved"]
                if not isinstance(resolved, list):
                    raise ValueError("resolved must be a list")
                names = [str(item["name"]) for item in resolved]
                if len(names) != len(set(names)) or not set(names).issubset(unresolved):
                    raise ValueError("resolved names are invalid or duplicated")
                merged = {
                    name: seeded.get(name, GroundedRequirement.unknown(name))
                    for name in REQUIREMENT_DIMENSIONS
                }
                for item in resolved:
                    name = str(item["name"])
                    value = RequirementValue(str(item["value"]))
                    has_strong_conflict = bool(
                        evidence.contradiction_refs
                        or any(
                            item.value is RequirementValue.CONFLICT
                            for item in seeded.values()
                        )
                    )
                    unsupported_scope_access = bool(
                        value is RequirementValue.TRUE
                        and (
                            (
                                name in _SCOPE_BOUND_ACCESS
                                and evidence.target_scope not in {"repository", "files"}
                                and not evidence.expected_files
                            )
                            or (
                                name == Capability.WEB_ACCESS.value
                                and evidence.target_scope != "web"
                            )
                        )
                    )
                    if has_strong_conflict or unsupported_scope_access:
                        continue
                    candidate = grounded_requirement(
                        name,
                        value,
                        RequirementEvidenceSource.SEMANTIC_INFERENCE,
                        "model:resolved_dimension",
                    )
                    merged[name] = merge_grounded_requirements(merged[name], candidate)
                grounded = GroundedTaskRequirements(
                    requirements=merged,
                    target_scope=(
                        evidence.target_scope
                        if evidence.target_scope != "unknown"
                        else str(raw["target_scope"])
                    ),
                    risk_level=RiskLevel(raw["risk_level"]),
                    ambiguity_material=bool(raw["ambiguity_material"]),
                    expected_files=evidence.expected_files,
                    forbidden_files=evidence.forbidden_files,
                    tests_requested=evidence.tests_requested,
                    prohibitions=evidence.prohibitions,
                    requested_agent=evidence.requested_agent,
                    requested_agent_source=evidence.requested_agent_source,
                    requested_agent_evidence_ref=evidence.requested_agent_evidence_ref,
                )
                return GroundedRequirementAnalysisResult(
                    grounded,
                    True,
                    attempt == 1,
                    attempt,
                    attempt,
                    round((time.perf_counter() - started) * 1000, 3),
                    choice.get("finish_reason"),
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = type(exc).__name__
                if attempt == 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The object was invalid. Return one object matching the "
                                f"same schema. Validation error: {last_error}."
                            ),
                        }
                    )
            except Exception as exc:
                last_error = type(exc).__name__
        return GroundedRequirementAnalysisResult(
            None,
            False,
            False,
            2,
            2,
            round((time.perf_counter() - started) * 1000, 3),
            None,
            None,
            None,
            last_error,
        )


@dataclass(frozen=True)
class GroundedAgentEligibility:
    base: AgentEligibility
    unresolved_safety_requirements: tuple[str, ...]
    conflict_requirements: tuple[str, ...]
    authorized_capabilities: tuple[Capability, ...]
    unknown_capabilities: tuple[Capability, ...]
    execution_safe: bool
    reason_codes: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        extra = tuple(
            f"EXECUTION_BLOCKED_UNKNOWN_{name.upper()}"
            for name in self.unresolved_safety_requirements
        ) + tuple(
            f"EXECUTION_BLOCKED_CONFLICT_{name.upper()}"
            for name in self.conflict_requirements
        )
        object.__setattr__(self, "reason_codes", (*self.base.reason_codes, *extra))

    @property
    def agent(self) -> Agent:
        return self.base.agent

    @property
    def eligible(self) -> bool:
        return self.base.eligible

    @property
    def executable_now(self) -> bool:
        return self.base.executable_now and self.execution_safe

    def as_dict(self) -> dict[str, Any]:
        value = self.base.as_dict()
        value.update(
            {
                "executable_now": self.executable_now,
                "execution_safe": self.execution_safe,
                "unresolved_safety_requirements": list(
                    self.unresolved_safety_requirements
                ),
                "conflict_requirements": list(self.conflict_requirements),
                "authorized_capabilities": [item.value for item in self.authorized_capabilities],
                "unknown_capabilities": [item.value for item in self.unknown_capabilities],
                "reason_codes": list(self.reason_codes),
            }
        )
        return value


class GroundedEligibilityEngine:
    """UNKNOWN does not eliminate an agent and never authorizes an operation."""

    def __init__(self) -> None:
        self.legacy = EligibilityEngine()

    def evaluate(
        self,
        requirements: GroundedTaskRequirements,
        profiles: Mapping[Agent, AgentCapabilityProfile],
        availability: Mapping[Agent, AgentRuntimeAvailability],
        permissions: Mapping[Agent, PermissionProfile] | None = None,
    ) -> dict[Agent, GroundedAgentEligibility]:
        base = self.legacy.evaluate(
            requirements.to_legacy(),
            profiles,
            availability,
            permissions,
        )
        mutation = requirements.requirements[MUTATION_REQUIRED]
        conflicts = requirements.conflict_dimensions
        unresolved_safety = (
            (MUTATION_REQUIRED,)
            if mutation.value is RequirementValue.UNKNOWN
            else ()
        )
        if mutation.value is RequirementValue.FALSE:
            unsafe_true = tuple(
                name
                for name in (
                    Capability.MUTATION.value,
                    Capability.REPOSITORY_WRITE.value,
                    Capability.FILESYSTEM_WRITE.value,
                    Capability.CODE_EDIT.value,
                )
                if requirements.requirements[name].value is RequirementValue.TRUE
            )
            conflicts = tuple(dict.fromkeys((*conflicts, *unsafe_true)))
        unknown_capabilities = tuple(
            item
            for item in Capability
            if requirements.requirements[item.value].value is RequirementValue.UNKNOWN
        )
        authorized = tuple(sorted(requirements.true_capabilities, key=lambda item: item.value))
        execution_safe = not unresolved_safety and not conflicts
        return {
            agent: GroundedAgentEligibility(
                result,
                unresolved_safety,
                conflicts,
                authorized,
                unknown_capabilities,
                execution_safe,
            )
            for agent, result in base.items()
        }


def load_grounded_fixture(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
