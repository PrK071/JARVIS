from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _bounded_score(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return number


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


class PredictiveFailureReason(str, Enum):
    INSUFFICIENT_STRUCTURAL_EVIDENCE = "INSUFFICIENT_STRUCTURAL_EVIDENCE"
    REASONER_UNAVAILABLE = "REASONER_UNAVAILABLE"
    INVALID_STRUCTURED_RESPONSE = "INVALID_STRUCTURED_RESPONSE"
    NO_GROUNDED_HYPOTHESIS = "NO_GROUNDED_HYPOTHESIS"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    NO_DISTINCT_CANDIDATE = "NO_DISTINCT_CANDIDATE"
    RANKING_FAILURE = "RANKING_FAILURE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    WEAKLY_SUPPORTED_HYPOTHESIS = "WEAKLY_SUPPORTED_HYPOTHESIS"
    EVIDENCE_TYPE_MISMATCH = "EVIDENCE_TYPE_MISMATCH"
    NO_ELIGIBLE_CANDIDATE = "NO_ELIGIBLE_CANDIDATE"
    DUPLICATE_SOLUTION_FAMILY = "DUPLICATE_SOLUTION_FAMILY"
    FORBIDDEN_CANDIDATE = "FORBIDDEN_CANDIDATE"
    AMBIGUOUS_RANKING = "AMBIGUOUS_RANKING"


class EvidenceKind(str, Enum):
    TRACEBACK_FRAME = "TRACEBACK_FRAME"
    SYMBOL_DEFINITION = "SYMBOL_DEFINITION"
    CALL_RELATION = "CALL_RELATION"
    IMPORT_RELATION = "IMPORT_RELATION"
    ASSIGNMENT = "ASSIGNMENT"
    RETURN_STATEMENT = "RETURN_STATEMENT"
    CONDITIONAL = "CONDITIONAL"
    TEST_RELATION = "TEST_RELATION"
    CONFIG_REFERENCE = "CONFIG_REFERENCE"
    GIT_CHANGE = "GIT_CHANGE"
    STRUCTURAL_RELATION = "STRUCTURAL_RELATION"


class ClaimSupport(str, Enum):
    DIRECT = "DIRECT"
    STRUCTURAL = "STRUCTURAL"
    INFERRED = "INFERRED"
    UNSUPPORTED = "UNSUPPORTED"


class ChangeKind(str, Enum):
    VALIDATION = "VALIDATION"
    UPSTREAM_FIX = "UPSTREAM_FIX"
    CONTROL_FLOW = "CONTROL_FLOW"
    RETURN_VALUE = "RETURN_VALUE"
    CALL_SITE = "CALL_SITE"
    CONFIGURATION = "CONFIGURATION"
    IMPORT = "IMPORT"
    ERROR_HANDLING = "ERROR_HANDLING"
    TEST_FIX = "TEST_FIX"
    OTHER = "OTHER"


class TestSupportLevel(str, Enum):
    NONE = "NONE"
    RELATED_FILE = "RELATED_FILE"
    RELATED_SYMBOL = "RELATED_SYMBOL"
    DIRECT_BEHAVIORAL = "DIRECT_BEHAVIORAL"


@dataclass(frozen=True)
class EvidenceAtom:
    id: str
    kind: EvidenceKind
    path: str
    start_line: int
    end_line: int
    statement: str
    strength: float
    symbol: str | None = None
    relation: str | None = None
    snippet: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.path.strip() or not self.statement.strip():
            raise ValueError("evidence atom identity, path and statement are required")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("evidence atom must identify a valid source range")
        object.__setattr__(self, "strength", _bounded_score("strength", self.strength))

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def as_dict(self, *, include_snippet: bool = True) -> dict[str, Any]:
        value = {
            "id": self.id,
            "kind": self.kind.value,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "relation": self.relation,
            "statement": self.statement,
            "strength": self.strength,
        }
        if include_snippet:
            value["snippet"] = self.snippet
        return value


@dataclass(frozen=True)
class EvidenceLedger:
    atoms: tuple[EvidenceAtom, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        ids = [atom.id for atom in self.atoms]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ledger IDs must be unique")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(atom.id for atom in self.atoms)

    def get(self, atom_id: str) -> EvidenceAtom | None:
        return next((atom for atom in self.atoms if atom.id == atom_id), None)

    def as_dict(self, *, include_snippets: bool = True) -> dict[str, Any]:
        return {
            "version": 1,
            "atoms": [atom.as_dict(include_snippet=include_snippets) for atom in self.atoms],
        }


@dataclass(frozen=True)
class HypothesisClaim:
    statement: str
    evidence_ids: tuple[str, ...]
    support: ClaimSupport

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("hypothesis claim statement is required")
        object.__setattr__(self, "evidence_ids", _unique_strings(self.evidence_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "evidence_ids": list(self.evidence_ids),
            "support": self.support.value,
        }


@dataclass(frozen=True)
class EvidenceExcerpt:
    ref: str
    path: str
    start_line: int
    end_line: int
    strength: str
    sources: tuple[str, ...]
    excerpt: str

    def __post_init__(self) -> None:
        if not self.ref or not self.path or self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("evidence must identify a concrete source range")
        if not self.excerpt.strip():
            raise ValueError("evidence excerpt must not be empty")
        object.__setattr__(self, "sources", _unique_strings(self.sources))

    def as_dict(self, *, include_excerpt: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "ref": self.ref,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "strength": self.strength,
            "sources": list(self.sources),
        }
        if include_excerpt:
            value["excerpt"] = self.excerpt
        return value


@dataclass(frozen=True)
class ProblemContext:
    problem: str
    project_path: str
    project_id: str
    evidence: tuple[EvidenceExcerpt, ...]
    related_files: tuple[str, ...]
    related_symbols: tuple[str, ...]
    related_tests: tuple[str, ...]
    test_relationships: tuple[tuple[str, str], ...] = ()
    evidence_ledger: EvidenceLedger = field(default_factory=EvidenceLedger)

    def __post_init__(self) -> None:
        if not self.problem.strip():
            raise ValueError("problem must not be empty")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "related_files", _unique_strings(self.related_files))
        object.__setattr__(self, "related_symbols", _unique_strings(self.related_symbols))
        object.__setattr__(self, "related_tests", _unique_strings(self.related_tests))
        object.__setattr__(
            self,
            "test_relationships",
            tuple(dict.fromkeys((str(source), str(test)) for source, test in self.test_relationships)),
        )

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return tuple(item.ref for item in self.evidence)

    @property
    def sufficient_evidence(self) -> bool:
        return any(item.strength in {"HARD", "STRONG"} for item in self.evidence)

    def reasoning_payload(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "project_id": self.project_id,
            "evidence_ledger": self.evidence_ledger.as_dict(include_snippets=True),
            "related_symbols": list(self.related_symbols),
            "related_tests": list(self.related_tests),
            "test_relationships": [
                {"production_file": source, "test_file": test}
                for source, test in self.test_relationships
            ],
        }


@dataclass(frozen=True)
class Hypothesis:
    id: str
    statement: str
    confidence: float
    evidence_refs: tuple[str, ...]
    claims: tuple[HypothesisClaim, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.statement.strip():
            raise ValueError("hypothesis id and statement must not be empty")
        object.__setattr__(self, "confidence", _bounded_score("confidence", self.confidence))
        object.__setattr__(self, "evidence_refs", _unique_strings(self.evidence_refs))
        object.__setattr__(self, "claims", tuple(self.claims))
        if not self.evidence_refs:
            raise ValueError("hypothesis must reference concrete evidence")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "claims": [claim.as_dict() for claim in self.claims],
        }

    @property
    def grounded(self) -> bool:
        return bool(self.claims) and any(
            claim.support in {ClaimSupport.DIRECT, ClaimSupport.STRUCTURAL}
            for claim in self.claims
        ) and all(claim.support is not ClaimSupport.UNSUPPORTED for claim in self.claims)


@dataclass(frozen=True)
class SolutionCandidate:
    id: str
    hypothesis_id: str
    action: str
    expected_outcome: str
    evidence_refs: tuple[str, ...]
    required_tests: tuple[str, ...]
    evidence_score: float
    risk_score: float
    cost_score: float
    reversibility_score: float
    estimated_success_score: float
    confidence: float
    test_support_score: float = 0.0
    ranking_score: float = 0.0
    score_explanation: str = ""
    change_kind: ChangeKind = ChangeKind.OTHER
    target_files: tuple[str, ...] = ()
    target_symbols: tuple[str, ...] = ()
    mechanism: str = "OTHER"
    evidence_ids: tuple[str, ...] = ()
    test_support_level: TestSupportLevel = TestSupportLevel.NONE
    eligible: bool = True
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.id, self.hypothesis_id, self.action, self.expected_outcome)
        ):
            raise ValueError("candidate identity, action and outcome must not be empty")
        object.__setattr__(self, "evidence_refs", _unique_strings(self.evidence_refs))
        object.__setattr__(self, "required_tests", _unique_strings(self.required_tests))
        object.__setattr__(self, "target_files", _unique_strings(self.target_files))
        object.__setattr__(self, "target_symbols", _unique_strings(self.target_symbols))
        object.__setattr__(self, "evidence_ids", _unique_strings(self.evidence_ids))
        object.__setattr__(self, "rejection_reasons", _unique_strings(self.rejection_reasons))
        if not self.evidence_refs:
            raise ValueError("candidate must reference concrete evidence")
        for name in (
            "evidence_score",
            "risk_score",
            "cost_score",
            "reversibility_score",
            "estimated_success_score",
            "confidence",
            "test_support_score",
            "ranking_score",
        ):
            object.__setattr__(self, name, _bounded_score(name, getattr(self, name)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "evidence_refs": list(self.evidence_refs),
            "required_tests": list(self.required_tests),
            "evidence_score": self.evidence_score,
            "risk_score": self.risk_score,
            "cost_score": self.cost_score,
            "reversibility_score": self.reversibility_score,
            "estimated_success_score": self.estimated_success_score,
            "confidence": self.confidence,
            "test_support_score": self.test_support_score,
            "ranking_score": self.ranking_score,
            "score_explanation": self.score_explanation,
            "change_kind": self.change_kind.value,
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "mechanism": self.mechanism,
            "evidence_ids": list(self.evidence_ids),
            "test_support_level": self.test_support_level.value,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @property
    def solution_signature(self) -> str:
        symbols = ",".join(sorted(self.target_symbols))
        files = ",".join(sorted(self.target_files))
        mechanism = " ".join(self.mechanism.casefold().split())
        return "|".join((self.change_kind.value, files, symbols, mechanism, self.hypothesis_id))


@dataclass(frozen=True)
class DecisionReport:
    problem: str
    hypotheses: tuple[Hypothesis, ...]
    candidates: tuple[SolutionCandidate, ...]
    recommended_candidate_id: str | None
    insufficient_evidence: bool
    recommendation_explanation: str
    failure_reason: PredictiveFailureReason | None = None
    diagnostic_codes: tuple[str, ...] = ()
    rejected_candidates: tuple[SolutionCandidate, ...] = ()
    ranking_margin: float | None = None
    ranking_ambiguous: bool = False
    requires_approval: bool = field(default=True, init=False)
    dry_run: bool = field(default=True, init=False)
    execution_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "diagnostic_codes", _unique_strings(self.diagnostic_codes))
        object.__setattr__(self, "rejected_candidates", tuple(self.rejected_candidates))
        if len(self.hypotheses) > 3 or len(self.candidates) > 3:
            raise ValueError("predictive reports are limited to three hypotheses and candidates")
        candidate_ids = {item.id for item in self.candidates}
        if self.insufficient_evidence:
            if self.recommended_candidate_id is not None:
                raise ValueError("insufficient evidence cannot have a recommendation")
        elif self.recommended_candidate_id is not None and self.recommended_candidate_id not in candidate_ids:
            raise ValueError("recommended candidate must exist in the report")
        if self.ranking_margin is not None:
            object.__setattr__(self, "ranking_margin", _bounded_score("ranking_margin", self.ranking_margin))
        if not self.recommendation_explanation.strip():
            raise ValueError("report must explain its outcome")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "problem": self.problem,
            "hypotheses": [item.as_dict() for item in self.hypotheses],
            "candidates": [item.as_dict() for item in self.candidates],
            "recommended_candidate_id": self.recommended_candidate_id,
            "insufficient_evidence": self.insufficient_evidence,
            "requires_approval": self.requires_approval,
            "recommendation_explanation": self.recommendation_explanation,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
            "diagnostic_codes": list(self.diagnostic_codes),
            "rejected_candidates": [item.as_dict() for item in self.rejected_candidates],
            "ranking_margin": self.ranking_margin,
            "ranking_ambiguous": self.ranking_ambiguous,
            "dry_run": self.dry_run,
            "execution_authorized": self.execution_authorized,
        }
