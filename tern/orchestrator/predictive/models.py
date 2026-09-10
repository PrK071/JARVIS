from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _bounded_score(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return number


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


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

    def __post_init__(self) -> None:
        if not self.problem.strip():
            raise ValueError("problem must not be empty")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "related_files", _unique_strings(self.related_files))
        object.__setattr__(self, "related_symbols", _unique_strings(self.related_symbols))
        object.__setattr__(self, "related_tests", _unique_strings(self.related_tests))

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
            "evidence": [item.as_dict(include_excerpt=True) for item in self.evidence],
            "related_symbols": list(self.related_symbols),
            "related_tests": list(self.related_tests),
        }


@dataclass(frozen=True)
class Hypothesis:
    id: str
    statement: str
    confidence: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.statement.strip():
            raise ValueError("hypothesis id and statement must not be empty")
        object.__setattr__(self, "confidence", _bounded_score("confidence", self.confidence))
        object.__setattr__(self, "evidence_refs", _unique_strings(self.evidence_refs))
        if not self.evidence_refs:
            raise ValueError("hypothesis must reference concrete evidence")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }


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

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (self.id, self.hypothesis_id, self.action, self.expected_outcome)
        ):
            raise ValueError("candidate identity, action and outcome must not be empty")
        object.__setattr__(self, "evidence_refs", _unique_strings(self.evidence_refs))
        object.__setattr__(self, "required_tests", _unique_strings(self.required_tests))
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
        }


@dataclass(frozen=True)
class DecisionReport:
    problem: str
    hypotheses: tuple[Hypothesis, ...]
    candidates: tuple[SolutionCandidate, ...]
    recommended_candidate_id: str | None
    insufficient_evidence: bool
    recommendation_explanation: str
    requires_approval: bool = field(default=True, init=False)
    dry_run: bool = field(default=True, init=False)
    execution_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if len(self.hypotheses) > 3 or len(self.candidates) > 3:
            raise ValueError("predictive reports are limited to three hypotheses and candidates")
        candidate_ids = {item.id for item in self.candidates}
        if self.insufficient_evidence:
            if self.recommended_candidate_id is not None:
                raise ValueError("insufficient evidence cannot have a recommendation")
        elif self.recommended_candidate_id not in candidate_ids:
            raise ValueError("recommended candidate must exist in the report")
        if not self.recommendation_explanation.strip():
            raise ValueError("report must explain its outcome")

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "hypotheses": [item.as_dict() for item in self.hypotheses],
            "candidates": [item.as_dict() for item in self.candidates],
            "recommended_candidate_id": self.recommended_candidate_id,
            "insufficient_evidence": self.insufficient_evidence,
            "requires_approval": self.requires_approval,
            "recommendation_explanation": self.recommendation_explanation,
            "dry_run": self.dry_run,
            "execution_authorized": self.execution_authorized,
        }
