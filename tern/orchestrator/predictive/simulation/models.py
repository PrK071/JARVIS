from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from ..models import DecisionReport
from ..repair import RepairTarget


class OutcomeStatus(str, Enum):
    PREDICTED = "PREDICTED"
    OBSERVED = "OBSERVED"
    NOT_TESTED = "NOT_TESTED"
    SIMULATION_FAILED = "SIMULATION_FAILED"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ChangeBudget:
    max_files: int = 2
    max_lines_added: int = 60
    max_lines_removed: int = 60
    max_diff_bytes: int = 32_768

    def __post_init__(self) -> None:
        if not 1 <= self.max_files <= 3 or not all(
            1 <= value <= 120 for value in (self.max_lines_added, self.max_lines_removed)
        ) or not 1 <= self.max_diff_bytes <= 65_536:
            raise ValueError("invalid conservative patch budget")


@dataclass(frozen=True)
class PatchProposal:
    candidate_id: str
    decision_report_id: str
    analyzed_commit: str
    root_cause_id: str
    strategy_kind: str
    targets: tuple[RepairTarget, ...]
    evidence_ids: tuple[str, ...]
    allowed_files: tuple[str, ...]
    diff: str
    prediction_rank: int
    prediction_score: float
    budget: ChangeBudget = field(default_factory=ChangeBudget)
    binding_slot: int | None = None
    binding_keyword: str | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not all((self.candidate_id, self.decision_report_id, self.analyzed_commit,
                    self.root_cause_id, self.evidence_ids, self.targets, self.allowed_files)):
            raise ValueError("proposal requires grounded candidate provenance")
        if self.prediction_rank < 1 or not 0 <= self.prediction_score <= 1:
            raise ValueError("invalid prediction rank/score")
        if self.binding_slot is not None and self.binding_slot < 0:
            raise ValueError("invalid argument binding slot")

    @property
    def patch_sha256(self) -> str:
        return hashlib.sha256(self.diff.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        value = asdict(self)
        value["targets"] = [target.as_dict() for target in self.targets]
        return value | {"patch_sha256": self.patch_sha256}

    @classmethod
    def from_dict(cls, value: dict) -> PatchProposal:
        from ..repair import RepairTargetKind
        if value.get("schema_version", 1) != 1:
            raise ValueError("unsupported patch proposal version")
        targets = tuple(RepairTarget(
            item["path"], RepairTargetKind(item["scope_kind"]),
            item.get("symbol"), item.get("parameter"), item.get("attribute"),
            item.get("expression_id"), item.get("line"),
        ) for item in value["targets"])
        return cls(
            value["candidate_id"], value["decision_report_id"], value["analyzed_commit"],
            value["root_cause_id"], value["strategy_kind"], targets,
            tuple(value["evidence_ids"]), tuple(value["allowed_files"]), value["diff"],
            value["prediction_rank"], value["prediction_score"], ChangeBudget(**value.get("budget", {})),
            value.get("binding_slot"), value.get("binding_keyword"),
        )

    @classmethod
    def from_report(cls, report: DecisionReport, candidate_id: str, diff: str, *, analyzed_commit: str) -> PatchProposal:
        from ..causal import RepairStrategyKind
        from ..policy import PredictiveCandidatePolicy
        from ..repair import validate_repair_target
        candidate = next((item for item in report.candidates if item.id == candidate_id), None)
        if candidate is None or not candidate.eligible or candidate.rejection_reasons or PredictiveCandidatePolicy().rejection_reasons(candidate):
            raise ValueError("candidate is not eligible")
        root = next((item for item in report.root_cause_candidates if item.id == candidate.root_cause_id), None)
        hypothesis = next((item for item in report.hypotheses if item.id == candidate.hypothesis_id), None)
        if root is None or hypothesis is None or not hypothesis.grounded or hypothesis.root_cause_id != root.id:
            raise ValueError("candidate root/hypothesis is not grounded")
        if not set(candidate.evidence_ids) <= set(root.evidence_ids):
            raise ValueError("candidate evidence is disconnected from root")
        if report.causal_slice is None or not all(validate_repair_target(
            RepairStrategyKind(candidate.strategy_kind), root, target, report.causal_slice,
        ) for target in candidate.repair_targets):
            raise ValueError("candidate repair target is incompatible")
        report_id = hashlib.sha256(json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        profile = root.responsibility
        return cls(candidate.id, report_id, analyzed_commit, root.id, candidate.strategy_kind,
                   candidate.repair_targets, candidate.evidence_ids, candidate.target_files, diff,
                   report.candidates.index(candidate) + 1, candidate.ranking_score,
                   binding_slot=profile.parameter_ordinal if profile else None,
                   binding_keyword=profile.keyword_binding if profile else None)


@dataclass(frozen=True)
class PatchValidation:
    passed: bool
    failure_codes: tuple[str, ...] = ()
    files_changed: tuple[str, ...] = ()
    lines_added: int = 0
    lines_removed: int = 0
    syntax_passed: bool | None = None
    preimage_sha256: tuple[tuple[str, str], ...] = ()
    preview_sha256: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObservedCandidateOutcome:
    candidate_id: str
    decision_report_id: str
    analyzed_commit: str
    prediction_rank: int
    prediction_score: float
    patch_sha256: str
    validation: PatchValidation
    status: OutcomeStatus
    baseline_tests: tuple[tuple[str, str], ...] = ()
    patched_tests: tuple[tuple[str, str], ...] = ()
    regressions: tuple[str, ...] = ()
    observed_evidence: tuple[str, ...] = ()
    timed_out: bool = False
    sandbox_violation: bool = False
    infrastructure_failure: bool = False
    failure_reason: str | None = None
    timings: tuple[tuple[str, float], ...] = ()
    schema_version: int = field(default=1, init=False)
    execution_authorized: bool = field(default=False, init=False)
    requires_approval: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.status is OutcomeStatus.OBSERVED and (
            not self.baseline_tests or not self.patched_tests or self.infrastructure_failure
        ):
            raise ValueError("observed outcome requires baseline and patched tests without infrastructure failure")
        if self.status in {OutcomeStatus.NOT_TESTED, OutcomeStatus.SANDBOX_UNAVAILABLE} and (
            self.baseline_tests or self.patched_tests
        ):
            raise ValueError("untested outcome cannot contain executed tests")

    def as_dict(self) -> dict:
        return asdict(self)
