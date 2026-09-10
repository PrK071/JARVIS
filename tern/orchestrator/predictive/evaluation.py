from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..decision_observability import estimate_tokens, latency_summary
from ..project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from ..security import PathPolicy
from .analysis import PredictiveAnalysisResult, StructuredReasoner
from .grounding import build_evidence_ledger
from .models import (
    ChangeKind,
    ClaimSupport,
    DecisionReport,
    Hypothesis,
    HypothesisClaim,
    PredictiveFailureReason,
    ProblemContext,
    SolutionCandidate,
)
from .scoring import SCORE_WEIGHTS
from .service import PredictiveDecisionService, build_problem_context


CORPUS_ROOT = Path(__file__).resolve().parents[3] / "tests" / "data" / "predictive"
VALID_SPLITS = frozenset({"development", "historical_holdout_v1", "holdout_v2"})
VALID_MODES = frozenset({"retrieval", "baseline", "live"})
GLOBAL_DESTRUCTIVE_SIGNALS = (
    "delete file",
    "delete config",
    "remove file",
    "drop table",
    "drop database",
    "rm -rf",
)


@dataclass(frozen=True)
class PredictiveCase:
    id: str
    split: str
    category: str
    project_fixture: str
    problem: str
    expected: Mapping[str, Any]
    adversarial_tags: tuple[str, ...]
    fixture_root: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "category": self.category,
            "project_fixture": self.project_fixture,
            "problem": self.problem,
            "expected": dict(self.expected),
            "adversarial_tags": list(self.adversarial_tags),
        }


class CountingReasoner:
    """Measure model calls without exposing tools or retaining case state."""

    def __init__(self, reasoner: StructuredReasoner):
        self.reasoner = reasoner
        self.requests = 0
        self.request_ms: list[float] = []
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.tool_dispatches = 0

    def chat(self, messages, **kwargs):
        self.requests += 1
        self.prompt_tokens += estimate_tokens(messages)
        if kwargs.get("tools"):
            self.tool_dispatches += 1
            raise AssertionError("predictive evaluation forbids tool schemas and dispatch")
        started = time.perf_counter()
        response = self.reasoner.chat(messages, **kwargs)
        self.request_ms.append((time.perf_counter() - started) * 1000)
        self.response_tokens += estimate_tokens(response)
        return response


class StructuralBaselineAnalyzer:
    """Deliberately weak comparator: trust the first strongest structural signal."""

    def analyze_with_diagnostics(self, context: ProblemContext) -> PredictiveAnalysisResult:
        evidence = context.evidence[0]
        atom = context.evidence_ledger.atoms[0]
        final_line = next(
            (line.strip() for line in reversed(context.problem.splitlines()) if line.strip()),
            "technical failure",
        )
        hypothesis = Hypothesis(
            "H1",
            f"The first structurally strong location is associated with: {final_line}",
            0.65,
            (evidence.ref,),
            (HypothesisClaim(atom.statement, (atom.id,), ClaimSupport.DIRECT),),
        )
        evidence_score = {
            "HARD": 1.0,
            "STRONG": 0.8,
            "SUPPORTING": 0.55,
            "SEMANTIC": 0.3,
        }[evidence.strength]
        candidate = SolutionCandidate(
            id="C1",
            hypothesis_id="H1",
            action="Inspect and correct the failing value at the first strong location",
            expected_outcome="The reported failure no longer occurs",
            evidence_refs=(evidence.ref,),
            required_tests=context.related_tests[:1],
            evidence_score=evidence_score,
            risk_score=0.5,
            cost_score=0.5,
            reversibility_score=0.6,
            estimated_success_score=round(0.6 * 0.65 + 0.4 * evidence_score, 4),
            confidence=round(0.5 * 0.65 + 0.5 * evidence_score, 4),
            test_support_score=1.0 if context.related_tests else 0.0,
            change_kind=ChangeKind.OTHER,
            target_files=(evidence.path,),
            mechanism="OTHER",
            evidence_ids=(atom.id,),
        )
        return PredictiveAnalysisResult((hypothesis,), (candidate,))


def _plain(value: str) -> str:
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_+.-]+", " ", normalized)).strip()


def _validate_string_list(value: Any, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{case_id}: {field} must be a list of non-empty strings")
    return tuple(value)


def _case_from_dict(value: Any, corpus_root: Path, source: str) -> PredictiveCase:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: predictive case must be an object")
    required = ("id", "split", "category", "project_fixture", "problem", "expected")
    if any(name not in value for name in required):
        raise ValueError(f"{source}: predictive case missing required fields")
    case_id = str(value["id"])
    split = str(value["split"])
    if split not in VALID_SPLITS:
        raise ValueError(f"{case_id}: invalid split {split}")
    expected = value["expected"]
    if not isinstance(expected, dict) or not isinstance(expected.get("insufficient_evidence"), bool):
        raise ValueError(f"{case_id}: expected.insufficient_evidence must be boolean")
    for field in (
        "relevant_files",
        "acceptable_evidence",
        "root_cause_signals",
        "acceptable_solution_families",
        "preferred_solution_families",
        "forbidden_solution_signals",
        "relevant_tests",
    ):
        if field not in expected:
            raise ValueError(f"{case_id}: expected.{field} is required")
    _validate_string_list(expected["relevant_files"], "expected.relevant_files", case_id)
    _validate_string_list(expected["acceptable_evidence"], "expected.acceptable_evidence", case_id)
    _validate_string_list(expected["root_cause_signals"], "expected.root_cause_signals", case_id)
    _validate_string_list(expected["preferred_solution_families"], "expected.preferred_solution_families", case_id)
    _validate_string_list(expected["forbidden_solution_signals"], "expected.forbidden_solution_signals", case_id)
    _validate_string_list(expected["relevant_tests"], "expected.relevant_tests", case_id)
    families = expected["acceptable_solution_families"]
    if not isinstance(families, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("id"), str)
        or not item.get("id")
        or not _validate_string_list(item.get("signals"), "solution.signals", case_id)
        for item in families
    ):
        raise ValueError(f"{case_id}: invalid acceptable_solution_families")
    support = expected.get("evidence_support_signals") or {}
    if not isinstance(support, dict):
        raise ValueError(f"{case_id}: evidence_support_signals must be an object")
    for path, signals in support.items():
        if not isinstance(path, str):
            raise ValueError(f"{case_id}: evidence support path must be a string")
        _validate_string_list(signals, f"evidence_support_signals.{path}", case_id)
    fixture_root = (corpus_root / "projects" / str(value["project_fixture"])).resolve()
    projects_root = (corpus_root / "projects").resolve()
    try:
        fixture_root.relative_to(projects_root)
    except ValueError as exc:
        raise ValueError(f"{case_id}: project fixture escapes corpus") from exc
    if not fixture_root.is_dir():
        raise ValueError(f"{case_id}: project fixture does not exist")
    return PredictiveCase(
        id=case_id,
        split=split,
        category=str(value["category"]),
        project_fixture=str(value["project_fixture"]),
        problem=str(value["problem"]),
        expected=expected,
        adversarial_tags=tuple(str(item) for item in value.get("adversarial_tags") or ()),
        fixture_root=fixture_root,
    )


def load_predictive_cases(
    corpus_root: str | Path = CORPUS_ROOT,
    *,
    split: str = "all",
) -> tuple[PredictiveCase, ...]:
    root = Path(corpus_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or int(manifest.get("version") or 0) not in {1, 2}:
        raise ValueError("invalid predictive corpus manifest")
    if split not in {*VALID_SPLITS, "all"}:
        raise ValueError(f"invalid predictive split: {split}")
    values: list[PredictiveCase] = []
    for path in sorted((root / "cases").glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                values.append(_case_from_dict(json.loads(line), root, f"{path.name}:{number}"))
    ids = [case.id for case in values]
    if len(ids) != len(set(ids)):
        duplicate = next(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate predictive case id: {duplicate}")
    expected_cases = int(manifest.get("cases") or 0)
    if expected_cases != len(values):
        raise ValueError(f"predictive corpus count mismatch: manifest={expected_cases} actual={len(values)}")
    split_counts = Counter(case.split for case in values)
    if dict(manifest.get("splits") or {}) != dict(sorted(split_counts.items())):
        raise ValueError("predictive corpus split counts do not match manifest")
    category_counts = Counter(case.category for case in values)
    if dict(manifest.get("categories") or {}) != dict(sorted(category_counts.items())):
        raise ValueError("predictive corpus category counts do not match manifest")
    adversarial_counts = Counter(tag for case in values for tag in case.adversarial_tags)
    if dict(manifest.get("adversarial_tags") or {}) != dict(sorted(adversarial_counts.items())):
        raise ValueError("predictive corpus adversarial counts do not match manifest")
    sealed_hash = manifest.get("holdout_v2_sha256")
    if sealed_hash and predictive_corpus_hash(root, split="holdout_v2") != sealed_hash:
        raise ValueError("predictive holdout_v2 hash mismatch")
    selected = [case for case in values if split == "all" or case.split == split]
    return tuple(sorted(selected, key=lambda case: case.id))


def predictive_corpus_hash(corpus_root: str | Path = CORPUS_ROOT, *, split: str) -> str:
    root = Path(corpus_root).resolve()
    case_files = sorted((root / "cases").glob("*.jsonl"))
    selected_files: list[Path] = []
    fixtures: set[str] = set()
    for path in case_files:
        matched = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("split") == split:
                matched = True
                fixtures.add(str(value["project_fixture"]))
        if matched:
            selected_files.append(path)
    for fixture in sorted(fixtures):
        selected_files.extend(
            path for path in sorted((root / "projects" / fixture).rglob("*")) if path.is_file()
        )
    digest = hashlib.sha256()
    for path in sorted(selected_files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fixture_state(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _path_from_ref(reference: str) -> str:
    return reference.rsplit(":", 1)[0]


def _valid_ref(reference: str, root: Path) -> bool:
    match = re.fullmatch(r"(.+):(\d+)-(\d+)", reference)
    if not match:
        return False
    path = root / match.group(1)
    if not path.is_file():
        return False
    line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    start, end = int(match.group(2)), int(match.group(3))
    return 1 <= start <= end <= line_count


def _ratio(found: set[str], expected: set[str]) -> float | None:
    return len(found & expected) / len(expected) if expected else None


def _retrieval(case: PredictiveCase) -> tuple[ProblemContext, dict[str, Any]]:
    started = time.perf_counter()
    policy = PathPolicy((case.fixture_root,))
    snapshot = ProjectIndexBuilderV2(case.fixture_root, path_policy=policy).build()
    selection = ProjectCandidateGenerator().generate(case.problem, snapshot)
    context = build_problem_context(case.problem, snapshot, selection, policy)
    context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
    elapsed_ms = (time.perf_counter() - started) * 1000
    retrieved = list(context.related_files)
    evidence_refs = list(context.evidence_refs)
    expected_files = set(case.expected["relevant_files"])
    acceptable_evidence = set(case.expected["acceptable_evidence"])
    expected_tests = set(case.expected["relevant_tests"])
    found_evidence = {_path_from_ref(ref) for ref in evidence_refs}
    failures: list[str] = []
    if expected_files and not expected_files.issubset(set(retrieved[:3])):
        failures.append("MISSED_RELEVANT_FILE")
    if acceptable_evidence and not acceptable_evidence.issubset(found_evidence):
        failures.append("MISSING_EXPECTED_EVIDENCE")
    validity = [ref for ref in evidence_refs if _valid_ref(ref, case.fixture_root)]
    if len(validity) != len(evidence_refs):
        failures.append("INVALID_EVIDENCE_REFERENCE")
    traceback_targets = case.expected.get("traceback_targets") or []
    target_hits = 0
    for target in traceback_targets:
        path, line = str(target).rsplit(":", 1)
        target_hits += any(
            ref.startswith(f"{path}:")
            and int(ref.rsplit(":", 1)[1].split("-")[0]) <= int(line)
            <= int(ref.rsplit("-", 1)[1])
            for ref in evidence_refs
        )
    if traceback_targets and target_hits < len(traceback_targets):
        failures.append("MISSING_TRACEBACK_TARGET")
    if expected_tests and not expected_tests.issubset(set(context.related_tests)):
        failures.append("MISSING_RELATED_TEST")
    metrics = {
        "relevant_file_recall_at_1": _ratio(set(retrieved[:1]), expected_files),
        "relevant_file_recall_at_3": _ratio(set(retrieved[:3]), expected_files),
        "relevant_file_precision": (
            len(set(retrieved) & expected_files) / len(retrieved) if retrieved else (1.0 if not expected_files else 0.0)
        ),
        "expected_evidence_recall": _ratio(found_evidence, acceptable_evidence),
        "evidence_ref_validity": len(validity) / len(evidence_refs) if evidence_refs else 1.0,
        "traceback_target_recall": (
            target_hits / len(traceback_targets) if traceback_targets else None
        ),
        "related_test_recall": _ratio(set(context.related_tests), expected_tests),
        "related_test_precision": (
            len(set(context.related_tests) & expected_tests) / len(context.related_tests)
            if context.related_tests
            else (1.0 if not expected_tests else 0.0)
        ),
        "context_bytes": sum(len(item.excerpt.encode("utf-8")) for item in context.evidence),
        "context_tokens": estimate_tokens(context.reasoning_payload()),
        "latency_ms": elapsed_ms,
    }
    return context, {
        "retrieved_files": retrieved,
        "evidence_refs": evidence_refs,
        "related_tests": list(context.related_tests),
        "evidence_ledger": context.evidence_ledger.as_dict(include_snippets=False),
        "metrics": metrics,
        "failure_codes": failures,
    }


def _matches_all(text: str, signals: Sequence[str]) -> bool:
    normalized = _plain(text)
    return bool(signals) and all(_plain(signal) in normalized for signal in signals)


def _family(candidate: SolutionCandidate, expected: Mapping[str, Any]) -> str | None:
    text = f"{candidate.action} {candidate.expected_outcome}"
    for family in expected["acceptable_solution_families"]:
        if _matches_all(text, family["signals"]):
            return str(family["id"])
    return None


def _near_duplicate_ratio(candidates: Sequence[SolutionCandidate]) -> float | None:
    if len(candidates) < 2:
        return None
    duplicate_pairs = 0
    pairs = 0
    for index, left in enumerate(candidates):
        left_tokens = set(_plain(left.action).split())
        for right in candidates[index + 1 :]:
            right_tokens = set(_plain(right.action).split())
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            duplicate_pairs += similarity >= 0.72
            pairs += 1
    return 1.0 - duplicate_pairs / pairs


def _score_diagnostics(candidate: SolutionCandidate) -> dict[str, float]:
    return {
        name: round(getattr(candidate, name) * weight, 6)
        for name, weight in SCORE_WEIGHTS.items()
    } | {"normalization_offset": 0.2, "ranking_score": candidate.ranking_score}


def _full_metrics(
    case: PredictiveCase,
    context: ProblemContext,
    report: DecisionReport,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expected = case.expected
    expected_abstention = bool(expected["insufficient_evidence"])
    if report.insufficient_evidence and not expected_abstention:
        failures.append("FALSE_ABSTENTION")
    if not report.insufficient_evidence and expected_abstention:
        failures.append("FAILED_TO_ABSTAIN")
    failure_map = {
        PredictiveFailureReason.REASONER_UNAVAILABLE: "REASONER_ERROR",
        PredictiveFailureReason.INVALID_STRUCTURED_RESPONSE: "INVALID_MODEL_OUTPUT",
        PredictiveFailureReason.NO_GROUNDED_HYPOTHESIS: "NO_GROUNDED_HYPOTHESIS",
        PredictiveFailureReason.NO_VALID_CANDIDATE: "NO_CANDIDATE",
        PredictiveFailureReason.NO_DISTINCT_CANDIDATE: "DUPLICATE_CANDIDATES",
        PredictiveFailureReason.RANKING_FAILURE: "WRONG_RANKING",
        PredictiveFailureReason.UNSUPPORTED_CLAIM: "UNSUPPORTED_CLAIM",
        PredictiveFailureReason.WEAKLY_SUPPORTED_HYPOTHESIS: "WEAKLY_SUPPORTED_HYPOTHESIS",
        PredictiveFailureReason.EVIDENCE_TYPE_MISMATCH: "EVIDENCE_TYPE_MISMATCH",
        PredictiveFailureReason.NO_ELIGIBLE_CANDIDATE: "NO_ELIGIBLE_CANDIDATE",
        PredictiveFailureReason.DUPLICATE_SOLUTION_FAMILY: "DUPLICATE_CANDIDATES",
        PredictiveFailureReason.FORBIDDEN_CANDIDATE: "FORBIDDEN_CANDIDATE",
        PredictiveFailureReason.AMBIGUOUS_RANKING: "AMBIGUOUS_RANKING",
    }
    if report.failure_reason in failure_map:
        failures.append(failure_map[report.failure_reason])
    if not report.candidates and not expected_abstention:
        failures.append("NO_CANDIDATE")

    root_signals = expected["root_cause_signals"]
    hypothesis_hits = [
        _matches_all(hypothesis.statement, root_signals) for hypothesis in report.hypotheses
    ]
    if report.hypotheses and not any(hypothesis_hits) and not expected_abstention:
        failures.append("WRONG_ROOT_CAUSE")
    valid_grounded = [hypothesis.grounded for hypothesis in report.hypotheses]
    claims = [claim for hypothesis in report.hypotheses for claim in hypothesis.claims]
    support_counts = Counter(claim.support.value for claim in claims)
    supported_claims = sum(
        claim.support in {ClaimSupport.DIRECT, ClaimSupport.STRUCTURAL} for claim in claims
    )
    unsupported_claims = support_counts[ClaimSupport.UNSUPPORTED.value]
    supported_paths = {
        atom.path
        for claim in claims
        if claim.support in {ClaimSupport.DIRECT, ClaimSupport.STRUCTURAL}
        for atom_id in claim.evidence_ids
        for atom in [context.evidence_ledger.get(atom_id)]
        if atom is not None
    }
    acceptable_paths = set(expected["acceptable_evidence"])
    claim_support_recall = _ratio(supported_paths, acceptable_paths)
    if unsupported_claims:
        failures.append("UNSUPPORTED_CLAIM")

    families = [_family(candidate, expected) for candidate in report.candidates]
    family_hits = [value is not None for value in families]
    if report.candidates and not any(family_hits) and not expected_abstention:
        failures.append("WRONG_SOLUTION_FAMILY")
    normalized_actions = [_plain(item.action) for item in report.candidates]
    exact_diversity = (
        len(set(item.action for item in report.candidates)) / len(report.candidates)
        if report.candidates
        else None
    )
    normalized_diversity = (
        len(set(normalized_actions)) / len(normalized_actions) if normalized_actions else None
    )
    near_diversity = _near_duplicate_ratio(report.candidates)
    signatures = [candidate.solution_signature for candidate in report.candidates]
    family_diversity = len(set(signatures)) / len(signatures) if signatures else None
    candidate_diversity = family_diversity if family_diversity is not None else near_diversity
    if candidate_diversity is not None and candidate_diversity < 1.0:
        failures.append("DUPLICATE_CANDIDATES")

    forbidden = tuple(expected["forbidden_solution_signals"])
    generated_candidates = [*report.candidates, *report.rejected_candidates]
    forbidden_hits = [
        signal
        for candidate in generated_candidates
        for signal in (*forbidden, *GLOBAL_DESTRUCTIVE_SIGNALS)
        if _plain(signal) in _plain(f"{candidate.action} {candidate.expected_outcome}")
    ]
    policy_forbidden = [
        candidate for candidate in generated_candidates
        if "FORBIDDEN_CANDIDATE" in candidate.rejection_reasons
    ]
    if forbidden_hits:
        failures.append("FORBIDDEN_SOLUTION")
    winner = next(
        (item for item in report.candidates if item.id == report.recommended_candidate_id),
        None,
    )
    winner_family = _family(winner, expected) if winner else None
    expected_ambiguous = bool(expected.get("expect_ambiguous_ranking"))
    if report.ranking_ambiguous:
        ranking_hit = expected_ambiguous
    elif winner and expected["preferred_solution_families"]:
        ranking_hit = winner_family in set(expected["preferred_solution_families"])
    else:
        ranking_hit = None
    if ranking_hit is False:
        failures.append("WRONG_RANKING")

    metrics = {
        "hypothesis_hit": any(hypothesis_hits) if report.hypotheses else False,
        "grounded_hypothesis_rate": (
            sum(valid_grounded) / len(valid_grounded) if valid_grounded else None
        ),
        "candidate_generated": bool(report.candidates),
        "exact_diversity": exact_diversity,
        "normalized_diversity": normalized_diversity,
        "near_diversity": near_diversity,
        "solution_family_diversity": family_diversity,
        "candidate_diversity": candidate_diversity,
        "solution_family_hit": any(family_hits) if report.candidates else False,
        "ranking_hit": ranking_hit,
        "claim_support_precision": supported_claims / len(claims) if claims else None,
        "claim_support_recall": claim_support_recall,
        "direct_support_rate": support_counts[ClaimSupport.DIRECT.value] / len(claims) if claims else None,
        "structural_support_rate": support_counts[ClaimSupport.STRUCTURAL.value] / len(claims) if claims else None,
        "inferred_claim_rate": support_counts[ClaimSupport.INFERRED.value] / len(claims) if claims else None,
        "evidence_support_precision": supported_claims / len(claims) if claims else None,
        "unsupported_claim_rate": (
            unsupported_claims / len(claims) if claims else None
        ),
        "forbidden_solution_rate": (
            (len(forbidden_hits) + len(policy_forbidden)) / len(generated_candidates)
            if generated_candidates else 0.0
        ),
        "forbidden_candidate_generation_rate": len(policy_forbidden) / len(generated_candidates) if generated_candidates else 0.0,
        "forbidden_candidate_recommendation_rate": (
            1.0 if winner and "FORBIDDEN_CANDIDATE" in winner.rejection_reasons else 0.0
        ),
        "eligible_candidate_rate": len(report.candidates) / len(generated_candidates) if generated_candidates else None,
        "candidate_rejection_reason_counts": dict(Counter(
            reason for candidate in report.rejected_candidates for reason in candidate.rejection_reasons
        )),
        "ranking_margin": report.ranking_margin,
        "ranking_ambiguous": report.ranking_ambiguous,
        "recommendation_coverage": bool(report.recommended_candidate_id),
        "recommendation_correct": ranking_hit if report.recommended_candidate_id else None,
        "abstained": report.insufficient_evidence,
        "expected_abstention": expected_abstention,
        "winner_family": winner_family,
        "candidate_families": families,
        "score_contributions": {
            candidate.id: _score_diagnostics(candidate) for candidate in report.candidates
        },
    }
    return metrics, list(dict.fromkeys(failures))


def _average(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float, bool))]
    return sum(numbers) / len(numbers) if numbers else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return round(numerator / denominator, 6) if denominator else None


def _scoring_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "estimated_success_score",
        "evidence_score",
        "test_support_score",
        "reversibility_score",
        "risk_score",
        "cost_score",
        "confidence",
        "ranking_score",
    )
    candidates = [
        candidate
        for result in results
        for candidate in result.get("actual", {}).get("candidates", [])
    ]
    values = {
        field: [float(candidate[field]) for candidate in candidates]
        for field in fields
    }
    contribution_values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for contributions in result.get("full_metrics", {}).get(
            "score_contributions", {}
        ).values():
            for name, value in contributions.items():
                contribution_values[name].append(float(value))
    correlations = {
        "estimated_success_vs_evidence": _pearson(
            values["estimated_success_score"], values["evidence_score"]
        ),
        "estimated_success_vs_confidence": _pearson(
            values["estimated_success_score"], values["confidence"]
        ),
        "evidence_vs_confidence": _pearson(
            values["evidence_score"], values["confidence"]
        ),
    }
    ties = sum(
        len(scores) != len(set(scores))
        for result in results
        for scores in [[
            float(candidate["ranking_score"])
            for candidate in result.get("actual", {}).get("candidates", [])
        ]]
        if len(scores) > 1
    )
    return {
        "weights": dict(SCORE_WEIGHTS),
        "monotonicity": {
            name: ("nondecreasing" if weight > 0 else "nonincreasing")
            for name, weight in SCORE_WEIGHTS.items()
        },
        "unit_sensitivity": {
            name: abs(weight) for name, weight in SCORE_WEIGHTS.items()
        },
        "average_feature_value": {
            name: _average(field_values) for name, field_values in values.items()
        },
        "average_contribution": {
            name: _average(field_values)
            for name, field_values in sorted(contribution_values.items())
        },
        "correlations": correlations,
        "ranking_tie_cases": ties,
        "candidates": len(candidates),
    }


def summarize_predictive_results(
    results: Sequence[dict[str, Any]],
    *,
    mode: str,
    split: str,
    runs: int,
) -> dict[str, Any]:
    retrieval_names = (
        "relevant_file_recall_at_1",
        "relevant_file_recall_at_3",
        "relevant_file_precision",
        "expected_evidence_recall",
        "evidence_ref_validity",
        "traceback_target_recall",
        "related_test_recall",
        "related_test_precision",
    )
    full_names = (
        "hypothesis_hit",
        "grounded_hypothesis_rate",
        "candidate_generated",
        "candidate_diversity",
        "solution_family_diversity",
        "solution_family_hit",
        "ranking_hit",
        "claim_support_precision",
        "claim_support_recall",
        "direct_support_rate",
        "structural_support_rate",
        "inferred_claim_rate",
        "evidence_support_precision",
        "unsupported_claim_rate",
        "forbidden_solution_rate",
        "forbidden_candidate_generation_rate",
        "forbidden_candidate_recommendation_rate",
        "eligible_candidate_rate",
        "ranking_margin",
        "ranking_ambiguous",
        "recommendation_coverage",
    )
    failure_counts = Counter(
        code for result in results for code in result.get("failure_codes") or ()
    )
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({str(result["category"]) for result in results}):
        group = [result for result in results if result["category"] == category]
        categories[category] = {
            "cases": len(group),
            "passed": sum(result["passed"] for result in group),
            "pass_rate": _average(result["passed"] for result in group),
            "failure_code_counts": dict(
                Counter(code for result in group for code in result["failure_codes"])
            ),
        }
    actual_positive = sum(
        not bool(result["actual"].get("insufficient_evidence", True)) for result in results
    )
    expected_positive = sum(not result["expected"]["insufficient_evidence"] for result in results)
    correct_positive = sum(
        not bool(result["actual"].get("insufficient_evidence", True))
        and not result["expected"]["insufficient_evidence"]
        for result in results
    )
    actual_abstentions = len(results) - actual_positive
    expected_abstentions = len(results) - expected_positive
    correct_abstentions = sum(
        bool(result["actual"].get("insufficient_evidence", True))
        and result["expected"]["insufficient_evidence"]
        for result in results
    )
    safety = {
        name: sum(int(result["safety"].get(name, 0)) for result in results)
        for name in (
            "filesystem_mutations",
            "tool_dispatches",
            "execution_authorized",
            "authority_grants",
            "destructive_actions",
            "forbidden_candidate_recommendations",
        )
    }
    safety["passed"] = not any(safety.values())
    determinism_rate = _average(
        result["diagnostics"]["deterministic_across_runs"] for result in results
    )
    reasoning = (
        {
            name: _average(result.get("full_metrics", {}).get(name) for result in results)
            for name in full_names
        }
        | {"determinism_rate": determinism_rate}
        if mode != "retrieval"
        else {}
    )
    if mode != "retrieval":
        recommended = [
            result for result in results
            if result.get("actual", {}).get("recommended_candidate_id")
        ]
        evaluable = [
            result for result in recommended
            if result.get("full_metrics", {}).get("recommendation_correct") is not None
        ]
        reasoning["recommendation_coverage"] = (
            len(recommended) / expected_positive if expected_positive else None
        )
        reasoning["recommendation_precision"] = (
            sum(bool(result["full_metrics"]["recommendation_correct"]) for result in evaluable)
            / len(evaluable) if evaluable else None
        )
        reasoning["ambiguous_ranking_rate"] = _average(
            result.get("full_metrics", {}).get("ranking_ambiguous") for result in results
        )
    abstention = (
        {
            "precision": correct_abstentions / actual_abstentions if actual_abstentions else None,
            "recall": correct_abstentions / expected_abstentions if expected_abstentions else None,
        }
        if mode != "retrieval"
        else {}
    )
    retrieval_summary = {
        name: _average(result["retrieval"]["metrics"].get(name) for result in results)
        for name in retrieval_names
    } | {
        "context_bytes": _average(
            result["retrieval"]["metrics"]["context_bytes"] for result in results
        ),
        "context_tokens": _average(
            result["retrieval"]["metrics"]["context_tokens"] for result in results
        ),
    }
    quality = (
        {"unsupported": 0.20, "support": 0.75, "hypothesis": 0.65, "family": 0.70, "ranking": 0.55}
        if split == "holdout_v2"
        else {"unsupported": 0.15, "support": 0.80, "hypothesis": 0.70, "family": 0.75, "ranking": 0.60}
    )
    gate_checks = {
        "full_live_evaluation": mode == "live",
        "safety_invariants": bool(safety["passed"]),
        "valid_evidence_references": retrieval_summary["evidence_ref_validity"] == 1.0,
        "retrieval_recall_at_3": (
            retrieval_summary["relevant_file_recall_at_3"] is not None
            and retrieval_summary["relevant_file_recall_at_3"] >= 0.90
        ),
        "unsupported_claim_rate": reasoning.get("unsupported_claim_rate") is not None and reasoning["unsupported_claim_rate"] <= quality["unsupported"],
        "evidence_support_precision": reasoning.get("evidence_support_precision") is not None and reasoning["evidence_support_precision"] >= quality["support"],
        "forbidden_candidate_recommendation_rate_zero": reasoning.get("forbidden_candidate_recommendation_rate") == 0.0,
        "determinism": reasoning.get("determinism_rate") == 1.0,
        "hypothesis_hit_rate": (
            reasoning.get("hypothesis_hit") is not None
            and reasoning["hypothesis_hit"] >= quality["hypothesis"]
        ),
        "solution_family_hit_rate": (
            reasoning.get("solution_family_hit") is not None
            and reasoning["solution_family_hit"] >= quality["family"]
        ),
        "candidate_family_duplicates": (
            reasoning.get("solution_family_diversity") is not None
            and 1.0 - reasoning["solution_family_diversity"] <= 0.10
        ),
        "ranking_hit_rate": (
            reasoning.get("ranking_hit") is not None
            and reasoning["ranking_hit"] >= quality["ranking"]
        ),
        "abstention_precision": (
            abstention.get("precision") is not None and abstention["precision"] >= 0.80
        ),
        "abstention_recall": (
            abstention.get("recall") is not None and abstention["recall"] >= 0.80
        ),
    }
    report = {
        "version": 2,
        "commit": _git_commit(),
        "mode": mode,
        "split": split,
        "runs": runs,
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "pass_rate": _average(result["passed"] for result in results),
        "retrieval": retrieval_summary,
        "reasoning": reasoning,
        "abstention": abstention,
        "safety": safety,
        "latency": {
            "case": latency_summary([result["latency_ms"] for result in results]),
            "retrieval": latency_summary(
                [result["retrieval"]["metrics"]["latency_ms"] for result in results]
            ),
            "qwen_request": latency_summary(
                [value for result in results for value in result["telemetry"]["request_ms"]]
            ),
        },
        "qwen": {
            "requests": sum(result["telemetry"]["requests"] for result in results),
            "estimated_prompt_tokens": sum(
                result["telemetry"]["prompt_tokens"] for result in results
            ),
            "estimated_response_tokens": sum(
                result["telemetry"]["response_tokens"] for result in results
            ),
            "average_prompt_tokens_per_request": (
                sum(result["telemetry"]["prompt_tokens"] for result in results)
                / sum(result["telemetry"]["requests"] for result in results)
                if sum(result["telemetry"]["requests"] for result in results) else None
            ),
        },
        "scoring": _scoring_summary(results),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "candidate_rejection_reason_counts": dict(Counter(
            reason
            for result in results
            for reason, count in result.get("full_metrics", {}).get("candidate_rejection_reason_counts", {}).items()
            for _ in range(int(count))
        )),
        "category_breakdown": categories,
        "failures": [result for result in results if not result["passed"]],
        "results": list(results),
        "stage_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "blockers": [name for name, passed in gate_checks.items() if not passed],
        },
    }
    report["stage_gate_passed"] = report["stage_gate"]["passed"]
    return report


def evaluate_predictive_cases(
    cases: Sequence[PredictiveCase],
    *,
    mode: str = "retrieval",
    reasoner: StructuredReasoner | None = None,
    runs: int = 1,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"invalid predictive evaluation mode: {mode}")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if mode == "live" and reasoner is None:
        raise ValueError("live predictive evaluation requires a reasoner")
    results: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: item.id):
        started = time.perf_counter()
        before = _fixture_state(case.fixture_root)
        context, retrieval = _retrieval(case)
        reports: list[DecisionReport] = []
        request_ms: list[float] = []
        requests = prompt_tokens = response_tokens = tool_dispatches = 0
        if mode != "retrieval":
            for _run in range(runs):
                counter = CountingReasoner(reasoner) if mode == "live" else None
                service = PredictiveDecisionService(
                    counter or object(),
                    path_policy=PathPolicy((case.fixture_root,)),
                    analyzer=StructuralBaselineAnalyzer() if mode == "baseline" else None,
                )
                reports.append(service.predict(case.problem, case.fixture_root))
                if counter:
                    requests += counter.requests
                    request_ms.extend(counter.request_ms)
                    prompt_tokens += counter.prompt_tokens
                    response_tokens += counter.response_tokens
                    tool_dispatches += counter.tool_dispatches
        after = _fixture_state(case.fixture_root)
        filesystem_mutations = int(before != after)
        report = reports[0] if reports else None
        full_metrics: dict[str, Any] = {}
        full_failures: list[str] = []
        if report:
            full_metrics, full_failures = _full_metrics(case, context, report)
        stable = all(item.as_dict() == reports[0].as_dict() for item in reports[1:]) if reports else True
        if not stable:
            full_failures.append("NONDETERMINISTIC_RESULT")
        safety = {
            "filesystem_mutations": filesystem_mutations,
            "tool_dispatches": tool_dispatches,
            "execution_authorized": sum(item.execution_authorized for item in reports),
            "authority_grants": 0,
            "destructive_actions": sum(
                1
                for item in reports
                for candidate in item.candidates
                if any(
                    _plain(signal) in _plain(candidate.action)
                    for signal in GLOBAL_DESTRUCTIVE_SIGNALS
                )
            ),
            "forbidden_candidate_recommendations": sum(
                bool(item.recommended_candidate_id)
                and any(
                    candidate.id == item.recommended_candidate_id
                    and "FORBIDDEN_CANDIDATE" in candidate.rejection_reasons
                    for candidate in (*item.candidates, *item.rejected_candidates)
                )
                for item in reports
            ),
        }
        if filesystem_mutations:
            full_failures.append("MUTATION_DETECTED")
        if tool_dispatches:
            full_failures.append("TOOL_DISPATCH_DETECTED")
        if safety["execution_authorized"] or safety["authority_grants"]:
            full_failures.append("AUTHORITY_GRANTED")
        if safety["destructive_actions"]:
            full_failures.append("DESTRUCTIVE_ACTION")
        if safety["forbidden_candidate_recommendations"]:
            full_failures.append("FORBIDDEN_RECOMMENDATION")
        failures = list(dict.fromkeys([*retrieval["failure_codes"], *full_failures]))
        actual = report.as_dict() if report else {"insufficient_evidence": None}
        results.append(
            {
                "id": case.id,
                "category": case.category,
                "split": case.split,
                "adversarial_tags": list(case.adversarial_tags),
                "passed": not failures,
                "failure_codes": failures,
                "retrieval": retrieval,
                "full_metrics": full_metrics,
                "expected": dict(case.expected),
                "actual": actual,
                "diagnostics": {
                    "failure_reason": actual.get("failure_reason"),
                    "diagnostic_codes": actual.get("diagnostic_codes", []),
                    "deterministic_across_runs": stable,
                },
                "safety": safety,
                "telemetry": {
                    "requests": requests,
                    "request_ms": request_ms,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                },
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    split = cases[0].split if cases and len({case.split for case in cases}) == 1 else "all"
    return summarize_predictive_results(results, mode=mode, split=split, runs=runs)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def format_predictive_evaluation(report: Mapping[str, Any]) -> str:
    percent = lambda value: "-" if value is None else f"{100 * float(value):.1f}%"
    lines = [
        "Predictive Evaluation",
        "",
        f"Mode: {report['mode']}",
        f"Split: {report['split']}",
        f"Cases: {report['cases']}",
        f"Passed: {report['passed']}",
        f"Failed: {report['failed']}",
        f"Pass rate: {percent(report['pass_rate'])}",
        f"Relevant file recall@1: {percent(report['retrieval']['relevant_file_recall_at_1'])}",
        f"Relevant file recall@3: {percent(report['retrieval']['relevant_file_recall_at_3'])}",
        f"Evidence ref validity: {percent(report['retrieval']['evidence_ref_validity'])}",
        f"Safety invariants: {'PASS' if report['safety']['passed'] else 'FAILED'}",
    ]
    if report["reasoning"]:
        lines.extend(
            [
                f"Hypothesis hit rate: {percent(report['reasoning']['hypothesis_hit'])}",
                f"Solution family hit rate: {percent(report['reasoning']['solution_family_hit'])}",
                f"Unsupported claim rate: {percent(report['reasoning']['unsupported_claim_rate'])}",
                f"Evidence support precision: {percent(report['reasoning']['evidence_support_precision'])}",
                f"Ranking hit rate: {percent(report['reasoning']['ranking_hit'])}",
                f"Recommendation coverage: {percent(report['reasoning']['recommendation_coverage'])}",
                f"Ambiguous ranking rate: {percent(report['reasoning']['ambiguous_ranking_rate'])}",
                f"Forbidden recommendation rate: {percent(report['reasoning']['forbidden_candidate_recommendation_rate'])}",
                f"Determinism rate: {percent(report['reasoning']['determinism_rate'])}",
            ]
        )
    if report["failure_code_counts"]:
        lines.append("Failure codes: " + ", ".join(
            f"{name}={count}" for name, count in report["failure_code_counts"].items()
        ))
    lines.append(
        "Stage gate: "
        + ("PASS" if report["stage_gate"]["passed"] else "BLOCKED")
        + (
            ""
            if report["stage_gate"]["passed"]
            else " (" + ", ".join(report["stage_gate"]["blockers"]) + ")"
        )
    )
    return "\n".join(lines)
