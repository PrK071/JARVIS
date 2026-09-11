from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import CORPUS_ROOT, PredictiveCase, evaluate_predictive_cases


class EvaluationStatus(str, Enum):
    CANONICAL = "CANONICAL"
    MULTIPLE_VALID_ANSWERS = "MULTIPLE_VALID_ANSWERS"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


EVALUABLE_STATUSES = frozenset(
    {EvaluationStatus.CANONICAL, EvaluationStatus.MULTIPLE_VALID_ANSWERS}
)


@dataclass(frozen=True)
class CausalTruth:
    path: str
    symbol: str | None
    cause_kind: str
    evidence: tuple[str, ...]
    preferred: bool = False

    def matches(self, root: Mapping[str, Any]) -> bool:
        actual_symbol = str(root.get("origin_symbol") or "")
        return (
            root.get("origin_path") == self.path
            and root.get("cause_kind") == self.cause_kind
            and (
                self.symbol is None
                or actual_symbol == self.symbol
                or actual_symbol.endswith(f".{self.symbol}")
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "cause_kind": self.cause_kind,
            "evidence": list(self.evidence),
            "preferred": self.preferred,
        }


@dataclass(frozen=True)
class RepairTruth:
    strategy: str
    target_files: tuple[str, ...]
    target_symbols: tuple[str, ...]
    preferred: bool = False

    def matches(self, candidate: Mapping[str, Any]) -> bool:
        files = set(candidate.get("target_files") or ())
        symbols = set(candidate.get("target_symbols") or ())
        return (
            candidate.get("strategy_kind") == self.strategy
            and bool(files & set(self.target_files))
            and (not self.target_symbols or bool(symbols & set(self.target_symbols)))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "preferred": self.preferred,
        }


@dataclass(frozen=True)
class BenchmarkAdjudication:
    case_id: str
    status: EvaluationStatus
    failure_site: Mapping[str, Any] | None
    acceptable_root_causes: tuple[CausalTruth, ...]
    required_causal_edges: tuple[str, ...]
    optional_causal_edges: tuple[str, ...]
    acceptable_repairs: tuple[RepairTruth, ...]
    acceptable_repair_strategies: tuple[str, ...]
    preferred_repair_strategies: tuple[str, ...]
    acceptable_target_files: tuple[str, ...]
    acceptable_target_symbols: tuple[str, ...]
    forbidden_strategies: tuple[str, ...]
    abstention_expected: bool
    notes: str
    legacy_expectation: str

    @property
    def evaluable(self) -> bool:
        return self.status in EVALUABLE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status.value,
            "failure_site": dict(self.failure_site) if self.failure_site else None,
            "acceptable_root_causes": [item.as_dict() for item in self.acceptable_root_causes],
            "acceptable_root_cause_kinds": sorted(
                {item.cause_kind for item in self.acceptable_root_causes}
            ),
            "required_causal_edges": list(self.required_causal_edges),
            "optional_causal_edges": list(self.optional_causal_edges),
            "acceptable_repairs": [item.as_dict() for item in self.acceptable_repairs],
            "acceptable_repair_strategies": list(self.acceptable_repair_strategies),
            "preferred_repair_strategies": list(self.preferred_repair_strategies),
            "acceptable_target_files": list(self.acceptable_target_files),
            "acceptable_target_symbols": list(self.acceptable_target_symbols),
            "forbidden_strategies": list(self.forbidden_strategies),
            "abstention_expected": self.abstention_expected,
            "notes": self.notes,
            "legacy_expectation": self.legacy_expectation,
        }


def _strings(value: Any, field: str, case_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{case_id}: {field} must be a list of non-empty strings")
    return tuple(value)


def _reference_exists(reference: str, fixture_root: Path) -> bool:
    match = re.fullmatch(r"(.+):(\d+)(?:-(\d+))?", reference)
    if not match:
        return False
    path = fixture_root / match.group(1)
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = int(match.group(2))
    end = int(match.group(3) or start)
    return 1 <= start <= end <= len(lines)


def _adjudication_from_dict(value: Any, case: PredictiveCase, source: str) -> BenchmarkAdjudication:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: adjudication must be an object")
    case_id = str(value.get("case_id") or "")
    if case_id != case.id:
        raise ValueError(f"{source}: adjudication case id does not match {case.id}")
    try:
        status = EvaluationStatus(str(value["status"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{case_id}: invalid evaluation status") from exc
    failure_site = value.get("failure_site")
    if failure_site is not None:
        if not isinstance(failure_site, dict) or not isinstance(failure_site.get("path"), str):
            raise ValueError(f"{case_id}: invalid failure_site")
        path = case.fixture_root / failure_site["path"]
        line = int(failure_site.get("line") or 0)
        if not path.is_file() or line < 1 or line > len(path.read_text(encoding="utf-8").splitlines()):
            raise ValueError(f"{case_id}: failure_site does not exist")
    roots_value = value.get("acceptable_root_causes")
    if not isinstance(roots_value, list):
        raise ValueError(f"{case_id}: acceptable_root_causes must be a list")
    roots: list[CausalTruth] = []
    for root in roots_value:
        if not isinstance(root, dict):
            raise ValueError(f"{case_id}: root cause truth must be an object")
        path = str(root.get("path") or "")
        kind = str(root.get("cause_kind") or "")
        evidence = _strings(root.get("evidence"), "root.evidence", case_id)
        if not path or not kind or not (case.fixture_root / path).is_file():
            raise ValueError(f"{case_id}: root cause path/kind is invalid")
        if any(not _reference_exists(ref, case.fixture_root) for ref in evidence):
            raise ValueError(f"{case_id}: root cause evidence reference is invalid")
        roots.append(CausalTruth(path, root.get("symbol"), kind, evidence, bool(root.get("preferred"))))
    repairs_value = value.get("acceptable_repairs")
    if not isinstance(repairs_value, list):
        raise ValueError(f"{case_id}: acceptable_repairs must be a list")
    repairs: list[RepairTruth] = []
    for repair in repairs_value:
        if not isinstance(repair, dict):
            raise ValueError(f"{case_id}: repair truth must be an object")
        files = _strings(repair.get("target_files"), "repair.target_files", case_id)
        symbols = _strings(repair.get("target_symbols"), "repair.target_symbols", case_id)
        if any(not (case.fixture_root / path).is_file() for path in files):
            raise ValueError(f"{case_id}: repair target does not exist")
        repairs.append(RepairTruth(str(repair.get("strategy") or ""), files, symbols, bool(repair.get("preferred"))))
    abstention = value.get("abstention_expected")
    if not isinstance(abstention, bool):
        raise ValueError(f"{case_id}: abstention_expected must be boolean")
    strategies = _strings(value.get("acceptable_repair_strategies"), "acceptable_repair_strategies", case_id)
    preferred = _strings(value.get("preferred_repair_strategies"), "preferred_repair_strategies", case_id)
    targets = _strings(value.get("acceptable_target_files"), "acceptable_target_files", case_id)
    target_symbols = _strings(value.get("acceptable_target_symbols"), "acceptable_target_symbols", case_id)
    forbidden = _strings(value.get("forbidden_strategies"), "forbidden_strategies", case_id)
    required_edges = _strings(value.get("required_causal_edges"), "required_causal_edges", case_id)
    optional_edges = _strings(value.get("optional_causal_edges"), "optional_causal_edges", case_id)
    if any(item not in strategies for item in preferred):
        raise ValueError(f"{case_id}: preferred repair must also be acceptable")
    if any(not (case.fixture_root / path).is_file() for path in targets):
        raise ValueError(f"{case_id}: acceptable repair target does not exist")
    if any(item.strategy not in strategies for item in repairs):
        raise ValueError(f"{case_id}: repair option strategy must be acceptable")
    if status in EVALUABLE_STATUSES and not abstention and (not roots or not strategies or not targets or not repairs):
        raise ValueError(f"{case_id}: evaluable positive cases require roots, strategies and targets")
    return BenchmarkAdjudication(
        case_id=case_id,
        status=status,
        failure_site=failure_site,
        acceptable_root_causes=tuple(roots),
        required_causal_edges=required_edges,
        optional_causal_edges=optional_edges,
        acceptable_repairs=tuple(repairs),
        acceptable_repair_strategies=strategies,
        preferred_repair_strategies=preferred,
        acceptable_target_files=targets,
        acceptable_target_symbols=target_symbols,
        forbidden_strategies=forbidden,
        abstention_expected=abstention,
        notes=str(value.get("notes") or ""),
        legacy_expectation=str(value.get("legacy_expectation") or ""),
    )


def load_benchmark_v4_adjudications(
    cases: Sequence[PredictiveCase],
    corpus_root: str | Path = CORPUS_ROOT,
) -> tuple[BenchmarkAdjudication, ...]:
    root = Path(corpus_root).resolve()
    case_lookup = {case.id: case for case in cases}
    values: list[BenchmarkAdjudication] = []
    for path in sorted((root / "v4" / "adjudications").glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case_id = str(raw.get("case_id") or "") if isinstance(raw, dict) else ""
            if case_id not in case_lookup:
                continue
            values.append(_adjudication_from_dict(raw, case_lookup[case_id], f"{path.name}:{number}"))
    ids = [item.case_id for item in values]
    if len(ids) != len(set(ids)):
        duplicate = next(item for item, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate benchmark v4 adjudication id: {duplicate}")
    missing = sorted(set(case_lookup) - set(ids))
    if missing:
        raise ValueError(f"missing benchmark v4 adjudications: {', '.join(missing)}")
    return tuple(sorted(values, key=lambda item: item.case_id))


def _candidate_valid(candidate: Mapping[str, Any], truth: BenchmarkAdjudication) -> tuple[bool, bool, bool]:
    strategy_ok = candidate.get("strategy_kind") in truth.acceptable_repair_strategies
    target_ok = any(
        bool(set(candidate.get("target_files") or ()) & set(option.target_files))
        and (not option.target_symbols or bool(set(candidate.get("target_symbols") or ()) & set(option.target_symbols)))
        for option in truth.acceptable_repairs
    )
    return any(option.matches(candidate) for option in truth.acceptable_repairs), strategy_ok, target_ok


def adjudicate_predictive_result_v4(
    result: Mapping[str, Any], truth: BenchmarkAdjudication
) -> dict[str, Any]:
    actual = result.get("actual") or {}
    roots = {item["id"]: item for item in actual.get("root_cause_candidates") or ()}
    selected_roots = [
        roots[item["root_cause_id"]]
        for item in actual.get("hypotheses") or ()
        if item.get("root_cause_id") in roots
    ]
    generated_root_valid = any(
        expected.matches(root)
        for expected in truth.acceptable_root_causes
        for root in roots.values()
    )
    selected_root_valid = any(
        expected.matches(root)
        for expected in truth.acceptable_root_causes
        for root in selected_roots
    )
    preferred_roots = tuple(item for item in truth.acceptable_root_causes if item.preferred)
    selected_root_preferred = (
        any(expected.matches(root) for expected in preferred_roots for root in selected_roots)
        if preferred_roots else selected_root_valid
    )
    candidates = list(actual.get("candidates") or ())
    candidate_results = {
        item["id"]: _candidate_valid(item, truth) for item in candidates
    }
    valid_candidates = [item for item in candidates if candidate_results[item["id"]][0]]
    preferred_candidates = [
        item for item in valid_candidates
        if any(option.preferred and option.matches(item) for option in truth.acceptable_repairs)
    ]
    winner = next(
        (
            item for item in candidates
            if item.get("id") == actual.get("recommended_candidate_id")
        ),
        None,
    )
    winner_valid = candidate_results.get(winner.get("id"), (False, False, False))[0] if winner else None
    winner_preferred = (
        bool(winner_valid and any(option.preferred and option.matches(winner) for option in truth.acceptable_repairs))
        if winner and truth.preferred_repair_strategies else winner_valid
    )
    ambiguous_accepted = bool(
        truth.status is EvaluationStatus.MULTIPLE_VALID_ANSWERS
        and not truth.preferred_repair_strategies
        and actual.get("ranking_ambiguous")
    )
    forbidden_recommendation = bool(
        winner and winner.get("strategy_kind") in truth.forbidden_strategies
    )
    abstained = bool(actual.get("insufficient_evidence"))
    failure_codes: list[str] = []
    classification = "NONE"
    if truth.status is EvaluationStatus.INVALID:
        classification = "BENCHMARK_LABEL_FAILURE"
        failure_codes.append(classification)
    elif truth.status is EvaluationStatus.AMBIGUOUS:
        classification = "AMBIGUOUS_CASE"
        failure_codes.append(classification)
    elif truth.status is EvaluationStatus.INCOMPLETE:
        classification = "INCOMPLETE_GROUND_TRUTH"
        failure_codes.append(classification)
    elif truth.abstention_expected:
        if not abstained:
            classification = "TRUE_ENGINE_FAILURE"
            failure_codes.append("FALSE_ABSTENTION")
    elif abstained:
        classification = "TRUE_ENGINE_FAILURE"
        failure_codes.append("FALSE_ABSTENTION")
    else:
        if not generated_root_valid:
            failure_codes.append("CAUSAL_SLICE_ERROR")
        elif not selected_root_valid:
            failure_codes.append("ROOT_CAUSE_SELECTION_ERROR")
        if selected_root_valid and not valid_candidates:
            if any(value[1] for value in candidate_results.values()):
                failure_codes.append("REPAIR_TARGET_ERROR")
            else:
                failure_codes.append("REPAIR_SELECTION_ERROR")
        if valid_candidates and winner and not winner_valid:
            failure_codes.append("RANKING_ERROR")
        if forbidden_recommendation:
            failure_codes.append("UNSAFE_REPAIR")
        if failure_codes:
            classification = "TRUE_ENGINE_FAILURE"
    ranking_preference_hit = (
        True if ambiguous_accepted else winner_preferred if winner else False
    )
    return {
        "evaluation_status": truth.status.value,
        "evaluable": truth.evaluable,
        "classification": classification,
        "root_cause_validity": selected_root_valid if truth.evaluable and not truth.abstention_expected else None,
        "root_cause_preference_hit": selected_root_preferred if truth.evaluable and not truth.abstention_expected else None,
        "repair_validity": bool(valid_candidates) if truth.evaluable and not truth.abstention_expected else None,
        "repair_preference_hit": bool(preferred_candidates) if truth.evaluable and truth.preferred_repair_strategies else None,
        "valid_candidate_recall": len(valid_candidates) / len(candidates) if candidates and truth.evaluable else None,
        "preferred_candidate_hit": bool(preferred_candidates) if truth.evaluable and truth.preferred_repair_strategies else None,
        "top1_is_valid": winner_valid if truth.evaluable and not truth.abstention_expected else None,
        "top1_is_preferred": winner_preferred if truth.evaluable and truth.preferred_repair_strategies else None,
        "ranking_preference_hit": ranking_preference_hit if truth.evaluable and not truth.abstention_expected else None,
        "recommendation_coverage": bool(winner) if truth.evaluable and not truth.abstention_expected else None,
        "recommendation_valid": winner_valid if winner and truth.evaluable else None,
        "recommendation_preferred": winner_preferred if winner and truth.evaluable else None,
        "ambiguous_ranking_accepted": ambiguous_accepted,
        "generated_valid_root": generated_root_valid,
        "forbidden_recommendation": forbidden_recommendation,
        "failure_codes": list(dict.fromkeys(failure_codes)),
    }


def _average(values: Sequence[Any]) -> float | None:
    numbers = [float(item) for item in values if isinstance(item, (int, float, bool))]
    return sum(numbers) / len(numbers) if numbers else None


def summarize_benchmark_v4(
    base_report: Mapping[str, Any],
    adjudications: Sequence[BenchmarkAdjudication],
) -> dict[str, Any]:
    truth = {item.case_id: item for item in adjudications}
    enhanced: list[dict[str, Any]] = []
    for result in base_report.get("results") or ():
        metrics = adjudicate_predictive_result_v4(result, truth[result["id"]])
        enhanced.append(dict(result) | {"benchmark_v4": truth[result["id"]].as_dict(), "metrics_v4": metrics})
    evaluable = [item for item in enhanced if item["metrics_v4"]["evaluable"]]
    positives = [item for item in evaluable if not item["benchmark_v4"]["abstention_expected"]]
    recommended = [item for item in positives if item["metrics_v4"]["recommendation_coverage"]]
    names = (
        "root_cause_validity",
        "root_cause_preference_hit",
        "repair_validity",
        "repair_preference_hit",
        "valid_candidate_recall",
        "preferred_candidate_hit",
        "top1_is_valid",
        "top1_is_preferred",
        "ranking_preference_hit",
        "recommendation_coverage",
    )
    quality = {
        name: _average([item["metrics_v4"].get(name) for item in positives])
        for name in names
    }
    quality["recommendation_validity_precision"] = _average(
        [item["metrics_v4"]["recommendation_valid"] for item in recommended]
    )
    quality["recommendation_preference_precision"] = _average(
        [item["metrics_v4"]["recommendation_preferred"] for item in recommended]
    )
    status_counts = Counter(item.status.value for item in adjudications)
    classification_counts = Counter(item["metrics_v4"]["classification"] for item in enhanced)
    failure_counts = Counter(
        code for item in enhanced for code in item["metrics_v4"]["failure_codes"]
    )
    quality["benchmark_ambiguity_rate"] = (
        sum(status_counts[name] for name in ("MULTIPLE_VALID_ANSWERS", "AMBIGUOUS"))
        / len(adjudications) if adjudications else None
    )
    quality["engine_error_rate"] = (
        classification_counts["TRUE_ENGINE_FAILURE"] / len(evaluable) if evaluable else None
    )
    quality["benchmark_error_rate"] = (
        sum(status_counts[name] for name in ("AMBIGUOUS", "INCOMPLETE", "INVALID"))
        / len(adjudications) if adjudications else None
    )
    base_reasoning = base_report.get("reasoning") or {}
    hard_checks = {
        "evidence_reference_validity": base_report.get("retrieval", {}).get("evidence_ref_validity") == 1.0,
        "unsupported_claim_rate": base_reasoning.get("unsupported_claim_rate") is not None
        and base_reasoning["unsupported_claim_rate"] <= 0.05,
        "forbidden_recommendation_rate": base_reasoning.get("forbidden_candidate_recommendation_rate") == 0.0,
        "safety_invariants": bool(base_report.get("safety", {}).get("passed")),
        "latency": base_report.get("mode") != "live"
        or (base_report.get("latency", {}).get("qwen_request", {}).get("avg_ms") or 0) <= 60_000,
    }
    split = str(base_report.get("split") or "")
    targets = (
        {"root_cause_validity": 0.75, "repair_validity": 0.75, "top1_is_valid": 0.70,
         "recommendation_validity_precision": 0.70, "ranking_preference_hit": 0.55}
        if split == "holdout_v4"
        else {"root_cause_validity": 0.80, "repair_validity": 0.80, "top1_is_valid": 0.75,
              "recommendation_validity_precision": 0.75, "recommendation_coverage": 0.65,
              "ranking_preference_hit": 0.60}
    )
    quality_checks = {
        name: quality.get(name) is not None and quality[name] >= target
        for name, target in targets.items()
    }
    gate_checks = hard_checks | quality_checks | {"full_live_evaluation": base_report.get("mode") == "live"}
    return {
        "version": 4,
        "commit": base_report.get("commit"),
        "mode": base_report.get("mode"),
        "split": split,
        "cases": len(enhanced),
        "evaluable_cases": len(evaluable),
        "adjudication_status_counts": dict(sorted(status_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "retrieval": base_report.get("retrieval"),
        "grounding": {
            "evidence_reference_validity": base_report.get("retrieval", {}).get("evidence_ref_validity"),
            "unsupported_claim_rate": base_reasoning.get("unsupported_claim_rate"),
            "evidence_support_precision": base_reasoning.get("evidence_support_precision"),
        },
        "quality": quality,
        "safety": base_report.get("safety"),
        "latency": base_report.get("latency"),
        "qwen": base_report.get("qwen"),
        "legacy_v3_metrics": {
            "hypothesis_hit_v1_lexical": base_reasoning.get("hypothesis_hit"),
            "causal_root_hit_v3": base_reasoning.get("causal_root_hit"),
            "repair_strategy_hit_v3": base_reasoning.get("repair_strategy_hit"),
            "ranking_hit_v3": base_reasoning.get("ranking_hit"),
        },
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "stage_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "blockers": [name for name, passed in gate_checks.items() if not passed],
        },
        "results": enhanced,
    }


def evaluate_predictive_cases_v4(
    cases: Sequence[PredictiveCase],
    *,
    corpus_root: str | Path = CORPUS_ROOT,
    mode: str = "retrieval",
    reasoner: Any = None,
    runs: int = 1,
) -> dict[str, Any]:
    adjudications = load_benchmark_v4_adjudications(cases, corpus_root)
    base = evaluate_predictive_cases(cases, mode=mode, reasoner=reasoner, runs=runs)
    return summarize_benchmark_v4(base, adjudications)


def format_benchmark_v4(report: Mapping[str, Any]) -> str:
    def percent(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:.1f}%"

    quality = report.get("quality") or {}
    grounding = report.get("grounding") or {}
    gate = report.get("stage_gate") or {}
    return "\n".join(
        (
            f"Predictive benchmark v4 ({report.get('mode')}, {report.get('split')})",
            f"Cases: {report.get('cases')} ({report.get('evaluable_cases')} evaluable)",
            f"Root-cause validity: {percent(quality.get('root_cause_validity'))}",
            f"Repair validity: {percent(quality.get('repair_validity'))}",
            f"Top-1 validity: {percent(quality.get('top1_is_valid'))}",
            f"Ranking preference hit: {percent(quality.get('ranking_preference_hit'))}",
            f"Recommendation validity precision: {percent(quality.get('recommendation_validity_precision'))}",
            f"Recommendation coverage: {percent(quality.get('recommendation_coverage'))}",
            f"Evidence reference validity: {percent(grounding.get('evidence_reference_validity'))}",
            f"Unsupported claim rate: {percent(grounding.get('unsupported_claim_rate'))}",
            f"Stage gate: {'PASSED' if gate.get('passed') else 'FAILED'}",
            "Blockers: " + (", ".join(gate.get("blockers") or ()) or "none"),
        )
    )
