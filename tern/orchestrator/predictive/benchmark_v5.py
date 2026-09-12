from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_v4 import (
    CausalTruth,
    EvaluationStatus,
    EVALUABLE_STATUSES,
    _reference_exists,
    load_benchmark_v4_adjudications,
)
from .evaluation import CORPUS_ROOT, PredictiveCase, evaluate_predictive_cases
from .repair import RepairTarget, RepairTargetKind, targets_compatible


@dataclass(frozen=True)
class RepairTruthV5:
    strategy: str
    targets: tuple[RepairTarget, ...]
    preferred: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "targets": [item.as_dict() for item in self.targets],
            "preferred": self.preferred,
        }


@dataclass(frozen=True)
class BenchmarkAdjudicationV5:
    case_id: str
    status: EvaluationStatus
    failure_site: Mapping[str, Any] | None
    acceptable_root_causes: tuple[CausalTruth, ...]
    acceptable_repairs: tuple[RepairTruthV5, ...]
    abstention_expected: bool
    notes: str

    @property
    def evaluable(self) -> bool:
        return self.status in EVALUABLE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status.value,
            "failure_site": dict(self.failure_site) if self.failure_site else None,
            "acceptable_root_causes": [item.as_dict() for item in self.acceptable_root_causes],
            "acceptable_repairs": [item.as_dict() for item in self.acceptable_repairs],
            "abstention_expected": self.abstention_expected,
            "notes": self.notes,
        }


def _target(value: Mapping[str, Any], case: PredictiveCase) -> RepairTarget:
    path = str(value.get("path") or "")
    if not path or not (case.fixture_root / path).is_file():
        raise ValueError(f"{case.id}: repair target path does not exist")
    return RepairTarget(
        path=path,
        scope_kind=RepairTargetKind(str(value["scope_kind"])),
        symbol=value.get("symbol"),
        parameter=value.get("parameter"),
        attribute=value.get("attribute"),
        expression_id=value.get("expression_id"),
        line=int(value["line"]) if value.get("line") is not None else None,
    )


def _from_v5(value: Mapping[str, Any], case: PredictiveCase) -> BenchmarkAdjudicationV5:
    if str(value.get("case_id") or "") != case.id:
        raise ValueError(f"{case.id}: adjudication id mismatch")
    status = EvaluationStatus(str(value["status"]))
    site = value.get("failure_site")
    if site is not None:
        path = case.fixture_root / str(site.get("path") or "")
        line = int(site.get("line") or 0)
        if not path.is_file() or not 1 <= line <= len(path.read_text(encoding="utf-8").splitlines()):
            raise ValueError(f"{case.id}: failure site does not exist")
    roots: list[CausalTruth] = []
    for item in value.get("acceptable_root_causes") or ():
        path = str(item.get("path") or "")
        evidence = tuple(str(ref) for ref in item.get("evidence") or ())
        if not path or not (case.fixture_root / path).is_file():
            raise ValueError(f"{case.id}: root path does not exist")
        if not evidence or any(not _reference_exists(ref, case.fixture_root) for ref in evidence):
            raise ValueError(f"{case.id}: root evidence is invalid")
        roots.append(CausalTruth(
            path, item.get("symbol"), str(item.get("cause_kind") or ""),
            evidence, bool(item.get("preferred")),
        ))
    repairs: list[RepairTruthV5] = []
    for item in value.get("acceptable_repairs") or ():
        strategy = str(item.get("strategy") or "")
        targets = tuple(_target(target, case) for target in item.get("targets") or ())
        if not strategy or not targets:
            raise ValueError(f"{case.id}: repair strategy and targets are required")
        repairs.append(RepairTruthV5(strategy, targets, bool(item.get("preferred"))))
    abstention = value.get("abstention_expected")
    if not isinstance(abstention, bool):
        raise ValueError(f"{case.id}: abstention_expected must be boolean")
    if status in EVALUABLE_STATUSES and not abstention and (not roots or not repairs):
        raise ValueError(f"{case.id}: positive evaluable case requires roots and repairs")
    return BenchmarkAdjudicationV5(
        case.id, status, site, tuple(roots), tuple(repairs), abstention,
        str(value.get("notes") or ""),
    )


def _legacy_target_kinds(strategy: str) -> tuple[RepairTargetKind, ...]:
    return {
        "CORRECT_ARGUMENT": (RepairTargetKind.CALL_SITE, RepairTargetKind.PARAMETER),
        "CORRECT_RETURN_VALUE": (RepairTargetKind.RETURN_SITE, RepairTargetKind.FUNCTION),
        "FIX_PRODUCER": (RepairTargetKind.FUNCTION, RepairTargetKind.RETURN_SITE, RepairTargetKind.PARAMETER, RepairTargetKind.ATTRIBUTE),
        "FIX_CONSUMER_CONTRACT": (RepairTargetKind.FUNCTION, RepairTargetKind.PARAMETER),
        "VALIDATE_BOUNDARY": (RepairTargetKind.FUNCTION, RepairTargetKind.PARAMETER, RepairTargetKind.CALL_SITE),
        "CORRECT_CONTROL_FLOW": (RepairTargetKind.FUNCTION, RepairTargetKind.RETURN_SITE),
        "CORRECT_CONFIGURATION": (RepairTargetKind.CONFIG_VALUE, RepairTargetKind.MODULE),
        "CORRECT_IMPORT": (RepairTargetKind.IMPORT_EDGE,),
        "CORRECT_TEST_EXPECTATION": (RepairTargetKind.TEST_EXPECTATION,),
    }.get(strategy, (RepairTargetKind.MODULE,))


def _adapt_v4(case: PredictiveCase, value: Any) -> BenchmarkAdjudicationV5:
    repairs: list[RepairTruthV5] = []
    for item in value.acceptable_repairs:
        targets: list[RepairTarget] = []
        symbols = item.target_symbols or (None,)
        for path in item.target_files:
            for symbol in symbols:
                for kind in _legacy_target_kinds(item.strategy):
                    kwargs: dict[str, Any] = {"path": path, "scope_kind": kind, "symbol": symbol}
                    if kind is RepairTargetKind.PARAMETER:
                        kwargs["parameter"] = symbol
                        kwargs["symbol"] = None
                    elif kind is RepairTargetKind.ATTRIBUTE:
                        kwargs["attribute"] = symbol
                        kwargs["symbol"] = None
                    targets.append(RepairTarget(**kwargs))
        repairs.append(RepairTruthV5(item.strategy, tuple(targets), item.preferred))
    return BenchmarkAdjudicationV5(
        case.id, value.status, value.failure_site, value.acceptable_root_causes,
        tuple(repairs), value.abstention_expected, value.notes,
    )


def load_benchmark_v5_adjudications(
    cases: Sequence[PredictiveCase],
    corpus_root: str | Path = CORPUS_ROOT,
) -> tuple[BenchmarkAdjudicationV5, ...]:
    root = Path(corpus_root).resolve()
    case_lookup = {case.id: case for case in cases}
    raw_values: dict[str, Mapping[str, Any]] = {}
    for path in sorted((root / "v5" / "adjudications").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            case_id = str(raw.get("case_id") or "")
            if case_id in case_lookup:
                if case_id in raw_values:
                    raise ValueError(f"duplicate benchmark v5 adjudication id: {case_id}")
                raw_values[case_id] = raw
    result = {
        case_id: _from_v5(raw, case_lookup[case_id])
        for case_id, raw in raw_values.items()
    }
    legacy_cases = [case for case in cases if case.id not in result]
    if legacy_cases:
        for item in load_benchmark_v4_adjudications(legacy_cases, root):
            result[item.case_id] = _adapt_v4(case_lookup[item.case_id], item)
    missing = sorted(set(case_lookup) - set(result))
    if missing:
        raise ValueError(f"missing benchmark v5 adjudications: {', '.join(missing)}")
    return tuple(result[key] for key in sorted(result))


def _actual_targets(candidate: Mapping[str, Any]) -> tuple[RepairTarget, ...]:
    structured = candidate.get("repair_targets") or ()
    if structured:
        return tuple(RepairTarget(
            path=str(item["path"]),
            scope_kind=RepairTargetKind(str(item["scope_kind"])),
            symbol=item.get("symbol"), parameter=item.get("parameter"),
            attribute=item.get("attribute"), expression_id=item.get("expression_id"),
            line=int(item["line"]) if item.get("line") is not None else None,
        ) for item in structured)
    strategy = str(candidate.get("strategy_kind") or "OTHER")
    symbols = tuple(candidate.get("target_symbols") or ()) or (None,)
    targets: list[RepairTarget] = []
    for path in candidate.get("target_files") or ():
        for symbol in symbols:
            for kind in _legacy_target_kinds(strategy):
                kwargs: dict[str, Any] = {"path": path, "scope_kind": kind, "symbol": symbol}
                if kind is RepairTargetKind.PARAMETER:
                    kwargs["parameter"], kwargs["symbol"] = symbol, None
                elif kind is RepairTargetKind.ATTRIBUTE:
                    kwargs["attribute"], kwargs["symbol"] = symbol, None
                targets.append(RepairTarget(**kwargs))
    return tuple(targets)


def _candidate_metrics(
    candidate: Mapping[str, Any], truth: BenchmarkAdjudicationV5
) -> tuple[bool, bool, bool, bool]:
    strategy = str(candidate.get("strategy_kind") or "")
    actual_targets = _actual_targets(candidate)
    strategy_valid = any(option.strategy == strategy for option in truth.acceptable_repairs)
    target_valid = any(
        targets_compatible(actual, expected)
        for option in truth.acceptable_repairs
        for expected in option.targets
        for actual in actual_targets
    )
    pair_valid = any(
        option.strategy == strategy
        and any(targets_compatible(actual, expected) for actual in actual_targets for expected in option.targets)
        for option in truth.acceptable_repairs
    )
    preferred = any(
        option.preferred and option.strategy == strategy
        and any(targets_compatible(actual, expected) for actual in actual_targets for expected in option.targets)
        for option in truth.acceptable_repairs
    )
    return strategy_valid, target_valid, pair_valid, preferred


def adjudicate_predictive_result_v5(
    result: Mapping[str, Any], truth: BenchmarkAdjudicationV5
) -> dict[str, Any]:
    actual = result.get("actual") or {}
    roots = {item["id"]: item for item in actual.get("root_cause_candidates") or ()}
    selected = [
        roots[item["root_cause_id"]]
        for item in actual.get("hypotheses") or ()
        if item.get("root_cause_id") in roots
    ]
    root_generated = any(expected.matches(root) for expected in truth.acceptable_root_causes for root in roots.values())
    root_valid = any(expected.matches(root) for expected in truth.acceptable_root_causes for root in selected)
    candidates = list(actual.get("candidates") or ())
    candidate_metrics = {item["id"]: _candidate_metrics(item, truth) for item in candidates}
    strategy_valid = any(value[0] for value in candidate_metrics.values())
    target_valid = any(value[1] for value in candidate_metrics.values())
    pair_valid = any(value[2] for value in candidate_metrics.values())
    winner = next((item for item in candidates if item.get("id") == actual.get("recommended_candidate_id")), None)
    winner_values = candidate_metrics.get(winner.get("id"), (False, False, False, False)) if winner else None
    abstained = bool(actual.get("insufficient_evidence"))
    failures: list[str] = []
    classification = "NONE"
    if truth.status is EvaluationStatus.INVALID:
        classification, failures = "BENCHMARK_LABEL_FAILURE", ["BENCHMARK_LABEL_FAILURE"]
    elif truth.status is EvaluationStatus.AMBIGUOUS:
        classification, failures = "AMBIGUOUS_CASE", ["AMBIGUOUS_CASE"]
    elif truth.status is EvaluationStatus.INCOMPLETE:
        classification, failures = "INCOMPLETE_GROUND_TRUTH", ["INCOMPLETE_GROUND_TRUTH"]
    elif truth.abstention_expected:
        if not abstained:
            classification, failures = "TRUE_ENGINE_FAILURE", ["FAILED_TO_ABSTAIN"]
    elif abstained:
        classification, failures = "TRUE_ENGINE_FAILURE", ["FALSE_ABSTENTION"]
    else:
        if not root_generated:
            failures.append("MISSED_STRUCTURAL_NEIGHBOR" if actual.get("retrieval_escalation", {}).get("attempted") else "CAUSAL_SLICE_ERROR")
        elif not root_valid:
            failures.append("ROOT_CAUSE_SELECTION_ERROR")
        if root_valid and not strategy_valid:
            failures.append("REPAIR_STRATEGY_ERROR")
        elif strategy_valid and not target_valid:
            failures.append("REPAIR_TARGET_ERROR")
        elif strategy_valid and target_valid and not pair_valid:
            failures.append("REPAIR_TARGET_GRANULARITY_ERROR")
        if winner and winner_values and not winner_values[2]:
            failures.append("RANKING_ERROR")
        if failures:
            classification = "TRUE_ENGINE_FAILURE"
    positive = truth.evaluable and not truth.abstention_expected
    return {
        "evaluation_status": truth.status.value,
        "evaluable": truth.evaluable,
        "classification": classification,
        "root_cause_validity": root_valid if positive else None,
        "repair_strategy_validity": strategy_valid if positive else None,
        "repair_target_validity": target_valid if positive else None,
        "repair_pair_validity": pair_valid if positive else None,
        "target_preference_hit": winner_values[3] if positive and winner_values else None,
        "top1_validity": winner_values[2] if positive and winner_values else None,
        "recommendation_coverage": bool(winner) if positive else None,
        "recommendation_valid": winner_values[2] if positive and winner_values else None,
        "false_abstention": abstained if positive else None,
        "generated_valid_root": root_generated,
        "failure_codes": list(dict.fromkeys(failures)),
    }


def _average(values: Sequence[Any]) -> float | None:
    numbers = [float(item) for item in values if isinstance(item, (bool, int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def summarize_benchmark_v5(
    base_report: Mapping[str, Any],
    adjudications: Sequence[BenchmarkAdjudicationV5],
) -> dict[str, Any]:
    truths = {item.case_id: item for item in adjudications}
    results = []
    for result in base_report.get("results") or ():
        metrics = adjudicate_predictive_result_v5(result, truths[result["id"]])
        results.append(dict(result) | {"benchmark_v5": truths[result["id"]].as_dict(), "metrics_v5": metrics})
    positives = [item for item in results if item["metrics_v5"]["evaluable"] and not item["benchmark_v5"]["abstention_expected"]]
    recommended = [item for item in positives if item["metrics_v5"]["recommendation_coverage"]]
    quality_names = (
        "root_cause_validity", "repair_strategy_validity", "repair_target_validity",
        "repair_pair_validity", "target_preference_hit", "top1_validity",
        "recommendation_coverage", "false_abstention",
    )
    quality = {name: _average([item["metrics_v5"].get(name) for item in positives]) for name in quality_names}
    quality["recommendation_validity_precision"] = _average([
        item["metrics_v5"]["recommendation_valid"] for item in recommended
    ])
    quality["ranking_preference_hit"] = quality["target_preference_hit"]
    import_cases = [item for item in positives if any(root.cause_kind == "IMPORT_RESOLUTION" for root in truths[item["id"]].acceptable_root_causes)]
    quality["import_cycle_detection_recall"] = _average([item["metrics_v5"]["generated_valid_root"] for item in import_cases])
    generated_imports = [
        root for item in results for root in (item.get("actual") or {}).get("root_cause_candidates") or ()
        if root.get("cause_kind") == "IMPORT_RESOLUTION"
    ]
    matched_imports = sum(
        any(expected.matches(root) for expected in truths[item["id"]].acceptable_root_causes)
        for item in results for root in (item.get("actual") or {}).get("root_cause_candidates") or ()
        if root.get("cause_kind") == "IMPORT_RESOLUTION"
    )
    quality["import_cycle_detection_precision"] = matched_imports / len(generated_imports) if generated_imports else None
    escalated = [item for item in results if (item.get("actual") or {}).get("retrieval_escalation", {}).get("attempted")]
    quality["retrieval_escalation_rate"] = len(escalated) / len(results) if results else None
    quality["retrieval_escalation_success"] = _average([
        item["actual"]["retrieval_escalation"].get("succeeded") for item in escalated
    ])
    reasoning = base_report.get("reasoning") or {}
    safety = base_report.get("safety") or {}
    hard = {
        "evidence_reference_validity": base_report.get("retrieval", {}).get("evidence_ref_validity") == 1.0,
        "unsupported_claim_rate": reasoning.get("unsupported_claim_rate") is not None and reasoning["unsupported_claim_rate"] <= 0.05,
        "forbidden_recommendations": reasoning.get("forbidden_candidate_recommendation_rate") == 0.0,
        "safety": bool(safety.get("passed")),
        "latency": base_report.get("mode") != "live" or (base_report.get("latency", {}).get("qwen_request", {}).get("avg_ms") or 0) <= 45_000,
    }
    holdout = str(base_report.get("split")) == "holdout_v5"
    targets = {
        "root_cause_validity": 0.80 if holdout else 0.85,
        "repair_strategy_validity": 0.80 if holdout else 0.90,
        "repair_target_validity": 0.75 if holdout else 0.85,
        "repair_pair_validity": 0.75 if holdout else 0.85,
        "top1_validity": 0.75 if holdout else 0.85,
        "recommendation_validity_precision": 0.75 if holdout else 0.85,
    }
    if not holdout:
        targets |= {"recommendation_coverage": 0.70}
    checks = hard | {name: quality.get(name) is not None and quality[name] >= threshold for name, threshold in targets.items()}
    checks["false_abstention"] = quality.get("false_abstention") is not None and quality["false_abstention"] <= 0.05
    checks["full_live_evaluation"] = base_report.get("mode") == "live"
    status_counts = Counter(item.status.value for item in adjudications)
    classification_counts = Counter(item["metrics_v5"]["classification"] for item in results)
    failure_counts = Counter(code for item in results for code in item["metrics_v5"]["failure_codes"])
    return {
        "version": 5,
        "commit": base_report.get("commit"), "mode": base_report.get("mode"),
        "split": base_report.get("split"), "cases": len(results),
        "evaluable_cases": sum(item.evaluable for item in adjudications),
        "adjudication_status_counts": dict(sorted(status_counts.items())),
        "classification_counts": dict(sorted(classification_counts.items())),
        "retrieval": base_report.get("retrieval"),
        "grounding": {
            "evidence_reference_validity": base_report.get("retrieval", {}).get("evidence_ref_validity"),
            "unsupported_claim_rate": reasoning.get("unsupported_claim_rate"),
            "evidence_support_precision": reasoning.get("evidence_support_precision"),
        },
        "quality": quality, "safety": safety,
        "latency": base_report.get("latency"), "qwen": base_report.get("qwen"),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "stage_gate": {"passed": all(checks.values()), "checks": checks, "blockers": [name for name, passed in checks.items() if not passed]},
        "results": results,
    }


def evaluate_predictive_cases_v5(
    cases: Sequence[PredictiveCase], *, corpus_root: str | Path = CORPUS_ROOT,
    mode: str = "retrieval", reasoner: Any = None, runs: int = 1,
    analyzer_factory: Any = None,
) -> dict[str, Any]:
    truths = load_benchmark_v5_adjudications(cases, corpus_root)
    return summarize_benchmark_v5(
        evaluate_predictive_cases(
            cases, mode=mode, reasoner=reasoner, runs=runs,
            analyzer_factory=analyzer_factory,
        ), truths
    )


def format_benchmark_v5(report: Mapping[str, Any]) -> str:
    percent = lambda value: "n/a" if value is None else f"{float(value) * 100:.1f}%"
    quality, gate = report.get("quality") or {}, report.get("stage_gate") or {}
    return "\n".join((
        f"Predictive benchmark v5 ({report.get('mode')}, {report.get('split')})",
        f"Cases: {report.get('cases')} ({report.get('evaluable_cases')} evaluable)",
        f"Root-cause validity: {percent(quality.get('root_cause_validity'))}",
        f"Repair strategy validity: {percent(quality.get('repair_strategy_validity'))}",
        f"Repair target validity: {percent(quality.get('repair_target_validity'))}",
        f"Repair pair validity: {percent(quality.get('repair_pair_validity'))}",
        f"Top-1 validity: {percent(quality.get('top1_validity'))}",
        f"Recommendation precision: {percent(quality.get('recommendation_validity_precision'))}",
        f"Recommendation coverage: {percent(quality.get('recommendation_coverage'))}",
        f"False abstention: {percent(quality.get('false_abstention'))}",
        f"Stage gate: {'PASSED' if gate.get('passed') else 'FAILED'}",
        "Blockers: " + (", ".join(gate.get("blockers") or ()) or "none"),
    ))
