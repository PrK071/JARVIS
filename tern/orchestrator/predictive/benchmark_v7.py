from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_v5 import evaluate_predictive_cases_v5
from .evaluation import CORPUS_ROOT, PredictiveCase
from .robustness import coverage_funnel_v7, coverage_trace_v7


def _average(values: Sequence[object]) -> float | None:
    numbers = [float(item) for item in values if isinstance(item, (bool, int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def summarize_benchmark_v7(report_v5: Mapping[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for source in report_v5.get("results") or ():
        trace = coverage_trace_v7(source)
        results.append(dict(source) | {"metrics_v7": {"coverage": trace}})
    positives = [
        item for item in results
        if item.get("metrics_v5", {}).get("evaluable")
        and not (item.get("benchmark_v5") or {}).get("abstention_expected")
    ]
    def recovery_of(item: Mapping[str, Any]) -> Mapping[str, Any]:
        return (
            ((item.get("actual") or {}).get("recovery") or {})
            or ((item.get("retrieval") or {}).get("recovery") or {})
        )

    def recovered_decision_is_valid(item: Mapping[str, Any]) -> bool:
        metrics = item.get("metrics_v5", {})
        if report_v5.get("mode") == "live":
            return bool(
                metrics.get("root_cause_validity")
                and metrics.get("recommendation_valid")
            )
        return bool(
            (item.get("retrieval") or {}).get("root_cause_candidates")
            or (item.get("actual") or {}).get("root_cause_candidates")
        )

    attempts = [
        item for item in positives
        if recovery_of(item).get("attempted")
    ]
    successful = [
        item for item in attempts
        if recovery_of(item).get("succeeded")
        and recovered_decision_is_valid(item)
    ]
    changed = [
        item for item in attempts
        if recovery_of(item).get("actions")
    ]
    precise = [
        item for item in changed
        if recovered_decision_is_valid(item)
    ]
    false_after = sum(
        bool(item.get("metrics_v5", {}).get("false_abstention"))
        for item in positives
    )
    false_before = sum(
        not bool(recovery_of(item).get("initial_sufficiency", {}).get("sufficient"))
        for item in positives
    )
    extra_files = [
        int(recovery_of(item).get("extra_files") or 0)
        for item in attempts
    ]
    extra_atoms = [
        int(recovery_of(item).get("extra_evidence_atoms") or 0)
        for item in attempts
    ]
    quality = dict(report_v5.get("quality") or {})
    recovery = {
        "positive_cases": len(positives),
        "attempts": len(attempts),
        "successful": len(successful),
        "decision_changes": len(changed),
        "correct_decision_changes": len(precise),
        "attempt_rate": len(attempts) / len(positives) if positives else None,
        "success_rate": len(successful) / len(attempts) if attempts else None,
        "precision": len(precise) / len(changed) if changed else None,
        "noise_rate": (len(changed) - len(precise)) / len(changed) if changed else None,
        "avg_extra_files": _average(extra_files),
        "avg_extra_evidence_atoms": _average(extra_atoms),
        "false_abstention_before": false_before / len(positives) if positives else None,
        "false_abstention_after": false_after / len(positives) if positives else None,
    }
    grounding = report_v5.get("grounding") or {}
    safety = report_v5.get("safety") or {}
    split = str(report_v5.get("split"))
    holdout = split == "holdout_v7"
    quality_targets = {
        "root_cause_validity": 0.80 if holdout else 0.88,
        "repair_pair_validity": 0.80 if holdout else 0.88,
        "top1_validity": 0.80 if holdout else 0.85,
        "recommendation_validity_precision": 0.80 if holdout else 0.85,
    }
    checks = {
        "evidence_reference_validity": grounding.get("evidence_reference_validity") == 1.0,
        "unsupported_claim_rate": (
            grounding.get("unsupported_claim_rate") is not None
            and grounding["unsupported_claim_rate"] <= 0.05
        ),
        "forbidden_recommendations": (
            (report_v5.get("reasoning") or {}).get(
                "forbidden_candidate_recommendation_rate", 0.0
            ) == 0.0
        ),
        "safety": bool(safety.get("passed")),
        "false_abstention": (
            quality.get("false_abstention") is not None
            and quality["false_abstention"] <= (0.10 if holdout else 0.05)
        ),
    }
    checks.update({
        name: quality.get(name) is not None and quality[name] >= target
        for name, target in quality_targets.items()
    })
    if report_v5.get("mode") == "live":
        latency = ((report_v5.get("latency") or {}).get("qwen_request") or {}).get(
            "average_ms"
        )
        checks["latency"] = latency is not None and latency <= 45_000
    if attempts:
        checks["recovery_precision"] = recovery["precision"] is not None and recovery["precision"] >= (0.75 if holdout else 0.80)
        if not holdout:
            checks["recovery_success_rate"] = recovery["success_rate"] is not None and recovery["success_rate"] >= 0.70
    failure_counts = Counter(
        code for item in results for code in item.get("metrics_v5", {}).get("failure_codes", ())
    )
    return dict(report_v5) | {
        "version": 7,
        "coverage_funnel_v7": coverage_funnel_v7(results),
        "recovery_v7": recovery,
        "failure_code_counts_v7": dict(sorted(failure_counts.items())),
        "stage_gate_v7": {
            "passed": all(checks.values()),
            "checks": checks,
            "blockers": [name for name, passed in checks.items() if not passed],
        },
        "results": results,
    }


def evaluate_predictive_cases_v7(
    cases: Sequence[PredictiveCase],
    *,
    corpus_root: str | Path = CORPUS_ROOT,
    mode: str = "retrieval",
    reasoner: Any = None,
    runs: int = 1,
    analyzer_factory: Any = None,
) -> dict[str, Any]:
    return summarize_benchmark_v7(evaluate_predictive_cases_v5(
        cases,
        corpus_root=corpus_root,
        mode=mode,
        reasoner=reasoner,
        runs=runs,
        analyzer_factory=analyzer_factory,
    ))


def format_benchmark_v7(report: Mapping[str, Any]) -> str:
    quality = report.get("quality") or {}
    recovery = report.get("recovery_v7") or {}
    percent = lambda value: "n/a" if value is None else f"{float(value) * 100:.1f}%"
    return "\n".join((
        f"Predictive benchmark v7 ({report.get('mode')}, {report.get('split')})",
        f"Root-cause validity: {percent(quality.get('root_cause_validity'))}",
        f"Repair pair validity: {percent(quality.get('repair_pair_validity'))}",
        f"Top-1 validity: {percent(quality.get('top1_validity'))}",
        f"Recommendation precision: {percent(quality.get('recommendation_validity_precision'))}",
        f"False abstention: {percent(quality.get('false_abstention'))}",
        f"Recovery: attempts={recovery.get('attempts', 0)}, success={percent(recovery.get('success_rate'))}, precision={percent(recovery.get('precision'))}",
        f"Stage gate: {'PASSED' if (report.get('stage_gate_v7') or {}).get('passed') else 'FAILED'}",
    ))
