from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_v5 import (
    BenchmarkAdjudicationV5,
    evaluate_predictive_cases_v5,
    load_benchmark_v5_adjudications,
)
from .evaluation import CORPUS_ROOT, PredictiveCase
from .robustness import METRIC_DEFINITIONS, coverage_funnel, feature_provenance


def _rate_details(results: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    values = [
        item["metrics_v5"].get(name)
        for item in results
        if isinstance(item.get("metrics_v5", {}).get(name), (bool, int, float))
    ]
    numerator = sum(bool(value) for value in values)
    denominator = len(values)
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def summarize_benchmark_v6(report_v5: Mapping[str, Any]) -> dict[str, Any]:
    results = []
    for source in report_v5.get("results") or ():
        stage = coverage_funnel((source,))["loss_stage_counts"]
        loss = next((name for name, count in stage.items() if count), "RECOMMENDED")
        results.append(dict(source) | {"metrics_v6": {"abstention_stage": loss}})
    statuses = Counter(
        str((item.get("benchmark_v5") or {}).get("evaluation_status") or "UNKNOWN")
        for item in results
    )
    positives = [
        item for item in results
        if item.get("metrics_v5", {}).get("evaluable")
        and not (item.get("benchmark_v5") or {}).get("abstention_expected")
    ]
    recommended = [
        item for item in positives
        if (item.get("actual") or {}).get("recommended_candidate_id")
    ]
    metric_details = {
        name: _rate_details(results, name)
        for name in (
            "root_cause_validity",
            "repair_strategy_validity",
            "repair_target_validity",
            "repair_pair_validity",
            "top1_validity",
            "recommendation_coverage",
            "false_abstention",
        )
    }
    metric_details["false_abstention_rate"] = metric_details.pop("false_abstention")
    metric_details["recommendation_validity_precision"] = {
        "value": (
            sum(bool(item["metrics_v5"].get("recommendation_valid")) for item in recommended)
            / len(recommended)
            if recommended else None
        ),
        "numerator": sum(
            bool(item["metrics_v5"].get("recommendation_valid")) for item in recommended
        ),
        "denominator": len(recommended),
    }
    provenance = [feature_provenance(item.get("actual") or {}) for item in results]
    ratios = [item["structural_decision_ratio"] for item in provenance if item["structural_decision_ratio"] is not None]
    attempted = [
        item for item in positives
        if ((item.get("actual") or {}).get("retrieval_escalation") or {}).get("attempted")
    ]
    recovered = [
        item for item in attempted
        if ((item.get("actual") or {}).get("retrieval_escalation") or {}).get("succeeded")
        and item.get("metrics_v5", {}).get("generated_valid_root")
    ]
    recovery = {
        "n_positive": len(positives),
        "n_attempted": len(attempted),
        "n_recovered": len(recovered),
        "n_noise": len(attempted) - len(recovered),
        "attempt_rate": len(attempted) / len(positives) if positives else None,
        "success_rate": len(recovered) / len(attempted) if attempted else None,
        "noise_rate": (len(attempted) - len(recovered)) / len(attempted) if attempted else None,
    }
    grounding = report_v5.get("grounding") or {}
    safety = report_v5.get("safety") or {}
    average_qwen_ms = ((report_v5.get("latency") or {}).get("qwen_request") or {}).get(
        "average_ms"
    )
    holdout = report_v5.get("split") == "holdout_v6"
    targets = {
        "root_cause_validity": 0.80 if holdout else 0.85,
        "repair_pair_validity": 0.80 if holdout else 0.85,
        "top1_validity": 0.80 if holdout else 0.85,
        "recommendation_validity_precision": 0.80 if holdout else 0.85,
    }
    checks = {
        "full_live_evaluation": report_v5.get("mode") == "live",
        "evidence_reference_validity": grounding.get("evidence_reference_validity") == 1.0,
        "unsupported_claim_rate": grounding.get("unsupported_claim_rate") is not None
        and grounding["unsupported_claim_rate"] <= 0.05,
        "safety": bool(safety.get("passed")),
        "latency_gate": average_qwen_ms is not None and average_qwen_ms <= 45_000,
        "false_abstention_rate": metric_details["false_abstention_rate"]["value"] is not None
        and metric_details["false_abstention_rate"]["value"] <= 0.10,
    }
    checks.update({
        name: metric_details[name]["value"] is not None
        and metric_details[name]["value"] >= threshold
        for name, threshold in targets.items()
    })
    report = dict(report_v5)
    report.update({
        "version": 6,
        "metric_definitions": [item.as_dict() for item in METRIC_DEFINITIONS],
        "metric_denominators": metric_details,
        "population": {
            "n_total": len(results),
            "n_evaluable": sum(item.get("metrics_v5", {}).get("evaluable", False) for item in results),
            "n_positive": len(positives),
            "n_recommended": len(recommended),
            "n_abstained": sum(bool((item.get("actual") or {}).get("insufficient_evidence")) for item in positives),
            "n_ambiguous": statuses["AMBIGUOUS"],
            "n_invalid": statuses["INVALID"],
        },
        "coverage_funnel": {"scope": "all benchmark cases", **coverage_funnel(results)},
        "recovery": recovery,
        "feature_provenance": {
            "structural_decision_ratio": sum(ratios) / len(ratios) if ratios else None,
            "cases": provenance,
        },
        "stage_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "blockers": [name for name, passed in checks.items() if not passed],
        },
        "results": results,
    })
    return report


def evaluate_predictive_cases_v6(
    cases: Sequence[PredictiveCase], *, corpus_root: str | Path = CORPUS_ROOT,
    mode: str = "retrieval", reasoner: Any = None, runs: int = 1,
    analyzer_factory: Any = None,
) -> dict[str, Any]:
    return summarize_benchmark_v6(evaluate_predictive_cases_v5(
        cases, corpus_root=corpus_root, mode=mode, reasoner=reasoner, runs=runs,
        analyzer_factory=analyzer_factory,
    ))


def format_benchmark_v6(report: Mapping[str, Any]) -> str:
    quality = report.get("quality") or {}
    population = report.get("population") or {}
    percent = lambda value: "n/a" if value is None else f"{float(value) * 100:.1f}%"
    detail = report.get("metric_denominators") or {}

    def metric(name: str) -> str:
        item = detail.get(name) or {}
        return f"{percent(item.get('value'))} ({item.get('numerator', 0)}/{item.get('denominator', 0)})"

    return "\n".join((
        f"Predictive benchmark v6 ({report.get('mode')}, {report.get('split')})",
        (
            f"Population: total={population.get('n_total', 0)}, "
            f"evaluable={population.get('n_evaluable', 0)}, "
            f"recommended={population.get('n_recommended', 0)}, "
            f"abstained={population.get('n_abstained', 0)}"
        ),
        f"Root-cause validity: {metric('root_cause_validity')}",
        f"Repair pair validity: {metric('repair_pair_validity')}",
        f"Top-1 validity: {metric('top1_validity')}",
        f"Recommendation precision: {metric('recommendation_validity_precision')}",
        f"Recommendation coverage: {metric('recommendation_coverage')}",
        f"False abstention: {metric('false_abstention_rate')}",
        f"Unsupported claims: {percent((report.get('grounding') or {}).get('unsupported_claim_rate'))}",
        f"Structural decision ratio: {percent((report.get('feature_provenance') or {}).get('structural_decision_ratio'))}",
        f"Stage gate: {'PASSED' if (report.get('stage_gate') or {}).get('passed') else 'FAILED'}",
    ))
