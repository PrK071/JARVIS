from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .benchmark_v5 import (
    BenchmarkAdjudicationV5,
    _candidate_metrics,
    evaluate_predictive_cases_v5,
    load_benchmark_v5_adjudications,
)
from .evaluation import CORPUS_ROOT, PredictiveCase
from .scoring import SCORE_FEATURE_PROVENANCE, SCORE_WEIGHTS


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    numerator = sum(values)
    denominator = len(values)
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _signature_key(value: Mapping[str, Any] | None) -> tuple[object, ...] | None:
    if not value:
        return None
    return (
        value.get("cause_kind"),
        tuple(value.get("causal_roles") or ()),
        value.get("origin_role"),
        value.get("relation_to_failure"),
        value.get("contract_role"),
        value.get("entity_identity"),
        value.get("responsibility_kind"),
        value.get("defect_bearing_relation"),
    )


def _signatures_equivalent(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> bool:
    if _signature_key(left) == _signature_key(right):
        return _signature_key(left) is not None
    if not left or not right:
        return False
    flow_contract_family = {"NULL_FLOW", "TYPE_FLOW", "RETURN_CONTRACT"}
    left_roles = set(left.get("causal_roles") or ())
    right_roles = set(right.get("causal_roles") or ())
    return (
        left.get("cause_kind") in flow_contract_family
        and right.get("cause_kind") in flow_contract_family
        and {"PRODUCER", "RETURN_SOURCE"} <= left_roles
        and {"PRODUCER", "RETURN_SOURCE"} <= right_roles
        and left.get("origin_role") == right.get("origin_role") == "SOURCE"
    )


def _semantic_root_metrics(
    actual: Mapping[str, Any], truth: BenchmarkAdjudicationV5
) -> tuple[bool, bool]:
    roots = {item["id"]: item for item in actual.get("root_cause_candidates") or ()}
    signatures = actual.get("root_cause_signatures") or {}
    expected_ids = {
        root_id
        for root_id, root in roots.items()
        if any(
            item.matches(root)
            or (
                item.cause_kind == root.get("cause_kind") == "IMPORT_RESOLUTION"
                and item.path == root.get("origin_path")
                and item.symbol is not None
                and str(root.get("origin_symbol") or "").rsplit(".", 1)[-1]
                == item.symbol.rsplit(".", 1)[-1]
            )
            for item in truth.acceptable_root_causes
        )
    }
    for root_id, root in roots.items():
        identity = root.get("import_scc") or {}
        members = set(identity.get("member_modules") or ())
        edge_symbols = {
            str(edge.get("symbol") or "").rsplit(".", 1)[-1]
            for edge in identity.get("production_edges") or ()
        }
        if any(
            item.cause_kind == root.get("cause_kind") == "IMPORT_RESOLUTION"
            and item.path in members
            and (
                item.symbol is None
                or item.symbol.rsplit(".", 1)[-1] in edge_symbols
            )
            for item in truth.acceptable_root_causes
        ):
            expected_ids.add(root_id)
    nodes = {
        item["id"]: item
        for item in ((actual.get("causal_slice") or {}).get("nodes") or ())
    }
    flow_family = {"NULL_FLOW", "TYPE_FLOW", "RETURN_CONTRACT"}
    for root_id, root in roots.items():
        path_nodes = [nodes.get(item) for item in root.get("causal_path") or ()]
        path_symbols = {
            str(item.get("symbol") or "").rsplit(".", 1)[-1]
            for item in path_nodes if item
        }
        if any(
            item.path == root.get("origin_path")
            and item.cause_kind in flow_family
            and root.get("cause_kind") in flow_family
            and (item.symbol is None or item.symbol.rsplit(".", 1)[-1] in path_symbols)
            for item in truth.acceptable_root_causes
        ):
            expected_ids.add(root_id)
    valid_ids = set(expected_ids)
    for root_id, root in roots.items():
        if root_id in valid_ids:
            continue
        if any(
            root.get("origin_path") == roots[expected_id].get("origin_path")
            and root.get("origin_line") == roots[expected_id].get("origin_line")
            and _signatures_equivalent(
                signatures.get(root_id), signatures.get(expected_id)
            )
            for expected_id in expected_ids
        ):
            valid_ids.add(root_id)
    selected_ids = {
        item.get("root_cause_id")
        for item in actual.get("hypotheses") or ()
        if item.get("root_cause_id") in roots
    }
    return bool(valid_ids), bool(selected_ids & valid_ids)


def _metrics_v8(
    result: Mapping[str, Any], truth: BenchmarkAdjudicationV5
) -> dict[str, Any]:
    legacy = dict(result.get("metrics_v5") or {})
    generated, selected = _semantic_root_metrics(result.get("actual") or {}, truth)
    if truth.evaluable and not truth.abstention_expected:
        legacy["root_cause_validity"] = selected
    legacy["generated_valid_root"] = generated
    failures = list(legacy.get("failure_codes") or ())
    if generated:
        failures = [
            item for item in failures
            if item not in {"CAUSAL_SLICE_ERROR", "MISSED_STRUCTURAL_NEIGHBOR"}
        ]
    if selected:
        failures = [item for item in failures if item != "ROOT_CAUSE_SELECTION_ERROR"]
    elif (
        truth.evaluable
        and not truth.abstention_expected
        and not legacy.get("false_abstention")
        and "ROOT_CAUSE_SELECTION_ERROR" not in failures
    ):
        failures.append("ROOT_CAUSE_SELECTION_ERROR")
    legacy["failure_codes"] = failures
    legacy["classification"] = (
        "NONE"
        if not failures and legacy.get("classification") == "TRUE_ENGINE_FAILURE"
        else "TRUE_ENGINE_FAILURE"
        if failures and legacy.get("classification") == "NONE"
        else legacy.get("classification")
    )
    return legacy


def _pairwise_metrics(
    actual: Mapping[str, Any], truth: BenchmarkAdjudicationV5
) -> tuple[list[bool], list[bool]]:
    roots = {item["id"]: item for item in actual.get("root_cause_candidates") or ()}
    preferred_roots = {
        root_id
        for root_id, root in roots.items()
        if any(item.preferred and item.matches(root) for item in truth.acceptable_root_causes)
    }
    selected_ids = {
        item.get("root_cause_id")
        for item in actual.get("hypotheses") or ()
        if item.get("root_cause_id")
    }
    root_pairs = [
        bool(selected_ids & preferred_roots)
        for _preferred in preferred_roots
        for _other in roots.keys() - preferred_roots
    ]

    candidates = list(actual.get("candidates") or ())
    values = {item["id"]: _candidate_metrics(item, truth) for item in candidates}
    ranking_pairs: list[bool] = []
    for preferred in candidates:
        if not values[preferred["id"]][3]:
            continue
        for alternate in candidates:
            if alternate["id"] == preferred["id"] or values[alternate["id"]][3]:
                continue
            ranking_pairs.append(
                float(preferred.get("ranking_score") or 0.0)
                > float(alternate.get("ranking_score") or 0.0)
            )
    return root_pairs, ranking_pairs


def summarize_benchmark_v8(report_v5: Mapping[str, Any]) -> dict[str, Any]:
    root_pairs: list[bool] = []
    ranking_pairs: list[bool] = []
    margins = {"correct": [], "wrong": [], "ambiguous": []}
    results = []
    for result in report_v5.get("results") or ():
        truth = result.get("benchmark_v5") or {}
        typed_truth = load_benchmark_v5_adjudications_for_result(result, truth)
        metrics = _metrics_v8(result, typed_truth)
        result = dict(result) | {"metrics_v8": metrics}
        results.append(result)
        roots, rankings = _pairwise_metrics(result.get("actual") or {}, typed_truth)
        root_pairs.extend(roots)
        ranking_pairs.extend(rankings)
        margin = (result.get("actual") or {}).get("ranking_margin")
        if margin is not None:
            bucket = (
                "ambiguous"
                if (result.get("actual") or {}).get("ranking_ambiguous")
                else "correct"
                if metrics.get("top1_validity")
                else "wrong"
            )
            margins[bucket].append(float(margin))

    positive = [
        item for item in results
        if (item.get("metrics_v8") or {}).get("evaluable")
        and not (item.get("benchmark_v5") or {}).get("abstention_expected")
    ]
    recommended = [
        item for item in positive
        if (item.get("metrics_v8") or {}).get("recommendation_coverage")
    ]
    seed_attempts = [
        item for item in results
        if bool(((item.get("actual") or {}).get("recovery") or {}).get(
            "seed_synthesis_attempted"
        ))
    ]
    seed_successes = [
        item for item in seed_attempts
        if bool(((item.get("actual") or {}).get("recovery") or {}).get(
            "seed_synthesis_succeeded"
        ))
    ]
    seed_correct = [
        item for item in seed_successes
        if bool((item.get("metrics_v8") or {}).get("root_cause_validity"))
    ]
    quality_names = (
        "root_cause_validity", "repair_strategy_validity",
        "repair_target_validity", "repair_pair_validity", "top1_validity",
        "recommendation_coverage", "false_abstention",
    )
    denominators = {
        name: _rate([
            bool((item.get("metrics_v8") or {}).get(name)) for item in positive
        ])
        for name in quality_names
    } | {
        "recommendation_validity_precision": _rate([
            bool((item.get("metrics_v8") or {}).get("recommendation_valid"))
            for item in recommended
        ]),
        "root_pairwise_accuracy": _rate(root_pairs),
        "candidate_pairwise_ranking_accuracy": _rate(ranking_pairs),
    }
    quality = dict(report_v5.get("quality") or {}) | {
        name: value["value"] for name, value in denominators.items()
    }
    split = str(report_v5.get("split") or "")
    holdout = split == "holdout_v8"
    thresholds = {
        "root_cause_validity": 0.82 if holdout else 0.90,
        "repair_pair_validity": 0.82 if holdout else 0.90,
        "top1_validity": 0.82 if holdout else 0.88,
        "recommendation_validity_precision": 0.82 if holdout else 0.88,
        "candidate_pairwise_ranking_accuracy": 0.80,
    }
    checks = {
        name: quality.get(name) is not None and float(quality[name]) >= limit
        for name, limit in thresholds.items()
    }
    checks |= {
        "evidence_reference_validity": (report_v5.get("grounding") or {}).get("evidence_reference_validity") == 1.0,
        "unsupported_claim_rate": ((report_v5.get("grounding") or {}).get("unsupported_claim_rate") or 0.0) <= 0.05,
        "forbidden_recommendations": int((report_v5.get("safety") or {}).get("forbidden_candidate_recommendations", 0)) == 0,
        "safety": bool((report_v5.get("safety") or {}).get("passed")),
        "latency": report_v5.get("mode") != "live" or (
            (report_v5.get("latency") or {})
            .get("qwen_request", {})
            .get("average_ms", 0)
            <= 45_000
        ),
        "full_live_evaluation": report_v5.get("mode") == "live",
    }
    if holdout:
        checks["false_abstention"] = (quality.get("false_abstention") or 0.0) <= 0.08
    return dict(report_v5) | {
        "version": 8,
        "results": results,
        "quality": quality,
        "metric_denominators_v8": denominators,
        "ranking_margin_distribution": {
            key: {
                "count": len(values),
                "average": sum(values) / len(values) if values else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
            for key, values in margins.items()
        },
        "ranking_features": [
            {
                "name": name,
                "source": SCORE_FEATURE_PROVENANCE[name],
                "range": [0.0, 1.0],
                "weight": weight,
            }
            for name, weight in SCORE_WEIGHTS.items()
        ],
        "seed_synthesis_v8": {
            "attempts": len(seed_attempts),
            "successes": len(seed_successes),
            "correct_diagnoses": len(seed_correct),
            "success_rate": (
                len(seed_successes) / len(seed_attempts) if seed_attempts else None
            ),
            "precision": (
                len(seed_correct) / len(seed_successes) if seed_successes else None
            ),
        },
        "stage_gate_v8": {"checks": checks, "passed": all(checks.values())},
    }


def load_benchmark_v5_adjudications_for_result(
    result: Mapping[str, Any], value: Mapping[str, Any]
) -> BenchmarkAdjudicationV5:
    from .benchmark_v4 import CausalTruth, EvaluationStatus
    from .repair import RepairTarget, RepairTargetKind
    from .benchmark_v5 import RepairTruthV5

    roots = tuple(CausalTruth(
        str(item["path"]), item.get("symbol"), str(item["cause_kind"]),
        tuple(item.get("evidence") or ()), bool(item.get("preferred")),
    ) for item in value.get("acceptable_root_causes") or ())
    repairs = tuple(RepairTruthV5(
        str(item["strategy"]),
        tuple(RepairTarget(
            path=str(target["path"]),
            scope_kind=RepairTargetKind(str(target["scope_kind"])),
            symbol=target.get("symbol"), parameter=target.get("parameter"),
            attribute=target.get("attribute"), expression_id=target.get("expression_id"),
            line=int(target["line"]) if target.get("line") is not None else None,
        ) for target in item.get("targets") or ()),
        bool(item.get("preferred")),
    ) for item in value.get("acceptable_repairs") or ())
    return BenchmarkAdjudicationV5(
        str(result["id"]), EvaluationStatus(str(value["status"])),
        value.get("failure_site"), roots, repairs,
        bool(value.get("abstention_expected")), str(value.get("notes") or ""),
    )


def evaluate_predictive_cases_v8(
    cases: Sequence[PredictiveCase], *, corpus_root: str | Path = CORPUS_ROOT,
    mode: str = "retrieval", reasoner: Any = None, runs: int = 1,
    analyzer_factory: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    return summarize_benchmark_v8(evaluate_predictive_cases_v5(
        cases, corpus_root=corpus_root, mode=mode, reasoner=reasoner,
        runs=runs, analyzer_factory=analyzer_factory,
    ))


def format_benchmark_v8(report: Mapping[str, Any]) -> str:
    quality = report.get("quality") or {}
    percent = lambda value: "n/a" if value is None else f"{float(value) * 100:.1f}%"
    return "\n".join((
        f"Predictive benchmark v8 ({report.get('mode')}, {report.get('split')})",
        f"Cases: {report.get('cases')} ({report.get('evaluable_cases')} evaluable)",
        f"Root validity: {percent(quality.get('root_cause_validity'))}",
        f"Repair-pair validity: {percent(quality.get('repair_pair_validity'))}",
        f"Top-1 validity: {percent(quality.get('top1_validity'))}",
        f"Recommendation precision: {percent(quality.get('recommendation_validity_precision'))}",
        f"Root pairwise accuracy: {percent(quality.get('root_pairwise_accuracy'))}",
        f"Candidate pairwise accuracy: {percent(quality.get('candidate_pairwise_ranking_accuracy'))}",
        f"Stage gate: {'PASS' if (report.get('stage_gate_v8') or {}).get('passed') else 'FAIL'}",
    ))


def summarize_robustness_v8(
    metamorphic: Mapping[str, Any], counterfactual: Mapping[str, Any]
) -> dict[str, Any]:
    base = dict(metamorphic.get("metrics") or {}) | dict(
        counterfactual.get("metrics") or {}
    )
    rows = list(metamorphic.get("results") or ())
    wrapper_rows = [
        item for item in rows if item.get("transformation") == "WRAPPER_FUNCTION"
    ]
    wrapper_checks = [
        bool((item.get("checks") or {}).get("root_cause_invariance"))
        and bool((item.get("checks") or {}).get("repair_strategy_invariance"))
        for item in wrapper_rows
    ]
    metrics = base | {
        "root_signature_invariance": base.get("root_cause_invariance"),
        "repair_invariance": base.get("repair_strategy_invariance"),
        "target_signature_invariance": base.get("repair_target_invariance"),
        "wording_invariance": base.get("problem_wording_invariance"),
        "wrapper_invariance": _rate(wrapper_checks)["value"],
        "root_sensitivity": base.get("counterfactual_root_sensitivity"),
        "repair_sensitivity": base.get("counterfactual_repair_sensitivity"),
        "target_sensitivity": base.get("counterfactual_target_sensitivity"),
        "variant_root_validity": base.get("variant_root_cause_validity"),
        "variant_repair_pair_validity": base.get("variant_repair_pair_validity"),
        "variant_top1_validity": base.get("variant_top1_validity"),
    }
    holdout = metamorphic.get("split") == "holdout_v8"
    targets = {
        "root_signature_invariance": 0.88 if holdout else 0.92,
        "repair_invariance": 0.88 if holdout else 0.92,
        "wording_invariance": 0.85 if holdout else 0.90,
        "variant_root_validity": 0.82 if holdout else 0.88,
        "variant_repair_pair_validity": 0.82 if holdout else 0.88,
    }
    if not holdout:
        targets |= {
            "target_signature_invariance": 0.92,
            "wrapper_invariance": 0.90,
            "variant_top1_validity": 0.85,
            "root_sensitivity": 0.90,
            "repair_sensitivity": 0.90,
        }
    checks = {
        name: metrics.get(name) is not None and float(metrics[name]) >= threshold
        for name, threshold in targets.items()
    }
    safety_names = (
        "filesystem_mutations", "tool_dispatches", "execution_authorized",
        "authority_grants", "destructive_actions",
        "forbidden_candidate_recommendations",
    )
    safety = {
        name: int((metamorphic.get("safety") or {}).get(name, 0))
        + int((counterfactual.get("safety") or {}).get(name, 0))
        for name in safety_names
    }
    safety["passed"] = not any(safety.values())
    return {
        "version": 8,
        "mode": metamorphic.get("mode"),
        "split": metamorphic.get("split"),
        "metrics": metrics,
        "metric_denominators": dict(metamorphic.get("metric_denominators") or {})
        | dict(counterfactual.get("metric_denominators") or {})
        | {"wrapper_invariance": _rate(wrapper_checks)},
        "metamorphic": metamorphic,
        "counterfactual": counterfactual,
        "safety": safety,
        "stage_gate_v8": {"checks": checks | {"safety": safety["passed"]}, "passed": all(checks.values()) and safety["passed"]},
    }


def format_robustness_v8(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    percent = lambda value: "n/a" if value is None else f"{float(value) * 100:.1f}%"
    return "\n".join((
        f"Predictive robustness v8 ({report.get('mode')}, {report.get('split')})",
        f"Root signature invariance: {percent(metrics.get('root_signature_invariance'))}",
        f"Repair invariance: {percent(metrics.get('repair_invariance'))}",
        f"Target signature invariance: {percent(metrics.get('target_signature_invariance'))}",
        f"Wording invariance: {percent(metrics.get('wording_invariance'))}",
        f"Wrapper invariance: {percent(metrics.get('wrapper_invariance'))}",
        f"Stage gate: {'PASS' if (report.get('stage_gate_v8') or {}).get('passed') else 'FAIL'}",
    ))
