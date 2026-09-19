from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .evaluation import (
    CORPUS_ROOT,
    PredictiveCase,
    evaluate_predictive_cases,
)
from .benchmark_v5 import (
    adjudicate_predictive_result_v5,
    load_benchmark_v5_adjudications,
)
from .benchmark_v8 import _metrics_v8
from .benchmark_v8 import summarize_robustness_v8


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    numerator = sum(values)
    denominator = len(values)
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _selected_root(actual: Mapping[str, Any]) -> str | None:
    return next((
        str(item["root_cause_id"])
        for item in actual.get("hypotheses") or ()
        if item.get("root_cause_id")
    ), None)


def _recommended_candidate(actual: Mapping[str, Any]) -> Mapping[str, Any] | None:
    recommended = actual.get("recommended_candidate_id")
    return next((
        item for item in actual.get("candidates") or ()
        if item.get("id") == recommended
    ), None)


def _root_matches(root: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    origins = expected.get("causal_origins") or ()
    responsibility = (root.get("responsibility") or {}).get("kind")
    v9 = expected.get("v9") or {}
    expected_responsibility = v9.get("responsibility")
    if expected_responsibility and responsibility != expected_responsibility:
        return False
    identity = root.get("import_scc") or {}
    expected_members = set(v9.get("import_scc_members") or ())
    if expected_members:
        return set(identity.get("member_modules") or ()) == expected_members
    for origin in origins:
        kind_ok = root.get("cause_kind") == origin.get("kind")
        if (
            origin.get("kind") == "RETURN_CONTRACT"
            and responsibility == "RETURN_CONTRACT_DEFECT"
        ):
            kind_ok = True
        if not kind_ok or root.get("origin_path") != origin.get("path"):
            continue
        symbol = origin.get("symbol")
        if symbol is None or str(root.get("origin_symbol") or "").rsplit(".", 1)[-1] == str(symbol).rsplit(".", 1)[-1]:
            return True
        return_site = (root.get("responsibility") or {}).get("return_site_id")
        if return_site:
            return True
    return not origins and expected_responsibility == responsibility


def _binding_matches(
    root: Mapping[str, Any], expected: Mapping[str, Any], causal_slice: Mapping[str, Any]
) -> bool | None:
    binding = ((expected.get("v9") or {}).get("binding") or {})
    if not binding:
        return None
    profile = root.get("responsibility") or {}
    if (
        expected.get("v9", {}).get("responsibility")
        == "CONSUMER_CONTRACT_DEFECT"
        and not profile.get("binding_id")
    ):
        # An external boundary has a formal parameter but no resolved
        # actual-argument/call-site pair. It remains evaluable as a consumer
        # contract, not as an ArgumentBindingIdentity.
        return None
    if not profile.get("binding_id"):
        return False
    if "ordinal" in binding and profile.get("parameter_ordinal") != binding["ordinal"]:
        return False
    if "keyword" in binding and profile.get("keyword_binding") != binding["keyword"]:
        return False
    nodes = {
        item.get("id"): item for item in causal_slice.get("nodes") or ()
    }
    formal = nodes.get(profile.get("formal_parameter_id")) or {}
    if binding.get("parameter") and formal.get("symbol") != binding["parameter"]:
        return False
    if binding.get("callee"):
        callee = str(formal.get("scope") or "").rsplit(".", 1)[-1]
        if callee != str(binding["callee"]).rsplit(".", 1)[-1]:
            return False
    return True


def _case_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    expected = result.get("expected") or {}
    abstention_expected = bool(expected.get("insufficient_evidence"))
    actual = result.get("actual") or {}
    retrieval = result.get("retrieval") or {}
    roots = {
        item["id"]: item
        for item in (
            actual.get("root_cause_candidates")
            or retrieval.get("root_cause_candidates")
            or ()
        )
    }
    causal_slice = actual.get("causal_slice") or retrieval.get("causal_slice") or {}
    selected_id = _selected_root(actual)
    selected = roots.get(selected_id)
    generated_valid = [root for root in roots.values() if _root_matches(root, expected)]
    binding_valid = _binding_matches(selected or {}, expected, causal_slice)
    expected_v9 = expected.get("v9") or {}
    responsibility = (selected or {}).get("responsibility") or {}
    binding_root_equivalent = bool(
        selected
        and binding_valid
        and expected_v9.get("responsibility") == responsibility.get("kind")
        and any(
            item.get("kind") == "ARGUMENT_BINDING"
            for item in expected.get("causal_origins") or ()
        )
        and responsibility.get("kind")
        in {
            "ARGUMENT_SOURCE_DEFECT",
            "ARGUMENT_BINDING_DEFECT",
            "CONSUMER_CONTRACT_DEFECT",
        }
    )
    root_valid = bool(
        selected
        and (_root_matches(selected, expected) or binding_root_equivalent)
    )
    candidate = _recommended_candidate(actual)
    accepted_strategies = set(expected.get("acceptable_repair_strategies") or ())
    strategy_valid = bool(
        candidate and candidate.get("strategy_kind") in accepted_strategies
    )
    accepted_paths = set(expected.get("repair_targets") or ())
    targets = candidate.get("repair_targets") or () if candidate else ()
    target_valid = bool(
        candidate
        and targets
        and any(item.get("path") in accepted_paths for item in targets)
    )
    repair_pair = strategy_valid and target_valid
    recommended = candidate is not None
    false_abstention = not abstention_expected and not recommended
    import_case = bool((expected.get("v9") or {}).get("import_scc_members"))
    selected_scc = (selected or {}).get("import_scc") or {}
    selected_import = bool(
        selected and selected.get("cause_kind") == "IMPORT_RESOLUTION"
    )
    production_edges = selected_scc.get("production_edges") or ()
    observer_edges = selected_scc.get("observer_edges") or ()
    observer_excluded = bool(
        selected_scc
        and all(
            edge.get("provenance") == "PRODUCTION_IMPORT"
            for edge in production_edges
        )
        and all(
            edge.get("provenance") != "PRODUCTION_IMPORT"
            for edge in observer_edges
        )
    )
    pairwise_case = len(roots) > 1
    binding_case = binding_valid is not None
    legacy = result.get("metrics_v8") or {}
    if not expected_v9 and legacy:
        root_valid = bool(legacy.get("root_cause_validity"))
        strategy_valid = bool(legacy.get("repair_strategy_validity"))
        target_valid = bool(legacy.get("repair_target_validity"))
        repair_pair = bool(legacy.get("repair_pair_validity"))
        false_abstention = bool(legacy.get("false_abstention"))
    return {
        "evaluable": bool(legacy.get("evaluable", True)),
        "v9_evaluable": bool(expected_v9),
        "abstention_expected": abstention_expected,
        "generated_valid_root": bool(generated_valid),
        "root_cause_validity": root_valid if not abstention_expected else not recommended,
        "root_entity_identity_validity": root_valid if not abstention_expected else not recommended,
        "root_responsibility_validity": root_valid if not abstention_expected else not recommended,
        "argument_binding_identity_validity": binding_valid,
        "argument_slot_target_accuracy": (
            bool(binding_valid and target_valid) if binding_case else None
        ),
        "argument_vs_return_accuracy": (
            root_valid
            if (expected.get("v9") or {}).get("responsibility")
            in {"ARGUMENT_SOURCE_DEFECT", "ARGUMENT_BINDING_DEFECT", "RETURN_CONTRACT_DEFECT"}
            else None
        ),
        "import_scc_identity_validity": root_valid if import_case else None,
        "import_root_validity": root_valid if import_case else None,
        "production_scc_recall": root_valid if import_case else None,
        "production_scc_precision": root_valid if selected_import else None,
        "observer_edge_exclusion_rate": observer_excluded if import_case else None,
        "root_pairwise_accuracy": root_valid if pairwise_case else None,
        "manifestation_vs_origin_accuracy": (
            root_valid
            if any(
                (item.get("responsibility") or {}).get("kind")
                == "MANIFESTATION_ONLY"
                for item in roots.values()
            )
            else None
        ),
        "repair_strategy_validity": strategy_valid if not abstention_expected else not recommended,
        "repair_target_validity": target_valid if not abstention_expected else not recommended,
        "repair_pair_validity": repair_pair if not abstention_expected else not recommended,
        "top1_validity": repair_pair if recommended else False,
        "recommendation_coverage": recommended,
        "recommendation_valid": root_valid and repair_pair,
        "false_abstention": false_abstention,
    }


def summarize_benchmark_v9(report: Mapping[str, Any]) -> dict[str, Any]:
    results = []
    for result in report.get("results") or ():
        metrics = _case_metrics(result)
        results.append(dict(result) | {"metrics_v9": metrics})
    regression_positive = [
        item for item in results
        if (item.get("metrics_v9") or {}).get("evaluable")
        and not (item.get("metrics_v9") or {}).get("abstention_expected")
    ]
    positive = [
        item for item in regression_positive
        if (item.get("metrics_v9") or {}).get("v9_evaluable")
    ]
    recommended = [
        item for item in positive
        if (item.get("metrics_v9") or {}).get("recommendation_coverage")
    ]
    names = (
        "root_cause_validity", "root_entity_identity_validity",
        "root_responsibility_validity", "repair_strategy_validity",
        "repair_target_validity", "repair_pair_validity", "top1_validity",
        "recommendation_coverage", "false_abstention",
    )
    denominators = {
        name: _rate([
            bool((item.get("metrics_v9") or {}).get(name)) for item in positive
        ])
        for name in names
    }
    for name in (
        "import_scc_identity_validity", "argument_binding_identity_validity",
        "argument_slot_target_accuracy", "argument_vs_return_accuracy",
        "import_root_validity", "production_scc_precision",
        "production_scc_recall", "observer_edge_exclusion_rate",
        "root_pairwise_accuracy", "manifestation_vs_origin_accuracy",
    ):
        population = [
            (item.get("metrics_v9") or {}).get(name) for item in positive
            if (item.get("metrics_v9") or {}).get(name) is not None
        ]
        denominators[name] = _rate([bool(item) for item in population])
    denominators["recommendation_validity_precision"] = _rate([
        bool((item.get("metrics_v9") or {}).get("recommendation_valid"))
        for item in recommended
    ])
    quality = {name: item["value"] for name, item in denominators.items()}
    regression_denominators = {
        name: _rate([
            bool((item.get("metrics_v9") or {}).get(name))
            for item in regression_positive
        ])
        for name in (
            "root_cause_validity", "repair_strategy_validity",
            "repair_target_validity", "repair_pair_validity", "top1_validity",
        )
    }
    split = str(report.get("split") or "")
    holdout = split == "holdout_v9"
    thresholds = {
        "root_cause_validity": 0.85 if holdout else 0.92,
        "repair_pair_validity": 0.85 if holdout else 0.92,
        "top1_validity": 0.85 if holdout else 0.90,
        "recommendation_validity_precision": 0.85 if holdout else 0.90,
        "import_scc_identity_validity": 0.90 if holdout else 0.95,
        "argument_vs_return_accuracy": 0.85 if holdout else 0.90,
        "argument_binding_identity_validity": 0.90 if holdout else 0.95,
        "argument_slot_target_accuracy": 0.90 if holdout else 0.95,
        "root_pairwise_accuracy": 0.80 if holdout else 0.90,
    }
    checks = {
        name: quality.get(name) is not None and float(quality[name]) >= value
        for name, value in thresholds.items()
    }
    checks["false_abstention"] = (quality.get("false_abstention") or 0.0) <= 0.05
    safety = report.get("safety") or {}
    checks["safety"] = bool(safety.get("passed"))
    grounding = {
        "evidence_reference_validity": (
            report.get("retrieval") or {}
        ).get("evidence_ref_validity"),
        "unsupported_claim_rate": (
            report.get("reasoning") or {}
        ).get("unsupported_claim_rate"),
    }
    checks["evidence_reference_validity"] = grounding["evidence_reference_validity"] == 1.0
    checks["unsupported_claim_rate"] = (grounding["unsupported_claim_rate"] or 0.0) <= 0.05
    checks["full_live_evaluation"] = report.get("mode") == "live"
    return dict(report) | {
        "version": 9,
        "results": results,
        "quality": quality,
        "grounding": grounding,
        "metric_denominators_v9": denominators,
        "historical_regression_denominators": regression_denominators,
        "stage_gate_v9": {"checks": checks, "passed": all(checks.values())},
    }


def evaluate_predictive_cases_v9(
    cases: Sequence[PredictiveCase],
    *,
    corpus_root: Path = CORPUS_ROOT,
    mode: str = "retrieval",
    reasoner: object | None = None,
    runs: int = 1,
    analyzer_factory: Callable[..., object] | None = None,
) -> dict[str, Any]:
    report = evaluate_predictive_cases(
        cases,
        mode=mode,
        reasoner=reasoner,
        runs=runs,
        analyzer_factory=analyzer_factory,
    )
    truths = {
        item.case_id: item
        for item in load_benchmark_v5_adjudications(cases, corpus_root)
    }
    results = []
    for result in report.get("results") or ():
        truth = truths[str(result["id"])]
        metrics_v5 = adjudicate_predictive_result_v5(result, truth)
        with_v5 = dict(result) | {"metrics_v5": metrics_v5}
        results.append(with_v5 | {"metrics_v8": _metrics_v8(with_v5, truth)})
    return summarize_benchmark_v9(dict(report) | {"results": results})


def format_benchmark_v9(report: Mapping[str, Any]) -> str:
    lines = [
        f"Predictive Benchmark v9 ({report.get('mode')} / {report.get('split')})",
        f"cases: {report.get('cases')}",
    ]
    denominators = report.get("metric_denominators_v9") or {}
    for name in (
        "root_cause_validity", "repair_pair_validity", "top1_validity",
        "recommendation_validity_precision", "recommendation_coverage",
        "false_abstention", "import_scc_identity_validity",
        "argument_vs_return_accuracy", "argument_binding_identity_validity",
        "argument_slot_target_accuracy", "root_pairwise_accuracy",
        "production_scc_precision", "production_scc_recall",
        "observer_edge_exclusion_rate",
    ):
        value = denominators.get(name) or {}
        score = value.get("value")
        rendered = "n/a" if score is None else f"{float(score):.1%}"
        lines.append(
            f"{name}: {rendered} ({value.get('numerator', 0)}/{value.get('denominator', 0)})"
        )
    gate = report.get("stage_gate_v9") or {}
    lines.append(f"stage gate: {'PASS' if gate.get('passed') else 'FAIL'}")
    return "\n".join(lines)


def summarize_robustness_v9(
    metamorphic: Mapping[str, Any], counterfactual: Mapping[str, Any]
) -> dict[str, Any]:
    report = summarize_robustness_v8(metamorphic, counterfactual)
    rows = list(metamorphic.get("results") or ())
    counterfactual_rows = list(counterfactual.get("results") or ())

    def signature(row: Mapping[str, Any], side: str) -> Sequence[Any]:
        return (row.get(side) or {}).get("root_signature") or ()

    def responsibility(row: Mapping[str, Any], side: str) -> str | None:
        value = signature(row, side)
        return str(value[-2]) if len(value) >= 2 else None

    def roles(row: Mapping[str, Any], side: str) -> Sequence[Any]:
        value = signature(row, side)
        return value[1] if len(value) > 1 else ()

    scc_rows = [
        row for row in rows
        if "IMPORT_CYCLE" in roles(row, "base_signature")
    ]
    binding_rows = [
        row for row in rows
        if len(signature(row, "base_signature")) > 5
        and str(signature(row, "base_signature")[5]).startswith("ARGUMENT_SLOT:")
    ]
    argument_return_rows = [
        row for row in counterfactual_rows
        if {
            responsibility(row, "base"),
            responsibility(row, "counterfactual"),
        } <= {"ARGUMENT_SOURCE_DEFECT", "ARGUMENT_BINDING_DEFECT", "RETURN_CONTRACT_DEFECT"}
        and responsibility(row, "base") != responsibility(row, "counterfactual")
    ]
    import_switch_rows = [
        row for row in counterfactual_rows
        if "IMPORT_GRAPH_DEFECT" in {
            responsibility(row, "base"),
            responsibility(row, "counterfactual"),
        }
    ]

    denominators = dict(report.get("metric_denominators") or {}) | {
        "scc_invariance": _rate([
            bool((row.get("checks") or {}).get("root_cause_invariance"))
            for row in scc_rows
        ]),
        "root_responsibility_invariance": _rate([
            responsibility(row, "base_signature")
            == responsibility(row, "variant_signature")
            for row in rows
        ]),
        "argument_binding_invariance": _rate([
            bool((row.get("checks") or {}).get("root_cause_invariance"))
            for row in binding_rows
        ]),
        "argument_return_switch_sensitivity": _rate([
            bool((row.get("checks") or {}).get("counterfactual_root_sensitivity"))
            for row in argument_return_rows
        ]),
        "import_cycle_removal_sensitivity": _rate([
            bool((row.get("checks") or {}).get("counterfactual_root_sensitivity"))
            for row in import_switch_rows
        ]),
    }
    metrics = dict(report.get("metrics") or {}) | {
        name: value["value"] for name, value in denominators.items()
    }
    holdout = metamorphic.get("split") == "holdout_v9"
    thresholds = {
        "scc_invariance": 0.90 if holdout else 0.95,
        "root_responsibility_invariance": 0.90 if holdout else 0.95,
        "argument_binding_invariance": 0.90 if holdout else 0.95,
        "repair_invariance": 0.90 if holdout else 0.95,
        "target_signature_invariance": 0.90 if holdout else 0.95,
        "wording_invariance": 0.90 if holdout else 0.95,
        "argument_return_switch_sensitivity": 0.90,
        "import_cycle_removal_sensitivity": 0.90,
    }
    checks = {
        name: metrics.get(name) is not None and float(metrics[name]) >= threshold
        for name, threshold in thresholds.items()
    }
    checks["safety"] = bool((report.get("safety") or {}).get("passed"))
    return dict(report) | {
        "version": 9,
        "metrics": metrics,
        "metric_denominators": denominators,
        "stage_gate_v9": {
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def format_robustness_v9(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    percent = lambda value: "n/a" if value is None else f"{float(value):.1%}"
    return "\n".join((
        f"Predictive robustness v9 ({report.get('mode')}, {report.get('split')})",
        f"SCC invariance: {percent(metrics.get('scc_invariance'))}",
        f"Responsibility invariance: {percent(metrics.get('root_responsibility_invariance'))}",
        f"Argument binding invariance: {percent(metrics.get('argument_binding_invariance'))}",
        f"Repair invariance: {percent(metrics.get('repair_invariance'))}",
        f"Target invariance: {percent(metrics.get('target_signature_invariance'))}",
        f"Wording invariance: {percent(metrics.get('wording_invariance'))}",
        f"Stage gate: {'PASS' if (report.get('stage_gate_v9') or {}).get('passed') else 'FAIL'}",
    ))
