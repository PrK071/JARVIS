from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from tern.orchestrator.predictive.benchmark_v8 import (
    _semantic_root_metrics,
    summarize_benchmark_v8,
)
from tern.orchestrator.predictive.causal import (
    RepairStrategyKind,
    RootCauseKind,
    build_causal_slice,
    structurally_dominant_root,
)
from tern.orchestrator.predictive.evaluation import (
    _retrieval,
    load_predictive_cases,
    predictive_corpus_hash,
)
from tern.orchestrator.predictive.repair import (
    RepairTarget,
    RepairTargetKind,
    derive_repair_targets,
    targets_compatible,
)
from tern.orchestrator.predictive.benchmark_v5 import load_benchmark_v5_adjudications
from tern.orchestrator.predictive.robustness import (
    load_counterfactual_cases,
    load_metamorphic_cases,
    materialize_metamorphic_case,
)
from tern.orchestrator.predictive.semantic import (
    CausalRole,
    PairwiseRootReason,
    TargetPreference,
    compare_root_candidates,
    is_transparent_wrapper,
    repair_target_signature,
    root_cause_signature,
    semantic_dominant_root,
    semantic_repair_strategy_score,
    semantic_repair_pair_score,
    semantic_target_equivalent,
    target_preference,
    validate_root_comparison,
)


def _case(identifier: str):
    return next(item for item in load_predictive_cases() if item.id == identifier)


def _context(identifier: str):
    return _retrieval(_case(identifier))[0]


def _root(context, *, kind=None, path=None, symbol=None):
    return next(
        item for item in context.root_cause_candidates
        if (kind is None or item.cause_kind is kind)
        and (path is None or item.origin_path == path)
        and (symbol is None or item.origin_symbol == symbol)
    )


def test_root_signature_uses_causal_role_not_names():
    dev = _context("SV8D-001")
    holdout = _context("SV8H-001")
    left = _root(dev, kind=RootCauseKind.NULL_FLOW, path="commerce/source.py")
    right = _root(holdout, kind=RootCauseKind.NULL_FLOW, path="inventory/provider.py")

    assert root_cause_signature(left, dev.causal_slice) == root_cause_signature(
        right, holdout.causal_slice
    )
    assert CausalRole.PRODUCER in root_cause_signature(left, dev.causal_slice).causal_roles


def test_pairwise_root_selection_rejects_transparent_wrapper():
    context = _context("PR7D-004")
    producer = _root(context, kind=RootCauseKind.NULL_FLOW, path="pkg/source.py")
    wrapper = _root(context, kind=RootCauseKind.RETURN_CONTRACT, path="pkg/bridge.py")

    assert is_transparent_wrapper(wrapper, context.causal_slice)
    comparison = compare_root_candidates(producer, wrapper, context.causal_slice)

    assert comparison.preferred_id == producer.id
    assert comparison.reason is PairwiseRootReason.INTERMEDIATE_WRAPPER
    assert validate_root_comparison(
        producer, wrapper, PairwiseRootReason.INTERMEDIATE_WRAPPER,
        context.causal_slice,
    )
    assert not validate_root_comparison(
        wrapper, producer, PairwiseRootReason.DIRECT_CAUSAL_ORIGIN,
        context.causal_slice,
    )
    assert semantic_dominant_root(
        context.root_cause_candidates, context.causal_slice
    ) == producer


def test_nontransparent_wrapper_is_not_collapsed():
    context = _context("SV8D-005")
    wrapper = _root(
        context, kind=RootCauseKind.RETURN_CONTRACT,
        path="commerce/bridge.py", symbol="adjusted_amount",
    )

    assert not is_transparent_wrapper(wrapper, context.causal_slice)


def test_call_with_locally_transformed_input_is_not_transparent_wrapper():
    context = _context("PD-012")
    transformed = _root(
        context, kind=RootCauseKind.RETURN_CONTRACT,
        path="pkg/loader.py", symbol="load",
    )

    assert not is_transparent_wrapper(transformed, context.causal_slice)


def test_literal_default_and_parameter_share_one_semantic_origin():
    context = _context("PC5D-007")
    dominant = semantic_dominant_root(
        context.root_cause_candidates, context.causal_slice
    )

    assert dominant is not None
    assert dominant.cause_kind is RootCauseKind.NULL_FLOW
    assert dominant.origin_line == 19


def test_demonstrated_return_contract_dominates_input_manifestation():
    for case_id, expected_symbol in (
        ("PC3D-003", "load_tax"),
        ("PD-019", "find_user"),
        ("PR7D-006", "decode"),
    ):
        context = _context(case_id)
        dominant = structurally_dominant_root(
            context.root_cause_candidates, context.problem, context.causal_slice
        )

        assert dominant is not None
        assert dominant.cause_kind is RootCauseKind.RETURN_CONTRACT
        assert dominant.origin_symbol == expected_symbol


def test_weak_identity_return_does_not_override_argument_source():
    context = _context("PC5D-009")
    dominant = structurally_dominant_root(
        context.root_cause_candidates, context.problem, context.causal_slice
    )

    assert dominant is not None
    assert dominant.cause_kind is RootCauseKind.ARGUMENT_BINDING


def test_repair_strategy_follows_structural_producer_kind():
    null_context = _context("SV8D-001")
    null_root = _root(
        null_context, kind=RootCauseKind.NULL_FLOW, path="commerce/source.py"
    )
    type_context = _context("SV8D-007")
    type_root = _root(
        type_context, kind=RootCauseKind.TYPE_FLOW, path="commerce/source.py"
    )

    assert semantic_repair_strategy_score(
        RepairStrategyKind.FIX_PRODUCER,
        null_root,
        null_context.causal_slice,
        0.5,
    ) > semantic_repair_strategy_score(
        RepairStrategyKind.CORRECT_RETURN_VALUE,
        null_root,
        null_context.causal_slice,
        0.5,
    )
    assert semantic_repair_strategy_score(
        RepairStrategyKind.CORRECT_RETURN_VALUE,
        type_root,
        type_context.causal_slice,
        0.5,
    ) > semantic_repair_strategy_score(
        RepairStrategyKind.CORRECT_ARGUMENT,
        type_root,
        type_context.causal_slice,
        0.5,
    )


def test_fix_producer_requires_a_producer_scope_to_outrank_return_fix():
    context = _context("PC5D-010")
    root = _root(context, kind=RootCauseKind.NULL_FLOW, path="app/storage.py")
    targets = derive_repair_targets(
        RepairStrategyKind.FIX_PRODUCER, root, context.causal_slice
    )
    return_site = next(
        item for item in targets if item.scope_kind is RepairTargetKind.RETURN_SITE
    )
    function = next(
        item for item in targets if item.scope_kind is RepairTargetKind.FUNCTION
    )

    assert semantic_repair_pair_score(
        RepairStrategyKind.FIX_PRODUCER,
        root,
        function,
        context.causal_slice,
        0.5,
    ) > semantic_repair_pair_score(
        RepairStrategyKind.FIX_PRODUCER,
        root,
        return_site,
        context.causal_slice,
        0.5,
    )
def test_compound_return_keeps_upstream_producer_in_causal_candidates():
    context = _context("SV8D-007")

    assert any(
        item.origin_path == "commerce/source.py"
        and item.cause_kind in {RootCauseKind.TYPE_FLOW, RootCauseKind.RETURN_CONTRACT}
        for item in context.root_cause_candidates
    )


def test_argument_source_dominates_downstream_parameter_manifestation():
    context = _context("SV8D-011")
    dominant = structurally_dominant_root(
        context.root_cause_candidates, context.problem, context.causal_slice
    )

    assert dominant is not None
    assert dominant.cause_kind is RootCauseKind.ARGUMENT_BINDING
    assert dominant.origin_symbol == "count"
    assert dominant.origin_line == 9


def test_boundary_target_preference_ignores_test_only_argument_source():
    context = _context("PC5D-005")
    root = _root(context, kind=RootCauseKind.ARGUMENT_BINDING, symbol="values")
    boundary = derive_repair_targets(
        RepairStrategyKind.VALIDATE_BOUNDARY, root, context.causal_slice
    )
    argument = derive_repair_targets(
        RepairStrategyKind.CORRECT_ARGUMENT, root, context.causal_slice
    )
    parameter = next(item for item in boundary if item.parameter == "values")
    call = next(item for item in argument if item.scope_kind.value == "CALL_SITE")

    assert target_preference(
        RepairStrategyKind.VALIDATE_BOUNDARY, root, parameter, context.causal_slice
    ) is TargetPreference.PREFERRED
    assert target_preference(
        RepairStrategyKind.CORRECT_ARGUMENT, root, call, context.causal_slice
    ) is TargetPreference.COMPATIBLE


def test_target_signature_normalizes_boundary_scope_granularity():
    context = _context("PC5D-005")
    root = _root(context, kind=RootCauseKind.ARGUMENT_BINDING, symbol="values")
    targets = derive_repair_targets(
        RepairStrategyKind.VALIDATE_BOUNDARY, root, context.causal_slice
    )
    parameter = next(item for item in targets if item.parameter == "values")
    function = next(item for item in targets if item.scope_kind.value == "FUNCTION")
    left = repair_target_signature(
        parameter, RepairStrategyKind.VALIDATE_BOUNDARY, root, context.causal_slice
    )
    right = repair_target_signature(
        function, RepairStrategyKind.VALIDATE_BOUNDARY, root, context.causal_slice
    )

    assert semantic_target_equivalent(left, right)


def test_callable_target_compatibility_normalizes_method_qualification():
    expected = RepairTarget(
        "pkg/models.py", RepairTargetKind.FUNCTION, "__init__"
    )
    actual = RepairTarget(
        "pkg/models.py", RepairTargetKind.METHOD, "Profile.__init__"
    )

    assert targets_compatible(actual, expected)


def test_semantic_signature_is_unchanged_by_problem_wording():
    context = _context("SV8D-001")
    root = _root(context, kind=RootCauseKind.NULL_FLOW, path="commerce/source.py")
    changed = replace(context, problem="Validation producer return boundary import None")

    assert root_cause_signature(root, context.causal_slice) == root_cause_signature(
        root, changed.causal_slice
    )


def test_v8_report_exposes_pairwise_denominators_and_features():
    report = summarize_benchmark_v8({
        "version": 5,
        "mode": "retrieval",
        "split": "development",
        "cases": 0,
        "evaluable_cases": 0,
        "results": [],
        "quality": {},
        "grounding": {"evidence_reference_validity": 1.0, "unsupported_claim_rate": 0.0},
        "safety": {"passed": True, "forbidden_candidate_recommendations": 0},
    })

    assert report["version"] == 8
    assert report["metric_denominators_v8"]["root_pairwise_accuracy"]["denominator"] == 0
    assert report["seed_synthesis_v8"] == {
        "attempts": 0,
        "successes": 0,
        "correct_diagnoses": 0,
        "success_rate": None,
        "precision": None,
    }
    assert {item["source"] for item in report["ranking_features"]} <= {
        "STRUCTURAL", "OBSERVED", "DERIVED", "MODEL", "LEXICAL"
    }


def test_v8_root_validity_uses_semantic_identity_for_target_granularity():
    context = _context("SV8D-012")
    roots = {item.id: item for item in context.root_cause_candidates}
    unsymbolized = next(
        item for item in roots.values()
        if item.cause_kind is RootCauseKind.CONFIGURATION
        and item.origin_symbol is None
    )
    truth = next(
        item for item in load_benchmark_v5_adjudications((_case("SV8D-012"),))
    )
    actual = {
        "root_cause_candidates": [item.as_dict() for item in roots.values()],
        "root_cause_signatures": {
            item.id: root_cause_signature(item, context.causal_slice).as_dict()
            for item in roots.values()
        },
        "hypotheses": [{"root_cause_id": unsymbolized.id}],
    }

    assert _semantic_root_metrics(actual, truth) == (True, True)


def test_v8_root_validity_treats_return_source_flow_as_same_causal_origin():
    context = _context("SV8D-005")
    flow = _root(
        context, kind=RootCauseKind.TYPE_FLOW, path="commerce/source.py"
    )
    truth = next(
        item for item in load_benchmark_v5_adjudications((_case("SV8D-005"),))
    )
    actual = {
        "root_cause_candidates": [
            item.as_dict() for item in context.root_cause_candidates
        ],
        "root_cause_signatures": {
            item.id: root_cause_signature(item, context.causal_slice).as_dict()
            for item in context.root_cause_candidates
        },
        "hypotheses": [{"root_cause_id": flow.id}],
    }

    assert _semantic_root_metrics(actual, truth) == (True, True)


def test_v8_import_root_accepts_qualified_and_local_edge_symbols():
    context = _context("SV8D-014")
    root = next(
        item for item in context.root_cause_candidates
        if item.cause_kind is RootCauseKind.IMPORT_RESOLUTION
    )
    truth = next(
        item for item in load_benchmark_v5_adjudications((_case("SV8D-014"),))
    )
    actual = {
        "root_cause_candidates": [item.as_dict() for item in context.root_cause_candidates],
        "root_cause_signatures": {
            item.id: root_cause_signature(item, context.causal_slice).as_dict()
            for item in context.root_cause_candidates
        },
        "hypotheses": [{"root_cause_id": root.id}],
    }

    assert _semantic_root_metrics(actual, truth) == (True, True)


def test_holdout_v8_is_sealed_before_live_evaluation():
    manifest = json.loads(
        (Path(__file__).parent / "data" / "predictive" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(load_predictive_cases(split="holdout_v8")) == 20
    assert predictive_corpus_hash(split="holdout_v8") == manifest["holdout_v8_sha256"]


def test_v8_metamorphic_and_counterfactual_suites_are_versioned(tmp_path):
    root = Path(__file__).parent / "data" / "predictive" / "v8"
    specs = load_metamorphic_cases(root, split="development")
    pairs = load_counterfactual_cases(root, split="development")
    cases = {item.id: item for item in load_predictive_cases()}
    truths = {
        item.case_id: item
        for item in load_benchmark_v5_adjudications(
            tuple(cases[case_id] for case_id in sorted({item.base_case_id for item in specs}))
        )
    }

    assert len(specs) >= 18
    assert len(pairs) >= 4
    for spec in specs:
        case, truth, _mapping = materialize_metamorphic_case(
            spec, cases[spec.base_case_id], truths[spec.base_case_id], tmp_path
        )
        assert case.fixture_root.is_dir()
        assert truth.acceptable_root_causes


def test_comment_and_docstring_distractors_do_not_change_root_signature(tmp_path):
    root = Path(__file__).parent / "data" / "predictive" / "v8"
    specs = load_metamorphic_cases(root, split="development")
    cases = {item.id: item for item in load_predictive_cases()}
    chosen = [
        next(item for item in specs if item.transformation.value == kind)
        for kind in ("HARMLESS_COMMENT", "DOCSTRING_DISTRACTOR")
    ]
    truths = {
        item.case_id: item
        for item in load_benchmark_v5_adjudications(
            tuple(cases[item.base_case_id] for item in chosen)
        )
    }

    for index, spec in enumerate(chosen):
        base_context = _context(spec.base_case_id)
        variant, _truth, _mapping = materialize_metamorphic_case(
            spec,
            cases[spec.base_case_id],
            truths[spec.base_case_id],
            tmp_path / f"variant-{index}",
        )
        variant_context = _retrieval(variant)[0]
        base_root = max(base_context.root_cause_candidates, key=lambda item: item.score)
        variant_root = max(
            variant_context.root_cause_candidates, key=lambda item: item.score
        )
        assert root_cause_signature(
            base_root, base_context.causal_slice
        ) == root_cause_signature(variant_root, variant_context.causal_slice)
