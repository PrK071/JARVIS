from __future__ import annotations

from tern.orchestrator.predictive.causal import (
    CausalResponsibilityKind,
    RepairStrategyKind,
    RootCauseKind,
)
from tern.orchestrator.predictive.benchmark_v9 import summarize_benchmark_v9
from tern.orchestrator.predictive.evaluation import _retrieval, load_predictive_cases
from tern.orchestrator.predictive.repair import (
    RepairTargetKind,
    derive_repair_targets,
    validate_repair_target,
)
from tern.orchestrator.predictive.semantic import (
    CausalRole,
    root_cause_signature,
)


def _case(identifier: str):
    return next(case for case in load_predictive_cases() if case.id == identifier)


def _context(identifier: str):
    return _retrieval(_case(identifier))[0]


def test_two_and_three_module_cycles_have_one_scc_root():
    two = _context("CI9D-001")
    three = _context("CI9D-003")

    assert len(two.root_cause_candidates) == 1
    assert len(three.root_cause_candidates) == 1
    assert len(two.root_cause_candidates[0].import_scc.member_modules) == 2
    assert len(three.root_cause_candidates[0].import_scc.member_modules) == 3
    assert all(
        root.responsibility.kind is CausalResponsibilityKind.IMPORT_GRAPH_DEFECT
        for root in (*two.root_cause_candidates, *three.root_cause_candidates)
    )


def test_import_scc_excludes_observers_and_has_module_identity():
    context = _context("CI9D-004")
    root = context.root_cause_candidates[0]
    identity = root.import_scc
    signature = root_cause_signature(root, context.causal_slice)

    assert identity is not None
    assert all(not edge.source.startswith("tests/") for edge in identity.production_edges)
    assert any(edge.source.startswith("tests/") for edge in identity.observer_edges)
    assert CausalRole.IMPORT_CYCLE in signature.causal_roles
    assert signature.entity_identity == "IMPORT_SCC:2:2"


def test_type_checking_and_local_imports_do_not_create_runtime_scc():
    assert not any(root.import_scc for root in _context("CI9D-005").root_cause_candidates)
    assert not any(root.import_scc for root in _context("CI9D-006").root_cause_candidates)


def test_import_observer_wording_selects_the_anchored_production_scc():
    context = _context("CI9D-018")
    identity = context.root_cause_candidates[0].import_scc

    assert identity is not None
    assert identity.member_modules == (
        "cycle3/blue.py", "cycle3/green.py", "cycle3/red.py"
    )


def test_argument_source_and_annotated_return_contract_are_distinct():
    argument = _context("CI9D-007")
    returned = _context("CI9D-009")

    assert any(
        root.responsibility.kind is CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT
        and root.responsibility.binding_id
        for root in argument.root_cause_candidates
    )
    assert any(
        root.cause_kind is RootCauseKind.RETURN_CONTRACT
        and root.contract_demonstrated
        and root.responsibility.kind
        is CausalResponsibilityKind.RETURN_CONTRACT_DEFECT
        for root in returned.root_cause_candidates
    )


def test_sibling_keyword_bindings_have_distinct_identities():
    context = _context("CI9D-012")
    roots = [
        root for root in context.root_cause_candidates
        if root.origin_symbol in {"total", "slots"}
        and root.responsibility.binding_id
        and root.origin_line == 9
    ]

    assert {root.responsibility.parameter_ordinal for root in roots} == {0, 1}
    assert {root.responsibility.keyword_binding for root in roots} >= {None, "slots"}
    assert len({root.responsibility.binding_id for root in roots}) == len(roots)


def test_default_argument_is_a_first_class_binding():
    context = _context("CI9D-014")
    root = next(
        item for item in context.root_cause_candidates
        if item.origin_line == 38
        and item.responsibility.kind
        is CausalResponsibilityKind.ARGUMENT_BINDING_DEFECT
    )

    assert root.responsibility.binding_id
    assert root.responsibility.parameter_ordinal == 0
    assert root.responsibility.defect_bearing_relation == (
        "default_argument_to_formal_parameter"
    )


def test_correct_argument_targets_exact_binding_not_sibling():
    context = _context("CI9D-012")
    root = next(
        item for item in context.root_cause_candidates
        if item.origin_symbol == "slots"
        and item.responsibility.keyword_binding == "slots"
    )
    targets = derive_repair_targets(
        RepairStrategyKind.CORRECT_ARGUMENT, root, context.causal_slice
    )

    assert targets
    assert any(
        target.scope_kind is RepairTargetKind.CALL_SITE
        and validate_repair_target(
            RepairStrategyKind.CORRECT_ARGUMENT,
            root,
            target,
            context.causal_slice,
        )
        for target in targets
    )
    assert not any(
        target.scope_kind is RepairTargetKind.PARAMETER
        and target.parameter == "total"
        for target in targets
    )


def test_scc_and_binding_ids_are_reproducible():
    first_scc = _context("CI9D-001").root_cause_candidates[0].import_scc.id
    second_scc = _context("CI9D-001").root_cause_candidates[0].import_scc.id
    first_bindings = {
        root.responsibility.binding_id
        for root in _context("CI9D-012").root_cause_candidates
        if root.responsibility.binding_id
    }
    second_bindings = {
        root.responsibility.binding_id
        for root in _context("CI9D-012").root_cause_candidates
        if root.responsibility.binding_id
    }

    assert first_scc == second_scc
    assert first_bindings == second_bindings


def test_v9_metrics_report_explicit_denominators():
    context = _context("CI9D-001")
    root = context.root_cause_candidates[0]
    report = summarize_benchmark_v9({
        "version": 3,
        "mode": "live",
        "split": "development",
        "cases": 1,
        "results": [{
            "id": "CI9D-001",
            "expected": _case("CI9D-001").expected,
            "actual": {
                "hypotheses": [{"root_cause_id": root.id}],
                "root_cause_candidates": [root.as_dict()],
                "causal_slice": context.causal_slice.as_dict(),
                "candidates": [{
                    "id": "C1",
                    "strategy_kind": "CORRECT_IMPORT",
                    "repair_targets": [{"path": "cycle2/alpha.py"}],
                }],
                "recommended_candidate_id": "C1",
            },
            "retrieval": {},
        }],
        "safety": {"passed": True},
        "retrieval": {"evidence_ref_validity": 1.0},
        "reasoning": {"unsupported_claim_rate": 0.0},
    })

    metric = report["metric_denominators_v9"]["import_scc_identity_validity"]
    assert metric == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert report["quality"]["repair_pair_validity"] == 1.0
    assert "root_pairwise_accuracy" in report["metric_denominators_v9"]
    assert "argument_slot_target_accuracy" in report["metric_denominators_v9"]
    assert "production_scc_precision" in report["metric_denominators_v9"]
