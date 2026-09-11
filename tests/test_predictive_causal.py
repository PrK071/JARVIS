from __future__ import annotations

import json

from tern.orchestrator.predictive.analysis import PredictiveAnalyzer
from tern.orchestrator.predictive.causal import (
    CausalEdgeKind,
    RepairStrategyKind,
    RootCauseKind,
    repair_locality,
    strategy_compatible,
)
from tern.orchestrator.predictive.evaluation import (
    CORPUS_ROOT,
    _retrieval,
    load_predictive_cases,
    predictive_corpus_hash,
)


def _case(identifier: str):
    return next(case for case in load_predictive_cases() if case.id == identifier)


def _context(identifier: str):
    return _retrieval(_case(identifier))[0]


def test_causal_ids_and_ordering_are_stable():
    first = _context("PC3D-001")
    second = _context("PC3D-001")

    assert first.causal_slice.as_dict() == second.causal_slice.as_dict()
    assert first.root_cause_candidates == second.root_cause_candidates
    assert first.root_cause_candidates[0].cause_kind is RootCauseKind.NULL_FLOW
    assert first.root_cause_candidates[0].origin_path == "pkg/repository.py"


def test_backward_slice_links_assignment_parameter_and_return_propagation():
    context = _context("PC3D-003")
    kinds = {edge.kind for edge in context.causal_slice.edges}

    assert CausalEdgeKind.ASSIGNED_FROM in kinds
    assert CausalEdgeKind.PASSED_AS_ARGUMENT in kinds
    assert CausalEdgeKind.RETURNED_FROM in kinds
    assert any(
        root.origin_path == "pkg/repository.py"
        and root.origin_symbol == "load_tax"
        and root.cause_kind is RootCauseKind.RETURN_CONTRACT
        for root in context.root_cause_candidates
    )


def test_parameter_default_is_a_distinct_binding_origin():
    context = _context("PC3D-004")

    assert any(
        root.cause_kind is RootCauseKind.ARGUMENT_BINDING
        and root.origin_symbol == "tax"
        and root.statement.startswith("default tax=0")
        for root in context.root_cause_candidates
    )


def test_nested_calls_and_branch_guards_remain_explicit():
    nested = _context("PC3D-008")
    guarded = _context("PC3D-006")

    assert sum(edge.kind is CausalEdgeKind.PASSED_AS_ARGUMENT for edge in nested.causal_slice.edges) >= 2
    assert CausalEdgeKind.GUARDED_BY in {edge.kind for edge in guarded.causal_slice.edges}
    assert any(root.cause_kind is RootCauseKind.CONTROL_FLOW for root in guarded.root_cause_candidates)


def test_configuration_origin_is_resolved_without_executing_code():
    context = _context("PC3D-009")

    assert CausalEdgeKind.CONFIGURED_BY in {edge.kind for edge in context.causal_slice.edges}
    assert context.root_cause_candidates[0].cause_kind is RootCauseKind.CONFIGURATION
    assert context.root_cause_candidates[0].origin_path == "pkg/settings.py"


def test_import_frame_and_guarded_raise_create_typed_roots():
    imported = _context("PD-013")
    guarded_raise = _context("PD-010")

    assert imported.root_cause_candidates[0].cause_kind is RootCauseKind.IMPORT_RESOLUTION
    assert any(
        root.cause_kind is RootCauseKind.CONTROL_FLOW
        for root in guarded_raise.root_cause_candidates
    )


def test_method_receiver_flows_back_to_the_function_parameter():
    context = _context("PD-005")

    assert context.root_cause_candidates[0].cause_kind is RootCauseKind.ARGUMENT_BINDING
    assert context.root_cause_candidates[0].origin_symbol == "user"


def test_failure_site_and_root_origin_are_distinct():
    context = _context("PC3D-001")
    root = context.root_cause_candidates[0]

    assert root.failure_path == "pkg/totals.py"
    assert root.origin_path == "pkg/repository.py"
    assert root.causal_path[-1] == context.causal_slice.failure_site_id
    edge_pairs = {(edge.source_id, edge.target_id) for edge in context.causal_slice.edges}
    assert all(pair in edge_pairs for pair in zip(root.causal_path, root.causal_path[1:]))


def test_unknown_flow_abstains_structurally():
    context = _context("PC3D-012")

    assert context.causal_slice.unknown_flow is True
    assert context.root_cause_candidates == ()


def test_repair_compatibility_and_locality_are_causal():
    context = _context("PC3D-001")
    root = context.root_cause_candidates[0]

    assert strategy_compatible(root.cause_kind, RepairStrategyKind.FIX_PRODUCER)
    assert not strategy_compatible(root.cause_kind, RepairStrategyKind.CORRECT_IMPORT)
    assert repair_locality(root, (root.origin_path,), (), context.causal_slice) == 1.0
    assert repair_locality(root, (root.failure_path,), (), context.causal_slice) == 0.7


def test_reasoner_protocol_is_compact_single_call_and_contains_no_raw_source_injection():
    contexts = (_context("PD-021"), _context("PV2-008"))

    class Recorder:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return {"choices": [{"message": {"content": '{"selections": []}'}}]}

    reasoner = Recorder()
    for context in contexts:
        PredictiveAnalyzer(reasoner).analyze_with_diagnostics(context)

    assert len(reasoner.calls) == 2
    for messages, kwargs in reasoner.calls:
        payload = json.loads(messages[-1]["content"])
        serialized = json.dumps(payload).casefold()
        assert set(payload) == {"problem", "project_id", "root_cause_candidates"}
        assert "evidence_ledger" not in payload
        assert "ignore all previous instructions" not in serialized
        assert "system message" not in serialized
        assert "delete database" not in serialized
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] <= 650


def test_holdout_v3_integrity_is_frozen_in_manifest():
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert len(load_predictive_cases(split="holdout_v3")) == 12
    assert predictive_corpus_hash(split="holdout_v3") == manifest["holdout_v3_sha256"]
