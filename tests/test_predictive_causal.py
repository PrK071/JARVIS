from __future__ import annotations

import json

from tern.orchestrator.predictive.analysis import PredictiveAnalyzer
from tern.orchestrator.predictive.causal import (
    CausalEdgeKind,
    RepairStrategyKind,
    RootCauseKind,
    RootSelectionReason,
    compatible_strategies_for_root,
    repair_locality,
    strategy_compatible,
    structurally_dominant_root,
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

    assert any(
        root.origin_path == "pkg/models.py"
        and root.cause_kind is RootCauseKind.NULL_FLOW
        for root in context.root_cause_candidates
    )
    assert CausalEdgeKind.READ_FROM_ATTRIBUTE in {
        edge.kind for edge in context.causal_slice.edges
    }


def test_attribute_assignment_propagates_to_external_receiver():
    context = _context("PD-002")

    assert any(
        root.origin_path == "pkg/orders.py"
        and root.cause_kind is RootCauseKind.NULL_FLOW
        for root in context.root_cause_candidates
    )
    root = next(
        item for item in context.root_cause_candidates
        if item.origin_path == "pkg/orders.py" and item.cause_kind is RootCauseKind.NULL_FLOW
    )
    assert RepairStrategyKind.CORRECT_RETURN_VALUE not in compatible_strategies_for_root(
        root, context.causal_slice
    )


def test_null_return_keeps_return_value_repair_compatible():
    context = _context("PC3D-001")
    root = context.root_cause_candidates[0]

    assert RepairStrategyKind.CORRECT_RETURN_VALUE in compatible_strategies_for_root(
        root, context.causal_slice
    )


def test_explicit_return_contract_can_dominate_failure_site():
    context = _context("PD-019")
    dominant = structurally_dominant_root(
        context.root_cause_candidates, context.problem
    )

    assert dominant is not None
    assert dominant.cause_kind is RootCauseKind.RETURN_CONTRACT
    assert dominant.origin_symbol == "find_user"


def test_upstream_contract_dominates_consumer_manifestation():
    for identifier, symbol in (
        ("PC5D-008", "parse_count"),
        ("PC5D-013", "decode_quantity"),
        ("PC5D-014", "decode_price"),
    ):
        context = _context(identifier)
        dominant = structurally_dominant_root(
            context.root_cause_candidates, context.problem, context.causal_slice
        )

        assert dominant is not None
        assert dominant.cause_kind is RootCauseKind.RETURN_CONTRACT
        assert dominant.origin_symbol == symbol


def test_explicit_parameter_owner_dominates_unrelated_flow():
    for identifier in ("PC5D-005", "PC5D-006"):
        context = _context(identifier)
        dominant = structurally_dominant_root(
            context.root_cause_candidates, context.problem, context.causal_slice
        )

        assert dominant is not None
        assert dominant.cause_kind is RootCauseKind.ARGUMENT_BINDING
        assert dominant.origin_symbol == "values"


def test_import_cycle_candidates_are_distinct_and_legacy_wording_is_supported():
    three_node = _context("PC5D-001")
    legacy = _context("PD-016")
    semantic_origins = {
        (root.cause_kind, root.origin_path, root.origin_symbol, root.origin_line)
        for root in three_node.root_cause_candidates
    }

    assert len(semantic_origins) == len(three_node.root_cause_candidates)
    assert len(three_node.root_cause_candidates) == 3
    assert legacy.root_cause_candidates
    assert all(
        root.cause_kind is RootCauseKind.IMPORT_RESOLUTION
        for root in legacy.root_cause_candidates
    )


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
        assert set(payload) == {
            "problem",
            "project_id",
            "root_cause_candidates",
            "structurally_dominant_root_id",
        }
        assert "evidence_ledger" not in payload
        assert "ignore all previous instructions" not in serialized
        assert "system message" not in serialized
        assert "delete database" not in serialized
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] <= 650


def test_reasoner_schema_requires_closed_comparative_reason_codes():
    context = _context("PD-019")

    class Recorder:
        def chat(self, _messages, **kwargs):
            schema = kwargs["response_format"]["json_schema"]["schema"]
            item = schema["properties"]["selections"]["items"]
            assert set(item["properties"]["selection_reason"]["enum"]) == {
                item.value for item in RootSelectionReason
            }
            assert "rejected" in item["required"]
            assert item["properties"]["root_cause_id"]["enum"] == [
                context.reasoning_payload()["structurally_dominant_root_id"]
            ]
            return {"choices": [{"message": {"content": '{"selections": []}'}}]}

    PredictiveAnalyzer(Recorder()).analyze_with_diagnostics(context)


def test_argument_repair_uses_call_site_even_when_model_names_parameter():
    context = _context("PC5D-005")
    root_id = context.reasoning_payload()["structurally_dominant_root_id"]

    class Reasoner:
        def chat(self, _messages, **_kwargs):
            value = {
                "selections": [{
                    "root_cause_id": root_id,
                    "selection_reason": "ARGUMENT_SOURCE",
                    "rejected": [],
                    "claim": "mean passes a zero count to ratio",
                    "strategies": [{
                        "kind": "CORRECT_ARGUMENT",
                        "target_file": "pkg/mathops.py",
                        "target_symbol": "values",
                        "rationale": "Correct the call argument derived from the empty input",
                        "reason": "ARGUMENT_SOURCE",
                    }],
                }],
            }
            return {"choices": [{"message": {"content": json.dumps(value)}}]}

    result = PredictiveAnalyzer(Reasoner()).analyze_with_diagnostics(context)

    assert result.candidates
    assert result.candidates[0].repair_targets[0].scope_kind.value == "CALL_SITE"


def test_explicit_boundary_limits_dominant_binding_to_validation_strategy():
    context = _context("PC5D-006")

    class Recorder:
        def chat(self, _messages, **kwargs):
            schema = kwargs["response_format"]["json_schema"]["schema"]
            strategy = schema["properties"]["selections"]["items"][
                "properties"
            ]["strategies"]["items"]["properties"]["kind"]
            assert strategy["enum"] == ["VALIDATE_BOUNDARY"]
            return {"choices": [{"message": {"content": '{"selections": []}'}}]}

    PredictiveAnalyzer(Recorder()).analyze_with_diagnostics(context)


def test_holdout_v3_integrity_is_frozen_in_manifest():
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert len(load_predictive_cases(split="holdout_v3")) == 12
    assert predictive_corpus_hash(split="holdout_v3") == manifest["holdout_v3_sha256"]
