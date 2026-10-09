from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tern.orchestrator import cli
from tern.orchestrator.predictive.testing.evaluation import (
    evaluate_generated_counterfactuals,
    evaluate_generated_structural,
    evaluate_oracle_injection,
    reduce_failing_spec,
    structural_case_metrics,
    live_case_metrics,
)
from tern.orchestrator.predictive.causal import (
    CausalResponsibilityKind,
    RootCauseKind,
    structurally_dominant_root,
)
from tern.orchestrator.predictive.evaluation import _retrieval
from tern.orchestrator.predictive.testing.generator import (
    CausalOracle,
    ProgramFamily,
    ProgramPopulation,
    ProgramSpec,
    generate_counterfactual_pairs,
    generate_holdout_v10_supplement,
    generate_program,
    generate_suite,
    stratified_sample,
    suite_sha256,
)


def test_program_spec_is_deterministic_and_validated():
    left = ProgramSpec(ProgramFamily.ARGUMENT_BINDING, 42, defective_slot=1)
    right = ProgramSpec(ProgramFamily.ARGUMENT_BINDING, 42, defective_slot=1)

    assert left.id == right.id
    assert left.canonical_json() == right.canonical_json()
    with pytest.raises(ValueError, match="defective slot"):
        ProgramSpec(ProgramFamily.ARGUMENT_BINDING, 42, defective_slot=3)


def test_generated_program_is_seed_reproducible(tmp_path):
    spec = ProgramSpec(
        ProgramFamily.ARGUMENT_BINDING,
        42,
        argument_style="keyword",
        defective_slot=1,
        wrapper_depth=2,
        temporary_variable=True,
        distractor=True,
    )
    first = generate_program(spec, tmp_path / "first")
    second = generate_program(spec, tmp_path / "second")

    assert first.files == second.files
    assert first.oracle == second.oracle
    assert first.case.problem == second.case.problem


@pytest.mark.parametrize("style", ["positional", "keyword", "mixed", "default"])
def test_argument_family_oracle_owns_ground_truth(style, tmp_path):
    spec = ProgramSpec(
        ProgramFamily.ARGUMENT_BINDING, 90, argument_style=style,
        slot_count=3, defective_slot=2,
    )
    program = generate_program(spec, tmp_path / style)

    assert "CORRECT_ARGUMENT" in program.oracle.expected_repair_strategies
    assert program.oracle.binding_slot == 2
    assert program.oracle.responsibility_kinds in {
        ("ARGUMENT_SOURCE_DEFECT",), ("ARGUMENT_BINDING_DEFECT",),
    }
    assert program.case.expected["v10_oracle"] == program.oracle.as_dict()


def test_return_family_oracle_is_not_derived_from_text(tmp_path):
    spec = ProgramSpec(
        ProgramFamily.RETURN_CONTRACT, 91, return_valid=False,
        contract_evidence="type_annotation", multiple_return_sites=True,
    )
    program = generate_program(spec, tmp_path / "return")

    assert program.oracle.responsibility_kinds == ("RETURN_CONTRACT_DEFECT",)
    assert program.oracle.preferred_repair_strategy == "CORRECT_RETURN_VALUE"
    assert program.oracle.preferred_target_role == "RETURN_SITE"


@pytest.mark.parametrize(
    ("binding_valid", "return_valid", "population"),
    [
        (True, True, ProgramPopulation.TRUE_ABSTENTION),
        (False, True, ProgramPopulation.SINGLE_ROOT),
        (True, False, ProgramPopulation.SINGLE_ROOT),
        (False, False, ProgramPopulation.MULTIPLE_ROOT),
    ],
)
def test_argument_return_factorial_oracle(
    binding_valid, return_valid, population, tmp_path
):
    spec = ProgramSpec(
        ProgramFamily.ARGUMENT_VS_RETURN, 100, binding_valid=binding_valid,
        return_valid=return_valid,
    )
    program = generate_program(spec, tmp_path / f"{binding_valid}-{return_valid}")

    assert program.oracle.population is population
    assert ("ARGUMENT_SOURCE_DEFECT" in program.oracle.responsibility_kinds) is (not binding_valid)
    assert ("RETURN_CONTRACT_DEFECT" in program.oracle.responsibility_kinds) is (not return_valid)


def test_import_oracle_distinguishes_runtime_scc_from_observers(tmp_path):
    cycle = generate_program(
        ProgramSpec(
            ProgramFamily.IMPORT_GRAPH, 101, scc_size=3, observer_import=True,
        ),
        tmp_path / "cycle",
    )
    type_only = generate_program(
        ProgramSpec(
            ProgramFamily.NEGATIVE_CONTROL, 102, scc_size=3,
            observer_import=True, type_checking_edge=True,
        ),
        tmp_path / "type-only",
    )

    assert len(cycle.oracle.import_scc_members) == 3
    assert cycle.oracle.responsibility_kinds == ("IMPORT_GRAPH_DEFECT",)
    assert type_only.oracle.abstention_expected is True
    assert type_only.oracle.import_scc_members == ()


def test_generated_suites_have_compositional_and_sealed_populations():
    development = generate_suite("development")
    compositional = generate_suite("compositional")
    holdout = generate_suite("holdout_v10")

    assert 150 <= len(development) <= 300
    assert len(holdout) == 48
    assert any(item.argument_style == "mixed" and item.wrapper_depth == 3 for item in compositional)
    assert not any(item.argument_style == "mixed" for item in development)
    assert {item.id for item in development}.isdisjoint(item.id for item in holdout)


def test_holdout_supplement_covers_previously_zero_denominators():
    supplement = generate_holdout_v10_supplement()

    assert len(supplement) == 12
    assert {item.family for item in supplement} == {
        ProgramFamily.ARGUMENT_VS_RETURN,
        ProgramFamily.IMPORT_GRAPH,
        ProgramFamily.NEGATIVE_CONTROL,
        ProgramFamily.PRODUCER_MANIFESTATION,
    }
    assert {item.seed for item in supplement} == set(range(3048, 3060))


def test_counterfactual_pairs_switch_argument_and_return_responsibility(tmp_path):
    pairs = generate_counterfactual_pairs(count=24)

    assert len(pairs) == 24
    base = generate_program(pairs[0][0], tmp_path / "base")
    changed = generate_program(pairs[0][1], tmp_path / "changed")
    assert base.oracle.responsibility_kinds == ("ARGUMENT_SOURCE_DEFECT",)
    assert changed.oracle.responsibility_kinds == ("RETURN_CONTRACT_DEFECT",)
    assert base.oracle.preferred_target_role == "CALL_SITE"
    assert changed.oracle.preferred_target_role == "RETURN_SITE"


def test_counterfactual_evaluation_requires_correct_semantic_switch(monkeypatch):
    pair = generate_counterfactual_pairs(count=1)
    base, changed = pair[0]

    def result(spec, responsibility, strategy, target):
        return {
            "id": spec.id,
            "actual": {
                "hypotheses": [{"root_cause_id": "RC1"}],
                "root_cause_candidates": [{
                    "id": "RC1", "responsibility": {"kind": responsibility},
                }],
                "recommended_candidate_id": "C1",
                "candidates": [{
                    "id": "C1", "strategy_kind": strategy,
                    "repair_targets": [{"scope_kind": target}],
                }],
            },
            "retrieval": {},
            "metrics_v10": {"root_validity": True, "repair_pair_validity": True},
        }

    fake = {
        "mode": "live", "cases": 2, "results": [
            result(base, "ARGUMENT_SOURCE_DEFECT", "CORRECT_ARGUMENT", "CALL_SITE"),
            result(changed, "RETURN_CONTRACT_DEFECT", "CORRECT_RETURN_VALUE", "RETURN_SITE"),
        ],
    }
    monkeypatch.setattr(
        "tern.orchestrator.predictive.testing.evaluation.evaluate_generated_live",
        lambda *_args, **_kwargs: fake,
    )

    report = evaluate_generated_counterfactuals(pair, reasoner=object())

    assert report["pairs"] == 1
    assert report["counterfactual_metrics"]["root_sensitivity"]["value"] == 1.0
    assert report["counterfactual_metrics"]["repair_sensitivity"]["value"] == 1.0
    assert report["counterfactual_metrics"]["target_sensitivity"]["value"] == 1.0


def test_true_abstention_is_not_argument_vs_return_accuracy_population(tmp_path):
    program = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 126,
            binding_valid=True, return_valid=True,
        ),
        tmp_path / "case",
    )
    metrics = live_case_metrics(program, {
        "actual": {"hypotheses": [], "candidates": [], "recommended_candidate_id": None},
        "retrieval": {},
    })

    assert metrics["root_validity"] is True
    assert metrics["argument_vs_return_accuracy"] is None


def test_stratified_sample_covers_families_and_is_deterministic():
    specs = generate_suite("development")
    first = stratified_sample(specs, 28)
    second = stratified_sample(reversed(specs), 28)

    assert first == second
    assert len({item.family for item in first}) == len({item.family for item in specs})
    factorial = [
        item for item in first
        if item.family is ProgramFamily.ARGUMENT_VS_RETURN
    ]
    assert {
        (item.binding_valid, item.return_valid) for item in factorial
    } == {(False, False), (False, True), (True, False), (True, True)}
    with pytest.raises(ValueError, match="positive"):
        stratified_sample(specs, 0)


def test_factorial_wrapper_depth_changes_topology_and_valid_control_has_no_traceback(tmp_path):
    direct = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 111,
            binding_valid=True, return_valid=False, wrapper_depth=0,
        ),
        tmp_path / "direct",
    )
    wrapped = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 111,
            binding_valid=True, return_valid=False, wrapper_depth=2,
        ),
        tmp_path / "wrapped",
    )
    valid = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 112,
            binding_valid=True, return_valid=True, wrapper_depth=2,
        ),
        tmp_path / "valid",
    )

    assert not any("relay_1_" in source for source in direct.files.values())
    assert any("relay_1_" in source for source in wrapped.files.values())
    assert "Traceback" not in valid.case.problem


def test_seed_changes_surface_but_preserves_structural_oracle(tmp_path):
    spec = ProgramSpec(
        ProgramFamily.ARGUMENT_BINDING, 120, argument_style="keyword",
        defective_slot=1,
    )
    renamed = replace(spec, seed=121)
    first = generate_program(spec, tmp_path / "first")
    second = generate_program(renamed, tmp_path / "second")

    assert first.files != second.files
    assert first.oracle.responsibility_kinds == second.oracle.responsibility_kinds
    assert first.oracle.binding_slot == second.oracle.binding_slot
    assert first.oracle.expected_repair_strategies == second.oracle.expected_repair_strategies


def test_proven_default_binding_preserves_responsibility_on_concrete_value_root(tmp_path):
    program = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_BINDING, 122, argument_style="default",
            slot_count=3, defective_slot=1, wrapper_depth=1,
        ),
        tmp_path / "case",
    )
    context, _retrieval_value = _retrieval(program.case)
    concrete = next(
        item for item in context.root_cause_candidates
        if item.cause_kind.value == "TYPE_FLOW" and item.origin_line == 1
    )

    assert concrete.responsibility is not None
    assert concrete.responsibility.kind is CausalResponsibilityKind.ARGUMENT_BINDING_DEFECT
    assert concrete.responsibility.parameter_ordinal == 1


def test_temporary_argument_lineage_preserves_slot_before_return(tmp_path):
    program = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_BINDING, 123, argument_style="keyword",
            slot_count=3, defective_slot=1, wrapper_depth=2,
            temporary_variable=True,
        ),
        tmp_path / "case",
    )
    context, _retrieval_value = _retrieval(program.case)
    concrete = next(
        item for item in context.root_cause_candidates
        if item.cause_kind is RootCauseKind.TYPE_FLOW
        and item.responsibility
        and item.responsibility.parameter_ordinal == 1
    )

    assert concrete.responsibility is not None
    assert concrete.responsibility.kind is CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT
    assert concrete.responsibility.return_site_id is None


def test_demonstrated_return_contract_dominates_generic_sibling_binding(tmp_path):
    program = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 124,
            binding_valid=True, return_valid=False, wrapper_depth=1,
        ),
        tmp_path / "case",
    )
    context, _retrieval_value = _retrieval(program.case)
    dominant = structurally_dominant_root(
        context.root_cause_candidates,
        context.problem,
        context.causal_slice,
    )

    assert dominant is not None
    assert dominant.responsibility is not None
    assert dominant.responsibility.kind is CausalResponsibilityKind.RETURN_CONTRACT_DEFECT
    assert dominant.contract_demonstrated is True


def test_satisfied_return_annotation_does_not_mask_bad_argument(tmp_path):
    program = generate_program(
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 125,
            binding_valid=False, return_valid=True, wrapper_depth=0,
        ),
        tmp_path / "case",
    )
    context, _retrieval_value = _retrieval(program.case)
    dominant = structurally_dominant_root(
        context.root_cause_candidates,
        context.problem,
        context.causal_slice,
    )

    assert dominant is not None
    assert dominant.responsibility is not None
    assert dominant.responsibility.kind is CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT
    satisfied_return = next(
        item for item in context.root_cause_candidates
        if item.cause_kind is RootCauseKind.RETURN_CONTRACT
        and item.contract_demonstrated
    )
    assert satisfied_return.responsibility is not None
    assert satisfied_return.responsibility.kind is CausalResponsibilityKind.MANIFESTATION_ONLY


def test_structural_metrics_decompose_generation_from_selection(tmp_path):
    program = generate_program(
        ProgramSpec(ProgramFamily.ARGUMENT_BINDING, 130, defective_slot=1),
        tmp_path / "case",
    )
    matching_root = {
        "cause_kind": "ARGUMENT_BINDING",
        "origin_path": next(iter(program.oracle.origin_paths)),
        "failure_path": next(iter(program.oracle.origin_paths)),
        "responsibility": {
            "kind": "ARGUMENT_SOURCE_DEFECT",
            "parameter_ordinal": 1,
            "keyword_binding": None,
            "defect_bearing_relation": "actual_argument_to_formal_parameter",
        },
    }
    metrics = structural_case_metrics(program, {
        "root_cause_candidates": [matching_root],
        "retrieved_files": list(program.oracle.origin_paths),
    })

    assert metrics["retrieval_accuracy"] is True
    assert metrics["oracle_root_candidate_recall"] is True
    assert metrics["oracle_relation_recall"] is True
    assert metrics["oracle_slot_identity_recall"] is True


def test_structural_evaluation_runs_without_reasoner():
    report = evaluate_generated_structural(generate_suite("development")[:4])

    assert report["mode"] == "structural"
    assert report["cases"] == 4
    assert report["metrics"]["oracle_root_candidate_recall"]["denominator"] == 4
    assert "argument_matrix" in report
    assert "failure_fingerprints" in report


def test_oracle_injection_stops_before_reasoner_when_generation_missed(tmp_path):
    program = generate_program(
        ProgramSpec(ProgramFamily.ARGUMENT_BINDING, 150), tmp_path / "case"
    )
    impossible = replace(
        program,
        oracle=replace(program.oracle, responsibility_kinds=("IMPORT_GRAPH_DEFECT",)),
    )

    class NeverCalled:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("reasoner must not run without an oracle candidate")

    value = evaluate_oracle_injection(impossible, reasoner=NeverCalled())
    assert value["oracle_candidates"] == 0
    assert value["analysis"] is None


def test_failure_reducer_minimizes_only_while_failure_survives():
    spec = ProgramSpec(
        ProgramFamily.ARGUMENT_BINDING, 160, wrapper_depth=3,
        temporary_variable=True, distractor=True,
    )
    reduced = reduce_failing_spec(spec, lambda item: item.wrapper_depth >= 1)

    assert reduced.wrapper_depth == 3
    assert reduced.temporary_variable is False
    assert reduced.distractor is False


def test_holdout_v10_manifest_hash_matches_generator():
    manifest = json.loads(
        (Path(__file__).parent / "data" / "predictive" / "v10" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    holdout = generate_suite("holdout_v10")

    assert manifest["generator_version"] == 4
    assert manifest["holdout_v10"]["count"] == len(holdout)
    assert manifest["holdout_v10"]["sha256"] == suite_sha256(holdout)


def test_generated_eval_cli_parser_exposes_live_as_opt_in():
    args = cli.build_parser().parse_args([
        "predictive-generated-eval", "--mode", "structural",
        "--split", "compositional", "--limit", "7", "--json",
    ])

    assert args.command == "predictive-generated-eval"
    assert args.mode == "structural"
    assert args.split == "compositional"
    assert args.limit == 7
    assert args.json_output is True

    counterfactual = cli.build_parser().parse_args([
        "predictive-generated-eval", "--mode", "live",
        "--split", "counterfactual", "--limit", "3",
    ])
    assert counterfactual.split == "counterfactual"
    assert counterfactual.limit == 3
