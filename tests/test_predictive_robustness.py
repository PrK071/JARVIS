from __future__ import annotations

import ast
import json
from pathlib import Path

from tern.orchestrator.predictive.evaluation import _retrieval, load_predictive_cases
from tern.orchestrator.predictive.benchmark_v5 import load_benchmark_v5_adjudications
from tern.orchestrator.predictive.robustness import (
    AbstentionStage,
    MetamorphicTransformation,
    classify_abstention_stage,
    coverage_funnel,
    evaluate_counterfactual_suite,
    evaluate_robustness_suite,
    evaluate_reasoner_order_bias,
    load_counterfactual_cases,
    load_metamorphic_cases,
    materialize_metamorphic_case,
    robustness_suite_hash,
    run_ablation,
    summarize_robustness_gate,
)


def _case(identifier: str):
    return next(item for item in load_predictive_cases() if item.id == identifier)


def test_v6_suite_has_required_development_transformations_and_counterfactuals():
    transformations = load_metamorphic_cases(split="development")
    kinds = {item.transformation for item in transformations}

    assert len({item.base_case_id for item in transformations}) >= 20
    assert len(transformations) >= 60
    assert {
        MetamorphicTransformation.SYMBOL_RENAME,
        MetamorphicTransformation.FILE_RENAME,
        MetamorphicTransformation.HARMLESS_COMMENT,
        MetamorphicTransformation.DOCSTRING_DISTRACTOR,
        MetamorphicTransformation.ERROR_PARAPHRASE,
        MetamorphicTransformation.FUNCTION_REORDER,
        MetamorphicTransformation.TEMPORARY_VARIABLE,
        MetamorphicTransformation.POSITIONAL_TO_KEYWORD,
        MetamorphicTransformation.LEXICAL_DISTRACTOR,
        MetamorphicTransformation.WRAPPER_FUNCTION,
    } <= kinds
    assert len(load_counterfactual_cases(split="development")) >= 10


def test_every_metamorphic_variant_remains_valid_python(tmp_path):
    specs = load_metamorphic_cases(split="development")
    cases = {item.id: item for item in load_predictive_cases()}
    base_cases = [cases[item] for item in sorted({spec.base_case_id for spec in specs})]
    truths = {item.case_id: item for item in load_benchmark_v5_adjudications(base_cases)}

    for spec in specs:
        transformed, _truth, _mapping = materialize_metamorphic_case(
            spec, cases[spec.base_case_id], truths[spec.base_case_id], tmp_path
        )
        for path in transformed.fixture_root.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_every_holdout_variant_materializes_before_live_evaluation(tmp_path):
    specs = load_metamorphic_cases(split="holdout_v6")
    cases = {item.id: item for item in load_predictive_cases()}
    base_cases = [cases[item] for item in sorted({spec.base_case_id for spec in specs})]
    truths = {item.case_id: item for item in load_benchmark_v5_adjudications(base_cases)}

    for spec in specs:
        transformed, _truth, _mapping = materialize_metamorphic_case(
            spec, cases[spec.base_case_id], truths[spec.base_case_id], tmp_path
        )
        assert transformed.fixture_root.is_dir()
        for path in transformed.fixture_root.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_symbol_and_file_renames_update_structural_truth(tmp_path):
    specs = [
        item for item in load_metamorphic_cases(split="development")
        if item.id.startswith("MV6D-015::1") or item.id.startswith("MV6D-016::1")
    ]
    cases = {item.id: item for item in load_predictive_cases()}
    base_cases = [cases[item.base_case_id] for item in specs]
    truths = {item.case_id: item for item in load_benchmark_v5_adjudications(base_cases)}

    values = [
        materialize_metamorphic_case(
            spec, cases[spec.base_case_id], truths[spec.base_case_id], tmp_path
        )
        for spec in specs
    ]

    assert values[0][2].symbols
    assert values[1][2].paths
    assert all(item.acceptable_root_causes for _case, item, _mapping in values)


def test_abstention_funnel_identifies_exact_loss_stage():
    results = [
        {"actual": {"root_cause_candidates": []}, "retrieval": {"failure_codes": []}},
        {"actual": {"root_cause_candidates": [{"id": "R1"}], "hypotheses": []}, "retrieval": {"failure_codes": []}},
        {"actual": {"root_cause_candidates": [{"id": "R1"}], "hypotheses": [{"id": "H1"}], "candidates": [], "rejected_candidates": []}, "retrieval": {"failure_codes": []}},
        {"actual": {"root_cause_candidates": [{"id": "R1"}], "hypotheses": [{"id": "H1"}], "candidates": [{"id": "C1"}], "ranking_ambiguous": True}, "retrieval": {"failure_codes": []}},
    ]

    assert classify_abstention_stage(results[0]) is AbstentionStage.CAUSAL_SLICE
    assert classify_abstention_stage(results[1]) is AbstentionStage.ROOT_SELECTION
    assert classify_abstention_stage(results[2]) is AbstentionStage.REPAIR_GENERATION
    assert classify_abstention_stage(results[3]) is AbstentionStage.RANKING
    assert coverage_funnel(results)["n_abstained"] == 4


def test_reasoner_selection_is_invariant_to_candidate_evidence_and_id_order():
    context = _retrieval(_case("PC5D-013"))[0]

    class StructuralReasoner:
        def chat(self, messages, **_kwargs):
            payload = json.loads(messages[-1]["content"])
            root = max(
                payload["root_cause_candidates"],
                key=lambda item: (item["structural_support"], item["score"]),
            )
            strategy = root["allowed_repair_strategies"][0]
            target_file, target_symbol = root["causal_targets"][0]
            value = {"selections": [{
                "root_cause_id": root["id"],
                "selection_reason": "STRONGER_CAUSAL_PATH",
                "rejected": [],
                "claim": "selected from the proven causal path",
                "strategies": [{
                    "kind": strategy,
                    "target_file": target_file,
                    "target_symbol": target_symbol or "",
                    "rationale": "repair the structural origin",
                    "reason": "STRONGER_CAUSAL_PATH",
                }],
            }]}
            return {"choices": [{"message": {"content": json.dumps(value)}}]}

    result = evaluate_reasoner_order_bias(context, StructuralReasoner)

    assert result["candidate_order_invariance"] is True
    assert result["evidence_order_invariance"] is True
    assert result["identifier_invariance"] is True


def test_ablation_runner_reports_on_off_delta():
    result = run_ablation(
        "lexical_rule",
        lambda enabled: {"invariance": 0.7 if enabled else 0.9},
    )

    assert result["delta"]["invariance"] == -0.2


def test_counterfactual_evaluation_reuses_known_structural_signatures(monkeypatch):
    from tern.orchestrator.predictive import robustness

    pairs = load_counterfactual_cases(split="holdout_v6")
    identifiers = {item for pair in pairs for item in (pair.base_case_id, pair.counterfactual_case_id)}
    signatures = {
        item: {
            "root_kind": "UNKNOWN", "root_path": item, "root_symbol": None,
            "repair_strategy": "OTHER", "target_kind": None, "target_path": None,
            "target_symbol": None, "abstained": False, "causal_path_roles": (),
        }
        for item in identifiers
    }
    monkeypatch.setattr(
        robustness,
        "evaluate_predictive_cases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("known signatures must not invoke the reasoner")
        ),
    )

    report = evaluate_counterfactual_suite(
        split="holdout_v6", mode="live", reasoner=object(),
        known_signatures=signatures,
    )

    assert report["pairs"] == len(pairs)
    assert report["telemetry"]["requests"] == 0
    assert report["safety"]["passed"] is True


def test_robustness_report_exposes_denominators_breakdown_and_aggregate_safety():
    metamorphic = evaluate_robustness_suite(
        split="development", mode="baseline", limit=1,
    )
    counterfactual = evaluate_counterfactual_suite(
        split="development", mode="baseline", limit=1,
        known_signatures=metamorphic["base_signatures"],
    )
    report = summarize_robustness_gate(metamorphic, counterfactual)

    root = report["metric_denominators"]["root_cause_invariance"]
    assert root["denominator"] == 1
    assert metamorphic["variant_population"]["n_total"] == 1
    assert "variant_root_cause_validity" in report["metric_denominators"]
    assert "variant_false_abstention_rate" in report["stage_gate"]["checks"]
    assert metamorphic["transformation_breakdown"]
    assert report["safety"]["passed"] is True
    assert report["telemetry"]["requests"] == 0


def test_invariant_but_invalid_variants_fail_the_quality_gate():
    zero_safety = {
        "filesystem_mutations": 0, "tool_dispatches": 0,
        "execution_authorized": 0, "authority_grants": 0,
        "destructive_actions": 0, "forbidden_candidate_recommendations": 0,
    }
    metamorphic = {
        "mode": "live", "split": "holdout_v6", "safety": zero_safety,
        "telemetry": {"requests": 1, "average_request_ms": 1000},
        "metrics": {
            "root_cause_invariance": 1.0, "repair_strategy_invariance": 1.0,
            "variant_root_cause_validity": 0.5,
            "variant_repair_pair_validity": 0.5, "variant_top1_validity": 1.0,
            "variant_recommendation_validity_precision": 1.0,
            "variant_false_abstention_rate": 0.0,
        },
    }
    counterfactual = {
        "safety": zero_safety, "telemetry": {},
        "metrics": {"counterfactual_root_sensitivity": 1.0},
    }

    gate = summarize_robustness_gate(metamorphic, counterfactual)["stage_gate"]

    assert gate["passed"] is False
    assert gate["checks"]["root_cause_invariance"] is True
    assert gate["checks"]["variant_root_cause_validity"] is False


def test_holdout_v6_is_sealed():
    manifest = json.loads(
        (Path(__file__).parent / "data" / "predictive" / "manifest.json").read_text(encoding="utf-8")
    )

    assert len(load_metamorphic_cases(split="holdout_v6")) == manifest["holdout_v6_cases"]
    assert robustness_suite_hash(split="holdout_v6") == manifest["holdout_v6_sha256"]
