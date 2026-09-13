from __future__ import annotations

from tern.orchestrator.predictive.benchmark_v7 import summarize_benchmark_v7
from tern.orchestrator.predictive.evaluation import (
    CORPUS_ROOT,
    load_predictive_cases,
    predictive_corpus_hash,
)
from tern.orchestrator.predictive.robustness import (
    CoverageStageV7,
    coverage_funnel_v7,
    coverage_trace_v7,
)


def _result(*, recovery, roots=(), hypotheses=(), strategies=(), candidates=(), winner=None):
    return {
        "id": "case",
        "expected": {"insufficient_evidence": False},
        "retrieval": {
            "retrieved_files": recovery.get("final_files", ()),
            "evidence_ledger": {"atoms": [{}] if recovery.get("final_files") else []},
            "metrics": {},
            "recovery": recovery,
        },
        "actual": {
            "recovery": recovery,
            "causal_slice": {"failure_site_id": "N1" if roots else None},
            "root_cause_candidates": list(roots),
            "hypotheses": list(hypotheses),
            "repair_strategies": list(strategies),
            "candidates": list(candidates),
            "rejected_candidates": [],
            "recommended_candidate_id": winner,
            "ranking_ambiguous": False,
        },
    }


def test_detailed_funnel_identifies_initial_retrieval_loss():
    result = _result(recovery={
        "attempted": True,
        "succeeded": False,
        "outcome": "RECOVERY_NO_NEW_EVIDENCE",
        "initial_files": [],
        "final_files": [],
    })

    trace = coverage_trace_v7(result)

    assert trace["loss_stage"] == CoverageStageV7.INITIAL_RETRIEVAL.value
    assert trace["recovery_attempted"] is True


def test_detailed_funnel_preserves_every_stage_count():
    recovery = {
        "attempted": True,
        "succeeded": True,
        "outcome": "RECOVERY_SUCCEEDED",
        "initial_files": [],
        "final_files": ["pkg/a.py"],
    }
    result = _result(
        recovery=recovery,
        roots=({"id": "R1"},),
        hypotheses=({"id": "H1"},),
        strategies=({"id": "S1"},),
        candidates=({"id": "C1", "repair_targets": [{"path": "pkg/a.py"}], "eligible": True},),
        winner="C1",
    )

    funnel = coverage_funnel_v7((result,))

    assert funnel["stage_output_counts"]["recommendation"] == 1
    assert funnel["loss_stage_counts"] == {"RECOMMENDED": 1}


def test_v7_recovery_metrics_use_explicit_denominators():
    recovery = {
        "attempted": True,
        "succeeded": True,
        "outcome": "RECOVERY_SUCCEEDED",
        "initial_sufficiency": {"sufficient": False},
        "initial_files": [],
        "final_files": ["pkg/a.py"],
        "extra_files": 1,
        "extra_evidence_atoms": 3,
        "actions": [{"reason": "EXPAND_SYMBOL_REFERENCE"}],
    }
    source = _result(recovery=recovery, roots=({"id": "R1"},)) | {
        "metrics_v5": {
            "evaluable": True,
            "root_cause_validity": True,
            "repair_pair_validity": True,
            "top1_validity": True,
            "recommendation_coverage": True,
            "recommendation_valid": True,
            "false_abstention": False,
            "failure_codes": [],
        },
        "benchmark_v5": {"abstention_expected": False},
    }
    report = summarize_benchmark_v7({
        "results": [source],
        "quality": {
            "root_cause_validity": 1.0,
            "repair_pair_validity": 1.0,
            "top1_validity": 1.0,
            "recommendation_validity_precision": 1.0,
            "false_abstention": 0.0,
        },
        "grounding": {"evidence_reference_validity": 1.0, "unsupported_claim_rate": 0.0},
        "reasoning": {"forbidden_candidate_recommendation_rate": 0.0},
        "safety": {"passed": True},
        "mode": "retrieval",
        "split": "development",
    })

    assert report["recovery_v7"]["attempts"] == 1
    assert report["recovery_v7"]["success_rate"] == 1.0
    assert report["recovery_v7"]["precision"] == 1.0
    assert report["recovery_v7"]["false_abstention_before"] == 1.0
    assert report["recovery_v7"]["false_abstention_after"] == 0.0


def test_live_recovery_precision_requires_valid_final_recommendation():
    recovery = {
        "attempted": True,
        "succeeded": True,
        "outcome": "RECOVERY_SUCCEEDED",
        "initial_sufficiency": {"sufficient": False},
        "initial_files": [],
        "final_files": ["pkg/a.py"],
        "extra_files": 1,
        "extra_evidence_atoms": 2,
        "actions": [{"reason": "EXPAND_CALLER"}],
    }
    source = _result(recovery=recovery, roots=({"id": "R1"},)) | {
        "metrics_v5": {
            "evaluable": True,
            "root_cause_validity": False,
            "recommendation_valid": False,
            "false_abstention": False,
            "failure_codes": ["ROOT_CAUSE_SELECTION_ERROR"],
        },
        "benchmark_v5": {"abstention_expected": False},
    }
    report = summarize_benchmark_v7({
        "results": [source],
        "quality": {},
        "grounding": {},
        "safety": {"passed": True},
        "mode": "live",
        "split": "development",
        "latency": {"qwen_request": {"average_ms": 1.0}},
    })

    assert report["recovery_v7"]["success_rate"] == 0.0
    assert report["recovery_v7"]["precision"] == 0.0


def test_holdout_v7_is_frozen_and_structurally_adjudicated():
    import json

    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    cases = load_predictive_cases(split="holdout_v7")

    assert len(cases) == manifest["holdout_v7_cases"] == 20
    assert predictive_corpus_hash(split="holdout_v7") == manifest["holdout_v7_sha256"]
    assert all(item.project_fixture != "recovery_chain_dev_v7" for item in cases)
