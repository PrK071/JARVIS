from __future__ import annotations

import json
from collections import Counter

import pytest

from tern.orchestrator import cli
from tern.orchestrator.predictive.benchmark_v4 import (
    EvaluationStatus,
    adjudicate_predictive_result_v4,
    evaluate_predictive_cases_v4,
    format_benchmark_v4,
    load_benchmark_v4_adjudications,
    summarize_benchmark_v4,
)
from tern.orchestrator.predictive.evaluation import (
    CORPUS_ROOT,
    evaluate_predictive_cases,
    load_predictive_cases,
    predictive_corpus_hash,
)


def test_all_development_cases_have_individual_v4_adjudications():
    cases = tuple(
        case for case in load_predictive_cases(split="development")
        if not case.id.startswith("PC5D-")
    )
    adjudications = load_benchmark_v4_adjudications(cases)

    assert len(adjudications) == 42
    assert Counter(item.status for item in adjudications) == {
        EvaluationStatus.CANONICAL: 16,
        EvaluationStatus.MULTIPLE_VALID_ANSWERS: 16,
        EvaluationStatus.AMBIGUOUS: 1,
        EvaluationStatus.INCOMPLETE: 7,
        EvaluationStatus.INVALID: 2,
    }
    assert all(item.notes and item.legacy_expectation for item in adjudications)


def test_holdout_v4_is_complete_and_matches_sealed_hash():
    cases = load_predictive_cases(split="holdout_v4")
    adjudications = load_benchmark_v4_adjudications(cases)
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert len(cases) == len(adjudications) == 18
    assert predictive_corpus_hash(split="holdout_v4") == manifest["holdout_v4_sha256"]


def test_adjudication_loader_rejects_missing_records():
    cases = load_predictive_cases(split="development")[:2]

    with pytest.raises(ValueError, match="missing benchmark v4 adjudications"):
        load_benchmark_v4_adjudications(cases, CORPUS_ROOT / "missing")


def _result(*, root_kind="NULL_FLOW", strategy="VALIDATE_BOUNDARY", target="pkg/totals.py", winner=True):
    candidate = {
        "id": "C1",
        "strategy_kind": strategy,
        "target_files": [target],
        "target_symbols": ["calculate" if target.endswith("totals.py") else "load_order"],
    }
    return {
        "id": "PC3D-001",
        "actual": {
            "insufficient_evidence": False,
            "ranking_ambiguous": False,
            "recommended_candidate_id": "C1" if winner else None,
            "hypotheses": [{"root_cause_id": "R1"}],
            "root_cause_candidates": [{
                "id": "R1",
                "origin_path": "pkg/repository.py",
                "origin_symbol": None,
                "cause_kind": root_kind,
            }],
            "candidates": [candidate],
        },
    }


def test_structural_root_correctness_does_not_depend_on_hypothesis_wording():
    case = next(item for item in load_predictive_cases(split="development") if item.id == "PC3D-001")
    truth = load_benchmark_v4_adjudications((case,))[0]

    metrics = adjudicate_predictive_result_v4(_result(), truth)

    assert metrics["root_cause_validity"] is True
    assert metrics["repair_validity"] is True
    assert metrics["top1_is_valid"] is True
    assert metrics["classification"] == "NONE"


def test_valid_nonpreferred_repair_is_not_a_correctness_failure():
    case = next(item for item in load_predictive_cases(split="development") if item.id == "PC3D-001")
    truth = load_benchmark_v4_adjudications((case,))[0]

    metrics = adjudicate_predictive_result_v4(_result(), truth)

    assert metrics["top1_is_valid"] is True
    assert metrics["top1_is_preferred"] is False
    assert metrics["recommendation_valid"] is True


def test_strategy_target_pair_prevents_unrelated_valid_strategy():
    case = next(item for item in load_predictive_cases(split="development") if item.id == "PC3D-001")
    truth = load_benchmark_v4_adjudications((case,))[0]

    metrics = adjudicate_predictive_result_v4(
        _result(strategy="FIX_PRODUCER", target="pkg/totals.py"), truth
    )

    assert metrics["repair_validity"] is False
    assert "REPAIR_TARGET_ERROR" in metrics["failure_codes"]


def test_v4_summary_separates_benchmark_and_engine_errors():
    cases = tuple(
        case for case in load_predictive_cases(split="development")
        if not case.id.startswith("PC5D-")
    )
    truths = load_benchmark_v4_adjudications(cases)
    base = evaluate_predictive_cases(cases, mode="retrieval")

    report = summarize_benchmark_v4(base, truths)

    assert report["version"] == 4
    assert report["adjudication_status_counts"]["INVALID"] == 2
    assert report["quality"]["benchmark_error_rate"] == pytest.approx(10 / 42)
    assert report["stage_gate"]["passed"] is False


def test_v4_retrieval_never_calls_reasoner_and_preserves_safety():
    case = next(item for item in load_predictive_cases(split="development") if item.id == "PC3D-001")
    report = evaluate_predictive_cases_v4((case,), mode="retrieval")

    assert report["qwen"]["requests"] == 0
    assert report["safety"]["filesystem_mutations"] == 0
    assert report["safety"]["tool_dispatches"] == 0
    assert report["safety"]["authority_grants"] == 0
    assert "Stage gate:" in format_benchmark_v4(report)


def test_predictive_eval_cli_supports_v4_without_starting_qwen(monkeypatch, capsys):
    class Runtime:
        def __init__(self, _settings):
            pass

        def ensure_llama_server(self, _wait):
            raise AssertionError("retrieval evaluation must not start Qwen")

    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "RuntimeManager", Runtime)

    assert cli.main([
        "predictive-eval",
        "--benchmark-version", "4",
        "--split", "development",
        "--mode", "retrieval",
        "--limit", "1",
        "--json",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == 4
    assert report["qwen"]["requests"] == 0
