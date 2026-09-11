from __future__ import annotations

import json

from tern.orchestrator import cli
from tern.orchestrator.predictive.benchmark_v5 import (
    adjudicate_predictive_result_v5,
    evaluate_predictive_cases_v5,
    load_benchmark_v5_adjudications,
)
from tern.orchestrator.predictive.evaluation import (
    CORPUS_ROOT,
    load_predictive_cases,
    predictive_corpus_hash,
)


def test_v5_development_and_holdout_truth_are_complete():
    development = load_predictive_cases(split="development")
    holdout = load_predictive_cases(split="holdout_v5")

    assert len(development) == len(load_benchmark_v5_adjudications(development)) == 57
    assert len(holdout) == len(load_benchmark_v5_adjudications(holdout)) == 18


def test_holdout_v5_matches_frozen_hash():
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert predictive_corpus_hash(split="holdout_v5") == manifest["holdout_v5_sha256"]
    assert manifest["holdout_v5_frozen_at"].endswith("Z")


def test_v5_separates_strategy_target_and_pair_validity():
    case = next(item for item in load_predictive_cases(split="development") if item.id == "PC5D-005")
    truth = load_benchmark_v5_adjudications((case,))[0]
    result = {
        "id": case.id,
        "actual": {
            "insufficient_evidence": False,
            "hypotheses": [{"root_cause_id": "R1"}],
            "root_cause_candidates": [{
                "id": "R1", "origin_path": "pkg/mathops.py",
                "origin_symbol": "values", "cause_kind": "ARGUMENT_BINDING",
            }],
            "recommended_candidate_id": "C1",
            "candidates": [{
                "id": "C1", "strategy_kind": "VALIDATE_BOUNDARY",
                "target_files": ["pkg/mathops.py"], "target_symbols": ["values"],
                "repair_targets": [{
                    "path": "pkg/mathops.py", "scope_kind": "PARAMETER",
                    "symbol": "mean", "parameter": "values", "attribute": None,
                    "expression_id": None, "line": 5,
                }],
            }],
        },
    }

    metrics = adjudicate_predictive_result_v5(result, truth)

    assert metrics["repair_strategy_validity"] is True
    assert metrics["repair_target_validity"] is True
    assert metrics["repair_pair_validity"] is True
    assert metrics["top1_validity"] is True


def test_v5_retrieval_never_calls_qwen():
    case = next(item for item in load_predictive_cases(split="development") if item.id == "PC5D-005")
    report = evaluate_predictive_cases_v5((case,), mode="retrieval")

    assert report["version"] == 5
    assert report["qwen"]["requests"] == 0
    assert report["safety"]["tool_dispatches"] == 0


def test_predictive_eval_cli_supports_v5_without_qwen(monkeypatch, capsys):
    class Runtime:
        def __init__(self, _settings):
            pass

        def ensure_llama_server(self, _wait):
            raise AssertionError("retrieval evaluation must not start Qwen")

    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "RuntimeManager", Runtime)

    assert cli.main([
        "predictive-eval", "--benchmark-version", "5", "--split", "development",
        "--mode", "retrieval", "--limit", "1", "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 5
