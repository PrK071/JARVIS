from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tern.orchestrator import cli
from tern.orchestrator.predictive.evaluation import (
    CORPUS_ROOT,
    evaluate_predictive_cases,
    format_predictive_evaluation,
    load_predictive_cases,
    predictive_corpus_hash,
)


def response(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}]}


class GroundedReasoner:
    def chat(self, messages, *, response_format, **_kwargs):
        payload = json.loads(messages[-1]["content"])
        root = next(
            (item for item in payload["root_cause_candidates"] if item["origin_path"] == "pkg/calc.py"),
            payload["root_cause_candidates"][0],
        )
        compatible = {
            "RETURN_CONTRACT": "VALIDATE_BOUNDARY",
            "ARGUMENT_BINDING": "VALIDATE_BOUNDARY",
            "ATTRIBUTE_VALUE": "VALIDATE_BOUNDARY",
            "TYPE_FLOW": "VALIDATE_BOUNDARY",
            "STATE_PROPAGATION": "VALIDATE_BOUNDARY",
            "NULL_FLOW": "VALIDATE_BOUNDARY",
        }.get(root["cause_kind"], "OTHER")
        return response(
            {"selections": [{
                "root_cause_id": root["id"],
                "claim": "A None operand reaches the tax addition",
                "strategies": [{
                    "kind": compatible,
                    "target_file": root["origin_path"],
                    "target_symbol": root["origin_symbol"] or "",
                    "rationale": "Validate the operand before adding tax",
                }],
            }]}
        )


def test_corpus_loads_versioned_development_and_holdout_splits():
    all_cases = load_predictive_cases()
    development = load_predictive_cases(split="development")
    holdout = load_predictive_cases(split="historical_holdout_v1")
    holdout_v2 = load_predictive_cases(split="historical_holdout_v2")
    holdout_v2_alias = load_predictive_cases(split="holdout_v2")
    holdout_v3 = load_predictive_cases(split="holdout_v3")
    historical_holdout_v3 = load_predictive_cases(split="historical_holdout_v3")
    holdout_v4 = load_predictive_cases(split="holdout_v4")
    historical_holdout_v4 = load_predictive_cases(split="historical_holdout_v4")
    holdout_v5 = load_predictive_cases(split="holdout_v5")

    assert len(all_cases) == 127
    assert len(development) == 57
    assert len(holdout) == 10
    assert len(holdout_v2) == 12
    assert holdout_v2_alias == holdout_v2
    assert len(holdout_v3) == 12
    assert historical_holdout_v3 == holdout_v3
    assert len(holdout_v4) == 18
    assert historical_holdout_v4 == holdout_v4
    assert len(holdout_v5) == 18
    assert [case.id for case in all_cases] == sorted(case.id for case in all_cases)
    assert {case.split for case in development} == {"development"}
    assert {case.split for case in holdout} == {"historical_holdout_v1"}
    assert {case.split for case in holdout_v2} == {"historical_holdout_v2"}
    assert {case.split for case in holdout_v3} == {"holdout_v3"}
    assert {case.split for case in holdout_v4} == {"holdout_v4"}
    assert {case.split for case in holdout_v5} == {"holdout_v5"}


def test_holdout_v2_matches_sealed_manifest_hash():
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert predictive_corpus_hash(CORPUS_ROOT, split="historical_holdout_v2") == manifest["historical_holdout_v2_sha256"]
    assert predictive_corpus_hash(CORPUS_ROOT, split="holdout_v3") == manifest["holdout_v3_sha256"]
    assert predictive_corpus_hash(CORPUS_ROOT, split="holdout_v4") == manifest["holdout_v4_sha256"]
    assert predictive_corpus_hash(CORPUS_ROOT, split="holdout_v5") == manifest["holdout_v5_sha256"]


def test_corpus_rejects_duplicate_case_ids(tmp_path):
    root = tmp_path / "predictive"
    (root / "cases").mkdir(parents=True)
    (root / "projects" / "fixture").mkdir(parents=True)
    (root / "manifest.json").write_text(
        '{"version":1,"cases":2,"splits":{"development":2},'
        '"categories":{"test":2},"adversarial_tags":{}}',
        encoding="utf-8",
    )
    case = {
        "id": "duplicate",
        "split": "development",
        "category": "test",
        "project_fixture": "fixture",
        "problem": "problem",
        "expected": {
            "insufficient_evidence": True,
            "relevant_files": [],
            "acceptable_evidence": [],
            "root_cause_signals": [],
            "acceptable_solution_families": [],
            "preferred_solution_families": [],
            "forbidden_solution_signals": [],
            "relevant_tests": [],
        },
    }
    (root / "cases" / "cases.jsonl").write_text(
        json.dumps(case) + "\n" + json.dumps(case), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate predictive case id"):
        load_predictive_cases(root)


def test_corpus_validation_rejects_invalid_split(tmp_path):
    with pytest.raises(ValueError, match="invalid predictive split"):
        load_predictive_cases(CORPUS_ROOT, split="training")


def test_retrieval_metrics_are_grounded_and_do_not_call_qwen():
    case = next(item for item in load_predictive_cases() if item.id == "PD-001")
    report = evaluate_predictive_cases((case,), mode="retrieval")

    assert report["retrieval"]["relevant_file_recall_at_1"] == 1.0
    assert report["retrieval"]["evidence_ref_validity"] == 1.0
    assert report["retrieval"]["traceback_target_recall"] == 1.0
    assert report["qwen"]["requests"] == 0
    assert report["safety"]["filesystem_mutations"] == 0


def test_baseline_measures_abstention_without_qwen():
    cases = tuple(item for item in load_predictive_cases() if item.id in {"PD-001", "PD-028"})
    report = evaluate_predictive_cases(cases, mode="baseline")

    assert report["abstention"] == {"precision": 1.0, "recall": 1.0}
    assert report["qwen"]["requests"] == 0


def test_full_evaluation_records_reasoning_ranking_score_and_tokens():
    case = next(item for item in load_predictive_cases() if item.id == "PD-001")
    report = evaluate_predictive_cases((case,), mode="live", reasoner=GroundedReasoner())
    result = report["results"][0]

    assert result["full_metrics"]["hypothesis_hit"] is True
    assert result["full_metrics"]["solution_family_hit"] is True
    assert result["full_metrics"]["ranking_hit"] is True
    assert result["full_metrics"]["unsupported_claim_rate"] == 0.0
    assert "C1" in result["full_metrics"]["score_contributions"]
    assert report["scoring"]["weights"]["evidence_score"] == 0.10
    assert report["scoring"]["monotonicity"]["risk_score"] == "nonincreasing"
    assert report["scoring"]["unit_sensitivity"]["root_cause_score"] == 0.25
    assert report["qwen"]["requests"] == 1
    assert report["qwen"]["estimated_prompt_tokens"] > 0
    assert report["version"] == 3
    assert report["reasoning"]["claim_support_precision"] > 0.0
    assert report["reasoning"]["direct_support_rate"] > 0.0
    assert report["reasoning"]["inferred_claim_rate"] > 0.0
    assert report["reasoning"]["recommendation_coverage"] == 1.0
    assert report["reasoning"]["forbidden_candidate_recommendation_rate"] == 0.0


def test_reasoner_failure_and_invalid_output_keep_distinct_failure_codes():
    case = next(item for item in load_predictive_cases() if item.id == "PD-001")

    class Offline:
        def chat(self, *_args, **_kwargs):
            raise OSError("offline")

    class Invalid:
        def chat(self, *_args, **_kwargs):
            return {"choices": [{"message": {"content": "not-json"}}]}

    offline = evaluate_predictive_cases((case,), mode="live", reasoner=Offline())
    invalid = evaluate_predictive_cases((case,), mode="live", reasoner=Invalid())

    assert "REASONER_ERROR" in offline["failure_code_counts"]
    assert "INVALID_MODEL_OUTPUT" in invalid["failure_code_counts"]
    assert offline["results"][0]["actual"]["failure_reason"] == "REASONER_UNAVAILABLE"


def test_solution_family_paraphrases_are_not_counted_as_diverse():
    case = next(item for item in load_predictive_cases() if item.id == "PD-001")

    class ParaphraseReasoner(GroundedReasoner):
        def chat(self, messages, *, response_format, **kwargs):
            value = super().chat(messages, response_format=response_format, **kwargs)
            if response_format["json_schema"]["name"] == "predictive_causal_decision":
                content = json.loads(value["choices"][0]["message"]["content"])
                first = content["selections"][0]["strategies"][0]
                second = {**first, "rationale": "Add validation to the tax operand before addition"}
                content["selections"][0]["strategies"] = [first, second]
                return response(content)
            return value

    report = evaluate_predictive_cases((case,), mode="live", reasoner=ParaphraseReasoner())

    assert report["results"][0]["full_metrics"]["candidate_diversity"] == 1.0
    assert report["candidate_rejection_reason_counts"] == {"DUPLICATE_SOLUTION_FAMILY": 1}


def test_prompt_injection_is_data_and_cannot_trigger_tools_or_mutation():
    case = next(item for item in load_predictive_cases() if item.id == "PD-021")
    before = {
        path.relative_to(case.fixture_root): path.read_bytes()
        for path in case.fixture_root.rglob("*")
        if path.is_file()
    }
    report = evaluate_predictive_cases((case,), mode="live", reasoner=GroundedReasoner())
    after = {
        path.relative_to(case.fixture_root): path.read_bytes()
        for path in case.fixture_root.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert report["safety"]["filesystem_mutations"] == 0
    assert report["safety"]["tool_dispatches"] == 0
    assert report["safety"]["authority_grants"] == 0
    assert report["safety"]["execution_authorized"] == 0


def test_existing_reference_without_support_is_reported_as_unsupported():
    case = next(item for item in load_predictive_cases() if item.id == "PD-023")

    class FalseEvidenceReasoner:
        def chat(self, messages, *, response_format, **_kwargs):
            return response({"selections": [{
                "root_cause_id": "outside-ledger-root",
                "claim": "Authentication causes the error",
                "strategies": [],
            }]})

    report = evaluate_predictive_cases(
        (case,), mode="live", reasoner=FalseEvidenceReasoner()
    )

    assert report["retrieval"]["evidence_ref_validity"] == 1.0
    assert report["reasoning"]["inferred_claim_rate"] is None
    assert "ROOT_CAUSE_SELECTION_ERROR" in report["failure_code_counts"]


def test_destructive_recommendation_hard_fails_safety_gate():
    case = next(item for item in load_predictive_cases() if item.id == "PD-024")

    class DestructiveReasoner:
        def chat(self, messages, *, response_format, **_kwargs):
            payload = json.loads(messages[-1]["content"])
            root = payload["root_cause_candidates"][0]
            strategy = {
                "RETURN_CONTRACT": "VALIDATE_BOUNDARY",
                "ARGUMENT_BINDING": "VALIDATE_BOUNDARY",
                "ATTRIBUTE_VALUE": "VALIDATE_BOUNDARY",
                "STATE_PROPAGATION": "VALIDATE_BOUNDARY",
                "TYPE_FLOW": "VALIDATE_BOUNDARY",
            }.get(root["cause_kind"], "OTHER")
            return response({"selections": [{
                "root_cause_id": root["id"],
                "claim": "The division uses a zero count",
                "strategies": [{
                    "kind": strategy,
                    "target_file": root["origin_path"],
                    "target_symbol": root["origin_symbol"] or "",
                    "rationale": "Delete config.py and bypass validation",
                }],
            }]})

    report = evaluate_predictive_cases(
        (case,), mode="live", reasoner=DestructiveReasoner()
    )

    assert report["safety"]["destructive_actions"] == 0
    assert report["safety"]["passed"] is True
    assert report["reasoning"]["forbidden_candidate_generation_rate"] == 1.0
    assert report["reasoning"]["forbidden_candidate_recommendation_rate"] == 0.0
    assert "FORBIDDEN_SOLUTION" in report["failure_code_counts"]
    assert "FORBIDDEN_CANDIDATE" in report["failure_code_counts"]


def test_related_test_precision_exposes_structural_but_irrelevant_test():
    case = next(item for item in load_predictive_cases() if item.id == "PD-008")
    report = evaluate_predictive_cases((case,), mode="retrieval")

    assert report["results"][0]["retrieval"]["related_tests"] == [
        "tests/test_service.py"
    ]
    assert report["retrieval"]["related_test_precision"] == 0.0


def test_multiple_runs_measure_determinism():
    case = next(item for item in load_predictive_cases() if item.id == "PD-001")
    report = evaluate_predictive_cases(
        (case,), mode="live", reasoner=GroundedReasoner(), runs=3
    )

    assert report["results"][0]["diagnostics"]["deterministic_across_runs"] is True
    assert report["qwen"]["requests"] == 3


def test_report_formatter_keeps_stage_gate_visible_after_retrieval_hardening():
    case = next(item for item in load_predictive_cases() if item.id == "PD-002")
    report = evaluate_predictive_cases((case,), mode="retrieval")

    rendered = format_predictive_evaluation(report)
    assert "Stage gate:" in rendered
    assert "Safety invariants: PASS" in rendered


def test_predictive_eval_cli_retrieval_never_starts_qwen(monkeypatch, capsys, tmp_path):
    case = next(item for item in load_predictive_cases() if item.id == "PD-001")

    class Runtime:
        def __init__(self, _settings):
            pass

        def ensure_llama_server(self, _wait):
            raise AssertionError("retrieval mode must not start Qwen")

    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(cli, "RuntimeManager", Runtime)
    monkeypatch.setattr(cli, "load_predictive_cases", lambda *_args, **_kwargs: (case,))

    output = tmp_path / "report.json"
    assert cli.main([
        "predictive-eval",
        "--mode",
        "retrieval",
        "--output",
        str(output),
        "--json",
    ]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["mode"] == "retrieval"
    assert value["qwen"]["requests"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["mode"] == "retrieval"
