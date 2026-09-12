from __future__ import annotations

from tern.orchestrator.predictive.benchmark_v6 import summarize_benchmark_v6


def test_v6_metrics_expose_population_and_denominators():
    positive = {
        "benchmark_v5": {"evaluation_status": "CANONICAL", "abstention_expected": False},
        "metrics_v5": {
            "evaluable": True, "root_cause_validity": False,
            "repair_strategy_validity": False, "repair_target_validity": False,
            "repair_pair_validity": False, "top1_validity": None,
            "recommendation_coverage": False, "false_abstention": True,
            "recommendation_valid": None,
        },
        "actual": {"insufficient_evidence": True, "root_cause_candidates": []},
        "retrieval": {"failure_codes": []},
    }
    recommended = {
        "benchmark_v5": {"evaluation_status": "CANONICAL", "abstention_expected": False},
        "metrics_v5": {
            "evaluable": True, "root_cause_validity": True,
            "repair_strategy_validity": True, "repair_target_validity": True,
            "repair_pair_validity": True, "top1_validity": True,
            "recommendation_coverage": True, "false_abstention": False,
            "recommendation_valid": True,
        },
        "actual": {
            "insufficient_evidence": False, "recommended_candidate_id": "C1",
            "root_cause_candidates": [{"id": "R1"}], "hypotheses": [{"id": "H1"}],
            "candidates": [{"id": "C1"}],
        },
        "retrieval": {"failure_codes": []},
    }
    report = summarize_benchmark_v6({
        "version": 5, "results": [positive, recommended], "quality": {},
    })

    assert report["metric_denominators"]["root_cause_validity"] == {
        "value": 0.5, "numerator": 1, "denominator": 2,
    }
    assert report["metric_denominators"]["top1_validity"] == {
        "value": 1.0, "numerator": 1, "denominator": 1,
    }
    assert report["population"]["n_positive"] == 2
    assert report["population"]["n_recommended"] == 1
    assert report["population"]["n_abstained"] == 1
    assert report["metric_denominators"]["false_abstention_rate"] == {
        "value": 0.5, "numerator": 1, "denominator": 2,
    }
    assert report["coverage_funnel"]["scope"] == "all benchmark cases"
