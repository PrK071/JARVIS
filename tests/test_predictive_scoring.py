from __future__ import annotations

from tern.orchestrator.predictive.models import SolutionCandidate
from tern.orchestrator.predictive.scoring import candidate_score, rank_candidates


def candidate(identifier: str, **overrides) -> SolutionCandidate:
    values = {
        "id": identifier,
        "hypothesis_id": "H1",
        "action": f"Action {identifier}",
        "expected_outcome": "Resolved",
        "evidence_refs": ("pkg/calc.py:1-4",),
        "required_tests": (),
        "evidence_score": 0.5,
        "risk_score": 0.5,
        "cost_score": 0.5,
        "reversibility_score": 0.5,
        "estimated_success_score": 0.5,
        "confidence": 0.5,
        "test_support_score": 0.5,
    }
    values.update(overrides)
    return SolutionCandidate(**values)


def test_strong_low_risk_candidate_beats_weak_high_risk_candidate():
    strong = candidate(
        "C2",
        evidence_score=1.0,
        risk_score=0.1,
        cost_score=0.2,
        reversibility_score=0.9,
        estimated_success_score=0.9,
        test_support_score=1.0,
    )
    weak = candidate(
        "C1",
        evidence_score=0.2,
        risk_score=0.9,
        cost_score=0.8,
        reversibility_score=0.2,
        estimated_success_score=0.3,
        test_support_score=0.0,
    )

    ranked = rank_candidates((weak, strong))

    assert [item.id for item in ranked] == ["C2", "C1"]
    assert ranked[0].score_explanation.startswith("score=")
    assert 0.0 <= candidate_score(strong) <= 1.0


def test_ranking_is_deterministic_and_uses_id_as_stable_tie_breaker():
    values = (candidate("C2"), candidate("C1"))

    assert rank_candidates(values) == rank_candidates(values)
    assert [item.id for item in rank_candidates(values)] == ["C1", "C2"]
