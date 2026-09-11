from __future__ import annotations

from tern.orchestrator.predictive.models import SolutionCandidate
from tern.orchestrator.predictive.scoring import candidate_score, rank_candidates, ranking_decision


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


def test_ranking_order_is_deterministic_but_tie_has_no_winner():
    values = (candidate("C2"), candidate("C1"))

    assert rank_candidates(values) == rank_candidates(values)
    assert [item.id for item in rank_candidates(values)] == ["C1", "C2"]
    assert ranking_decision(values).winner_id is None


def test_equal_scores_are_reported_as_ambiguous_instead_of_a_winner():
    decision = ranking_decision((candidate("C2"), candidate("C1")))
    assert decision.margin == 0.0
    assert decision.ambiguous is True
    assert decision.winner_id is None


def test_ineligible_candidate_never_reaches_ranking():
    rejected = candidate("C1", eligible=False, rejection_reasons=("FORBIDDEN_CANDIDATE",))
    accepted = candidate("C2", evidence_score=0.4)
    decision = ranking_decision((rejected, accepted))
    assert [item.id for item in decision.candidates] == ["C2"]
    assert decision.winner_id == "C2"


def test_root_cause_quality_and_repair_locality_contribute_independently():
    at_origin = candidate(
        "C1", root_cause_score=0.9, repair_locality_score=1.0,
        evidence_score=0.8,
        repair_strategy_score=1.0,
    )
    symptom_mask = candidate(
        "C2", root_cause_score=0.4, repair_locality_score=0.5,
        evidence_score=0.8,
        repair_strategy_score=0.5,
    )

    decision = ranking_decision((symptom_mask, at_origin))

    assert decision.winner_id == "C1"
    assert "causa=+0.225" in decision.candidates[0].score_explanation
    assert "estrategia=+0.200" in decision.candidates[0].score_explanation
    assert "localidade=+0.150" in decision.candidates[0].score_explanation
