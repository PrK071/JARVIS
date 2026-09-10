from __future__ import annotations

import dataclasses

import pytest

from tern.orchestrator.predictive.models import (
    DecisionReport,
    Hypothesis,
    SolutionCandidate,
)


def candidate(**overrides) -> SolutionCandidate:
    values = {
        "id": "C1",
        "hypothesis_id": "H1",
        "action": "Validate the operand before addition",
        "expected_outcome": "None never reaches the addition",
        "evidence_refs": ("pkg/calc.py:1-8",),
        "required_tests": ("tests/test_calc.py",),
        "evidence_score": 0.9,
        "risk_score": 0.2,
        "cost_score": 0.2,
        "reversibility_score": 0.9,
        "estimated_success_score": 0.8,
        "confidence": 0.8,
    }
    values.update(overrides)
    return SolutionCandidate(**values)


@pytest.mark.parametrize(
    "field",
    [
        "evidence_score",
        "risk_score",
        "cost_score",
        "reversibility_score",
        "estimated_success_score",
        "confidence",
        "test_support_score",
        "ranking_score",
    ],
)
def test_predictive_scores_are_bounded(field):
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        candidate(**{field: 1.01})


def test_hypothesis_requires_concrete_evidence_reference():
    with pytest.raises(ValueError, match="concrete evidence"):
        Hypothesis("H1", "A cause", 0.5, ())


def test_report_cannot_grant_itself_execution_authority():
    hypothesis = Hypothesis("H1", "None reaches the operation", 0.8, ("pkg/calc.py:1-8",))
    report = DecisionReport(
        problem="TypeError",
        hypotheses=(hypothesis,),
        candidates=(candidate(),),
        recommended_candidate_id="C1",
        insufficient_evidence=False,
        recommendation_explanation="C1 has the strongest evidence.",
    )

    assert report.requires_approval is True
    assert report.execution_authorized is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.execution_authorized = True
    with pytest.raises(TypeError):
        DecisionReport(
            problem="TypeError",
            hypotheses=(),
            candidates=(),
            recommended_candidate_id=None,
            insufficient_evidence=True,
            recommendation_explanation="No evidence.",
            execution_authorized=True,
        )


def test_report_limits_hypotheses_and_candidates_to_three():
    hypotheses = tuple(
        Hypothesis(f"H{index}", f"Cause {index}", 0.5, (f"pkg/m{index}.py:1-2",))
        for index in range(1, 5)
    )
    with pytest.raises(ValueError, match="limited to three"):
        DecisionReport("problem", hypotheses, (), None, True, "Too many hypotheses.")
