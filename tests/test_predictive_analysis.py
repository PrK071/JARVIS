from __future__ import annotations

import json

from tern.orchestrator.predictive.analysis import PredictiveAnalyzer
from tern.orchestrator.predictive.models import EvidenceExcerpt, ProblemContext


def response(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}]}


class FakeReasoner:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.values.pop(0)


def context() -> ProblemContext:
    return ProblemContext(
        problem='File "pkg/calc.py", line 2\nTypeError: unsupported operand type(s)',
        project_path="/repo",
        project_id="repo",
        evidence=(
            EvidenceExcerpt(
                "pkg/calc.py:1-4",
                "pkg/calc.py",
                1,
                4,
                "HARD",
                ("TRACEBACK_REFERENCE",),
                "1: def add(a, b):\n2:     return a + b",
            ),
            EvidenceExcerpt(
                "tests/test_calc.py:1-3",
                "tests/test_calc.py",
                1,
                3,
                "STRONG",
                ("TEST_RELATIONSHIP",),
                "1: def test_add():\n2:     assert add(1, 2) == 3",
            ),
        ),
        related_files=("pkg/calc.py", "tests/test_calc.py"),
        related_symbols=("add@pkg/calc.py:1-2",),
        related_tests=("tests/test_calc.py",),
    )


def valid_responses():
    return [
        response(
            {
                "hypotheses": [
                    {
                        "id": "H1",
                        "statement": "add receives None from its caller",
                        "confidence_level": "HIGH",
                        "evidence_refs": ["pkg/calc.py:1-4"],
                    }
                ]
            }
        ),
        response(
            {
                "candidates": [
                    {
                        "id": "C1",
                        "hypothesis_id": "H1",
                        "action": "Validate operands at the add boundary",
                        "expected_outcome": "Reject None before arithmetic",
                        "evidence_refs": ["pkg/calc.py:1-4"],
                        "required_tests": ["tests/test_calc.py"],
                        "risk_level": "LOW",
                        "cost_level": "LOW",
                        "reversibility_level": "HIGH",
                    },
                    {
                        "id": "C2",
                        "hypothesis_id": "H1",
                        "action": "Validate operands at the add boundary",
                        "expected_outcome": "Same wording must be filtered",
                        "evidence_refs": ["pkg/calc.py:1-4"],
                        "required_tests": ["tests/test_calc.py"],
                        "risk_level": "LOW",
                        "cost_level": "LOW",
                        "reversibility_level": "HIGH",
                    },
                ]
            }
        ),
    ]


def test_traceback_evidence_generates_grounded_hypothesis_and_distinct_candidate():
    reasoner = FakeReasoner(valid_responses())
    hypotheses, candidates = PredictiveAnalyzer(reasoner).analyze(context())

    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_refs == ("pkg/calc.py:1-4",)
    assert len(candidates) == 1
    assert candidates[0].required_tests == ("tests/test_calc.py",)
    assert all(0.0 <= value <= 1.0 for value in (
        candidates[0].evidence_score,
        candidates[0].risk_score,
        candidates[0].cost_score,
        candidates[0].reversibility_score,
        candidates[0].estimated_success_score,
        candidates[0].confidence,
    ))
    assert all(call[1]["temperature"] == 0.0 for call in reasoner.calls)


def test_analysis_rejects_invented_evidence_and_does_not_generate_candidates():
    reasoner = FakeReasoner(
        [
            response(
                {
                    "hypotheses": [
                        {
                            "id": "H1",
                            "statement": "Invented cause",
                            "confidence_level": "HIGH",
                            "evidence_refs": ["missing.py:1-2"],
                        }
                    ]
                }
            )
        ]
    )

    hypotheses, candidates = PredictiveAnalyzer(reasoner).analyze(context())

    assert hypotheses == ()
    assert candidates == ()
    assert len(reasoner.calls) == 1


def test_same_structured_input_is_reproducible():
    first = PredictiveAnalyzer(FakeReasoner(valid_responses())).analyze(context())
    second = PredictiveAnalyzer(FakeReasoner(valid_responses())).analyze(context())

    assert first == second
