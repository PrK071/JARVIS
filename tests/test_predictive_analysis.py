from __future__ import annotations

import json

from tern.orchestrator.predictive.analysis import PredictiveAnalyzer
from tern.orchestrator.predictive.models import (
    EvidenceAtom,
    EvidenceExcerpt,
    EvidenceKind,
    EvidenceLedger,
    PredictiveFailureReason,
    ProblemContext,
)


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
        test_relationships=(("pkg/calc.py", "tests/test_calc.py"),),
        evidence_ledger=EvidenceLedger((
            EvidenceAtom("E1", EvidenceKind.RETURN_STATEMENT, "pkg/calc.py", 2, 2, "add returns a + b", 1.0, symbol="add", relation="returns", snippet="return a + b"),
            EvidenceAtom("E2", EvidenceKind.TEST_RELATION, "tests/test_calc.py", 1, 3, "tests/test_calc.py is structurally related to pkg/calc.py", 0.7, relation="pkg/calc.py"),
            EvidenceAtom("E3", EvidenceKind.STRUCTURAL_RELATION, "tests/test_calc.py", 2, 2, "test_add asserts behavior involving add", 0.9, relation="add"),
        )),
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
                        "claims": [
                            {"statement": "add returns a + b", "claim_type": "FACT", "evidence_ids": ["E1"]},
                            {"statement": "one operand may be None", "claim_type": "INFERENCE", "evidence_ids": ["E1"]}
                        ],
                    }
                ],
                "candidates": [
                    {
                        "id": "C1",
                        "hypothesis_id": "H1",
                        "action": "Validate operands at the add boundary",
                        "expected_outcome": "Reject None before arithmetic",
                        "change_kind": "VALIDATION",
                        "target_files": ["pkg/calc.py"],
                        "target_symbols": ["add"],
                        "mechanism": "GUARD_CLAUSE",
                        "evidence_ids": ["E1"],
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
                        "change_kind": "VALIDATION",
                        "target_files": ["pkg/calc.py"],
                        "target_symbols": ["add"],
                        "mechanism": "GUARD_CLAUSE",
                        "evidence_ids": ["E1"],
                        "required_tests": ["tests/test_calc.py"],
                        "risk_level": "LOW",
                        "cost_level": "LOW",
                        "reversibility_level": "HIGH",
                    },
                ]
            }
        )
    ]


def test_traceback_evidence_generates_grounded_hypothesis_and_distinct_candidate():
    reasoner = FakeReasoner(valid_responses())
    hypotheses, candidates = PredictiveAnalyzer(reasoner).analyze(context())

    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_refs == ("pkg/calc.py:2-2",)
    assert len(candidates) == 1
    assert candidates[0].required_tests == ("tests/test_calc.py",)
    assert candidates[0].test_support_score == 1.0
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
                            "claims": [{"statement": "Invented", "claim_type": "FACT", "evidence_ids": ["missing"]}],
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


def test_unlabelled_causal_summary_is_demoted_to_linked_inference():
    values = valid_responses()
    payload = json.loads(values[0]["choices"][0]["message"]["content"])
    payload["hypotheses"][0]["claims"] = payload["hypotheses"][0]["claims"][:1]

    result = PredictiveAnalyzer(
        FakeReasoner([response(payload)])
    ).analyze_with_diagnostics(context())

    assert result.hypotheses[0].statement == "add receives None from its caller"
    assert result.hypotheses[0].claims[-1].support.value == "INFERRED"
    assert result.hypotheses[0].claims[-1].evidence_ids == ("E1",)
    assert result.hypotheses[0].confidence == 0.65
    assert "INFERENCE_NORMALIZED" in result.diagnostic_codes


def test_same_structured_input_is_reproducible():
    first = PredictiveAnalyzer(FakeReasoner(valid_responses())).analyze(context())
    second = PredictiveAnalyzer(FakeReasoner(valid_responses())).analyze(context())

    assert first == second


def test_reasoner_failure_is_not_reported_as_missing_code_evidence():
    class Unavailable:
        def chat(self, *_args, **_kwargs):
            raise OSError("offline")

    result = PredictiveAnalyzer(Unavailable()).analyze_with_diagnostics(context())

    assert result.failure_reason is PredictiveFailureReason.REASONER_UNAVAILABLE


def test_invalid_json_has_distinct_failure_reason():
    result = PredictiveAnalyzer(
        FakeReasoner([{"choices": [{"message": {"content": "not-json"}}]}])
    ).analyze_with_diagnostics(context())

    assert result.failure_reason is PredictiveFailureReason.INVALID_STRUCTURED_RESPONSE


def test_invalid_schema_shape_has_distinct_failure_reason():
    result = PredictiveAnalyzer(
        FakeReasoner([response({"hypotheses": "not-a-list"})])
    ).analyze_with_diagnostics(context())

    assert result.failure_reason is PredictiveFailureReason.INVALID_STRUCTURED_RESPONSE


def test_near_duplicate_candidates_are_collapsed_with_diagnostic():
    values = valid_responses()
    candidate_payload = json.loads(values[0]["choices"][0]["message"]["content"])
    candidate_payload["candidates"][0]["action"] = "Validate operands before adding tax"
    candidate_payload["candidates"][1]["action"] = "Validate the operands before adding tax"
    values[0] = response(candidate_payload)

    result = PredictiveAnalyzer(FakeReasoner(values)).analyze_with_diagnostics(context())

    assert len(result.candidates) == 1
    assert result.diagnostic_codes == ("DUPLICATE_SOLUTION_FAMILY",)
