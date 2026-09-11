from __future__ import annotations

import json

from tern.orchestrator.predictive.analysis import PredictiveAnalyzer
from tern.orchestrator.predictive.causal import (
    CausalNode,
    CausalNodeKind,
    CausalSlice,
    RootCauseCandidate,
    RootCauseKind,
)
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
    origin = CausalNode("N1", CausalNodeKind.RETURN_VALUE, "pkg/calc.py", 2, "add", "return a + b", ("E1",))
    failure = CausalNode("N2", CausalNodeKind.EXCEPTION_SITE, "pkg/calc.py", 2, None, "exception", ("E1",))
    root = RootCauseCandidate(
        "R1", RootCauseKind.RETURN_CONTRACT, "pkg/calc.py", "add", 2,
        "pkg/calc.py", 2, ("N1", "N2"), ("E1",), 1.0, 1.0, 1,
        1.0, 0.9, "return a + b at pkg/calc.py:2 flows to the failure site",
    )
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
        causal_slice=CausalSlice("N2", (origin, failure), ()),
        root_cause_candidates=(root,),
    )


def valid_responses():
    return [
        response(
            {
                "selections": [
                    {
                        "root_cause_id": "R1",
                        "claim": "one operand may be None",
                        "strategies": [
                            {"kind": "VALIDATE_BOUNDARY", "target_file": "pkg/calc.py", "target_symbol": "add", "rationale": "Validate operands at the add boundary"},
                            {"kind": "VALIDATE_BOUNDARY", "target_file": "pkg/calc.py", "target_symbol": "add", "rationale": "Check operands before addition"},
                        ],
                    },
                ],
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
                {"selections": [{"root_cause_id": "missing", "claim": "Invented", "strategies": []}]}
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
    payload["selections"][0]["claim"] = "add receives None from its caller"

    result = PredictiveAnalyzer(
        FakeReasoner([response(payload)])
    ).analyze_with_diagnostics(context())

    assert result.hypotheses[0].statement.endswith("Assessment: add receives None from its caller")
    assert result.hypotheses[0].claims[-1].support.value == "INFERRED"
    assert result.hypotheses[0].claims[-1].evidence_ids == ("E1",)
    assert result.hypotheses[0].confidence == 0.9


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
        FakeReasoner([response({"selections": "not-a-list"})])
    ).analyze_with_diagnostics(context())

    assert result.failure_reason is PredictiveFailureReason.INVALID_STRUCTURED_RESPONSE


def test_near_duplicate_candidates_are_collapsed_with_diagnostic():
    values = valid_responses()
    candidate_payload = json.loads(values[0]["choices"][0]["message"]["content"])
    candidate_payload["selections"][0]["strategies"][0]["rationale"] = "Validate operands before adding tax"
    candidate_payload["selections"][0]["strategies"][1]["rationale"] = "Validate the operands before adding tax"
    values[0] = response(candidate_payload)

    result = PredictiveAnalyzer(FakeReasoner(values)).analyze_with_diagnostics(context())

    assert len(result.candidates) == 1
    assert result.diagnostic_codes == ("DUPLICATE_SOLUTION_FAMILY",)
