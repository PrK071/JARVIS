from __future__ import annotations

from tern.orchestrator.predictive.models import ChangeKind, SolutionCandidate
from tern.orchestrator.predictive.policy import PredictiveCandidatePolicy


def candidate(action: str) -> SolutionCandidate:
    return SolutionCandidate(
        "C1", "H1", action, "Resolve the failure", ("pkg/a.py:1-1",), (),
        0.8, 0.2, 0.2, 0.9, 0.8, 0.8,
        change_kind=ChangeKind.VALIDATION,
        target_files=("pkg/a.py",),
        mechanism="GUARD_CLAUSE",
        evidence_ids=("E1",),
    )


def test_candidate_policy_rejects_generic_dangerous_repairs():
    policy = PredictiveCandidatePolicy()
    for action in (
        "Delete config.py to resolve it",
        "Disable the tests",
        "Remove validation and bypass the guard",
        "Catch all errors and swallow the exception",
    ):
        value = policy.apply(candidate(action))
        assert value.eligible is False
        assert value.rejection_reasons == ("FORBIDDEN_CANDIDATE",)


def test_candidate_policy_preserves_local_safe_repairs():
    value = PredictiveCandidatePolicy().apply(candidate("Validate the input before addition"))
    assert value.eligible is True
    assert value.rejection_reasons == ()


def test_solution_signature_collapses_paraphrases_but_preserves_distinct_mechanisms():
    left = candidate("Validate before addition")
    paraphrase = candidate("Check the value before adding")
    upstream = SolutionCandidate(**{
        **left.__dict__,
        "id": "C2",
        "change_kind": ChangeKind.UPSTREAM_FIX,
        "mechanism": "FIX_PRODUCER",
    })
    assert left.solution_signature == paraphrase.solution_signature
    assert left.solution_signature != upstream.solution_signature
