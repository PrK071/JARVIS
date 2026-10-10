"""Pure consultative comparison; no execution or patch application capability."""
from dataclasses import dataclass, field

from .models import ObservedCandidateOutcome, OutcomeStatus


@dataclass(frozen=True)
class ObservedComparison:
    prediction_order: tuple[str, ...]
    observed_order: tuple[str, ...]
    recommended_candidate_id: str | None
    explanations: tuple[tuple[str, str], ...]
    execution_authorized: bool = field(default=False, init=False)
    requires_approval: bool = field(default=True, init=False)


def compare_observed(outcomes: tuple[ObservedCandidateOutcome, ...]) -> ObservedComparison:
    if len({item.candidate_id for item in outcomes}) != len(outcomes):
        raise ValueError("duplicate candidate outcomes")
    assessments = []
    for item in outcomes:
        before, after = dict(item.baseline_tests), dict(item.patched_tests)
        if item.infrastructure_failure or item.status is not OutcomeStatus.OBSERVED:
            tier, reason = 0, "NOT_EVALUABLE"
        elif item.timed_out or item.sandbox_violation or not item.validation.passed or not item.validation.syntax_passed:
            tier, reason = 1, "INTEGRITY_OR_SANDBOX_FAILURE"
        elif not before or before.keys() != after.keys():
            tier, reason = 0, "BASELINE_COMPARISON_INCOMPLETE"
        elif item.regressions or any(value != "PASS" for value in after.values()):
            tier, reason = 1, "TEST_OR_REGRESSION_FAILURE"
        elif not any(before[key] == "FAIL" and after[key] == "PASS" for key in before):
            tier, reason = 2, "NO_DEMONSTRATED_IMPROVEMENT"
        else:
            tier, reason = 3, "OBSERVED_TARGETED_IMPROVEMENT_NO_RECORDED_REGRESSION"
        assessments.append((tier, item, reason))
    ordered = sorted(assessments, key=lambda row: (-row[0], row[1].prediction_rank))
    improved = [row for row in ordered if row[0] == 3]
    # Equivalent observed evidence remains ambiguous; prediction alone cannot
    # manufacture an observed winner.
    winner = improved[0][1].candidate_id if len(improved) == 1 else None
    return ObservedComparison(
        tuple(item.candidate_id for item in sorted(outcomes, key=lambda item: item.prediction_rank)),
        tuple(item.candidate_id for _, item, _ in ordered), winner,
        tuple((item.candidate_id, reason) for _, item, reason in ordered),
    )
