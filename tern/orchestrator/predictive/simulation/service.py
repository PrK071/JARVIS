from __future__ import annotations

import time
from pathlib import Path

from .models import ObservedCandidateOutcome, OutcomeStatus, PatchProposal
from .patching import validate_patch


def assess_proposal(proposal: PatchProposal, project: Path) -> ObservedCandidateOutcome:
    """Static-only entry point; runtime promotion is a separate conditional stage."""
    started = time.monotonic()
    validation = validate_patch(proposal, project)
    return ObservedCandidateOutcome(
        proposal.candidate_id, proposal.decision_report_id, proposal.analyzed_commit,
        proposal.prediction_rank, proposal.prediction_score, proposal.patch_sha256,
        validation, OutcomeStatus.NOT_TESTED,
        failure_reason=None if validation.passed else "PATCH_VALIDATION_REJECTED",
        timings=(("patch_validation", time.monotonic() - started),),
    )
