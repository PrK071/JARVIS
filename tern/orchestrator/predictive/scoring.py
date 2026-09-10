from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .models import SolutionCandidate


SCORE_WEIGHTS = {
    "estimated_success_score": 0.35,
    "evidence_score": 0.20,
    "test_support_score": 0.15,
    "reversibility_score": 0.10,
    "risk_score": -0.15,
    "cost_score": -0.05,
}


def candidate_score(candidate: SolutionCandidate) -> float:
    """Return a normalized, deterministic comparative score."""

    raw = sum(getattr(candidate, name) * weight for name, weight in SCORE_WEIGHTS.items())
    # The specified weighted range is [-0.20, 0.80]. Map it onto [0, 1]
    # without hiding the formula or allowing a model to choose the final score.
    return round(min(1.0, max(0.0, raw + 0.20)), 6)


def explain_score(candidate: SolutionCandidate, score: float) -> str:
    positive = (
        candidate.estimated_success_score * SCORE_WEIGHTS["estimated_success_score"]
        + candidate.evidence_score * SCORE_WEIGHTS["evidence_score"]
        + candidate.test_support_score * SCORE_WEIGHTS["test_support_score"]
        + candidate.reversibility_score * SCORE_WEIGHTS["reversibility_score"]
    )
    penalty = (
        candidate.risk_score * abs(SCORE_WEIGHTS["risk_score"])
        + candidate.cost_score * abs(SCORE_WEIGHTS["cost_score"])
    )
    return (
        f"score={score:.3f}; sinais positivos={positive:.3f}; "
        f"penalidades de risco/custo={penalty:.3f}; normalizacao=+0.200"
    )


def rank_candidates(candidates: Iterable[SolutionCandidate]) -> tuple[SolutionCandidate, ...]:
    scored = []
    for candidate in candidates:
        score = candidate_score(candidate)
        scored.append(
            replace(
                candidate,
                ranking_score=score,
                score_explanation=explain_score(candidate, score),
            )
        )
    return tuple(sorted(scored, key=lambda item: (-item.ranking_score, item.id)))
