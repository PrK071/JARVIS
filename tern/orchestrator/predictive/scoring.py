from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .models import SolutionCandidate


SCORE_WEIGHTS = {
    "root_cause_score": 0.25,
    "repair_strategy_score": 0.20,
    "evidence_score": 0.10,
    "test_support_score": 0.10,
    "repair_locality_score": 0.15,
    "reversibility_score": 0.10,
    "risk_score": -0.07,
    "cost_score": -0.03,
}


def candidate_score(candidate: SolutionCandidate) -> float:
    """Return a normalized, deterministic comparative score."""

    raw = sum(getattr(candidate, name) * weight for name, weight in SCORE_WEIGHTS.items())
    # The causal weighted range is [-0.10, 0.90]. Map it onto [0, 1]
    # without hiding the formula or allowing a model to choose the final score.
    return round(min(1.0, max(0.0, raw + 0.10)), 6)


def explain_score(candidate: SolutionCandidate, score: float) -> str:
    contributions = {
        name: getattr(candidate, name) * weight for name, weight in SCORE_WEIGHTS.items()
    }
    return (
        f"score={score:.3f}; causa={contributions['root_cause_score']:+.3f}; "
        f"estrategia={contributions['repair_strategy_score']:+.3f}; "
        f"evidencia={contributions['evidence_score']:+.3f}; "
        f"testes={contributions['test_support_score']:+.3f}; "
        f"localidade={contributions['repair_locality_score']:+.3f}; "
        f"reversibilidade={contributions['reversibility_score']:+.3f}; "
        f"risco={contributions['risk_score']:+.3f}; "
        f"custo={contributions['cost_score']:+.3f}; normalizacao=+0.100"
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


@dataclass(frozen=True)
class RankingDecision:
    candidates: tuple[SolutionCandidate, ...]
    winner_id: str | None
    margin: float | None
    ambiguous: bool


def ranking_decision(candidates: Iterable[SolutionCandidate]) -> RankingDecision:
    ranked = rank_candidates(item for item in candidates if item.eligible)
    if not ranked:
        return RankingDecision((), None, None, False)
    if len(ranked) == 1:
        return RankingDecision(ranked, ranked[0].id, None, False)
    margin = round(ranked[0].ranking_score - ranked[1].ranking_score, 6)
    # Development v1 showed no monotonic relation between arbitrary small margins
    # and correctness. Only a real score tie is treated as indistinguishable.
    ambiguous = margin == 0.0
    return RankingDecision(ranked, None if ambiguous else ranked[0].id, margin, ambiguous)
