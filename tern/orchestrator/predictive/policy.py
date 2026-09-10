from __future__ import annotations

import re
from dataclasses import replace

from .models import SolutionCandidate


class PredictiveCandidatePolicy:
    """Read-only eligibility policy; it owns no execution or tool capability."""

    _FORBIDDEN = (
        re.compile(r"\b(?:delete|remove)\s+(?:the\s+)?(?:file|config(?:uration)?|database|code)\b", re.I),
        re.compile(r"\b(?:delete|remove|skip|disable)\s+(?:the\s+)?(?:test|tests|assertion|assertions)\b", re.I),
        re.compile(r"\b(?:disable|remove|weaken|bypass)\s+(?:the\s+)?(?:validation|security|guard|check)\b", re.I),
        re.compile(r"\b(?:except\s+Exception|broad\s+except|catch\s+all).*(?:pass|ignore|swallow)", re.I),
        re.compile(r"\b(?:ignore|swallow|silence|hide)\s+(?:the\s+)?(?:exception|error|failure)\b", re.I),
        re.compile(r"\brm\s+-rf\b", re.I),
        re.compile(r"\bdrop\s+(?:table|database)\b", re.I),
    )

    def rejection_reasons(self, candidate: SolutionCandidate) -> tuple[str, ...]:
        text = f"{candidate.action} {candidate.expected_outcome} {candidate.mechanism}"
        return ("FORBIDDEN_CANDIDATE",) if any(rule.search(text) for rule in self._FORBIDDEN) else ()

    def apply(self, candidate: SolutionCandidate) -> SolutionCandidate:
        reasons = tuple(dict.fromkeys((*candidate.rejection_reasons, *self.rejection_reasons(candidate))))
        return replace(candidate, eligible=not reasons, rejection_reasons=reasons)
