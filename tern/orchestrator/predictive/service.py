from __future__ import annotations

import re
from pathlib import Path

from ..project_intelligence_v2 import (
    EvidenceStrength,
    ProjectCandidateGenerator,
    ProjectCandidateSelection,
    ProjectIndexBuilderV2,
    ProjectSnapshotV2,
    RelevantFileCandidate,
)
from ..security import PathPolicy
from .analysis import PredictiveAnalyzer, StructuredReasoner
from .models import DecisionReport, EvidenceExcerpt, ProblemContext
from .scoring import rank_candidates


_STRENGTH_ORDER = {
    EvidenceStrength.SEMANTIC: 1,
    EvidenceStrength.SUPPORTING: 2,
    EvidenceStrength.STRONG: 3,
    EvidenceStrength.HARD: 4,
}


def _line_target(candidate: RelevantFileCandidate) -> int | None:
    prefix = re.escape(candidate.path)
    for evidence in candidate.evidences:
        match = re.fullmatch(rf"{prefix}:(\d+)", evidence.target)
        if match:
            return int(match.group(1))
    return None


def build_problem_context(
    problem: str,
    snapshot: ProjectSnapshotV2,
    selection: ProjectCandidateSelection,
    path_policy: PathPolicy,
) -> ProblemContext:
    slices = {item.path: item for item in selection.context_slices}
    excerpts: list[EvidenceExcerpt] = []
    related_symbols: list[str] = []
    selected_paths = {item.path for item in selection.selected}
    root = Path(snapshot.project_path)

    for candidate in selection.selected:
        selected_slice = slices.get(candidate.path)
        path = path_policy.resolve(str(root / candidate.path))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            continue
        target_line = _line_target(candidate)
        if target_line is not None:
            start = max(1, target_line - 12)
            end = min(len(lines), target_line + 12)
        elif selected_slice is not None:
            start = max(1, selected_slice.start_line)
            end = min(len(lines), max(start, selected_slice.end_line))
        else:
            start, end = 1, min(len(lines), 80)
        if end < start:
            continue
        excerpt = "\n".join(
            f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
        )
        strongest = max(candidate.evidences, key=lambda item: _STRENGTH_ORDER[item.strength])
        excerpts.append(
            EvidenceExcerpt(
                ref=f"{candidate.path}:{start}-{end}",
                path=candidate.path,
                start_line=start,
                end_line=end,
                strength=strongest.strength.value,
                sources=tuple(item.source.value for item in candidate.evidences),
                excerpt=excerpt,
            )
        )
        indexed = snapshot.file_index[candidate.path]
        related_symbols.extend(
            f"{symbol.qualified_name}@{symbol.file}:{symbol.line}-{symbol.end_line}"
            for symbol in indexed.symbols
            if start <= symbol.line <= end or start <= symbol.end_line <= end
        )

    related_tests = sorted(
        {
            relation.test_file
            for relation in snapshot.test_relationships
            if relation.production_file in selected_paths or relation.test_file in selected_paths
        }
        | {item.path for item in selection.selected if item.is_test}
    )
    return ProblemContext(
        problem=problem,
        project_path=snapshot.project_path,
        project_id=snapshot.project_id,
        evidence=tuple(excerpts),
        related_files=tuple(item.path for item in selection.selected),
        related_symbols=tuple(related_symbols),
        related_tests=tuple(related_tests),
    )


class PredictiveDecisionService:
    """Read-only reasoning service. It owns no tools or execution authority."""

    def __init__(
        self,
        reasoner: StructuredReasoner,
        *,
        path_policy: PathPolicy,
        analyzer: PredictiveAnalyzer | None = None,
        candidate_generator: ProjectCandidateGenerator | None = None,
    ) -> None:
        self.path_policy = path_policy
        self.analyzer = analyzer or PredictiveAnalyzer(reasoner)
        self.candidate_generator = candidate_generator or ProjectCandidateGenerator()

    def predict(self, problem: str, project_path: str | Path) -> DecisionReport:
        if not problem.strip():
            raise ValueError("problem must not be empty")
        root = self.path_policy.resolve(str(project_path))
        snapshot = ProjectIndexBuilderV2(root, path_policy=self.path_policy).build()
        selection = self.candidate_generator.generate(problem, snapshot)
        context = build_problem_context(problem, snapshot, selection, self.path_policy)
        if not context.sufficient_evidence:
            return self._insufficient(problem, "Nenhuma evidência estrutural forte foi localizada.")

        hypotheses, candidates = self.analyzer.analyze(context)
        if not hypotheses:
            return self._insufficient(problem, "Não foi possível sustentar uma hipótese nas evidências disponíveis.")
        if not candidates:
            return DecisionReport(
                problem=problem,
                hypotheses=hypotheses,
                candidates=(),
                recommended_candidate_id=None,
                insufficient_evidence=True,
                recommendation_explanation=(
                    "As hipóteses possuem evidência, mas nenhuma solução candidata válida e distinta foi produzida."
                ),
            )

        ranked = rank_candidates(candidates)
        winner = ranked[0]
        explanation = (
            f"{winner.id} venceu pelo maior score determinístico ({winner.ranking_score:.3f}). "
            f"{winner.score_explanation} A recomendação é consultiva e exige aprovação."
        )
        return DecisionReport(
            problem=problem,
            hypotheses=hypotheses,
            candidates=ranked,
            recommended_candidate_id=winner.id,
            insufficient_evidence=False,
            recommendation_explanation=explanation,
        )

    @staticmethod
    def _insufficient(problem: str, explanation: str) -> DecisionReport:
        return DecisionReport(
            problem=problem,
            hypotheses=(),
            candidates=(),
            recommended_candidate_id=None,
            insufficient_evidence=True,
            recommendation_explanation=explanation,
        )
