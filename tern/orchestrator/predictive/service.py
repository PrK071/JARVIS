from __future__ import annotations

import re
from dataclasses import replace
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
from .causal import build_causal_slice, expand_context_for_causal_flow
from .grounding import build_evidence_ledger
from .models import (
    DecisionReport,
    EvidenceExcerpt,
    PredictiveFailureReason,
    ProblemContext,
)
from .policy import PredictiveCandidatePolicy
from .scoring import ranking_decision


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
    test_relationships = tuple(
        sorted(
            (relation.production_file, relation.test_file)
            for relation in snapshot.test_relationships
            if relation.production_file in selected_paths or relation.test_file in selected_paths
        )
    )
    return ProblemContext(
        problem=problem,
        project_path=snapshot.project_path,
        project_id=snapshot.project_id,
        evidence=tuple(excerpts),
        related_files=tuple(item.path for item in selection.selected),
        related_symbols=tuple(related_symbols),
        related_tests=tuple(related_tests),
        test_relationships=test_relationships,
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
        candidate_policy: PredictiveCandidatePolicy | None = None,
    ) -> None:
        self.path_policy = path_policy
        self.analyzer = analyzer or PredictiveAnalyzer(reasoner)
        self.candidate_generator = candidate_generator or ProjectCandidateGenerator()
        self.candidate_policy = candidate_policy or PredictiveCandidatePolicy()

    def predict(self, problem: str, project_path: str | Path) -> DecisionReport:
        if not problem.strip():
            raise ValueError("problem must not be empty")
        root = self.path_policy.resolve(str(project_path))
        snapshot = ProjectIndexBuilderV2(root, path_policy=self.path_policy).build()
        selection = self.candidate_generator.generate(problem, snapshot)
        context = build_problem_context(problem, snapshot, selection, self.path_policy)
        context = expand_context_for_causal_flow(context, snapshot, self.path_policy)
        context = replace(
            context,
            evidence_ledger=build_evidence_ledger(context, snapshot, self.path_policy),
        )
        causal_slice, root_causes = build_causal_slice(
            context, snapshot, self.path_policy
        )
        context = replace(
            context,
            causal_slice=causal_slice,
            root_cause_candidates=root_causes,
        )
        if not context.sufficient_evidence:
            return self._insufficient(problem, "Nenhuma evidência estrutural forte foi localizada.")

        analysis = self.analyzer.analyze_with_diagnostics(context)
        if analysis.failure_reason is not None:
            return DecisionReport(
                problem=problem,
                hypotheses=analysis.hypotheses,
                candidates=analysis.candidates,
                recommended_candidate_id=None,
                insufficient_evidence=True,
                recommendation_explanation=self._failure_message(analysis.failure_reason),
                failure_reason=analysis.failure_reason,
                diagnostic_codes=analysis.diagnostic_codes,
                rejected_candidates=analysis.rejected_candidates,
                causal_slice=context.causal_slice,
                root_cause_candidates=context.root_cause_candidates,
                repair_strategies=analysis.repair_strategies,
            )

        policy_candidates = tuple(self.candidate_policy.apply(item) for item in analysis.candidates)
        rejected = tuple(
            (*analysis.rejected_candidates, *(item for item in policy_candidates if not item.eligible))
        )
        eligible = tuple(item for item in policy_candidates if item.eligible)
        if not eligible:
            return DecisionReport(
                problem=problem,
                hypotheses=analysis.hypotheses,
                candidates=(),
                recommended_candidate_id=None,
                insufficient_evidence=False,
                recommendation_explanation="Nenhuma solução candidata segura e grounded permaneceu elegível.",
                failure_reason=(
                    PredictiveFailureReason.FORBIDDEN_CANDIDATE
                    if any("FORBIDDEN_CANDIDATE" in item.rejection_reasons for item in rejected)
                    else PredictiveFailureReason.NO_ELIGIBLE_CANDIDATE
                ),
                diagnostic_codes=tuple(dict.fromkeys(
                    (*analysis.diagnostic_codes, *(reason for item in rejected for reason in item.rejection_reasons))
                )),
                rejected_candidates=rejected,
                causal_slice=context.causal_slice,
                root_cause_candidates=context.root_cause_candidates,
                repair_strategies=analysis.repair_strategies,
            )
        try:
            decision = ranking_decision(eligible)
        except Exception:
            return self._insufficient(
                problem,
                "Não foi possível ordenar as soluções candidatas.",
                PredictiveFailureReason.RANKING_FAILURE,
            )
        if not decision.candidates:
            return self._insufficient(
                problem,
                "Nenhuma solução candidata válida foi produzida.",
                PredictiveFailureReason.NO_VALID_CANDIDATE,
            )
        if decision.ambiguous:
            return DecisionReport(
                problem=problem,
                hypotheses=analysis.hypotheses,
                candidates=decision.candidates,
                recommended_candidate_id=None,
                insufficient_evidence=False,
                recommendation_explanation=(
                    f"Os candidatos mais fortes são indistinguíveis com a evidência atual "
                    f"(margem={decision.margin:.3f}); nenhuma recomendação forte foi emitida."
                ),
                failure_reason=PredictiveFailureReason.AMBIGUOUS_RANKING,
                diagnostic_codes=tuple(dict.fromkeys((*analysis.diagnostic_codes, "AMBIGUOUS_RANKING"))),
                rejected_candidates=rejected,
                ranking_margin=decision.margin,
                ranking_ambiguous=True,
                causal_slice=context.causal_slice,
                root_cause_candidates=context.root_cause_candidates,
                repair_strategies=analysis.repair_strategies,
            )
        winner = next(item for item in decision.candidates if item.id == decision.winner_id)
        explanation = (
            f"{winner.id} venceu pelo maior score determinístico ({winner.ranking_score:.3f}). "
            f"{winner.score_explanation} A recomendação é consultiva e exige aprovação."
        )
        return DecisionReport(
            problem=problem,
            hypotheses=analysis.hypotheses,
            candidates=decision.candidates,
            recommended_candidate_id=winner.id,
            insufficient_evidence=False,
            recommendation_explanation=explanation,
            diagnostic_codes=analysis.diagnostic_codes,
            rejected_candidates=rejected,
            ranking_margin=decision.margin,
            causal_slice=context.causal_slice,
            root_cause_candidates=context.root_cause_candidates,
            repair_strategies=analysis.repair_strategies,
        )

    @staticmethod
    def _insufficient(
        problem: str,
        explanation: str,
        failure_reason: PredictiveFailureReason = (
            PredictiveFailureReason.INSUFFICIENT_STRUCTURAL_EVIDENCE
        ),
    ) -> DecisionReport:
        return DecisionReport(
            problem=problem,
            hypotheses=(),
            candidates=(),
            recommended_candidate_id=None,
            insufficient_evidence=True,
            recommendation_explanation=explanation,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _failure_message(reason: PredictiveFailureReason) -> str:
        return {
            PredictiveFailureReason.REASONER_UNAVAILABLE: (
                "O analisador semântico está indisponível; a evidência estrutural não foi descartada."
            ),
            PredictiveFailureReason.INVALID_STRUCTURED_RESPONSE: (
                "O analisador semântico retornou uma resposta estruturada inválida."
            ),
            PredictiveFailureReason.NO_GROUNDED_HYPOTHESIS: (
                "Não foi possível sustentar uma hipótese nas evidências disponíveis."
            ),
            PredictiveFailureReason.NO_VALID_CANDIDATE: (
                "As hipóteses possuem evidência, mas nenhuma solução candidata válida foi produzida."
            ),
            PredictiveFailureReason.NO_DISTINCT_CANDIDATE: (
                "As soluções propostas não eram suficientemente distintas."
            ),
            PredictiveFailureReason.RANKING_FAILURE: (
                "Não foi possível ordenar as soluções candidatas."
            ),
            PredictiveFailureReason.INSUFFICIENT_STRUCTURAL_EVIDENCE: (
                "Nenhuma evidência estrutural forte foi localizada."
            ),
            PredictiveFailureReason.UNSUPPORTED_CLAIM: "As afirmações produzidas não foram sustentadas pelo ledger.",
            PredictiveFailureReason.WEAKLY_SUPPORTED_HYPOTHESIS: "As hipóteses possuem apenas inferências fracas.",
            PredictiveFailureReason.EVIDENCE_TYPE_MISMATCH: "O tipo de evidência não sustenta a afirmação.",
            PredictiveFailureReason.NO_ELIGIBLE_CANDIDATE: "Nenhuma solução candidata permaneceu elegível.",
            PredictiveFailureReason.DUPLICATE_SOLUTION_FAMILY: "As soluções pertencem à mesma família técnica.",
            PredictiveFailureReason.FORBIDDEN_CANDIDATE: "Todas as soluções candidatas violaram a política consultiva.",
            PredictiveFailureReason.AMBIGUOUS_RANKING: "Não há margem de evidência para distinguir os candidatos.",
            PredictiveFailureReason.CAUSAL_SLICE_MISSED_ORIGIN: "O slice causal não localizou uma origem verificável.",
            PredictiveFailureReason.ROOT_CAUSE_CANDIDATE_MISSING: "Nenhum candidato causal estrutural foi encontrado.",
            PredictiveFailureReason.ROOT_CAUSE_SELECTION_ERROR: "O reasoner não selecionou uma causa causal válida.",
            PredictiveFailureReason.ROOT_CAUSE_AMBIGUOUS: "As causas estruturais permanecem indistinguíveis.",
            PredictiveFailureReason.REPAIR_STRATEGY_ERROR: "A estratégia proposta não é compatível com a causa.",
            PredictiveFailureReason.REPAIR_TARGET_ERROR: "O alvo proposto não pertence ao caminho causal.",
        }[reason]
