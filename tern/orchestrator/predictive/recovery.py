from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable

from ..project_intelligence_v2 import ProjectSnapshotV2, SymbolKind
from ..security import PathPolicy
from .causal import CausalNodeKind, CausalSlice, RootCauseCandidate, build_causal_slice
from .grounding import build_evidence_ledger
from .models import EvidenceExcerpt, ProblemContext, RetrievalEscalation


class UnresolvedRelation(str, Enum):
    CALLER_UNKNOWN = "CALLER_UNKNOWN"
    CALLEE_RETURN_UNKNOWN = "CALLEE_RETURN_UNKNOWN"
    ATTRIBUTE_ORIGIN_UNKNOWN = "ATTRIBUTE_ORIGIN_UNKNOWN"
    IMPORT_ORIGIN_UNKNOWN = "IMPORT_ORIGIN_UNKNOWN"
    CONFIG_ORIGIN_UNKNOWN = "CONFIG_ORIGIN_UNKNOWN"
    ARGUMENT_SOURCE_UNKNOWN = "ARGUMENT_SOURCE_UNKNOWN"
    RETURN_SOURCE_UNKNOWN = "RETURN_SOURCE_UNKNOWN"
    SYMBOL_REFERENCE_UNKNOWN = "SYMBOL_REFERENCE_UNKNOWN"


class RecoveryReason(str, Enum):
    EXPAND_CALLER = "EXPAND_CALLER"
    EXPAND_CALLEE = "EXPAND_CALLEE"
    EXPAND_RETURN_PRODUCER = "EXPAND_RETURN_PRODUCER"
    EXPAND_ARGUMENT_SOURCE = "EXPAND_ARGUMENT_SOURCE"
    EXPAND_ATTRIBUTE_ORIGIN = "EXPAND_ATTRIBUTE_ORIGIN"
    EXPAND_IMPORT_NEIGHBOR = "EXPAND_IMPORT_NEIGHBOR"
    EXPAND_CONFIG_SOURCE = "EXPAND_CONFIG_SOURCE"
    EXPAND_SYMBOL_REFERENCE = "EXPAND_SYMBOL_REFERENCE"


class RecoveryOutcome(str, Enum):
    RECOVERY_NOT_NEEDED = "RECOVERY_NOT_NEEDED"
    RECOVERY_SUCCEEDED = "RECOVERY_SUCCEEDED"
    RECOVERY_BUDGET_EXHAUSTED = "RECOVERY_BUDGET_EXHAUSTED"
    RECOVERY_NO_NEW_EVIDENCE = "RECOVERY_NO_NEW_EVIDENCE"
    RECOVERY_CYCLE_STOP = "RECOVERY_CYCLE_STOP"
    RECOVERY_INSUFFICIENT = "RECOVERY_INSUFFICIENT"
    TRUE_INSUFFICIENT_EVIDENCE = "TRUE_INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class RecoveryBudget:
    max_recovery_attempts: int = 1
    max_depth: int = 2
    max_extra_files: int = 4
    max_extra_symbols: int = 12
    max_extra_evidence_atoms: int = 30

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if int(value) < 0:
                raise ValueError(f"{name} must not be negative")

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class UnresolvedCausalFrontier:
    node_id: str | None
    symbol: str | None
    path: str | None
    unresolved_relation: UnresolvedRelation
    evidence_ids: tuple[str, ...] = ()
    candidate_paths: tuple[str, ...] = ()
    depth: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "symbol": self.symbol,
            "path": self.path,
            "unresolved_relation": self.unresolved_relation.value,
            "evidence_ids": list(self.evidence_ids),
            "candidate_paths": list(self.candidate_paths),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class CausalSufficiency:
    has_failure_site: bool
    has_origin_candidate: bool
    has_structural_path: bool
    has_supporting_evidence: bool
    has_repairable_target: bool

    @property
    def sufficient(self) -> bool:
        return all((
            self.has_failure_site,
            self.has_origin_candidate,
            self.has_structural_path,
            self.has_supporting_evidence,
            self.has_repairable_target,
        ))

    def as_dict(self) -> dict[str, bool]:
        return {
            "has_failure_site": self.has_failure_site,
            "has_origin_candidate": self.has_origin_candidate,
            "has_structural_path": self.has_structural_path,
            "has_supporting_evidence": self.has_supporting_evidence,
            "has_repairable_target": self.has_repairable_target,
            "sufficient": self.sufficient,
        }


@dataclass(frozen=True)
class RecoveryAction:
    reason: RecoveryReason
    frontier: UnresolvedCausalFrontier
    added_files: tuple[str, ...]
    added_symbols: tuple[str, ...]
    added_evidence_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "frontier": self.frontier.as_dict(),
            "added_files": list(self.added_files),
            "added_symbols": list(self.added_symbols),
            "added_evidence_ids": list(self.added_evidence_ids),
        }


@dataclass(frozen=True)
class StructuralRecoveryTrace:
    outcome: RecoveryOutcome
    budget: RecoveryBudget
    initial_sufficiency: CausalSufficiency
    final_sufficiency: CausalSufficiency
    frontiers: tuple[UnresolvedCausalFrontier, ...] = ()
    actions: tuple[RecoveryAction, ...] = ()
    initial_files: tuple[str, ...] = ()
    final_files: tuple[str, ...] = ()
    initial_root_count: int = 0
    final_root_count: int = 0
    initial_evidence_atom_count: int = 0
    final_evidence_atom_count: int = 0

    @property
    def attempted(self) -> bool:
        return bool(self.actions) or self.outcome not in {
            RecoveryOutcome.RECOVERY_NOT_NEEDED,
            RecoveryOutcome.TRUE_INSUFFICIENT_EVIDENCE,
        }

    @property
    def succeeded(self) -> bool:
        return self.outcome is RecoveryOutcome.RECOVERY_SUCCEEDED

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "outcome": self.outcome.value,
            "budget": self.budget.as_dict(),
            "initial_sufficiency": self.initial_sufficiency.as_dict(),
            "final_sufficiency": self.final_sufficiency.as_dict(),
            "frontiers": [item.as_dict() for item in self.frontiers],
            "actions": [item.as_dict() for item in self.actions],
            "initial_files": list(self.initial_files),
            "final_files": list(self.final_files),
            "initial_root_count": self.initial_root_count,
            "final_root_count": self.final_root_count,
            "initial_evidence_atom_count": self.initial_evidence_atom_count,
            "final_evidence_atom_count": self.final_evidence_atom_count,
            "extra_files": len(set(self.final_files) - set(self.initial_files)),
            "extra_evidence_atoms": max(
                0, self.final_evidence_atom_count - self.initial_evidence_atom_count
            ),
        }


@dataclass(frozen=True)
class StructuralRecoveryResult:
    context: ProblemContext
    causal_slice: CausalSlice
    root_causes: tuple[RootCauseCandidate, ...]
    trace: StructuralRecoveryTrace


_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)")


def causal_sufficiency(
    causal_slice: CausalSlice,
    roots: Iterable[RootCauseCandidate],
) -> CausalSufficiency:
    values = tuple(roots)
    failure_id = causal_slice.failure_site_id
    nodes = {item.id: item for item in causal_slice.nodes}
    grounded_origins = tuple(
        item for item in values
        if item.causal_path
        and nodes.get(item.causal_path[0]) is not None
        and nodes[item.causal_path[0]].evidence_ids
    )
    return CausalSufficiency(
        has_failure_site=failure_id is not None,
        has_origin_candidate=bool(values),
        has_structural_path=any(
            item.causal_path and item.causal_path[-1] == failure_id
            and item.structural_support > 0.0
            for item in values
        ),
        has_supporting_evidence=bool(grounded_origins),
        has_repairable_target=any(item.origin_path for item in grounded_origins),
    )


_RELATION_PRIORITY = {
    UnresolvedRelation.CALLEE_RETURN_UNKNOWN: 0,
    UnresolvedRelation.RETURN_SOURCE_UNKNOWN: 1,
    UnresolvedRelation.ARGUMENT_SOURCE_UNKNOWN: 2,
    UnresolvedRelation.CALLER_UNKNOWN: 3,
    UnresolvedRelation.ATTRIBUTE_ORIGIN_UNKNOWN: 4,
    UnresolvedRelation.IMPORT_ORIGIN_UNKNOWN: 5,
    UnresolvedRelation.CONFIG_ORIGIN_UNKNOWN: 6,
    UnresolvedRelation.SYMBOL_REFERENCE_UNKNOWN: 7,
}


_REASON_FOR_RELATION = {
    UnresolvedRelation.CALLER_UNKNOWN: RecoveryReason.EXPAND_CALLER,
    UnresolvedRelation.CALLEE_RETURN_UNKNOWN: RecoveryReason.EXPAND_CALLEE,
    UnresolvedRelation.RETURN_SOURCE_UNKNOWN: RecoveryReason.EXPAND_RETURN_PRODUCER,
    UnresolvedRelation.ARGUMENT_SOURCE_UNKNOWN: RecoveryReason.EXPAND_ARGUMENT_SOURCE,
    UnresolvedRelation.ATTRIBUTE_ORIGIN_UNKNOWN: RecoveryReason.EXPAND_ATTRIBUTE_ORIGIN,
    UnresolvedRelation.IMPORT_ORIGIN_UNKNOWN: RecoveryReason.EXPAND_IMPORT_NEIGHBOR,
    UnresolvedRelation.CONFIG_ORIGIN_UNKNOWN: RecoveryReason.EXPAND_CONFIG_SOURCE,
    UnresolvedRelation.SYMBOL_REFERENCE_UNKNOWN: RecoveryReason.EXPAND_SYMBOL_REFERENCE,
}


class StructuralRecoveryService:
    """Resolve one bounded, deterministic causal frontier from an existing index."""

    def __init__(self, budget: RecoveryBudget | None = None) -> None:
        self.budget = budget or RecoveryBudget()

    def recover(
        self,
        context: ProblemContext,
        snapshot: ProjectSnapshotV2,
        path_policy: PathPolicy,
        causal_slice: CausalSlice,
        roots: tuple[RootCauseCandidate, ...],
    ) -> StructuralRecoveryResult:
        initial = self._with_origin_coverage(
            causal_sufficiency(causal_slice, roots), context, roots, snapshot
        )
        frontiers = self._frontiers(context, snapshot, causal_slice)
        root_path_nodes = {
            node_id for item in roots for node_id in item.causal_path
        }
        unresolved_grounded_origin = any(
            item.unresolved_relation is UnresolvedRelation.ATTRIBUTE_ORIGIN_UNKNOWN
            and item.node_id in root_path_nodes
            for item in frontiers
        )
        if initial.sufficient and not unresolved_grounded_origin:
            trace = self._trace(
                RecoveryOutcome.RECOVERY_NOT_NEEDED, context, context,
                initial, initial, (), (), roots, roots,
            )
            return StructuralRecoveryResult(context, causal_slice, roots, trace)
        if self.budget.max_recovery_attempts == 0:
            trace = self._trace(
                RecoveryOutcome.RECOVERY_BUDGET_EXHAUSTED, context, context,
                initial, initial, (), (), roots, roots,
            )
            return StructuralRecoveryResult(context, causal_slice, roots, trace)

        if not frontiers:
            outcome = (
                RecoveryOutcome.TRUE_INSUFFICIENT_EVIDENCE
                if not context.related_files
                else RecoveryOutcome.RECOVERY_NO_NEW_EVIDENCE
            )
            trace = self._trace(
                outcome, context, context, initial, initial, (), (), roots, roots,
            )
            return StructuralRecoveryResult(context, causal_slice, roots, trace)

        chosen_paths: list[str] = []
        chosen: list[UnresolvedCausalFrontier] = []
        for frontier in frontiers:
            if frontier.depth > self.budget.max_depth:
                continue
            new_paths = [
                item for item in frontier.candidate_paths
                if item not in context.related_files and item not in chosen_paths
            ]
            if not new_paths:
                continue
            remaining = self.budget.max_extra_files - len(chosen_paths)
            if remaining <= 0:
                break
            chosen_paths.extend(new_paths[:remaining])
            chosen.append(frontier)

        if not chosen_paths:
            trace = self._trace(
                RecoveryOutcome.RECOVERY_NO_NEW_EVIDENCE, context, context,
                initial, initial, frontiers, (), roots, roots,
            )
            return StructuralRecoveryResult(context, causal_slice, roots, trace)

        # Resolve at most one bounded structural chain in the same recovery
        # attempt. This is not a new repository search: every hop comes from
        # the already-built import or unique symbol-definition indexes.
        visited = set(context.related_files) | set(chosen_paths)
        queue = [(path, 1) for path in chosen_paths]
        while queue and len(chosen_paths) < self.budget.max_extra_files:
            current, depth = queue.pop(0)
            if depth >= self.budget.max_depth:
                continue
            indexed = snapshot.file_index.get(current)
            if not indexed:
                continue
            neighbours = set(snapshot.import_graph.get(current, ())) | set(
                snapshot.reverse_import_graph.get(current, ())
            )
            for name in indexed.referenced_names:
                definitions = {item.file for item in snapshot.symbol_index.get(name, ())}
                if len(definitions) == 1:
                    neighbours.update(definitions)
            for path in sorted(neighbours):
                candidate = snapshot.file_index.get(path)
                if (
                    path in visited or not candidate or candidate.is_test
                    or not candidate.analyzed or candidate.language.casefold() != "python"
                ):
                    continue
                visited.add(path)
                chosen_paths.append(path)
                derived = UnresolvedCausalFrontier(
                    None, None, current,
                    UnresolvedRelation.IMPORT_ORIGIN_UNKNOWN,
                    (), (path,), depth + 1,
                )
                chosen.append(derived)
                queue.append((path, depth + 1))
                if len(chosen_paths) >= self.budget.max_extra_files:
                    break

        recovered, actions, atom_budget_exhausted = self._enrich(
            context, snapshot, path_policy, tuple(chosen_paths), tuple(chosen)
        )
        recovered = replace(
            recovered,
            evidence_ledger=build_evidence_ledger(recovered, snapshot, path_policy),
        )
        recovered_slice, recovered_roots = build_causal_slice(
            recovered, snapshot, path_policy
        )
        final = self._with_origin_coverage(
            causal_sufficiency(recovered_slice, recovered_roots),
            recovered,
            recovered_roots,
            snapshot,
        )
        outcome = (
            RecoveryOutcome.RECOVERY_BUDGET_EXHAUSTED
            if atom_budget_exhausted
            else RecoveryOutcome.RECOVERY_SUCCEEDED
            if final.sufficient
            else RecoveryOutcome.RECOVERY_BUDGET_EXHAUSTED
            if len(chosen_paths) >= self.budget.max_extra_files
            else RecoveryOutcome.RECOVERY_INSUFFICIENT
        )
        escalation = RetrievalEscalation(
            attempted=True,
            initial_candidates=context.related_files,
            expanded_candidates=tuple(chosen_paths),
            reasons=tuple(_REASON_FOR_RELATION[item.unresolved_relation].value for item in chosen),
            causal_relations=tuple(item.unresolved_relation.value for item in chosen),
            max_depth=self.budget.max_depth,
            max_extra_files=self.budget.max_extra_files,
            max_extra_symbols=self.budget.max_extra_symbols,
        )
        recovered = replace(recovered, retrieval_escalation=escalation)
        trace = self._trace(
            outcome, context, recovered, initial, final, frontiers, actions,
            roots, recovered_roots,
        )
        return StructuralRecoveryResult(
            recovered, recovered_slice, recovered_roots, trace
        )

    @staticmethod
    def _with_origin_coverage(
        sufficiency: CausalSufficiency,
        context: ProblemContext,
        roots: tuple[RootCauseCandidate, ...],
        snapshot: ProjectSnapshotV2,
    ) -> CausalSufficiency:
        uncovered = any(
            item.origin_path not in context.related_files
            and item.origin_path in snapshot.file_index
            and not snapshot.file_index[item.origin_path].is_test
            for item in roots
        )
        if not uncovered:
            return sufficiency
        return replace(
            sufficiency,
            has_supporting_evidence=False,
            has_repairable_target=False,
        )

    def _frontiers(
        self,
        context: ProblemContext,
        snapshot: ProjectSnapshotV2,
        causal_slice: CausalSlice,
    ) -> tuple[UnresolvedCausalFrontier, ...]:
        selected = set(context.related_files)
        values: list[UnresolvedCausalFrontier] = []
        symbol_count = 0

        if not selected:
            folded_problem = context.problem.casefold()
            for indexed in snapshot.files:
                module = (indexed.module or "").casefold()
                if not module or not re.search(
                    rf"(?<![A-Za-z0-9_.]){re.escape(module)}(?![A-Za-z0-9_.])",
                    folded_problem,
                ):
                    continue
                values.append(UnresolvedCausalFrontier(
                    None, indexed.module, None,
                    UnresolvedRelation.IMPORT_ORIGIN_UNKNOWN,
                    (), (indexed.path,), 1,
                ))
            for token in dict.fromkeys(_IDENTIFIER.findall(context.problem)):
                records = snapshot.symbol_index.get(token, ())
                paths = tuple(sorted({item.file for item in records}))
                if len(paths) == 1:
                    values.append(UnresolvedCausalFrontier(
                        None, token, None,
                        UnresolvedRelation.SYMBOL_REFERENCE_UNKNOWN,
                        (), paths, 1,
                    ))
                    symbol_count += 1
                    if symbol_count >= self.budget.max_extra_symbols:
                        break

        for path in sorted(selected):
            indexed = snapshot.file_index.get(path)
            if not indexed:
                continue
            for neighbour in sorted(set(snapshot.import_graph.get(path, ())) - selected):
                values.append(UnresolvedCausalFrontier(
                    None, None, path, UnresolvedRelation.IMPORT_ORIGIN_UNKNOWN,
                    (), (neighbour,), 1,
                ))
            for neighbour in sorted(set(snapshot.reverse_import_graph.get(path, ())) - selected):
                if snapshot.file_index[neighbour].is_test and not indexed.is_test:
                    continue
                values.append(UnresolvedCausalFrontier(
                    None, None, path, UnresolvedRelation.CALLER_UNKNOWN,
                    (), (neighbour,), 1,
                ))
            for name in indexed.referenced_names:
                if symbol_count >= self.budget.max_extra_symbols:
                    break
                symbol_count += 1
                records = snapshot.symbol_index.get(name, ())
                definitions = tuple(sorted({
                    item.file for item in records
                    if item.file not in selected
                }))
                if len(definitions) == 1:
                    relation = (
                        UnresolvedRelation.ATTRIBUTE_ORIGIN_UNKNOWN
                        if any(item.kind in {
                            SymbolKind.CLASS_ATTRIBUTE,
                            SymbolKind.INSTANCE_ATTRIBUTE,
                        } for item in records)
                        else UnresolvedRelation.CALLEE_RETURN_UNKNOWN
                    )
                    values.append(UnresolvedCausalFrontier(
                        None, name, path,
                        relation,
                        (), definitions, 1,
                    ))
            defined = {item.name for item in indexed.symbols}
            for other in snapshot.files:
                if other.path in selected or (other.is_test and not indexed.is_test):
                    continue
                shared = sorted(defined & set(other.referenced_names))
                if shared:
                    values.append(UnresolvedCausalFrontier(
                        None, shared[0], path,
                        UnresolvedRelation.ARGUMENT_SOURCE_UNKNOWN,
                        (), (other.path,), 1,
                    ))

        incoming = {edge.target_id for edge in causal_slice.edges}
        for node in causal_slice.nodes:
            if node.id in incoming:
                continue
            if node.kind is CausalNodeKind.ATTRIBUTE and node.symbol:
                name = node.symbol.rsplit(".", 1)[-1]
                paths = tuple(sorted({
                    item.path for item in snapshot.files
                    if item.path not in selected and name in item.referenced_names
                }))
                if paths:
                    values.append(UnresolvedCausalFrontier(
                        node.id, node.symbol, node.path,
                        UnresolvedRelation.ATTRIBUTE_ORIGIN_UNKNOWN,
                        node.evidence_ids, paths, 1,
                    ))

        unique: dict[tuple[object, ...], UnresolvedCausalFrontier] = {}
        for item in values:
            key = (item.unresolved_relation, item.symbol, item.path, item.candidate_paths)
            unique.setdefault(key, item)
        return tuple(sorted(
            unique.values(),
            key=lambda item: (
                _RELATION_PRIORITY[item.unresolved_relation],
                item.depth,
                item.path or "",
                item.symbol or "",
                item.candidate_paths,
            ),
        ))

    def _enrich(
        self,
        context: ProblemContext,
        snapshot: ProjectSnapshotV2,
        path_policy: PathPolicy,
        paths: tuple[str, ...],
        frontiers: tuple[UnresolvedCausalFrontier, ...],
    ) -> tuple[ProblemContext, tuple[RecoveryAction, ...], bool]:
        excerpts = list(context.evidence)
        related_symbols = list(context.related_symbols)
        root = Path(snapshot.project_path)
        symbols_added = 0
        action_rows: list[tuple[RecoveryReason, UnresolvedCausalFrontier, list[str]]] = []
        for frontier in frontiers:
            action_rows.append((_REASON_FOR_RELATION[frontier.unresolved_relation], frontier, []))
        for path in paths:
            indexed = snapshot.file_index[path]
            lines = path_policy.resolve(str(root / path)).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if not lines:
                continue
            relevant = [
                item for item in indexed.symbols
                if any(
                    item.name == frontier.symbol or item.qualified_name == frontier.symbol
                    for frontier in frontiers if frontier.symbol
                )
            ]
            if relevant:
                start = max(1, min(item.line for item in relevant) - 2)
                end = min(len(lines), max(item.end_line for item in relevant) + 2)
            else:
                start, end = 1, min(len(lines), 80)
            excerpts.append(EvidenceExcerpt(
                ref=f"{path}:{start}-{end}",
                path=path,
                start_line=start,
                end_line=end,
                strength="STRONG",
                sources=("STRUCTURAL_RECOVERY",),
                excerpt="\n".join(
                    f"{number}: {lines[number - 1]}"
                    for number in range(start, end + 1)
                ),
            ))
            for symbol in indexed.symbols:
                if symbols_added >= self.budget.max_extra_symbols:
                    break
                if start <= symbol.line <= end or start <= symbol.end_line <= end:
                    related_symbols.append(
                        f"{symbol.qualified_name}@{symbol.file}:{symbol.line}-{symbol.end_line}"
                    )
                    symbols_added += 1

        related_files = tuple(dict.fromkeys((*context.related_files, *paths)))
        related_tests = tuple(sorted(set(context.related_tests) | {
            relation.test_file for relation in snapshot.test_relationships
            if relation.production_file in related_files
        }))
        recovered = replace(
            context,
            evidence=tuple(excerpts),
            related_files=related_files,
            related_symbols=tuple(related_symbols),
            related_tests=related_tests,
        )
        before_ids = set(context.evidence_ledger.ids)
        provisional = build_evidence_ledger(recovered, snapshot, path_policy)
        added_ids = tuple(item for item in provisional.ids if item not in before_ids)
        if len(added_ids) > self.budget.max_extra_evidence_atoms:
            # Evidence is an all-or-nothing structural input.  Silently
            # retaining atoms above the declared budget would make the trace
            # misleading; fail closed and let the caller report exhaustion.
            return context, (), True
        actions = tuple(
            RecoveryAction(
                reason, frontier,
                tuple(path for path in paths if path in frontier.candidate_paths),
                tuple(
                    item for item in recovered.related_symbols
                    if item not in context.related_symbols
                    and item.split("@", 1)[-1].split(":", 1)[0] in frontier.candidate_paths
                ),
                tuple(
                    atom_id for atom_id in added_ids
                    if provisional.get(atom_id)
                    and provisional.get(atom_id).path in frontier.candidate_paths
                )[: self.budget.max_extra_evidence_atoms],
            )
            for reason, frontier, _unused in action_rows
            if set(paths) & set(frontier.candidate_paths)
        )
        return recovered, actions, False

    def _trace(
        self,
        outcome: RecoveryOutcome,
        initial_context: ProblemContext,
        final_context: ProblemContext,
        initial: CausalSufficiency,
        final: CausalSufficiency,
        frontiers: tuple[UnresolvedCausalFrontier, ...],
        actions: tuple[RecoveryAction, ...],
        initial_roots: tuple[RootCauseCandidate, ...],
        final_roots: tuple[RootCauseCandidate, ...],
    ) -> StructuralRecoveryTrace:
        return StructuralRecoveryTrace(
            outcome=outcome,
            budget=self.budget,
            initial_sufficiency=initial,
            final_sufficiency=final,
            frontiers=frontiers,
            actions=actions,
            initial_files=initial_context.related_files,
            final_files=final_context.related_files,
            initial_root_count=len(initial_roots),
            final_root_count=len(final_roots),
            initial_evidence_atom_count=len(initial_context.evidence_ledger.atoms),
            final_evidence_atom_count=len(final_context.evidence_ledger.atoms),
        )
