"""Categorical symptom bridges, not numerical relevance or semantic entailment.

A graph edge is a fact. Its involvement in an observed failure is a separate
claim. No model confidence, benchmark labels or corrected diffs are used here.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .import_graph import ImportCycleCause


class SymptomKind(str, Enum):
    EXCEPTION_FAILURE = "EXCEPTION_FAILURE"
    BEHAVIORAL_MISMATCH = "BEHAVIORAL_MISMATCH"
    IMPORT_INITIALIZATION = "IMPORT_INITIALIZATION"
    UNRESOLVED = "UNRESOLVED"


class RelevanceStatus(str, Enum):
    STRUCTURAL_RELATION_CONFIRMED = "STRUCTURAL_RELATION_CONFIRMED"
    RELEVANCE_TO_SYMPTOM_CONFIRMED = "RELEVANCE_TO_SYMPTOM_CONFIRMED"
    DEFECT_ORIGIN_SUPPORTED = "DEFECT_ORIGIN_SUPPORTED"
    INSUFFICIENT_CAUSAL_BRIDGE = "INSUFFICIENT_CAUSAL_BRIDGE"


@dataclass(frozen=True)
class ProblemSymptom:
    kind: SymptomKind
    reported_path: str | None
    reported_line: int | None
    report: str
    # Unknown expectations stay unknown. Python does not infer a behavioral
    # contract merely from a function name or the existence of a return node.
    expected_behavior: str | None = None
    contract_evidence_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"version": 1, "kind": self.kind.value,
                "reported_path": self.reported_path, "reported_line": self.reported_line,
                "report": self.report, "expected_behavior": self.expected_behavior,
                "contract_evidence_ids": list(self.contract_evidence_ids)}


@dataclass(frozen=True)
class CausalRelevanceAssessment:
    entity_id: str
    status: RelevanceStatus
    structural_relation_confirmed: bool
    relevance_to_symptom_confirmed: bool
    defect_origin_supported: bool
    bridge_refs: tuple[str, ...] = ()
    reason: str = "INSUFFICIENT_CAUSAL_BRIDGE"

    def __post_init__(self) -> None:
        if self.defect_origin_supported and (
            not self.structural_relation_confirmed or not self.relevance_to_symptom_confirmed
            or not self.bridge_refs
        ):
            raise ValueError("Origem sem ponte causal verificavel.")

    def as_dict(self) -> dict[str, object]:
        return {"entity_id": self.entity_id, "status": self.status.value,
                "structural_relation_confirmed": self.structural_relation_confirmed,
                "relevance_to_symptom_confirmed": self.relevance_to_symptom_confirmed,
                "defect_origin_supported": self.defect_origin_supported,
                "bridge_refs": list(self.bridge_refs), "reason": self.reason}


def describe_symptom(problem: str, path: str | None, line: int | None) -> ProblemSymptom:
    # A reported location is not itself evidence that an exception happened.
    has_frame = bool(re.search(r'File\s+["\'].+?["\'],\s*line\s+\d+', problem))
    has_exception = bool(re.search(r"(?m)^\s*(?:[\w.]+(?:Error|Exception)|StopIteration)\s*:", problem))
    partial_module = bool(re.search(r"partially initialized|\binitializing\s+[\w.]+", problem, re.I))
    kind = (SymptomKind.EXCEPTION_FAILURE if has_frame and has_exception
            else SymptomKind.IMPORT_INITIALIZATION if partial_module
            else SymptomKind.BEHAVIORAL_MISMATCH if problem.strip()
            else SymptomKind.UNRESOLVED)
    return ProblemSymptom(kind, path, line, problem[:1024])


def _top_level_definitions(tree: ast.Module) -> dict[str, int]:
    result: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = node.lineno
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.lineno
    return result


def assess_import_cycle(
    cycle: ImportCycleCause,
    symptom: ProblemSymptom,
    trees: Mapping[str, ast.Module],
) -> CausalRelevanceAssessment:
    """Require both an observed initialization anchor and an eager cyclic lookup.

    Conservative subset: a module-level ``from`` lookup requests a member
    defined only after that module's own cycle imports. Dynamic initialization,
    exception text alone and benign module cycles do not prove this relation.
    This supports an origin; it is not an execution trace or a universal proof.
    """
    identity = cycle.identity()
    missing = CausalRelevanceAssessment(identity.id, RelevanceStatus.INSUFFICIENT_CAUSAL_BRIDGE,
                                       True, False, False)
    # A traceback inside a consumer does not make its module imports causal.
    observed_edge = next((edge for edge in cycle.production_edges
                          if edge.source == symptom.reported_path and edge.line == symptom.reported_line), None)
    if observed_edge is None:
        observed_edge = next((edge for edge in cycle.observer_edges
                              if edge.source == symptom.reported_path and edge.line == symptom.reported_line
                              and edge.scope == "module"), None)
    partial = re.search(r"partially initialized module\s+['\"]?([\w.]+)", symptom.report, re.I)
    initializing = re.search(r"\binitializing\s+['\"]?([\w.]+)", symptom.report, re.I)
    loading = re.search(r"\b(?:imports?|importing|loading)\s+['\"]?([\w.]+)", symptom.report, re.I)
    observation = partial or initializing or loading
    module = observation.group(1).rstrip(".") if observation else None
    named_member = (
        bool(re.search(r"partially initialized|\binitializing\s+[\w.]+", symptom.report, re.I))
        and module is not None
        and any(module == member.removesuffix(".py").replace("/", ".")
                for member in cycle.strongly_connected_component)
    )
    if observed_edge is None and not named_member:
        return missing
    refs = []
    for edge in cycle.production_edges:
        source, target = trees.get(edge.source), trees.get(edge.target)
        if source is None or target is None:
            continue
        lookup = next((node for node in source.body
                       if isinstance(node, ast.ImportFrom) and node.lineno == edge.line), None)
        if lookup is None:
            continue
        definitions = _top_level_definitions(target)
        target_edges = [other for other in cycle.production_edges if other.source == edge.target]
        for name in lookup.names:
            definition = definitions.get(name.name)
            if definition and any(other.line < definition for other in target_edges):
                refs.extend((f"{edge.source}:{edge.line}", f"{edge.target}:{definition}"))
    if not refs:
        return missing
    if observed_edge:
        refs.append(f"{observed_edge.source}:{observed_edge.line}")
    return CausalRelevanceAssessment(identity.id, RelevanceStatus.DEFECT_ORIGIN_SUPPORTED,
                                     True, True, True, tuple(sorted(set(refs))),
                                     "OBSERVED_INITIALIZATION_AND_EAGER_CYCLIC_LOOKUP")
