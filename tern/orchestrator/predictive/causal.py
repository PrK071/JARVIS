from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..project_intelligence_v2 import ProjectSnapshotV2
from ..security import PathPolicy
from .models import EvidenceExcerpt, EvidenceLedger, ProblemContext


class CausalNodeKind(str, Enum):
    PARAMETER = "PARAMETER"
    LOCAL_VARIABLE = "LOCAL_VARIABLE"
    ATTRIBUTE = "ATTRIBUTE"
    CALL = "CALL"
    RETURN_VALUE = "RETURN_VALUE"
    CONDITION = "CONDITION"
    EXCEPTION_SITE = "EXCEPTION_SITE"
    CONFIG_VALUE = "CONFIG_VALUE"
    IMPORT = "IMPORT"
    TEST_EXPECTATION = "TEST_EXPECTATION"
    LITERAL = "LITERAL"


class CausalEdgeKind(str, Enum):
    ASSIGNED_FROM = "ASSIGNED_FROM"
    PASSED_AS_ARGUMENT = "PASSED_AS_ARGUMENT"
    RETURNED_FROM = "RETURNED_FROM"
    READ_FROM_ATTRIBUTE = "READ_FROM_ATTRIBUTE"
    CALLS = "CALLS"
    GUARDED_BY = "GUARDED_BY"
    RAISES_AT = "RAISES_AT"
    CONFIGURED_BY = "CONFIGURED_BY"
    ASSERTED_BY = "ASSERTED_BY"


class RootCauseKind(str, Enum):
    NULL_FLOW = "NULL_FLOW"
    TYPE_FLOW = "TYPE_FLOW"
    ARGUMENT_BINDING = "ARGUMENT_BINDING"
    RETURN_CONTRACT = "RETURN_CONTRACT"
    ATTRIBUTE_VALUE = "ATTRIBUTE_VALUE"
    CONTROL_FLOW = "CONTROL_FLOW"
    CONFIGURATION = "CONFIGURATION"
    IMPORT_RESOLUTION = "IMPORT_RESOLUTION"
    STATE_PROPAGATION = "STATE_PROPAGATION"
    TEST_EXPECTATION = "TEST_EXPECTATION"
    UNKNOWN = "UNKNOWN"


class RepairStrategyKind(str, Enum):
    FIX_PRODUCER = "FIX_PRODUCER"
    FIX_CONSUMER_CONTRACT = "FIX_CONSUMER_CONTRACT"
    VALIDATE_BOUNDARY = "VALIDATE_BOUNDARY"
    CORRECT_ARGUMENT = "CORRECT_ARGUMENT"
    CORRECT_RETURN_VALUE = "CORRECT_RETURN_VALUE"
    CORRECT_CONTROL_FLOW = "CORRECT_CONTROL_FLOW"
    CORRECT_CONFIGURATION = "CORRECT_CONFIGURATION"
    CORRECT_IMPORT = "CORRECT_IMPORT"
    CORRECT_TEST_EXPECTATION = "CORRECT_TEST_EXPECTATION"
    ERROR_HANDLING = "ERROR_HANDLING"
    OTHER = "OTHER"


class RootSelectionReason(str, Enum):
    STRUCTURAL_DOMINANCE = "STRUCTURAL_DOMINANCE"
    STRONGER_DIRECT_SUPPORT = "STRONGER_DIRECT_SUPPORT"
    STRONGER_CAUSAL_PATH = "STRONGER_CAUSAL_PATH"
    UPSTREAM_ORIGIN = "UPSTREAM_ORIGIN"
    MANIFESTATION_NOT_ORIGIN = "MANIFESTATION_NOT_ORIGIN"
    CONFIG_SOURCE = "CONFIG_SOURCE"
    RETURN_SOURCE = "RETURN_SOURCE"
    ARGUMENT_SOURCE = "ARGUMENT_SOURCE"
    CONTROL_FLOW_SOURCE = "CONTROL_FLOW_SOURCE"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
    EQUIVALENT_CANDIDATES = "EQUIVALENT_CANDIDATES"


_COMPATIBILITY: Mapping[RootCauseKind, frozenset[RepairStrategyKind]] = {
    RootCauseKind.NULL_FLOW: frozenset({RepairStrategyKind.FIX_PRODUCER, RepairStrategyKind.VALIDATE_BOUNDARY, RepairStrategyKind.CORRECT_RETURN_VALUE}),
    RootCauseKind.TYPE_FLOW: frozenset({RepairStrategyKind.FIX_PRODUCER, RepairStrategyKind.FIX_CONSUMER_CONTRACT, RepairStrategyKind.VALIDATE_BOUNDARY, RepairStrategyKind.CORRECT_ARGUMENT, RepairStrategyKind.CORRECT_RETURN_VALUE}),
    RootCauseKind.ARGUMENT_BINDING: frozenset({RepairStrategyKind.CORRECT_ARGUMENT, RepairStrategyKind.VALIDATE_BOUNDARY}),
    RootCauseKind.RETURN_CONTRACT: frozenset({RepairStrategyKind.FIX_PRODUCER, RepairStrategyKind.CORRECT_RETURN_VALUE, RepairStrategyKind.VALIDATE_BOUNDARY}),
    RootCauseKind.ATTRIBUTE_VALUE: frozenset({RepairStrategyKind.FIX_PRODUCER, RepairStrategyKind.FIX_CONSUMER_CONTRACT, RepairStrategyKind.VALIDATE_BOUNDARY}),
    RootCauseKind.CONTROL_FLOW: frozenset({RepairStrategyKind.CORRECT_CONTROL_FLOW, RepairStrategyKind.CORRECT_RETURN_VALUE}),
    RootCauseKind.CONFIGURATION: frozenset({RepairStrategyKind.CORRECT_CONFIGURATION, RepairStrategyKind.VALIDATE_BOUNDARY}),
    RootCauseKind.IMPORT_RESOLUTION: frozenset({RepairStrategyKind.CORRECT_IMPORT}),
    RootCauseKind.STATE_PROPAGATION: frozenset({RepairStrategyKind.FIX_PRODUCER, RepairStrategyKind.FIX_CONSUMER_CONTRACT, RepairStrategyKind.VALIDATE_BOUNDARY}),
    RootCauseKind.TEST_EXPECTATION: frozenset({RepairStrategyKind.CORRECT_TEST_EXPECTATION, RepairStrategyKind.FIX_PRODUCER}),
    RootCauseKind.UNKNOWN: frozenset({RepairStrategyKind.OTHER}),
}


def strategy_compatible(cause: RootCauseKind, strategy: RepairStrategyKind) -> bool:
    return strategy in _COMPATIBILITY[cause]


def compatible_strategies(cause: RootCauseKind) -> tuple[RepairStrategyKind, ...]:
    return tuple(sorted(_COMPATIBILITY[cause], key=lambda item: item.value))


def compatible_strategies_for_root(
    root: RootCauseCandidate, causal_slice: CausalSlice | None
) -> tuple[RepairStrategyKind, ...]:
    strategies = list(compatible_strategies(root.cause_kind))
    if (
        root.cause_kind is RootCauseKind.NULL_FLOW
        and causal_slice is not None
        and not any(
            node.id in root.causal_path
            and node.path == root.origin_path
            and node.kind is CausalNodeKind.RETURN_VALUE
            for node in causal_slice.nodes
        )
    ):
        strategies = [
            item for item in strategies
            if item is not RepairStrategyKind.CORRECT_RETURN_VALUE
        ]
    return tuple(strategies)


def _stable(prefix: str, *parts: object) -> str:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return prefix + hashlib.sha256(raw).hexdigest()[:12].upper()


@dataclass(frozen=True)
class CausalNode:
    id: str
    kind: CausalNodeKind
    path: str
    line: int
    symbol: str | None
    expression: str
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind.value, "path": self.path,
                "line": self.line, "symbol": self.symbol,
                "expression": self.expression, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class CausalEdge:
    id: str
    source_id: str
    target_id: str
    kind: CausalEdgeKind
    deterministic: bool
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "source_id": self.source_id,
                "target_id": self.target_id, "kind": self.kind.value,
                "deterministic": self.deterministic,
                "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class CausalSlice:
    failure_site_id: str | None
    nodes: tuple[CausalNode, ...]
    edges: tuple[CausalEdge, ...]
    unknown_flow: bool = False

    def __post_init__(self) -> None:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("causal node IDs must be unique")
        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("causal edge IDs must be unique")
        known = set(ids)
        if any(edge.source_id not in known or edge.target_id not in known for edge in self.edges):
            raise ValueError("causal edges must reference known nodes")

    def as_dict(self) -> dict[str, object]:
        return {"version": 1, "failure_site_id": self.failure_site_id,
                "unknown_flow": self.unknown_flow,
                "nodes": [node.as_dict() for node in self.nodes],
                "edges": [edge.as_dict() for edge in self.edges]}


@dataclass(frozen=True)
class RootCauseCandidate:
    id: str
    cause_kind: RootCauseKind
    origin_path: str
    origin_symbol: str | None
    origin_line: int
    failure_path: str
    failure_line: int
    causal_path: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    direct_support: float
    structural_support: float
    causal_distance: int
    completeness: float
    score: float
    statement: str

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "cause_kind": self.cause_kind.value,
                "origin_path": self.origin_path, "origin_symbol": self.origin_symbol,
                "origin_line": self.origin_line, "failure_path": self.failure_path,
                "failure_line": self.failure_line, "causal_path": list(self.causal_path),
                "evidence_ids": list(self.evidence_ids), "direct_support": self.direct_support,
                "structural_support": self.structural_support,
                "causal_distance": self.causal_distance, "completeness": self.completeness,
                "score": self.score, "statement": self.statement}


@dataclass(frozen=True)
class RepairStrategy:
    id: str
    strategy_kind: RepairStrategyKind
    root_cause_id: str
    target_files: tuple[str, ...]
    target_symbols: tuple[str, ...]
    mechanism: str
    rationale: str
    evidence_ids: tuple[str, ...]
    compatible: bool
    locality_score: float

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "strategy_kind": self.strategy_kind.value,
                "root_cause_id": self.root_cause_id,
                "target_files": list(self.target_files),
                "target_symbols": list(self.target_symbols), "mechanism": self.mechanism,
                "rationale": self.rationale, "evidence_ids": list(self.evidence_ids),
                "compatible": self.compatible, "locality_score": self.locality_score}


@dataclass(frozen=True)
class RejectedRootCause:
    root_cause_id: str
    reason: RootSelectionReason

    def as_dict(self) -> dict[str, str]:
        return {"root_cause_id": self.root_cause_id, "reason": self.reason.value}


@dataclass(frozen=True)
class RootCauseSelection:
    selected_id: str
    reason: RootSelectionReason
    rejected: tuple[RejectedRootCause, ...] = ()
    structurally_dominant: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_id": self.selected_id,
            "reason": self.reason.value,
            "rejected": [item.as_dict() for item in self.rejected],
            "structurally_dominant": self.structurally_dominant,
        }


@dataclass
class _Function:
    key: str
    name: str
    path: str
    line: int
    params: list[str]
    param_nodes: dict[str, str]
    returns: list[str]


@dataclass
class _Call:
    node_id: str
    path: str
    scope: str
    target: str
    args: list[tuple[str | None, list[str]]]


def _expr(node: ast.AST | None) -> str:
    if node is None:
        return "None"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return f"str[{len(node.value)}]"
        return repr(node.value)
    try:
        return ast.unparse(node)[:120]
    except Exception:
        return type(node).__name__


def _refs(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    values = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    values |= {
        _expr(item) for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
    }
    return tuple(sorted(values))


def _traceback(problem: str) -> tuple[str | None, int | None]:
    matches = list(re.finditer(
        r'(?:File\s+["\'](?P<file>[^"\']+)["\'],\s+line\s+(?P<line>\d+))|(?P<short>[^\s]+\.py):(?P<short_line>\d+)',
        problem,
    ))
    if not matches:
        return None, None
    match = matches[-1]
    return ((match.group("file") or match.group("short")).replace("\\", "/"),
            int(match.group("line") or match.group("short_line")))


def expand_context_for_causal_flow(
    context: ProblemContext,
    snapshot: ProjectSnapshotV2,
    path_policy: PathPolicy,
    *,
    max_added_files: int = 6,
) -> ProblemContext:
    """Add only uniquely resolved symbol neighbours to the read-only context.

    Project Intelligence already owns discovery and indexing.  This function
    consumes its symbol/reference indexes to close a small caller/callee gap;
    it does not create a second project index or infer dynamic dispatch.
    """
    selected = list(context.related_files)
    selected_set = set(selected)
    added: list[str] = []
    symbol_index = snapshot.symbol_index
    file_index = snapshot.file_index
    frontier = list(selected)
    while frontier and len(added) < max_added_files:
        current_path = frontier.pop(0)
        indexed = file_index.get(current_path)
        if not indexed:
            continue
        neighbours: set[str] = set(snapshot.import_graph.get(current_path, ()))
        neighbours.update(snapshot.reverse_import_graph.get(current_path, ()))
        for name in indexed.referenced_names:
            records = symbol_index.get(name, ())
            defining_files = {record.file for record in records}
            if len(defining_files) == 1:
                neighbours.update(defining_files)
        defined_names = {symbol.name for symbol in indexed.symbols}
        for other in snapshot.files:
            if other.is_test and not indexed.is_test:
                continue
            if defined_names & set(other.referenced_names):
                neighbours.add(other.path)
        for path in sorted(neighbours):
            candidate = file_index.get(path)
            if (
                path in selected_set
                or not candidate
                or not candidate.analyzed
                or candidate.language.casefold() != "python"
            ):
                continue
            selected_set.add(path)
            selected.append(path)
            added.append(path)
            frontier.append(path)
            if len(added) >= max_added_files:
                break
    if not added:
        return context

    excerpts = list(context.evidence)
    symbols = list(context.related_symbols)
    root = Path(snapshot.project_path)
    for path in added:
        indexed = file_index[path]
        source = path_policy.resolve(str(root / path)).read_text(
            encoding="utf-8", errors="replace"
        )
        lines = source.splitlines()
        if not lines:
            continue
        end = min(len(lines), 160)
        excerpts.append(EvidenceExcerpt(
            ref=f"{path}:1-{end}",
            path=path,
            start_line=1,
            end_line=end,
            strength="SUPPORTING",
            sources=("CAUSAL_SYMBOL_RELATION",),
            excerpt="\n".join(f"{line}: {lines[line - 1]}" for line in range(1, end + 1)),
        ))
        symbols.extend(
            f"{symbol.qualified_name}@{symbol.file}:{symbol.line}-{symbol.end_line}"
            for symbol in indexed.symbols
            if symbol.line <= end
        )
    related_tests = set(context.related_tests)
    relationships = set(context.test_relationships)
    for relation in snapshot.test_relationships:
        if relation.production_file in selected_set or relation.test_file in selected_set:
            related_tests.add(relation.test_file)
            relationships.add((relation.production_file, relation.test_file))
    return ProblemContext(
        problem=context.problem,
        project_path=context.project_path,
        project_id=context.project_id,
        evidence=tuple(excerpts),
        related_files=tuple(selected),
        related_symbols=tuple(dict.fromkeys(symbols)),
        related_tests=tuple(sorted(related_tests)),
        test_relationships=tuple(sorted(relationships)),
    )


def _evidence_at(ledger: EvidenceLedger, path: str, line: int) -> tuple[str, ...]:
    exact = tuple(atom.id for atom in ledger.atoms if atom.path == path and atom.start_line <= line <= atom.end_line)
    if exact:
        return exact
    return tuple(atom.id for atom in ledger.atoms if atom.path == path)[:2]


class _FlowBuilder(ast.NodeVisitor):
    def __init__(self, path: str, ledger: EvidenceLedger):
        self.path = path
        self.ledger = ledger
        self.nodes: dict[str, CausalNode] = {}
        self.edges: dict[str, CausalEdge] = {}
        self.functions: list[_Function] = []
        self.calls: list[_Call] = []
        self.scope = "module"
        self.class_name: str | None = None
        self.values: dict[tuple[str, str], str] = {}
        self.current: _Function | None = None
        self.attribute_reads: set[str] = set()
        self.attribute_writes: set[str] = set()

    def node(self, kind: CausalNodeKind, line: int, symbol: str | None, expression: str) -> str:
        node_id = _stable("N", kind.value, self.path, line, self.scope, symbol, expression)
        self.nodes.setdefault(node_id, CausalNode(
            node_id, kind, self.path, max(1, line), symbol, expression,
            _evidence_at(self.ledger, self.path, max(1, line)),
        ))
        return node_id

    def edge(self, source: str, target: str, kind: CausalEdgeKind) -> None:
        evidence = tuple(dict.fromkeys((*self.nodes[source].evidence_ids, *self.nodes[target].evidence_ids)))
        edge_id = _stable("G", source, target, kind.value)
        self.edges.setdefault(edge_id, CausalEdge(edge_id, source, target, kind, True, evidence))

    def value(self, name: str, line: int, kind: CausalNodeKind | None = None) -> str:
        key = (self.scope, name)
        if key in self.values:
            return self.values[key]
        inferred = kind or (CausalNodeKind.ATTRIBUTE if "." in name else CausalNodeKind.LOCAL_VARIABLE)
        value = self.node(inferred, line, name, name)
        self.values[key] = value
        return value

    def source_nodes(self, node: ast.AST | None, line: int) -> list[str]:
        if isinstance(node, ast.Constant):
            kind = CausalNodeKind.CONFIG_VALUE if self.scope == "module" else CausalNodeKind.LITERAL
            return [self.node(kind, line, None, _expr(node))]
        if isinstance(node, ast.Call):
            return [self._call(node)]
        refs = _refs(node)
        values = [self.value(name, line) for name in refs]
        self.attribute_reads.update(
            value for name, value in zip(refs, values) if "." in name
        )
        return values

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.class_name
        self.class_name = node.name
        self.generic_visit(node)
        self.class_name = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous_scope, previous_current = self.scope, self.current
        name = f"{self.class_name}.{node.name}" if self.class_name else node.name
        self.scope = name
        positional = [*node.args.posonlyargs, *node.args.args]
        params = [item.arg for item in positional] + [item.arg for item in node.args.kwonlyargs]
        function = _Function(f"{self.path}:{name}", name, self.path, node.lineno, params, {}, [])
        self.current = function
        for parameter in params:
            function.param_nodes[parameter] = self.value(parameter, node.lineno, CausalNodeKind.PARAMETER)
        defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
        for parameter, default in zip([item.arg for item in positional], defaults):
            if default is not None:
                line = getattr(default, "lineno", node.lineno)
                source = self.node(
                    CausalNodeKind.PARAMETER,
                    line,
                    parameter,
                    f"default {parameter}={_expr(default)}",
                )
                for value_source in self.source_nodes(default, line):
                    self.edge(value_source, source, CausalEdgeKind.ASSIGNED_FROM)
                self.edge(source, function.param_nodes[parameter], CausalEdgeKind.ASSIGNED_FROM)
        self.functions.append(function)
        for statement in node.body:
            self.visit(statement)
        self.scope, self.current = previous_scope, previous_current

    visit_AsyncFunctionDef = visit_FunctionDef

    def _call(self, node: ast.Call) -> str:
        target = _expr(node.func)
        call_id = self.node(CausalNodeKind.CALL, node.lineno, target, f"{target}()")
        if not any(item.node_id == call_id for item in self.calls):
            args = [(None, list(self.source_nodes(value, node.lineno))) for value in node.args]
            args.extend((keyword.arg, list(self.source_nodes(keyword.value, node.lineno))) for keyword in node.keywords)
            if isinstance(node.func, ast.Attribute):
                args.append((None, list(self.source_nodes(node.func.value, node.lineno))))
            self.calls.append(_Call(call_id, self.path, self.scope, target, args))
            for _keyword, sources in args:
                for source in sources:
                    self.edge(source, call_id, CausalEdgeKind.PASSED_AS_ARGUMENT)
        return call_id

    def visit_Call(self, node: ast.Call) -> None:
        self._call(node)

    def _assign(self, targets: Sequence[ast.AST], value: ast.AST | None, line: int) -> None:
        sources = self.source_nodes(value, line)
        for target in targets:
            name = _expr(target)
            kind = CausalNodeKind.CONFIG_VALUE if self.scope == "module" and name.isupper() else None
            target_id = self.value(name, line, kind)
            if isinstance(target, ast.Attribute):
                self.attribute_writes.add(target_id)
            for source in sources:
                edge_kind = CausalEdgeKind.RETURNED_FROM if self.nodes[source].kind is CausalNodeKind.CALL else CausalEdgeKind.ASSIGNED_FROM
                if isinstance(target, ast.Attribute):
                    edge_kind = CausalEdgeKind.READ_FROM_ATTRIBUTE
                self.edge(source, target_id, edge_kind)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._assign(node.targets, node.value, node.lineno)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assign((node.target,), node.value, node.lineno)
        if node.value:
            self.generic_visit(node.value)

    def visit_Return(self, node: ast.Return) -> None:
        return_id = self.node(CausalNodeKind.RETURN_VALUE, node.lineno, self.scope, f"return {_expr(node.value)}")
        for source in self.source_nodes(node.value, node.lineno):
            self.edge(source, return_id, CausalEdgeKind.RETURNED_FROM)
        if self.current:
            self.current.returns.append(return_id)
        if node.value:
            self.generic_visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        condition = self.node(CausalNodeKind.CONDITION, node.lineno, self.scope, _expr(node.test))
        for source in self.source_nodes(node.test, node.lineno):
            self.edge(source, condition, CausalEdgeKind.GUARDED_BY)
        before = set(self.nodes)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)
        for created in set(self.nodes) - before:
            if self.nodes[created].kind in {
                CausalNodeKind.RETURN_VALUE,
                CausalNodeKind.EXCEPTION_SITE,
            }:
                self.edge(condition, created, CausalEdgeKind.GUARDED_BY)

    def visit_Assert(self, node: ast.Assert) -> None:
        assertion = self.node(CausalNodeKind.TEST_EXPECTATION, node.lineno, self.scope, _expr(node.test))
        for source in self.source_nodes(node.test, node.lineno):
            self.edge(source, assertion, CausalEdgeKind.ASSERTED_BY)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        raised = self.node(
            CausalNodeKind.EXCEPTION_SITE,
            node.lineno,
            self.scope,
            f"raise {_expr(node.exc)}",
        )
        for source in self.source_nodes(node.exc, node.lineno):
            self.edge(source, raised, CausalEdgeKind.RAISES_AT)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.node(CausalNodeKind.IMPORT, node.lineno, alias.asname or alias.name, f"import {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.node(
                CausalNodeKind.IMPORT,
                node.lineno,
                alias.asname or alias.name,
                f"from {module} import {alias.name}",
            )


def _node_kind_for_cause(node: CausalNode, problem: str) -> RootCauseKind:
    expression = node.expression.casefold()
    if expression in {"none", "null"} or "nonetype" in problem.casefold():
        if expression == "none":
            return RootCauseKind.NULL_FLOW
    if node.kind is CausalNodeKind.CONFIG_VALUE:
        return RootCauseKind.CONFIGURATION
    if node.kind is CausalNodeKind.RETURN_VALUE:
        return RootCauseKind.RETURN_CONTRACT
    if node.kind is CausalNodeKind.PARAMETER:
        return RootCauseKind.ARGUMENT_BINDING
    if node.kind is CausalNodeKind.ATTRIBUTE:
        return RootCauseKind.ATTRIBUTE_VALUE
    if node.kind is CausalNodeKind.CONDITION:
        return RootCauseKind.CONTROL_FLOW
    if node.kind is CausalNodeKind.IMPORT:
        return RootCauseKind.IMPORT_RESOLUTION
    if node.kind is CausalNodeKind.TEST_EXPECTATION:
        return RootCauseKind.TEST_EXPECTATION
    if node.kind is CausalNodeKind.LITERAL and expression.startswith("str["):
        return RootCauseKind.TYPE_FLOW
    return RootCauseKind.STATE_PROPAGATION


def build_causal_slice(
    context: ProblemContext,
    snapshot: ProjectSnapshotV2,
    path_policy: PathPolicy,
) -> tuple[CausalSlice, tuple[RootCauseCandidate, ...]]:
    builders: list[_FlowBuilder] = []
    # Use the existing Project Intelligence scope, plus its directly related
    # imports/callers. The result is facts, not another project index.
    paths = set(context.related_files)
    for path in tuple(paths):
        paths.update(snapshot.import_graph.get(path, ()))
        paths.update(snapshot.reverse_import_graph.get(path, ()))
    for path in sorted(paths):
        indexed = snapshot.file_index.get(path)
        if not indexed or not indexed.analyzed or not path.endswith(".py"):
            continue
        source = path_policy.resolve(str(Path(snapshot.project_path) / path)).read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        builder = _FlowBuilder(path, context.evidence_ledger)
        builder.visit(tree)
        builders.append(builder)

    nodes = {node.id: node for builder in builders for node in builder.nodes.values()}
    edges = {edge.id: edge for builder in builders for edge in builder.edges.values()}
    functions = [function for builder in builders for function in builder.functions]
    by_name: dict[str, list[_Function]] = {}
    for function in functions:
        by_name.setdefault(function.name.rsplit(".", 1)[-1], []).append(function)
    calls = [call for builder in builders for call in builder.calls]
    for call in calls:
        matches = by_name.get(call.target.rsplit(".", 1)[-1], ())
        if len(matches) != 1:
            continue
        callee = matches[0]
        for index, (keyword, sources) in enumerate(call.args):
            parameter = keyword or (callee.params[index] if index < len(callee.params) else None)
            if parameter not in callee.param_nodes:
                continue
            for source in sources:
                edge_id = _stable("G", source, callee.param_nodes[parameter], CausalEdgeKind.PASSED_AS_ARGUMENT.value)
                evidence = tuple(dict.fromkeys((*nodes[source].evidence_ids, *nodes[callee.param_nodes[parameter]].evidence_ids)))
                edges[edge_id] = CausalEdge(edge_id, source, callee.param_nodes[parameter], CausalEdgeKind.PASSED_AS_ARGUMENT, True, evidence)
        for returned in callee.returns:
            edge_id = _stable("G", returned, call.node_id, CausalEdgeKind.RETURNED_FROM.value)
            evidence = tuple(dict.fromkeys((*nodes[returned].evidence_ids, *nodes[call.node_id].evidence_ids)))
            edges[edge_id] = CausalEdge(edge_id, returned, call.node_id, CausalEdgeKind.RETURNED_FROM, True, evidence)

    # Join a uniquely assigned instance attribute (``self.profile = profile``)
    # to reads through another receiver (``user.profile``).  Attribute-name
    # uniqueness is required; ambiguous class fields remain unresolved.
    attribute_writes = {
        node_id for builder in builders for node_id in builder.attribute_writes
    }
    attribute_reads = {
        node_id for builder in builders for node_id in builder.attribute_reads
    }
    writes_by_name: dict[str, list[str]] = {}
    for node_id in attribute_writes:
        symbol = nodes[node_id].symbol
        if symbol:
            writes_by_name.setdefault(symbol.rsplit(".", 1)[-1], []).append(node_id)
    for read_id in sorted(attribute_reads):
        symbol = nodes[read_id].symbol
        if not symbol:
            continue
        matches = writes_by_name.get(symbol.rsplit(".", 1)[-1], ())
        if len(matches) != 1 or matches[0] == read_id:
            continue
        source = matches[0]
        edge_id = _stable("G", source, read_id, CausalEdgeKind.READ_FROM_ATTRIBUTE.value)
        evidence = tuple(dict.fromkeys((*nodes[source].evidence_ids, *nodes[read_id].evidence_ids)))
        edges[edge_id] = CausalEdge(
            edge_id, source, read_id, CausalEdgeKind.READ_FROM_ATTRIBUTE, True, evidence
        )

    # Resolve ``settings.NAME`` to a unique module-level NAME assignment.  The
    # edge is deterministic only because ambiguous definitions are ignored.
    config_by_name: dict[str, list[str]] = {}
    for node in nodes.values():
        if node.kind is CausalNodeKind.CONFIG_VALUE and node.symbol:
            config_by_name.setdefault(node.symbol, []).append(node.id)
    for node in tuple(nodes.values()):
        if node.kind is CausalNodeKind.CONFIG_VALUE or not node.symbol:
            continue
        matches = config_by_name.get(node.symbol.rsplit(".", 1)[-1], ())
        if len(matches) != 1:
            continue
        source = matches[0]
        edge_id = _stable("G", source, node.id, CausalEdgeKind.CONFIGURED_BY.value)
        evidence = tuple(dict.fromkeys((*nodes[source].evidence_ids, *node.evidence_ids)))
        edges[edge_id] = CausalEdge(
            edge_id, source, node.id, CausalEdgeKind.CONFIGURED_BY, True, evidence
        )

    failure_path, failure_line = _traceback(context.problem)
    symbolic_failure_nodes: list[CausalNode] = []
    if failure_path is None:
        folded_problem = context.problem.casefold()
        import_failure = "importerror" in folded_problem or "modulenotfounderror" in folded_problem or "import cycle" in folded_problem
        symbolic_failure_nodes = [
            node for node in nodes.values()
            if node.symbol
            and re.search(rf"\b{re.escape(node.symbol.rsplit('.', 1)[-1].casefold())}\b", folded_problem)
            and node.kind in (
                {CausalNodeKind.IMPORT}
                if import_failure
                else {CausalNodeKind.RETURN_VALUE, CausalNodeKind.CONDITION}
            )
        ]
        if symbolic_failure_nodes:
            symbolic_failure_nodes.sort(key=lambda item: (item.path, item.line, item.id))
            failure_path, failure_line = symbolic_failure_nodes[0].path, symbolic_failure_nodes[0].line
    failure_id: str | None = None
    if failure_path and failure_line:
        failure_id = _stable("N", CausalNodeKind.EXCEPTION_SITE.value, failure_path, failure_line)
        nodes[failure_id] = CausalNode(
            failure_id, CausalNodeKind.EXCEPTION_SITE, failure_path, failure_line,
            None, f"exception at {failure_path}:{failure_line}",
            _evidence_at(context.evidence_ledger, failure_path, failure_line),
        )
        candidates_at_site = symbolic_failure_nodes or [
            node for node in nodes.values()
            if node.path == failure_path and node.line == failure_line
            and node.id != failure_id
        ]
        for node in candidates_at_site:
            edge_id = _stable("G", node.id, failure_id, CausalEdgeKind.RAISES_AT.value)
            edges[edge_id] = CausalEdge(edge_id, node.id, failure_id, CausalEdgeKind.RAISES_AT, True, tuple(dict.fromkeys((*node.evidence_ids, *nodes[failure_id].evidence_ids))))

    causal_slice = CausalSlice(
        failure_id,
        tuple(sorted(nodes.values(), key=lambda item: item.id)),
        tuple(sorted(edges.values(), key=lambda item: item.id)),
        unknown_flow=failure_id is None,
    )
    if failure_id is None:
        return causal_slice, ()

    incoming: dict[str, list[CausalEdge]] = {}
    for edge in edges.values():
        incoming.setdefault(edge.target_id, []).append(edge)
    paths_to_failure: dict[str, tuple[str, ...]] = {failure_id: (failure_id,)}
    queue = [failure_id]
    while queue:
        target = queue.pop(0)
        if len(paths_to_failure[target]) >= 8:
            continue
        for edge in sorted(incoming.get(target, ()), key=lambda item: item.id):
            if edge.source_id in paths_to_failure:
                continue
            paths_to_failure[edge.source_id] = (edge.source_id, *paths_to_failure[target])
            queue.append(edge.source_id)

    root_candidates: list[RootCauseCandidate] = []
    for node_id, path_ids in paths_to_failure.items():
        if node_id == failure_id:
            continue
        node = nodes[node_id]
        if node.kind in {CausalNodeKind.CALL, CausalNodeKind.EXCEPTION_SITE}:
            continue
        evidence_ids = tuple(dict.fromkeys(
            evidence_id for path_node_id in path_ids for evidence_id in nodes[path_node_id].evidence_ids
        ))
        if not evidence_ids:
            continue
        distance = len(path_ids) - 1
        direct = sum(
            context.evidence_ledger.get(item).strength
            for item in node.evidence_ids if context.evidence_ledger.get(item)
        ) / max(1, len(node.evidence_ids))
        deterministic_edges = 0
        for left, right in zip(path_ids, path_ids[1:]):
            deterministic_edges += any(edge.source_id == left and edge.target_id == right and edge.deterministic for edge in edges.values())
        structural = deterministic_edges / max(1, distance)
        completeness = 1.0 if path_ids[-1] == failure_id else 0.0
        kind = _node_kind_for_cause(node, context.problem)
        is_test_input = (
            node.kind is CausalNodeKind.LITERAL
            and (node.path.startswith("tests/") or "/test" in node.path)
        )
        test_origin_penalty = (
            0.20
            if (node.path.startswith("tests/") or "/test" in node.path)
            and not (failure_path.startswith("tests/") or "/test" in failure_path)
            else 0.0
        )
        source_prior = {
            RootCauseKind.NULL_FLOW: 1.0,
            RootCauseKind.CONFIGURATION: 1.0,
            RootCauseKind.CONTROL_FLOW: 0.9,
            RootCauseKind.TYPE_FLOW: 0.9,
            RootCauseKind.RETURN_CONTRACT: 0.75,
            RootCauseKind.TEST_EXPECTATION: 0.55,
            RootCauseKind.ARGUMENT_BINDING: 0.45,
            RootCauseKind.STATE_PROPAGATION: 0.35,
            RootCauseKind.ATTRIBUTE_VALUE: 0.25,
            RootCauseKind.IMPORT_RESOLUTION: 0.8,
            RootCauseKind.UNKNOWN: 0.0,
        }[kind]
        upstream = node.path != failure_path
        explicit_return_contract = bool(
            kind is RootCauseKind.RETURN_CONTRACT
            and node.path == failure_path
            and node.line == failure_line
            and re.search(r"\breturn(?:s|ed|ing)?\b", context.problem, re.IGNORECASE)
            and re.search(r"\b(?:expect(?:s|ed)?|contract|caller)\b", context.problem, re.IGNORECASE)
        )
        failure_manifestation = (
            node.path == failure_path
            and node.line == failure_line
            and kind in {
                RootCauseKind.RETURN_CONTRACT,
                RootCauseKind.ATTRIBUTE_VALUE,
                RootCauseKind.STATE_PROPAGATION,
            }
            and not explicit_return_contract
        )
        score = round(min(
            1.0,
            direct * 0.20
            + structural * 0.20
            + completeness * 0.15
            + source_prior * (0.08 if is_test_input else 0.25)
            + (0.10 if upstream else 0.0)
            + min(distance, 8) / 8 * 0.10
            + (0.20 if explicit_return_contract else 0.0)
            - (0.20 if failure_manifestation else 0.0)
            - test_origin_penalty,
        ), 6)
        flow = " -> ".join(nodes[path_node].expression for path_node in path_ids[:7])
        statement = (
            f"{node.expression} at {node.path}:{node.line} flows to the failure site: {flow}"
        )
        root_candidates.append(RootCauseCandidate(
            _stable("R", kind.value, node.path, node.line, node.symbol, path_ids),
            kind, node.path, node.symbol, node.line, failure_path, failure_line,
            path_ids, evidence_ids, round(direct, 4), round(structural, 4),
            distance, completeness, score, statement,
        ))
    ordered = sorted(
        root_candidates,
        key=lambda item: (-item.score, item.causal_distance, item.origin_path, item.origin_line, item.id),
    )
    return causal_slice, tuple(ordered[:8])


def structurally_dominant_root(
    candidates: Sequence[RootCauseCandidate], problem: str
) -> RootCauseCandidate | None:
    """Return a root only when deterministic evidence makes alternatives weaker."""
    explicit_contract = [
        item for item in candidates
        if item.cause_kind is RootCauseKind.RETURN_CONTRACT
        and item.origin_path == item.failure_path
        and item.origin_line == item.failure_line
        and re.search(r"\breturn(?:s|ed|ing)?\b", problem, re.IGNORECASE)
        and re.search(r"\b(?:expect(?:s|ed)?|contract|caller)\b", problem, re.IGNORECASE)
    ]
    if len(explicit_contract) == 1:
        return explicit_contract[0]
    strong_origins = [
        item for item in candidates
        if item.origin_path != item.failure_path
        and item.cause_kind in {
            RootCauseKind.NULL_FLOW,
            RootCauseKind.CONFIGURATION,
        }
        and item.direct_support >= 0.8
        and item.structural_support == 1.0
        and item.completeness == 1.0
    ]
    return strong_origins[0] if len(strong_origins) == 1 else None


def repair_locality(
    root: RootCauseCandidate,
    target_files: Iterable[str],
    target_symbols: Iterable[str],
    causal_slice: CausalSlice,
) -> float:
    files = set(target_files)
    symbols = {item.rsplit(".", 1)[-1] for item in target_symbols}
    if root.origin_path in files and (not root.origin_symbol or root.origin_symbol.rsplit(".", 1)[-1] in symbols):
        return 1.0
    path_files = {node.path for node in causal_slice.nodes if node.id in root.causal_path}
    if files & path_files:
        return 0.70
    if root.failure_path in files:
        return 0.50
    return 0.0


def repair_target_compatible(
    root: RootCauseCandidate,
    strategy: RepairStrategyKind,
    target_file: str,
    target_symbol: str | None,
    causal_slice: CausalSlice,
) -> bool:
    path_nodes = [node for node in causal_slice.nodes if node.id in root.causal_path]
    if (
        strategy is RepairStrategyKind.CORRECT_RETURN_VALUE
        and not any(
            node.path == root.origin_path
            and node.kind is CausalNodeKind.RETURN_VALUE
            for node in path_nodes
        )
    ):
        return False
    path_files = {node.path for node in path_nodes}
    origin_strategies = {
        RepairStrategyKind.FIX_PRODUCER,
        RepairStrategyKind.CORRECT_RETURN_VALUE,
        RepairStrategyKind.CORRECT_CONFIGURATION,
        RepairStrategyKind.CORRECT_CONTROL_FLOW,
        RepairStrategyKind.CORRECT_IMPORT,
        RepairStrategyKind.CORRECT_TEST_EXPECTATION,
    }
    if strategy in origin_strategies and target_file != root.origin_path:
        return False
    if target_file not in path_files:
        return False
    if not target_symbol:
        return True
    path_symbols = {
        node.symbol
        for node in path_nodes
        if node.path == target_file and node.symbol
    }
    return target_symbol in path_symbols or any(
        symbol.endswith(f".{target_symbol}") or target_symbol.endswith(f".{symbol}")
        for symbol in path_symbols
    )
