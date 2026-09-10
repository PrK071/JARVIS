from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..project_intelligence_v2 import ProjectSnapshotV2
from ..security import PathPolicy
from .models import (
    ClaimSupport,
    EvidenceAtom,
    EvidenceKind,
    EvidenceLedger,
    HypothesisClaim,
    ProblemContext,
)


_STRENGTH = {"HARD": 1.0, "STRONG": 0.8, "SUPPORTING": 0.55, "SEMANTIC": 0.3}
_STRUCTURAL_KINDS = {
    EvidenceKind.CALL_RELATION,
    EvidenceKind.IMPORT_RELATION,
    EvidenceKind.TEST_RELATION,
    EvidenceKind.STRUCTURAL_RELATION,
    EvidenceKind.GIT_CHANGE,
}


def _atom_id(
    kind: EvidenceKind,
    path: str,
    start: int,
    end: int,
    statement: str,
) -> str:
    material = f"{kind.value}\0{path}\0{start}\0{end}\0{statement}".encode("utf-8")
    return "E" + hashlib.sha256(material).hexdigest()[:12].upper()


def _atom(
    kind: EvidenceKind,
    path: str,
    start: int,
    end: int,
    statement: str,
    strength: float,
    *,
    symbol: str | None = None,
    relation: str | None = None,
    snippet: str = "",
) -> EvidenceAtom:
    return EvidenceAtom(
        _atom_id(kind, path, start, end, statement),
        kind,
        path,
        start,
        end,
        statement,
        strength,
        symbol,
        relation,
        snippet[:320],
    )


def _name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return "expression"


def _safe_expression(node: ast.AST | None) -> str:
    if node is None:
        return "None"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return f"string literal of length {len(node.value)}"
        return repr(node.value)
    try:
        return ast.unparse(node)[:160]
    except Exception:
        return type(node).__name__


@dataclass
class _FactVisitor(ast.NodeVisitor):
    path: str
    source: str
    allowed_lines: set[int]
    atoms: list[EvidenceAtom]
    owner: str | None = None

    def _inside(self, node: ast.AST) -> bool:
        return int(getattr(node, "lineno", 0)) in self.allowed_lines

    def _snippet(self, node: ast.AST) -> str:
        return (ast.get_source_segment(self.source, node) or "").strip()[:320]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous = self.owner
        self.owner = node.name
        self.generic_visit(node)
        self.owner = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self._inside(node):
            target = _name(node.func)
            statement = (
                f"{self.owner or 'module'} calls {target} with "
                f"{len(node.args)} positional and {len(node.keywords)} keyword arguments"
            )
            self.atoms.append(
                _atom(
                    EvidenceKind.CALL_RELATION,
                    self.path,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    statement,
                    0.8,
                    symbol=self.owner,
                    relation=target,
                    snippet=self._snippet(node),
                )
            )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self._inside(node):
            statement = f"{self.owner or 'module'} returns {_safe_expression(node.value)}"
            self.atoms.append(
                _atom(
                    EvidenceKind.RETURN_STATEMENT,
                    self.path,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    statement,
                    0.8,
                    symbol=self.owner,
                    relation="returns",
                    snippet=self._snippet(node),
                )
            )
        self.generic_visit(node)

    def _assignment(self, node: ast.Assign | ast.AnnAssign, targets: Sequence[ast.AST]) -> None:
        if not self._inside(node):
            return
        value = node.value
        names = tuple(_name(target) for target in targets)
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and not any(
            name.isupper() for name in names
        ):
            return
        for name in names:
            statement = f"{name} is assigned {_safe_expression(value)}"
            kind = EvidenceKind.CONFIG_REFERENCE if name.isupper() else EvidenceKind.ASSIGNMENT
            self.atoms.append(
                _atom(
                    kind,
                    self.path,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    statement,
                    0.8,
                    symbol=name,
                    relation="assigned",
                    snippet=self._snippet(node),
                )
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        self._assignment(node, node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assignment(node, (node.target,))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._inside(node):
            self.atoms.append(
                _atom(
                    EvidenceKind.CONDITIONAL,
                    self.path,
                    node.lineno,
                    getattr(node.test, "end_lineno", node.lineno),
                    f"{self.owner or 'module'} checks {_safe_expression(node.test)}",
                    0.8,
                    symbol=self.owner,
                    relation="condition",
                    snippet=self._snippet(node.test),
                )
            )
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if self._inside(node):
            for call in (item for item in ast.walk(node.test) if isinstance(item, ast.Call)):
                target = _name(call.func)
                self.atoms.append(
                    _atom(
                        EvidenceKind.STRUCTURAL_RELATION,
                        self.path,
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                        f"{self.owner or 'test'} asserts behavior involving {target}",
                        0.9,
                        symbol=self.owner,
                        relation=target,
                        snippet=self._snippet(node),
                    )
                )
        self.generic_visit(node)


def build_evidence_ledger(
    context: ProblemContext,
    snapshot: ProjectSnapshotV2,
    path_policy: PathPolicy,
) -> EvidenceLedger:
    atoms: list[EvidenceAtom] = []
    excerpts = {item.path: item for item in context.evidence}
    traceback_lines: dict[str, set[int]] = {}
    for match in re.finditer(r'(?:File\s+["\'](?P<file>[^"\']+)["\'],\s+line\s+(?P<line>\d+))|(?P<short>[^\s]+\.py):(?P<short_line>\d+)', context.problem):
        path = (match.group("file") or match.group("short")).replace("\\", "/")
        traceback_lines.setdefault(path, set()).add(int(match.group("line") or match.group("short_line")))

    for path, excerpt in sorted(excerpts.items()):
        resolved = path_policy.resolve(str(Path(snapshot.project_path) / path))
        source = resolved.read_text(encoding="utf-8", errors="replace")
        for line in sorted(traceback_lines.get(path, ())):
            if excerpt.start_line <= line <= excerpt.end_line:
                atoms.append(
                    _atom(
                        EvidenceKind.TRACEBACK_FRAME,
                        path,
                        line,
                        line,
                        f"the traceback reports execution at {path}:{line}",
                        1.0,
                        relation="traceback",
                    )
                )
        indexed = snapshot.file_index[path]
        for symbol in indexed.symbols:
            if excerpt.start_line <= symbol.line <= excerpt.end_line:
                atoms.append(
                    _atom(
                        EvidenceKind.SYMBOL_DEFINITION,
                        path,
                        symbol.line,
                        symbol.end_line,
                        f"{symbol.kind.value} {symbol.qualified_name} is defined in {path}",
                        0.8,
                        symbol=symbol.qualified_name,
                        relation="defines",
                    )
                )
        for record in indexed.imports:
            if excerpt.start_line <= record.line <= excerpt.end_line:
                imported = record.module or ",".join(record.names)
                atoms.append(
                    _atom(
                        EvidenceKind.IMPORT_RELATION,
                        path,
                        record.line,
                        record.line,
                        f"{path} imports {imported}",
                        0.8,
                        relation=imported,
                    )
                )
        if path.endswith(".py"):
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError:
                tree = None
            if tree is not None:
                _FactVisitor(
                    path,
                    source,
                    set(range(excerpt.start_line, excerpt.end_line + 1)),
                    atoms,
                ).visit(tree)

    for relation in snapshot.test_relationships:
        if relation.production_file in excerpts or relation.test_file in excerpts:
            target = excerpts.get(relation.test_file) or excerpts.get(relation.production_file)
            if target:
                atoms.append(
                    _atom(
                        EvidenceKind.TEST_RELATION,
                        target.path,
                        target.start_line,
                        target.end_line,
                        f"{relation.test_file} is structurally related to {relation.production_file}",
                        0.7 if relation.strength.value == "STRONG" else 0.55,
                        relation=relation.production_file,
                    )
                )

    # Reduce related tests to structural call/assertion facts. This separates a
    # merely related file from a test that deterministically exercises a symbol,
    # without exposing comments, docstrings, or arbitrary string contents.
    for test_path in sorted(context.related_tests):
        if test_path in excerpts or not test_path.endswith(".py"):
            continue
        resolved = path_policy.resolve(str(Path(snapshot.project_path) / test_path))
        source = resolved.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=test_path)
        except SyntaxError:
            continue
        test_atoms: list[EvidenceAtom] = []
        _FactVisitor(
            test_path,
            source,
            set(range(1, len(source.splitlines()) + 1)),
            test_atoms,
        ).visit(tree)
        atoms.extend(
            atom
            for atom in test_atoms
            if atom.kind in {EvidenceKind.CALL_RELATION, EvidenceKind.STRUCTURAL_RELATION}
        )

    unique = {atom.id: atom for atom in atoms}
    return EvidenceLedger(tuple(sorted(unique.values(), key=lambda item: item.id)))


def ground_claim(
    statement: str,
    evidence_ids: Iterable[str],
    claim_type: str,
    ledger: EvidenceLedger,
) -> HypothesisClaim:
    ids = tuple(dict.fromkeys(str(value) for value in evidence_ids))
    atoms = [ledger.get(atom_id) for atom_id in ids]
    if not ids or any(atom is None for atom in atoms):
        return HypothesisClaim(statement, ids, ClaimSupport.UNSUPPORTED)
    concrete = [atom for atom in atoms if atom is not None]
    if claim_type == "INFERENCE":
        return HypothesisClaim(statement, ids, ClaimSupport.INFERRED)
    canonical = "; ".join(atom.statement for atom in concrete)
    support = (
        ClaimSupport.STRUCTURAL
        if all(atom.kind in _STRUCTURAL_KINDS for atom in concrete)
        else ClaimSupport.DIRECT
    )
    return HypothesisClaim(canonical, ids, support)


def validate_claim_mapping(claims: Sequence[HypothesisClaim], ledger: EvidenceLedger) -> bool:
    return bool(claims) and all(
        claim.support is not ClaimSupport.UNSUPPORTED
        and claim.evidence_ids
        and all(ledger.get(atom_id) is not None for atom_id in claim.evidence_ids)
        for claim in claims
    )
