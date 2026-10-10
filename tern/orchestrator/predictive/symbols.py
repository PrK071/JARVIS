"""Bounded AST symbol context. Selection is observation, never defect proof."""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ..project_intelligence_v2 import SymbolKind
from .models import EvidenceExcerpt


@dataclass(frozen=True)
class SymbolRecoveryBudget:
    max_attempts: int = 1
    max_extra_files: int = 2
    max_extra_symbols: int = 16
    max_structural_depth: int = 2
    max_symbol_lines: int = 240
    max_context_bytes: int = 48_000

    def __post_init__(self):
        if any(value < 0 for value in self.__dict__.values()) or self.max_attempts > 1:
            raise ValueError("Orcamento de simbolos invalido.")


@dataclass(frozen=True)
class RelevantSymbolContext:
    symbol_id: str
    path: str
    qualified_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    selection_reason: str
    structural_anchors: tuple[str, ...]
    related_calls: tuple[str, ...] = ()
    related_returns: tuple[int, ...] = ()
    related_assignments: tuple[int, ...] = ()
    contract_evidence: tuple[str, ...] = ()
    depth: int = 0

    def __post_init__(self):
        if (not self.symbol_id or not self.path or not self.qualified_name
                or self.start_line < 1 or self.end_line < self.start_line
                or self.depth < 0 or not self.structural_anchors):
            raise ValueError("Contexto de simbolo sem origem verificavel.")

    def as_dict(self):
        return dict(self.__dict__, structural_anchors=list(self.structural_anchors),
                    related_calls=list(self.related_calls), related_returns=list(self.related_returns),
                    related_assignments=list(self.related_assignments), contract_evidence=list(self.contract_evidence),
                    status="SYMBOL_RETRIEVED_NOT_PROVEN_DEFECTIVE")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _body_nodes(node):
    # Do not inherit the body/contract of an unrelated nested callable.
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _body_nodes(child)


def _reported_identifiers(problem):
    result = set()
    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?![\w.])", problem):
        token = match.group(1)
        if "." in token or "_" in token or re.match(r"\s*\(", problem[match.end():]):
            result.add(token)
        elif f"`{token}`" in problem:
            result.add(token)
    return result


def recover_symbol_context(context, snapshot, policy, *, budget=None):
    """One incremental recovery, no reindexing or lexical file search."""
    budget = budget or SymbolRecoveryBudget()
    if not budget.max_attempts or not budget.max_extra_symbols:
        return context
    reported = _reported_identifiers(context.problem)
    selected_paths = set(context.related_files)
    initial_paths = set(selected_paths)
    known = snapshot.file_index
    records = [symbol for path in sorted(selected_paths) for symbol in known[path].symbols
               if symbol.kind in {SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION,
                                  SymbolKind.METHOD, SymbolKind.ASYNC_METHOD}]
    parsed = {}

    def tree(path):
        if path not in parsed:
            source = policy.resolve(str(Path(snapshot.project_path) / path)).read_text(encoding="utf-8", errors="replace")
            try:
                parsed[path] = (source, ast.parse(source))
            except SyntaxError:
                parsed[path] = (source, None)
        return parsed[path]

    def node_for(symbol):
        _, module = tree(symbol.file)
        if module is None:
            return None
        return next((node for node in ast.walk(module)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.lineno == symbol.line), None)

    def resolved_callees(symbol, node, call):
        _, module = tree(symbol.file)
        names = call.split(".")
        local_bound = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                           *node.args.kwonlyargs)}
        local_bound.update(child.id for child in _body_nodes(node)
                           if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store))
        if names[0] in local_bound:
            return []
        import_bindings = []
        for statement in module.body:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                for alias in statement.names:
                    if len(names) == 1 and names[0] == (alias.asname or alias.name):
                        import_bindings.append((statement.module, alias.name))
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if names[0] == bound and len(names) == 2:
                        import_bindings.append((alias.name, names[1]))
        candidates = []
        if import_bindings:
            for imported_module, imported_name in import_bindings:
                candidates.extend(item for item in by_name.get(imported_name, ())
                                  if known[item.file].module == imported_module
                                  and item.file in snapshot.import_graph.get(symbol.file, ()))
        elif len(names) == 1:
            candidates.extend(item for item in by_name.get(call, ()) if item.file == symbol.file)
        return [item for item in candidates
                if item.kind in {SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION}]

    selections = {}
    queue = []
    for symbol in records:
        qualified = symbol.qualified_name
        anchors = sorted(token for token in reported if token == qualified
                         or token.endswith("." + qualified) or token == symbol.name)
        if anchors:
            selections[(symbol.file, qualified, symbol.line)] = (symbol, "REPORTED_API", tuple(anchors), 0)
            queue.append(symbol)
    # A bare receiver.method is ambiguous: retain all indexed matches in the
    # already retrieved scope; no name is allowed to prove responsibility.
    for symbol in records:
        if (symbol.file, symbol.qualified_name, symbol.line) in selections:
            continue
        anchors = sorted(token for token in reported if "." in token
                         and token.rsplit(".", 1)[-1] == symbol.name)
        if anchors:
            selections[(symbol.file, symbol.qualified_name, symbol.line)] = (symbol, "REPORTED_API_AMBIGUOUS", tuple(anchors), 0)
            queue.append(symbol)
    # Exact structural reads can anchor a configuration behavior even when its
    # private helper is not named in the report. Comments/docstrings do not count.
    for symbol in records:
        if (symbol.file, symbol.qualified_name, symbol.line) in selections:
            continue
        node = node_for(symbol)
        if node is None:
            continue
        reads = {_call_name(child) for child in _body_nodes(node)
                 if isinstance(child, (ast.Name, ast.Attribute)) and isinstance(child.ctx, ast.Load)}
        anchors = sorted(token for token in reported if token in reads)
        if anchors:
            selections[(symbol.file, symbol.qualified_name, symbol.line)] = (symbol, "REPORTED_VALUE_READ", tuple(anchors), 0)
            queue.append(symbol)

    # Only a uniquely resolved call through the existing module import graph
    # may add a file. Dynamic receiver dispatch remains unresolved.
    by_name = snapshot.symbol_index
    position = 0
    while position < len(queue) and len(selections) < budget.max_extra_symbols:
        symbol = queue[position]
        position += 1
        depth = selections[(symbol.file, symbol.qualified_name, symbol.line)][3]
        if depth >= budget.max_structural_depth:
            continue
        node = node_for(symbol)
        if node is None:
            continue
        calls = sorted({_call_name(child.func) for child in _body_nodes(node) if isinstance(child, ast.Call)})
        for call in calls:
            possible = resolved_callees(symbol, node, call)
            if len(possible) != 1:
                continue
            candidate = possible[0]
            key = (candidate.file, candidate.qualified_name, candidate.line)
            if key in selections:
                continue
            if candidate.file not in selected_paths:
                if len(selected_paths - initial_paths) >= budget.max_extra_files:
                    continue
                selected_paths.add(candidate.file)
            selections[key] = (candidate, "RESOLVED_CALL_NEIGHBOR", (f"{symbol.file}::{symbol.qualified_name}->{call}",), depth + 1)
            queue.append(candidate)
            if len(selections) >= budget.max_extra_symbols:
                break

    windows = list(context.evidence)
    contexts = []
    used_bytes = 0
    # Prioritize direct reported APIs over structural reads and their callees.
    ordered = sorted(selections.values(), key=lambda item: (
        item[3], 0 if item[1].startswith("REPORTED_API") else 1,
        item[0].file, item[0].line, item[0].qualified_name))
    for symbol, reason, anchors, depth in ordered[:budget.max_extra_symbols]:
        node = node_for(symbol)
        if node is None:
            continue
        source, _ = tree(symbol.file)
        lines = source.splitlines()
        end = min(symbol.end_line, symbol.line + budget.max_symbol_lines - 1)
        excerpt = "\n".join(f"{line}: {lines[line-1]}" for line in range(symbol.line, end + 1))
        size = len(excerpt.encode("utf-8"))
        if used_bytes + size > budget.max_context_bytes:
            continue
        used_bytes += size
        parts = [child for child in _body_nodes(node)
                 if symbol.line <= getattr(child, "lineno", 0) <= end]
        sid = "S" + hashlib.sha256(f"{symbol.file}\0{symbol.qualified_name}\0{symbol.line}".encode()).hexdigest()[:12]
        contracts = ("AST_RETURN_ANNOTATION",) if node.returns else ()
        contexts.append(RelevantSymbolContext(
            sid, symbol.file, symbol.qualified_name, symbol.kind.value, symbol.line, end, reason, anchors,
            tuple(sorted({_call_name(child.func) for child in parts if isinstance(child, ast.Call)})),
            tuple(sorted({child.lineno for child in parts if isinstance(child, ast.Return)})),
            tuple(sorted({child.lineno for child in parts if isinstance(child, (ast.Assign, ast.AnnAssign))})), contracts, depth))
        if not any(item.path == symbol.file and item.start_line <= symbol.line and item.end_line >= end for item in windows):
            windows.append(EvidenceExcerpt(f"{symbol.file}:{symbol.line}-{end}", symbol.file, symbol.line, end,
                                          "STRONG", (reason,), excerpt))
    actual_paths = set(initial_paths) | {item.path for item in contexts}
    return replace(context, evidence=tuple(windows),
                   related_files=tuple(dict.fromkeys((*context.related_files, *sorted(actual_paths - initial_paths)))),
                   related_symbols=tuple(dict.fromkeys((*context.related_symbols,
                       *(f"{item.qualified_name}@{item.path}:{item.start_line}-{item.end_line}" for item in contexts)))),
                   relevant_symbol_contexts=tuple(contexts))
