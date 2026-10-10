"""Caminhos AST limitados. Alcancabilidade nunca demonstra responsabilidade."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from ..project_intelligence_v2 import SymbolKind
from .models import EvidenceExcerpt, EvidenceKind
from .symbols import RelevantSymbolContext, _body_nodes, _call_name, _reported_identifiers


class RelationSupport(str, Enum):
    AST_DEMONSTRATED = "AST_DEMONSTRATED"
    STATICALLY_RESOLVED = "STATICALLY_RESOLVED"
    INFERRED_FROM_OBSERVATION = "INFERRED_FROM_OBSERVATION"
    UNRESOLVED = "UNRESOLVED"


class EffectKind(str, Enum):
    RETURNS_VALUE = "RETURNS_VALUE"
    WRITES_ATTRIBUTE = "WRITES_ATTRIBUTE"
    MUTATES_CONTAINER = "MUTATES_CONTAINER"
    YIELDS_VALUE = "YIELDS_VALUE"
    RAISES_EXCEPTION = "RAISES_EXCEPTION"
    HANDLES_EXCEPTION = "HANDLES_EXCEPTION"


@dataclass(frozen=True)
class ImplementationBudget:
    max_depth: int = 2
    max_extra_files: int = 2
    max_symbols: int = 16
    max_context_bytes: int = 48_000
    max_symbol_lines: int = 240
    max_source_bytes: int = 4_000_000

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in self.__dict__.values()):
            raise ValueError("Orcamento de implementacao invalido.")


def _id(prefix, *parts):
    return prefix + hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ResolvedCallable:
    id: str
    path: str
    qualified_name: str
    owner: str | None
    module: str
    start_line: int
    end_line: int
    depth: int

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class ImplementationRelation:
    id: str
    source_id: str
    target_id: str
    path: str
    line: int
    end_line: int
    call_expression: str
    resolution: str
    support: RelationSupport = RelationSupport.STATICALLY_RESOLVED
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self):
        return dict(self.__dict__, support=self.support.value, evidence_ids=list(self.evidence_ids),
                    status="REACHABLE_NOT_PROVEN_DEFECTIVE")


@dataclass(frozen=True)
class BehavioralEffect:
    id: str
    callable_id: str
    kind: EffectKind
    path: str
    owner: str
    line: int
    end_line: int
    subject: str | None
    support: RelationSupport = RelationSupport.AST_DEMONSTRATED
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self):
        return dict(self.__dict__, kind=self.kind.value, support=self.support.value,
                    evidence_ids=list(self.evidence_ids), status="EFFECT_OBSERVED_NOT_PROVEN_INCORRECT")


@dataclass(frozen=True)
class UnresolvedDispatch:
    source_id: str
    path: str
    line: int
    expression: str
    reason: str

    def as_dict(self):
        return dict(self.__dict__, support=RelationSupport.UNRESOLVED.value)


@dataclass(frozen=True)
class BehavioralGraph:
    entry_ids: tuple[str, ...] = ()
    callables: tuple[ResolvedCallable, ...] = ()
    relations: tuple[ImplementationRelation, ...] = ()
    effects: tuple[BehavioralEffect, ...] = ()
    unresolved: tuple[UnresolvedDispatch, ...] = ()
    reason_codes: tuple[str, ...] = ()
    initial_windows: tuple[tuple[str, int, int], ...] = ()
    added_context_bytes: int = 0

    def __post_init__(self):
        ids = {item.id for item in self.callables}
        if (len(ids) != len(self.callables) or not set(self.entry_ids) <= ids
                or any(r.source_id not in ids or r.target_id not in ids for r in self.relations)
                or any(e.callable_id not in ids for e in self.effects)):
            raise ValueError("Grafo comportamental com identidades inconsistentes.")

    def as_dict(self):
        return {"version": 1, "entry_ids": list(self.entry_ids),
            "callables": [item.as_dict() for item in self.callables],
            "relations": [item.as_dict() for item in self.relations],
            "effects": [item.as_dict() for item in self.effects],
            "unresolved_dispatch": [item.as_dict() for item in self.unresolved],
            "reason_codes": list(self.reason_codes), "initial_windows": list(self.initial_windows),
            "added_context_bytes": self.added_context_bytes,
            "sufficiency": {"observation_anchored": bool(self.entry_ids),
                "implementation_reached": bool(self.callables), "effect_connected": bool(self.effects),
                "contract_supported": False},
            "limitation": "AST paths/effects do not establish a violated behavioral contract."}


def _effects(callable_, node):
    for part in _body_nodes(node):
        kind, subject = None, None
        if isinstance(part, ast.Return):
            kind = EffectKind.RETURNS_VALUE
        elif isinstance(part, (ast.Yield, ast.YieldFrom)):
            kind = EffectKind.YIELDS_VALUE
        elif isinstance(part, ast.Raise):
            kind, subject = EffectKind.RAISES_EXCEPTION, _call_name(part.exc)
        elif isinstance(part, ast.ExceptHandler):
            kind, subject = EffectKind.HANDLES_EXCEPTION, _call_name(part.type)
        elif isinstance(part, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = part.targets if isinstance(part, ast.Assign) else (part.target,)
            for target in targets:
                target_kind = (EffectKind.WRITES_ATTRIBUTE if isinstance(target, ast.Attribute)
                               else EffectKind.MUTATES_CONTAINER if isinstance(target, ast.Subscript) else None)
                if target_kind:
                    yield BehavioralEffect(_id("BE", callable_.id, target_kind.value, part.lineno,
                        ast.dump(target)), callable_.id, target_kind, callable_.path,
                        callable_.qualified_name, part.lineno, part.end_lineno, _call_name(target))
        if kind:
            yield BehavioralEffect(_id("BE", callable_.id, kind.value, part.lineno), callable_.id,
                kind, callable_.path, callable_.qualified_name, part.lineno,
                getattr(part, "end_lineno", part.lineno), subject)


def resolve_behavioral_context(context, snapshot, policy, *, budget=None, materialize=True):
    """Uma travessia estatica; sem oracle, busca textual ampla ou reindexacao."""
    budget = budget or ImplementationBudget()
    reported = _reported_identifiers(context.problem)
    kinds = {SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION, SymbolKind.METHOD, SymbolKind.ASYNC_METHOD}
    original_paths = set(context.related_files)
    seeds = sorted((symbol for path in original_paths for symbol in snapshot.file_index[path].symbols
        if symbol.kind in kinds and any(token == symbol.qualified_name or token == symbol.name
            or token.endswith('.' + symbol.qualified_name) for token in reported)),
        key=lambda symbol: (symbol.file, symbol.line, symbol.qualified_name))
    if not seeds or not budget.max_symbols:
        return replace(context, behavioral_graph=BehavioralGraph(reason_codes=("NO_STRUCTURAL_API_ANCHOR",)))
    # An ambiguous API does not become a unique seed by sorting it.
    seed_keys = {(symbol.file, symbol.qualified_name) for symbol in seeds}
    parsed, reasons = {}, set()
    source_bytes = 0

    def module(path):
        nonlocal source_bytes
        if path not in parsed:
            resolved = policy.resolve(str(Path(snapshot.project_path) / path))
            size = resolved.stat().st_size
            if source_bytes + size > budget.max_source_bytes:
                reasons.add("BUDGET_EXHAUSTED")
                parsed[path] = ("", None, {}, {})
                return parsed[path]
            source_bytes += size
            source = resolved.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except (SyntaxError, RecursionError):
                parsed[path] = (source, None, {}, {})
                reasons.add("MISSING_IMPLEMENTATION_EDGE")
                return parsed[path]
            definitions = {(part.lineno, part.name): part for part in ast.walk(tree)
                           if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef))}
            classes = {part.name: part for part in tree.body if isinstance(part, ast.ClassDef)}
            parsed[path] = (source, tree, definitions, classes)
        return parsed[path]

    def target_for(symbol, node, call):
        source, tree, _, classes = module(symbol.file)
        expr = _call_name(call.func)
        receiver = call.func
        while isinstance(receiver, ast.Attribute):
            receiver = receiver.value
        if not isinstance(receiver, ast.Name):
            return None, "UNRESOLVED_RECEIVER"
        locals_ = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        locals_.update(part.id for part in _body_nodes(node)
                       if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store))
        def local_definitions(statements):
            for statement in statements:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    yield statement.name
                elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                    yield from (a.asname or a.name.split('.')[0] for a in statement.names)
                else:
                    yield from local_definitions(ast.iter_child_nodes(statement))
        locals_.update(local_definitions(node.body))
        if node.args.vararg:
            locals_.add(node.args.vararg.arg)
        if node.args.kwarg:
            locals_.add(node.args.kwarg.arg)
        if expr.startswith('self.') and expr.count('.') == 1:
            owner = symbol.qualified_name.rpartition('.')[0]
            cls = classes.get(owner)
            rebound = any(isinstance(part, ast.Name) and part.id == 'self' and isinstance(part.ctx, ast.Store)
                          for part in _body_nodes(node))
            if not cls or cls.bases or cls.decorator_list or node.decorator_list or rebound:
                return None, "UNRESOLVED_RECEIVER"
            member = expr.split('.')[1]
            overwritten = any(isinstance(part, ast.Attribute) and isinstance(part.ctx, ast.Store)
                and _call_name(part) == expr for part in ast.walk(cls))
            if overwritten or not node.args.args or node.args.args[0].arg != 'self':
                return None, "SHADOWED_OR_DYNAMIC_RECEIVER"
            qualified = owner + '.' + expr.split('.')[1]
            candidates = [s for s in snapshot.file_index[symbol.file].symbols if s.qualified_name == qualified]
            resolution = "LEXICAL_SELF_METHOD"
        elif expr.split('.')[0] in locals_:
            return None, "SHADOWED_OR_DYNAMIC_RECEIVER"
        else:
            candidates, bindings = [], []
            for statement in tree.body:
                if isinstance(statement, ast.ImportFrom):
                    for alias in statement.names:
                        if expr == (alias.asname or alias.name):
                            record = snapshot.file_index[symbol.file]
                            package = record.package.split('.') if record.package else []
                            if statement.level:
                                remove = statement.level - 1
                                base = '.'.join(package[:len(package)-remove]) if remove < len(package) else ''
                                imported = '.'.join(filter(None, (base, statement.module)))
                            else:
                                imported = statement.module or ''
                            bindings.append((imported, alias.name))
                elif isinstance(statement, ast.Import):
                    for alias in statement.names:
                        prefix = alias.asname or alias.name
                        if expr.startswith(prefix + '.'):
                            bindings.append((alias.name, expr[len(prefix)+1:]))
            if bindings:
                for imported, name in bindings:
                    paths = [p for p, record in snapshot.file_index.items()
                             if record.module == imported or record.module.endswith('.' + imported)]
                    if len(paths) == 1:
                        candidates.extend(s for s in snapshot.file_index[paths[0]].symbols
                                          if s.qualified_name == name and s.kind in kinds)
                resolution = "AST_IMPORTED_CALL"
            elif '.' not in expr:
                candidates = [s for s in snapshot.file_index[symbol.file].symbols
                              if s.qualified_name == expr and s.kind in kinds]
                resolution = "AST_LOCAL_CALL"
            else:
                return None, "UNRESOLVED_RECEIVER"
            head = expr.split('.')[0]
            if any(isinstance(part, ast.Name) and part.id == head and isinstance(part.ctx, ast.Store)
                   for statement in tree.body if not isinstance(statement,
                       (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for part in ast.walk(statement)):
                return None, "SHADOWED_OR_DYNAMIC_RECEIVER"
        if len(candidates) != 1:
            return None, "UNRESOLVED_DISPATCH"
        target = candidates[0]
        if target.file not in original_paths and target.file not in parsed:
            already_extra = len(set(parsed) - original_paths)
            if already_extra >= budget.max_extra_files:
                reasons.add("BUDGET_EXHAUSTED")
                return None, "BUDGET_EXHAUSTED"
        _, _, definitions, _ = module(target.file)
        target_node = definitions.get((target.line, target.name))
        if target_node is None or target_node.decorator_list:
            return None, "DECORATED_OR_MISSING_CALLABLE"
        return target, resolution

    windows, symbol_contexts = list(context.evidence), list(context.relevant_symbol_contexts)
    initial_bytes = sum(len(w.excerpt.encode()) for w in windows)
    if initial_bytes > budget.max_context_bytes:
        reasons.add("INITIAL_CONTEXT_OVER_BUDGET")
    added_bytes, added_paths = 0, set()
    callables, relations, effects, unresolved, entries = {}, {}, {}, [], []
    queue = [(s, 0) for s in seeds[:budget.max_symbols]]
    if len(seeds) > budget.max_symbols:
        reasons.add("BUDGET_EXHAUSTED")
    while queue:
        symbol, depth = queue.pop(0)
        key = (symbol.file, symbol.qualified_name)
        if key in callables:
            continue
        if len(callables) >= budget.max_symbols:
            reasons.add("BUDGET_EXHAUSTED")
            continue
        existing_symbol = any(s.path == symbol.file and s.qualified_name == symbol.qualified_name
                              for s in symbol_contexts)
        if not existing_symbol and len(symbol_contexts) >= budget.max_symbols:
            reasons.add("BUDGET_EXHAUSTED")
            continue
        if symbol.file not in original_paths and symbol.file not in added_paths:
            if len(added_paths) >= budget.max_extra_files:
                reasons.add("BUDGET_EXHAUSTED")
                continue
        source, _, definitions, _ = module(symbol.file)
        node = definitions.get((symbol.line, symbol.name))
        if node is None:
            continue
        end = min(symbol.end_line, symbol.line + budget.max_symbol_lines - 1)
        covered = any(w.path == symbol.file and w.start_line <= symbol.line and w.end_line >= end for w in windows)
        if not covered and materialize:
            lines = source.splitlines()
            excerpt = '\n'.join(f'{n}: {lines[n-1]}' for n in range(symbol.line, end+1))
            size = len(excerpt.encode())
            if initial_bytes + added_bytes + size > budget.max_context_bytes:
                reasons.add("BUDGET_EXHAUSTED")
                continue
            added_bytes += size
            windows.append(EvidenceExcerpt(f'{symbol.file}:{symbol.line}-{end}', symbol.file,
                symbol.line, end, 'STRONG', ('AST_IMPLEMENTATION_PATH',), excerpt))
        if symbol.file not in original_paths:
            added_paths.add(symbol.file)
        item = ResolvedCallable(_id('BC', symbol.file, symbol.qualified_name, symbol.line),
            symbol.file, symbol.qualified_name, symbol.qualified_name.rpartition('.')[0] or None,
            symbol.module, symbol.line, end, depth)
        callables[key] = item
        if key in seed_keys:
            entries.append(item.id)
        if not any(s.path == symbol.file and s.qualified_name == symbol.qualified_name for s in symbol_contexts):
            symbol_contexts.append(RelevantSymbolContext(item.id, item.path, item.qualified_name,
                symbol.kind.value, item.start_line, item.end_line,
                'REPORTED_API' if key in seed_keys else 'AST_IMPLEMENTATION_PATH',
                (f'{item.path}:{item.start_line}',), depth=depth))
        for effect in _effects(item, node):
            if effect.end_line <= end:
                effects[effect.id] = effect
        for call in (part for part in _body_nodes(node) if isinstance(part, ast.Call) and part.end_lineno <= end):
            target, resolution = target_for(symbol, node, call)
            if target is None:
                unresolved.append(UnresolvedDispatch(item.id, symbol.file, call.lineno,
                    _call_name(call.func), resolution))
                continue
            if depth >= budget.max_depth:
                reasons.add("BUDGET_EXHAUSTED")
                continue
            target_id = _id('BC', target.file, target.qualified_name, target.line)
            relation = ImplementationRelation(_id('BR', item.id, target_id, call.lineno),
                item.id, target_id, symbol.file, call.lineno, call.end_lineno,
                _call_name(call.func), resolution)
            relations[relation.id] = relation
            queue.append((target, depth+1))
    known_ids = {c.id for c in callables.values()}
    relations = {key: r for key, r in relations.items() if r.target_id in known_ids}
    if unresolved:
        reasons.add("UNRESOLVED_DISPATCH")
    if not effects:
        reasons.add("MISSING_BEHAVIORAL_EFFECT")
    reasons.add("NO_SUPPORTED_CONTRACT")
    graph = BehavioralGraph(tuple(sorted(entries)), tuple(sorted(callables.values(), key=lambda c: c.id)),
        tuple(sorted(relations.values(), key=lambda r: r.id)), tuple(sorted(effects.values(), key=lambda e: e.id)),
        tuple(sorted(unresolved, key=lambda u: (u.path, u.line, u.expression))), tuple(sorted(reasons)),
        tuple((w.path, w.start_line, w.end_line) for w in context.evidence), added_bytes)
    return replace(context, evidence=tuple(windows), relevant_symbol_contexts=tuple(symbol_contexts),
        related_files=tuple(dict.fromkeys((*context.related_files, *sorted(added_paths)))), behavioral_graph=graph)


def bind_graph_evidence(graph, ledger):
    """Identidade exata de fato/owner/site; nenhuma evidencia fallback por arquivo."""
    if graph is None:
        return None
    owners = {c.id: c.qualified_name for c in graph.callables}
    relations = tuple(replace(r, evidence_ids=tuple(a.id for a in ledger.for_path(r.path)
        if a.kind is EvidenceKind.CALL_RELATION and a.start_line == r.line
        and a.symbol == owners[r.source_id] and a.relation == r.call_expression)) for r in graph.relations)
    effects = tuple(replace(e, evidence_ids=tuple(a.id for a in ledger.for_path(e.path)
        if a.start_line == e.line and a.symbol == e.owner
        and ((e.kind is EffectKind.RETURNS_VALUE and a.kind is EvidenceKind.RETURN_STATEMENT)
             or (a.kind is EvidenceKind.BEHAVIORAL_EFFECT and a.relation == e.kind.value)))) for e in graph.effects)
    reasons = set(graph.reason_codes)
    if any(not r.evidence_ids for r in relations) or any(not e.evidence_ids for e in effects):
        reasons.add("MISSING_LEDGER_RELATION")
    return replace(graph, relations=relations, effects=effects, reason_codes=tuple(sorted(reasons)))
