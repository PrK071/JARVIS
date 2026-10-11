"""Expectativa, aplicabilidade e violacao sao verificacoes diferentes."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .behavioral import EffectKind, RelationSupport, _id
from .models import EvidenceKind
from .symbols import _body_nodes, _call_name


class ExpectationSource(str, Enum):
    USER_REPORTED_EXPECTATION = 'USER_REPORTED_EXPECTATION'
    TEST_ASSERTION = 'TEST_ASSERTION'
    STATIC_ANNOTATION = 'STATIC_ANNOTATION'
    SOURCE_GUARD = 'SOURCE_GUARD'
    EXPLICIT_CONTRACT = 'EXPLICIT_CONTRACT'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class EffectObservationLink:
    entry_id: str
    effect_id: str
    implementation_relation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    support: RelationSupport = RelationSupport.INFERRED_FROM_OBSERVATION
    condition: str = 'VERIFY_EXECUTION_AND_OBSERVATION'

    def as_dict(self):
        return {'entry_id': self.entry_id, 'effect_id': self.effect_id,
            'implementation_relation_ids': list(self.implementation_relation_ids),
            'evidence_ids': list(self.evidence_ids), 'support': self.support.value,
            'condition': self.condition, 'execution_demonstrated': False,
            'causal_responsibility_demonstrated': False}


@dataclass(frozen=True)
class EffectFlowRelation:
    kind: str
    source_effect_id: str
    consumer_id: str
    evidence_ids: tuple[str, ...]
    support: RelationSupport = RelationSupport.AST_DEMONSTRATED

    def as_dict(self):
        return dict(self.__dict__, evidence_ids=list(self.evidence_ids), support=self.support.value,
            condition='SOMENTE_SE_O_CAMINHO_EXECUTAR', responsibility_demonstrated=False)


@dataclass(frozen=True)
class BehavioralContractContext:
    id: str
    callable_id: str
    path: str
    symbol: str
    observation: str
    expected_behavior: str | None
    expectation_source: ExpectationSource
    evidence_ids: tuple[str, ...]
    links: tuple[EffectObservationLink, ...]
    contract_exists: bool
    contract_applies: bool
    violation_supported: bool
    return_site_line: int | None = None
    verification_reason: str = 'NO_SUPPORTED_CONTRACT_VIOLATION'
    effect_flow_relations: tuple[EffectFlowRelation, ...] = ()

    def __post_init__(self):
        if self.violation_supported and (not self.contract_exists or not self.contract_applies
                or self.expectation_source is not ExpectationSource.STATIC_ANNOTATION
                or not self.return_site_line or not self.evidence_ids):
            raise ValueError('Violacao sem contrato estatico verificavel.')

    def as_dict(self):
        return dict(self.__dict__, expectation_source=self.expectation_source.value,
            evidence_ids=list(self.evidence_ids), links=[link.as_dict() for link in self.links],
            effect_flow_relations=[r.as_dict() for r in self.effect_flow_relations],
            limitation='Aplicabilidade estatica nao prova execucao nem explica sozinha o sintoma.')


def _primitive_annotation(node, tree):
    annotation = node.returns
    name = annotation.id if isinstance(annotation, ast.Name) else (
        'None' if isinstance(annotation, ast.Constant) and annotation.value is None else None)
    if name not in {'int', 'float', 'str', 'bool', 'None'}:
        return None
    # A nomeacao de builtins pode ser alterada; nao inferir tipos nesse caso.
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if statement.name == name:
                return None
            continue
        if any(isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store) and part.id == name
               for part in ast.walk(statement)):
            return None
        if isinstance(statement, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split('.')[0]) == name or alias.name == '*'
                for alias in statement.names):
            return None
    return name


def _literal_return_violation(node, tree):
    """Subconjunto deliberado: retorno literal incondicional, sem dispatch."""
    annotation = _primitive_annotation(node, tree)
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if (annotation is None or node.decorator_list or isinstance(node, ast.AsyncFunctionDef)
            or len(body) != 1 or not isinstance(body[0], ast.Return)
            or not isinstance(body[0].value, ast.Constant)):
        return None, None
    returned = body[0]
    actual = type(returned.value.value)
    allowed = {'int': (int, bool), 'float': (int, float, bool), 'str': (str,),
        'bool': (bool,), 'None': (type(None),)}[annotation]
    return actual not in allowed, returned.lineno


def _implementation_paths(graph):
    """Arestas verificadas apenas: nenhuma proximidade textual entre simbolos."""
    result = {}
    relations = sorted(graph.relations, key=lambda r: r.id)
    for entry in sorted(graph.entry_ids):
        queue = [(entry, ())]
        seen = set()
        while queue:
            current, path = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result[(entry, current)] = path
            for relation in relations:
                if (relation.source_id == current and relation.evidence_ids
                        and relation.support is RelationSupport.STATICALLY_RESOLVED):
                    queue.append((relation.target_id, (*path, relation)))
    return result


def _effect_flows(callable_, node, graph, ledger):
    result = []
    own_effects = [e for e in graph.effects if e.callable_id == callable_.id and e.evidence_ids]
    for effect in own_effects:
        if effect.kind is not EffectKind.RETURNS_VALUE:
            continue
        returned = next((n for n in _body_nodes(node) if isinstance(n, ast.Return)
                         and n.lineno == effect.line), None)
        if returned is None or not isinstance(returned.value, ast.Call):
            continue
        for relation in graph.relations:
            if (relation.source_id == callable_.id and relation.line == returned.value.lineno
                    and relation.call_expression == _call_name(returned.value.func)
                    and relation.evidence_ids):
                for producer in graph.effects:
                    if (producer.callable_id == relation.target_id
                            and producer.kind is EffectKind.RETURNS_VALUE and producer.evidence_ids):
                        ids = tuple(dict.fromkeys((*producer.evidence_ids, *relation.evidence_ids,
                                                  *effect.evidence_ids)))
                        result.append(EffectFlowRelation('RETURN_CONSUMED_BY_API', producer.id,
                            effect.id, ids))
    for statement in _body_nodes(node):
        if not isinstance(statement, ast.If):
            continue
        guards = tuple(a.id for a in ledger.for_path(callable_.path)
            if a.kind is EvidenceKind.CONDITIONAL and a.start_line == statement.lineno
            and a.symbol == callable_.qualified_name)
        if not guards:
            continue
        guarded_lines = {n.lineno for child in (*statement.body, *statement.orelse)
            for n in ast.walk(child) if hasattr(n, 'lineno')}
        for effect in own_effects:
            if effect.line in guarded_lines:
                result.append(EffectFlowRelation('CONDITION_GOVERNS_EFFECT', effect.id,
                    guards[0], tuple(dict.fromkeys((*guards, *effect.evidence_ids)))))
    return tuple(result)


def build_contract_contexts(context, policy):
    graph = context.behavioral_graph
    if not graph or not graph.entry_ids:
        return ()
    ledger = context.evidence_ledger
    paths = _implementation_paths(graph)
    expected = re.search(r'^Expected:\s*(.+)$', context.problem, re.M | re.I)
    reported_expected = expected.group(1)[:512] if expected else None
    modules, total_bytes = {}, 0
    results = []
    for callable_ in sorted(graph.callables, key=lambda c: (c.depth, c.path, c.start_line)):
        definitions = tuple(a.id for a in ledger.for_path(callable_.path)
            if a.kind is EvidenceKind.SYMBOL_DEFINITION and a.symbol == callable_.qualified_name
            and a.start_line == callable_.start_line)
        effects = tuple(e for e in graph.effects if e.callable_id == callable_.id
            and e.evidence_ids and all(ledger.get(i) for i in e.evidence_ids))
        if not definitions or not effects:
            continue
        links = tuple(EffectObservationLink(entry, effect.id,
            tuple(r.id for r in path), tuple(dict.fromkeys((*definitions, *effect.evidence_ids,
                *(i for r in path for i in r.evidence_ids)))))
            for effect in effects for (entry, target), path in paths.items()
            if target == callable_.id)
        if not links:
            continue
        if callable_.path not in modules:
            source_path = policy.resolve(str(Path(context.project_path) / callable_.path))
            size = source_path.stat().st_size
            if total_bytes + size > 4_000_000:
                continue
            total_bytes += size
            try:
                modules[callable_.path] = ast.parse(source_path.read_text(encoding='utf-8', errors='replace'))
            except (SyntaxError, RecursionError):
                continue
        tree = modules[callable_.path]
        node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.lineno == callable_.start_line and n.name == callable_.qualified_name.rsplit('.', 1)[-1]), None)
        if node is None:
            continue
        annotation = _primitive_annotation(node, tree)
        mismatch, return_line = _literal_return_violation(node, tree)
        return_evidence = tuple(i for e in effects if e.kind is EffectKind.RETURNS_VALUE
            and e.line == return_line for i in e.evidence_ids)
        violation = mismatch is True and bool(return_evidence)
        source = ExpectationSource.STATIC_ANNOTATION if annotation else (
            ExpectationSource.USER_REPORTED_EXPECTATION if reported_expected else ExpectationSource.UNKNOWN)
        results.append(BehavioralContractContext(
            _id('CT', callable_.id, source.value), callable_.id, callable_.path,
            callable_.qualified_name, context.problem[:2048], annotation or reported_expected,
            source, tuple(dict.fromkeys((*definitions, *return_evidence))), links,
            source is not ExpectationSource.UNKNOWN, bool(annotation), violation,
            return_line if violation else None,
            'UNCONDITIONAL_LITERAL_ANNOTATION_MISMATCH' if violation else (
                'ANNOTATION_SATISFIED_FOR_LITERAL_RETURN' if mismatch is False
                else 'NO_SUPPORTED_CONTRACT_VIOLATION'),
            _effect_flows(callable_, node, graph, ledger)))
    return tuple(results)
