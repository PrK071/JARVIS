from dataclasses import replace
import hashlib

import pytest

from tern.orchestrator.predictive.behavioral import (
    BehavioralGraph, EffectKind, ImplementationBudget, RelationSupport,
    bind_graph_evidence, resolve_behavioral_context,
)
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.causal import build_causal_slice, CausalNodeKind, CausalEdgeKind
from tern.orchestrator.predictive.models import EvidenceExcerpt, EvidenceKind, ProblemContext
from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.predictive.testing.behavioral_families import behavioral_families
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def prepare(tmp_path, files, problem, *, budget=None, windows=None):
    for path, source in files.items():
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(source)
    policy = PathPolicy((tmp_path,))
    snapshot = ProjectIndexBuilderV2(tmp_path, path_policy=policy).build()
    names = tuple(path for path in files if 'api' in path)
    excerpts = windows if windows is not None else tuple(EvidenceExcerpt(
        f'{path}:1-1', path, 1, 1, 'STRONG', (), files[path].splitlines()[0]) for path in names)
    context = ProblemContext(problem, str(tmp_path), 'test', excerpts, names, (), ())
    context = resolve_behavioral_context(context, snapshot, policy, budget=budget)
    ledger = build_evidence_ledger(context, snapshot, policy)
    context = replace(context, evidence_ledger=ledger,
                      behavioral_graph=bind_graph_evidence(context.behavioral_graph, ledger))
    return context, snapshot, policy


@pytest.mark.parametrize('family', behavioral_families(), ids=lambda family: family.name)
def test_family_path_effect_and_no_defect_promotion(tmp_path, family):
    context, _, _ = prepare(tmp_path, family.files, family.problem)
    graph = context.behavioral_graph
    if family.implementation:
        assert any(c.qualified_name == family.implementation for c in graph.callables)
    else:
        assert not graph.relations
        assert graph.unresolved
    if family.effect:
        assert any(e.kind.value == family.effect for e in graph.effects)
    else:
        assert not graph.effects
    assert not graph.as_dict()['sufficiency']['contract_supported']
    assert 'NO_SUPPORTED_CONTRACT' in graph.reason_codes
    for relation in graph.relations:
        assert relation.support is RelationSupport.STATICALLY_RESOLVED
        assert relation.evidence_ids
        assert all(context.evidence_ledger.get(e).start_line == relation.line for e in relation.evidence_ids)
    assert all(effect.evidence_ids for effect in graph.effects)


def test_no_same_file_evidence_fallback(tmp_path):
    context, _, _ = prepare(tmp_path, {'api.py': 'def read():\n    return helper()\ndef helper():\n    return 1\n'}, 'read() wrong output')
    graph = bind_graph_evidence(context.behavioral_graph, replace(context.evidence_ledger,
        atoms=tuple(a for a in context.evidence_ledger.atoms if a.kind is EvidenceKind.SYMBOL_DEFINITION)))
    assert all(not r.evidence_ids for r in graph.relations)
    assert all(not e.evidence_ids for e in graph.effects)
    assert 'MISSING_LEDGER_RELATION' in graph.reason_codes


def test_void_effect_transport_does_not_invent_cause(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {'api.py':
        'class Api:\n    def update(self, item):\n        self.current = item\n'}, 'Api.update() wrong state')
    causal, roots = build_causal_slice(context, snapshot, policy)
    effects = [n for n in causal.nodes if n.kind is CausalNodeKind.BEHAVIORAL_EFFECT]
    assert effects and all(n.evidence_ids for n in effects)
    assert any(e.kind is CausalEdgeKind.HAS_EFFECT for e in causal.edges)
    assert not roots
    assert not any(n.kind is CausalNodeKind.EXCEPTION_SITE for n in causal.nodes)


def test_unresolved_receiver_cannot_inherit_unrelated_return(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {'api.py':
        'def read(receiver):\n    return receiver.helper()\ndef helper():\n    return None\n'}, 'read() wrong output')
    causal, roots = build_causal_slice(context, snapshot, policy)
    helper_returns = {n.id for n in causal.nodes if n.scope == 'helper' and n.kind is CausalNodeKind.RETURN_VALUE}
    assert not any(e.source_id in helper_returns and e.kind is CausalEdgeKind.RETURNED_FROM for e in causal.edges)
    assert not any(r.origin_symbol == 'helper' for r in roots)


def test_bound_method_resolution_uses_correct_argument_slot(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {'api.py':
        'class Api:\n    def read(self, item):\n        return self.helper(item)\n    def helper(self, payload):\n        return payload\n'}, 'Api.read() wrong output')
    causal, _ = build_causal_slice(context, snapshot, policy)
    bindings = [e.argument_binding for e in causal.edges if e.argument_binding
                and e.argument_binding.callee_id.endswith('Api.helper')]
    assert bindings
    targets = {n.id: n.symbol for n in causal.nodes}
    assert all(targets[b.formal_parameter_id] == 'payload' for b in bindings)


@pytest.mark.parametrize('source', [
    'class Api(Base):\n    def read(self):\n        return self.helper()\n    def helper(self):\n        return 1\n',
    'class Api:\n    def __init__(self):\n        self.helper = override\n    def read(self):\n        return self.helper()\n    def helper(self):\n        return 1\n',
    'def helper():\n    return 1\nhelper = external\ndef read():\n    return helper()\n',
    'def helper():\n    return 1\ndef read(helper):\n    return helper()\n',
    'def helper():\n    return 1\ndef read():\n    return factory().helper()\n',
    'def helper():\n    return 1\ndef read():\n    def helper():\n        return 2\n    return helper()\n',
])
def test_ambiguous_or_shadowed_dispatch_not_guessed(tmp_path, source):
    context, _, _ = prepare(tmp_path, {'api.py': source}, 'api.py read() wrong output')
    assert not context.behavioral_graph.relations
    assert context.behavioral_graph.unresolved


def test_budget_and_cycle_determinism(tmp_path):
    files = {'api.py': 'from impl import helper\ndef read():\n    return helper()\n',
             'impl.py': 'from api import read\ndef helper():\n    return read()\n'}
    context, snapshot, policy = prepare(tmp_path, files, 'read() wrong output')
    assert len(context.behavioral_graph.callables) == 2
    assert len(context.behavioral_graph.relations) == 2
    again = resolve_behavioral_context(replace(context, evidence=tuple(context.evidence[:1]),
        relevant_symbol_contexts=(), behavioral_graph=None), snapshot, policy)
    assert bind_graph_evidence(again.behavioral_graph, context.evidence_ledger).relations == context.behavioral_graph.relations
    disabled = resolve_behavioral_context(context, snapshot, policy, budget=ImplementationBudget(max_symbols=0))
    assert not disabled.behavioral_graph.callables
    depth0 = resolve_behavioral_context(context, snapshot, policy, budget=ImplementationBudget(max_depth=0))
    assert not depth0.behavioral_graph.relations
    assert 'BUDGET_EXHAUSTED' in depth0.behavioral_graph.reason_codes
    with pytest.raises(ValueError):
        ImplementationBudget(max_extra_files=-1)


def test_context_byte_and_file_budget(tmp_path):
    files = {'api.py': 'from impl import helper\ndef read():\n    return helper()\n',
             'impl.py': 'def helper():\n    return 1\n'}
    limited, _, _ = prepare(tmp_path, files, 'read() wrong output', budget=ImplementationBudget(max_extra_files=0))
    assert set(limited.related_files) == {'api.py'}
    assert not limited.behavioral_graph.relations
    tiny, _, _ = prepare(tmp_path, files, 'read() wrong output', budget=ImplementationBudget(max_context_bytes=1))
    assert not tiny.behavioral_graph.callables
    assert 'BUDGET_EXHAUSTED' in tiny.behavioral_graph.reason_codes


def test_rename_comments_wording_and_counterfactual(tmp_path):
    one, _, _ = prepare(tmp_path/'a', {'api.py': 'def read():\n    return helper()\ndef helper():\n    yield 1\n'}, 'read() wrong iteration')
    two, _, _ = prepare(tmp_path/'b', {'api_alt.py': 'def decode():\n    """validation return import None"""\n    # helper producer boundary\n    return produce()\ndef produce():\n    yield 1\n'}, 'decode() unexpected behavior')
    assert sorted(e.kind for e in one.behavioral_graph.effects) == sorted(e.kind for e in two.behavioral_graph.effects)
    assert len(one.behavioral_graph.relations) == len(two.behavioral_graph.relations) == 1
    changed, _, _ = prepare(tmp_path/'c', {'api.py': 'def read():\n    return 1\ndef helper():\n    yield 1\n'}, 'read() wrong iteration')
    assert not changed.behavioral_graph.relations
    assert not any(e.kind is EffectKind.YIELDS_VALUE for e in changed.behavioral_graph.effects)


def test_inadequate_line_anchor_preserved_and_no_authority(tmp_path):
    source = 'def unrelated():\n    return 1\nclass Api:\n    def update(self, item):\n        self.current = item\n'
    path = tmp_path/'api.py'
    path.write_text(source)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    class InvestigationReason:
        def chat(self, *args, **kwargs):
            calls.append(1)
            assert kwargs['response_format']['json_schema']['name'] == 'investigable_locations'
            return {'choices': [{'message': {'content': '{"investigations": []}'}}]}
    report = PredictiveDecisionService(InvestigationReason(), path_policy=PathPolicy((tmp_path,))).predict(
        'api.py:2 Api.update() stores unexpected state', tmp_path)
    graph = report.behavioral_graph
    assert graph and any(e.kind is EffectKind.WRITES_ATTRIBUTE for e in graph.effects)
    assert graph.initial_windows
    assert not report.execution_authorized and report.requires_approval
    assert report.as_dict()['behavioral_graph_schema_version'] == 1
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(calls) == 1
    assert not report.root_cause_candidates and not report.candidates
    assert any(a.reason == 'ANCHOR_EXISTS_BUT_UNRESOLVED' for a in report.causal_slice.relevance_assessments)


def test_unknown_api_true_insufficiency(tmp_path):
    context, _, _ = prepare(tmp_path, {'api.py': 'def unrelated():\n    return None\n'}, 'api.py unknown behavior')
    assert not context.behavioral_graph.entry_ids
    assert context.behavioral_graph.reason_codes == ('NO_STRUCTURAL_API_ANCHOR',)
    with pytest.raises(ValueError):
        BehavioralGraph(entry_ids=('not-existing',))
