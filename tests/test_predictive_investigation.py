import json
from dataclasses import replace

import pytest

from test_predictive_behavioral_graph import prepare
from tern.orchestrator.predictive.contracts import build_contract_contexts, ExpectationSource
from tern.orchestrator.predictive.investigation import (
    build_investigable_hypotheses, select_investigations, validate_hypothesis_promotion,
    InvestigationStatus,
)
from tern.orchestrator.predictive.causal import build_causal_slice, RootCauseKind
from tern.orchestrator.predictive.models import PredictiveFailureReason
from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.security import PathPolicy
from tern.orchestrator.predictive.testing.contract_families import contract_families


def contract_case(tmp_path, source, problem='read() wrong output\nExpected: integer\nObserved: None'):
    context, snapshot, policy = prepare(tmp_path, {'api.py': source}, problem)
    contracts = build_contract_contexts(context, policy)
    return context, contracts, snapshot, policy


@pytest.mark.parametrize('family', contract_families(), ids=lambda f: f.name)
def test_family_contract_and_investigation_population(tmp_path, family):
    context, contracts, _, _ = contract_case(tmp_path, family.source, family.problem)
    hypotheses = build_investigable_hypotheses(context, contracts)
    assert any(c.violation_supported for c in contracts) == family.violation
    if family.investigation_symbol:
        assert any(h.symbol == family.investigation_symbol for h in hypotheses)
    else:
        assert not hypotheses
    for h in hypotheses:
        assert h.status is InvestigationStatus.INVESTIGABLE
        assert h.evidence_ids
        assert all(context.evidence_ledger.get(i) for i in h.evidence_ids)
        assert not h.as_dict()['confirmed_root']
        assert not h.as_dict()['recommendation_eligible']
    for contract in contracts:
        assert all(not link.as_dict()['execution_demonstrated'] for link in contract.links)


@pytest.mark.parametrize('source', [
    'def read() -> int:\n    return 1\n',
    'def read() -> int:\n    return True\n',
    'def read() -> float:\n    return False\n',
    'def read(flag) -> int:\n    if flag:\n        return None\n    return 1\n',
    'def read() -> int:\n    return external()\n',
    'int = Custom\ndef read() -> int:\n    return None\n',
    '@decorate\ndef read() -> int:\n    return None\n',
    'def read() -> int:\n    """Return None, probably producer boundary defect."""\n    return 1\n',
])
def test_annotation_or_report_is_not_automatically_violation(tmp_path, source):
    _, contracts, _, _ = contract_case(tmp_path, source)
    assert not any(c.violation_supported for c in contracts)


def test_reported_expectation_is_unverified_and_no_promotion(tmp_path):
    context, contracts, _, _ = contract_case(tmp_path, 'def read(item):\n    return item\n')
    contract = contracts[0]
    assert contract.expectation_source is ExpectationSource.USER_REPORTED_EXPECTATION
    assert contract.contract_exists and not contract.contract_applies
    hypothesis = build_investigable_hypotheses(context, contracts)[0]
    outcome = validate_hypothesis_promotion(hypothesis, contract, None, None, context.evidence_ledger)
    assert not outcome.approved
    assert 'CONTRACT_VIOLATION_NOT_SUPPORTED' in outcome.reasons


def test_promotion_requires_real_bridge_and_compatible_evidence(tmp_path):
    context, contracts, snapshot, policy = contract_case(tmp_path, 'def read() -> int:\n    return None\n')
    causal, roots = build_causal_slice(context, snapshot, policy)
    hypothesis = build_investigable_hypotheses(context, contracts)[0]
    root = next(r for r in roots if r.cause_kind is RootCauseKind.RETURN_CONTRACT)
    result = validate_hypothesis_promotion(hypothesis, contracts[0], root, causal, context.evidence_ledger)
    assert result.approved
    assert result.root_id == root.id
    assert not validate_hypothesis_promotion(replace(hypothesis, status=InvestigationStatus.CONTRADICTED),
        contracts[0], root, causal, context.evidence_ledger).approved
    assert not validate_hypothesis_promotion(hypothesis, contracts[0], replace(root, evidence_ids=()),
        causal, context.evidence_ledger).approved
    assert not validate_hypothesis_promotion(hypothesis, contracts[0], root,
        replace(causal, edges=()), context.evidence_ledger).approved


class InvestigationReasoner:
    def __init__(self, response=None):
        self.calls = []
        self.response = response

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        locations = json.loads(messages[1]['content'])['locations']
        answer = self.response if self.response is not None else {'investigations': [
            {'id': locations[0]['id'], 'missing_evidence': ['VERIFY_EFFECT_TO_OBSERVATION']}]}
        return {'choices': [{'message': {'content': json.dumps(answer)}}]}


def test_one_call_ids_only_and_requirements_cannot_be_erased(tmp_path):
    context, contracts, _, _ = contract_case(tmp_path, 'def read():\n    yield 1\n')
    hypotheses = build_investigable_hypotheses(context, contracts)
    reasoner = InvestigationReasoner()
    selected, failure = select_investigations(reasoner, context, hypotheses)
    assert failure is None and selected == hypotheses
    assert len(reasoner.calls) == 1
    assert 'VERIFY_CONTRACT' in selected[0].as_dict()['required_evidence']
    assert 'scores' not in reasoner.calls[0][1]['response_format']['json_schema']['schema']['properties']


@pytest.mark.parametrize('answer', [
    {'investigations': [{'id': 'unknown', 'missing_evidence': ['VERIFY_CONTRACT']}]},
    {'investigations': [], 'root_cause': 'confirmed'},
    {'investigations': None},
    {'investigations': [{'id': 'unknown', 'missing_evidence': [], 'confidence': 1}]},
])
def test_invalid_reasoner_cannot_promote_or_invent_locations(tmp_path, answer):
    context, contracts, _, _ = contract_case(tmp_path, 'def read():\n    yield 1\n')
    selected, failure = select_investigations(InvestigationReasoner(answer), context,
        build_investigable_hypotheses(context, contracts))
    assert not selected
    assert failure is PredictiveFailureReason.INVALID_STRUCTURED_RESPONSE


def test_no_anchor_no_reasoner_call(tmp_path):
    context, contracts, _, _ = contract_case(tmp_path, 'def unknown():\n    return None\n', 'Unspecified wrong result')
    reasoner = InvestigationReasoner()
    assert select_investigations(reasoner, context, build_investigable_hypotheses(context, contracts)) == ((), None)
    assert not reasoner.calls


def test_hypothesis_only_service_never_recommends_or_grants_authority(tmp_path):
    (tmp_path / 'api.py').write_text('class Api:\n    def update(self, item):\n        self.current = item\n')
    reasoner = InvestigationReasoner()
    report = PredictiveDecisionService(reasoner, path_policy=PathPolicy((tmp_path,))).predict(
        'Api.update() stores unexpected state', tmp_path)
    assert len(reasoner.calls) == 1
    assert report.investigable_hypotheses
    assert not report.hypotheses and not report.root_cause_candidates and not report.candidates
    assert report.recommended_candidate_id is None
    assert report.insufficient_evidence and not report.execution_authorized and report.requires_approval
    value = report.as_dict()
    assert value['investigation_schema_version'] == 1
    assert value['behavioral_contract_contexts']
    with pytest.raises(ValueError):
        replace(report, insufficient_evidence=False)


def test_return_forwarding_and_branch_guard_are_not_execution_proofs(tmp_path):
    _, contracts, _, _ = contract_case(tmp_path, 'def read():\n    return helper()\ndef helper():\n    return None\n')
    flows = [r for c in contracts for r in c.effect_flow_relations]
    assert any(r.kind == 'RETURN_CONSUMED_BY_API' for r in flows)
    assert all(not r.as_dict()['responsibility_demonstrated'] for r in flows)
    _, branch, _, _ = contract_case(tmp_path / 'branch', 'def read(flag):\n    if flag:\n        return None\n    return 1\n')
    assert any(r.kind == 'CONDITION_GOVERNS_EFFECT' for c in branch for r in c.effect_flow_relations)
    _, discarded, _, _ = contract_case(tmp_path / 'discard', 'def read():\n    helper()\n    return 1\ndef helper():\n    return None\n')
    assert not any(r.kind == 'RETURN_CONSUMED_BY_API' for c in discarded for r in c.effect_flow_relations)


def test_comment_and_wording_do_not_create_contract_violation(tmp_path):
    context, contracts, _, _ = contract_case(tmp_path / 'a', 'def read() -> int:\n    return 1\n')
    _, distractor, _, _ = contract_case(tmp_path / 'b',
        '# validation return None producer import\ndef read() -> int:\n    """Broken boundary contract."""\n    return 1\n',
        'read() apparently broken return contract')
    assert all(not c.violation_supported for c in (*contracts, *distractor))
    _, counterfactual, _, _ = contract_case(tmp_path / 'c', 'def read() -> int:\n    return None\n')
    assert counterfactual[0].violation_supported


def test_self_helper_and_constructor_state_are_investigable_not_proven_wrong(tmp_path):
    context, contracts, _, _ = contract_case(tmp_path,
        'class Api:\n    def __init__(self, item):\n        self.current = item\n'
        '    def read(self):\n        return self._helper()\n'
        '    def _helper(self):\n        return self.current\n',
        'Api.__init__() Api.read() wrong state\nExpected: item\nObserved: different state')
    assert any(r.resolution == 'LEXICAL_SELF_METHOD' for r in context.behavioral_graph.relations)
    assert any(c.symbol == 'Api.__init__' for c in contracts)
    assert any(c.symbol == 'Api._helper' for c in contracts)
    assert not any(c.violation_supported for c in contracts)


def test_known_instantiation_remains_unresolved_until_receiver_binding_supported(tmp_path):
    context, _, _, _ = contract_case(tmp_path,
        'class Api:\n    def read(self):\n        return 1\n'
        'def read():\n    instance = Api()\n    return instance.read()\n')
    assert any(u.expression == 'instance.read' for u in context.behavioral_graph.unresolved)
    assert not any(r.call_expression == 'instance.read' for r in context.behavioral_graph.relations)
