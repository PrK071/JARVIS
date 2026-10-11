import json
from dataclasses import replace

from test_predictive_investigation import contract_case, InvestigationReasoner
from test_predictive_service import repository, FakeReasoner
from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.predictive.sufficiency import assess_evidence_sufficiency, EvidenceSufficiency
from tern.orchestrator.security import PathPolicy


def prepared(tmp_path, source, problem=None):
    args = {} if problem is None else {'problem': problem}
    context, contracts, snapshot, policy = contract_case(tmp_path, source, **args)
    slice_, roots = build_causal_slice(context, snapshot, policy)
    context = replace(context, causal_slice=slice_, root_cause_candidates=roots)
    return context, contracts


def test_real_exception_bridge(tmp_path):
    root = repository(tmp_path)
    report = PredictiveDecisionService(FakeReasoner(), path_policy=PathPolicy((root,))).predict(
        'Traceback\n  File "pkg/calc.py", line 2, in add\nTypeError: unsupported operand type(s)', root)
    assert report.evidence_sufficiency.supported_root_ids, json.dumps({
        'symptom': report.causal_slice.symptom.as_dict(),
        'roots': [r.as_dict() for r in report.root_cause_candidates],
        'assessment': report.evidence_sufficiency.as_dict()})
    assert not report.insufficient_evidence, json.dumps({
        'failure': str(report.failure_reason), 'roots': [r.as_dict() for r in report.root_cause_candidates],
        'assessment': report.evidence_sufficiency.as_dict()})


def test_present_root_is_not_proof_of_behavioral_violation(tmp_path):
    context, contracts = prepared(tmp_path, 'def read(flag):\n    if flag:\n        return None\n    return 1\n')
    assert context.root_cause_candidates
    result = assess_evidence_sufficiency(context, contracts)
    assert not result.supported_root_ids
    assert result.classification is EvidenceSufficiency.SUFFICIENT_FOR_INVESTIGATION
    assert result.requests
    assert all(not r.as_dict()['execution_requested'] for r in result.requests)


def test_demonstrated_literal_violation_is_supported(tmp_path):
    context, contracts = prepared(tmp_path, 'def read() -> int:\n    return None\n')
    result = assess_evidence_sufficiency(context, contracts)
    assert result.supported_root_ids
    assert result.classification is EvidenceSufficiency.SUFFICIENT_FOR_DIAGNOSIS
    assert not result.requests


def test_unsupported_root_routes_to_one_investigation_without_erasing_ids(tmp_path):
    (tmp_path / 'api.py').write_text('def read(flag):\n    if flag:\n        return None\n    return 1\n')
    reasoner = InvestigationReasoner()
    report = PredictiveDecisionService(reasoner, path_policy=PathPolicy((tmp_path,))).predict(
        'read() wrong output\nExpected: integer\nObserved: None', tmp_path)
    assert report.root_cause_candidates
    assert report.investigable_hypotheses
    assert len(reasoner.calls) == 1
    assert not report.candidates and not report.recommended_candidate_id
    assert not report.execution_authorized
    assert report.requires_approval
    assert 'ROOT_PRESENT_BUT_UNSUPPORTED' in report.diagnostic_codes
    assert {r.id for r in report.root_cause_candidates} == {r.root_id for r in report.evidence_sufficiency.roots}


def test_no_api_request_not_generic_context(tmp_path):
    context, contracts = prepared(tmp_path, 'def unrelated():\n    return None\n', 'Unspecified wrong result')
    result = assess_evidence_sufficiency(context, contracts)
    assert not result.investigation_eligible
    assert not result.supported_root_ids
    assert result.requests[0].missing_fact == 'AFFECTED_API_INVOCATION'
    assert not any(r.related_symbols for r in result.requests)


def test_missing_path_blocks_promotion(tmp_path):
    context, contracts = prepared(tmp_path, 'def read() -> int:\n    return None\n')
    context = replace(context, causal_slice=replace(context.causal_slice, edges=()))
    assert not assess_evidence_sufficiency(context, contracts).supported_root_ids
