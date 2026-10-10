import gzip
import hashlib
import json

from scripts.audit_predictive_external_artifacts import audit


def test_audit_configurable_snapshots_and_forbidden_gate(tmp_path):
    root = tmp_path / 'sources' / 'one'
    root.mkdir(parents=True)
    (root / 'api.py').write_text('def decode():\n    return None\n')
    decision = {'hypotheses': [], 'candidates': [{'id': 'C1', 'evidence_refs': ['api.py:1-2'],
                 'eligible': False, 'rejection_reasons': ['FORBIDDEN_CANDIDATE']}],
                'recommended_candidate_id': 'C1', 'execution_authorized': False,
                'requires_approval': True, 'dry_run': True}
    raw = json.dumps({'decision': decision}).encode()
    trace = tmp_path / 'trace.json.gz'
    trace.write_bytes(gzip.compress(raw))
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'results': [{'case_id': 'one', 'formulation': 1,
        'raw_trace_archive': str(trace), 'raw_trace_sha256': hashlib.sha256(raw).hexdigest()}]}))
    value = audit(report, snapshots=tmp_path / 'sources')
    assert value['evidence_reference_validity']['numerator'] == 1
    assert value['forbidden_recommendations'] == 1
    assert not value['passed']
