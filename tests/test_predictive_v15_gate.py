import hashlib
import json

import pytest

from scripts import record_predictive_v15 as gate


def _journal(tmp_path, monkeypatch, events):
    store = tmp_path / 'permanent'
    store.mkdir()
    raw = ''.join(json.dumps(event) + '\n' for event in events).encode()
    (store / 'checkpoints.jsonl').write_bytes(raw)
    result = tmp_path / 'results'
    result.mkdir()
    manifest = {'destination': str(store), 'checkpoints': [
        {'relative_path': 'checkpoints.jsonl', 'sha256': hashlib.sha256(raw).hexdigest()}]}
    (result / 'v15-artifact-preservation-final.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(gate, 'RESULTS', result)
    return store


def test_checkpoint_audit_counts_censored_request(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, [
        {'phase': 'qwen_analysis', 'state': 'START', 'wall_seconds': 10},
        {'phase': 'qwen_request', 'state': 'START', 'wall_seconds': 14}])
    audit = gate.checkpoint_audit()
    assert audit['requests_started'] == audit['requests_unfinished'] == 1
    assert audit['requests_finished'] == 0
    assert audit['rows'][0]['pre_request_seconds'] == 4
    assert audit['rows'][0]['last_checkpoint']['phase'] == 'qwen_request'


def test_checkpoint_audit_keeps_completed_request(tmp_path, monkeypatch):
    _journal(tmp_path, monkeypatch, [
        {'phase': 'qwen_request', 'state': 'START', 'wall_seconds': 2},
        {'phase': 'qwen_request', 'state': 'END', 'wall_seconds': 5,
         'elapsed_seconds': 3, 'working_set_bytes': 100}])
    audit = gate.checkpoint_audit()
    assert audit['requests_started'] == audit['requests_finished'] == 1
    assert audit['requests_unfinished'] == 0
    assert audit['rows'][0]['phase_seconds']['qwen_request'] == 3
    assert audit['rows'][0]['pre_request_seconds'] is None


def test_checkpoint_audit_rejects_backup_hash_mismatch(tmp_path, monkeypatch):
    store = _journal(tmp_path, monkeypatch, [])
    (store / 'checkpoints.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Hash'):
        gate.checkpoint_audit()
