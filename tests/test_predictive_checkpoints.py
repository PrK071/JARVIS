import json
from pathlib import Path

import pytest

from scripts.predictive_checkpoints import CheckpointJournal, last_checkpoint
from tern.orchestrator.predictive.operational import AnalysisTimer, observe_checkpoints, observe_phase


def test_start_checkpoint_survives_unfinished_operation(tmp_path):
    journal = CheckpointJournal(tmp_path / 'observations' / 'phases.jsonl', tmp_path / 'snapshot')
    def operation():
        value = last_checkpoint(journal.path)
        assert value['phase'] == 'diagnostic.index' and value['state'] == 'START'
        assert 'working_set_bytes' in value
        raise TimeoutError()
    with observe_checkpoints(journal), pytest.raises(TimeoutError):
        observe_phase('diagnostic.index', operation)
    assert last_checkpoint(journal.path)['state'] == 'ERROR'
    assert last_checkpoint(journal.path)['error_type'] == 'TimeoutError'
    assert 'queue' not in last_checkpoint(journal.path)['phase']


def test_phase_observer_is_optional_and_context_local(tmp_path):
    events = []
    with observe_checkpoints(events.append):
        assert AnalysisTimer().measure('ledger', lambda: 42) == 42
    assert [e['state'] for e in events] == ['START', 'END']
    assert observe_phase('unobserved', lambda: 7) == 7
    assert len(events) == 2


def test_checkpoint_cannot_write_into_external_snapshot(tmp_path):
    snapshot = tmp_path / 'sources'
    with pytest.raises(ValueError):
        CheckpointJournal(snapshot / 'checkpoint.jsonl', snapshot)
    journal = CheckpointJournal(tmp_path / 'observations.jsonl', snapshot)
    journal({'phase': 'qwen_request', 'state': 'START'})
    with pytest.raises(ValueError):
        CheckpointJournal(journal.path, snapshot)
    assert last_checkpoint(journal.path)['model_timing_resolution'].endswith('UNKNOWN')


def test_partially_written_final_event_preserves_last_complete(tmp_path):
    path = tmp_path / 'checkpoints.jsonl'
    path.write_text(json.dumps({'phase': 'qwen_request', 'state': 'START'}) + '\n{"partial"', encoding='utf-8')
    assert last_checkpoint(path)['phase'] == 'qwen_request'
