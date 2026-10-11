import hashlib
from scripts.preserve_predictive_v13_artifacts import preserve


def test_completed_and_censored_journals_are_backed_up_without_cleanup(tmp_path, monkeypatch):
    # Somente a classificacao do storage e simulada; os testes v13 preservam
    # a rejeicao obrigatoria de Temp na politica real.
    monkeypatch.setattr('scripts.preserve_predictive_v13_artifacts._temporary_destination', lambda path: False)
    source = tmp_path / 'source'
    source.mkdir()
    journal = source / 'checkpoints.jsonl'
    journal.write_text('{"state": "START", "phase": "causal_slice"}\n')
    result = preserve(source, tmp_path / 'permanent', include_checkpoints=True)
    assert len(result['checkpoints']) == 1
    assert (tmp_path / 'permanent' / journal.name).read_bytes() == journal.read_bytes()
    assert result['checkpoints'][0]['sha256'] == hashlib.sha256(journal.read_bytes()).hexdigest()
    assert journal.exists() and result['originals_retained']
