import gzip
import hashlib
import json

import pytest

from scripts.preserve_predictive_v13_artifacts import preserve


def test_backup_refuses_temporary_destination_and_preserves_original(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    raw = json.dumps({'observed': 'trace'}).encode()
    name = hashlib.sha256(raw).hexdigest() + '.json.gz'
    original = source / name
    original.write_bytes(gzip.compress(raw, mtime=0))
    with pytest.raises(ValueError, match='Temp'):
        preserve(source, tmp_path / 'Temp' / 'backup')
    assert original.read_bytes() == gzip.compress(raw, mtime=0)


def test_backup_rejects_wrong_content_hash_before_copy(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / ('0' * 64 + '.json.gz')).write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError):
        preserve(source, source / 'nested')


def test_v12_persistent_backup_manifest_is_complete():
    from pathlib import Path
    index = json.loads(Path('tests/data/predictive/results/v13-artifact-preservation.json').read_text())
    assert index['operator_backup_required'] is False
    assert index['separate_copy_created'] is True
    assert index['portable_zip_sha256'] == '4c4acc4f4f038283a7e388367402b7022b40474cffafff0c319308e6c49f4b05'
