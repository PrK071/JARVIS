"""Copia integra de traces para armazenamento local fora de Temp, sem limpeza."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def _temporary_destination(path):
    return 'temp' in {part.casefold() for part in path.parts}


def preserve(source: Path, destination: Path, *, include_checkpoints=False):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.is_relative_to(source) or _temporary_destination(destination):
        raise ValueError('Destino permanente nao pode estar em Temp ou na origem.')
    rows = []
    for original in sorted(source.rglob('*.json.gz')):
        raw = gzip.decompress(original.read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if original.name != digest + '.json.gz':
            raise ValueError('Hash de conteudo divergente.')
        target = destination / original.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                raise ValueError('Backup existente divergente; nao sobrescrever.')
        else:
            shutil.copy2(original, target)
        if target.read_bytes() != original.read_bytes():
            raise ValueError('Copia nao verificada.')
        rows.append({'relative_path': original.relative_to(source).as_posix(),
                     'raw_sha256': digest,
                     'compressed_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                     'compressed_bytes': target.stat().st_size})
    checkpoints = []
    if include_checkpoints:
        for original in sorted(source.rglob('checkpoints.jsonl')):
            if original.is_symlink() or not original.resolve().is_relative_to(source):
                raise ValueError('Checkpoint fora da origem autorizada.')
            raw = original.read_bytes()
            if len(raw) > 5_000_000:
                raise ValueError('Journal excede limite de preservacao.')
            target = destination / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('Checkpoint preservado divergente.')
            if not target.exists():
                shutil.copy2(original, target)
            if target.read_bytes() != raw:
                raise ValueError('Checkpoint nao verificado.')
            checkpoints.append({'relative_path': original.relative_to(source).as_posix(),
                'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    return {'version': 13, 'destination': str(destination), 'traces': rows,
            'checkpoints': checkpoints,
            'count': len(rows), 'verified': True, 'originals_retained': True,
            'limitation': 'Verified same-host local backup, not off-machine disaster recovery.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('.orchestrator/predictive-artifacts/v13'))
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-checkpoints', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Nao sobrescrever indice preservado.')
    result = preserve(args.source, args.destination, include_checkpoints=args.include_checkpoints)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Traces verificados: {result['count']}")


if __name__ == '__main__':
    main()
