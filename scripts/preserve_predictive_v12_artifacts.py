"""Copy benchmark traces to a durable local archive, verifying every byte.

This is evaluation retention, not operational predictive memory. Never deletes
source archives or overwrites differing files. Does not rewrite Git history.
"""
import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--destination", type=Path)
    group.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    source = Path(".orchestrator/predictive-artifacts/v12").resolve(strict=True)
    destination = args.destination.resolve() if args.destination else source
    if args.destination and (destination == source or destination.is_relative_to(source)):
        raise ValueError("Arquivo duravel deve ficar fora do checkout temporario.")
    destination.mkdir(parents=True, exist_ok=True, mode=0o777)
    rows = []
    for path in sorted(source.rglob("*.json.gz")):
        raw = gzip.decompress(path.read_bytes())
        sha = hashlib.sha256(raw).hexdigest()
        if path.name != sha+".json.gz":
            raise ValueError("Nome do trace nao corresponde ao hash.")
        target = destination/path.name if args.destination else path
        if target.exists():
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError("Arquivo duravel conflitante; nao sobrescrever.")
        else:
            shutil.copyfile(path, target)
        assert gzip.decompress(target.read_bytes()) == raw
        rows.append({"raw_sha256": sha, "raw_bytes": len(raw),
                     "gzip_bytes": target.stat().st_size, "file": target.name,
                     "archive_path": str(target)})
    args.index.write_text(json.dumps({"version": 12, "local_archive": str(destination),
        "source_archives_retained": True, "separate_copy_created": bool(args.destination),
        "operator_backup_required": True,
        "files": rows,
        "note": "Verified local evaluation archives; not uploaded or predictive memory. Catalog-only preserves existing files but does not guarantee persistence of temporary storage. Compact reports/specs remain in Git."}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"archives": len(rows), "verified": True, "destination": str(destination)}))
    if args.bundle:
        if args.bundle.exists():
            raise ValueError("Pacote ja existe; nao sobrescrever.")
        with zipfile.ZipFile(args.bundle, "w", compression=zipfile.ZIP_STORED) as bundle:
            bundle.write(args.index, "manifest.json")
            for row in rows:
                bundle.write(row["archive_path"], row["file"])
        with zipfile.ZipFile(args.bundle) as bundle:
            for row in rows:
                assert hashlib.sha256(gzip.decompress(bundle.read(row["file"]))).hexdigest() == row["raw_sha256"]
        print(json.dumps({"portable_bundle": str(args.bundle),
                          "sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(), "verified": True}))


if __name__ == "__main__":
    main()
