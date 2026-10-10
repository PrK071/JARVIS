"""Compact benchmark reports only after preserving and verifying complete traces."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def compact_report(path: Path, archive_root: Path, *, source_commit: str,
                   source_blob: bytes | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    report = json.loads(raw)
    if "artifact_retention" in report:
        metadata = report["artifact_retention"]
        archived = gzip.decompress((archive_root / metadata["archive_name"]).read_bytes())
        if hashlib.sha256(archived).hexdigest() != metadata["raw_sha256"]:
            raise ValueError("preserved artifact hash mismatch")
        if source_blob is not None:
            if archived.replace(b"\r\n", b"\n") != source_blob.replace(b"\r\n", b"\n"):
                raise ValueError("archive does not match historical Git blob")
            metadata["git_blob_sha256"] = hashlib.sha256(source_blob).hexdigest()
            metadata["git_blob_bytes"] = len(source_blob)
            metadata["line_endings"] = "archive retains checkout bytes; Git blob may use LF"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
    if source_blob is not None and raw.replace(b"\r\n", b"\n") != source_blob.replace(b"\r\n", b"\n"):
        raise ValueError("report differs from preservation commit")
    digest = hashlib.sha256(raw).hexdigest()
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"{digest}.json.gz"
    compressed = gzip.compress(raw, mtime=0)
    if archive.exists() and gzip.decompress(archive.read_bytes()) != raw:
        raise ValueError("existing artifact archive differs")
    archive.write_bytes(compressed)
    if gzip.decompress(archive.read_bytes()) != raw:
        raise ValueError("artifact archive verification failed")
    rows = report.pop("results", [])
    report["case_index"] = [{
        "id": row.get("id"),
        "family": row.get("family") or row.get("spec", {}).get("family"),
        "population": row.get("population") or row.get("oracle", {}).get("population"),
        "metrics": row.get("metrics_v10") or row.get("metrics"),
        "failure_codes": row.get("failure_codes", []),
    } for row in rows]
    report["failures"] = [{
        "id": row.get("id"),
        "failure_codes": row.get("failure_codes", []),
        "failure_fingerprint": row.get("failure_fingerprint"),
    } for row in report.get("failures", [])]
    metadata = {
        "version": 1, "raw_sha256": digest,
        "raw_bytes": len(raw), "archive_bytes": len(compressed),
        "archive_name": archive.name,
        "source_commit": source_commit,
        "source_path": path.as_posix().split("tests/data/", 1)[-1],
        "recovery": "git show <source_commit>:tests/data/<source_path>",
        "note": "complete traces preserved in verified local gzip and original Git commit",
    }
    report["artifact_retention"] = metadata
    if source_blob is not None:
        metadata["git_blob_sha256"] = hashlib.sha256(source_blob).hexdigest()
        metadata["git_blob_bytes"] = len(source_blob)
        metadata["line_endings"] = "archive retains checkout bytes; Git blob may use LF"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
