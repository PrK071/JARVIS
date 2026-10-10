"""Compact newly generated v12 legacy diagnostics after verifying the archive."""
import gzip
import hashlib
import json
from pathlib import Path


def main():
    path = Path("tests/data/predictive/results/v12-synthetic-live-regression.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    raw = gzip.decompress(Path(report["raw_trace_archive"]).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == report["raw_trace_sha256"]
    full = json.loads(raw)
    assert full["layer_metrics"] == report["layer_metrics"]
    report["failures"] = [{"id": row["id"], "failure_codes": row.get("failure_codes", [])}
                          for row in full.get("failures", [])]
    report["legacy_diagnostics_note"] = "Inherited v1 diagnostic mismatches are separate from v10 oracle metrics; full raw records retained in verified gzip."
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
