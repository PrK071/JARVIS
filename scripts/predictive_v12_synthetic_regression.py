"""Separate v10 regression, preserving large raw traces outside tracked reports."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.testing.evaluation import evaluate_generated_structural, evaluate_generated_live
from tern.orchestrator.predictive.testing.generator import generate_suite, stratified_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Nao sobrescrever regressao preservada.")
    specs = generate_suite("development")
    root = Path(".orchestrator/predictive-v12-regression")
    if args.live:
        report = evaluate_generated_live(stratified_sample(specs, 14), reasoner=LlamaClient("http://127.0.0.1:8080", 180), workspace_root=root/"live")
    else:
        report = evaluate_generated_structural(specs, output_root=root/"structural")
    raw = json.dumps(report, ensure_ascii=False).encode()
    sha = hashlib.sha256(raw).hexdigest()
    archive = Path(".orchestrator/predictive-artifacts/v12")/(sha+".json.gz")
    archive.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
    archive.write_bytes(gzip.compress(raw, mtime=0))
    assert gzip.decompress(archive.read_bytes()) == raw
    rows = report.pop("results")
    report["case_index"] = [{"id": row["id"], "family": row.get("family"),
                             "metrics": row.get("metrics_v10", row.get("metrics")),
                             "failure_fingerprint": row.get("failure_fingerprint")} for row in rows]
    # Preserve legacy diagnostic codes, not duplicate their entire raw graphs.
    report["failures"] = [{"id": row["id"], "failure_codes": row.get("failure_codes", [])}
                          for row in report.get("failures", [])]
    report["raw_trace_archive"] = str(archive)
    report["raw_trace_sha256"] = sha
    report["scope"] = "v10 regression only; does not establish external readiness"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report.get("metrics") or report.get("layer_metrics"), indent=2))


if __name__ == "__main__":
    main()
