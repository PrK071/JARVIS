"""Explicit opt-in live external evaluation; no external project execution."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.external_evaluation import evaluate_external_cases, load_external_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "holdout", "all"), required=True)
    parser.add_argument("--mode", choices=("structural", "live"), default="structural")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("tests/data/predictive/external_v12"))
    parser.add_argument("--snapshots", type=Path, default=Path(".orchestrator/external-v12/snapshots"))
    parser.add_argument("--archive", type=Path, default=Path(".orchestrator/predictive-artifacts/v12"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Nao sobrescrever resultado historico.")
    corpus = args.corpus
    cases = load_external_cases(corpus, split=args.split)
    value = evaluate_external_cases(cases, args.snapshots,
        args.archive / args.output.stem,
        reasoner=LlamaClient("http://127.0.0.1:8080", 180) if args.mode == "live" else None)
    value["engine_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    value["corpus_seal"] = json.loads((corpus/"manifest.json").read_text())["cases_sha256"]
    value["split"] = args.split
    value["runtime_source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(Path("tern/orchestrator/predictive").glob("*.py"))
        if path.name != "external_evaluation.py"
    }
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value["summary"], indent=2))


if __name__ == "__main__":
    main()
