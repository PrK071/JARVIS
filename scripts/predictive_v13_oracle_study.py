"""Opt-in coverage interventions, kept separate from real-engine accuracy."""
import argparse
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.external_evaluation import load_external_cases, python_source_hash, snapshot_inventory
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.oracle_coverage_evaluation import oracle_interventions
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2, ProjectCandidateGenerator
from tern.orchestrator.security import PathPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Nao sobrescrever estudo preservado.")
    corpus = Path("tests/data/predictive/external_v13")
    rows = []
    for case in load_external_cases(corpus, split="development"):
        root = Path(".orchestrator/external-v13/snapshots") / case["id"]
        before = python_source_hash(root)
        inventory = snapshot_inventory(root)
        if before != case["source_python_sha256"]:
            raise ValueError("Snapshot divergente.")
        policy = PathPolicy((root.resolve(),))
        snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
        context = build_problem_context(case["problem_formulations"][0], snapshot,
                                        ProjectCandidateGenerator().generate(case["problem_formulations"][0], snapshot), policy)
        context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
        result = oracle_interventions(context, snapshot, policy, case["oracle"],
                                       reasoner=LlamaClient("http://127.0.0.1:8080", 180) if args.live else None)
        raw = json.dumps(result, sort_keys=True).encode()
        sha = hashlib.sha256(raw).hexdigest()
        archive = Path(".orchestrator/predictive-artifacts/v13/oracle-study") / (sha + ".json.gz")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(gzip.compress(raw, mtime=0))
        rows.append({"id": case["id"], "experiments": result["experiments"],
                     "raw_trace_archive": str(archive), "raw_trace_sha256": sha})
        if before != python_source_hash(root) or inventory != snapshot_inventory(root):
            raise AssertionError("Snapshot externo modificado.")
        print(case["id"], flush=True)
    args.output.write_text(json.dumps({"version": 13, "experimental_only": True,
        "corpus_seal": json.loads((corpus / "manifest.json").read_text())["cases_sha256"],
        "split": "development", "results": rows,
        "limitation": "A location oracle cannot invent missing causal semantics. Mode D is unavailable when no correctly localized grounded root is representable."}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
