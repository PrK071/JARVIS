"""Read-only external study. Never imports or executes the inspected repository."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import time
import pstats
from pathlib import Path

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.service import PredictiveDecisionService
from tern.orchestrator.security import PathPolicy

CASES = (
    "Requests 2.25.0: for an HTTP response with Content-Type application/json without charset, response.text decodes a short UTF-8 JSON body incorrectly while response.json() returns the expected Unicode. Adding an unrelated uuid field changes the result. Diagnose this encoding inconsistency.",
    "Requests 2.25.0: the same Unicode JSON payload is read differently by Response.text and Response.json when the server sends application/json and omits the charset parameter. The result depends on payload length. What code causes this behavior?",
)


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class MeasuredReasoner:
    def __init__(self, client) -> None:
        self.client = client
        self.requests = []

    def chat(self, *args, **kwargs):
        started = time.monotonic()
        response = self.client.chat(*args, **kwargs)
        self.requests.append({"wall_seconds": time.monotonic() - started,
                              "usage": response.get("usage", {})})
        return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    root = args.project.resolve(strict=True)
    before = source_hash(root)
    reasoner = MeasuredReasoner(LlamaClient(args.base_url, 180))
    service = PredictiveDecisionService(reasoner, path_policy=PathPolicy((root,)))
    rows = []
    for index, problem in enumerate(CASES[:args.limit]):
        started = time.monotonic()
        profile = cProfile.Profile()
        report = profile.runcall(service.predict, problem, root)
        stats = pstats.Stats(profile).stats
        def phase(filename, function):
            return sum(item[3] for (path, _line, name), item in stats.items()
                       if path.endswith(filename) and name == function)
        selected = {item.root_cause_id for item in report.hypotheses}
        roots = [item for item in report.root_cause_candidates if item.id in selected]
        root_hit = any(item.origin_path == "requests/utils.py" and item.origin_symbol == "get_encoding_from_headers" for item in roots)
        rows.append({"id": f"requests-5667-wording-{index + 1}", "problem": problem,
            "root_matches_upstream": root_hit,
            "wall_seconds": time.monotonic() - started,
            "phase_seconds": {
                "project_index": phase("project_intelligence_v2.py", "build"),
                "retrieval_selection": phase("project_intelligence_v2.py", "generate"),
                "context": phase("service.py", "build_problem_context"),
                "evidence_ledger": phase("grounding.py", "build_evidence_ledger"),
                "causal_analysis": phase("causal.py", "build_causal_slice"),
                "structural_recovery_inclusive": phase("recovery.py", "recover"),
                "qwen": phase("client.py", "chat"),
                "scoring": phase("scoring.py", "ranking_decision"),
                "patch_proposal": None, "sandbox_setup": None,
                "baseline_tests": None, "patched_tests": None, "cleanup": None,
            },
            "phase_note": "cProfile cumulative wall times; inclusive recovery may overlap causal analysis",
            "decision": report.as_dict()})
    unchanged = before == source_hash(root)
    output = {
        "version": 1, "study": "external-shadow-v11", "independent_defects": 1,
        "wording_scenarios": len(rows), "project": "psf/requests",
        "source_commit": "03957eb1c2b9a1e5e6d61f5e930d7c5ed7cfe853",
        "issue": "https://github.com/psf/requests/issues/5667",
        "ground_truth": {"source": "https://github.com/psf/requests/pull/5673",
            "target": "requests/utils.py::get_encoding_from_headers",
            "defect": "missing application/json default UTF-8 return when charset absent"},
        "freeze": "6b288b84fed38ce0904f0535a1748472bae5a59d",
        "project_source_unchanged": unchanged, "source_tree_sha256": before,
        "qwen_requests": reasoner.requests, "results": rows,
        "not_a_population_accuracy_estimate": True,
        "no_project_execution": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"root_hits": sum(row["root_matches_upstream"] for row in rows),
        "scenarios": len(rows), "project_unchanged": unchanged}))


if __name__ == "__main__":
    main()
