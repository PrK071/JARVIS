"""Reconstruct historical decisions from preserved traces and frozen retrieval."""
import gzip
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def main():
    summary = Path("tests/data/predictive/results/v11-external-shadow.json")
    metadata = json.loads(summary.read_text())["artifact_retention"]
    archive = Path(".orchestrator/predictive-artifacts/v11") / metadata["archive_name"]
    if archive.exists():
        raw = gzip.decompress(archive.read_bytes())
    else:
        raw = subprocess.check_output(["git", "show", f"{metadata['source_commit']}:tests/data/{metadata['source_path']}"])
        if hashlib.sha256(raw).hexdigest() != metadata["raw_sha256"]:
            # Restore the original Windows text artifact, then verify exact bytes.
            raw = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert hashlib.sha256(raw).hexdigest() == metadata["raw_sha256"]
    history = json.loads(raw)
    project = Path(".orchestrator/external/requests-v2.25.0").resolve()
    policy = PathPolicy((project,))
    snapshot = ProjectIndexBuilderV2(project, path_policy=policy).build()
    rows = []
    for row in history["results"]:
        selection = ProjectCandidateGenerator().generate(row["problem"], snapshot)
        context = build_problem_context(row["problem"], snapshot, selection, policy)
        context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
        atoms = [atom.as_dict() for atom in context.evidence_ledger.atoms
                 if atom.path == "requests/utils.py" and atom.symbol == "get_encoding_from_headers"]
        decision = row["decision"]
        roots = decision["root_cause_candidates"]
        site_id = decision["causal_slice"]["failure_site_id"]
        site = next(node for node in decision["causal_slice"]["nodes"] if node["id"] == site_id)
        rows.append({
            "id": row["id"], "problem": row["problem"],
            "initial_files": context.related_files,
            "correct_file_retrieved": "requests/utils.py" in context.related_files,
            "correct_symbol_in_snapshot": any(symbol.name == "get_encoding_from_headers"
                for symbol in snapshot.file_index["requests/utils.py"].symbols),
            "correct_symbol_ledger_atoms": atoms,
            "correct_origin_candidates": [item for item in roots if item["origin_path"] == "requests/utils.py"
                                          and item["origin_symbol"] == "get_encoding_from_headers"],
            "selected_root_ids": [item["root_cause_id"] for item in decision["hypotheses"]],
            "failure_site": site,
            "traceback_present": "Traceback" in row["problem"],
            "initial_evidence_refs": context.evidence_refs,
            "domination": decision["root_cause_selections"],
            "fabricated_bridge_edges": [edge for edge in decision["causal_slice"]["edges"]
                                         if edge["target_id"] == site_id and edge["kind"] == "RAISES_AT"],
        })
    output = Path("tests/data/predictive/results/v12-requests-decomposition.json")
    output.write_text(json.dumps({"version": 12, "source_trace_sha256": metadata["raw_sha256"],
        "independent_defects": 1, "formulations": len(rows), "results": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps({key: row[key] for key in (
            "id", "initial_files", "correct_file_retrieved", "correct_symbol_in_snapshot",
            "correct_symbol_ledger_atoms", "correct_origin_candidates", "failure_site", "domination",
        )}, indent=2))


if __name__ == "__main__":
    main()
