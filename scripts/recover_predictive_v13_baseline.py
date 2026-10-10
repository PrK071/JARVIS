"""Recover completed baseline observations without rerunning or inventing rows."""
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from tern.orchestrator.predictive.external_evaluation import load_external_cases, matches_origin, rate


def main():
    target = Path("tests/data/predictive/results/v13-development-engine-frozen.json")
    if target.exists():
        raise ValueError("Baseline ja preservado.")
    cases = load_external_cases(Path("tests/data/predictive/external_v13"), split="development")
    archive = Path(".orchestrator/predictive-artifacts/v13/v13-development-engine-frozen")
    blobs = []
    for path in archive.glob("*.json.gz"):
        raw = gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest() != path.name[:-8]:
            raise ValueError("Trace divergente.")
        value = json.loads(raw)
        # Interrupted first attempt with Qwen unavailable is retained but never
        # substituted for the healthy frozen-engine run.
        if value["decision"].get("failure_reason") != "REASONER_UNAVAILABLE":
            blobs.append((path.stat().st_mtime_ns, path, value))
    rows, pending = [], []
    for case in cases:
        options = [item for item in blobs if item[2]["decision"].get("problem") == case["problem_formulations"][0]]
        if not options:
            pending.append({"id": case["id"], "status": "NOT_COMPLETED_OPERATIONAL_LIMIT",
                            "reason": "Frozen analysis exceeded fifteen minutes; not counted as root error or abstention."})
            continue
        _, path, trace = max(options, key=lambda item: item[0])
        raw, oracle = trace["decision"], case["oracle"]
        atoms = trace["initial_context"]["ledger"]["atoms"]
        roots = raw["root_cause_candidates"]
        valid = {root["id"] for root in roots if matches_origin(root, oracle)}
        selected = {hypothesis["root_cause_id"] for hypothesis in raw["hypotheses"]}
        ledger_hit = any(atom["path"] == oracle["root_path"] and atom["kind"] == "SYMBOL_DEFINITION"
                         and atom["symbol"] == oracle["root_symbol"] for atom in atoms)
        fact_hit = any(atom["path"] == oracle["root_path"] and atom["kind"] != "SYMBOL_DEFINITION"
                       and any(site["line"] <= atom["start_line"] <= site["end_line"] for site in oracle["source_locations"])
                       for atom in atoms)
        def repair_ok(candidate):
            return (candidate["root_cause_id"] in valid and candidate["strategy_kind"] in oracle["repair_strategies"]
                    and bool(candidate["repair_targets"])
                    and all(target["path"] in oracle["target_paths"] for target in candidate["repair_targets"]))
        winner = next((candidate for candidate in raw["candidates"] if candidate["id"] == raw["recommended_candidate_id"]), None)
        rows.append({"case_id": case["id"], "formulation": 1, "raw_trace_archive": str(path),
                     "raw_trace_sha256": path.name[:-8], "expected": oracle,
                     "correct_file_retrieved": oracle["root_path"] in trace["initial_context"]["files"],
                     "correct_symbol_in_ledger": ledger_hit, "correct_symbol_facts_in_ledger": fact_hit,
                     "correct_candidate_exists": bool(valid),
                     "root_correct": bool(selected) and selected <= valid,
                     "repair_correct": any(repair_ok(candidate) for candidate in raw["candidates"]),
                     "recommendation_correct": bool(winner and repair_ok(winner)), "recommended": winner is not None,
                     "failure_reason": raw["failure_reason"], "phase_seconds": None,
                     "measurement_limit": "Completed raw decision recovered; discarded in-memory counters/timings cannot be reconstructed reliably."})
    n = len(rows)
    metrics = {name: rate(sum(bool(row[field]) for row in rows), n, "completed frozen formulations only; incomplete case excluded explicitly")
               for name, field in (("relevant_file_retrieval", "correct_file_retrieved"),
                                   ("correct_symbol_ledger_recall", "correct_symbol_in_ledger"),
                                   ("correct_root_candidate_recall", "correct_candidate_exists"),
                                   ("root_validity", "root_correct"), ("repair_pair_validity", "repair_correct"),
                                   ("recommendation_coverage", "recommended"))}
    recommended = [row for row in rows if row["recommended"]]
    metrics["recommendation_precision"] = rate(sum(row["recommendation_correct"] for row in recommended), len(recommended), "completed recommendations")
    metrics["false_abstention_rate"] = rate(sum(not row["recommended"] for row in rows), n, "completed positive formulations")
    engine_files = ("analysis.py", "causal.py", "grounding.py", "models.py", "policy.py", "recovery.py", "repair.py", "scoring.py", "service.py", "relevance.py")
    source_hashes = {name: hashlib.sha256(subprocess.check_output(["git", "show", f"00cbcd9:tern/orchestrator/predictive/{name}"]).replace(b"\r\n", b"\n")).hexdigest() for name in engine_files}
    target.write_text(json.dumps({"schema_version": 13, "engine_commit": "a0a47093fe62e0e8875367a8e2915e1ec8cb67fe",
        "runtime_source_sha256": source_hashes, "n_total": len(cases), "n_evaluable": n,
        "n_incomplete": len(pending), "pending": pending, "metrics": metrics, "results": rows,
        "preservation": "All preliminary and completed traces retained. Missing timing/call counters not fabricated. Engine runtime files matched original commit throughout frozen process."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
