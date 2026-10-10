"""Verify preserved full traces and report reference grounding independently."""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from tern.orchestrator.predictive.external_evaluation import rate


def audit(report_path: Path, *, snapshots: Path = Path(".orchestrator/external-v12/snapshots")) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for row in report["results"]:
        raw = gzip.decompress(Path(row["raw_trace_archive"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != row["raw_trace_sha256"]:
            raise ValueError("Trace preservado adulterado.")
        decision = json.loads(raw)["decision"]
        root = snapshots/row["case_id"]
        hypotheses = decision.get("hypotheses", [])
        candidates = decision.get("candidates", [])
        refs = sorted({ref for item in (*hypotheses, *candidates, *decision.get("rejected_candidates", []))
                       for ref in item.get("evidence_refs", [])})
        valid = 0
        for ref in refs:
            match = re.fullmatch(r"(.+):(\d+)(?:-(\d+))?", ref)
            if not match:
                continue
            path = root/match.group(1)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                continue
            length = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            start, end = int(match.group(2)), int(match.group(3) or match.group(2))
            valid += 1 <= start <= end <= length
        claims = [claim for item in hypotheses for claim in item.get("claims", [])]
        winner = next((item for item in candidates if item["id"] == decision.get("recommended_candidate_id")), None)
        rows.append({"case_id": row["case_id"], "formulation": row["formulation"],
                     "valid_refs": valid, "refs": len(refs),
                     "unsupported_claims": sum(claim["support"] == "UNSUPPORTED" for claim in claims),
                     "claims": len(claims), "forbidden_recommendations": int(bool(winner and (not winner["eligible"] or "FORBIDDEN_CANDIDATE" in winner["rejection_reasons"]))),
                     "read_only_contract": decision.get("execution_authorized") is False
                     and decision.get("requires_approval") is True and decision.get("dry_run") is True})
    refs, valid = sum(row["refs"] for row in rows), sum(row["valid_refs"] for row in rows)
    claims, unsupported = sum(row["claims"] for row in rows), sum(row["unsupported_claims"] for row in rows)
    return {"source_report": str(report_path), "rows": rows,
            "evidence_reference_validity": rate(valid, refs, "unique references per formulation"),
            "unsupported_claim_rate": rate(unsupported, claims, "claims classified by deterministic grounding; not causal semantic support"),
            "forbidden_recommendations": sum(row["forbidden_recommendations"] for row in rows),
            "read_only_contract": all(row["read_only_contract"] for row in rows),
            "passed": valid == refs and unsupported == 0
            and not any(row["forbidden_recommendations"] for row in rows)
            and all(row["read_only_contract"] for row in rows),
            "note": "Preserved outputs independently verified; no model calls or external project execution. Causal relevance is evaluated separately against oracle."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshots", type=Path, default=Path(".orchestrator/external-v12/snapshots"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Nao sobrescrever auditoria preservada.")
    args.output.write_text(json.dumps({"version": 13, "audits": [audit(path, snapshots=args.snapshots) for path in args.reports]}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
