"""Opt-in, read-only external evaluation. Gold annotations never enter prompts."""
from __future__ import annotations

import cProfile
import gzip
import hashlib
import json
import math
import pstats
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from ..security import PathPolicy
from .analysis import StructuredReasoner
from .causal import build_causal_slice
from .evaluation import CountingReasoner
from .grounding import build_evidence_ledger
from .service import PredictiveDecisionService, build_problem_context


def python_source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Snapshot externo com escape por symlink.")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def snapshot_inventory(root: Path) -> tuple[str, ...]:
    # Acquisition creates Python-only projections. Check namespace changes too:
    # an added .pyc, log, cache or directory is still a project mutation.
    return tuple(sorted(path.relative_to(root).as_posix() + ("/" if path.is_dir() else "")
                        for path in root.rglob("*")))


def load_external_cases(corpus: Path, *, split: str) -> tuple[Mapping[str, Any], ...]:
    if split not in {"development", "holdout", "all"}:
        raise ValueError("Split externo invalido.")
    raw = (corpus / "cases.json").read_bytes()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != manifest["cases_sha256"]:
        raise ValueError("Seal do corpus externo invalido.")
    cases = json.loads(raw)
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or len(cases) != manifest["independent_defects"]:
        raise ValueError("Populacao externa invalida.")
    for case in cases:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case["id"]):
            raise ValueError("Identificador externo invalido.")
        if case["split"] not in {"development", "holdout"}:
            raise ValueError("Split do caso invalido.")
        if not re.fullmatch(r"[0-9a-f]{40}", case["base_commit"]):
            raise ValueError("Commit nao fixado.")
        if not case["problem_formulations"] or not case["oracle"]["source_locations"]:
            raise ValueError("Caso sem input ou adjudicacao.")
    return tuple(sorted((case for case in cases if split == "all" or case["split"] == split),
                        key=lambda case: case["id"]))


def matches_origin(root: Mapping[str, Any], oracle: Mapping[str, Any]) -> bool:
    symbol = root.get("origin_symbol") or ""
    expected = oracle["root_symbol"]
    return (root.get("origin_path") == oracle["root_path"]
            and (symbol == expected or symbol.endswith("." + expected))
            and any(site["line"] <= root.get("origin_line", 0) <= site["end_line"]
                    for site in oracle["source_locations"]))


def rate(numerator: int, denominator: int, population: str) -> dict[str, Any]:
    """Wilson 95% interval, descriptive only for this purposive sample."""
    if not denominator:
        return {"numerator": numerator, "denominator": 0, "rate": None,
                "population": population, "wilson_95": None}
    p, z = numerator / denominator, 1.959963984540054
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1-p) / denominator + z*z / (4*denominator*denominator)) / scale
    return {"numerator": numerator, "denominator": denominator, "rate": p,
            "population": population, "wilson_95": [center-radius, center+radius]}


def summarize_external_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommended = [row for row in rows if row["recommended"]]
    candidates = [row for row in rows if row["correct_candidate_exists"]]
    live = [row for row in rows if row["mode"] == "live"]
    selected_scc = [row for row in live if row["selected_scc"]]
    metrics = {
        "relevant_file_retrieval": rate(sum(row["correct_file_retrieved"] for row in rows), len(rows), "formulations"),
        "correct_candidate_recall": rate(sum(row["correct_candidate_exists"] for row in rows), len(rows), "formulations"),
        "external_root_validity": rate(sum(row["root_correct"] for row in live), len(live), "live formulations including abstentions"),
        "root_selection_given_correct_candidate": rate(sum(row["root_correct"] for row in candidates if row["mode"] == "live"), sum(row["mode"] == "live" for row in candidates), "live formulations with oracle-location candidate"),
        "repair_pair_validity": rate(sum(row["repair_correct"] for row in live), len(live), "live formulations including abstentions"),
        "recommendation_correctness": rate(sum(row["recommendation_correct"] for row in recommended), len(recommended), "recommendations only"),
        "recommendation_coverage": rate(len(recommended), len(live), "live formulations"),
        "false_abstention_rate": rate(sum(not row["recommended"] and not row["abstention_expected"] for row in live), sum(not row["abstention_expected"] for row in live), "live positive formulations"),
        "correct_abstention": rate(sum(not row["recommended"] and row["abstention_expected"] for row in live), sum(row["abstention_expected"] for row in live), "true-insufficient formulations"),
        "irrelevant_SCC_selection_rate": rate(sum(row["selected_scc"] and not row["import_causal"] for row in live), len(live), "live formulations; independently adjudicated import causality"),
        "causal_relevance_precision": rate(sum(row["root_correct"] for row in live if row["selected_root"]), sum(row["selected_root"] for row in live), "selected roots; oracle-location proxy, NOT semantic entailment"),
        "behavioral_defect_root_validity": rate(sum(row["root_correct"] for row in live if row["category"] == "BEHAVIORAL_MISMATCH"), sum(row["category"] == "BEHAVIORAL_MISMATCH" for row in live), "live behavioral formulations"),
        "SCC_relevance_precision": rate(sum(row["import_causal"] for row in selected_scc), len(selected_scc), "selected SCCs"),
    }
    groups: dict[str, list] = {}
    for row in live:
        groups.setdefault(row["case_id"], []).append(row)
    metrics["independent_defect_root_validity"] = rate(sum(all(row["root_correct"] for row in group) for group in groups.values()), len(groups), "independent defects; all formulations must match")
    return {"metrics": metrics, "n_independent_defects": len({row["case_id"] for row in rows}),
            "n_formulations": len(rows), "n_recommended": len(recommended),
            "n_abstained": len(live)-len(recommended),
            "failure_code_counts": dict(sorted(Counter(code for row in rows for code in row["failure_codes"]).items())),
            "qwen_requests": sum(row["qwen"]["requests"] for row in rows),
            "avg_qwen_seconds": (sum(sum(row["qwen"]["request_ms"]) for row in rows)/1000 /
                                 sum(row["qwen"]["requests"] for row in rows)) if any(row["qwen"]["requests"] for row in rows) else None,
            "safety": {key: sum(row["safety"][key] for row in rows) for key in (
                "filesystem_mutations", "tool_dispatches", "authority_grants", "execution_authorized", "destructive_actions")},
            "limitations": ["Purposive small sample; intervals do not establish population readiness.",
                            "Root and repair metrics use adjudicated source locations and strategy IDs, not an LLM judge.",
                            "No third-party tests executed; Python-only snapshots omit non-Python assets.",
                            "Zero selected roots is undefined relevance precision, not perfect grounding."]}


def evaluate_external_cases(
    cases: Sequence[Mapping[str, Any]], snapshots: Path, archive: Path,
    *, reasoner: StructuredReasoner | None = None,
) -> dict[str, Any]:
    rows = []
    if archive.resolve().is_relative_to(snapshots.resolve()):
        raise ValueError("Artefatos nao podem modificar snapshots externos.")
    archive.mkdir(parents=True, exist_ok=True, mode=0o777)
    for case in cases:
        root = (snapshots / case["id"]).resolve(strict=True)
        if not root.is_relative_to(snapshots.resolve(strict=True)):
            raise ValueError("Snapshot fora da raiz autorizada.")
        before = python_source_hash(root)
        before_inventory = snapshot_inventory(root)
        if before != case["source_python_sha256"]:
            raise ValueError("Snapshot externo adulterado.")
        policy = PathPolicy((root,))
        snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
        for index, problem in enumerate(case["problem_formulations"]):
            context = build_problem_context(problem, snapshot, ProjectCandidateGenerator().generate(problem, snapshot), policy)
            context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
            initial_slice, initial_roots = build_causal_slice(context, snapshot, policy)
            measured = CountingReasoner(reasoner) if reasoner else None
            profile = cProfile.Profile()
            started = time.perf_counter()
            report = profile.runcall(PredictiveDecisionService(measured, path_policy=policy).predict, problem, root) if measured else None
            wall = time.perf_counter()-started
            raw = report.as_dict() if report else {"root_cause_candidates": [item.as_dict() for item in initial_roots]}
            roots = raw["root_cause_candidates"]
            selected_ids = {item["root_cause_id"] for item in raw.get("hypotheses", [])}
            selected = [item for item in roots if item["id"] in selected_ids]
            oracle = case["oracle"]
            correct = [item for item in roots if matches_origin(item, oracle)]
            valid_ids = {item["id"] for item in correct}
            candidates = raw.get("candidates", [])
            def repair_ok(candidate):
                return (candidate.get("root_cause_id") in valid_ids
                        and candidate["strategy_kind"] in oracle["repair_strategies"]
                        and bool(candidate.get("repair_targets"))
                        and all(target["path"] in oracle["target_paths"] for target in candidate["repair_targets"]))
            winner = next((item for item in candidates if item["id"] == raw.get("recommended_candidate_id")), None)
            root_correct = bool(selected) and all(item["id"] in valid_ids for item in selected)
            stats = pstats.Stats(profile).stats if report else {}
            def phase(filename, function):
                return sum(item[3] for (path, _line, name), item in stats.items()
                           if path.endswith(filename) and name == function)
            safety = {"filesystem_mutations": int(before != python_source_hash(root)
                      or before_inventory != snapshot_inventory(root)),
                      "tool_dispatches": measured.tool_dispatches if measured else 0,
                      "authority_grants": 0, "execution_authorized": int(report.execution_authorized) if report else 0,
                      "destructive_actions": int(any(item.strategy_kind == "OTHER" and not item.eligible for item in report.candidates)) if report else 0}
            if any(safety.values()):
                raise AssertionError("Invariante read-only violada.")
            codes = []
            if oracle["root_path"] not in context.related_files: codes.append("RELEVANT_FILE_MISSED")
            if not correct: codes.append("CORRECT_ROOT_CANDIDATE_MISSING")
            elif report and not root_correct: codes.append("ROOT_SELECTION_ERROR")
            if report and winner is None and not oracle["abstention_expected"]: codes.append("FALSE_ABSTENTION")
            if report and any(item.get("import_scc") for item in selected) and not oracle["import_causal"]: codes.append("IRRELEVANT_SCC_SELECTED")
            if winner and not repair_ok(winner): codes.append("WRONG_REPAIR_PAIR")
            blob = json.dumps({"decision": raw, "initial_context": context.as_dict() if hasattr(context, "as_dict") else {"files": list(context.related_files), "ledger": context.evidence_ledger.as_dict()}, "expected": oracle}, ensure_ascii=False, sort_keys=True).encode()
            sha = hashlib.sha256(blob).hexdigest()
            (archive / (sha + ".json.gz")).write_bytes(gzip.compress(blob, mtime=0))
            row = {"case_id": case["id"], "formulation": index+1, "mode": "live" if report else "structural", "category": case["category"],
                   "correct_file_retrieved": oracle["root_path"] in context.related_files,
                   "initial_files": list(context.related_files), "initial_correct_candidate": any(matches_origin(item.as_dict(), oracle) for item in initial_roots),
                   "correct_symbol_in_ledger": any(atom.path == oracle["root_path"] and atom.symbol and (atom.symbol == oracle["root_symbol"] or atom.symbol.endswith("."+oracle["root_symbol"])) for atom in context.evidence_ledger.atoms),
                   "correct_candidate_exists": bool(correct), "selected_root": bool(selected), "root_correct": root_correct,
                   "selected_scc": any(item.get("import_scc") for item in selected), "import_causal": oracle["import_causal"],
                   "selected_roots": [{key: item[key] for key in ("id", "cause_kind", "origin_path", "origin_symbol", "origin_line")} for item in selected],
                   "root_candidates": [{key: item[key] for key in ("id", "cause_kind", "origin_path", "origin_symbol", "origin_line")} for item in roots],
                   "repair_correct": any(repair_ok(item) for item in candidates), "recommendation_correct": bool(winner and repair_ok(winner)),
                   "recommended": winner is not None, "abstention_expected": oracle["abstention_expected"], "winner": winner,
                   "failure_reason": raw.get("failure_reason"), "failure_codes": codes, "wall_seconds": wall,
                   "phase_seconds": {"index": phase("project_intelligence_v2.py", "build"), "retrieval": phase("project_intelligence_v2.py", "generate"), "ledger": phase("grounding.py", "build_evidence_ledger"), "causal": phase("causal.py", "build_causal_slice"), "recovery_inclusive": phase("recovery.py", "recover"), "scoring": phase("scoring.py", "ranking_decision")},
                   "qwen": {"requests": measured.requests if measured else 0, "request_ms": measured.request_ms if measured else [], "prompt_tokens": measured.prompt_tokens if measured else 0, "response_tokens": measured.response_tokens if measured else 0},
                   "safety": safety, "raw_trace_sha256": sha, "raw_trace_archive": str(archive / (sha + ".json.gz")), "expected": oracle}
            from .coverage_evaluation import layer_coverage
            row["layer_coverage"] = layer_coverage(snapshot, context, initial_slice, initial_roots,
                                                   oracle, decision=report)
            rows.append(row)
            print(json.dumps({"id": case["id"], "root_correct": root_correct, "candidate_exists": bool(correct), "failures": codes}), flush=True)
    from .coverage_evaluation import summarize_layer_coverage
    summary = summarize_external_results(rows)
    summary["layer_coverage"] = summarize_layer_coverage(rows)
    return {"schema_version": 13, "summary": summary, "results": rows}
