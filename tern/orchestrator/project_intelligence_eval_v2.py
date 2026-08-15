from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import LlamaClient
from .project_intelligence import ProjectFileSelector, ProjectSnapshotBuilder
from .project_intelligence_v2 import (
    ProjectCandidateGenerator,
    ProjectCandidateRanker,
    ProjectIndexBuilderV2,
    ProjectSnapshotV2,
    RelevantFileEvidenceSource,
)


SOURCE_ERROR_CLASS = {
    RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE: "MISSED_EXPLICIT_FILE",
    RelevantFileEvidenceSource.EXPLICIT_DIRECTORY_REFERENCE: "SCOPE_EXPANSION",
    RelevantFileEvidenceSource.EXPLICIT_SYMBOL_REFERENCE: "MISSED_EXPLICIT_SYMBOL",
    RelevantFileEvidenceSource.SYMBOL_DEFINITION: "MISSED_SYMBOL_DEFINITION",
    RelevantFileEvidenceSource.SYMBOL_REFERENCE: "MISSED_SYMBOL_DEFINITION",
    RelevantFileEvidenceSource.IMPORT_DEPENDENCY: "MISSED_DEPENDENCY",
    RelevantFileEvidenceSource.REVERSE_IMPORT: "MISSED_CALLER",
    RelevantFileEvidenceSource.TEST_RELATIONSHIP: "MISSED_TEST",
    RelevantFileEvidenceSource.TRACEBACK_REFERENCE: "MISSED_TRACEBACK_FILE",
    RelevantFileEvidenceSource.ERROR_REFERENCE: "MISSED_TRACEBACK_FILE",
    RelevantFileEvidenceSource.GIT_MODIFIED_FILE: "MISSED_MODIFIED_FILE",
    RelevantFileEvidenceSource.GIT_DIFF_RELATIONSHIP: "MISSED_MODIFIED_FILE",
    RelevantFileEvidenceSource.CONFIG_RELATIONSHIP: "WRONG_MODULE",
    RelevantFileEvidenceSource.ENTRYPOINT_RELATIONSHIP: "WRONG_MODULE",
    RelevantFileEvidenceSource.PROJECT_STRUCTURE: "WRONG_MODULE",
    RelevantFileEvidenceSource.SEMANTIC_INFERENCE: "FALSE_SEMANTIC_MATCH",
}


BREAKDOWN_SOURCES = {
    "explicit_file_recall": {RelevantFileEvidenceSource.EXPLICIT_FILE_REFERENCE},
    "explicit_symbol_recall": {RelevantFileEvidenceSource.EXPLICIT_SYMBOL_REFERENCE},
    "symbol_definition_recall": {RelevantFileEvidenceSource.SYMBOL_DEFINITION},
    "dependency_recall": {RelevantFileEvidenceSource.IMPORT_DEPENDENCY},
    "reverse_dependency_recall": {RelevantFileEvidenceSource.REVERSE_IMPORT},
    "test_relationship_recall": {RelevantFileEvidenceSource.TEST_RELATIONSHIP},
    "traceback_file_recall": {RelevantFileEvidenceSource.TRACEBACK_REFERENCE},
    "modified_file_recall": {
        RelevantFileEvidenceSource.GIT_MODIFIED_FILE,
        RelevantFileEvidenceSource.GIT_DIFF_RELATIONSHIP,
    },
    "config_file_recall": {RelevantFileEvidenceSource.CONFIG_RELATIONSHIP},
    "entrypoint_recall": {RelevantFileEvidenceSource.ENTRYPOINT_RELATIONSHIP},
    "semantic_only_recall": {RelevantFileEvidenceSource.SEMANTIC_INFERENCE},
}


@dataclass(frozen=True)
class ProjectIntelligenceCase:
    id: str
    category: str
    task: str
    requirements: Mapping[str, Any]
    required_files: frozenset[str]
    relevant_files: frozenset[str]
    optional_supporting_files: frozenset[str]
    known_irrelevant_files: frozenset[str]
    expected_evidence: Mapping[str, frozenset[RelevantFileEvidenceSource]]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectIntelligenceCase":
        return cls(
            str(value["id"]),
            str(value["category"]),
            str(value["input"]),
            dict(value.get("requirements") or {}),
            frozenset(str(item) for item in value.get("required_files") or ()),
            frozenset(str(item) for item in value.get("relevant_files") or ()),
            frozenset(
                str(item) for item in value.get("optional_supporting_files") or ()
            ),
            frozenset(str(item) for item in value.get("known_irrelevant_files") or ()),
            {
                str(path): frozenset(
                    RelevantFileEvidenceSource(item) for item in sources
                )
                for path, sources in (value.get("expected_evidence") or {}).items()
            },
        )

    @property
    def relevant_or_required(self) -> frozenset[str]:
        return self.required_files | self.relevant_files

    @property
    def accepted(self) -> frozenset[str]:
        return self.relevant_or_required | self.optional_supporting_files


def load_project_intelligence_cases(
    path: str | Path,
) -> list[ProjectIntelligenceCase]:
    cases = [
        ProjectIntelligenceCase.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate project intelligence case id")
    return cases


@dataclass(frozen=True)
class ObservedSelection:
    selected_files: tuple[str, ...]
    candidate_files: tuple[str, ...]
    evidence: Mapping[str, frozenset[RelevantFileEvidenceSource]]
    valid: bool
    latency_ms: float
    semantic_ranking_ms: float
    context_bytes: int
    context_tokens: int
    selected_file_bytes: int
    context_budget_violation: bool
    hard_evidence_dropped: int
    semantic_ranking_used: bool
    raw: Any = None


def observe_baseline(
    cases: Sequence[ProjectIntelligenceCase],
    *,
    endpoint: str,
    project_path: str | Path,
    request_timeout: int = 120,
) -> tuple[list[ObservedSelection], dict[str, Any]]:
    started = time.perf_counter()
    snapshot = ProjectSnapshotBuilder(project_path).build()
    compact = snapshot.compact()
    build_ms = (time.perf_counter() - started) * 1000
    context_bytes = len(json.dumps(compact, ensure_ascii=False).encode("utf-8"))
    sizes = {item.path: item.size for item in snapshot.repo_map}
    selector = ProjectFileSelector(LlamaClient(endpoint, timeout=request_timeout))
    results: list[ObservedSelection] = []
    for index, case in enumerate(cases, start=1):
        print(f"baseline {index}/{len(cases)} {case.id}", flush=True)
        result = selector.select(case.task, compact)
        selected_bytes = sum(sizes.get(path, 0) for path in result.selected_files)
        results.append(
            ObservedSelection(
                result.selected_files,
                result.selected_files,
                {},
                result.valid,
                result.latency_ms,
                result.latency_ms,
                context_bytes,
                math.ceil(context_bytes / 4),
                selected_bytes,
                context_bytes > 24_000,
                0,
                True,
                result,
            )
        )
    return results, {
        "repo_scan_time_ms": round(build_ms, 3),
        "index_build_time_ms": round(build_ms, 3),
        "files_discovered": len(snapshot.repo_map),
        "files_reused": snapshot.files_reused,
        "files_indexed": snapshot.files_analyzed,
        "snapshot_context_bytes": context_bytes,
    }


def observe_candidate(
    cases: Sequence[ProjectIntelligenceCase],
    *,
    project_path: str | Path,
    cache_path: str | Path | None = None,
    endpoint: str | None = None,
    request_timeout: int = 120,
) -> tuple[list[ObservedSelection], dict[str, Any], ProjectSnapshotV2]:
    snapshot = ProjectIndexBuilderV2(project_path, cache_path=cache_path).build()
    ranker = (
        ProjectCandidateRanker(LlamaClient(endpoint, timeout=request_timeout))
        if endpoint
        else None
    )
    generator = ProjectCandidateGenerator(ranker=ranker)
    results: list[ObservedSelection] = []
    for index, case in enumerate(cases, start=1):
        print(f"candidate {index}/{len(cases)} {case.id}", flush=True)
        result = generator.generate(
            case.task,
            snapshot,
            requirements=case.requirements,
        )
        evidence = {
            candidate.path: frozenset(item.source for item in candidate.evidences)
            for candidate in result.candidates
        }
        results.append(
            ObservedSelection(
                result.selected_files,
                tuple(item.path for item in result.candidates),
                evidence,
                True,
                result.metrics.generation_ms,
                result.metrics.semantic_ranking_ms,
                result.metrics.context_bytes,
                result.metrics.estimated_context_tokens,
                result.metrics.selected_file_bytes,
                result.metrics.context_budget_violation,
                result.metrics.hard_evidence_dropped,
                result.metrics.semantic_ranking_used,
                result,
            )
        )
    metrics = snapshot.metrics
    return results, {
        "repo_scan_time_ms": metrics.repo_scan_ms,
        "index_build_time_ms": metrics.index_build_ms,
        "incremental_update_time_ms": metrics.incremental_update_ms,
        "files_discovered": metrics.files_discovered,
        "files_reused": metrics.files_reused,
        "files_indexed": metrics.files_indexed,
        "bytes_hashed": metrics.bytes_hashed,
        "bytes_parsed": metrics.bytes_parsed,
        "symbol_count": sum(len(item.symbols) for item in snapshot.files),
        "import_edge_count": sum(len(item) for item in snapshot.import_graph.values()),
        "test_relationship_count": len(snapshot.test_relationships),
    }, snapshot


def _div(numerator: int | float, denominator: int | float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def _missing_error_class(
    case: ProjectIntelligenceCase,
    path: str,
) -> str:
    sources = case.expected_evidence.get(path, frozenset())
    if sources:
        strongest = max(
            sources,
            key=lambda item: {
                "MISSED_EXPLICIT_FILE": 10,
                "MISSED_TRACEBACK_FILE": 9,
                "MISSED_EXPLICIT_SYMBOL": 8,
                "MISSED_SYMBOL_DEFINITION": 7,
                "MISSED_TEST": 6,
                "MISSED_DEPENDENCY": 5,
                "MISSED_CALLER": 4,
                "MISSED_MODIFIED_FILE": 3,
                "WRONG_MODULE": 2,
                "FALSE_SEMANTIC_MATCH": 1,
                "SCOPE_EXPANSION": 0,
            }[SOURCE_ERROR_CLASS[item]],
        )
        return SOURCE_ERROR_CLASS[strongest]
    return "WRONG_MODULE"


def evaluate_project_intelligence(
    cases: Sequence[ProjectIntelligenceCase],
    observations: Sequence[ObservedSelection],
    *,
    variant: str,
    snapshot_files: frozenset[str],
    index_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if len(cases) != len(observations):
        raise ValueError("cases and observations must have equal length")
    required_hit = required_total = relevant_hit = relevant_total = 0
    accepted_selected = selected_total = irrelevant_selected = 0
    provenance_hit = provenance_total = 0
    valid = 0
    cross_project_leakage = 0
    context_violations = hard_dropped = semantic_calls = 0
    latencies: list[float] = []
    semantic_latencies: list[float] = []
    candidate_counts: list[int] = []
    selected_counts: list[int] = []
    context_bytes: list[int] = []
    context_tokens: list[int] = []
    selected_bytes: list[int] = []
    breakdown = {
        name: {"hit": 0, "total": 0}
        for name in BREAKDOWN_SOURCES
    }
    errors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for case, observed in zip(cases, observations):
        selected = frozenset(observed.selected_files)
        required_hit += len(selected & case.required_files)
        required_total += len(case.required_files)
        relevant_hit += len(selected & case.relevant_or_required)
        relevant_total += len(case.relevant_or_required)
        accepted_selected += len(selected & case.accepted)
        selected_total += len(selected)
        extras = selected - case.accepted
        irrelevant_selected += len(extras)
        valid += int(observed.valid)
        cross_project_leakage += len(selected - snapshot_files)
        context_violations += int(observed.context_budget_violation)
        hard_dropped += observed.hard_evidence_dropped
        semantic_calls += int(observed.semantic_ranking_used)
        latencies.append(observed.latency_ms)
        semantic_latencies.append(observed.semantic_ranking_ms)
        candidate_counts.append(len(observed.candidate_files))
        selected_counts.append(len(selected))
        context_bytes.append(observed.context_bytes)
        context_tokens.append(observed.context_tokens)
        selected_bytes.append(observed.selected_file_bytes)

        for path, expected_sources in case.expected_evidence.items():
            actual_sources = observed.evidence.get(path, frozenset())
            for source in expected_sources:
                provenance_total += 1
                provenance_hit += int(source in actual_sources)
                for name, sources in BREAKDOWN_SOURCES.items():
                    if source in sources:
                        breakdown[name]["total"] += 1
                        breakdown[name]["hit"] += int(path in selected)
        for path in sorted(case.relevant_or_required - selected):
            errors.append(
                {
                    "case_id": case.id,
                    "task": case.task,
                    "expected_file": path,
                    "actual_candidates": list(observed.candidate_files),
                    "error_class": _missing_error_class(case, path),
                    "stage": variant,
                }
            )
        for path in sorted(extras):
            sources = observed.evidence.get(path, frozenset())
            error_class = (
                "FALSE_SEMANTIC_MATCH"
                if variant == "baseline"
                or RelevantFileEvidenceSource.SEMANTIC_INFERENCE in sources
                else "UNRELATED_NEIGHBOR_FILE"
                if RelevantFileEvidenceSource.PROJECT_STRUCTURE in sources
                else "SCOPE_EXPANSION"
            )
            errors.append(
                {
                    "case_id": case.id,
                    "task": case.task,
                    "expected_file": None,
                    "actual_file": path,
                    "actual_candidates": list(observed.candidate_files),
                    "error_class": error_class,
                    "stage": variant,
                }
            )
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "required_files": sorted(case.required_files),
                "relevant_files": sorted(case.relevant_files),
                "optional_supporting_files": sorted(case.optional_supporting_files),
                "selected_files": list(observed.selected_files),
                "candidate_files": list(observed.candidate_files),
                "evidence": {
                    path: sorted(source.value for source in sources)
                    for path, sources in observed.evidence.items()
                },
                "valid": observed.valid,
                "latency_ms": observed.latency_ms,
                "context_bytes": observed.context_bytes,
                "context_tokens": observed.context_tokens,
            }
        )

    precision = _div(accepted_selected, selected_total, 1.0)
    recall = _div(relevant_hit, relevant_total, 1.0)
    f1 = _div(2 * precision * recall, precision + recall, 0.0)
    rendered_breakdown = {
        name: {
            "recall": (
                _div(value["hit"], value["total"])
                if value["total"]
                else None
            ),
            "hit": value["hit"],
            "support": value["total"],
        }
        for name, value in breakdown.items()
    }
    return {
        "variant": variant,
        "cases": len(cases),
        "dry_run": True,
        "execution_count": 0,
        "metrics": {
            "required_file_recall": _div(required_hit, required_total, 1.0),
            "relevant_file_recall": recall,
            "precision": precision,
            "f1": f1,
            "irrelevant_file_selection_rate": _div(
                irrelevant_selected, selected_total, 0.0
            ),
            "explicit_reference_preservation": rendered_breakdown[
                "explicit_file_recall"
            ]["recall"],
            "symbol_resolution_accuracy": rendered_breakdown[
                "symbol_definition_recall"
            ]["recall"],
            "traceback_resolution_accuracy": rendered_breakdown[
                "traceback_file_recall"
            ]["recall"],
            "test_relationship_accuracy": rendered_breakdown[
                "test_relationship_recall"
            ]["recall"],
            "dependency_relationship_accuracy": _div(
                rendered_breakdown["dependency_recall"]["hit"]
                + rendered_breakdown["reverse_dependency_recall"]["hit"],
                rendered_breakdown["dependency_recall"]["support"]
                + rendered_breakdown["reverse_dependency_recall"]["support"],
                1.0,
            ),
            "provenance_accuracy": _div(
                provenance_hit, provenance_total, 1.0 if variant != "baseline" else 0.0
            ),
            "cross_project_file_leakage": cross_project_leakage,
            "context_budget_violations": context_violations,
            "hard_evidence_dropped": hard_dropped,
            "json_validity": _div(valid, len(cases), 1.0),
            "semantic_ranking_calls": semantic_calls,
            "candidate_count_mean": statistics.fmean(candidate_counts)
            if candidate_counts
            else 0.0,
            "selected_count_mean": statistics.fmean(selected_counts)
            if selected_counts
            else 0.0,
            "selected_file_bytes_mean": statistics.fmean(selected_bytes)
            if selected_bytes
            else 0.0,
            "context_bytes_mean": statistics.fmean(context_bytes)
            if context_bytes
            else 0.0,
            "context_tokens_mean": statistics.fmean(context_tokens)
            if context_tokens
            else 0.0,
            "candidate_generation_p50_ms": _percentile(latencies, 0.50),
            "candidate_generation_p90_ms": _percentile(latencies, 0.90),
            "candidate_generation_p95_ms": _percentile(latencies, 0.95),
            "semantic_ranking_time_ms": sum(semantic_latencies),
        },
        "breakdown": rendered_breakdown,
        "index_metrics": dict(index_metrics),
        "error_taxonomy": errors,
        "error_counts": dict(Counter(item["error_class"] for item in errors)),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project Intelligence V1/V2 evaluator (dry-run only)"
    )
    parser.add_argument(
        "--corpus",
        default="tests/data/project_intelligence_v2_diagnostic.jsonl",
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--variant", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument("--cache")
    parser.add_argument("--output")
    parser.add_argument("--request-timeout", type=int, default=120)
    args = parser.parse_args(argv)
    cases = load_project_intelligence_cases(args.corpus)
    snapshot_v2 = ProjectIndexBuilderV2(args.project, cache_path=args.cache).build()
    snapshot_files = frozenset(snapshot_v2.file_index)
    report: dict[str, Any] = {}
    if args.variant in {"baseline", "both"}:
        if not args.endpoint:
            parser.error("--endpoint is required for the live baseline")
        observed, index_metrics = observe_baseline(
            cases,
            endpoint=args.endpoint,
            project_path=args.project,
            request_timeout=args.request_timeout,
        )
        report["baseline"] = evaluate_project_intelligence(
            cases,
            observed,
            variant="baseline",
            snapshot_files=snapshot_files,
            index_metrics=index_metrics,
        )
    if args.variant in {"candidate", "both"}:
        observed, index_metrics, candidate_snapshot = observe_candidate(
            cases,
            project_path=args.project,
            cache_path=args.cache,
            endpoint=args.endpoint,
            request_timeout=args.request_timeout,
        )
        report["candidate"] = evaluate_project_intelligence(
            cases,
            observed,
            variant="candidate",
            snapshot_files=frozenset(candidate_snapshot.file_index),
            index_metrics=index_metrics,
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
