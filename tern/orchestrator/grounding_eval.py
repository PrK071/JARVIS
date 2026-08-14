from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .autonomy_eval import diagnostic_baseline
from .autonomy_foundation import (
    Agent,
    Capability,
    EligibilityEngine,
    TaskRequirementAnalyzer,
    TaskRequirements,
    propose_agent_selection,
)
from .client import LlamaClient
from .explicit_agent_binding import detect_explicit_agent_binding
from .project_intelligence import ProjectSnapshotBuilder
from .task_requirement_grounding import (
    MUTATION_REQUIRED,
    REQUIREMENT_DIMENSIONS,
    ExplicitTaskEvidenceExtractor,
    GroundedEligibilityEngine,
    GroundedRequirement,
    GroundedRequirementAnalysisResult,
    GroundedTaskRequirementAnalyzer,
    GroundedTaskRequirements,
    RequirementEvidenceSource,
    RequirementValue,
    requirement_safety_class,
)


CRITICAL_REQUIREMENTS = frozenset(
    {
        MUTATION_REQUIRED,
        Capability.MUTATION.value,
        Capability.REPOSITORY_WRITE.value,
        Capability.FILESYSTEM_WRITE.value,
        Capability.CODE_EDIT.value,
        Capability.TEST_EXECUTION.value,
        Capability.WEB_ACCESS.value,
    }
)


@dataclass(frozen=True)
class ExpectedRequirement:
    value: RequirementValue
    source: RequirementEvidenceSource


@dataclass(frozen=True)
class GroundingCase:
    id: str
    category: str
    input: str
    expected: Mapping[str, ExpectedRequirement]
    expected_eligible_agents: frozenset[Agent]
    expected_requested_agent: Agent | None
    expected_ambiguity: bool
    expected_prohibitions: frozenset[str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundingCase":
        expected_block = value["expected"]
        overrides = expected_block.get("requirements") or {}
        expected: dict[str, ExpectedRequirement] = {}
        for name in REQUIREMENT_DIMENSIONS:
            item = overrides.get(name)
            expected[name] = (
                ExpectedRequirement(
                    RequirementValue(item["value"]),
                    RequirementEvidenceSource(item["source"]),
                )
                if item
                else ExpectedRequirement(
                    RequirementValue.UNKNOWN,
                    RequirementEvidenceSource.INSUFFICIENT_EVIDENCE,
                )
            )
        return cls(
            str(value["id"]),
            str(value["category"]),
            str(value["input"]),
            expected,
            frozenset(Agent(item) for item in expected_block["eligible_agents"]),
            Agent(expected_block["requested_agent"])
            if expected_block.get("requested_agent")
            else None,
            bool(expected_block.get("ambiguity_material", False)),
            frozenset(expected_block.get("prohibitions") or ()),
        )


def load_grounding_cases(path: str | Path) -> list[GroundingCase]:
    return [
        GroundingCase.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class RequirementPrediction:
    value: RequirementValue
    source: RequirementEvidenceSource


def legacy_predictions(
    requirements: TaskRequirements | None,
) -> dict[str, RequirementPrediction]:
    if requirements is None:
        return {
            name: RequirementPrediction(
                RequirementValue.UNKNOWN,
                RequirementEvidenceSource.INSUFFICIENT_EVIDENCE,
            )
            for name in REQUIREMENT_DIMENSIONS
        }
    result = {
        item.value: RequirementPrediction(
            RequirementValue.TRUE
            if item in requirements.capabilities
            else RequirementValue.FALSE,
            RequirementEvidenceSource.SEMANTIC_INFERENCE,
        )
        for item in Capability
    }
    result[MUTATION_REQUIRED] = RequirementPrediction(
        RequirementValue.TRUE if requirements.mutation_required else RequirementValue.FALSE,
        RequirementEvidenceSource.SEMANTIC_INFERENCE,
    )
    result["read_only_required"] = RequirementPrediction(
        RequirementValue.TRUE if requirements.read_only_required else RequirementValue.FALSE,
        RequirementEvidenceSource.SEMANTIC_INFERENCE,
    )
    return result


def grounded_predictions(
    requirements: GroundedTaskRequirements | None,
) -> dict[str, RequirementPrediction]:
    if requirements is None:
        return legacy_predictions(None)
    return {
        name: RequirementPrediction(item.value, item.source)
        for name, item in requirements.requirements.items()
    }


def _safe_div(numerator: int | float, denominator: int | float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((len(ordered) - 1) * quantile)
    return ordered[index]


def _error_class(
    expected: ExpectedRequirement,
    actual: RequirementPrediction,
) -> str:
    if actual.value is RequirementValue.CONFLICT:
        return "CONTRADICTION"
    if expected.source in {
        RequirementEvidenceSource.EXPLICIT_USER,
        RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
    } and actual.source not in {
        RequirementEvidenceSource.EXPLICIT_USER,
        RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
    }:
        return "EXPLICIT_EVIDENCE_IGNORED"
    if (
        expected.source is RequirementEvidenceSource.PROJECT_FACT
        and actual.source is not RequirementEvidenceSource.PROJECT_FACT
    ):
        return "PROJECT_FACT_IGNORED"
    if expected.value is RequirementValue.UNKNOWN and actual.value in {
        RequirementValue.TRUE,
        RequirementValue.FALSE,
    }:
        return "UNKNOWN_SHOULD_HAVE_BEEN_USED"
    if expected.value is RequirementValue.TRUE and actual.value is RequirementValue.FALSE:
        return "FALSE_NOT_REQUIRED"
    if expected.value is RequirementValue.TRUE:
        return "MISSING_REQUIREMENT"
    if expected.value is RequirementValue.FALSE and actual.value is RequirementValue.TRUE:
        return (
            "CONSTRAINT_IGNORED"
            if expected.source is RequirementEvidenceSource.EXPLICIT_USER_NEGATION
            else "FALSE_REQUIRED"
        )
    return "OVERCONFIDENT_INFERENCE"


class GroundingEvaluator:
    def __init__(self, baseline: Any | None = None):
        self.baseline = baseline or diagnostic_baseline()

    def evaluate_variant(
        self,
        cases: Sequence[GroundingCase],
        analyses: Sequence[Any],
        *,
        variant: str,
    ) -> dict[str, Any]:
        if len(cases) != len(analyses):
            raise ValueError("cases and analyses must have equal length")
        per_field = {
            name: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "correct": 0,
                "total": 0,
                "unknown_predicted": 0,
                "definite_expected": 0,
            }
            for name in REQUIREMENT_DIMENSIONS
        }
        errors: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        provenance_correct = explicit_total = explicit_correct = 0
        deterministic_total = deterministic_correct = 0
        semantic_total = semantic_correct = 0
        expected_unknown = predicted_unknown = correct_unknown = 0
        overconfident = 0
        critical_tp = critical_fn = critical_fp = 0
        eligibility_tp = eligibility_fp = eligibility_fn = 0
        single_total = single_correct = no_total = no_correct = multi_total = multi_correct = 0
        constraint_total = constraint_correct = 0
        prohibition_total = prohibition_correct = 0
        ambiguity_correct = 0
        requested_total = requested_correct = 0
        false_mutation_authorizations = 0
        invalid = 0
        latencies: list[float] = []
        prompt_tokens = completion_tokens = retries = inference_count = 0
        legacy_engine = EligibilityEngine()
        grounded_engine = GroundedEligibilityEngine()

        for case, analysis in zip(cases, analyses):
            requirements = analysis.requirements
            invalid += int(not analysis.valid)
            latencies.append(float(analysis.latency_ms))
            prompt_tokens += analysis.prompt_tokens or 0
            completion_tokens += analysis.generated_tokens or 0
            retries += int(analysis.attempts > 1)
            inference_count += getattr(analysis, "inference_count", analysis.attempts)
            predictions = (
                legacy_predictions(requirements)
                if variant == "baseline"
                else grounded_predictions(requirements)
            )

            for name in REQUIREMENT_DIMENSIONS:
                expected = case.expected[name]
                actual = predictions[name]
                stats = per_field[name]
                stats["total"] += 1
                stats["correct"] += int(expected.value == actual.value)
                stats["unknown_predicted"] += int(actual.value is RequirementValue.UNKNOWN)
                stats["definite_expected"] += int(expected.value is not RequirementValue.UNKNOWN)
                expected_true = expected.value is RequirementValue.TRUE
                actual_true = actual.value is RequirementValue.TRUE
                stats["tp"] += int(expected_true and actual_true)
                stats["fp"] += int(not expected_true and actual_true)
                stats["fn"] += int(expected_true and not actual_true)
                provenance_correct += int(
                    expected.value == actual.value and expected.source == actual.source
                )
                if expected.source in {
                    RequirementEvidenceSource.EXPLICIT_USER,
                    RequirementEvidenceSource.EXPLICIT_USER_NEGATION,
                }:
                    explicit_total += 1
                    explicit_correct += int(
                        expected.value == actual.value and expected.source == actual.source
                    )
                    if expected.source is RequirementEvidenceSource.EXPLICIT_USER_NEGATION:
                        constraint_total += 1
                        constraint_correct += int(expected.value == actual.value)
                if expected.source in {
                    RequirementEvidenceSource.PROJECT_FACT,
                    RequirementEvidenceSource.RUNTIME_FACT,
                }:
                    deterministic_total += 1
                    deterministic_correct += int(
                        expected.value == actual.value and expected.source == actual.source
                    )
                if expected.source is RequirementEvidenceSource.SEMANTIC_INFERENCE:
                    semantic_total += 1
                    semantic_correct += int(expected.value == actual.value)
                if expected.value is RequirementValue.UNKNOWN:
                    expected_unknown += 1
                    correct_unknown += int(actual.value is RequirementValue.UNKNOWN)
                    overconfident += int(
                        actual.value in {RequirementValue.TRUE, RequirementValue.FALSE}
                    )
                predicted_unknown += int(actual.value is RequirementValue.UNKNOWN)
                if name in CRITICAL_REQUIREMENTS:
                    critical_tp += int(expected_true and actual_true)
                    critical_fn += int(expected_true and not actual_true)
                    critical_fp += int(not expected_true and actual_true)
                if expected.value != actual.value or expected.source != actual.source:
                    errors.append(
                        {
                            "case_id": case.id,
                            "user_request": case.input,
                            "requirement": name,
                            "expected": expected.value.value,
                            "actual": actual.value.value,
                            "expected_source": expected.source.value,
                            "actual_source": actual.source.value,
                            "error_class": _error_class(expected, actual),
                            "source_stage": variant,
                        }
                    )

            if variant == "baseline":
                evaluations = (
                    legacy_engine.evaluate(
                        requirements,
                        self.baseline.profiles,
                        self.baseline.availability,
                    )
                    if requirements is not None
                    else {}
                )
                binding = detect_explicit_agent_binding(case.input)
                requested = Agent(binding.requested_agent) if binding else None
                mutation_authorized = bool(
                    requirements is not None and requirements.mutation_required
                )
                actual_prohibitions = frozenset()
                predicted_ambiguity = bool(
                    requirements is not None and requirements.ambiguity_material
                )
            else:
                evaluations = (
                    grounded_engine.evaluate(
                        requirements,
                        self.baseline.profiles,
                        self.baseline.availability,
                    )
                    if requirements is not None
                    else {}
                )
                requested = requirements.requested_agent if requirements else None
                mutation_authorized = bool(
                    requirements and requirements.mutation_authorized_by_requirements
                )
                actual_prohibitions = frozenset(
                    requirements.prohibitions if requirements else ()
                )
                predicted_ambiguity = bool(
                    requirements is not None and requirements.ambiguity_material
                )
            ambiguity_correct += int(predicted_ambiguity == case.expected_ambiguity)
            if case.expected_prohibitions:
                prohibition_total += 1
                prohibition_correct += int(
                    case.expected_prohibitions.issubset(actual_prohibitions)
                )
            actual_eligible = frozenset(
                agent for agent, item in evaluations.items() if item.eligible
            )
            eligibility_tp += len(actual_eligible & case.expected_eligible_agents)
            eligibility_fp += len(actual_eligible - case.expected_eligible_agents)
            eligibility_fn += len(case.expected_eligible_agents - actual_eligible)
            if len(case.expected_eligible_agents) == 1:
                single_total += 1
                single_correct += int(actual_eligible == case.expected_eligible_agents)
            elif not case.expected_eligible_agents:
                no_total += 1
                no_correct += int(not actual_eligible)
            else:
                multi_total += 1
                multi_correct += int(actual_eligible == case.expected_eligible_agents)
            if case.expected_requested_agent is not None:
                requested_total += 1
                requested_correct += int(requested == case.expected_requested_agent)
            expected_mutation = case.expected[MUTATION_REQUIRED].value
            false_mutation_authorizations += int(
                mutation_authorized and expected_mutation is not RequirementValue.TRUE
            )
            proposal = (
                propose_agent_selection(evaluations, requested_agent=requested)
                if evaluations
                else None
            )
            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "predictions": {
                        name: {
                            "value": item.value.value,
                            "source": item.source.value,
                        }
                        for name, item in predictions.items()
                    },
                    "eligible_agents": sorted(item.value for item in actual_eligible),
                    "expected_eligible_agents": sorted(
                        item.value for item in case.expected_eligible_agents
                    ),
                    "requested_agent": requested.value if requested else None,
                    "proposal": proposal.as_dict() if proposal else None,
                    "analysis": analysis.__dict__,
                }
            )

        rendered_fields: dict[str, Any] = {}
        supported_precision = []
        supported_recall = []
        supported_f1 = []
        for name, stats in per_field.items():
            precision = _safe_div(stats["tp"], stats["tp"] + stats["fp"], 1.0)
            recall = _safe_div(stats["tp"], stats["tp"] + stats["fn"], 1.0)
            f1 = _safe_div(2 * precision * recall, precision + recall, 0.0)
            rendered_fields[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive": stats["fp"],
                "false_negative": stats["fn"],
                "unknown_rate": _safe_div(stats["unknown_predicted"], stats["total"]),
                "accuracy": _safe_div(stats["correct"], stats["total"]),
                "positive_support": stats["tp"] + stats["fn"],
                "definite_support": stats["definite_expected"],
            }
            if stats["tp"] + stats["fn"]:
                supported_precision.append(precision)
                supported_recall.append(recall)
                supported_f1.append(f1)

        total_predictions = len(cases) * len(REQUIREMENT_DIMENSIONS)
        total_tp = sum(item["tp"] for item in per_field.values())
        total_fp = sum(item["fp"] for item in per_field.values())
        total_fn = sum(item["fn"] for item in per_field.values())
        metrics = {
            "macro_precision": _safe_div(sum(supported_precision), len(supported_precision), 1.0),
            "macro_recall": _safe_div(sum(supported_recall), len(supported_recall), 1.0),
            "macro_f1": _safe_div(sum(supported_f1), len(supported_f1), 1.0),
            "micro_precision": _safe_div(total_tp, total_tp + total_fp, 1.0),
            "micro_recall": _safe_div(total_tp, total_tp + total_fn, 1.0),
            "accuracy": _safe_div(
                sum(item["correct"] for item in per_field.values()), total_predictions
            ),
            "critical_requirement_recall": _safe_div(
                critical_tp, critical_tp + critical_fn, 1.0
            ),
            "critical_false_positive_rate": _safe_div(
                critical_fp, critical_tp + critical_fp, 0.0
            ),
            "provenance_accuracy": _safe_div(provenance_correct, total_predictions),
            "explicit_evidence_preservation": _safe_div(
                explicit_correct, explicit_total, 1.0
            ),
            "deterministic_fact_preservation": _safe_div(
                deterministic_correct, deterministic_total, 1.0
            ),
            "semantic_inference_accuracy": _safe_div(
                semantic_correct, semantic_total, 1.0
            ),
            "unknown_precision": _safe_div(correct_unknown, predicted_unknown, 1.0),
            "unknown_recall": _safe_div(correct_unknown, expected_unknown, 1.0),
            "overconfidence_rate": _safe_div(overconfident, expected_unknown),
            "explicit_constraint_retention": _safe_div(
                constraint_correct, constraint_total, 1.0
            ),
            "explicit_prohibition_retention": _safe_div(
                prohibition_correct, prohibition_total, 1.0
            ),
            "ambiguity_accuracy": _safe_div(ambiguity_correct, len(cases), 1.0),
            "false_mutation_authorization": false_mutation_authorizations,
            "eligibility_precision": _safe_div(
                eligibility_tp, eligibility_tp + eligibility_fp, 1.0
            ),
            "eligibility_recall": _safe_div(
                eligibility_tp, eligibility_tp + eligibility_fn, 1.0
            ),
            "false_eligible_agent": eligibility_fp,
            "missed_eligible_agent": eligibility_fn,
            "single_candidate_correct": _safe_div(single_correct, single_total, 1.0),
            "no_candidate_correct": _safe_div(no_correct, no_total, 1.0),
            "multi_candidate_correct": _safe_div(multi_correct, multi_total, 1.0),
            "explicit_agent_preservation": _safe_div(
                requested_correct, requested_total, 1.0
            ),
            "json_validity": _safe_div(len(cases) - invalid, len(cases), 1.0),
            "retry_rate": _safe_div(retries, len(cases)),
            "extra_inference_count": 0,
            "total_inference_attempts": inference_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p90_ms": _percentile(latencies, 0.90),
            "latency_p95_ms": _percentile(latencies, 0.95),
        }
        return {
            "variant": variant,
            "cases": len(cases),
            "dry_run": True,
            "execution_count": 0,
            "metrics": metrics,
            "per_field": rendered_fields,
            "error_taxonomy": errors,
            "rows": rows,
        }


def run_live(
    cases: Sequence[GroundingCase],
    *,
    endpoint: str,
    project_snapshot: Mapping[str, Any] | None,
    variant: str,
) -> list[Any]:
    client = LlamaClient(endpoint, timeout=600)
    analyzer: Any = (
        TaskRequirementAnalyzer(client)
        if variant == "baseline"
        else GroundedTaskRequirementAnalyzer(client)
    )
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"{variant} {index}/{len(cases)} {case.id}", flush=True)
        results.append(analyzer.analyze(case.input, project_snapshot=project_snapshot))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task requirement grounding A/B evaluator")
    parser.add_argument(
        "--corpus",
        default="tests/data/task_requirement_grounding_diagnostic.jsonl",
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--project")
    parser.add_argument("--variant", choices=("baseline", "grounded", "both"), default="both")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    cases = load_grounding_cases(args.corpus)
    snapshot = None
    if args.project:
        project = ProjectSnapshotBuilder(args.project).build()
        selected_entries = sorted(
            project.repo_map,
            key=lambda item: (
                not item.path.startswith("tern/"),
                not item.path.startswith("tests/"),
                item.path.casefold(),
            ),
        )[:40]
        snapshot = {
            "project_path": project.project_path,
            "languages": list(project.languages),
            "git_branch": project.git_branch,
            "modified_files": list(project.modified_files[:20]),
            "tests_present": bool(project.tests),
            "repo_map": [
                {
                    "path": item.path,
                    "file_type": item.file_type,
                    "module": item.module,
                }
                for item in selected_entries
            ],
        }
    evaluator = GroundingEvaluator()
    variants = ("baseline", "grounded") if args.variant == "both" else (args.variant,)
    report = {
        variant: evaluator.evaluate_variant(
            cases,
            run_live(
                cases,
                endpoint=args.endpoint,
                project_snapshot=snapshot,
                variant=variant,
            ),
            variant=variant,
        )
        for variant in variants
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
