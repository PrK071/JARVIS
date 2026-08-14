from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .autonomy_foundation import (
    Agent,
    AgentRuntimeAvailability,
    Capability,
    CapabilityProfileBuilder,
    EligibilityEngine,
    TaskRequirementAnalyzer,
    TaskRequirements,
    propose_agent_selection,
)
from .client import LlamaClient
from .explicit_agent_binding import detect_explicit_agent_binding
from .project_intelligence import ProjectSnapshotBuilder


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    input: str
    expected_requirements: TaskRequirements
    expected_eligible_agents: frozenset[Agent]
    expected_proposed_agent: Agent | None
    expected_requested_agent: Agent | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationCase":
        expected = value["expected"]
        return cls(
            str(value["id"]),
            str(value["input"]),
            TaskRequirements.from_dict(expected["requirements"]),
            frozenset(Agent(item) for item in expected["eligible_agents"]),
            Agent(expected["proposed_agent"]) if expected.get("proposed_agent") else None,
            Agent(expected["requested_agent"]) if expected.get("requested_agent") else None,
        )


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvaluationCase.from_dict(json.loads(line)))
    return cases


class _OfflineRegistry:
    def __init__(self, *, deepseek_available: bool = True):
        self._names = (
            "filesystem_list",
            "filesystem_read_text",
            "filesystem_write_text",
            "filesystem_delete",
            "find_project_files",
            "web_search",
            "web_open",
            "web_extract",
            "delegate_to_codex",
            "get_codex_job_status",
            "delegate_to_deepseek",
            "review_deepseek_session",
        )
        self.codex = SimpleNamespace(sessions=SimpleNamespace())
        self.deepseek = SimpleNamespace(
            client=SimpleNamespace(enabled=True, configured=deepseek_available)
        )

    def names(self) -> tuple[str, ...]:
        return self._names


def diagnostic_baseline(*, deepseek_available: bool = True):
    return CapabilityProfileBuilder.from_registry(
        _OfflineRegistry(deepseek_available=deepseek_available),
        local_model_available=True,
        codex_available=True,
    )


def capability_profile_metrics(baseline: Any) -> dict[str, float]:
    declared = 0
    evidenced = 0
    for profile in baseline.profiles.values():
        for capability in profile.capabilities:
            declared += 1
            evidenced += int(
                any(item.capability == capability for item in profile.evidence)
            )
    available = diagnostic_baseline(deepseek_available=True)
    unavailable = diagnostic_baseline(deepseek_available=False)
    unchanged = sum(
        available.profiles[agent].capabilities
        == unavailable.profiles[agent].capabilities
        for agent in Agent
    )
    return {
        "capability_profile_accuracy": evidenced / declared if declared else 1.0,
        "availability_separation_accuracy": unchanged / len(Agent),
    }


class AutonomyEvaluator:
    def __init__(self, baseline: Any):
        self.baseline = baseline
        self.eligibility = EligibilityEngine()

    def evaluate(
        self,
        cases: Iterable[EvaluationCase],
        *,
        analyzer: TaskRequirementAnalyzer | None = None,
        project_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows = []
        requirement_exact = 0
        requirement_fields_correct = 0
        constraint_fields_correct = 0
        proposed_correct = 0
        explicit_correct = 0
        explicit_total = 0
        single_total = 0
        single_correct = 0
        tp = fp = fn = 0
        analysis_latencies: list[float] = []
        prompt_tokens = generated_tokens = 0
        valid = first_pass_valid = retry_valid = 0
        retries = 0
        no_execution = 0
        for case in cases:
            analysis = (
                analyzer.analyze(case.input, project_snapshot=project_snapshot)
                if analyzer is not None
                else None
            )
            requirements = case.expected_requirements if analysis is None else analysis.requirements
            if analysis is not None:
                analysis_latencies.append(analysis.latency_ms)
                valid += int(analysis.valid)
                first_pass_valid += int(analysis.first_pass_valid)
                retry_valid += int(analysis.valid and not analysis.first_pass_valid)
                retries += int(analysis.attempts > 1)
                prompt_tokens += analysis.prompt_tokens or 0
                generated_tokens += analysis.generated_tokens or 0
            binding = detect_explicit_agent_binding(case.input)
            requested = Agent(binding.requested_agent) if binding else None
            explicit_total += int(case.expected_requested_agent is not None)
            if requirements is None:
                fn += len(case.expected_eligible_agents)
                if len(case.expected_eligible_agents) == 1 and case.expected_requested_agent is None:
                    single_total += 1
                no_execution += 1
                rows.append(
                    {
                        "id": case.id,
                        "requirements": None,
                        "requirements_correct": False,
                        "proposal": None,
                        "requested_agent_detected": requested.value if requested else None,
                        "expected_eligible_agents": sorted(
                            item.value for item in case.expected_eligible_agents
                        ),
                        "analysis": analysis.__dict__,
                    }
                )
                continue
            exact_requirements = requirements.as_dict() == case.expected_requirements.as_dict()
            if analysis is None or exact_requirements:
                requirement_exact += 1
            expected_requirements = case.expected_requirements
            requirement_fields_correct += sum(
                (
                    requirements.capabilities == expected_requirements.capabilities,
                    requirements.mutation_required == expected_requirements.mutation_required,
                    requirements.read_only_required == expected_requirements.read_only_required,
                    requirements.risk_level == expected_requirements.risk_level,
                    requirements.ambiguity_material == expected_requirements.ambiguity_material,
                )
            )
            constraint_fields_correct += sum(
                (
                    requirements.mutation_required == expected_requirements.mutation_required,
                    requirements.read_only_required == expected_requirements.read_only_required,
                    requirements.risk_level == expected_requirements.risk_level,
                    requirements.ambiguity_material == expected_requirements.ambiguity_material,
                    requirements.forbidden_files == expected_requirements.forbidden_files,
                )
            )
            evaluations = self.eligibility.evaluate(
                requirements,
                self.baseline.profiles,
                self.baseline.availability,
            )
            proposal = propose_agent_selection(evaluations, requested_agent=requested)
            actual_eligible = frozenset(proposal.eligible_agents)
            tp += len(actual_eligible & case.expected_eligible_agents)
            fp += len(actual_eligible - case.expected_eligible_agents)
            fn += len(case.expected_eligible_agents - actual_eligible)
            proposed_correct += int(proposal.proposed_agent == case.expected_proposed_agent)
            if case.expected_requested_agent is not None:
                explicit_correct += int(
                    requested == case.expected_requested_agent
                    and proposal.selected_agent == case.expected_requested_agent
                )
            if len(case.expected_eligible_agents) == 1 and case.expected_requested_agent is None:
                single_total += 1
                single_correct += int(proposal.proposed_agent in case.expected_eligible_agents)
            no_execution += int(proposal.dry_run and not proposal.execution_authorized)
            rows.append(
                {
                    "id": case.id,
                    "requirements": requirements.as_dict(),
                    "requirements_correct": exact_requirements,
                    "proposal": proposal.as_dict(),
                    "expected_eligible_agents": sorted(item.value for item in case.expected_eligible_agents),
                    "analysis": analysis.__dict__ if analysis is not None else None,
                }
            )
        count = len(rows)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        sorted_latency = sorted(analysis_latencies)

        def percentile(percent: float) -> float | None:
            if not sorted_latency:
                return None
            index = round((len(sorted_latency) - 1) * percent)
            return sorted_latency[index]

        return {
            "dry_run": True,
            "execution_count": 0,
            "cases": count,
            "metrics": {
                **capability_profile_metrics(self.baseline),
                "task_requirement_accuracy": requirement_fields_correct / (count * 5) if count else 1.0,
                "task_requirement_exact_match": requirement_exact / count if count else 1.0,
                "constraint_accuracy": constraint_fields_correct / (count * 5) if count else 1.0,
                "eligibility_precision": precision,
                "eligibility_recall": recall,
                "false_eligible_agent_rate": fp / (tp + fp) if tp + fp else 0.0,
                "missed_eligible_agent_rate": fn / (tp + fn) if tp + fn else 0.0,
                "single_candidate_resolution_accuracy": single_correct / single_total if single_total else 1.0,
                "explicit_agent_override_accuracy": explicit_correct / explicit_total if explicit_total else 1.0,
                "no_execution_from_dry_run_accuracy": no_execution / count if count else 1.0,
                "json_validity": valid / count if analyzer and count else None,
                "first_pass_semantic_validity": first_pass_valid / count if analyzer and count else None,
                "valid_after_retry": valid / count if analyzer and count else None,
                "retry_recovery_rate": retry_valid / retries if analyzer and retries else 0.0 if analyzer else None,
                "retry_rate": retries / count if analyzer and count else None,
                "fallback_rate": 0.0,
                "latency_p50_ms": percentile(0.50),
                "latency_p90_ms": percentile(0.90),
                "latency_p95_ms": percentile(0.95),
                "prompt_tokens": prompt_tokens if analyzer else None,
                "generated_tokens": generated_tokens if analyzer else None,
                "proposal_accuracy": proposed_correct / count if count else 1.0,
            },
            "rows": rows,
        }


def project_understanding_metrics(expected: Iterable[str], selected: Iterable[str]) -> dict[str, float]:
    expected_set = frozenset(expected)
    selected_set = frozenset(selected)
    relevant = len(expected_set & selected_set)
    irrelevant = len(selected_set - expected_set)
    return {
        "relevant_file_recall": relevant / len(expected_set) if expected_set else 1.0,
        "irrelevant_file_selection_rate": irrelevant / len(selected_set) if selected_set else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomy foundation evaluator (dry-run only)")
    parser.add_argument("--corpus", default="tests/data/autonomy_foundation_diagnostic.jsonl")
    parser.add_argument("--endpoint", help="llama-server endpoint for live requirement extraction")
    parser.add_argument("--project", help="optional project snapshot root")
    parser.add_argument("--output")
    parser.add_argument("--deepseek-unavailable", action="store_true")
    args = parser.parse_args(argv)
    analyzer = TaskRequirementAnalyzer(LlamaClient(args.endpoint)) if args.endpoint else None
    snapshot = None
    if args.project:
        snapshot = ProjectSnapshotBuilder(args.project).build().compact()
    report = AutonomyEvaluator(
        diagnostic_baseline(deepseek_available=not args.deepseek_unavailable)
    ).evaluate(load_cases(args.corpus), analyzer=analyzer, project_snapshot=snapshot)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
