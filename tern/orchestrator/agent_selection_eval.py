"""Dry-run evaluator for automatic agent selection and selection provenance.

The evaluator never executes an agent, never creates a job or session and never
touches the filesystem outside the report file it is asked to write.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .autonomy_foundation import (
    Agent,
    AgentRuntimeAvailability,
    RiskLevel,
    propose_agent_selection,
)
from .agent_selection import (
    AgentSelectionEngine,
    AgentSelectionProposal,
    OperationalFacts,
    SelectionFactorType,
    SelectionPolicy,
    SelectionProjectContext,
    SelectionSource,
    SemanticAgentSelector,
)
from .autonomy_eval import diagnostic_baseline
from .client import LlamaClient
from .local_model_runtime import OpenAICompatibleLocalRuntime, RuntimeDescriptor
from .task_requirement_grounding import (
    MUTATION_REQUIRED,
    READ_ONLY_REQUIRED,
    REQUIREMENT_DIMENSIONS,
    GroundedEligibilityEngine,
    GroundedTaskRequirements,
    RequirementEvidenceSource,
    RequirementValue,
    grounded_requirement,
)


ERROR_CODES = (
    "EXPLICIT_AGENT_OVERRIDDEN",
    "INELIGIBLE_AGENT_SELECTED",
    "UNAVAILABLE_AGENT_SELECTED",
    "WRONG_MULTI_AGENT_SELECTION",
    "UNJUSTIFIED_SELECTION",
    "FAILED_TO_SELECT_SINGLE_CANDIDATE",
    "FALSE_UNRESOLVED",
    "OVERCONFIDENT_SELECTION",
    "AVAILABILITY_CORRUPTED_ELIGIBILITY",
    "BAD_SELECTION_FACTOR",
    "MODEL_PARSE_FAILURE",
)


def grounded_requirements_from_labels(labels: Mapping[str, Any]) -> GroundedTaskRequirements:
    """Build grounded requirements from corpus ground truth, not from keywords."""

    capabilities = frozenset(str(item) for item in labels.get("capabilities") or ())
    unknown = frozenset(str(item) for item in labels.get("unknown_dimensions") or ())
    flags = {
        MUTATION_REQUIRED: bool(labels.get("mutation_required")),
        READ_ONLY_REQUIRED: bool(labels.get("read_only_required")),
    }
    requirements = {}
    for name in REQUIREMENT_DIMENSIONS:
        if name in unknown:
            value = RequirementValue.UNKNOWN
            source = RequirementEvidenceSource.INSUFFICIENT_EVIDENCE
            ref = "corpus:unknown_dimension"
        else:
            positive = flags[name] if name in flags else name in capabilities
            value = RequirementValue.TRUE if positive else RequirementValue.FALSE
            source = RequirementEvidenceSource.TASK_IMPLICATION
            ref = "corpus:labelled_requirement"
        requirements[name] = grounded_requirement(name, value, source, ref)
    return GroundedTaskRequirements(
        requirements=requirements,
        target_scope=str(labels.get("target_scope") or "unknown"),
        risk_level=RiskLevel(str(labels.get("risk_level") or "low")),
        ambiguity_material=bool(labels.get("ambiguity_material")),
        expected_files=tuple(labels.get("expected_files") or ()),
        forbidden_files=tuple(labels.get("forbidden_files") or ()),
        tests_requested=tuple(labels.get("tests_requested") or ()),
        prohibitions=tuple(labels.get("prohibitions") or ()),
    )


@dataclass(frozen=True)
class SelectionCase:
    id: str
    category: str
    input: str
    requirements: GroundedTaskRequirements
    requested_agent: Agent | None
    availability: Mapping[Agent, bool]
    operational: OperationalFacts
    policy: SelectionPolicy
    expected_eligible_agents: frozenset[Agent]
    expected_available_agents: frozenset[Agent]
    acceptable_selected_agents: frozenset[Agent]
    preferred_agent: Agent | None
    expected_selection_sources: tuple[SelectionSource, ...]
    required_factors: tuple[SelectionFactorType, ...]
    forbidden_agents: frozenset[Agent]
    unresolved_acceptable: bool = False
    contrast_of: str | None = None

    @property
    def expects_unresolved(self) -> bool:
        return not self.acceptable_selected_agents

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionCase":
        expected = value["expected"]
        operational = value.get("operational") or {}
        availability = {
            agent: bool((value.get("availability") or {}).get(agent.value, True))
            for agent in Agent
        }
        return cls(
            str(value["id"]),
            str(value["category"]),
            str(value["input"]),
            grounded_requirements_from_labels(value["requirements"]),
            Agent(value["requested_agent"]) if value.get("requested_agent") else None,
            availability,
            OperationalFacts(
                bool(operational.get("reusable_codex_session")),
                bool(operational.get("codex_project_affinity")),
                bool(operational.get("deepseek_project_session")),
                operational.get("project_id"),
            ),
            SelectionPolicy(
                deepseek_auto_escalation=bool(
                    (value.get("policy") or {}).get("deepseek_auto_escalation", False)
                )
            ),
            frozenset(Agent(item) for item in expected["eligible_agents"]),
            frozenset(Agent(item) for item in expected["available_agents"]),
            frozenset(Agent(item) for item in expected.get("acceptable_selected_agents") or ()),
            Agent(expected["preferred_agent"]) if expected.get("preferred_agent") else None,
            tuple(SelectionSource(item) for item in expected["selection_sources"]),
            tuple(SelectionFactorType(item) for item in expected.get("required_factors") or ()),
            frozenset(Agent(item) for item in expected.get("forbidden_agents") or ()),
            bool(value.get("unresolved_acceptable")),
            value.get("contrast_of"),
        )


def load_selection_cases(path: str | Path) -> list[SelectionCase]:
    return [
        SelectionCase.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def availability_map(
    baseline_availability: Mapping[Agent, AgentRuntimeAvailability],
    overrides: Mapping[Agent, bool],
) -> dict[Agent, AgentRuntimeAvailability]:
    result: dict[Agent, AgentRuntimeAvailability] = {}
    for agent, state in baseline_availability.items():
        available = overrides.get(agent, state.available)
        result[agent] = AgentRuntimeAvailability(
            agent,
            available,
            state.enabled,
            state.configured,
            state.reason_code if available else f"{agent.value.upper()}_UNAVAILABLE",
        )
    return result


class SelectionEvaluator:
    def __init__(
        self,
        *,
        semantic_selector: SemanticAgentSelector | None = None,
        baseline: Any | None = None,
    ):
        self.baseline = baseline or diagnostic_baseline(deepseek_available=True)
        self.eligibility = GroundedEligibilityEngine()
        self.semantic_selector = semantic_selector

    def _evaluate_case(self, case: SelectionCase) -> dict[str, Any]:
        availability = availability_map(self.baseline.availability, case.availability)
        evaluations = self.eligibility.evaluate(
            case.requirements,
            self.baseline.profiles,
            availability,
        )
        # eligibility must not depend on availability
        all_available = availability_map(
            self.baseline.availability, {agent: True for agent in Agent}
        )
        control = self.eligibility.evaluate(
            case.requirements,
            self.baseline.profiles,
            all_available,
        )
        eligibility_stable = all(
            control[agent].eligible == evaluations[agent].eligible for agent in Agent
        )
        engine = AgentSelectionEngine(
            policy=case.policy,
            semantic_selector=self.semantic_selector,
        )
        proposal = engine.propose(
            case.requirements,
            evaluations,
            capability_profiles=self.baseline.profiles,
            availability=availability,
            operational=case.operational,
            project_context=SelectionProjectContext(
                project_id=case.operational.project_id,
                languages=("python",),
                test_roots=("tests",),
            ),
            requested_agent=case.requested_agent,
        )
        legacy = propose_agent_selection(
            {agent: item.base for agent, item in evaluations.items()},
            requested_agent=case.requested_agent,
        )
        return {
            "proposal": proposal,
            "legacy": legacy,
            "evaluations": evaluations,
            "availability": availability,
            "eligibility_stable": eligibility_stable,
        }

    def run(
        self,
        cases: Iterable[SelectionCase],
        *,
        replay: int = 1,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        errors: dict[str, int] = {code: 0 for code in ERROR_CODES}
        counters = {
            "explicit_total": 0,
            "explicit_ok": 0,
            "single_total": 0,
            "single_ok": 0,
            "no_eligible_total": 0,
            "no_eligible_ok": 0,
            "availability_total": 0,
            "availability_ok": 0,
            "multi_total": 0,
            "multi_ok": 0,
            "proposal_total": 0,
            "acceptable_total": 0,
            "ineligible_selected": 0,
            "unavailable_selected": 0,
            "unjustified_selected": 0,
            "provenance_ok": 0,
            "factor_total": 0,
            "factor_ok": 0,
            "unresolved_predicted": 0,
            "unresolved_expected": 0,
            "unresolved_true_positive": 0,
            "model_calls": 0,
            "avoided_explicit": 0,
            "avoided_single": 0,
            "avoided_no_eligible": 0,
            "avoided_only_available": 0,
            "avoided_deterministic": 0,
            "avoided_unresolved": 0,
            "legacy_proposals": 0,
            "legacy_acceptable": 0,
            "jobs_created": 0,
            "delegations": 0,
            "filesystem_mutations": 0,
            "session_resolved": 0,
        }
        deterministic_latencies: list[float] = []
        semantic_latencies: list[float] = []
        consistency_rows: list[dict[str, Any]] = []

        for case in cases:
            outcome = self._evaluate_case(case)
            proposal: AgentSelectionProposal = outcome["proposal"]
            legacy = outcome["legacy"]
            eligible = frozenset(proposal.eligible_agents)
            available_eligible = frozenset(proposal.available_eligible_agents)
            proposed = proposal.proposed_agent
            source = proposal.selection_source
            case_errors: list[str] = []

            if not outcome["eligibility_stable"]:
                case_errors.append("AVAILABILITY_CORRUPTED_ELIGIBILITY")
            provenance_ok = source in case.expected_selection_sources
            counters["provenance_ok"] += int(provenance_ok)
            eligibility_exact = eligible == case.expected_eligible_agents
            availability_exact = available_eligible == frozenset(
                agent for agent in case.expected_available_agents if agent in case.expected_eligible_agents
            )

            counters["model_calls"] += proposal.model_calls
            if proposal.model_calls:
                semantic_latencies.append(proposal.model_latency_ms)
            deterministic_latencies.append(proposal.deterministic_latency_ms)
            if source is SelectionSource.EXPLICIT_USER:
                counters["avoided_explicit"] += 1
            elif source is SelectionSource.SINGLE_ELIGIBLE_AGENT:
                counters["avoided_single"] += 1
            elif source is SelectionSource.NO_ELIGIBLE_AGENT:
                counters["avoided_no_eligible"] += 1
            elif source is SelectionSource.ONLY_AVAILABLE_ELIGIBLE_AGENT:
                counters["avoided_only_available"] += 1
            elif source is SelectionSource.DETERMINISTIC_SELECTION:
                counters["avoided_deterministic"] += 1
            elif source is SelectionSource.UNRESOLVED and proposal.model_calls == 0:
                counters["avoided_unresolved"] += 1

            if case.requested_agent is not None:
                counters["explicit_total"] += 1
                preserved = (
                    proposed == case.requested_agent
                    and source is SelectionSource.EXPLICIT_USER
                    and proposal.model_calls == 0
                )
                counters["explicit_ok"] += int(preserved)
                if not preserved:
                    case_errors.append("EXPLICIT_AGENT_OVERRIDDEN")
            elif len(case.expected_eligible_agents) == 1:
                counters["single_total"] += 1
                ok = (
                    proposed is not None
                    and proposed in case.expected_eligible_agents
                    and source is SelectionSource.SINGLE_ELIGIBLE_AGENT
                    and proposal.model_calls == 0
                )
                counters["single_ok"] += int(ok)
                if not ok:
                    case_errors.append("FAILED_TO_SELECT_SINGLE_CANDIDATE")
            elif not case.expected_eligible_agents:
                counters["no_eligible_total"] += 1
                ok = proposed is None and source is SelectionSource.NO_ELIGIBLE_AGENT
                counters["no_eligible_ok"] += int(ok)

            if case.category.startswith("availability"):
                counters["availability_total"] += 1
                counters["availability_ok"] += int(
                    availability_exact and eligibility_exact and provenance_ok
                )

            if source is SelectionSource.SEMANTIC_MULTI_AGENT:
                counters["multi_total"] += 1
                target = (
                    {case.preferred_agent}
                    if case.preferred_agent is not None
                    else case.acceptable_selected_agents
                )
                ok = proposed in target
                counters["multi_ok"] += int(ok)
                if not ok:
                    case_errors.append("WRONG_MULTI_AGENT_SELECTION")

            if proposed is not None and source is not SelectionSource.EXPLICIT_USER:
                counters["proposal_total"] += 1
                acceptable = proposed in case.acceptable_selected_agents
                counters["acceptable_total"] += int(acceptable)
                if proposed not in eligible:
                    counters["ineligible_selected"] += 1
                    case_errors.append("INELIGIBLE_AGENT_SELECTED")
                if proposed not in available_eligible:
                    counters["unavailable_selected"] += 1
                    case_errors.append("UNAVAILABLE_AGENT_SELECTED")
                if proposed in case.forbidden_agents:
                    case_errors.append("INELIGIBLE_AGENT_SELECTED")
                justified = bool(proposal.factors) or source in {
                    SelectionSource.SINGLE_ELIGIBLE_AGENT,
                    SelectionSource.ONLY_AVAILABLE_ELIGIBLE_AGENT,
                    SelectionSource.NO_ELIGIBLE_AGENT,
                }
                if not justified:
                    counters["unjustified_selected"] += 1
                    case_errors.append("UNJUSTIFIED_SELECTION")
                if (
                    proposal.confidence.value == "SUPPORTED"
                    and case.preferred_agent is None
                    and len(case.acceptable_selected_agents) > 1
                    and len(proposal.factors)
                    <= max(
                        (
                            len(proposal.profiles[agent].support_factors)
                            for agent in proposal.candidate_agents
                            if agent is not proposed
                        ),
                        default=0,
                    )
                ):
                    case_errors.append("OVERCONFIDENT_SELECTION")

            if case.required_factors and proposed is not None:
                counters["factor_total"] += 1
                present = {item.type for item in proposal.factors} | {
                    item.type for item in proposal.task_factors
                }
                ok = set(case.required_factors).issubset(present)
                counters["factor_ok"] += int(ok)
                if not ok:
                    case_errors.append("BAD_SELECTION_FACTOR")

            unresolved_predicted = source in {
                SelectionSource.UNRESOLVED,
                SelectionSource.NO_AVAILABLE_ELIGIBLE_AGENT,
                SelectionSource.INVALID_SELECTION,
            }
            unresolved_expected = (
                case.expects_unresolved
                and case.requested_agent is None
                and SelectionSource.NO_ELIGIBLE_AGENT not in case.expected_selection_sources
            )
            unresolved_tolerated = unresolved_expected or case.unresolved_acceptable
            counters["unresolved_predicted"] += int(unresolved_predicted)
            counters["unresolved_expected"] += int(unresolved_expected)
            counters["unresolved_true_positive"] += int(unresolved_predicted and unresolved_tolerated)
            if unresolved_predicted and not unresolved_tolerated:
                case_errors.append("FALSE_UNRESOLVED")
            for item in proposal.errors:
                if item.startswith("BAD_SELECTION_FACTOR"):
                    case_errors.append("BAD_SELECTION_FACTOR")
                elif item in ERROR_CODES:
                    case_errors.append(item)
                else:
                    case_errors.append("MODEL_PARSE_FAILURE")

            counters["jobs_created"] += proposal.jobs_created
            counters["delegations"] += proposal.delegations
            counters["filesystem_mutations"] += proposal.filesystem_mutations
            counters["session_resolved"] += int(proposal.session_resolved)

            if legacy.proposed_agent is not None or legacy.selected_agent is not None:
                counters["legacy_proposals"] += 1
                chosen = legacy.selected_agent or legacy.proposed_agent
                counters["legacy_acceptable"] += int(
                    chosen == case.requested_agent
                    if case.requested_agent is not None
                    else chosen in case.acceptable_selected_agents
                )

            for code in case_errors:
                if code in errors:
                    errors[code] += 1

            replays: list[str | None] = []
            if replay > 1 and proposal.model_calls:
                for _ in range(replay - 1):
                    repeat = self._evaluate_case(case)["proposal"]
                    replays.append(
                        repeat.proposed_agent.value if repeat.proposed_agent else None
                    )
                    counters["model_calls"] += repeat.model_calls
                    semantic_latencies.append(repeat.model_latency_ms)
                first = proposed.value if proposed else None
                consistency_rows.append(
                    {
                        "id": case.id,
                        "decisions": [first, *replays],
                        "stable": all(item == first for item in replays),
                    }
                )

            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "input": case.input,
                    "contrast_of": case.contrast_of,
                    "eligibility_exact": eligibility_exact,
                    "availability_exact": availability_exact,
                    "provenance_ok": provenance_ok,
                    "expected_selection_sources": [item.value for item in case.expected_selection_sources],
                    "acceptable_selected_agents": sorted(
                        item.value for item in case.acceptable_selected_agents
                    ),
                    "preferred_agent": case.preferred_agent.value if case.preferred_agent else None,
                    "errors": sorted(set(case_errors)),
                    "replay_decisions": replays,
                    "proposal": proposal.as_dict(),
                    "legacy_baseline": legacy.as_dict(),
                }
            )

        def ratio(ok: str, total: str) -> float:
            return counters[ok] / counters[total] if counters[total] else 1.0

        def percentile(values: Sequence[float], percent: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[round((len(ordered) - 1) * percent)]

        consistent = [item for item in consistency_rows if item["stable"]]
        metrics = {
            "explicit_agent_preservation": ratio("explicit_ok", "explicit_total"),
            "single_eligible_selection_accuracy": ratio("single_ok", "single_total"),
            "no_eligible_accuracy": ratio("no_eligible_ok", "no_eligible_total"),
            "availability_handling_accuracy": ratio("availability_ok", "availability_total"),
            "multi_agent_selection_accuracy": ratio("multi_ok", "multi_total"),
            "acceptable_selection_rate": ratio("acceptable_total", "proposal_total"),
            "ineligible_agent_selection_rate": (
                counters["ineligible_selected"] / counters["proposal_total"]
                if counters["proposal_total"]
                else 0.0
            ),
            "unavailable_agent_selection_rate": (
                counters["unavailable_selected"] / counters["proposal_total"]
                if counters["proposal_total"]
                else 0.0
            ),
            "unjustified_selection_rate": (
                counters["unjustified_selected"] / counters["proposal_total"]
                if counters["proposal_total"]
                else 0.0
            ),
            "unresolved_precision": (
                counters["unresolved_true_positive"] / counters["unresolved_predicted"]
                if counters["unresolved_predicted"]
                else 1.0
            ),
            "unresolved_recall": (
                counters["unresolved_true_positive"] / counters["unresolved_expected"]
                if counters["unresolved_expected"]
                else 1.0
            ),
            "selection_provenance_accuracy": counters["provenance_ok"] / len(rows) if rows else 1.0,
            "selection_factor_accuracy": ratio("factor_ok", "factor_total"),
            "eligibility_exactness": (
                sum(item["eligibility_exact"] for item in rows) / len(rows) if rows else 1.0
            ),
            "availability_separation_accuracy": (
                0.0 if errors["AVAILABILITY_CORRUPTED_ELIGIBILITY"] else 1.0
            ),
            "extra_model_calls": counters["model_calls"],
            "model_calls_per_case": counters["model_calls"] / len(rows) if rows else 0.0,
            "selection_consistency": (
                len(consistent) / len(consistency_rows) if consistency_rows else None
            ),
            "deterministic_latency_p50_ms": percentile(deterministic_latencies, 0.50),
            "deterministic_latency_p90_ms": percentile(deterministic_latencies, 0.90),
            "deterministic_latency_p95_ms": percentile(deterministic_latencies, 0.95),
            "semantic_latency_mean_ms": (
                sum(semantic_latencies) / len(semantic_latencies) if semantic_latencies else None
            ),
            "semantic_latency_p50_ms": percentile(semantic_latencies, 0.50),
            "semantic_latency_p90_ms": percentile(semantic_latencies, 0.90),
            "semantic_latency_p95_ms": percentile(semantic_latencies, 0.95),
        }
        safety = {
            "execution_authorized": False,
            "automatic_tool_calls": 0,
            "automatic_delegations": counters["delegations"],
            "filesystem_mutations": counters["filesystem_mutations"],
            "codex_jobs_created": counters["jobs_created"],
            "deepseek_jobs_created": 0,
            "sessions_resolved": counters["session_resolved"],
        }
        calls_avoided = {
            "explicit_agent": counters["avoided_explicit"],
            "single_eligible": counters["avoided_single"],
            "no_eligible": counters["avoided_no_eligible"],
            "only_available_eligible": counters["avoided_only_available"],
            "deterministic_selection": counters["avoided_deterministic"],
            "deterministic_unresolved": counters["avoided_unresolved"],
        }
        calls_avoided["total"] = sum(calls_avoided.values())
        baseline = {
            "name": "NO_SELECTION",
            "proposals": counters["legacy_proposals"],
            "acceptable": counters["legacy_acceptable"],
            "acceptable_rate": (
                counters["legacy_acceptable"] / counters["legacy_proposals"]
                if counters["legacy_proposals"]
                else 0.0
            ),
            "multi_agent_proposals": 0,
        }
        return {
            "dry_run": True,
            "cases": len(rows),
            "metrics": metrics,
            "errors": errors,
            "safety": safety,
            "calls_avoided": calls_avoided,
            "baseline_a": baseline,
            "consistency": consistency_rows,
            "rows": rows,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent selection evaluator (dry-run only)")
    parser.add_argument("--corpus", default="tests/data/agent_selection_diagnostic.jsonl")
    parser.add_argument("--endpoint", help="llama-server endpoint for semantic selection")
    parser.add_argument("--provider", default="llama.cpp")
    parser.add_argument("--model", default="qwen3.5-4b-q4_k_m")
    parser.add_argument("--runtime", default="stock")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--replay", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    selector = None
    if args.endpoint:
        selector = SemanticAgentSelector(
            OpenAICompatibleLocalRuntime(
                LlamaClient(args.endpoint, timeout=args.timeout),
                RuntimeDescriptor(args.provider, args.model, args.runtime),
            )
        )
    started = time.perf_counter()
    report = SelectionEvaluator(semantic_selector=selector).run(
        load_selection_cases(args.corpus),
        replay=args.replay,
    )
    report["wall_clock_seconds"] = round(time.perf_counter() - started, 3)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
