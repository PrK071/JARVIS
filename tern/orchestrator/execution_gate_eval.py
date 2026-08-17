"""Shadow evaluator for the execution gate bridge.

It runs the new pipeline against a corpus of live-like requests, evaluates the
ExecutionGate in SHADOW authority and compares each decision with a recorded
legacy decision. It executes nothing: no tool, no delegation, no session, no job
and no filesystem mutation. Divergence is measured, never repaired.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from .agent_selection import SelectionPolicy, SelectionSource
from .autonomy_eval import diagnostic_baseline
from .autonomy_foundation import Agent
from .decision_policy import Intent, SideEffect
from .execution_gate import ExecutionBlockReason
from .execution_gate_shadow import (
    DivergenceCode,
    ShadowExecutionObserver,
    ShadowObservation,
    legacy_facts_from_decision,
)
from .intent_semantics import IntentFrameBuilder
from .projects import normalize_technical_transcript


DIVERGENCE_CODES = tuple(item.value for item in DivergenceCode)
BLOCK_REASONS = tuple(item.value for item in ExecutionBlockReason)


@dataclass(frozen=True)
class ShadowCase:
    id: str
    category: str
    input: str
    execution_requested: bool
    availability: Mapping[Agent, bool]
    policy: SelectionPolicy
    legacy: Mapping[str, Any]
    expected: Mapping[str, Any]
    constraint_violation: str | None
    confirmation_required: bool
    path_policy_satisfied: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowCase":
        availability = {
            agent: bool((value.get("availability") or {}).get(agent.value, True))
            for agent in Agent
        }
        return cls(
            str(value["id"]),
            str(value["category"]),
            str(value["input"]),
            bool(value.get("execution_requested")),
            availability,
            SelectionPolicy(
                deepseek_auto_escalation=bool(
                    (value.get("policy") or {}).get("deepseek_auto_escalation", False)
                )
            ),
            value.get("legacy") or {},
            value.get("expected") or {},
            value.get("constraint_violation"),
            bool(value.get("confirmation_required")),
            bool(value.get("path_policy_satisfied", True)),
        )


def load_shadow_cases(path: str | Path) -> list[ShadowCase]:
    return [
        ShadowCase.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def legacy_decision_stub(values: Mapping[str, Any]) -> Any:
    """Rebuild a decision-shaped object so the real adapter is exercised."""

    intent_name = str(values.get("intent") or "ANSWER_DIRECTLY")
    try:
        intent = Intent(intent_name)
    except ValueError:
        intent = SimpleNamespace(value=intent_name)
    tools = tuple(
        item for item in (values.get("selected_action"),) if isinstance(item, str)
    )
    return SimpleNamespace(
        intent=intent,
        tools=tools,
        selected_action=tools[0] if tools else None,
        side_effects=tuple(
            SideEffect(item) for item in (values.get("side_effects") or ())
        ),
        requested_agent=values.get("requested_agent"),
        requested_agent_source=values.get("requested_agent_source"),
        constraint_violation=values.get("constraint_violation"),
        execution_allowed=values.get("execution_allowed"),
        intent_frame=None,
    )


class ShadowGateEvaluator:
    def __init__(
        self,
        *,
        semantic_selector: Any | None = None,
        baseline: Any | None = None,
    ):
        self.baseline = baseline or diagnostic_baseline(deepseek_available=True)
        self.semantic_selector = semantic_selector

    def _observe(self, case: ShadowCase) -> ShadowObservation:
        observer = ShadowExecutionObserver(
            policy=case.policy,
            semantic_selector=self.semantic_selector,
        )
        frame, _ = IntentFrameBuilder().build(
            normalize_technical_transcript(case.input),
            SimpleNamespace(active_project=None, project_root=None, known_projects=()),
        )
        return observer.observe(
            case.input,
            baseline=self.baseline,
            execution_requested=case.execution_requested,
            intent_frame=frame,
            legacy=legacy_facts_from_decision(legacy_decision_stub(case.legacy)),
            availability_override=case.availability,
            constraint_violation=case.constraint_violation,
            confirmation_required=case.confirmation_required,
            path_policy_satisfied=case.path_policy_satisfied,
        )

    def run(self, cases: Iterable[ShadowCase]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        divergences = {code: 0 for code in DIVERGENCE_CODES}
        blocks = {reason: 0 for reason in BLOCK_REASONS}
        counters = {
            "observations": 0,
            "observation_failures": 0,
            "proposal_valid": 0,
            "agent_agreement": 0,
            "execution_agreement": 0,
            "mutation_agreement": 0,
            "comparisons": 0,
            "explicit_total": 0,
            "explicit_preserved": 0,
            "ineligible_execution": 0,
            "unavailable_execution": 0,
            "policy_excluded_execution": 0,
            "execution_without_request": 0,
            "mutation_without_authorization": 0,
            "mutation_without_execution": 0,
            "provenance_complete": 0,
            "expectation_total": 0,
            "expectation_ok": 0,
            "bad_selection_factor": 0,
            "model_calls": 0,
            "executor_calls": 0,
            "job_creations": 0,
            "session_resolutions": 0,
            "mutations": 0,
        }
        deterministic_latencies: list[float] = []
        semantic_latencies: list[float] = []
        total_latencies: list[float] = []

        for case in cases:
            try:
                observation = self._observe(case)
            except Exception as exc:  # pragma: no cover - defensive
                counters["observation_failures"] += 1
                rows.append({"id": case.id, "error": type(exc).__name__})
                continue
            counters["observations"] += 1
            decision = observation.decision
            comparison = observation.comparison
            expected = case.expected
            failures: list[str] = []

            if decision.selection_valid:
                counters["proposal_valid"] += 1
            counters["provenance_complete"] += int(decision.provenance_complete)
            counters["model_calls"] += observation.model_calls
            counters["job_creations"] += decision.jobs_created
            counters["session_resolutions"] += decision.sessions_resolved
            counters["mutations"] += decision.filesystem_mutations
            counters["executor_calls"] += decision.delegations
            deterministic_latencies.append(observation.deterministic_latency_ms)
            total_latencies.append(observation.total_latency_ms)
            if observation.model_calls:
                semantic_latencies.append(observation.semantic_latency_ms)
            for item in observation.proposal.errors:
                if item.startswith("BAD_SELECTION_FACTOR"):
                    counters["bad_selection_factor"] += 1

            # hard safety counters
            if decision.execution_allowed and not decision.agent_eligible:
                counters["ineligible_execution"] += 1
            if decision.execution_allowed and not decision.agent_available:
                counters["unavailable_execution"] += 1
            if (
                decision.execution_allowed
                and decision.candidate_agent
                in case.policy.agents_requiring_explicit_request()
                and decision.selection_source is not SelectionSource.EXPLICIT_USER
            ):
                counters["policy_excluded_execution"] += 1
            if decision.execution_allowed and not decision.execution_requested:
                counters["execution_without_request"] += 1
            if decision.mutation_authorized and not decision.mutation_requested:
                counters["mutation_without_authorization"] += 1
            if decision.mutation_authorized and not decision.execution_allowed:
                counters["mutation_without_execution"] += 1

            if decision.block_reason is not None:
                blocks[decision.block_reason.value] += 1

            if comparison is not None:
                counters["comparisons"] += 1
                counters["agent_agreement"] += int(comparison.agent_agreement)
                counters["execution_agreement"] += int(comparison.execution_agreement)
                counters["mutation_agreement"] += int(comparison.mutation_agreement)
                for code in comparison.divergence_codes:
                    divergences[code.value] += 1

            if decision.selection_source is SelectionSource.EXPLICIT_USER:
                counters["explicit_total"] += 1
                requested = decision.requested_agent
                counters["explicit_preserved"] += int(
                    requested is not None
                    and decision.candidate_agent == requested
                    and observation.model_calls == 0
                )

            # corpus expectations
            checks: list[tuple[str, bool]] = []
            unresolved_ok = bool(expected.get("unresolved_acceptable"))
            if "candidate_agent" in expected:
                want = expected["candidate_agent"]
                actual = (
                    decision.candidate_agent.value if decision.candidate_agent else None
                )
                checks.append(("candidate_agent", actual == want))
            if expected.get("selection_sources"):
                checks.append(
                    (
                        "selection_source",
                        decision.selection_source.value in expected["selection_sources"],
                    )
                )
            if "execution_allowed" in expected:
                checks.append(
                    (
                        "execution_allowed",
                        decision.execution_allowed == bool(expected["execution_allowed"]),
                    )
                )
            if "mutation_requested" in expected:
                checks.append(
                    (
                        "mutation_requested",
                        decision.mutation_requested
                        == bool(expected["mutation_requested"]),
                    )
                )
            if "mutation_authorized" in expected:
                checks.append(
                    (
                        "mutation_authorized",
                        decision.mutation_authorized
                        == bool(expected["mutation_authorized"]),
                    )
                )
            if "block_reason" in expected:
                want_block = expected["block_reason"]
                actual_block = (
                    decision.block_reason.value if decision.block_reason else None
                )
                ok = actual_block == want_block
                if not ok and unresolved_ok and actual_block == "SELECTION_UNRESOLVED":
                    ok = True
                checks.append(("block_reason", ok))
            if expected.get("block_reason_includes"):
                present = {item.value for item in decision.block_reasons}
                checks.append(
                    (
                        "block_reason_includes",
                        set(expected["block_reason_includes"]).issubset(present),
                    )
                )
            if "expect_agent_agreement" in expected and comparison is not None:
                checks.append(
                    (
                        "agent_agreement",
                        comparison.agent_agreement
                        == bool(expected["expect_agent_agreement"]),
                    )
                )
            if expected.get("expect_divergence_codes") and comparison is not None:
                present = {item.value for item in comparison.divergence_codes}
                checks.append(
                    (
                        "divergence_codes",
                        set(expected["expect_divergence_codes"]).issubset(present),
                    )
                )
            for name, ok in checks:
                counters["expectation_total"] += 1
                counters["expectation_ok"] += int(ok)
                if not ok:
                    failures.append(name)

            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "input": case.input,
                    "finding": case.expected.get("finding"),
                    "failures": failures,
                    "shadow": decision.as_dict(),
                    "comparison": comparison.as_dict() if comparison else None,
                    "requirements": {
                        "unknown_dimensions": list(
                            observation.requirements.unknown_dimensions
                        ),
                        "conflict_dimensions": list(
                            observation.requirements.conflict_dimensions
                        ),
                    },
                    "model_calls": observation.model_calls,
                    "deterministic_latency_ms": observation.deterministic_latency_ms,
                    "total_latency_ms": observation.total_latency_ms,
                }
            )

        def ratio(ok: str, total: str) -> float:
            return counters[ok] / counters[total] if counters[total] else 1.0

        def rate(count: str) -> float:
            return (
                counters[count] / counters["observations"]
                if counters["observations"]
                else 0.0
            )

        def percentile(values: Sequence[float], percent: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[round((len(ordered) - 1) * percent)]

        total_cases = counters["observations"] + counters["observation_failures"]
        metrics = {
            "shadow_pipeline_success_rate": (
                counters["observations"] / total_cases if total_cases else 1.0
            ),
            "selection_proposal_valid_rate": rate("proposal_valid"),
            "legacy_shadow_agent_agreement": ratio("agent_agreement", "comparisons"),
            "legacy_shadow_execution_agreement": ratio(
                "execution_agreement", "comparisons"
            ),
            "legacy_shadow_mutation_agreement": ratio(
                "mutation_agreement", "comparisons"
            ),
            "explicit_agent_preservation": ratio("explicit_preserved", "explicit_total"),
            "ineligible_shadow_execution_rate": rate("ineligible_execution"),
            "unavailable_shadow_execution_rate": rate("unavailable_execution"),
            "policy_excluded_shadow_execution_rate": rate("policy_excluded_execution"),
            "execution_without_request_shadow_rate": rate("execution_without_request"),
            "mutation_without_authorization_shadow_rate": rate(
                "mutation_without_authorization"
            ),
            "mutation_without_execution_shadow_rate": rate("mutation_without_execution"),
            "selection_provenance_completeness": rate("provenance_complete"),
            "corpus_expectation_accuracy": ratio("expectation_ok", "expectation_total"),
            "bad_selection_factor_rate": rate("bad_selection_factor"),
            "model_calls": counters["model_calls"],
            "model_calls_per_case": (
                counters["model_calls"] / counters["observations"]
                if counters["observations"]
                else 0.0
            ),
            "deterministic_latency_p50_ms": percentile(deterministic_latencies, 0.50),
            "deterministic_latency_p90_ms": percentile(deterministic_latencies, 0.90),
            "deterministic_latency_p95_ms": percentile(deterministic_latencies, 0.95),
            "shadow_total_latency_p50_ms": percentile(total_latencies, 0.50),
            "shadow_total_latency_p90_ms": percentile(total_latencies, 0.90),
            "shadow_total_latency_p95_ms": percentile(total_latencies, 0.95),
            "semantic_latency_p50_ms": percentile(semantic_latencies, 0.50),
            "semantic_latency_p95_ms": percentile(semantic_latencies, 0.95),
        }
        safety = {
            "authority": "SHADOW",
            "live_authority": False,
            "shadow_executor_calls": counters["executor_calls"],
            "shadow_job_creations": counters["job_creations"],
            "shadow_session_resolutions": counters["session_resolutions"],
            "shadow_mutations": counters["mutations"],
        }
        return {
            "mode": "SHADOW",
            "cases": total_cases,
            "metrics": metrics,
            "safety": safety,
            "divergences": divergences,
            "findings": sorted(
                {
                    str(row["finding"])
                    for row in rows
                    if row.get("finding")
                }
            ),
            "block_reasons": {
                name: value for name, value in blocks.items() if value
            },
            "rows": rows,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execution gate shadow evaluator (observation only)"
    )
    parser.add_argument("--corpus", default="tests/data/execution_gate_shadow.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = ShadowGateEvaluator().run(load_shadow_cases(args.corpus))
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
