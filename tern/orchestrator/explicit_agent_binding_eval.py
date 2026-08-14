from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .decision_policy import AgentDecisionPolicy, Decision, Intent
from .explicit_agent_binding import (
    availability_for_requested_agent,
    detect_explicit_agent_binding,
)
from .projects import normalize_technical_transcript
from .semantic_pass import QwenSemanticInterpreter, SemanticPassResult


CASES_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "explicit_agent_binding_diagnostic.jsonl"
)
REGISTERED_AGENT_TOOLS = frozenset(
    {"delegate_to_codex", "delegate_to_deepseek"}
)


SemanticProvider = Callable[[str, str, Any], SemanticPassResult]


def load_explicit_agent_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _route_agent(decision: Decision) -> str | None:
    if decision.intent is Intent.DEEPSEEK_DELEGATE:
        return "deepseek"
    if decision.intent is Intent.CODEX_DELEGATE:
        return "codex"
    return None


def _apply_semantic_fallback(
    policy: AgentDecisionPolicy,
    decision: Decision,
    semantic_result: SemanticPassResult,
) -> Decision:
    if semantic_result.used and not semantic_result.parse_valid:
        return policy.safe_fallback_decision(decision)
    return decision


def _score_variant(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    total = len(records)
    routing_hits = 0
    false_clarify = 0
    missed_tool = 0
    wrong_tool = 0
    wrong_agent = 0
    explicit = 0
    availability_corruption = 0
    unavailable_explicit = 0
    for record in records:
        expected = record["expected"]
        actual = record[variant]
        if actual["intent"] == expected["intent"]:
            routing_hits += 1
        if actual["intent"] == "CLARIFY" and expected["intent"] != "CLARIFY":
            false_clarify += 1
        expected_tool = expected.get("tool")
        actual_tool = actual.get("tool")
        if expected_tool and actual_tool is None:
            missed_tool += 1
        elif expected_tool != actual_tool and actual_tool is not None:
            wrong_tool += 1
        requested = expected.get("requested_agent")
        if requested:
            explicit += 1
            if actual.get("route_agent") != requested:
                wrong_agent += 1
            if (
                expected.get("tool_available") is False
            ):
                unavailable_explicit += 1
                if (
                    actual.get("route_agent") != requested
                    or actual.get("requested_agent") != requested
                ):
                    availability_corruption += 1
    return {
        "routing_accuracy": routing_hits / total if total else 0.0,
        "false_clarify": false_clarify,
        "missed_tool": missed_tool,
        "wrong_tool": wrong_tool,
        "wrong_agent_rate": wrong_agent / explicit if explicit else 0.0,
        "availability_intent_corruption_rate": (
            availability_corruption / unavailable_explicit
            if unavailable_explicit
            else 0.0
        ),
    }


def evaluate_explicit_agent_binding_ab(
    *,
    cases: Iterable[dict[str, Any]] | None = None,
    semantic_provider: SemanticProvider | None = None,
    registered_tools: Iterable[str] = REGISTERED_AGENT_TOOLS,
) -> dict[str, Any]:
    """Apply A and B to the exact same semantic result for each case."""

    selected = list(cases if cases is not None else load_explicit_agent_cases())
    registered = frozenset(registered_tools)
    records: list[dict[str, Any]] = []
    semantic_calls = 0
    semantic_retries = 0
    semantic_fallbacks = 0
    semantic_latency_ms = 0.0
    policy_latency_a = 0.0
    policy_latency_b = 0.0
    true_positive = false_positive = false_negative = 0

    for case in selected:
        text = str(case["input"])
        routing_text = normalize_technical_transcript(text)
        fixture = dict(case.get("context") or {})
        policy_a = AgentDecisionPolicy()
        policy_b = AgentDecisionPolicy()
        context_a = policy_a.build_context(fixture_context=fixture)
        context_b = policy_b.build_context(fixture_context=fixture)
        if semantic_provider is None:
            semantic_result = QwenSemanticInterpreter.skipped()
        else:
            semantic_result = semantic_provider(text, routing_text, context_a)
            semantic_calls += 1
            semantic_retries += int(semantic_result.repair_used)
            semantic_fallbacks += int(
                semantic_result.used and not semantic_result.parse_valid
            )
            semantic_latency_ms += semantic_result.latency_ms

        started = time.perf_counter()
        decision_a = policy_a.decide(
            text,
            context=context_a,
            semantic_decision=semantic_result.decision,
        )
        decision_a = _apply_semantic_fallback(policy_a, decision_a, semantic_result)
        policy_latency_a += (time.perf_counter() - started) * 1000

        binding = detect_explicit_agent_binding(routing_text)
        expected_agent = case["expected"].get("requested_agent")
        detected_agent = binding.requested_agent if binding else None
        if detected_agent and detected_agent == expected_agent:
            true_positive += 1
        elif detected_agent:
            false_positive += 1
        elif expected_agent:
            false_negative += 1

        started = time.perf_counter()
        decision_b = policy_b.decide(
            text,
            context=context_b,
            semantic_decision=semantic_result.decision,
            explicit_agent_binding=binding,
        )
        decision_b = _apply_semantic_fallback(policy_b, decision_b, semantic_result)
        if binding is not None:
            availability = availability_for_requested_agent(
                binding,
                context_b,
                registered,
            )
            decision_b = replace(
                decision_b,
                tool_registered=availability.tool_registered,
                tool_available=availability.tool_available,
                execution_allowed=(
                    availability.execution_allowed
                    and decision_b.constraint_violation is None
                ),
                availability_reason=availability.reason,
            )
        policy_latency_b += (time.perf_counter() - started) * 1000

        def snapshot(decision: Decision) -> dict[str, Any]:
            return {
                "intent": decision.intent.value,
                "tool": decision.selected_action,
                "route_agent": _route_agent(decision),
                "requested_agent": decision.requested_agent,
                "requested_agent_source": decision.requested_agent_source,
                "tool_registered": decision.tool_registered,
                "tool_available": decision.tool_available,
                "execution_allowed": decision.execution_allowed,
                "availability_reason": decision.availability_reason,
                "reason_code": decision.reason_code,
            }

        records.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": text,
                "expected": case["expected"],
                "semantic": semantic_result.as_dict(),
                "A": snapshot(decision_a),
                "B": snapshot(decision_b),
            }
        )

    positives = true_positive + false_negative
    precision_denominator = true_positive + false_positive
    deepseek_records = [
        record
        for record in records
        if record["expected"].get("requested_agent") == "deepseek"
    ]
    codex_records = [
        record
        for record in records
        if record["expected"].get("requested_agent") == "codex"
    ]
    negative_records = [
        record
        for record in records
        if record["expected"].get("requested_agent") is None
    ]

    def route_accuracy(items: list[dict[str, Any]], agent: str) -> float:
        if not items:
            return 0.0
        return sum(record["B"]["route_agent"] == agent for record in items) / len(items)

    preservation = sum(
        record["B"]["requested_agent"] == record["expected"].get("requested_agent")
        for record in [*deepseek_records, *codex_records]
    )
    explicit_count = len(deepseek_records) + len(codex_records)
    report = {
        "experiment": "explicit_agent_binding",
        "cases": len(records),
        "ab_control": {
            "same_semantic_result_per_case": True,
            "semantic_calls": semantic_calls,
            "additional_inference_calls_B": 0,
            "schema_changed": False,
            "prompt_changed": False,
        },
        "metrics": {
            "explicit_agent_detection_precision": (
                true_positive / precision_denominator if precision_denominator else 1.0
            ),
            "explicit_agent_detection_recall": (
                true_positive / positives if positives else 1.0
            ),
            "explicit_deepseek_route_accuracy": route_accuracy(
                deepseek_records, "deepseek"
            ),
            "explicit_codex_route_accuracy": route_accuracy(codex_records, "codex"),
            "explicit_agent_preservation": (
                preservation / explicit_count if explicit_count else 1.0
            ),
            "false_agent_binding_rate": (
                false_positive / len(negative_records) if negative_records else 0.0
            ),
            "A": _score_variant(records, "A"),
            "B": _score_variant(records, "B"),
            "retry": semantic_retries,
            "fallback": semantic_fallbacks,
            "latency_ms": {
                "semantic_average": (
                    semantic_latency_ms / semantic_calls if semantic_calls else 0.0
                ),
                "policy_A_average": policy_latency_a / len(records) if records else 0.0,
                "policy_B_average": policy_latency_b / len(records) if records else 0.0,
            },
            "tokens": {
                "additional_prompt_tokens_B": 0,
                "additional_completion_tokens_B": 0,
            },
        },
        "records": records,
    }
    return report


def live_semantic_provider(interpreter: QwenSemanticInterpreter) -> SemanticProvider:
    def provide(original: str, normalized: str, context: Any) -> SemanticPassResult:
        return interpreter.interpret(original, normalized, context)

    return provide
