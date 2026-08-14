from __future__ import annotations

import json
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .decision_observability import estimate_tokens, latency_summary
from .decision_policy import AgentDecisionPolicy, Intent
from .projects import normalize_technical_transcript
from .semantic_pass import QwenSemanticInterpreter
from .semantic_pass import SemanticValidationCode


CASES_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "command_routing_diagnostic.jsonl"
)


def load_command_routing_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Load the separate command-routing diagnostic corpus.

    Command type and expected route are fixture labels. This evaluator never infers
    either label from keywords in the user text.
    """
    cases: list[dict[str, Any]] = []
    valid_routes = {item.value for item in Intent}
    valid_types = {
        "EXPLICIT_COMMAND",
        "ACTUALLY_AMBIGUOUS_COMMAND",
        "INFORMATIONAL_REQUEST",
        "UNAVAILABLE_CAPABILITY",
    }
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid command-routing case at line {number}")
        missing = {"id", "input", "command_type", "context"} - set(value)
        if missing:
            raise ValueError(
                f"command-routing case missing {sorted(missing)} at line {number}"
            )
        if value["command_type"] not in valid_types:
            raise ValueError(f"invalid command_type at line {number}")
        expected_route = value.get("expected_route")
        if expected_route is not None and expected_route not in valid_routes:
            raise ValueError(f"invalid expected_route at line {number}")
        expected_tools = value.get("expected_tools", [])
        if not isinstance(expected_tools, list) or not all(
            isinstance(item, str) for item in expected_tools
        ):
            raise ValueError(f"invalid expected_tools at line {number}")
        cases.append(value)
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate command-routing case ids")
    return cases


class _InstrumentedClient:
    def __init__(self, wrapped: Any):
        self.wrapped = wrapped
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        started = time.perf_counter()
        event: dict[str, Any] = {
            "prompt_tokens_estimate": estimate_tokens(messages),
            "tools_exposed": bool(kwargs.get("tools")),
        }
        try:
            response = self.wrapped.chat(messages, **kwargs)
        except Exception as exc:
            event.update(
                {
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.requests.append(event)
            raise
        choice = response.get("choices", [{}])[0] if isinstance(response, dict) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        usage = response.get("usage") if isinstance(response, dict) else None
        event.update(
            {
                "latency_ms": (time.perf_counter() - started) * 1000,
                "response_content": content,
                "generated_tokens_estimate": estimate_tokens(content),
                "usage": usage if isinstance(usage, dict) else None,
                "finish_reason": (
                    choice.get("finish_reason") if isinstance(choice, dict) else None
                ),
            }
        )
        self.requests.append(event)
        return response


def observe_current_routing(
    case: dict[str, Any],
    *,
    client: Any,
    available_tools: set[str],
    enabled_tools: set[str],
    cross_field_invariants: frozenset[SemanticValidationCode] = frozenset(),
    semantic_system_prompt: str | None = None,
) -> dict[str, Any]:
    """Observe production semantic gating and policy mapping without execution."""
    started = time.perf_counter()
    text = str(case["input"])
    policy = AgentDecisionPolicy(context_cache_enabled=True)
    context = policy.build_context(fixture_context=dict(case.get("context") or {}))
    normalized = normalize_technical_transcript(text)
    instrumented = _InstrumentedClient(client)
    interpreter = QwenSemanticInterpreter(
        instrumented,
        cross_field_invariants=cross_field_invariants,
        system_prompt=semantic_system_prompt,
    )
    semantic_needed = interpreter.needs_semantic_pass(normalized, context)
    semantic = (
        interpreter.interpret(text, normalized, context)
        if semantic_needed
        else interpreter.skipped()
    )
    decision = policy.decide(
        text,
        context=context,
        semantic_decision=semantic.decision,
    )
    if semantic.used and not semantic.parse_valid:
        decision = policy.safe_fallback_decision(decision)

    case_disabled = {str(item) for item in case.get("disabled_tools") or []}
    effective_enabled = set(enabled_tools) - case_disabled
    selected_tool = decision.selected_action
    requests = instrumented.requests
    semantic_attempts: list[dict[str, Any]] = []
    for item in requests:
        try:
            raw_attempt = json.loads(str(item.get("response_content") or ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            semantic_attempts.append({"parseable": False})
            continue
        semantic_attempts.append(
            {
                "parseable": isinstance(raw_attempt, dict),
                "primary_intent": (
                    raw_attempt.get("primary_intent")
                    if isinstance(raw_attempt, dict)
                    else None
                ),
                "operation": (
                    raw_attempt.get("operation")
                    if isinstance(raw_attempt, dict)
                    else None
                ),
                "agent": (
                    raw_attempt.get("agent")
                    if isinstance(raw_attempt, dict)
                    else None
                ),
                "execution_requested": (
                    raw_attempt.get("execution_requested")
                    if isinstance(raw_attempt, dict)
                    else None
                ),
                "constraints": (
                    raw_attempt.get("constraints")
                    if isinstance(raw_attempt, dict)
                    and isinstance(raw_attempt.get("constraints"), list)
                    else []
                ),
            }
        )
    exact_prompt_tokens = sum(
        int((item.get("usage") or {}).get("prompt_tokens") or 0)
        for item in requests
    )
    exact_generated_tokens = sum(
        int((item.get("usage") or {}).get("completion_tokens") or 0)
        for item in requests
    )
    semantic_frame = semantic.decision.as_dict() if semantic.decision else None
    reference = decision.resolved_reference.as_dict() if decision.resolved_reference else {}
    target_type = (
        reference.get("type")
        or ((semantic_frame or {}).get("target") or {}).get("type")
    )
    target_value = decision.target or reference.get("id")
    ambiguity_present = bool(
        (semantic_frame or {}).get("ambiguity", {}).get("present")
        or reference.get("ambiguous")
    )
    configured_execution_allowed = case.get("execution_allowed")
    execution_allowed = (
        bool(configured_execution_allowed)
        if configured_execution_allowed is not None
        else (selected_tool in effective_enabled if selected_tool else None)
    )
    return {
        "request_id": f"command-routing-{uuid.uuid4()}",
        "decision": decision.as_dict(),
        "semantic_pass_used": semantic.used,
        "semantic_parse_valid": semantic.parse_valid,
        "semantic_repair_used": semantic.repair_used,
        "semantic_validation_error_codes": list(
            semantic.validation_error_codes
        ),
        "semantic_attempts": semantic_attempts,
        "semantic_frame": semantic_frame,
        "retry": max(0, len(requests) - 1),
        "fallback": decision.reason_code.startswith("semantic_parse_failed"),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": exact_prompt_tokens or sum(
            int(item.get("prompt_tokens_estimate") or 0) for item in requests
        ),
        "prompt_tokens_estimated": exact_prompt_tokens == 0 and bool(requests),
        "generated_tokens": exact_generated_tokens or sum(
            int(item.get("generated_tokens_estimate") or 0) for item in requests
        ),
        "generated_tokens_estimated": exact_generated_tokens == 0 and bool(requests),
        "finish_reason": requests[-1].get("finish_reason") if requests else None,
        "available_tools": sorted(available_tools),
        "enabled_tools": sorted(effective_enabled),
        "disabled_tools": sorted(available_tools - effective_enabled),
        "selected_tool": selected_tool,
        "tool_available": (
            selected_tool in effective_enabled if selected_tool is not None else None
        ),
        "explicit_target_detected": bool(
            target_type not in {None, "none"}
            and target_value
            and not ambiguity_present
        ),
        "target_type": target_type,
        "ambiguity_present": ambiguity_present,
        "clarify_reason": (
            decision.reason_code if decision.intent is Intent.CLARIFY else None
        ),
        "execution_attempted": False,
        "execution_allowed": execution_allowed,
        "constraint_violation": decision.constraint_violation,
    }


def evaluate_command_routing(
    cases: list[dict[str, Any]],
    observe: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    route_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    preservation: Counter[str] = Counter()
    preservation_cases: dict[str, list[str]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    preservation_scored = 0
    preservation_correct = 0
    imperative_total = 0
    imperative_correct = 0
    execution_true_positive = 0
    execution_false_positive = 0
    execution_false_negative = 0
    informational_total = 0
    operation_total = 0
    operation_correct = 0
    agent_total = 0
    agent_correct = 0
    speech_act_total = 0
    speech_act_correct = 0

    for case in cases:
        observed = observe(case)
        decision = observed["decision"]
        predicted_route = str(decision["intent"])
        expected_route = case.get("expected_route")
        scored = expected_route is not None
        routing_correct = predicted_route == expected_route if scored else None
        expected_tools = list(case.get("expected_tools") or [])
        predicted_tools = list(decision.get("tools") or [])
        plan_valid = predicted_tools == expected_tools if scored else None
        frame = observed.get("semantic_frame") or decision.get("intent_frame") or {}
        semantic_parse_valid = bool(
            observed.get("semantic_parse_valid", observed.get("semantic_valid", True))
        )
        semantic_checks = []
        if case.get("expected_speech_act") is not None:
            semantic_checks.append(
                frame.get("speech_act") == case.get("expected_speech_act")
            )
        if case.get("expected_execution_requested") is not None:
            semantic_checks.append(
                frame.get("execution_requested")
                is case.get("expected_execution_requested")
            )
        if case.get("expected_target_type") is not None:
            semantic_checks.append(
                observed.get("target_type") == case.get("expected_target_type")
                and bool(observed.get("explicit_target_detected"))
            )
        semantic_primary_intent = (
            (observed.get("semantic_frame") or {}).get("primary_intent")
        )
        if (
            observed.get("semantic_pass_used")
            and expected_route is not None
            and semantic_primary_intent is not None
        ):
            semantic_checks.append(semantic_primary_intent == expected_route)
        semantic_valid = semantic_parse_valid and all(semantic_checks)
        record = {
            "case_id": case["id"],
            "source_case_id": case.get("source_case_id"),
            "command_type": case["command_type"],
            "input": case["input"],
            "expected_route": expected_route,
            "predicted_route": predicted_route,
            "routing_correct": routing_correct,
            "expected_tools": expected_tools,
            "planned_tools": predicted_tools,
            "plan_valid": plan_valid,
            "semantic_parse_valid": semantic_parse_valid,
            "semantic_valid": semantic_valid,
            "semantic_primary_intent": semantic_primary_intent,
            "reason_code": decision.get("reason_code"),
            "speech_act": frame.get("speech_act"),
            "operation": frame.get("operation"),
            "agent": frame.get("agent"),
            "execution_requested": frame.get("execution_requested"),
            **{
                key: value
                for key, value in observed.items()
                if key not in {"decision", "semantic_parse_valid", "semantic_valid"}
            },
        }
        records.append(record)
        if scored:
            route_counts[str(expected_route)] += 1
            predicted_counts[predicted_route] += 1
            correct_counts[str(expected_route)] += bool(routing_correct)
            confusion[str(expected_route)][predicted_route] += 1
            if not routing_correct:
                transitions[f"{expected_route} -> {predicted_route}"] += 1

        issues: list[str] = []
        explicit_command = case["command_type"] == "EXPLICIT_COMMAND"
        if (
            explicit_command
            and case.get("expected_speech_act") == "COMMAND"
            and frame.get("speech_act") is not None
            and frame.get("speech_act") != "COMMAND"
        ):
            issues.append("imperative_to_non_command")
        if (
            explicit_command
            and case.get("expected_execution_requested") is True
            and frame.get("execution_requested") is False
        ):
            issues.append("execution_request_to_informational")
        if (
            case.get("expected_execution_requested") is False
            and frame.get("execution_requested") is True
        ):
            issues.append("informational_request_to_execution")
        if (
            case["command_type"] == "EXPLICIT_COMMAND"
            and case.get("expected_target_type")
            and not observed.get("explicit_target_detected")
        ):
            issues.append("explicit_target_to_ambiguous_reference")
        if expected_tools and predicted_route == Intent.ANSWER_DIRECTLY.value:
            issues.append("tool_command_to_answer_directly")
        for issue in issues:
            preservation[issue] += 1
            preservation_cases[issue].append(str(case["id"]))
        record["command_preservation_issues"] = issues

        preservation_checks: list[bool] = []
        expected_speech_act = case.get("expected_speech_act")
        if expected_speech_act is not None:
            speech_act_total += 1
            speech_matches = frame.get("speech_act") == expected_speech_act
            speech_act_correct += bool(speech_matches)
            preservation_checks.append(speech_matches)
            if expected_speech_act == "COMMAND":
                imperative_total += 1
                imperative_correct += bool(speech_matches)
        if "expected_execution_requested" in case:
            expected_execution = case["expected_execution_requested"]
            predicted_execution = frame.get("execution_requested")
            execution_matches = predicted_execution is expected_execution
            preservation_checks.append(execution_matches)
            if expected_execution is True:
                execution_true_positive += predicted_execution is True
                execution_false_negative += predicted_execution is not True
            else:
                informational_total += 1
                execution_false_positive += predicted_execution is True
        if case.get("expected_operation") is not None:
            operation_total += 1
            operation_matches = frame.get("operation") == case["expected_operation"]
            operation_correct += bool(operation_matches)
            preservation_checks.append(operation_matches)
        if "expected_agent" in case:
            agent_total += 1
            agent_matches = frame.get("agent") == case["expected_agent"]
            agent_correct += bool(agent_matches)
            preservation_checks.append(agent_matches)
        if preservation_checks:
            preservation_scored += 1
            preservation_correct += all(preservation_checks)
        record["command_preservation_correct"] = (
            all(preservation_checks) if preservation_checks else None
        )

    scored_records = [item for item in records if item["routing_correct"] is not None]
    count = len(scored_records)
    routes = sorted(set(route_counts) | set(predicted_counts))
    accuracy_by_route = {
        route: correct_counts[route] / route_counts[route]
        for route in routes
        if route_counts[route]
    }
    precision_by_route: dict[str, float | None] = {}
    recall_by_route: dict[str, float | None] = {}
    for route in routes:
        true_positive = confusion[route][route]
        predicted = sum(row[route] for row in confusion.values())
        expected = sum(confusion[route].values())
        precision_by_route[route] = true_positive / predicted if predicted else None
        recall_by_route[route] = true_positive / expected if expected else None

    expected_tool = [item for item in scored_records if item["expected_tools"]]
    expected_no_tool = [item for item in scored_records if not item["expected_tools"]]
    predicted_clarify = [
        item for item in scored_records if item["predicted_route"] == Intent.CLARIFY.value
    ]
    expected_not_clarify = [
        item for item in scored_records if item["expected_route"] != Intent.CLARIFY.value
    ]
    false_clarify = [
        item for item in predicted_clarify if item["expected_route"] != Intent.CLARIFY.value
    ]
    missed_tool = [item for item in expected_tool if not item["planned_tools"]]
    unnecessary_tool = [item for item in expected_no_tool if item["planned_tools"]]
    wrong_tool = [
        item
        for item in expected_tool
        if item["planned_tools"] and not item["plan_valid"]
    ]
    selected = [item for item in records if item.get("selected_tool")]
    unavailable = [item for item in selected if item.get("tool_available") is False]
    invalid_frames = [
        item
        for item in records
        if item.get("semantic_validation_error_codes")
    ]
    invalid_recovered = [
        item for item in invalid_frames if item.get("routing_correct") is True
    ]
    invalid_fallback = [item for item in invalid_frames if item.get("fallback")]
    safety_violations = [
        item
        for item, case in zip(records, cases)
        if case.get("execution_allowed") is False
        and (
            item.get("execution_attempted")
            or item.get("execution_allowed") is True
        )
    ]
    quadrants = Counter(
        (
            "routing_correct" if item["routing_correct"] else "routing_wrong",
            "plan_valid" if item["plan_valid"] else "plan_invalid",
        )
        for item in scored_records
    )
    quadrant_names = (
        "routing_correct / plan_valid",
        "routing_correct / plan_invalid",
        "routing_wrong / plan_valid",
        "routing_wrong / plan_invalid",
    )
    return {
        "cases": len(records),
        "scored_cases": count,
        "routing_accuracy": (
            sum(bool(item["routing_correct"]) for item in scored_records) / count
            if count
            else None
        ),
        "accuracy_by_route": accuracy_by_route,
        "precision_by_route": precision_by_route,
        "recall_by_route": recall_by_route,
        "false_clarify_rate": (
            len(false_clarify) / len(expected_not_clarify)
            if expected_not_clarify
            else None
        ),
        "clarify_precision": (
            sum(item["expected_route"] == Intent.CLARIFY.value for item in predicted_clarify)
            / len(predicted_clarify)
            if predicted_clarify
            else None
        ),
        "missed_tool_rate": (
            len(missed_tool) / len(expected_tool) if expected_tool else None
        ),
        "unnecessary_tool_rate": (
            len(unnecessary_tool) / len(expected_no_tool)
            if expected_no_tool
            else None
        ),
        "wrong_tool_rate": (
            len(wrong_tool) / len(expected_tool) if expected_tool else None
        ),
        "unavailable_tool_selection_rate": (
            len(unavailable) / len(selected) if selected else None
        ),
        "invalid_frames_caught": len(invalid_frames),
        "invalid_frames_recovered_after_retry": len(invalid_recovered),
        "invalid_frames_fallback": len(invalid_fallback),
        "route_plan_quadrants": {
            name: quadrants[tuple(name.split(" / "))]
            for name in quadrant_names
        },
        "confusion_matrix": {
            expected: dict(predicted)
            for expected, predicted in sorted(confusion.items())
        },
        "top_route_errors": dict(transitions.most_common()),
        "false_clarify_cases": [item["case_id"] for item in false_clarify],
        "clarify_reason_codes": dict(
            Counter(
                str(item.get("clarify_reason") or "none")
                for item in predicted_clarify
            )
        ),
        "command_preservation": {
            "counts": dict(preservation),
            "cases": dict(preservation_cases),
        },
        "command_preservation_accuracy": (
            preservation_correct / preservation_scored
            if preservation_scored else None
        ),
        "imperative_preservation_accuracy": (
            imperative_correct / imperative_total if imperative_total else None
        ),
        "execution_request_recall": (
            execution_true_positive
            / (execution_true_positive + execution_false_negative)
            if execution_true_positive + execution_false_negative else None
        ),
        "execution_request_precision": (
            execution_true_positive
            / (execution_true_positive + execution_false_positive)
            if execution_true_positive + execution_false_positive else None
        ),
        "operation_preservation_accuracy": (
            operation_correct / operation_total if operation_total else None
        ),
        "agent_preservation_accuracy": (
            agent_correct / agent_total if agent_total else None
        ),
        "speech_act_accuracy": (
            speech_act_correct / speech_act_total if speech_act_total else None
        ),
        "false_informational_rate": (
            execution_false_negative
            / (execution_true_positive + execution_false_negative)
            if execution_true_positive + execution_false_negative else None
        ),
        "false_execution_rate": (
            execution_false_positive / informational_total
            if informational_total else None
        ),
        "informational_to_execution_regressions": execution_false_positive,
        "retry": sum(int(item.get("retry") or 0) for item in records),
        "retry_rate": (
            sum(bool(item.get("retry")) for item in records) / len(records)
            if records
            else None
        ),
        "fallback": sum(bool(item.get("fallback")) for item in records),
        "fallback_rate": (
            sum(bool(item.get("fallback")) for item in records) / len(records)
            if records
            else None
        ),
        "safety_violations": len(safety_violations),
        "constraint_violations": sum(
            bool(item.get("constraint_violation")) for item in records
        ),
        "latency": latency_summary(
            [float(item["latency_ms"]) for item in records]
        ),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in records),
        "generated_tokens": sum(
            int(item.get("generated_tokens") or 0) for item in records
        ),
        "finish_reasons": dict(
            Counter(
                str(item.get("finish_reason") or "none") for item in records
            )
        ),
        "records": records,
    }
