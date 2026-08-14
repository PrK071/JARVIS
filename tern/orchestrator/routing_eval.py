from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .decision_observability import estimate_tokens, latency_summary


CASES_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "agent_routing_cases.jsonl"
SEMANTIC_V2_MANIFEST = CASES_PATH.parent / "agent_routing_semantic_regression_v2.json"
V3_FAILURE_AUDIT_PATH = CASES_PATH.parent / "agent_routing_v3_failure_audit.json"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "expected" not in value:
            raise ValueError(f"invalid routing case at line {number}")
        cases.append(value)
    return cases


def load_semantic_regression_v2(
    manifest_path: Path = SEMANTIC_V2_MANIFEST,
) -> list[dict[str, Any]]:
    """Resolve the known v3 failures plus new semantic variations."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("invalid semantic regression v2 manifest")
    root = manifest_path.parent
    result_path = root / "agent_routing_test_v3_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    failure_ids = set(result.get("failure_ids") or [])
    known_v3 = [
        case
        for case in load_cases(root / "agent_routing_test_v3.jsonl")
        if case.get("id") in failure_ids
    ]
    variations = load_cases(root / "agent_routing_semantic_regression_v2_variations.jsonl")
    cases = [*known_v3, *variations]
    expected = int(manifest.get("total_cases") or 0)
    if len(known_v3) != 33 or len(cases) != expected:
        raise ValueError(
            f"semantic regression v2 composition mismatch: v3={len(known_v3)} total={len(cases)}"
        )
    ids = [str(case.get("id")) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("semantic regression v2 contains duplicate ids")
    return cases


def load_v3_failure_audit() -> dict[str, Any]:
    """Join sealed v3 cases with preserved cause analysis and heuristic frames."""
    from .decision_policy import AgentDecisionPolicy

    source = json.loads(V3_FAILURE_AUDIT_PATH.read_text(encoding="utf-8"))
    result = json.loads(
        (V3_FAILURE_AUDIT_PATH.parent / "agent_routing_test_v3_result.json").read_text(
            encoding="utf-8"
        )
    )
    failures = source.get("failures") if isinstance(source, dict) else None
    if not isinstance(failures, list):
        raise ValueError("invalid v3 failure audit")
    expected_ids = {str(item) for item in result.get("failure_ids") or []}
    cases = {
        str(case.get("id")): case
        for case in load_cases(V3_FAILURE_AUDIT_PATH.parent / "agent_routing_test_v3.jsonl")
    }
    records: list[dict[str, Any]] = []
    for failure in failures:
        if not isinstance(failure, dict):
            raise ValueError("invalid v3 failure audit record")
        case_id = str(failure.get("id") or "")
        case = cases.get(case_id)
        if case is None or case_id not in expected_ids:
            raise ValueError(f"v3 failure audit case missing: {case_id}")
        policy = AgentDecisionPolicy()
        current = policy.decide(
            str(case["input"]), fixture_context=dict(case.get("context") or {})
        ).as_dict()
        records.append(
            {
                "id": case_id,
                "phrase": case["input"],
                "context": case.get("context") or {},
                "heuristic_frame": current.get("intent_frame"),
                "expected": case["expected"],
                "heuristic_decision": {
                    "intent": current.get("intent"),
                    "tools": current.get("tools"),
                    "reason_code": current.get("reason_code"),
                },
                "expected_decision": {
                    "intent": (case.get("expected") or {}).get("intent"),
                    "tools": (case.get("expected") or {}).get("tools", []),
                },
                "lost": failure.get("lost") or [],
                "origin": failure.get("origin"),
                "cause": failure.get("cause"),
            }
        )
    if {record["id"] for record in records} != expected_ids:
        raise ValueError("v3 failure audit does not cover sealed failures")
    return {
        "failure_origin": source.get("failure_origin") or {},
        "records": records,
    }


def legacy_decision(text: str, context: dict[str, Any]) -> dict[str, Any]:
    """Frozen view of the deterministic routing gates that predate AgentDecisionPolicy."""
    from .agent import (
        _codex_job_intent,
        _deepseek_intent,
        _is_codex_history_request,
        _normalized,
        _project_lookup_only,
    )
    # Deliberately frozen: later STT improvements must not rewrite the BEFORE
    # measurement. These are exactly the technical corrections available when
    # the baseline was captured.
    routed = text
    if re.search(
        r"(?i)\b(?:arquivo|codigo|c[oó]digo|projeto|assistente|jarves|jarvis|"
        r"config|diretorio|pasta|sessao|thread|teste|bridge|voz|provider|"
        r"codex|c[oó]dex|c[oó]digo ex|terne|ll?ama|lama ponto cpp)\b",
        text,
    ):
        for pattern, replacement in (
            (r"(?i)\bc[oó]digo\s+ex\b|\bc[oó]dex\b", "Codex"),
            (r"(?i)\bterne\b", "Tern"),
            (r"(?i)\blama\s+ponto\s+cpp\b", "llama.cpp"),
            (r"(?i)\bjarves\b", "Jarvis"),
        ):
            routed = re.sub(pattern, replacement, routed)
    normalized = _normalized(routed)
    active = context.get("active_project")
    deepseek = _deepseek_intent(routed)
    history = _is_codex_history_request(routed)
    if deepseek:
        tools: list[str] = []
        if "codex" in normalized and history:
            tools.append("review_codex_session")
        tools.append(deepseek)
        if "codex" in normalized and any(
            term in normalized
            for term in ("depois", "implemente", "implementar", "mande o codex", "peca ao codex")
        ):
            tools.append("delegate_to_codex")
        return {
            "intent": "DEEPSEEK_REVIEW" if deepseek == "review_deepseek_session" else "DEEPSEEK_DELEGATE",
            "tools": tools,
            "project": active,
            "new_codex_turn": "delegate_to_codex" in tools,
            "reason_code": "legacy_deepseek_gate",
            "confidence": 0.70,
        }
    job = _codex_job_intent(routed)
    if job:
        intent = {
            "get_codex_job_status": "CODEX_STATUS",
            "steer_codex_job": "CODEX_STEER",
            "cancel_codex_job": "CODEX_CANCEL",
        }[job]
        return {
            "intent": intent,
            "tools": [job],
            "project": active,
            "new_codex_turn": False,
            "reason_code": "legacy_codex_job_gate",
            "confidence": 0.75,
        }
    if history:
        return {
            "intent": "CODEX_REVIEW",
            "tools": ["review_codex_session"],
            "project": active,
            "new_codex_turn": False,
            "reason_code": "legacy_codex_history_gate",
            "confidence": 0.75,
        }
    if _project_lookup_only(routed):
        return {
            "intent": "LOCAL_SEARCH",
            "tools": ["resolve_project", "find_project_files"],
            "project": active,
            "new_codex_turn": False,
            "reason_code": "legacy_project_lookup_gate",
            "confidence": 0.60,
        }
    return {
        "intent": "ANSWER_DIRECTLY",
        "tools": [],
        "project": active,
        "new_codex_turn": False,
        "reason_code": "legacy_unrestricted_qwen",
        "confidence": 0.40,
    }


def _failure_codes(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    codes: list[str] = []
    expected_tools = list(expected.get("tools") or [])
    actual_tools = list(actual.get("tools") or [])
    if actual.get("intent") != expected.get("intent"):
        codes.append("WRONG_INTENT")
    if actual_tools != expected_tools:
        codes.append("WRONG_TOOL" if actual_tools else "UNNECESSARY_TOOL" if expected_tools == [] else "WRONG_TOOL")
    if any(tool.startswith("delegate_to_") for tool in actual_tools) and not any(
        tool.startswith("delegate_to_") for tool in expected_tools
    ):
        codes.append("UNNECESSARY_DELEGATION")
    if actual.get("intent") == "CLARIFY" and expected.get("intent") != "CLARIFY":
        codes.append("UNNECESSARY_CLARIFICATION")
    if expected.get("project") != actual.get("project"):
        codes.append("WRONG_PROJECT")
    if len(actual_tools) > int(expected.get("max_tool_calls", len(expected_tools))):
        codes.append("EXCESS_CALLS")
    if any(tool in set(expected.get("forbidden_tools") or []) for tool in actual_tools):
        codes.append("FORBIDDEN_TOOL")
    if not expected.get("new_codex_turn") and actual.get("new_codex_turn"):
        codes.append("NEW_CODEX_TURN")
    if actual.get("tool_loop"):
        codes.append("TOOL_LOOP")
    frame = actual.get("intent_frame") if isinstance(actual.get("intent_frame"), dict) else {}
    reference = actual.get("resolved_reference") if isinstance(actual.get("resolved_reference"), dict) else {}
    if "speech_act" in expected and frame.get("speech_act") != expected.get("speech_act"):
        codes.append("WRONG_SPEECH_ACT")
    if "execution_requested" in expected and frame.get("execution_requested") != expected.get("execution_requested"):
        codes.append("WRONG_EXECUTION_REQUEST")
    if "constraints" in expected and not set(expected.get("constraints") or []).issubset(
        set(frame.get("constraints") or [])
    ):
        codes.append("CONSTRAINT_VIOLATION")
    if "resolved_reference_type" in expected and reference.get("type") != expected.get("resolved_reference_type"):
        codes.append("WRONG_REFERENCE")
    if "followup_type" in expected and frame.get("followup_type") != expected.get("followup_type"):
        codes.append("WRONG_FOLLOWUP")
    return list(dict.fromkeys(codes))


def evaluate_cases(
    cases: list[dict[str, Any]],
    decide: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    totals = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    tool_calls = 0
    pair_results: dict[str, list[bool]] = defaultdict(list)
    category_results: dict[str, list[bool]] = defaultdict(list)
    for case in cases:
        expected = case["expected"]
        actual = decide(str(case["input"]), dict(case.get("context") or {}))
        codes = _failure_codes(expected, actual)
        totals["cases"] += 1
        totals["intent_correct"] += actual.get("intent") == expected.get("intent")
        totals["tools_correct"] += list(actual.get("tools") or []) == list(expected.get("tools") or [])
        totals["project_correct"] += actual.get("project") == expected.get("project")
        frame = actual.get("intent_frame") if isinstance(actual.get("intent_frame"), dict) else {}
        reference = actual.get("resolved_reference") if isinstance(actual.get("resolved_reference"), dict) else {}
        if "speech_act" in expected:
            totals["speech_act_cases"] += 1
            totals["speech_act_correct"] += frame.get("speech_act") == expected.get("speech_act")
        if "constraints" in expected:
            totals["constraint_cases"] += 1
            totals["constraint_correct"] += set(expected.get("constraints") or []).issubset(set(frame.get("constraints") or []))
        if "resolved_reference_type" in expected:
            totals["reference_cases"] += 1
            totals["reference_correct"] += reference.get("type") == expected.get("resolved_reference_type")
        if "execution_requested" in expected:
            totals["execution_cases"] += 1
            totals["execution_correct"] += frame.get("execution_requested") == expected.get("execution_requested")
        totals["passed"] += not codes
        if case.get("pair_id"):
            pair_results[str(case["pair_id"])].append(not codes)
        category_results[str(case.get("category") or "uncategorized")].append(not codes)
        totals["unnecessary_delegations"] += "UNNECESSARY_DELEGATION" in codes
        totals["unnecessary_clarifications"] += "UNNECESSARY_CLARIFICATION" in codes
        totals["forbidden_tool_calls"] += sum(
            tool in set(expected.get("forbidden_tools") or [])
            for tool in actual.get("tools") or []
        )
        totals["excess_tool_calls"] += max(
            0,
            len(actual.get("tools") or []) - int(expected.get("max_tool_calls", 0)),
        )
        totals["new_turn_violations"] += "NEW_CODEX_TURN" in codes
        totals["tool_loop_violations"] += "TOOL_LOOP" in codes
        tool_calls += len(actual.get("tools") or [])
        confusion[str(expected.get("intent"))][str(actual.get("intent"))] += 1
        if codes:
            failures.append(
                {
                    "id": case["id"],
                    "split": case.get("split", "sealed"),
                    "input": case["input"],
                    "expected": expected,
                    "actual": actual,
                    "failure_codes": codes,
                }
            )
    count = max(1, totals["cases"])
    complete_pairs = [values for values in pair_results.values() if len(values) >= 2]
    return {
        "cases": totals["cases"],
        "passed": totals["passed"],
        "failed": totals["cases"] - totals["passed"],
        "overall_accuracy": totals["passed"] / count,
        "intent_accuracy": totals["intent_correct"] / count,
        "tool_selection_accuracy": totals["tools_correct"] / count,
        "project_accuracy": totals["project_correct"] / count,
        "speech_act_accuracy": totals["speech_act_correct"] / max(1, totals["speech_act_cases"]),
        "constraint_satisfaction_accuracy": totals["constraint_correct"] / max(1, totals["constraint_cases"]),
        "reference_resolution_accuracy": totals["reference_correct"] / max(1, totals["reference_cases"]),
        "execution_request_accuracy": totals["execution_correct"] / max(1, totals["execution_cases"]),
        "semantic_case_counts": {
            "speech_act": totals["speech_act_cases"],
            "constraints": totals["constraint_cases"],
            "reference": totals["reference_cases"],
            "execution_requested": totals["execution_cases"],
        },
        "unnecessary_delegations": totals["unnecessary_delegations"],
        "unnecessary_clarifications": totals["unnecessary_clarifications"],
        "forbidden_tool_calls": totals["forbidden_tool_calls"],
        "excess_tool_calls": totals["excess_tool_calls"],
        "new_turn_violations": totals["new_turn_violations"],
        "tool_loop_violations": totals["tool_loop_violations"],
        "average_tool_calls": tool_calls / count,
        "minimal_pair_accuracy": (
            sum(all(values) for values in complete_pairs) / len(complete_pairs)
            if complete_pairs else None
        ),
        "minimal_pairs": len(complete_pairs),
        "category_breakdown": {
            category: {
                "cases": len(values),
                "passed": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for category, values in sorted(category_results.items())
        },
        "confusion_matrix": {
            expected: dict(predicted) for expected, predicted in sorted(confusion.items())
        },
        "failures": failures,
    }


def evaluate_live_semantic_qwen(
    *,
    cases: list[dict[str, Any]],
    client: Any,
    server_state: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One isolated semantic Qwen pass per case; never dispatches tools.

    This evaluator is intentionally separate from production routing.  Each case
    owns its policy and semantic cache so a fixture, cache entry, or focus value
    cannot become input for the following case.  The optional state callback is
    read-only instrumentation supplied by the CLI runtime manager.
    """
    from .decision_policy import AgentDecisionPolicy
    from .projects import normalize_technical_transcript
    from .semantic_pass import QwenSemanticInterpreter

    class InstrumentedClient:
        """Capture synchronous semantic HTTP calls without changing their payload."""

        def __init__(self, wrapped: Any, case_started: float):
            self.wrapped = wrapped
            self.case_started = case_started
            self.requests: list[dict[str, Any]] = []
            self._previous_completed: float | None = None

        def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            started = time.perf_counter()
            previous_completed = self._previous_completed
            event: dict[str, Any] = {
                "request_index": len(self.requests) + 1,
                "started_ms": (started - self.case_started) * 1000,
                "messages": len(messages),
                "prompt_chars": sum(len(str(item.get("content") or "")) for item in messages),
                "prompt_tokens_estimate": estimate_tokens(messages),
                "system_prompt_tokens_estimate": estimate_tokens(
                    next((item.get("content") for item in messages if item.get("role") == "system"), "")
                ),
                "context_tokens_estimate": estimate_tokens(
                    next((item.get("content") for item in messages if item.get("role") == "user"), "")
                ),
                "tools_exposed": bool(kwargs.get("tools")),
                "response_format": bool(kwargs.get("response_format")),
                "max_tokens": kwargs.get("max_tokens"),
                "wait_since_previous_request_ms": (
                    (started - previous_completed) * 1000
                    if previous_completed is not None
                    else 0.0
                ),
                "overlaps_previous_request": bool(
                    previous_completed is not None and started < previous_completed
                ),
            }
            try:
                response = self.wrapped.chat(messages, **kwargs)
            except Exception as exc:
                completed = time.perf_counter()
                event.update(
                    {
                        "request_ms": (completed - started) * 1000,
                        "response_received_ms": (completed - self.case_started) * 1000,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                self._previous_completed = completed
                self.requests.append(event)
                raise
            completed = time.perf_counter()
            choice = response.get("choices", [{}])[0] if isinstance(response, dict) else {}
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content") if isinstance(message, dict) else ""
            event.update(
                {
                    "request_ms": (completed - started) * 1000,
                    "response_received_ms": (completed - self.case_started) * 1000,
                    "response_content_chars": len(str(content or "")),
                    "response_tokens_estimate": estimate_tokens(content),
                    "usage": response.get("usage") if isinstance(response, dict) else None,
                }
            )
            self._previous_completed = completed
            self.requests.append(event)
            return response

    def snapshot() -> tuple[dict[str, Any] | None, float]:
        if server_state is None:
            return None, 0.0
        started = time.perf_counter()
        try:
            return server_state(), (time.perf_counter() - started) * 1000
        except Exception as exc:  # diagnostic collection must not change evaluation behavior
            return {"error": f"{type(exc).__name__}: {exc}"}, (time.perf_counter() - started) * 1000

    timings: list[dict[str, Any]] = []

    def decide(text: str, fixture: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        before, before_state_ms = snapshot()
        initialization_started = time.perf_counter()
        # Fresh objects are deliberate: no semantic cache or ConversationFocus is
        # shared between independent benchmark fixtures.
        policy = AgentDecisionPolicy(context_cache_enabled=True)
        instrumented = InstrumentedClient(client, started)
        interpreter = QwenSemanticInterpreter(instrumented)
        initialization_ms = (time.perf_counter() - initialization_started) * 1000
        context_started = time.perf_counter()
        context = policy.build_context(fixture_context=fixture)
        context_build_ms = (time.perf_counter() - context_started) * 1000
        normalization_started = time.perf_counter()
        normalized = normalize_technical_transcript(text)
        normalization_ms = (time.perf_counter() - normalization_started) * 1000
        semantic = interpreter.interpret(text, normalized, context)
        policy_started = time.perf_counter()
        if semantic.decision is not None:
            decision = policy.decide(text, context=context, semantic_decision=semantic.decision)
        else:
            decision = policy.safe_fallback_decision(policy.decide(text, context=context))
        policy_mapping_ms = (time.perf_counter() - policy_started) * 1000
        after, after_state_ms = snapshot()
        requests = instrumented.requests
        timings.append(
            {
                "case_id": None,
                "preparation_ms": initialization_ms + context_build_ms + normalization_ms,
                "initialization_ms": initialization_ms,
                "context_build_ms": context_build_ms,
                "normalization_ms": normalization_ms,
                "semantic_ms": semantic.latency_ms,
                "qwen_request_ms": sum(float(item["request_ms"]) for item in requests),
                "time_to_first_response_ms": (
                    float(requests[0]["response_received_ms"]) if requests else None
                ),
                "qwen_requests": requests,
                "qwen_request_count": len(requests),
                "retry_count": max(0, len(requests) - 1),
                "semantic_repair_used": semantic.repair_used,
                "semantic_canonicalization_reason": semantic.canonicalization_reason,
                "semantic_cache_hit": semantic.cache_hit,
                "policy_mapping_ms": policy_mapping_ms,
                "prompt_tokens_estimate": sum(
                    int(item["prompt_tokens_estimate"]) for item in requests
                ),
                "context_tokens_estimate": estimate_tokens(context.prompt_text()),
                "server_before": before,
                "server_after": after,
                "server_before_inspection_ms": before_state_ms,
                "server_after_inspection_ms": after_state_ms,
                "all_requests_serial": not any(
                    bool(item["overlaps_previous_request"]) for item in requests
                ),
                "total_ms": (time.perf_counter() - started) * 1000,
            }
        )
        value = decision.as_dict()
        value["semantic_pass"] = semantic.as_dict()
        return value

    report = evaluate_cases(cases, decide)
    for timing, case in zip(timings, cases):
        timing["case_id"] = case.get("id")
    report["live_qwen"] = True
    report["semantic_first"] = True
    report["dry_run"] = {
        "tool_dispatches": 0,
        "codex_turns_created": 0,
        "deepseek_requests": 0,
        "filesystem_mutations": 0,
        "semantic_tools_exposed": any(
            item["tools_exposed"]
            for timing in timings
            for item in timing["qwen_requests"]
        ),
    }
    report["latency"] = {
        "semantic": latency_summary([item["semantic_ms"] for item in timings]),
        "decision": latency_summary([item["total_ms"] for item in timings]),
        "preparation": latency_summary([item["preparation_ms"] for item in timings]),
        "qwen_request": latency_summary([item["qwen_request_ms"] for item in timings]),
        "first_response": latency_summary(
            [item["time_to_first_response_ms"] for item in timings if item["time_to_first_response_ms"] is not None]
        ),
        "policy_mapping": latency_summary([item["policy_mapping_ms"] for item in timings]),
    }
    report["timing_cases"] = timings
    return report


def corpus_summary(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cases": len(cases),
        "development": sum(case.get("split") == "development" for case in cases),
        "holdout": sum(case.get("split") == "holdout" for case in cases),
        "multi_turn_scenarios": len(
            {
                case.get("sequence")
                for case in cases
                if case.get("sequence")
            }
        ),
        "stt_cases": sum(case.get("category") == "stt" for case in cases),
    }


def routing_group(case: dict[str, Any]) -> str:
    intent = str((case.get("expected") or {}).get("intent") or "")
    if intent in {"ANSWER_DIRECTLY", "CLARIFY", "NO_ACTION"}:
        return "DIRECT"
    if intent in {"LOCAL_READ", "LOCAL_SEARCH", "PROJECT_RESOLUTION", "LOCAL_ACTION"}:
        return "READ"
    if intent in {"CODEX_DELEGATE", "CODEX_REVIEW"}:
        return "CODEX"
    if intent.startswith("DEEPSEEK_"):
        return "DEEPSEEK"
    return "FOLLOWUP"


def balanced_live_sample(
    cases: list[dict[str, Any]],
    *,
    per_group: int = 4,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group in ("DIRECT", "READ", "CODEX", "DEEPSEEK", "FOLLOWUP"):
        result.extend(
            case for case in cases if routing_group(case) == group
        )
        result = result[: len(result) - max(0, sum(routing_group(item) == group for item in result) - per_group)]
    return result


def _intent_from_tools(tools: list[str], content: str = "") -> str:
    if tools:
        if tools == ["review_codex_session"]:
            return "CODEX_REVIEW"
        if tools == ["get_codex_job_status"]:
            return "CODEX_STATUS"
        if tools == ["steer_codex_job"]:
            return "CODEX_STEER"
        if tools == ["cancel_codex_job"]:
            return "CODEX_CANCEL"
        if tools == ["review_deepseek_session"]:
            return "DEEPSEEK_REVIEW"
        if "delegate_to_deepseek" in tools:
            return "DEEPSEEK_DELEGATE"
        if "delegate_to_codex" in tools:
            return "CODEX_DELEGATE"
        if "filesystem_read_text" in tools:
            return "LOCAL_READ"
        if "find_project_files" in tools:
            return "LOCAL_SEARCH"
        if tools == ["resolve_project"]:
            return "PROJECT_RESOLUTION"
        return "LOCAL_ACTION"
    return "CLARIFY" if content.rstrip().endswith("?") else "ANSWER_DIRECTLY"


def evaluate_live_qwen(
    *,
    cases: list[dict[str, Any]],
    client: Any,
    tool_specs: list[dict[str, Any]],
    fast_path: bool = False,
) -> dict[str, Any]:
    """Exercise Qwen's real tool choice without dispatching any tool."""
    from .decision_policy import AgentDecisionPolicy, tool_specs_for_decision
    from .prompt import SYSTEM_PROMPT
    from .projects import normalize_technical_transcript

    policy = AgentDecisionPolicy(context_cache_enabled=True)
    timing: list[dict[str, Any]] = []
    category_by_input = {
        str(case.get("input")): routing_group(case) for case in cases
    }

    def decide(text: str, fixture: dict[str, Any]) -> dict[str, Any]:
        local_started = time.perf_counter()
        context_started = time.perf_counter()
        context = policy.build_context(fixture_context=fixture)
        context_build_ms = (time.perf_counter() - context_started) * 1000
        policy_started = time.perf_counter()
        recommendation = policy.decide(text, context=context)
        policy_ms = (time.perf_counter() - policy_started) * 1000
        prompt_started = time.perf_counter()
        prompt = (
            SYSTEM_PROMPT
            + "\n\nDRY RUN: assess routing only; choose a tool call if needed, but "
            + "no tool will execute. When the recommendation is ANSWER_DIRECTLY, "
            + "return a brief placeholder answer without asking for missing details; "
            + "the fixture represents conversation content that is already available."
            + "\n"
            + context.prompt_text()
            + "\nDecision recommendation:\n"
            + json.dumps(recommendation.as_dict(), ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": normalize_technical_transcript(text)},
        ]
        available = tool_specs_for_decision(tool_specs, recommendation)
        exposed_tool_schema_tokens = estimate_tokens(available)
        names: list[str] = []
        calls: list[dict[str, Any]] = []
        message: dict[str, Any] = {}
        prompt_build_ms = (time.perf_counter() - prompt_started) * 1000
        started = time.perf_counter()
        first_tool_ms: float | None = None
        first_response_ms: float | None = None
        budget = max(1, recommendation.max_tool_calls) if available else 0
        shortcut = policy.fast_path(recommendation, context, text) if fast_path else None
        if shortcut is not None:
            first_tool_ms = (time.perf_counter() - local_started) * 1000
            names.append(shortcut.tool)
            calls.append(
                {
                    "id": "dry-fast",
                    "type": "function",
                    "function": {
                        "name": shortcut.tool,
                        "arguments": json.dumps(shortcut.arguments),
                    },
                }
            )
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "dry-fast",
                        "name": shortcut.tool,
                        "content": json.dumps(
                            {"ok": True, "dry_run": True, "tool": shortcut.tool}
                        ),
                    },
                ]
            )
            response_started = time.perf_counter()
            response = client.chat(messages, tools=None, max_tokens=256)
            first_response_ms = (time.perf_counter() - response_started) * 1000
            message = response.get("choices", [{}])[0].get("message", {})
            available = []
            budget = 0
        for _step in range(0 if shortcut is not None else budget + 1):
            response = client.chat(
                messages,
                tools=available or None,
                max_tokens=256,
            )
            if first_response_ms is None:
                first_response_ms = (time.perf_counter() - started) * 1000
            message = response.get("choices", [{}])[0].get("message", {})
            current_calls = message.get("tool_calls") or []
            if not current_calls:
                break
            if first_tool_ms is None:
                first_tool_ms = (time.perf_counter() - started) * 1000
            messages.append(message)
            for call in current_calls:
                if len(names) >= budget:
                    break
                name = call.get("function", {}).get("name", "")
                if not name:
                    continue
                names.append(name)
                calls.append(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"dry-{len(names)}"),
                        "name": name,
                        "content": json.dumps(
                            {"ok": True, "dry_run": True, "tool": name},
                            ensure_ascii=False,
                        ),
                    }
                )
            if len(names) >= budget:
                break
            used = set(names)
            available = [
                item
                for item in available
                if item.get("function", {}).get("name") not in used
            ]
            messages.append(
                {
                    "role": "user",
                    "content": "Continue the recommended dry-run plan using the remaining tool, if any.",
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        decision_elapsed_ms = (
            first_tool_ms
            if shortcut is not None and first_tool_ms is not None
            else first_response_ms
            if first_response_ms is not None
            else elapsed_ms
        )
        project = recommendation.project
        for call in calls:
            raw = call.get("function", {}).get("arguments", {})
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(arguments, dict):
                candidate = (
                    arguments.get("project_path")
                    or arguments.get("working_directory")
                    or arguments.get("project_id")
                )
                if candidate:
                    normalized_candidate = str(candidate).replace("/", "\\").rstrip("\\").casefold()
                    project = {
                        r"d:\tern": "tern",
                        r"d:\llama.cpp": "llama.cpp",
                        r"d:\sasori_review": "sasori_review",
                    }.get(normalized_candidate, str(candidate))
                    break
        timing.append(
            {
                "category": category_by_input.get(text, "UNKNOWN"),
                "time_to_decision_ms": decision_elapsed_ms,
                "response_ms": elapsed_ms,
                "first_response_ms": first_response_ms,
                "time_to_first_tool_call_ms": first_tool_ms,
                "context_build_ms": context_build_ms,
                "policy_ms": policy_ms,
                "prompt_build_ms": prompt_build_ms,
                "system_prompt_tokens": estimate_tokens(SYSTEM_PROMPT),
                "tool_schema_tokens": exposed_tool_schema_tokens,
                "decision_context_tokens": estimate_tokens(context.prompt_text()),
                "conversation_tokens": estimate_tokens(
                    normalize_technical_transcript(text)
                ),
                "fast_path": shortcut is not None,
            }
        )
        return {
            "intent": _intent_from_tools(names, str(message.get("content") or "")),
            "tools": names,
            "project": project,
            "new_codex_turn": "delegate_to_codex" in names,
            "reason_code": "live_qwen_dry_run",
            "confidence": recommendation.confidence,
            "intent_frame": (
                recommendation.intent_frame.as_dict()
                if recommendation.intent_frame
                else None
            ),
            "resolved_reference": (
                recommendation.resolved_reference.as_dict()
                if recommendation.resolved_reference
                else None
            ),
        }

    report = evaluate_cases(cases, decide)
    values = [item["time_to_decision_ms"] for item in timing]
    first = [
        item["time_to_first_tool_call_ms"]
        for item in timing
        if item["time_to_first_tool_call_ms"] is not None
    ]
    report["live_qwen"] = True
    report["fast_path"] = fast_path
    report["average_time_to_decision_ms"] = sum(values) / max(1, len(values))
    report["average_time_to_first_tool_call_ms"] = (
        sum(first) / len(first) if first else None
    )
    report["latency"] = {
        "decision": latency_summary(values),
        "first_tool": latency_summary(first),
        "first_response": latency_summary(
            [item["first_response_ms"] for item in timing if item["first_response_ms"] is not None]
        ),
        "response": latency_summary([item["response_ms"] for item in timing]),
        "context_build": latency_summary([item["context_build_ms"] for item in timing]),
        "policy": latency_summary([item["policy_ms"] for item in timing]),
        "prompt_build": latency_summary([item["prompt_build_ms"] for item in timing]),
    }
    report["categories"] = {}
    for category in ("DIRECT", "READ", "CODEX", "DEEPSEEK", "FOLLOWUP"):
        category_times = [
            item["time_to_decision_ms"]
            for item in timing
            if item["category"] == category
        ]
        report["categories"][category] = latency_summary(category_times)
    report["prompt_sizes"] = {
        key: latency_summary([float(item[key]) for item in timing])
        for key in (
            "system_prompt_tokens",
            "tool_schema_tokens",
            "decision_context_tokens",
            "conversation_tokens",
        )
    }
    report["timing_cases"] = timing
    return report


def evaluate(
    *,
    mode: str = "legacy",
    split: str | None = None,
    cases_path: Path = CASES_PATH,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    if split:
        cases = [case for case in cases if case.get("split") == split]
    if mode == "legacy":
        decide = legacy_decision
    elif mode == "policy":
        from .decision_policy import AgentDecisionPolicy

        policy = AgentDecisionPolicy()
        decide = lambda text, context: policy.decide(text, fixture_context=context).as_dict()
    else:
        raise ValueError(f"unknown routing evaluation mode: {mode}")
    return evaluate_cases(cases, decide)


def format_report(report: dict[str, Any], *, title: str = "Agent Routing Evaluation") -> str:
    percent = lambda value: f"{100 * float(value):.1f}%"
    lines = [
        title,
        "",
        f"Cases: {report['cases']}",
        f"Passed: {report['passed']}",
        f"Failed: {report['failed']}",
        f"Overall accuracy: {percent(report['overall_accuracy'])}",
        f"Intent accuracy: {percent(report['intent_accuracy'])}",
        f"Tool selection: {percent(report['tool_selection_accuracy'])}",
        f"Project resolution: {percent(report['project_accuracy'])}",
        f"Unnecessary delegation: {report['unnecessary_delegations']}",
        f"Unnecessary clarification: {report['unnecessary_clarifications']}",
        f"Forbidden tool calls: {report['forbidden_tool_calls']}",
        f"Excess tool calls: {report['excess_tool_calls']}",
        f"New Codex turn violations: {report['new_turn_violations']}",
        f"Tool loops: {report['tool_loop_violations']}",
        f"Average tool calls/case: {report['average_tool_calls']:.2f}",
    ]
    semantic_counts = report.get("semantic_case_counts") or {}
    if semantic_counts.get("speech_act"):
        lines.append(f"Speech act: {percent(report['speech_act_accuracy'])}")
    if semantic_counts.get("constraints"):
        lines.append(f"Constraint satisfaction: {percent(report['constraint_satisfaction_accuracy'])}")
    if semantic_counts.get("reference"):
        lines.append(f"Reference resolution: {percent(report['reference_resolution_accuracy'])}")
    if report.get("live_qwen"):
        latency = report.get("latency") or {}
        decision_latency = latency.get("decision") or {}
        average = report.get("average_time_to_decision_ms", decision_latency.get("average_ms"))
        if isinstance(average, (int, float)):
            lines.append(f"Average time to decision: {average:.1f} ms")
        lines.append(
            "Decision latency percentiles: "
            f"p50={decision_latency.get('p50_ms', 0):.1f} ms, "
            f"p90={decision_latency.get('p90_ms', 0):.1f} ms, "
            f"p95={decision_latency.get('p95_ms', 0):.1f} ms"
        )
        if report.get("semantic_first"):
            semantic_latency = latency.get("semantic") or {}
            lines.append(
                "Semantic latency percentiles: "
                f"p50={semantic_latency.get('p50_ms', 0):.1f} ms, "
                f"p90={semantic_latency.get('p90_ms', 0):.1f} ms, "
                f"p95={semantic_latency.get('p95_ms', 0):.1f} ms"
            )
        else:
            lines.append(f"Fast path: {str(bool(report.get('fast_path'))).lower()}")
        first = report.get("average_time_to_first_tool_call_ms")
        lines.append(
            "Average time to first tool call: "
            + (f"{first:.1f} ms" if isinstance(first, (int, float)) else "-")
        )
    if report["failures"]:
        lines.extend(["", "Failures:"])
        for failure in report["failures"][:20]:
            lines.extend(
                [
                    f"[{failure['id']}] {failure['input']}",
                    f"  Expected: {failure['expected']['intent']} {failure['expected'].get('tools', [])}",
                    f"  Actual: {failure['actual'].get('intent')} {failure['actual'].get('tools', [])}",
                    f"  Reason: {', '.join(failure['failure_codes'])}",
                ]
            )
    return "\n".join(lines)


def format_confusion(report: dict[str, Any]) -> str:
    matrix = report.get("confusion_matrix") or {}
    predicted = sorted(
        {name for row in matrix.values() for name in row}
        | set(matrix)
    )
    if not predicted:
        return "Intent confusion matrix: empty"
    width = max(8, *(len(value) for value in predicted))
    lines = ["Intent confusion matrix", "expected\\predicted".ljust(width + 3) + " ".join(value.rjust(width) for value in predicted)]
    for expected in sorted(matrix):
        lines.append(
            expected.ljust(width + 3)
            + " ".join(str(matrix[expected].get(value, 0)).rjust(width) for value in predicted)
        )
    return "\n".join(lines)
