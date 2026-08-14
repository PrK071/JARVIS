from __future__ import annotations

import json

from tern.orchestrator.command_routing_eval import (
    evaluate_command_routing,
    load_command_routing_cases,
    observe_current_routing,
)


ALL_TOOLS = {
    "resolve_project",
    "find_project_files",
    "filesystem_list",
    "filesystem_read_text",
    "filesystem_write_text",
    "filesystem_delete",
    "web_search",
    "web_open",
    "web_open_browser",
    "web_extract",
    "review_codex_session",
    "get_codex_job_status",
    "steer_codex_job",
    "cancel_codex_job",
    "delegate_to_codex",
    "review_deepseek_session",
    "delegate_to_deepseek",
}


class NoSemanticCall:
    def chat(self, _messages, **_kwargs):
        raise AssertionError("explicit URL must bypass semantic inference")


class StructuredClient:
    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def chat(self, _messages, **_kwargs):
        self.calls += 1
        return {
            "choices": [
                {
                    "message": {"content": json.dumps(self.frame)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }


def semantic_frame(**changes):
    value = {
        "speech_act": "COMMAND",
        "primary_intent": "CLARIFY",
        "operation": "clarify",
        "execution_requested": False,
        "agent": None,
        "target": {"type": "none", "reference": None},
        "constraints": [],
        "followup_type": "REFERENCE_FOLLOWUP",
        "continuation": True,
        "compound_plan": [],
        "ambiguity": {"present": True, "candidates": ["previous_entity"]},
        "confidence": 0.9,
    }
    value.update(changes)
    return value


def test_diagnostic_corpus_is_separate_and_uses_real_routes():
    cases = load_command_routing_cases()
    assert len(cases) >= 20
    assert all("v4" not in str(case["id"]).casefold() for case in cases)
    assert {case["command_type"] for case in cases} == {
        "EXPLICIT_COMMAND",
        "ACTUALLY_AMBIGUOUS_COMMAND",
        "INFORMATIONAL_REQUEST",
        "UNAVAILABLE_CAPABILITY",
    }


def test_command_preservation_cases_have_objective_semantic_labels():
    cases = {case["id"]: case for case in load_command_routing_cases()}
    expected = {
        "CR-FILE-READ": ("COMMAND", True, "read", "filesystem"),
        "CR-FILE-WRITE": ("COMMAND", True, None, "filesystem"),
        "CR-FILE-DELETE-BLOCKED": ("COMMAND", True, "delete", "filesystem"),
        "CR-EXEC-CODEX": ("COMMAND", True, "delegate", "codex"),
        "CR-DEEPSEEK": ("COMMAND", True, "delegate", "deepseek"),
        "CR-INFO-PYTEST": ("QUESTION", False, "answer", "qwen"),
        "CR-INFO-CODEX": ("QUESTION", False, "answer", "qwen"),
    }
    for case_id, (speech, execution, operation, agent) in expected.items():
        case = cases[case_id]
        assert case["expected_speech_act"] == speech
        assert case["expected_execution_requested"] is execution
        assert case.get("expected_operation") == operation
        assert case["expected_agent"] == agent


def test_command_preservation_corpus_has_informational_guardrails():
    cases = {case["id"]: case for case in load_command_routing_cases()}
    guardrails = {
        "CR-INFO-PYTEST",
        "CR-INFO-RM",
        "CR-INFO-READ-FILE",
        "CR-INFO-OPEN-SITE",
        "CR-INFO-CODEX",
        "CR-INFO-JSON",
    }
    assert all(
        cases[case_id]["command_type"] == "INFORMATIONAL_REQUEST"
        and cases[case_id]["expected_execution_requested"] is False
        and cases[case_id]["expected_route"] == "ANSWER_DIRECTLY"
        for case_id in guardrails
    )


def test_explicit_url_observation_uses_production_bypass_and_runtime_catalog():
    case = next(
        case
        for case in load_command_routing_cases()
        if case["id"] == "CR-WEB-URL"
    )
    observed = observe_current_routing(
        case,
        client=NoSemanticCall(),
        available_tools=ALL_TOOLS,
        enabled_tools=ALL_TOOLS,
    )
    assert observed["decision"]["intent"] == "WEB_OPEN"
    assert observed["selected_tool"] == "web_open_browser"
    assert observed["tool_available"] is True
    assert observed["semantic_pass_used"] is False
    assert observed["explicit_target_detected"] is True
    assert observed["execution_attempted"] is False


def test_ambiguous_reference_can_clarify_without_executing():
    case = next(
        case
        for case in load_command_routing_cases()
        if case["id"] == "CR-AMB-ISSO"
    )
    client = StructuredClient(semantic_frame())
    observed = observe_current_routing(
        case,
        client=client,
        available_tools=ALL_TOOLS,
        enabled_tools=ALL_TOOLS,
    )
    assert client.calls == 1
    assert observed["decision"]["intent"] == "CLARIFY"
    assert observed["clarify_reason"] == "semantic_reference_ambiguous"
    assert observed["semantic_parse_valid"] is True
    assert observed["execution_attempted"] is False


def test_semantic_correctness_is_separate_from_structural_parse_validity():
    case = {
        "id": "semantic-wrong",
        "input": "leia config.json",
        "command_type": "EXPLICIT_COMMAND",
        "context": {},
        "expected_route": "LOCAL_READ",
        "expected_tools": ["filesystem_read_text"],
        "expected_speech_act": "COMMAND",
        "expected_execution_requested": True,
    }
    observed = {
        "decision": {
            "intent": "ANSWER_DIRECTLY",
            "tools": [],
            "reason_code": "qwen_semantic_frame",
            "intent_frame": {
                "speech_act": "COMMAND",
                "execution_requested": True,
            },
        },
        "semantic_pass_used": True,
        "semantic_frame": {"primary_intent": "ANSWER_DIRECTLY"},
        "semantic_parse_valid": True,
        "request_id": "semantic-wrong",
        "available_tools": sorted(ALL_TOOLS),
        "enabled_tools": sorted(ALL_TOOLS),
        "disabled_tools": [],
        "selected_tool": None,
        "tool_available": None,
        "explicit_target_detected": True,
        "clarify_reason": None,
        "retry": 0,
        "fallback": False,
        "latency_ms": 1.0,
        "prompt_tokens": 1,
        "generated_tokens": 1,
        "finish_reason": "stop",
        "execution_attempted": False,
        "execution_allowed": None,
    }
    record = evaluate_command_routing([case], lambda _case: observed)["records"][0]
    assert record["semantic_parse_valid"] is True
    assert record["semantic_valid"] is False
    assert record["routing_correct"] is False


def test_disabled_tool_is_reported_after_correct_routing():
    cases = load_command_routing_cases()
    web_case = next(case for case in cases if case["id"] == "CR-WEB-DISABLED")
    observed = observe_current_routing(
        web_case,
        client=NoSemanticCall(),
        available_tools=ALL_TOOLS,
        enabled_tools=ALL_TOOLS,
    )
    report = evaluate_command_routing([web_case], lambda _case: observed)
    record = report["records"][0]
    assert record["routing_correct"] is True
    assert record["plan_valid"] is True
    assert record["tool_available"] is False
    assert record["execution_allowed"] is False
    assert report["unavailable_tool_selection_rate"] == 1.0


def test_safety_block_is_separate_from_correct_routing():
    case = next(
        case
        for case in load_command_routing_cases()
        if case["id"] == "CR-WEB-SAFETY-BLOCK"
    )
    observed = observe_current_routing(
        case,
        client=NoSemanticCall(),
        available_tools=ALL_TOOLS,
        enabled_tools=ALL_TOOLS,
    )
    record = evaluate_command_routing([case], lambda _case: observed)["records"][0]
    assert record["routing_correct"] is True
    assert record["tool_available"] is True
    assert record["execution_attempted"] is False
    assert record["execution_allowed"] is False


def test_unavailable_capability_records_current_unscored_contract():
    case = next(
        case
        for case in load_command_routing_cases()
        if case["id"] == "CR-UNAVAILABLE-FAX"
    )
    observed = observe_current_routing(
        case,
        client=NoSemanticCall(),
        available_tools=ALL_TOOLS,
        enabled_tools=ALL_TOOLS,
    )
    report = evaluate_command_routing([case], lambda _case: observed)
    record = report["records"][0]
    assert report["scored_cases"] == 0
    assert record["predicted_route"] == "ANSWER_DIRECTLY"
    assert record["routing_correct"] is None
    assert record["selected_tool"] is None


def test_route_and_plan_metrics_are_independent():
    cases = [
        {
            "id": "both-valid",
            "input": "x",
            "command_type": "EXPLICIT_COMMAND",
            "context": {},
            "expected_route": "LOCAL_READ",
            "expected_tools": ["filesystem_read_text"],
        },
        {
            "id": "route-valid-plan-invalid",
            "input": "x",
            "command_type": "EXPLICIT_COMMAND",
            "context": {},
            "expected_route": "LOCAL_READ",
            "expected_tools": ["filesystem_read_text"],
        },
        {
            "id": "route-wrong-plan-valid",
            "input": "x",
            "command_type": "EXPLICIT_COMMAND",
            "context": {},
            "expected_route": "LOCAL_READ",
            "expected_tools": ["filesystem_read_text"],
        },
        {
            "id": "both-invalid",
            "input": "x",
            "command_type": "EXPLICIT_COMMAND",
            "context": {},
            "expected_route": "LOCAL_READ",
            "expected_tools": ["filesystem_read_text"],
        },
    ]
    values = {
        "both-valid": ("LOCAL_READ", ["filesystem_read_text"]),
        "route-valid-plan-invalid": ("LOCAL_READ", ["find_project_files"]),
        "route-wrong-plan-valid": ("LOCAL_SEARCH", ["filesystem_read_text"]),
        "both-invalid": ("ANSWER_DIRECTLY", []),
    }

    def observe(case):
        route, tools = values[case["id"]]
        return {
            "decision": {
                "intent": route,
                "tools": tools,
                "reason_code": "fixture",
                "intent_frame": {
                    "speech_act": "COMMAND",
                    "execution_requested": True,
                },
            },
            "request_id": case["id"],
            "available_tools": sorted(ALL_TOOLS),
            "enabled_tools": sorted(ALL_TOOLS),
            "disabled_tools": [],
            "selected_tool": tools[0] if tools else None,
            "tool_available": bool(tools) or None,
            "explicit_target_detected": True,
            "clarify_reason": None,
            "semantic_valid": True,
            "retry": 0,
            "fallback": False,
            "latency_ms": 1.0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "finish_reason": None,
            "execution_attempted": False,
            "execution_allowed": None,
        }

    report = evaluate_command_routing(cases, observe)
    assert report["route_plan_quadrants"] == {
        "routing_correct / plan_invalid": 1,
        "routing_correct / plan_valid": 1,
        "routing_wrong / plan_invalid": 1,
        "routing_wrong / plan_valid": 1,
    }
