from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from tern.orchestrator.decision_policy import (
    AgentDecisionPolicy,
    Constraint,
    Intent,
    SideEffect,
    TOOL_EFFECTS,
    constraint_violation_for_tool,
    tool_catalog_audit,
)
from tern.orchestrator.semantic_pass import (
    QwenSemanticInterpreter,
    SemanticValidationError,
    semantic_json_schema,
    validate_semantic_decision,
)
from tern.orchestrator.routing_eval import (
    load_semantic_regression_v2,
    load_v3_failure_audit,
)


def raw_frame(**changes):
    value = {
        "speech_act": "QUESTION",
        "primary_intent": "ANSWER_DIRECTLY",
        "operation": "answer",
        "execution_requested": False,
        "agent": "qwen",
        "target": {"type": "none", "reference": None},
        "constraints": [],
        "followup_type": "NEW_REQUEST",
        "continuation": False,
        "compound_plan": [],
        "ambiguity": {"present": False, "candidates": []},
        "confidence": 0.96,
    }
    value.update(changes)
    return value


def context(**changes):
    values = {
        "active_project": "tern",
        "project_root": r"D:\tern",
        "known_projects": ({"id": "tern", "root": r"D:\tern"},),
        "codex_job_status": None,
        "codex_job_id": None,
        "codex_running_jobs": 0,
        "codex_thread_available": True,
        "deepseek_enabled": True,
        "deepseek_configured": True,
        "deepseek_active_session": "ds-local",
        "pending_action": None,
        "focused_agent": None,
        "focused_project": "tern",
        "focused_project_root": r"D:\tern",
        "focused_file": None,
        "focused_job": None,
        "focused_session": None,
        "content_available": False,
        "ambiguous_target": False,
        "recent_tools": (),
        "last_user_intent": None,
        "last_user_text": None,
        "turn_index": 1,
        "recent_entities": (),
    }
    values.update(changes)
    return SimpleNamespace(**values)


class StructuredClient:
    supports_structured_output = True

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        value = self.values.pop(0)
        return {"choices": [{"message": {"content": value}}]}


def specs(*names):
    return [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in names
    ]


def test_schema_is_strict_and_reuses_existing_enums():
    schema = semantic_json_schema()["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["properties"]["primary_intent"]["enum"]) == {
        item.value for item in Intent
    }


def test_question_and_command_are_contrastive():
    question = validate_semantic_decision(raw_frame())
    command = validate_semantic_decision(
        raw_frame(
            speech_act="COMMAND",
            primary_intent="CODEX_CANCEL",
            operation="cancel",
            execution_requested=True,
            agent="codex",
            target={"type": "codex_job", "reference": "latest_codex_job"},
        )
    )
    assert question.execution_requested is False
    assert command.execution_requested is True


def test_side_effect_intent_requires_execution_requested():
    with pytest.raises(SemanticValidationError, match="execution_requested"):
        validate_semantic_decision(
            raw_frame(primary_intent="CODEX_CANCEL", operation="cancel", agent="codex")
        )


def test_conditional_plan_requires_order_and_deepseek_precondition():
    value = raw_frame(
        speech_act="COMMAND",
        primary_intent="CODEX_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="codex",
        compound_plan=[
            {
                "intent": "CODEX_DELEGATE",
                "operation": "delegate",
                "agent": "codex",
                "target_type": "task",
                "target_reference": "user_mentioned_target",
                "condition": "positive_recommendation",
            }
        ],
    )
    with pytest.raises(SemanticValidationError, match="ORDERED"):
        validate_semantic_decision(value)
    value["constraints"] = ["ORDERED"]
    with pytest.raises(SemanticValidationError, match="cannot have a condition"):
        validate_semantic_decision(value)


def test_semantic_reference_cannot_invent_path_or_uuid():
    with pytest.raises(SemanticValidationError, match="semantic"):
        validate_semantic_decision(raw_frame(target={"type": "file", "reference": r"D:\tern\x.py"}))


def test_unexpected_fields_are_rejected():
    value = raw_frame()
    value["explanation"] = "thought"
    with pytest.raises(SemanticValidationError, match="unexpected fields"):
        validate_semantic_decision(value)


def test_semantic_pass_has_zero_tools_and_strict_response_format():
    client = StructuredClient([json.dumps(raw_frame())])
    result = QwenSemanticInterpreter(client).interpret("o que é isso?", "o que e isso?", context())
    assert result.parse_valid
    assert client.calls[0][1]["tools"] is None
    assert client.calls[0][1]["response_format"]["type"] == "json_schema"


def test_one_repair_contains_only_schema_error_invalid_object_and_schema():
    client = StructuredClient(["not json", json.dumps(raw_frame())])
    result = QwenSemanticInterpreter(client).interpret("cancela ele", "cancela ele", context())
    assert result.parse_valid and result.repair_used
    assert len(client.calls) == 2
    repair = json.loads(client.calls[1][0][1]["content"])
    assert set(repair) == {"schema_error", "invalid_object", "expected_schema"}


def test_second_parse_failure_stops_without_loop():
    client = StructuredClient(["bad", "still bad"])
    result = QwenSemanticInterpreter(client).interpret("faz isso", "faz isso", context())
    assert not result.parse_valid and result.error == "semantic_parse_failed"
    assert len(client.calls) == 2


def test_semantic_cache_uses_message_and_context_fingerprint():
    client = StructuredClient([json.dumps(raw_frame()), json.dumps(raw_frame())])
    interpreter = QwenSemanticInterpreter(client)
    first = interpreter.interpret("abre ele", "abre ele", context(focused_file="x.py"))
    cached = interpreter.interpret("abre ele", "abre ele", context(focused_file="x.py"))
    changed = interpreter.interpret("abre ele", "abre ele", context(focused_file="y.py"))
    assert first.parse_valid and cached.cache_hit and not changed.cache_hit
    assert len(client.calls) == 2


def test_live_semantic_eval_isolates_case_cache_and_never_exposes_tools():
    from tern.orchestrator.routing_eval import evaluate_live_semantic_qwen

    class Client:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return {"choices": [{"message": {"content": json.dumps(raw_frame())}}]}

    case = {
        "id": "isolated",
        "input": "abre ele",
        "context": {"focused_file": "x.py"},
        "expected": {
            "intent": "ANSWER_DIRECTLY",
            "tools": [],
            "project": None,
            "max_tool_calls": 0,
            "new_codex_turn": False,
        },
    }
    client = Client()
    report = evaluate_live_semantic_qwen(cases=[case, dict(case, id="isolated-2")], client=client)

    # Identical inputs would be one call with a shared interpreter cache.  A
    # benchmark fixture must instead be a fresh, independent semantic pass.
    assert len(client.calls) == 2
    assert all(kwargs["tools"] is None for _messages, kwargs in client.calls)
    assert all(len(messages) == 2 for messages, _kwargs in client.calls)
    assert all(not item["semantic_cache_hit"] for item in report["timing_cases"])
    assert report["dry_run"] == {
        "tool_dispatches": 0,
        "codex_turns_created": 0,
        "deepseek_requests": 0,
        "filesystem_mutations": 0,
        "semantic_tools_exposed": False,
    }


def test_selector_skips_simple_knowledge_and_selects_operational_language():
    assert not QwenSemanticInterpreter.needs_semantic_pass("o que é overfitting?", context())
    assert QwenSemanticInterpreter.needs_semantic_pass("não manda pro Codex, explica", context())
    assert QwenSemanticInterpreter.needs_semantic_pass("abre ele", context(focused_file="x.py"))


@pytest.mark.parametrize(
    ("constraint", "tool"),
    [
        (Constraint.FORBID_CODEX, "delegate_to_codex"),
        (Constraint.FORBID_DEEPSEEK, "delegate_to_deepseek"),
        (Constraint.FORBID_CANCEL, "cancel_codex_job"),
        (Constraint.FORBID_NEW_TURN, "delegate_to_codex"),
        (Constraint.READ_ONLY, "steer_codex_job"),
        (Constraint.ANSWER_SELF, "delegate_to_deepseek"),
    ],
)
def test_constraint_envelope_blocks_incompatible_tools(constraint, tool):
    semantic = validate_semantic_decision(
        raw_frame(constraints=[constraint.value])
    )
    policy = AgentDecisionPolicy()
    decision = policy.decide("texto irrelevante", context=context(), semantic_decision=semantic)
    frame = decision.intent_frame
    assert constraint_violation_for_tool(tool, frame) == constraint.value


def test_execution_false_blocks_all_effectful_tool_classes():
    semantic = validate_semantic_decision(raw_frame())
    policy = AgentDecisionPolicy()
    decision = policy.decide("meta pergunta", context=context(), semantic_decision=semantic)
    for tool, effect in TOOL_EFFECTS.items():
        if effect is not SideEffect.READ_ONLY:
            assert constraint_violation_for_tool(tool, decision.intent_frame) == "EXECUTION_NOT_REQUESTED"


@pytest.mark.parametrize(
    ("constraint", "blocked"),
    [
        (Constraint.FORBID_CODEX, lambda tool, _effect: "codex" in tool),
        (Constraint.FORBID_DEEPSEEK, lambda tool, _effect: "deepseek" in tool),
        (Constraint.FORBID_CANCEL, lambda tool, _effect: tool == "cancel_codex_job"),
        (Constraint.FORBID_NEW_TURN, lambda tool, _effect: tool == "delegate_to_codex"),
        (Constraint.FORBID_DELEGATION, lambda tool, _effect: tool.startswith("delegate_to_")),
        (Constraint.READ_ONLY, lambda _tool, effect: effect is not SideEffect.READ_ONLY),
        (Constraint.ANSWER_SELF, lambda tool, _effect: tool.startswith("delegate_to_")),
    ],
)
def test_constraint_properties_reject_every_incompatible_tool(constraint, blocked):
    semantic = validate_semantic_decision(raw_frame(constraints=[constraint.value]))
    policy = AgentDecisionPolicy()
    decision = policy.decide("texto", context=context(), semantic_decision=semantic)
    for tool, effect in TOOL_EFFECTS.items():
        if blocked(tool, effect):
            assert constraint_violation_for_tool(tool, decision.intent_frame) == constraint.value


def test_policy_consumes_semantic_frame_instead_of_reinterpreting_words():
    semantic = validate_semantic_decision(
        raw_frame(
            speech_act="COMMAND",
            primary_intent="DEEPSEEK_DELEGATE",
            operation="delegate",
            execution_requested=True,
            agent="deepseek",
            target={"type": "task", "reference": "user_mentioned_target"},
        )
    )
    decision = AgentDecisionPolicy().decide(
        "não contém nenhuma palavra conhecida",
        context=context(),
        semantic_decision=semantic,
    )
    assert decision.intent is Intent.DEEPSEEK_DELEGATE
    assert decision.tools == ("delegate_to_deepseek",)
    assert decision.reason_code == "qwen_semantic_frame"


def test_safe_fallback_never_preserves_generation_or_mutation():
    policy = AgentDecisionPolicy()
    unsafe = policy.decide("corrige", context=context())
    fallback = policy.safe_fallback_decision(unsafe)
    assert fallback.intent is Intent.CLARIFY
    assert fallback.tools == ()


def test_compound_order_and_condition_are_preserved():
    value = raw_frame(
        speech_act="COMMAND",
        primary_intent="DEEPSEEK_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="deepseek",
        constraints=["ORDERED"],
        compound_plan=[
            {
                "intent": "DEEPSEEK_DELEGATE",
                "operation": "delegate",
                "agent": "deepseek",
                "target_type": "task",
                "target_reference": "user_mentioned_target",
                "condition": None,
            },
            {
                "intent": "CODEX_DELEGATE",
                "operation": "delegate",
                "agent": "codex",
                "target_type": "task",
                "target_reference": "deepseek_recommendation",
                "condition": "positive_recommendation",
            },
        ],
    )
    semantic = validate_semantic_decision(value)
    decision = AgentDecisionPolicy().decide("faça o plano", context=context(), semantic_decision=semantic)
    assert decision.tools == ("delegate_to_deepseek", "delegate_to_codex")
    assert semantic.compound_plan[1].condition == "positive_recommendation"


def test_catalog_audit_reports_constraint_rejection():
    semantic = validate_semantic_decision(raw_frame(constraints=["FORBID_DEEPSEEK"]))
    decision = AgentDecisionPolicy().decide("responda", context=context(), semantic_decision=semantic)
    audit = tool_catalog_audit(specs("delegate_to_deepseek", "review_codex_session"), decision)
    rejected = {item["tool"]: item["reason"] for item in audit["rejected"]}
    assert rejected["delegate_to_deepseek"] == "constraint:FORBID_DEEPSEEK"


def test_semantic_regression_v2_contains_v3_failures_and_thirty_pairs():
    cases = load_semantic_regression_v2()
    ids = {case["id"] for case in cases}
    assert len(cases) == 100
    assert len({case.get("pair_id") for case in cases if case.get("pair_id")}) == 30
    result = json.loads(
        (Path(__file__).parent / "data" / "agent_routing_test_v3_result.json").read_text(encoding="utf-8")
    )
    assert set(result["failure_ids"]).issubset(ids)


def test_v3_failure_audit_has_full_records_and_origin_breakdown():
    audit = load_v3_failure_audit()
    assert audit["failure_origin"] == {
        "semantic_interpretation_before_intent_frame": 30,
        "reference_resolution_after_correct_type": 3,
        "policy_mapping_after_correct_frame": 0,
        "contextual_tool_catalog": 0,
        "final_qwen_action_pass": 0,
    }
    assert len(audit["records"]) == 33
    assert all(
        record["phrase"]
        and record["context"] is not None
        and record["heuristic_frame"] is not None
        and record["expected_decision"]["intent"]
        and record["lost"]
        for record in audit["records"]
    )
