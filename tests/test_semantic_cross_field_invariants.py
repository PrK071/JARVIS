from __future__ import annotations

import copy
import json

import pytest

from tern.orchestrator.decision_policy import Intent
from tern.orchestrator.semantic_pass import (
    HARD_CROSS_FIELD_INVARIANTS,
    QwenSemanticInterpreter,
    SemanticValidationCode,
    SemanticValidationError,
    validate_semantic_decision,
)


def frame(**changes):
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
        "confidence": 0.95,
    }
    value.update(changes)
    return value


class SequenceClient:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return {
            "choices": [
                {"message": {"content": json.dumps(self.values.pop(0))}}
            ]
        }


def context():
    from types import SimpleNamespace

    return SimpleNamespace(
        active_project="tern",
        project_root=r"D:\tern",
        known_projects=({"id": "tern", "root": r"D:\tern"},),
        codex_job_status=None,
        codex_job_id=None,
        codex_running_jobs=0,
        codex_thread_available=True,
        deepseek_enabled=True,
        deepseek_configured=True,
        deepseek_active_session=None,
        pending_action=None,
        focused_agent=None,
        focused_project="tern",
        focused_project_root=r"D:\tern",
        focused_file=None,
        focused_job=None,
        focused_session=None,
        content_available=False,
        ambiguous_target=False,
        recent_tools=(),
        last_user_intent=None,
        last_user_text=None,
        turn_index=0,
        recent_entities=(),
    )


@pytest.mark.parametrize("intent,agent", [
    ("CODEX_DELEGATE", "codex"),
    ("DEEPSEEK_DELEGATE", "deepseek"),
])
def test_read_only_delegation_conflict_is_rejected_individually(intent, agent):
    raw = frame(
        speech_act="COMMAND",
        primary_intent=intent,
        operation="delegate",
        execution_requested=True,
        agent=agent,
        target={"type": "task", "reference": "user_mentioned_target"},
        constraints=["READ_ONLY"],
    )
    original = copy.deepcopy(raw)
    with pytest.raises(SemanticValidationError) as captured:
        validate_semantic_decision(
            raw,
            cross_field_invariants=frozenset(
                {SemanticValidationCode.READ_ONLY_TOOL_INTENT_CONFLICT}
            ),
        )
    assert captured.value.code is SemanticValidationCode.READ_ONLY_TOOL_INTENT_CONFLICT
    assert raw == original


@pytest.mark.parametrize(
    "raw,expected",
    [
        (frame(), Intent.ANSWER_DIRECTLY),
        (
            frame(
                speech_act="COMMAND",
                primary_intent="LOCAL_READ",
                operation="read",
                execution_requested=True,
                agent="filesystem",
                target={"type": "file", "reference": "focused_file"},
                constraints=["READ_ONLY"],
            ),
            Intent.LOCAL_READ,
        ),
        (
            frame(
                speech_act="COMMAND",
                primary_intent="LOCAL_ACTION",
                operation="delete",
                execution_requested=True,
                agent="filesystem",
                target={"type": "file", "reference": "focused_file"},
            ),
            Intent.LOCAL_ACTION,
        ),
        (
            frame(
                speech_act="COMMAND",
                primary_intent="CODEX_DELEGATE",
                operation="delegate",
                execution_requested=True,
                agent="codex",
                target={"type": "task", "reference": "user_mentioned_target"},
            ),
            Intent.CODEX_DELEGATE,
        ),
        (
            frame(
                primary_intent="CLARIFY",
                operation="clarify",
                agent=None,
                ambiguity={"present": True, "candidates": ["previous_entity"]},
            ),
            Intent.CLARIFY,
        ),
        (
            frame(
                speech_act="COMMAND",
                primary_intent="WEB_OPEN",
                operation="open_url",
                execution_requested=True,
                agent="web",
                target={"type": "url", "reference": "https://example.com"},
            ),
            Intent.WEB_OPEN,
        ),
        (
            frame(
                speech_act="COMMAND",
                primary_intent="LOCAL_SEARCH",
                operation="search",
                execution_requested=True,
                agent="project",
                target={"type": "project", "reference": "active_project"},
                constraints=["READ_ONLY"],
            ),
            Intent.LOCAL_SEARCH,
        ),
        (
            frame(
                primary_intent="CODEX_REVIEW",
                operation="review",
                agent="codex",
                target={"type": "codex_session", "reference": "shared_codex_session"},
                constraints=["READ_ONLY"],
            ),
            Intent.CODEX_REVIEW,
        ),
        (
            frame(
                primary_intent="DEEPSEEK_REVIEW",
                operation="review",
                agent="deepseek",
                target={"type": "deepseek_session", "reference": "active_deepseek_session"},
                constraints=["READ_ONLY"],
            ),
            Intent.DEEPSEEK_REVIEW,
        ),
    ],
)
def test_valid_neighboring_route_frames_remain_valid(raw, expected):
    result = validate_semantic_decision(
        raw,
        cross_field_invariants=HARD_CROSS_FIELD_INVARIANTS,
    )
    assert result.primary_intent is expected
    assert validate_semantic_decision(
        raw,
        cross_field_invariants=HARD_CROSS_FIELD_INVARIANTS,
    ) == result


def test_inconsistent_frame_uses_existing_retry_without_local_correction():
    invalid = frame(
        speech_act="COMMAND",
        primary_intent="CODEX_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="codex",
        target={"type": "task", "reference": "user_mentioned_target"},
        constraints=["READ_ONLY"],
    )
    valid = frame(
        speech_act="COMMAND",
        primary_intent="CODEX_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="codex",
        target={"type": "task", "reference": "user_mentioned_target"},
    )
    client = SequenceClient([invalid, valid])
    result = QwenSemanticInterpreter(
        client,
        cross_field_invariants=HARD_CROSS_FIELD_INVARIANTS,
    ).interpret("mande ao Codex", "mande ao codex", context())
    assert result.parse_valid and result.repair_used
    assert result.decision.primary_intent is Intent.CODEX_DELEGATE
    assert result.validation_error_codes == (
        SemanticValidationCode.READ_ONLY_TOOL_INTENT_CONFLICT.value,
    )
    assert len(client.calls) == 2
    repair = json.loads(client.calls[1][0][1]["content"])
    assert repair["invalid_object"] == json.dumps(invalid)
    assert set(repair) == {"schema_error", "invalid_object", "expected_schema"}


def test_second_inconsistent_frame_uses_existing_fallback_signal():
    invalid = frame(
        speech_act="COMMAND",
        primary_intent="CODEX_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="codex",
        target={"type": "task", "reference": "user_mentioned_target"},
        constraints=["READ_ONLY"],
    )
    client = SequenceClient([invalid, invalid])
    result = QwenSemanticInterpreter(
        client,
        cross_field_invariants=HARD_CROSS_FIELD_INVARIANTS,
    ).interpret("mande ao Codex", "mande ao codex", context())
    assert not result.parse_valid
    assert result.decision is None
    assert result.error == "semantic_parse_failed"
    assert result.validation_error_codes == (
        SemanticValidationCode.READ_ONLY_TOOL_INTENT_CONFLICT.value,
        SemanticValidationCode.READ_ONLY_TOOL_INTENT_CONFLICT.value,
    )
    assert len(client.calls) == 2


def test_experimental_invariants_are_inactive_without_explicit_enablement():
    raw = frame(
        speech_act="COMMAND",
        primary_intent="CODEX_DELEGATE",
        operation="delegate",
        execution_requested=True,
        agent="codex",
        target={"type": "task", "reference": "user_mentioned_target"},
        constraints=["READ_ONLY"],
    )
    accepted = validate_semantic_decision(raw)
    assert accepted.primary_intent is Intent.CODEX_DELEGATE
